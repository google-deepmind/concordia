# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Lifeways world running for a page, which a person plays as Experimenter.

The session is a harness: it builds a simulation, plays it in a thread of
its own, and reads its state. The person on the page is the Experimenter
entity, through the stock `HumanActComponent` and a `PageInput` adapter: each
year the Experimenter answers with the move the person queued, or else the
person's standing choice (watching, unless the person sets another), at the
pace the person sets. The adapter never invents a move: when the page closes
the session it raises, and the run ends.

A session given a save path outlives its server: at a year's close, every
so many seconds and once more when the server stops, it saves the stock
simulation checkpoint (`Simulation.make_checkpoint_data`) with the page's
census rows; a new session of the same world restores the last save
(`simulation.load_checkpoint`) and plays on from that year.
"""

import collections
import json
import os
import threading
import time
from typing import Any

from absl import logging
from concordia.components.agent import human_act_component
from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import person as person_lib
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import land as land_lib
from examples.lifeways.world import world as world_lib
from concordia.typing import human_input

# Census rows a session keeps for the page's graphs.
ROWS_KEPT = 400
# The form of a session's save; a save of another form is set aside.
SAVE_VERSION = 1
# The name the asynchronous engine gives its entity loops' threads.
ENGINE_THREADS = 'entity_loop'


class Closed(Exception):
  """The page closed the session: the Experimenter has no answer to give."""


class PageInput:
  """The Experimenter's answers from the page, at the page's pace.

  A `HumanInput`: called once a year with the Experimenter's question, it
  waits until the person's pace allows another year (and the person has not
  paused), then returns the person's next queued move, or else the person's
  standing choice (watching).
  """

  def __init__(self, years_per_second: float = 1.0, standing: str = 'watch'):
    """Initializes the input.

    Args:
      years_per_second: How fast the person lets the years pass.
      standing: What the person does in a year it queued nothing for: the
        person's standing choice, watching.
    """
    self._years_per_second = float(years_per_second)
    self._standing = standing
    self._queue: collections.deque[str] = collections.deque()
    self._paused = False
    self._closed = False
    self._due = 0.0
    self._changed = threading.Condition()

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    del request
    with self._changed:
      while True:
        if self._closed:
          raise Closed('The page closed the session.')
        wait = self._due - time.monotonic()
        if not self._paused and wait <= 0:
          break
        self._changed.wait(timeout=0.1 if self._paused else min(0.1, wait))
      self._due = time.monotonic() + 1.0 / max(self._years_per_second, 1e-3)
      if self._queue:
        return self._queue.popleft()
      return self._standing

  def queue(self, move: str) -> None:
    """Queues a move the person chose, for the next year.

    Args:
      move: The move.
    """
    with self._changed:
      self._queue.append(move)
      self._changed.notify_all()

  def pace(
      self, years_per_second: float | None = None, paused: bool | None = None
  ) -> None:
    """Sets how fast the years pass, or pauses them.

    Args:
      years_per_second: The pace (0.05 to 50), or None to keep it.
      paused: Whether the years wait, or None to keep it.
    """
    with self._changed:
      if years_per_second is not None:
        self._years_per_second = max(0.05, min(50.0, float(years_per_second)))
      if paused is not None:
        self._paused = bool(paused)
      self._changed.notify_all()

  def close(self) -> None:
    """Closes the input: the Experimenter's next question raises."""
    with self._changed:
      self._closed = True
      self._changed.notify_all()

  def settings(self) -> dict[str, Any]:
    """The pace, whether paused, the moves queued and the standing choice."""
    with self._changed:
      return {
          'years_per_second': self._years_per_second,
          'paused': self._paused,
          'queued': list(self._queue),
          'standing': self._standing,
      }


class Session:
  """One world, played in a thread, read by a page."""

  def __init__(
      self,
      scenario: str = 'sandbox',
      seed: int = 0,
      years_per_second: float = 1.0,
      save_path: str | None = None,
      save_seconds: float = 10.0,
      **overrides: Any,
  ):
    """Builds the world, or restores its last save (it starts with `start`).

    Args:
      scenario: The scenario or preset.
      seed: The world seed.
      years_per_second: The pace the years start at.
      save_path: Optional. The file the world is saved to and restored from.
        A save of another world (scenario, seed, settings) or one that no
        longer restores is set aside, and a new world begins.
      save_seconds: At most how often the world is saved, in seconds.
      **overrides: More settings.
    """
    self.scenario = scenario
    self.input = PageInput(years_per_second)
    self._settings = scenarios.settings(scenario, seed=seed, **overrides)
    # What a save must match to be restored.
    self._identity = json.loads(
        json.dumps(
            {'scenario': scenario, 'seed': seed, 'overrides': overrides},
            sort_keys=True,
            default=str,
        )
    )
    self._rows: collections.deque[dict[str, Any]] = collections.deque(
        maxlen=ROWS_KEPT
    )
    self._rows_lock = threading.Lock()
    self._thread: threading.Thread | None = None
    self.error: str = ''
    # When the person last did anything (a move, the pace, a stop).
    self._played = 0.0
    self._save_path = save_path
    self._save_seconds = float(save_seconds)
    self._saving = threading.Condition()
    self._last_save = time.monotonic()
    self._save_wanted = False
    self._saves = 0
    self.saved_year: int | None = None
    self.restored_year: int | None = None
    self.run = self._build()
    if save_path:
      self._restore(save_path)

  def _build(self) -> simulation.Run:
    run = simulation.build_simulation(
        self._settings,
        # People wait for their turn without spinning: the page's years are
        # slow anyway.
        sleep_time=0.05,
        archive=simulation.Archive(keep=0),
        experimenter=lambda order: human_act_component.HumanActComponent(
            self.input, order
        ),
    )
    run.lifecycle.year_closed = self._year_closed
    return run

  def start(self) -> None:
    """Plays the world in a thread of its own until it ends or is closed."""
    self._thread = threading.Thread(
        target=self._play, name='lifeways-session', daemon=True
    )
    self._thread.start()

  def _play(self) -> None:
    try:
      # The page reads the world and its own census rows, never the raw log.
      self.run.play(log_retention=0)
    except Closed:
      pass
    except Exception as error:  # pylint: disable=broad-exception-caught
      self.error = f'{type(error).__name__}: {error}'

  def close(self, timeout: float = 10.0) -> None:
    """Ends the run: the Experimenter's next answer raises.

    The engine then tells everyone's loop to end, but returns without
    waiting for loops in the middle of a turn; this waits for them too, so
    that a process exiting next leaves no turn half done.

    Args:
      timeout: How long to wait for the run's threads.
    """
    deadline = time.monotonic() + timeout
    self.input.close()
    self.run.engine.play()
    if self._thread is not None:
      self._thread.join(timeout)
    for thread in threading.enumerate():
      if thread.name.startswith(ENGINE_THREADS):
        thread.join(max(0.0, deadline - time.monotonic()))

  def _year_closed(self, year: int) -> None:
    # In the World GM's lock: the world may be read here, and is whole.
    world = self.run.world()
    row = {
        **(world.history[-1] if world.history else {'year': year - 1}),
        'held': self.run.customs_census(),
        'peoples': self.run.peoples(),
    }
    with self._rows_lock:
      self._rows.append(row)
    if self._save_path is None:
      return
    with self._saving:
      if (
          self._save_wanted
          or time.monotonic() - self._last_save >= self._save_seconds
      ):
        self._save(self._save_path, year)

  # --- Saves ------------------------------------------------------------

  def _save(self, path: str, year: int) -> None:
    """Writes the world at a year's close (in the World GM's lock)."""
    checkpoint = self.run.sim.make_checkpoint_data()
    checkpoint['raw_log'] = []  # Nobody reads it; a restore needs none.
    with self._rows_lock:
      rows = list(self._rows)
    saved = {
        'version': SAVE_VERSION,
        'world': self._identity,
        'year': year,
        'rows': rows,
        'pace': self.input.settings(),
        'checkpoint': checkpoint,
    }
    try:
      with open(path + '.tmp', 'w') as f:
        json.dump(saved, f, default=str)
      os.replace(path + '.tmp', path)
    except OSError as error:
      logging.warning('Lifeways: the world was not saved: %s', error)
    else:
      self.saved_year = year
    self._last_save = time.monotonic()
    self._save_wanted = False
    self._saves += 1
    self._saving.notify_all()

  def save_soon(self, timeout: float = 5.0) -> bool:
    """Saves the world at the next year's close, as a server stops.

    Args:
      timeout: How long to wait for the year to close.

    Returns:
      Whether a save was made (not if the world is paused, stopped, or
      slower than the timeout).
    """
    if self._save_path is None or self.input.settings()['paused']:
      return False
    if self._thread is None or not self._thread.is_alive():
      return False
    with self._saving:
      before = self._saves
      self._save_wanted = True
      return self._saving.wait_for(lambda: self._saves > before, timeout)

  def _restore(self, path: str) -> None:
    """Restores the world's last save, if it is a save of this world."""
    try:
      with open(path) as f:
        saved = json.load(f)
    except FileNotFoundError:
      return
    except (OSError, ValueError) as error:
      self._set_aside(path, f'it does not read ({error})')
      return
    if not isinstance(saved, dict) or saved.get('version') != SAVE_VERSION:
      self._set_aside(path, 'it is of another form')
      return
    if saved.get('world') != self._identity:
      self._set_aside(path, f'it is another world ({saved.get("world")})')
      return
    try:
      simulation.load_checkpoint(self.run, saved['checkpoint'])
    except Exception as error:  # pylint: disable=broad-exception-caught
      # A save from an older build may no longer fit: begin anew.
      self.run = self._build()
      self._set_aside(path, f'it no longer restores ({error!r})')
      return
    with self._rows_lock:
      self._rows.extend(saved.get('rows', ()))
    pace = saved.get('pace', {})
    self.input.pace(pace.get('years_per_second'), pace.get('paused'))
    if pace.get('paused'):
      self.run.engine.pause()
    for move in pace.get('queued', ()):
      self.input.queue(move)
    self.restored_year = self.saved_year = int(saved['year'])

  def _set_aside(self, path: str, why: str) -> None:
    logging.warning(
        'Lifeways: a new world begins; the save is set aside: %s', why
    )
    try:
      os.replace(path, path + '.set-aside')
    except OSError:
      pass

  # --- What the page reads ----------------------------------------------

  def census(self, since: int = -1) -> list[dict[str, Any]]:
    """The census rows of the years after `since`.

    Args:
      since: A year (-1: all the rows kept).

    Returns:
      The rows, oldest first.
    """
    with self._rows_lock:
      return [row for row in self._rows if row['year'] > since]

  def state(self) -> dict[str, Any]:
    """The world as the page draws it.

    Returns:
      The year, the map, the people, the places, the pace, the interviews,
      and what the page needs to know of the game (kinds, domains, the age
      of the grown, the highest skill level, the rows kept).
    """
    world = self.run.world_state()
    land = world['land']
    people = [
        {
            'name': p['name'],
            'tile': p['tile'],
            'age': int(world['year']) - int(p['born']),
            'sex': p['sex'],
            'food': round(float(p['food']), 1),
            'partner': p['partner'],
            'descent': p.get('descent', ''),
            'speech': p.get('speech', ''),
        }
        for p in world['people']
    ]
    places = [
        {
            'id': p['id'],
            'kind': p['kind'],
            'tile': p['tile'],
            'shelf': len(p['shelf']),
            'used': p['used'],
        }
        for p in world.get('places', {}).get('places', ())
    ]
    return {
        'scenario': self.scenario,
        'year': world['year'],
        'rows': land['rows'],
        'cols': land['cols'],
        'kinds': list(land_lib.TERRAIN),
        'terrain': land['terrain'],
        'food': [[round(f, 1) for f in row] for row in land['food']],
        'people': people,
        'places': places,
        'pace': self.input.settings(),
        'running': self._thread is not None and self._thread.is_alive(),
        'error': self.error,
        'idle_seconds': (
            round(time.time() - self._played) if self._played else None
        ),
        'release_form': world_lib.RELEASE_FORM,
        'interviews': dict(world.get('interviews', {})),
        'transcripts': list(world.get('transcripts', ()))[-10:],
        'custom_kinds': list(culture_lib.KINDS),
        'domains': list(culture_lib.DOMAINS),
        'grown_age': world_lib.ADULT_AGE,
        'max_level': culture_lib.MAX_LEVEL,
        'rows_kept': ROWS_KEPT,
        'saved_year': self.saved_year,
        'restored_year': self.restored_year,
    }

  def person(self, name: str) -> dict[str, Any] | None:
    """A living person as the inspector shows it: customs, kin, body.

    Args:
      name: The person's name.

    Returns:
      The person, or None if nobody of that name lives.
    """
    world = self.run.world_state()
    body = next((p for p in world['people'] if p['name'] == name), None)
    entity = next(
        (e for e in list(self.run.sim.entities) if e.name == name), None
    )
    if body is None or entity is None:
      return None
    learner = entity.get_component(  # pyrefly: ignore[missing-attribute]
        person_lib.LEARNER, type_=culture.Learner
    )
    came_by = {held['id']: held['via'] for held in learner.get_state()['held']}
    customs = [
        {
            'id': c.id,
            'kind': c.kind,
            'domain': c.domain,
            'text': c.spec['text'],
            'params': dict(c.spec.get('params', {})),
            'via': came_by.get(c.id, ''),
        }
        for c in entity.get_all_context_components().values()  # pyrefly: ignore[missing-attribute]
        if isinstance(c, culture.Custom)
    ]
    return {
        **{k: body[k] for k in ('name', 'born', 'sex', 'tile', 'partner')},
        'food': round(float(body['food']), 1),
        'parents': body.get('parents', []),
        'children': body.get('children', []),
        'descent': body.get('descent', ''),
        'customs': sorted(customs, key=lambda c: (c['domain'], c['kind'])),
    }

  # --- What the page does -----------------------------------------------

  def stop(self) -> None:
    """The person stops the world: the Experimenter's next move ends it."""
    self._played = time.time()
    self.input.queue(world_lib.STOP)
    self.input.pace(paused=False)
    self.run.engine.play()

  def move(self, move: str) -> None:
    """Queues the Experimenter's move (as the person typed or built it).

    Args:
      move: The move.
    """
    move = ' '.join(str(move).split())
    if move:
      self._played = time.time()
      self.input.queue(move)

  def pace(
      self,
      years_per_second: float | None = None,
      paused: bool | None = None,
  ) -> dict[str, Any]:
    """Sets the pace, or pauses: everyone waits (the engine's own pause).

    Args:
      years_per_second: The pace, or None to keep it.
      paused: Whether the world waits, or None to keep it.

    Returns:
      The input's settings.
    """
    self._played = time.time()
    self.input.pace(years_per_second, paused)
    if paused is True:
      self.run.engine.pause()
    elif paused is False:
      self.run.engine.play()
    return self.input.settings()
