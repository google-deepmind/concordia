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
"""

import collections
import threading
import time
from typing import Any

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


class Closed(Exception):
  """The page closed the session: the Experimenter has no answer to give."""


class PageInput:
  """The Experimenter's answers from the page, at the page's pace.

  A `HumanInput`: called once a year with the Experimenter's question, it
  waits until the person's pace allows another year (and the person has not
  paused), then returns the person's next queued move, or else the person's
  standing choice.
  """

  def __init__(self, years_per_second: float = 1.0, standing: str = 'watch'):
    """Initializes the input.

    Args:
      years_per_second: How fast the person lets the years pass.
      standing: What the person does in a year it queued nothing for.
    """
    self._years_per_second = float(years_per_second)
    self._standing = standing
    self._queue: collections.deque[str] = collections.deque()
    self._paused = False
    self._closed = False
    self._due = 0.0
    self._changed = threading.Condition()
    self.last_request: human_input.HumanInputRequest | None = None

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    with self._changed:
      self.last_request = request
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
    """Queues a move the person chose, for the next year."""
    with self._changed:
      self._queue.append(move)
      self._changed.notify_all()

  def pace(
      self, years_per_second: float | None = None, paused: bool | None = None
  ) -> None:
    """Sets how fast the years pass, or pauses them."""
    with self._changed:
      if years_per_second is not None:
        self._years_per_second = max(0.05, min(50.0, float(years_per_second)))
      if paused is not None:
        self._paused = bool(paused)
      self._changed.notify_all()

  def close(self) -> None:
    with self._changed:
      self._closed = True
      self._changed.notify_all()

  def settings(self) -> dict[str, Any]:
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
      **overrides: Any,
  ):
    """Builds the world (it starts with `start`).

    Args:
      scenario: The scenario or preset.
      seed: The world seed.
      years_per_second: The pace the years start at.
      **overrides: More settings.
    """
    self.scenario = scenario
    self.input = PageInput(years_per_second)
    self._rows: collections.deque[dict[str, Any]] = collections.deque(
        maxlen=ROWS_KEPT
    )
    self._rows_lock = threading.Lock()
    self.run = simulation.build_simulation(
        scenarios.settings(scenario, seed=seed, **overrides),
        archive=simulation.Archive(keep=0),
        experimenter=lambda order: human_act_component.HumanActComponent(
            self.input, order
        ),
    )
    self.run.lifecycle.year_closed = self._year_closed
    self._thread: threading.Thread | None = None
    self.error: str = ''
    # When the person last did anything (a move, the pace, a stop).
    self._played = 0.0

  def start(self) -> None:
    """Plays the world in a thread of its own until it ends or is closed."""
    self._thread = threading.Thread(
        target=self._play, name='lifeways-session', daemon=True
    )
    self._thread.start()

  def _play(self) -> None:
    try:
      self.run.play()
    except Closed:
      pass
    except Exception as error:  # pylint: disable=broad-exception-caught
      self.error = f'{type(error).__name__}: {error}'

  def close(self, timeout: float = 10.0) -> None:
    """Ends the run: the Experimenter's next answer raises."""
    self.input.close()
    if self._thread is not None:
      self._thread.join(timeout)

  def _year_closed(self, year: int) -> None:
    # In the World GM's lock: the world may be read here.
    world = self.run.world()
    row = {
        **world.history[-1],
        'held': self.run.customs_census(),
        'peoples': self.run.peoples(),
    }
    del year
    with self._rows_lock:
      self._rows.append(row)

  # --- What the page reads ----------------------------------------------

  def census(self, since: int = -1) -> list[dict[str, Any]]:
    """The census rows of the years after `since`."""
    with self._rows_lock:
      return [row for row in self._rows if row['year'] > since]

  def state(self) -> dict[str, Any]:
    """The world as the page draws it."""
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
        'custom_kinds': list(culture_lib.KINDS),
        'domains': list(culture_lib.DOMAINS),
    }

  def person(self, name: str) -> dict[str, Any] | None:
    """A living person as the inspector shows it: customs, kin, body."""
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

  def move(self, move: str) -> None:
    """Queues the Experimenter's move (as the person typed or built it)."""
    move = ' '.join(str(move).split())
    if move:
      self._played = time.time()
      self.input.queue(move)

  def pace(
      self,
      years_per_second: float | None = None,
      paused: bool | None = None,
  ) -> dict[str, Any]:
    self._played = time.time()
    self.input.pace(years_per_second, paused)
    return self.input.settings()
