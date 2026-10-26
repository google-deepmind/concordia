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

The session keeps the person's lab notebook (DESIGN §9): what the person
writes (hypotheses, observations, notes), the moves the person makes, the
World's account of each (from the census, `reports`), and what changed in
the valley that a scientist would want noted. It is saved with the world.

The person may fork a twin: a second session, restored from the same year's
checkpoint with one lever pulled (`scenarios.with_lever`), plays beside the
first. The person's pace and releases reach both worlds, so the lever stays
the one difference between them; the page draws their census together.
"""

import collections
from collections.abc import Mapping, Sequence
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
# Variants nobody holds any more that the page remembers, for lineages.
FORGOTTEN_KEPT = 2000
# The most variants of one lineage the page is sent.
LINEAGE_SHOWN = 80
# What the person writes in the notebook, and the longest entry.
NOTE_KINDS = ('hypothesis', 'observation', 'note')
NOTE_LENGTH = 1000
# What the session writes there: the person's moves, the World's account of
# them, and changes in the valley.
MOVE, WORLD, EVENT = 'move', 'world', 'event'
# The notebook keeps every entry the person wrote and at most this many of
# its own.
NOTEBOOK_KEPT = 1000
# Fewer living than this is noted.
FEW_LEFT = 10
# The person's latest releases the page follows, and the years of each it
# sends.
RELEASES_FOLLOWED = 12
RELEASE_YEARS_SENT = 120


def _number(custom_id: str) -> int:
  """A custom's number, for ordering (c12 after c9)."""
  return int(custom_id[1:]) if custom_id[1:].isdigit() else 0


def await_engines(timeout: float = 10.0) -> None:
  """Waits for every engine loop in the process to finish its turn.

  A closed session's engine tells everyone's loop to end, but returns without
  waiting for loops in the middle of a turn. A process about to exit calls
  this once its sessions are closed, so that no turn is cut off half done.

  Args:
    timeout: How long to wait in all.
  """
  deadline = time.monotonic() + timeout
  for thread in threading.enumerate():
    if thread.name.startswith(ENGINE_THREADS):
      thread.join(max(0.0, deadline - time.monotonic()))


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
    self._seed = seed
    self._overrides = dict(overrides)
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
    # Variants the world forgot (nobody holds them), latest last.
    self._forgotten: collections.OrderedDict[str, dict[str, Any]] = (
        collections.OrderedDict()
    )
    # The lab notebook, oldest first; ids count up from 1.
    self._notebook: list[dict[str, Any]] = []
    self._entries = 0
    self._year = 0
    # The customs that have been the most widely held (noted once each).
    self._tops: set[str] = set()
    # The person's releases, followed year by year (`_follow_releases`).
    self._releases: list[dict[str, Any]] = []
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
    # The twin playing beside this world, and a snapshot taken for it.
    self.twin: Session | None = None
    self._twin_lock = threading.Lock()
    self._fork_wanted = False
    self._fork: dict[str, Any] | None = None
    # If this world is a twin: the lever pulled, its value, the year forked.
    self.twin_of: dict[str, Any] | None = None
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
    run.lifecycle.forgotten = self._forget
    return run

  def _forget(self, customs: Sequence[Mapping[str, Any]]) -> None:
    # In the World GM's lock, as the year closes.
    with self._rows_lock:
      for custom in customs:
        self._forgotten[str(custom['id'])] = dict(custom)
      while len(self._forgotten) > FORGOTTEN_KEPT:
        self._forgotten.popitem(last=False)

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
    """Ends the run, and its twin's: the Experimenter's next answer raises.

    Args:
      timeout: How long to wait for the run's thread.
    """
    deadline = time.monotonic() + timeout
    self.input.close()
    self.run.engine.play()
    if self._thread is not None:
      self._thread.join(timeout)
    twin = self.end_twin(noted=False)
    if twin is not None:
      twin.close(max(0.0, deadline - time.monotonic()))

  def _year_closed(self, year: int) -> None:
    # In the World GM's lock: the world may be read here, and is whole.
    world = self.run.world()
    row = {
        **(world.history[-1] if world.history else {'year': year - 1}),
        'held': self.run.customs_census(),
        'peoples': self.run.peoples(),
    }
    self._note_year(year, row)
    self._follow_releases(row)
    with self._rows_lock:
      self._rows.append(row)
    with self._saving:
      if self._fork_wanted:
        # A copy that shares nothing with the live world.
        self._fork = json.loads(json.dumps(self._snapshot(year), default=str))
        self._fork_wanted = False
        self._saving.notify_all()
      if self._save_path is not None and (
          self._save_wanted
          or time.monotonic() - self._last_save >= self._save_seconds
      ):
        self._save(self._save_path, year)

  # --- Saves ------------------------------------------------------------

  def _snapshot(self, year: int) -> dict[str, Any]:
    """The world at a year's close (in the World GM's lock), as saved."""
    checkpoint = self.run.sim.make_checkpoint_data()
    checkpoint['raw_log'] = []  # Nobody reads it; a restore needs none.
    with self._rows_lock:
      rows = list(self._rows)
      forgotten = list(self._forgotten.values())
      notebook = [dict(e) for e in self._notebook]
      tops = sorted(self._tops)
      releases = json.loads(json.dumps(self._releases))
    return {
        'version': SAVE_VERSION,
        'world': self._identity,
        'year': year,
        'rows': rows,
        'forgotten': forgotten,
        'notebook': notebook,
        'tops': tops,
        'releases': releases,
        'pace': self.input.settings(),
        'checkpoint': checkpoint,
    }

  def _save(self, path: str, year: int) -> None:
    """Writes the world at a year's close (in the World GM's lock)."""
    try:
      with open(path + '.tmp', 'w') as f:
        json.dump(self._snapshot(year), f, default=str)
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
      self._load(saved)
    except Exception as error:  # pylint: disable=broad-exception-caught
      # A save from an older build may no longer fit: begin anew.
      self.run = self._build()
      self._set_aside(path, f'it no longer restores ({error!r})')
      return
    self.restored_year = self.saved_year = int(saved['year'])

  def _load(self, saved: Mapping[str, Any]) -> None:
    """Loads a snapshot into this (new) session's world, with its rows."""
    simulation.load_checkpoint(self.run, saved['checkpoint'])
    with self._rows_lock:
      self._rows.extend(saved.get('rows', ()))
      for custom in saved.get('forgotten', ()):
        self._forgotten[str(custom['id'])] = dict(custom)
      self._notebook = [dict(e) for e in saved.get('notebook', ())]
      self._entries = max((int(e['id']) for e in self._notebook), default=0)
      self._tops = {str(t) for t in saved.get('tops', ())}
      self._releases = [dict(r) for r in saved.get('releases', ())]
    self._year = int(saved['year'])
    pace = saved.get('pace', {})
    self.input.pace(pace.get('years_per_second'), pace.get('paused'))
    if pace.get('paused'):
      self.run.engine.pause()
    for move in pace.get('queued', ()):
      self.input.queue(move)

  def _set_aside(self, path: str, why: str) -> None:
    logging.warning(
        'Lifeways: a new world begins; the save is set aside: %s', why
    )
    try:
      os.replace(path, path + '.set-aside')
    except OSError:
      pass

  # --- The person's releases -------------------------------------------

  def _follow_releases(self, row: Mapping[str, Any]) -> None:
    """Counts who holds each release or a variant of it (in the GM's lock).

    A release lives on while it or a variant descended from it is in the
    world (held, or kept in a place); it died out the first year none is.
    """
    world = self.run.world()
    closed = int(row['year'])
    customs = world.culture.get_state()['customs']
    with self._rows_lock:
      for release in row.get('releases', ()):
        custom = next(
            (c for c in customs if c['id'] == release['id']),
            self._forgotten.get(release['id'], {}),
        )
        self._releases.append({
            **release,
            'year': closed,
            'kind': custom.get('kind', ''),
            'domain': custom.get('domain', ''),
            'text': custom.get('text', ''),
            'holders': [],
            'peak': 0,
            'peak_year': closed,
            'variants': 0,
            'died': None,
        })
      del self._releases[:-RELEASES_FOLLOWED]
      following = {r['id'] for r in self._releases if r['died'] is None}
      if not following:
        return
      parents = {
          i: str(c.get('parent') or '') for i, c in self._forgotten.items()
      }
    parents.update({c['id']: str(c.get('parent') or '') for c in customs})
    found: dict[str, str] = {}

    def release_of(custom_id: str) -> str:
      """The followed release a variant descends from ('' if none)."""
      path, at = [], custom_id
      while at and at not in found and at not in following and at not in path:
        path.append(at)
        at = parents.get(at, '')
      root = at if at in following else found.get(at, '')
      for step in path:
        found[step] = root
      return root

    held = self.run.held_counts()
    holders: collections.Counter[str] = collections.Counter()
    variants: dict[str, set[str]] = collections.defaultdict(set)
    for custom_id, n in held.items():
      root = release_of(custom_id)
      if root:
        holders[root] += n
        variants[root].add(custom_id)
    alive = {release_of(c['id']) for c in customs} - {''}
    died = []
    with self._rows_lock:
      for release in self._releases:
        if release['died'] is not None:
          continue
        n = holders[release['id']]
        release['holders'] = (release['holders'] + [[closed, n]])[
            -RELEASE_YEARS_SENT:
        ]
        if n > release['peak']:
          release['peak'], release['peak_year'] = n, closed
        release['variants'] = len(variants[release['id']])
        if release['id'] not in alive:
          release['died'] = closed
          died.append(dict(release))
    for release in died:  # The notebook takes the lock itself.
      self._write(
          EVENT,
          f'Your release {release["id"]} died out: "{release["text"]}"'
          f' (at most {release["peak"]} held it).',
          closed,
      )

  def releases(self) -> list[dict[str, Any]]:
    """The person's latest releases, as followed year by year.

    With a twin, each release made since the fork also says how it fares
    there (`twin`): the twin's release of the same custom (its own id).

    Returns:
      The releases, oldest first.
    """
    with self._rows_lock:
      releases = json.loads(json.dumps(self._releases))
    twin = self.twin
    if twin is None or twin.twin_of is None:
      return releases
    forked = int(twin.twin_of['year'])
    theirs = twin.releases()
    for release in releases:
      if release['year'] < forked:
        continue
      same = [
          r
          for r in theirs
          if (r['kind'], r['domain'], r['text'])
          == (release['kind'], release['domain'], release['text'])
      ]
      if same:
        mirror = same[-1]
        release['twin'] = {
            'id': mirror['id'],
            'heard': mirror['heard'],
            'now': mirror['holders'][-1][1] if mirror['holders'] else 0,
            'peak': mirror['peak'],
            'died': mirror['died'],
        }
    return releases

  # --- The notebook -----------------------------------------------------

  def _write(
      self, kind: str, text: str, year: int | None = None
  ) -> dict[str, Any]:
    """Adds an entry; the oldest of the session's own go past the bound."""
    with self._rows_lock:
      self._entries += 1
      entry = {
          'id': self._entries,
          'year': self._year if year is None else year,
          'kind': kind,
          'text': text,
      }
      self._notebook.append(entry)
      own = [e for e in self._notebook if e['kind'] not in NOTE_KINDS]
      if len(own) > NOTEBOOK_KEPT:
        gone = own[0]['id']
        self._notebook = [e for e in self._notebook if e['id'] != gone]
      return dict(entry)

  def _note_year(self, year: int, row: Mapping[str, Any]) -> None:
    """Notes the World's reports and what changed (in the World GM's lock)."""
    self._year = year
    closed = int(row['year'])
    for report in row.get('reports', ()):
      self._write(WORLD, report, closed)
    for disaster in row.get('disasters', ()):
      if disaster['by'] != world_lib.EXPERIMENTER:  # Sent by the world.
        self._write(
            EVENT,
            f'A {disaster["kind"]} struck at'
            f' {disaster["tile"][0]},{disaster["tile"][1]}:'
            f' {disaster["near"]} lived within reach, {disaster["dead"]} died.',
            closed,
        )
    with self._rows_lock:
      last: Mapping[str, Any] = self._rows[-1] if self._rows else {}
    if 'population' not in row or 'population' not in last:
      return  # No census to compare: the first year, or none kept.
    for what, now, was in (
        ('peoples', row['peoples']['peoples'], last['peoples']['peoples']),
        ('tongues spoken', row.get('tongues'), last.get('tongues')),
    ):
      if now != was:
        self._write(
            EVENT, f'The valley has {now} {what} (it had {was}).', closed
        )
    living, lived = int(row['population']), int(last['population'])
    if living == 0 < lived:
      self._write(EVENT, 'Nobody is left in the valley.', closed)
    elif living < FEW_LEFT <= lived:
      self._write(EVENT, f'Only {living} people are left.', closed)
    top = row['held'].get('top') or []
    if top and top[0][0] not in self._tops:
      custom_id, share = top[0]
      self._tops.add(custom_id)
      world = self.run.world()
      text = (
          world.culture.get(custom_id).text
          if custom_id in world.culture
          else ''
      )
      self._write(
          EVENT,
          f'{custom_id} is the most widely held custom for the first time'
          f' ({round(100 * share)}% hold it): "{text}"',
          closed,
      )

  def note(self, kind: str, text: str) -> dict[str, Any]:
    """Writes the person's entry in the notebook, dated this year.

    Args:
      kind: One of `NOTE_KINDS`.
      text: What the person writes.

    Returns:
      The entry.

    Raises:
      ValueError: If the kind is not one, or the text is empty or too long.
    """
    text = str(text).strip()
    if kind not in NOTE_KINDS:
      raise ValueError(f'An entry is a {", ".join(NOTE_KINDS)}.')
    if not text or len(text) > NOTE_LENGTH:
      raise ValueError(f'Write 1 to {NOTE_LENGTH} characters.')
    self._played = time.time()
    return self._write(kind, text)

  def notebook(self, since: int = 0) -> list[dict[str, Any]]:
    """The notebook's entries after an id, oldest first.

    Args:
      since: An entry's id (0: all).

    Returns:
      The entries.
    """
    with self._rows_lock:
      return [dict(e) for e in self._notebook if e['id'] > since]

  def notebook_markdown(self) -> str:
    """The notebook as a Markdown page, to keep."""
    lines = [
        f'# Lifeways notebook: {self.scenario}, seed {self._seed}',
        '',
        f'Overrides: {json.dumps(self._overrides, default=str)}',
    ]
    year = None
    for entry in self.notebook():
      if entry['year'] != year:
        year = entry['year']
        lines += ['', f'## Year {year}', '']
      lines.append(f'- **{entry["kind"]}**: {entry["text"]}')
    return '\n'.join(lines) + '\n'

  # --- The twin ---------------------------------------------------------

  def fork(self, lever: str, value: Any, timeout: float = 10.0) -> 'Session':
    """Forks a twin at the next year's close, with one lever pulled.

    The twin is a new session of the same world restored from that year's
    checkpoint, with the lever's setting changed (`scenarios.with_lever`);
    it replaces any earlier twin and plays at once.

    Args:
      lever: A key of `scenarios.LEVERS`, or `scenarios.WITHOUT`.
      value: The lever's value, or the channel the twin goes without.
      timeout: How long to wait for the year to close.

    Returns:
      The twin.

    Raises:
      ValueError: If the lever or its value is not one.
      RuntimeError: If the world is not playing, or no year closed in time.
    """
    twin_settings = scenarios.with_lever(self._settings, lever, value)
    changed = {
        key: twin_settings[key]
        for key in twin_settings
        if twin_settings[key] != self._settings.get(key)
    }
    if self._thread is None or not self._thread.is_alive():
      raise RuntimeError('The world is not playing.')
    if self.input.settings()['paused']:
      raise RuntimeError("The world is paused: a twin forks at a year's close.")
    with self._saving:
      self._fork, self._fork_wanted = None, True
      self._saving.wait_for(lambda: self._fork is not None, timeout)
      saved, self._fork, self._fork_wanted = self._fork, None, False
    if saved is None:
      raise RuntimeError('No year closed in time to fork a twin.')
    twin = Session(
        self.scenario,
        seed=self._seed,
        years_per_second=self.input.settings()['years_per_second'],
        **{**self._overrides, **changed},
    )
    twin._load(saved)  # pylint: disable=protected-access
    twin.twin_of = {'lever': lever, 'value': value, 'year': saved['year']}
    pulled = (
        f'without {value}' if lever == scenarios.WITHOUT else f'{lever} {value}'
    )
    self._write(MOVE, f'You fork a twin in year {saved["year"]}: {pulled}.')
    with self._twin_lock:
      earlier, self.twin = self.twin, twin
    if earlier is not None:
      earlier.close()
    twin.start()
    return twin

  def end_twin(self, noted: bool = True) -> 'Session | None':
    """Lets go of the twin; the caller closes it.

    Args:
      noted: Whether the notebook says so (the person let it go).

    Returns:
      The twin, if there was one.
    """
    with self._twin_lock:
      twin, self.twin = self.twin, None
    if noted and twin is not None and twin.twin_of is not None:
      self._write(
          MOVE,
          f'You let the twin go (forked in year {twin.twin_of["year"]}); it'
          f' had reached year {twin.run.world_state()["year"]}.',
      )
    return twin

  def _twin_state(self) -> dict[str, Any] | None:
    twin = self.twin
    if twin is None or twin.twin_of is None:
      return None
    world = twin.run.world_state()
    return {
        **twin.twin_of,
        'year_now': world['year'],
        'people': len(world['people']),
        'running': twin._thread is not None and twin._thread.is_alive(),  # pylint: disable=protected-access
        'error': twin.error,
    }

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
        'child_acts_age': world_lib.CHILD_ACTS_AGE,
        'disaster_kinds': list(world_lib.DISASTERS),
        'disaster_radius': int(
            (self._settings.get('disasters') or {}).get(
                'radius', world_lib.DISASTER_RADIUS
            )
        ),
        # The chance each who hears a released song takes it up.
        'song_uptake': person_lib.DEFAULT_UPTAKE.get(culture_lib.RELEASE, 0),
        'sight': int(self._settings.get('sight', 2)),
        'max_level': culture_lib.MAX_LEVEL,
        'rows_kept': ROWS_KEPT,
        'releases': self.releases(),
        'saved_year': self.saved_year,
        'restored_year': self.restored_year,
        'twin': self._twin_state(),
        'levers': {
            lever: [low, high, self._settings.get(lever)]
            for lever, (low, high) in scenarios.LEVERS.items()
        },
        'channels': sorted(
            self._settings.get('channels', world_lib.DEFAULT_CHANNELS)
        ),
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
    came_by = {held['id']: held for held in learner.get_state()['held']}
    customs = [
        {
            'id': c.id,
            'kind': c.kind,
            'domain': c.domain,
            'text': c.spec['text'],
            'params': dict(c.spec.get('params', {})),
            'via': came_by.get(c.id, {}).get('via', ''),
            'from': came_by.get(c.id, {}).get('from', ''),
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

  def lineage(self, custom_id: str) -> dict[str, Any] | None:
    """A custom's family of variants, and who holds each and from whom.

    The variants are the world's living culture and the variants it forgot
    that the page remembers (`FORGOTTEN_KEPT`), from the oldest ancestor
    remembered down, depth first. Who holds each is read from the living's
    Learners (measurement): how it came to them, and from whom.

    Args:
      custom_id: Any variant of the lineage.

    Returns:
      The root, the parent lost beyond it (if any), the variants with their
      depth and holders (at most `LINEAGE_SHOWN`), and how many more there
      are; or None if the variant is not known.
    """
    world = self.run.world_state()
    with self._rows_lock:
      known: dict[str, dict[str, Any]] = {
          i: {**c, 'living': False} for i, c in self._forgotten.items()
      }
    known.update({
        str(c['id']): {**c, 'living': True} for c in world['culture']['customs']
    })
    if custom_id not in known:
      return None
    root, seen = custom_id, {custom_id}
    while True:
      parent = str(known[root].get('parent') or '')
      if parent not in known or parent in seen:
        break
      root = parent
      seen.add(root)
    children: dict[str, list[str]] = collections.defaultdict(list)
    for i, custom in known.items():
      if custom.get('parent'):
        children[str(custom['parent'])].append(i)
    holders = self._holders(world)
    variants, stack, visited = [], [(root, 0)], set()
    while stack:
      i, depth = stack.pop()
      if i in visited:
        continue
      visited.add(i)
      custom = known[i]
      variants.append({
          **{
              k: custom.get(k)
              for k in ('id', 'kind', 'domain', 'text', 'parent', 'origin')
          },
          'year': custom.get('year', 0),
          'params': {
              k: v
              for k, v in dict(custom.get('params', {})).items()
              if k != 'words'
          },
          'living': custom['living'],
          'depth': depth,
          'holders': holders.get(i, []),
      })
      stack.extend(
          (child, depth + 1)
          for child in sorted(children[i], key=_number, reverse=True)
      )
    return {
        'id': custom_id,
        'root': root,
        'lost_parent': str(known[root].get('parent') or ''),
        'variants': variants[:LINEAGE_SHOWN],
        'more': max(0, len(variants) - LINEAGE_SHOWN),
    }

  def _holders(
      self, world: Mapping[str, Any]
  ) -> dict[str, list[dict[str, str]]]:
    """Who holds each variant, how it came and from whom (their Learners)."""
    living = {p['name'] for p in world['people']}
    holders: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for entity in list(self.run.sim.entities):
      if entity.name not in living:
        continue
      learner = entity.get_component(  # pyrefly: ignore[missing-attribute]
          person_lib.LEARNER, type_=culture.Learner
      )
      for held in learner.get_state()['held']:
        holders[held['id']].append({
            'name': entity.name,
            'via': held['via'],
            'from': held.get('from', ''),
        })
    return holders

  def kin(self, name: str) -> dict[str, Any] | None:
    """A person's family, three generations, the living and remembered dead.

    Args:
      name: A living person, or one of the dead the world remembers.

    Returns:
      The person, partner, grandparents, parents, siblings, children and
      grandchildren, each with sex, birth, death (if dead) and descent; or
      None if nobody of that name is known.
    """
    world = self.run.world_state()
    living = {p['name']: p for p in world['people']}
    dead = {d['name']: d for d in world.get('dead', ())}
    if name not in living and name not in dead:
      return None

    def record(who: str) -> Mapping[str, Any]:
      return living.get(who) or dead.get(who) or {}

    def entry(who: str) -> dict[str, Any]:
      found = record(who)
      return {
          'name': who,
          'living': who in living,
          'known': bool(found),
          'sex': found.get('sex', ''),
          'born': found.get('born'),
          'died': found.get('died'),
          'descent': found.get('descent', ''),
      }

    parents = list(record(name).get('parents', ()))
    children = list(record(name).get('children', ()))
    siblings = sorted(
        {s for p in parents for s in record(p).get('children', ())} - {name}
    )
    partner = str(record(name).get('partner', '') or '')
    return {
        'person': entry(name),
        'partner': entry(partner) if partner else None,
        'grandparents': [
            entry(g) for p in parents for g in record(p).get('parents', ())
        ],
        'parents': [entry(p) for p in parents],
        'siblings': [entry(s) for s in siblings],
        'children': [entry(c) for c in children],
        'grandchildren': [
            entry(g) for c in children for g in record(c).get('children', ())
        ],
    }

  # --- What the page does -----------------------------------------------

  def stop(self) -> None:
    """The person stops the world: the Experimenter's next move ends it."""
    self._played = time.time()
    self.input.queue(world_lib.STOP)
    self._write(MOVE, 'You stop the world.')
    self.input.pace(paused=False)
    self.run.engine.play()

  def move(self, move: str) -> None:
    """Queues the Experimenter's move (as the person typed or built it).

    A release goes to the twin as well; a question or a stop does not.

    Args:
      move: The move.
    """
    move = ' '.join(str(move).split())
    if move:
      self._played = time.time()
      self.input.queue(move)
      self._write(MOVE, f'You queue a move: {move}')
      twin = self.twin
      if twin is not None and move.lower().startswith(world_lib.RELEASE + ' '):
        # A release reaches the twin too: the lever stays the one difference.
        twin.move(move)

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
    twin = self.twin
    if twin is not None:
      twin.pace(years_per_second, paused)
    return self.input.settings()
