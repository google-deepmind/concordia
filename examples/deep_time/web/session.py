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

"""A game you can play (or watch) from a phone: one Concordia run on a thread.

The simulation is the ordinary Concordia one (`simulation.build_simulation`,
the Simultaneous engine, every other entity on the mock model). In *play*
mode one role is a person: the role's entity acts through Concordia's
`HumanActComponent`, and `GameSession` is its `HumanInput` reader, a blocking
queue serviced by the web page. The page shows the request's context, which
is exactly what a model playing the role would read: the Steward's late,
noisy instruments, or Mission Control's relay from the colony, years old. The
true state of the world is revealed when the game ends. In *watch* mode the
mock plays every role, and each step is paced so a viewer can follow it.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-long-ternary,g-short-docstring-punctuation,missing-function-docstring

from collections.abc import Callable, Mapping, Sequence
import colorsys
import functools
import json
import math
import os
import re
import threading
import time
from typing import Any, cast
import zlib

from concordia.agents import entity_agent
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.agent import pubsub
from concordia.contrib.prefabs.entity import basic_with_async_deliberation
from examples.deep_time import actors as actors_lib
from examples.deep_time import crises as crises_lib
from examples.deep_time import forecast as forecast_lib
from examples.deep_time import model_setup
from examples.deep_time import motion as motion_lib
from examples.deep_time import scenarios
from examples.deep_time import simulation
from examples.deep_time import views
from examples.deep_time.components import world_step
from examples.deep_time.world import data
from examples.deep_time.world import media as media_lib
from examples.deep_time.world import publishing
from concordia.prefabs.entity import minimal
from concordia.typing import human_input
import numpy as np

NEWS_TOPICS = (
    'gaia.omens',
    'geo.events',
    'bio.events',
    'ocean.alerts',
    'civ.events',
    'colony.events',
)


class Stopped(Exception):  # pylint: disable=g-bad-exception-name
  """The game was cancelled: raised to the waiting entity and the engine."""


# How the page describes each kind of role, by the rules that drive it. The
# first two lead their scenarios and are offered first.
_KINDS = (
    (actors_lib.StewardActor, 'the Steward'),
    (actors_lib.MissionActor, ''),
    (actors_lib.SpeciesActor, 'a species'),
    (actors_lib.GrazerActor, 'a grazing herd'),
    (actors_lib.PolityActor, 'a people'),
    (actors_lib.ColonyActor, 'the colony'),
)


@functools.cache
def roles(scenario: str) -> tuple[tuple[str, str], ...]:
  """The roles a person can play in a scenario: (entity name, label).

  Every entity a model plays at the start (the Chronicler acts by a rule, and
  species that arise later cannot be chosen); the Steward, or Mission
  Control, comes first.
  """
  sim, context = simulation.build_simulation(
      scenario, model_setup.make_models('mock')
  )
  found = []
  for entity in sim.get_entities():
    actor = context.actors.get(entity.name)
    for rank, (cls, kind) in enumerate(_KINDS):
      if isinstance(actor, cls):
        label = f'{entity.name}, {kind}' if kind else entity.name
        found.append((rank > 1, entity.name, label))
        break
  found.sort(key=lambda role: role[0])  # Stable: leaders first.
  return tuple((name, label) for _, name, label in found)


# What a role does when its person runs out of time: nothing new.
# A sowing stays outlined on the life map this many ticks after a census
# first sees it.
PLANTINGS_SHOWN = 6

NEUTRAL_OPTIONS = (
    'wait and save omega',
    actors_lib.KEEP_LABOR,
    'hold steady',
    actors_lib.LISTEN,
    'browse lightly',
)


# Moves planned ahead: at most this many at once, each at
# most this many turns ahead (the forecast's course is six turns long).
MAX_PLANS = 3
PLAN_AHEAD = 5


def short_move(option: str) -> str:
  """An option without its price or push, to match it from turn to turn."""
  return option.split(' [')[0].split(' (')[0].strip().lower()


def neutral(options: Sequence[str]) -> str:
  """The option that changes least (the first option if none is known)."""
  return next((o for o in options if o in NEUTRAL_OPTIONS), options[0])


# What a turn's telemetry keeps of what the person looked at.
LOOKS = 24
LOOK_CHARS = 48


class Seat(human_input.HumanInput):
  """One person's place in a game: the `HumanInput` reader of their entity.

  `HumanActComponent` calls the seat and blocks until the person's page
  answers, the game is cancelled, or (with a turn timer) time runs out and
  the role makes its most neutral move.
  """

  def __init__(self, entity: str, role: str, turn_seconds: float | None = None):
    self.entity = entity
    self.role = role
    self.turn_seconds = turn_seconds
    self._asked_at = 0.0
    self._lock = threading.Lock()
    self._answered = threading.Event()
    self._request: human_input.HumanInputRequest | None = None
    self._answer = ''
    self._cancelled = False
    self.choices: list[str] = []  # What the person chose, in order.
    # The same moves with the tick each was sent on, how ('chose', 'held',
    # 'timed out') and where on the map it was aimed (for the page).
    self.sent: list[dict[str, Any]] = []
    # Every turn of the game, kept for play telemetry.
    self.played: list[dict[str, Any]] = []
    self.clock: Callable[[], int] | None = None
    # Holding course: the person's neutral move for this many more turns,
    # unless `guard` finds a reason to stop (then kept in `stopped`).
    self.hold = 0
    self.stopped = ''
    self.guard: Callable[[], str] | None = None
    # True once the game is decided: the engine's last question then needs
    # no one, and is answered with the neutral move at once.
    self.over: Callable[[], bool] | None = None
    # Seconds a held turn lasts, so the map can show each one go by.
    self.hold_pace = 1.5
    # How many of a turn's options carry a measured push, when it was sent.
    self.count_pushes: Callable[[Sequence[str]], int] = lambda offered: 0
    # The world as the turn stood, for play telemetry: what the
    # player read and what was true, to tell a rescue from a mistake.
    self.world: Callable[[], dict[str, Any]] = lambda: {}
    # Moves planned for later turns, by tick: on that turn
    # the seat sends the move itself, unless `can_make` says why it cannot
    # (then it asks the person, saying so in `stopped`).
    self.plans: dict[int, str] = {}
    self.can_make: Callable[[str], str] = lambda option: ''
    # What the person looked at while this turn was asked (the move sheet,
    # the cards they read, the folds they opened), kept with the turn.
    self.looked: list[str] = []
    # The moves a free question offers (a Steward that may speak an omen
    # answers in words, and its moves are still there to tap).
    self.moves: Callable[[], Sequence[str]] = lambda: ()
    # Why an answer to a free question cannot be taken ('' if it can): a
    # person's free question is a choice that may be aimed, so its answer
    # is one of the moves (unless the role may answer in words).
    self.check: Callable[[str], str] = lambda response: ''

  def offered(self, request: human_input.HumanInputRequest) -> tuple[str, ...]:
    """A question's options: its choices, or the moves of a free one."""
    return tuple(request.action_spec.options) or tuple(self.moves())

  def saved(self) -> dict[str, Any]:
    """What a save keeps of the seat."""
    with self._lock:
      return {
          'choices': list(self.choices),
          'sent': [dict(s) for s in self.sent],
          'played': [dict(t) for t in self.played],
          'plans': {str(t): m for t, m in self.plans.items()},
          'hold': self.hold,
          'stopped': self.stopped,
      }

  def restore(self, saved: Mapping[str, Any]) -> None:
    """Takes back what `saved` kept."""
    with self._lock:
      self.choices = list(saved.get('choices', ()))
      self.sent = [dict(s) for s in saved.get('sent', ())]
      self.played = [dict(t) for t in saved.get('played', ())]
      self.plans = {int(t): str(m) for t, m in saved.get('plans', {}).items()}
      self.hold = int(saved.get('hold', 0))
      self.stopped = str(saved.get('stopped', ''))

  def _planned(self, request: human_input.HumanInputRequest) -> str | None:
    """The move planned for this turn, if it can be made (under the lock)."""
    now = self.clock() if self.clock is not None else 0
    for missed in [t for t in self.plans if t < now]:
      del self.plans[missed]
    option = self.plans.pop(now, None)
    if option is None:
      return None
    options = self.offered(request)
    found = next((o for o in options if o == option), None) or next(
        (o for o in options if short_move(o) == short_move(option)), None
    )
    reason = self.can_make(found or option) or (
        'it is not on offer this turn' if found is None else ''
    )
    if reason or found is None:
      self.hold = 0
      self.stopped = f'your plan to {short_move(option)} failed: {reason}'
      return None
    self.choices.append(found)
    self._note(found, 'planned', list(options))
    return found

  def _held(self, request: human_input.HumanInputRequest) -> str | None:
    """The neutral move while holding course; None to ask the person."""
    options = self.offered(request)
    calm = [o for o in options if o in NEUTRAL_OPTIONS]
    if self.hold <= 0:
      return None
    if not calm:  # No move that changes nothing: the person must choose.
      self.hold, self.stopped = 0, 'this question needs your answer'
      return None
    reason = self.guard() if self.guard is not None else ''
    if reason:  # Something needs the person: stop holding and ask.
      self.hold, self.stopped = 0, reason
      return None
    self.hold -= 1
    self.choices.append(f'(holding course) {calm[0]}')
    self._note(calm[0], 'held', list(options))
    return calm[0]

  def _note(
      self,
      move: str,
      how: str,
      offered: Sequence[str] = (),
      seconds: float | None = None,
  ) -> None:
    """Keeps a sent move, with its tick, for the page (call under the lock).

    Args:
      move: The option sent.
      how: 'chose', 'held' or 'timed out'.
      offered: The options the turn offered (for play telemetry).
      seconds: How long the person took to answer, if they did.
    """
    tick = self.clock() if self.clock is not None else 0
    pushes = self.count_pushes(offered) if offered else 0
    self.sent.append({'tick': tick, 'move': move, 'how': how, 'target': None})
    del self.sent[:-60]
    self.played.append({
        'tick': tick,
        'move': move,
        'how': how,
        'offered': list(offered),
        'pushes': pushes,
        'seconds': None if seconds is None else round(seconds, 1),
        'looked': self.looked,
        **self.world(),
    })
    self.looked = []

  def look(self, what: str) -> None:
    """Notes that the person looked at something on the page this turn."""
    what = ' '.join(str(what).split())[:LOOK_CHARS]
    with self._lock:
      if what and what not in self.looked and len(self.looked) < LOOKS:
        self.looked.append(what)

  def hold_course(self, turns: int) -> None:
    """Makes the neutral move for `turns` turns (0 stops); answers now if asked."""
    with self._lock:
      self.hold, self.stopped = max(0, min(10, int(turns))), ''
      request = self._request
      held = self._held(request) if request is not None and self.hold else None
      if held is not None:
        self._answer = held
        self._request = None
    if held is not None:
      self._answered.set()

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    """Waits for the page's answer to `request` (raises if cancelled)."""
    with self._lock:
      if self._cancelled:
        raise Stopped()
      decided = self.over() if self.over is not None else False
      if decided:  # The game is decided: nothing to ask.
        self.hold = 0
        options = self.offered(request)
        return neutral(options) if options else ''
      planned = self._planned(request)
      if planned is not None:
        return planned
      held = self._held(request)
    if held is not None:
      deadline = time.monotonic() + self.hold_pace
      while time.monotonic() < deadline:  # Paced, but stops if cancelled.
        if self._cancelled:
          with self._lock:  # Never sent: forget the held move.
            for kept in (self.choices, self.sent, self.played):
              if kept:
                kept.pop()
          raise Stopped()
        time.sleep(0.05)
      return held
    with self._lock:
      if self._cancelled:
        raise Stopped()
      self._request = request
      self._asked_at = time.monotonic()
      self._answered.clear()
    answered = self._answered.wait(self.turn_seconds)
    with self._lock:
      if self._cancelled:
        self._request = None
        raise Stopped()
      if not answered and self._request is not None:
        # Out of time: the role makes its most neutral move.
        options = self.offered(request)
        self._answer = neutral(options) if options else ''
        self.choices.append(f'(out of time) {self._answer}')
        self._note(self._answer, 'timed out', list(options))
      self._request = None
      return self._answer

  def seconds_left(self) -> int | None:
    """Seconds before a pending question times out (None: no timer)."""
    with self._lock:
      if self._request is None or self.turn_seconds is None:
        return None
      left = self.turn_seconds - (time.monotonic() - self._asked_at)
      return max(0, int(left))

  def pending(self) -> human_input.HumanInputRequest | None:
    with self._lock:
      return self._request

  def choose(
      self, request_id: str, response: str, move: str | None = None
  ) -> str | None:
    """Answers the pending request; None if accepted, else why not.

    Args:
      request_id: The request answered.
      response: The answer the act component gets (a move, words, or an
        aimed move `"<move> @ row,col"`).
      move: The move as the page shows it, if not the response itself.

    Returns:
      None if accepted, otherwise why not.
    """
    with self._lock:
      request = self._request
      if request is None or request.request_id != request_id:
        return 'That question has already been answered.'
      try:
        human_act_component.validate_response(response, request.action_spec)
      except ValueError as error:
        return str(error)
      reason = self.check(response)
      if reason:
        return reason
      self._answer = response
      self._request = None  # Answered: a second tap must not count.
      self.choices.append(move or response)
      self._note(
          move or response,
          'chose',
          list(self.offered(request)),
          time.monotonic() - self._asked_at,
      )
    self._answered.set()
    return None

  def cancel(self) -> None:
    with self._lock:
      self._cancelled = True
    self._answered.set()


# How many entries the story feed keeps.
STORY_ITEMS = 40

# Replays: a decided game played with the mock's minds keeps its
# players' moves, so the same world can be played again exactly.
REPLAYS = 'replays'
REPLAYS_LISTED = 6


class ReplaySeat(human_input.HumanInput):
  """Answers a seat's turns with the moves its player made.

  Each recorded move is sent on its turn, aimed where it was aimed; a turn
  with nothing recorded (or a move not on offer) makes the neutral move.
  """

  def __init__(
      self,
      moves: Sequence[Sequence[Any]],
      clock: Callable[[], int],
  ):
    self._moves = {int(m[0]): (str(m[1]), m[2]) for m in moves}
    self._clock = clock
    # The moves of a free question (as `Seat.moves`).
    self.moves: Callable[[], Sequence[str]] = lambda: ()

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    options = tuple(request.action_spec.options) or tuple(self.moves())
    move, target = self._moves.get(self._clock(), (None, None))
    if move is None or (options and move not in options):
      return neutral(options) if options else ''
    if target:  # Aimed where it was aimed, in the action itself.
      return f'{move} @ {int(target[0])},{int(target[1])}'
    return with_draft(request, move)  # A draft's words, as the role reads.


# Moves that enact a slow mind's draft, and the role's context component that
# holds it: the move carries the draft's words ("<move>: <words>").
DRAFTS = {
    actors_lib.EVOLVE: 'Evolving toward',
    actors_lib.PUBLISH_BROADSHEET: "Our printers' sheet",
    actors_lib.PRINT_RETELLING: "Our printers' retelling",
    actors_lib.ADOPT_ORDERS: "Our scribes' draft",
}
_AGED = re.compile(r'\s*\[tick [^\]]*\]\s*$')


def with_draft(request: human_input.HumanInputRequest, move: str) -> str:
  """A move enacting a draft, with the draft's words as the role reads them."""
  for enact, key in DRAFTS.items():
    if not move.strip().lower().startswith(enact.lower()):
      continue
    if move.strip()[len(enact) :].lstrip().startswith(':'):
      return move  # Its words are given.
    text = _AGED.sub('', str(request.contexts.get(key, ''))).strip()
    if enact == actors_lib.ADOPT_ORDERS:  # The orders as the scribes wrote.
      text = text.split('As written: ', 1)[1] if 'As written: ' in text else ''
    return f'{move}: {text}' if text else move
  return move


def list_replays(telemetry_dir: str) -> list[dict[str, Any]]:
  """The newest replays there, without their moves (newest first)."""
  folder = os.path.join(telemetry_dir, REPLAYS) if telemetry_dir else ''
  try:
    names = [n for n in os.listdir(folder) if n.endswith('.json')]
  except OSError:
    return []
  names.sort(key=lambda n: -os.path.getmtime(os.path.join(folder, n)))
  found = []
  for name in names[:REPLAYS_LISTED]:
    record = load_replay(os.path.join(folder, name))
    if record is not None:
      found.append({k: v for k, v in record.items() if k != 'moves'})
  return found


def load_replay(path: str) -> dict[str, Any] | None:
  """A replay record, or None if missing or unreadable."""
  try:
    with open(path, encoding='utf-8') as f:
      found = json.load(f)
  except (OSError, ValueError):
    return None
  return found if isinstance(found, dict) and 'moves' in found else None


# Saves of played games: a folder beside the telemetry, one file a
# game, removed when the game is decided or given up.
SAVES = 'saves'
# 2: roles' params are plain (a species founded in play saves whole).
# 3: the World GM carries the readings in flight (`Deliveries`).
SAVE_VERSION = 3
# A limited rewind: a game may go back this many turns in all, to
# saves kept in memory for the last few turns.
REWINDS = 3
KEPT_TURNS = 4


def latest_save(telemetry_dir: str) -> dict[str, Any] | None:
  """The newest save there (without its checkpoint), or None."""
  folder = os.path.join(telemetry_dir, SAVES) if telemetry_dir else ''
  try:
    names = [n for n in os.listdir(folder) if n.endswith('.json')]
  except OSError:
    return None
  names.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)))
  for name in reversed(names):
    found = load_save(os.path.join(folder, name))
    if found is not None:
      return {k: v for k, v in found.items() if k != 'checkpoint'}
  return None


def load_save(path: str) -> dict[str, Any] | None:
  """A save file, or None if it is missing, torn or of another version."""
  try:
    with open(path, encoding='utf-8') as f:
      found = json.load(f)
  except (OSError, ValueError):
    return None
  if not isinstance(found, dict) or found.get('version') != SAVE_VERSION:
    return None
  return found


# A slow mind's answer on its way: (short label, what it is).
_MIND_WORDS = {
    'Advisor': ('Advisor', "The Advisor's next briefing"),
    'Adaptation': ('Evolution', 'Which trait to evolve next'),
    'Pact': ('Pact', 'An answer on a pact'),
    'Treaty': ('Treaty', 'An answer to a treaty offer'),
    'Press': ('Broadsheet', 'The next broadsheet'),
    'Policy': ('Orders', 'New standing orders'),
    'Chronicle': ('Chronicle', 'The next chronicle'),
}


class GameSession:
  """One game on a background thread, and the seats of the people in it.

  A pending question blocks one of the engine's worker threads, which Python
  joins at exit: `cancel` a game you no longer need, or use it as a context
  manager (`with GameSession(...).start() as game:`), which cancels on exit.
  """

  def __init__(
      self,
      scenario: str,
      mode: str = 'play',
      seed: int = 0,
      pace_seconds: float = 0.8,
      role: str | None = None,
      players: Sequence[str] | None = None,
      turn_seconds: float | None = None,
      telemetry_dir: str | None = None,
      minds: str = 'mock',
      max_calls: int = 1500,
      on_end: Callable[['GameSession'], None] | None = None,
      resume: Mapping[str, Any] | None = None,
      replay: Mapping[str, Any] | None = None,
      bots: Sequence[str] = (),
  ):
    """Builds the game (start it with `start`).

    Args:
      scenario: Key in `scenarios.SCENARIOS`.
      mode: 'play' (people play `players`) or 'watch' (the mock plays all).
      seed: Seed of the planet's random processes.
      pace_seconds: Pause after each step in watch mode.
      role: The entity one person plays (see `roles`); the Steward by default.
        Ignored if `players` is given.
      players: The entities people play, one seat (and phone) each; the first is
        the default view.
      turn_seconds: If set, a person who has not answered in this long makes
        their role's most neutral move, so no one stalls a shared game.
      telemetry_dir: If set, a played game writes how it went there when it ends
        (`telemetry`), for the play audit.
      minds: Who plays the roles no person plays, and voices the Advisor,
        Chronicler and presses: 'mock', 'together' or 'ollama'. A live mode
        without its credentials falls back to the mock.
      max_calls: The most calls each model may make in this game.
      on_end: Called with the game when it ends (e.g. to log its spend).
      resume: A save (`saved`) to play on from, of this scenario, seed and
        seats: its standard checkpoint is loaded and the seats take back what
          they had. A save may carry `rewinds` (turns already taken back) and
          `kept` (earlier turns' saves, compressed, for further rewinds).
      replay: A replay record (`replay_record`): the game is watched while its
        players' seats make the moves they made.
      bots: Roles the mock's skilled policies play, inside their own entities,
        whoever plays the rest (e.g. a watched game of live minds with a
        skilled Steward, to measure the minds).

    Raises:
      KeyError: For an unknown scenario or role.
      ValueError: For an unknown mode, or a role taken twice.
    """
    if scenario not in scenarios.SCENARIOS:
      raise KeyError(scenario)
    if mode not in ('play', 'watch'):
      raise ValueError(mode)
    self.scenario = scenario
    self.mode = mode
    self.seed = seed
    self._telemetry_dir = telemetry_dir
    self.pace = pace_seconds
    # Time controls for watching: paused, normal (1x) or fast (4x), and a
    # single step from a pause.
    self._base_pace = pace_seconds
    self.paused = False
    self._step_once = False
    labels = dict(roles(scenario))
    names = list(players or [role or next(iter(labels))])
    if len(set(names)) != len(names):
      raise ValueError(f'a role is taken twice: {names}')
    for name in [*names, *bots]:
      if name not in labels:
        raise KeyError(name)
    if mode == 'play' and set(bots) & set(names):
      raise ValueError(f'a role is played by a person and a bot: {bots}')
    self.seats = {
        name: Seat(name, labels[name], turn_seconds) for name in names
    }
    # A seat decides from its request and what the World GM publishes for
    # its role (`views`): its menu, the tick, the outcome, what needs it.
    for name, place in self.seats.items():
      place.over = self._decided
      place.clock = self._tick
      place.count_pushes = lambda offered, name=name: len(
          set(offered) & set(views.menu(self.context, name).get('pushes', {}))
      )
      place.can_make = lambda option, name=name: self._can_make(name, option)
      place.world = lambda name=name: self._turn_world(name)
      place.moves = lambda name=name: self._free_moves(name)
      place.check = lambda response, name=name: self._check(name, response)
    self.entity, self.role = names[0], labels[names[0]]  # The first seat.
    self._cancelled = False
    self.error = ''
    self.finished = False
    self.minds = minds
    self.models = model_setup.make_models(minds, max_calls=max_calls)
    self._on_end = on_end
    # Its place in the campaign once decided, set as it ends.
    self.chapter: dict[str, Any] | None = None
    humans: dict[str, human_input.HumanInput] | None = (
        dict(self.seats) if mode == 'play' else None
    )
    self.replay_of: str | None = None
    if replay is not None:  # Watched, with the players' moves played back.
      self.replay_of = str(replay.get('game_id', ''))
      humans = {}
      for name, moves in (replay.get('moves') or {}).items():
        if name not in labels:
          continue
        seat = ReplaySeat(moves, clock=self._tick)
        seat.moves = lambda name=name: self._free_moves(name)
        humans[name] = seat
    bot = (
        model_setup.make_models('mock', max_calls=max_calls).fast
        if bots
        else None
    )
    self._sim, self.context = simulation.build_simulation(
        scenario,
        self.models,
        seed=seed,
        humans=humans,
        bots={name: bot for name in bots} if bot is not None else None,
        # Live characters play without the playbook lines the mock's rules
        # read (a scenario flag).
        overrides=None if self.models.mock is not None else {'coached': False},
        # The World GM publishes each role's view for the page.
        views=mode,
    )
    self._thread = threading.Thread(target=self._run, daemon=True)
    # Identifies this game to pages caching its static terrain.
    self.game_id = f'{scenario}-{seed}-{id(self):x}'
    self._colours: dict[str, dict[str, list[int]]] = {}
    # Saves: a played game's standard checkpoint after every step,
    # beside the telemetry, so a restarted server can play it on.
    self._saves_dir = (
        os.path.join(telemetry_dir, SAVES) if telemetry_dir else ''
    )
    self.turn_seconds = turn_seconds
    self.started_at = time.monotonic()
    self.resumed_at: int | None = None
    self._spent_before: dict[str, Any] = {}
    # The last turns' saves, compressed, oldest first, for rewinds.
    self._kept: list[tuple[int, bytes]] = []
    self.rewinds = 0
    # Set when a rewind replaces this game: it then leaves its save, spend
    # and telemetry to the game that replaces it.
    self.superseded = False
    if resume is not None:
      simulation.load_checkpoint(self._sim, resume['checkpoint'])
      for name, kept in (resume.get('seats') or {}).items():
        if name in self.seats:
          self.seats[name].restore(kept)
      self.game_id = str(resume.get('game_id', self.game_id))
      self.resumed_at = self.context.planet.tick
      self._spent_before = dict(resume.get('spend') or {})
      self.rewinds = int(resume.get('rewinds', 0))
      self._kept = [(int(t), z) for t, z in resume.get('kept', ())]

  @property
  def choices(self) -> list[str]:
    """What the first seat's person chose, in order."""
    return self.seats[self.entity].choices

  def start(self) -> 'GameSession':
    self.started_at = time.monotonic()
    self._thread.start()
    return self

  def _run(self) -> None:
    try:
      # (No stdout redirection: it is process-wide, not per thread.)
      if self.resumed_at is None:
        self._sim.play(step_callback=self._paced, get_state_callback=self._save)
      else:  # Played on from a save: the premise was heard long ago.
        planet = self.context.planet
        self._sim.play(
            premise='',
            max_steps=max(1, planet.max_ticks - planet.tick + 2),
            step_callback=self._paced,
            get_state_callback=self._save,
        )
    except Stopped:
      pass
    except Exception as error:  # pylint: disable=broad-exception-caught
      self.error = f'{type(error).__name__}: {error}'
    finally:
      if not self.superseded:  # (A rewind's new game carries all on.)
        if self.context.planet.outcome or self._cancelled or self.error:
          self._drop_save()  # Decided or given up: nothing to play on.
        self._write_telemetry()
        self._write_replay()
      self.finished = True  # Once its records are written.
      if not self.superseded and self._on_end is not None:
        self._on_end(self)

  def spend(self) -> dict[str, Any]:
    """The game's minds and what they have cost so far."""
    used = model_setup.usage(self.models, self.minds)
    for key in ('calls', 'tokens', 'usd'):  # And before a restart.
      if key in self._spent_before and key in used:
        used[key] = type(used[key])(used[key] + self._spent_before[key])
    if 'usd' in used:
      used['usd'] = round(used['usd'], 4)
    return {'minds': self.minds, 'label': self.models.label, **used}

  def saved(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """A save of the game: the checkpoint, and what the seats hold."""
    planet = self.context.planet
    return {
        'version': SAVE_VERSION,
        'game_id': self.game_id,
        'scenario': self.scenario,
        'mode': self.mode,
        'seed': self.seed,
        'players': list(self.seats),
        'minds': self.minds,
        'turn_seconds': self.turn_seconds,
        'tick': planet.tick,
        'max_ticks': planet.max_ticks,
        'saved_at': time.strftime('%Y-%m-%d %H:%M'),
        'spend': self.spend(),
        'seats': {name: place.saved() for name, place in self.seats.items()},
        'rewinds': self.rewinds,
        'checkpoint': checkpoint,
    }

  def _save_path(self) -> str:
    return os.path.join(self._saves_dir, f'{self.game_id}.json')

  def _save(self, checkpoint: Mapping[str, Any]) -> None:
    """Saves the game after a step (played games only).

    The save is kept in memory for the last few turns (compressed, for
    rewinds) and written beside the telemetry (to play on after a restart).

    Args:
      checkpoint: The simulation's standard checkpoint after the step.
    """
    if self.mode != 'play' or self._cancelled or self.context.planet.outcome:
      return  # Watched, given up or decided: nothing to play on.
    text = json.dumps(self.saved(checkpoint), default=str)
    tick = self.context.planet.tick
    self._kept = [k for k in self._kept if k[0] < tick][-(KEPT_TURNS - 1) :]
    self._kept.append((tick, zlib.compress(text.encode('utf-8'), 1)))
    if not self._saves_dir:
      return
    try:
      os.makedirs(self._saves_dir, exist_ok=True)
      path = self._save_path()
      with open(path + '.tmp', 'w', encoding='utf-8') as f:
        f.write(text)
      os.replace(path + '.tmp', path)  # Whole, or the last one stays.
    except OSError:
      pass  # A save never stops a game.

  def rewind_save(self) -> tuple[dict[str, Any] | None, str]:
    """The save a rewind goes back to, or None and why not.

    Going back a turn replays the turn before the one being asked: the save
    from the end of the turn before that. The returned save carries the
    earlier kept saves, the rewinds used (one more) and all the spend so
    far (live minds' calls are not refunded by going back).
    """
    if self.mode != 'play':
      return None, 'Only a played game can go back.'
    if len(self.seats) > 1:  # It would take a friend's turn back too.
      return None, 'Only a game for one player can go back.'
    if self.finished or self.context.planet.outcome:
      return None, 'The game is decided.'
    if self.rewinds >= REWINDS:
      return None, f'All {REWINDS} rewinds are used.'
    target = self.context.planet.tick - 2
    found = [(t, z) for t, z in self._kept if t == target]
    if not found:
      return None, 'There is no earlier turn kept to go back to.'
    save = json.loads(zlib.decompress(found[-1][1]).decode('utf-8'))
    save['kept'] = [(t, z) for t, z in self._kept if t < target]
    save['rewinds'] = self.rewinds + 1
    save['spend'] = self.spend()
    return save, ''

  def replay_record(self) -> dict[str, Any]:
    """A decided game's replay: its world and its players' moves."""
    planet = self.context.planet
    moves = {}
    for name, place in self.seats.items():
      aimed = {int(s['tick']): s.get('target') for s in place.sent}
      moves[name] = [
          [int(t['tick']), t['move'], aimed.get(int(t['tick']))]
          for t in place.played
      ]
    return {
        'version': 1,
        'game_id': self.game_id,
        'scenario': self.scenario,
        'title': str(self.context.scenario.get('title', self.scenario)),
        'seed': self.seed,
        'players': list(self.seats),
        'outcome': planet.outcome,
        'stars': planet.star_count(),
        'ticks': planet.tick,
        'when': time.strftime('%Y-%m-%d %H:%M'),
        'moves': moves,
    }

  def _write_replay(self) -> None:
    """Keeps a decided played game for replays, if the mock played it.

    Live minds answer differently each time, so only the mock's games can be
    played again exactly.
    """
    if (
        not self._telemetry_dir
        or self.mode != 'play'
        or self.minds != 'mock'
        or not self.context.planet.outcome
    ):
      return
    folder = os.path.join(self._telemetry_dir, REPLAYS)
    try:
      os.makedirs(folder, exist_ok=True)
      with open(
          os.path.join(folder, f'{self.game_id}.json'), 'w', encoding='utf-8'
      ) as f:
        json.dump(self.replay_record(), f)
    except OSError:
      pass  # A replay never stops a game.

  def _drop_save(self) -> None:
    if self._saves_dir:
      try:
        os.remove(self._save_path())
      except OSError:
        pass

  def telemetry(self) -> dict[str, Any]:
    """How a played game went, turn by turn, per seat.

    Each turn: its tick, the move, how it was made (chose, held, timed out),
    how many moves were on offer and how many of them carry a measured push,
    whether it was the role's neutral move, and how long the person took.
    """
    planet = self.context.planet
    seats = {}
    for name, place in self.seats.items():
      turns = []
      for turn in list(place.played):
        offered = turn['offered']
        turns.append({
            'tick': turn['tick'],
            'move': turn['move'],
            'how': turn['how'],
            'options': len(offered),
            'pushes': turn['pushes'],
            'neutral': turn['move'] in NEUTRAL_OPTIONS,
            'seconds': turn['seconds'],
            'looked': list(turn.get('looked', ())),
            **{
                k: turn[k]
                for k in ('omega', 'read_c', 'forecast_c', 'true_c', 'life')
                if k in turn
            },
        })
      seats[name] = {'role': place.role, 'turns': turns}
    return {
        'version': 1,
        'game_id': self.game_id,
        'scenario': self.scenario,
        'mode': self.mode,
        'seed': self.seed,
        'ticks': planet.tick,
        'max_ticks': planet.max_ticks,
        'outcome': planet.outcome or ('abandoned' if self._cancelled else ''),
        'reason': planet.outcome_reason,
        'stars': planet.star_count(),
        'rewinds': self.rewinds,
        'minds': self.spend(),
        'seats': seats,
    }

  def _write_telemetry(self) -> None:
    """Writes `telemetry` for a played game, if a folder was given."""
    if not self._telemetry_dir or self.mode != 'play':
      return
    if not any(place.played for place in self.seats.values()):
      return  # Nobody played a turn.
    try:
      os.makedirs(self._telemetry_dir, exist_ok=True)
      stamp = time.strftime('%Y%m%d-%H%M%S')
      path = os.path.join(self._telemetry_dir, f'{stamp}-{self.game_id}.json')
      with open(path, 'w') as f:
        json.dump(self.telemetry(), f, indent=1)
    except OSError:
      pass  # Telemetry never stops a game.

  def _paced(self, step_data: Any) -> None:
    del step_data
    if self._cancelled:
      raise Stopped()
    if self.mode != 'watch':
      return
    if self._step_once:  # One step taken: pause again.
      self._step_once = False
      self.paused = True
    deadline = time.monotonic() + self.pace
    while self.paused or time.monotonic() < deadline:
      if self._cancelled:
        raise Stopped()
      if self._step_once:
        self.paused = False
        return
      time.sleep(0.05)

  def hold_course(self, seat: str | None, turns: int) -> str | None:
    """A player holds course for some turns (0 stops); None, or why not."""
    if self.mode != 'play' or self._spectating(seat):
      return 'Only a player can hold course.'
    if self.finished or self.context.planet.outcome:
      return 'The game is decided.'
    entity = seat if seat in self.seats else self.entity
    place = self.seats[entity]
    if place.guard is None:
      place.guard = lambda: self._needs_you(entity)
    place.hold_course(turns)
    return None

  def look(self, seat: str | None, what: str) -> str | None:
    """A player looked at something on the page; None, or why not noted."""
    if self.mode != 'play' or self._spectating(seat):
      return "Only a player's looks are kept."
    entity = seat if seat in self.seats else self.entity
    self.seats[entity].look(what)
    return None

  def plan(self, seat: str | None, move: str, turns: int) -> str | None:
    """A player plans a move for a later turn; None, or why not."""
    if self.mode != 'play' or self._spectating(seat):
      return 'Only a player can plan a move.'
    if self.finished or self.context.planet.outcome:
      return 'The game is decided.'
    entity = seat if seat in self.seats else self.entity
    place = self.seats[entity]
    if not 1 <= turns <= PLAN_AHEAD:
      return f'A move can be planned 1 to {PLAN_AHEAD} turns ahead.'
    with place._lock:  # pylint: disable=protected-access
      request = place._request  # pylint: disable=protected-access
      offered = (
          list(place.offered(request))
          if request is not None
          else (place.played[-1]['offered'] if place.played else [])
      )
      # Plannable too: the moves it is saving for, as the GM's menu says.
      offered += list(views.menu(self.context, entity).get('later', ()))
      if move not in offered:
        return 'That move is not one of yours.'
      if move in NEUTRAL_OPTIONS:
        return 'Holding course needs no plan.'
      tick = self._tick() + turns
      if tick in place.plans:
        return f'Turn {tick} already has a plan: one move a turn.'
      if len(place.plans) >= MAX_PLANS:
        return f'At most {MAX_PLANS} moves can be planned at once.'
      place.plans[tick] = move
    return None

  def cancel_plan(self, seat: str | None, tick: int) -> str | None:
    """Drops a player's plan for a turn; None, or why not."""
    if self.mode != 'play' or self._spectating(seat):
      return 'Only a player can cancel a plan.'
    place = self.seats[seat if seat in self.seats else self.entity]
    with place._lock:  # pylint: disable=protected-access
      if place.plans.pop(tick, None) is None:
        return f'Nothing is planned for turn {tick}.'
    return None

  def _omen(self, actor: Any, options: Sequence[str]) -> dict[str, Any] | None:
    """The omen's signs, words and price for the page's composer, or None."""
    if not isinstance(actor, actors_lib.StewardActor) or actor.omen is None:
      return None
    omen = actor.omen
    return {
        'option': omen.option,
        'signs': list(omen.effect['signs']),
        'max_words': int(omen.effect['max_words']),
        'cost': omen.cost,
        # Off its recharge and paid for: it may be spoken this turn.
        'ready': omen.option in options,
    }

  def _take_history(self) -> list[tuple[int, str, str]]:
    """Every people's recent takes on the omens, from its topic: (tick,
    people key, take), oldest first (words: the world keeps no record)."""
    planet = self.context.planet
    if planet.gazette is None or planet.civ is None:
      return []
    found = []
    for key in planet.civ.polities:
      for message in self.context.bus.history(f'civ.{key}.takes'):
        take = str(message.payload).strip()
        if take:
          tick = int(message.time) if message.time is not None else 0
          found.append((tick, message.seq, key, take))
    return [(t, key, take) for t, _, key, take in sorted(found)]

  def _takes(self) -> list[dict[str, Any]]:
    """Each people's newest take on an omen: {who, tick, text}."""
    planet = self.context.planet
    if planet.civ is None:
      return []
    newest = {polity: (t, take) for t, polity, take in self._take_history()}
    return [
        {'who': planet.civ.polities[k].name, 'tick': t, 'text': take}
        for k, (t, take) in sorted(newest.items())
        if k in planet.civ.polities
    ]

  def _free_moves(self, entity: str) -> list[str]:
    """The moves of a free question (a Steward that may speak an omen, or a
    person whose move may be aimed): the menu the World GM offered."""
    return list(views.menu(self.context, entity).get('offered', ()))

  def _check(self, entity: str, response: str) -> str:
    """Why a free answer is not a move on offer ('' if it is, or words)."""
    menu = views.menu(self.context, entity)
    if menu.get('words'):
      return ''  # The Steward may speak an omen in its own words.
    moves = list(menu.get('offered', ()))
    move, _ = actors_lib.aimed(response)
    move = move.strip()
    if moves and not any(
        move == m or (move.startswith(m) and move[len(m) :].lstrip()[:1] == ':')
        for m in moves
    ):
      return f'"{move}" is not one of the moves this turn.'
    return ''

  def _can_make(self, entity: str, option: str) -> str:
    """Why a planned move cannot be made now ('' if it can)."""
    return str(
        views.menu(self.context, entity).get('why_not', {}).get(option, '')
    )

  def _tick(self) -> int:
    """The tick, as the World GM announced it (`world.tick`)."""
    message = self.context.bus.latest('world.tick')
    return int(message.time or 0) if message is not None else 0

  def _decided(self) -> bool:
    """Whether the World GM has announced the game decided."""
    return self.context.bus.latest(world_step.OUTCOME_TOPIC) is not None

  def _crises(self, entity: str | None) -> list[dict[str, Any]]:
    """A role's crises, as the World GM worked them out (`views.crises`)."""
    if entity is None:
      return []
    return list(views.latest(self.context, entity, views.CRISES, []))

  def _turn_world(self, entity: str) -> dict[str, Any]:
    """A turn's world for telemetry: Ω, the last reading, the forecast, the truth.

    The climate for the Steward (it is its to read), and each people's hands
    and food where there are peoples: kept for review after the game, never
    shown during it.
    """
    actor = self.context.actors.get(entity)
    planet = self.context.planet
    # Each people's hands and food, to review live peoples by.
    peoples = _peoples_now(planet)
    if not isinstance(actor, actors_lib.StewardActor):
      return {'peoples': peoples} if peoples else {}
    found = forecast_lib.of_planet(planet, acts_in=self._acts_in(entity))
    seen = [
        r.get('believed_temp_c')
        for r in planet.history
        if r.get('believed_temp_c') is not None
    ]
    return {
        'omega': round(float(planet.omega), 1),
        'read_c': round(float(seen[-1]), 1) if seen else None,
        'forecast_c': found.now if found is not None else None,
        'true_c': round(float(planet.temperature_c()), 1),
        # The Omega meter is the Steward's own sense: exact, and its own.
        'life': round(float(planet.biodiversity()), 2),
        **({'peoples': peoples} if peoples else {}),
    }

  def _held_run(self, entity: str) -> dict[str, Any] | None:
    """The course a Steward just held, start and now, if it held 2+ turns."""
    place = self.seats[entity]
    if place.hold > 0 or not isinstance(
        self.context.actors.get(entity), actors_lib.StewardActor
    ):
      return None
    played = list(place.played)
    run = []
    for turn in reversed(played):
      if turn['how'] != 'held':
        break
      run.append(turn)
    if len(run) < 2:
      return None
    first, now = run[-1], self._turn_world(entity)
    pair = lambda k: [first.get(k), now.get(k)]
    return {
        'from': int(first['tick']),
        'to': int(run[0]['tick']),
        'turns': len(run),
        'read_c': pair('read_c'),
        'omega': pair('omega'),
        'life': pair('life'),
        'stopped': place.stopped,
    }

  def _advice(
      self, entity: str, options: Sequence[str]
  ) -> dict[str, Any] | None:
    """The move the Steward's last briefing recommends, if it is on offer."""
    message = self.context.received.get((entity, 'Advisor briefing'))
    if message is None or message.time is None:
      return None
    found = re.findall(r'Recommend:\s*([^.\[\n]+)', str(message.payload))
    if not found:
      return None
    pick = found[-1].strip().lower()
    move = next(
        (
            o
            for o in options
            if o.lower().startswith(pick) or short_move(o).lower() == pick
        ),
        None,
    )
    if move is None:
      return None
    latency = int(data.load('advisor.yaml')['latency_ticks'])
    return {
        'move': move,
        # Turns since the Advisor read the world it advises on.
        'age': self.context.planet.tick - int(message.time) + latency,
    }

  def _what_if(
      self, entity: str, options: Sequence[str]
  ) -> dict[str, dict[str, Any]]:
    """Each climate move's course in the Steward's forecast (its logbook)."""
    log = views.latest(self.context, entity, views.LOGBOOK, None)
    found = views.forecast(log, self._planned_moves(entity))
    if log is None or found is None:
      return {}
    out = {}
    band = tuple(log['band']) if log['band'] else (-1e9, 1e9)
    for option in options:
      key = log['keys'].get(option)
      shown = (
          forecast_lib.what_if(found, log['profiles'], key, band)
          if key is not None
          else None
      )
      if shown is not None:
        out[option] = shown
    return out

  def _planned_moves(self, entity: str) -> list[tuple[int, str]]:
    """A Steward seat's plans as (tick, intervention key), for forecasts."""
    place = self.seats.get(entity)
    if place is None:
      return []
    log = views.latest(self.context, entity, views.LOGBOOK, None) or {}
    keys = log.get('keys', {})
    return [
        (tick, keys[option])
        for tick, option in sorted(place.plans.items())
        if option in keys
    ]

  def _needs_you(self, entity: str) -> str:
    """Why a player holding course should look now, as the World GM says."""
    return str(views.latest(self.context, entity, views.NEEDS, ''))

  def set_speed(self, speed: str) -> str:
    """Watching: 'pause', 'play' (1x), 'fast' (4x) or 'step' (one tick)."""
    if speed in ('pause', 'play', 'fast'):
      self._step_once = False  # A step under way gives way to the new speed.
    if speed == 'pause':
      self.paused = True
    elif speed == 'play':
      self.paused, self.pace = False, self._base_pace
    elif speed == 'fast':
      self.paused, self.pace = False, self._base_pace / 4.0
    elif speed == 'step':
      self._step_once = True
    else:
      raise ValueError(f'Unknown speed {speed!r}')
    return self.speed()

  def speed(self) -> str:
    if self.paused:
      return 'pause'
    return 'fast' if self.pace < self._base_pace else 'play'

  def _targetable(self, entity: str, option: str) -> bool:
    """Whether a move may be aimed on the map, as the GM's menu says."""
    return option in views.menu(self.context, entity).get('aimable', ())

  def choose(
      self,
      request_id: str,
      response: str,
      target: tuple[int, int] | None = None,
  ) -> str | None:
    """Answers whichever seat's pending request has this id.

    Args:
      request_id: The request the page is answering; stale ids are refused.
      response: An option's exact text (or free text, for a free action).
      target: Where on the map the move should land (row from the north,
        column), for moves that can be aimed; the option text stays exact.

    Returns:
      None if accepted, otherwise why not (for the page to show).
    """
    for name, seat in self.seats.items():
      request = seat.pending()
      if request is not None and request.request_id == request_id:
        aim = target is not None and self._targetable(name, response)
        # A move enacting a draft carries its words, as the role reads them
        # (the move itself is what is kept, for the page and replays).
        action = with_draft(request, response)
        error = seat.choose(
            request_id,
            # The aim travels in the action; the actor sets it as the World
            # GM resolves the move.
            f'{action} @ {int(target[0])},{int(target[1])}'
            if aim and target is not None
            else action,
            move=response,
        )
        if error is None and aim and target is not None:
          seat.sent[-1]['target'] = [int(target[0]), int(target[1])]
        return error
    return 'That question has already been answered.'

  def cancel(self) -> None:
    """Ends the game at its next step."""
    self._cancelled = True
    for seat in self.seats.values():
      seat.cancel()

  def join(self, timeout: float | None = None) -> None:
    """Waits for the game's thread to end."""
    self._thread.join(timeout)

  def __enter__(self) -> 'GameSession':
    return self

  def __exit__(self, *exc_info: object) -> None:
    self.cancel()
    self.join(30)

  # What the page shows.

  def role_entity(
      self, entity: str | None = None
  ) -> entity_agent.EntityAgent | None:
    """A seat's entity (the first seat's by default)."""
    name = entity or self.entity
    for found in self._sim.get_entities():
      if found.name == name and isinstance(found, entity_agent.EntityAgent):
        return found
    return None

  def timing(self, entity: str | None = None) -> list[dict[str, Any]]:
    """How old each thing a role knows is (its own subscribers' readings).

    Every reading ends with its age ("[tick 3, 2 ticks old]", "[fresh]"):
    lag is the game's central mechanic, so the page shows it per source.
    """
    found = self.role_entity(entity)
    if found is None:
      return []
    now = self.context.planet.tick
    out = []
    for key, component in found.get_all_context_components().items():
      if not isinstance(component, pubsub.Subscriber):
        continue
      match = _AGE_RE.search(component.get_current_value())
      if match is None:
        continue
      age = 0 if match.group(2) == 'fresh' else int(match.group(3))
      if key == 'Advisor briefing':
        # As old as the readings the Advisor wrote it from, as the move it
        # recommends is tagged: it lands its latency after it read them.
        age += int(data.load('advisor.yaml')['latency_ticks'])
      out.append({'label': key, 'age': age, 'tick': now - age})
    return out

  def _story(
      self, entity: str | None, reveal: bool, counselled: bool
  ) -> list[dict[str, Any]]:
    """The story so far as a role may know it, newest first.

    The Advisor's briefings (to whom it counsels), the Gazette's
    broadsheets (public), the news the role has heard (its own digests) and,
    once the world is revealed, the Chronicler's chapters (written from the
    true world).

    Args:
      entity: The role, or None for a spectator.
      reveal: Whether the true world is shown.
      counselled: Whether the role hears the Advisor.

    Returns:
      Up to `STORY_ITEMS` entries: {tick, kind, who, text}.
    """
    bus = self.context.bus
    items: list[dict[str, Any]] = []
    if counselled:
      for message in bus.history('gaia.briefings'):
        items.append({
            'tick': int(message.time or 0),
            'kind': 'advisor',
            'who': 'Advisor',
            'text': message.payload,
        })
    seen_sheets = set()
    # The printers' front pages about the Steward's omens, in full.
    gazette = self.context.planet.gazette
    stances = gazette.stances if gazette is not None else {}
    for message in bus.history('media.omens'):
      body = media_lib.without_stance(message.payload, stances)
      seen_sheets.add(media_lib.headline(body))
      items.append({
          'tick': int(message.time or 0),
          'kind': 'press',
          'who': 'The Gazette on the omen',
          'text': body,
      })
    names = (
        {k: p.name for k, p in self.context.planet.civ.polities.items()}
        if self.context.planet.civ is not None
        else {}
    )
    for tick, polity, take in self._take_history():
      items.append({
          'tick': tick,
          'kind': 'take',
          'who': names.get(polity, polity),
          'text': take,
      })
    for message in bus.history('media.gazette'):
      try:
        sheets = json.loads(message.payload).get('broadsheets') or []
      except (TypeError, ValueError, AttributeError):
        continue
      for sheet in sheets:
        key = (sheet.get('author'), sheet.get('title'))
        if key in seen_sheets or sheet.get('title') in seen_sheets:
          continue
        seen_sheets.add(key)
        items.append({
            'tick': int(message.time or 0),
            'kind': 'press',
            'who': str(sheet.get('author', '')),
            'text': f'"{sheet.get("title", "")}"',
        })
    if entity is not None:
      for reading in self.instruments(entity):
        reading_data = reading.get('data') or {}
        if (
            'news' not in reading['label'].lower()
            or 'items' not in reading_data
        ):
          continue
        for item in reading_data['items']:
          items.append({
              'tick': int(item.get('tick') or 0),
              'kind': 'news',
              'who': reading['label'],
              'text': str(item.get('text', '')),
          })
    if reveal:
      for message in bus.history('chronicle'):
        items.append({
            'tick': int(message.time or 0),
            'kind': 'chronicle',
            'who': 'Chronicler',
            'text': message.payload,
        })
    order = {'chronicle': 0, 'advisor': 1, 'press': 2, 'take': 3, 'news': 4}
    items.sort(key=lambda i: (-i['tick'], order[i['kind']]))
    return items[:STORY_ITEMS]

  def instruments(self, entity: str | None = None) -> list[dict[str, Any]]:
    """A role's readings as data, as they reached it.

    One per instrument (a `pubsub.Subscriber` on one topic): its label,
    topic, age and the payload it last delivered (JSON as published, or
    {'text': ...}); None before its first reading. Readings that pass a
    transport are recorded as they arrived (late, noisy, or lost); others
    are the topic's latest message, which they receive at once. Digests of
    several topics (news) are left to the news list.
    """
    found = self.role_entity(entity)
    if found is None:
      return []
    name, context = found.name, self.context
    actor = context.actors.get(name)
    plain = dict(_PLAIN_TOPICS)  # A people's and the colony's own plans.
    if isinstance(actor, actors_lib.PolityActor):
      plain['Current plan'] = f'civ.{actor.polity}.plan'
    elif isinstance(actor, actors_lib.ColonyActor):
      plain['Current plan'] = 'colony.plan'
    ages = {t['label']: t['age'] for t in self.timing(entity)}
    # Swapped out of the Steward's attention, as the GM says (`views`).
    unwatched = set(views.latest(context, name, views.UNWATCHED, []))
    out = []
    for key, component in found.get_all_context_components().items():
      if not isinstance(component, pubsub.Subscriber):
        continue
      match = _TOPIC_RE.search(component.get_pre_act_label())
      topic = match.group(1) if match is not None else plain.get(key)
      if key == 'Advisor briefing':
        continue
      items = (
          _DIGEST_RE.findall(component.get_current_value())
          if topic is None or ',' in topic or 'news' in key.lower()
          else []
      )
      if topic is None or ',' in topic or items:  # A digest of news.
        if items:
          out.append({
              'label': key,
              'topic': 'news',
              'age': None,
              'data': {
                  'items': [
                      {'tick': int(t) if t else None, 'text': text.strip()}
                      for t, text in items[-6:]
                  ]
              },
          })
        continue
      if (name, key) in context.tapped:
        message = context.received.get((name, key))
      else:
        message = context.bus.latest(topic)
      payload: Any = None
      if message is not None:
        try:
          payload = json.loads(message.payload)
        except (TypeError, ValueError):
          payload = None
        if not isinstance(payload, (dict, list)):
          payload = {'text': str(message.payload)}
      if topic == 'bio.census' and isinstance(payload, dict):
        # Names and colours (public, as on the map) to draw the census by.
        keys = list(payload.get('cover', {}))
        payload['look'] = {
            k: {
                'name': context.planet.species_name(k),
                **self._species_look(k, keys),
            }
            for k in keys
        }
      if (
          isinstance(payload, dict)
          and 'tick' not in payload
          and message is not None
      ):
        if message.time is not None:  # When it was published, for trends.
          payload['tick'] = int(message.time)
      out.append({
          'label': key,
          'topic': topic,
          'age': ages.get(key),
          'data': payload,
          # Swapped out of the Steward's attention: the last reading, frozen.
          'unwatched': key in unwatched,
      })
    genome = views.latest(context, name, views.TRAITS, None)
    if genome is not None:  # A species knows its own make-up, always fresh.
      out.insert(
          0, {'label': 'Genome', 'topic': 'genome', 'age': 0, 'data': genome}
      )
    return out

  def in_flight(self, entity: str | None = None) -> list[dict[str, Any]]:
    """What is on its way to a role, and the turn it lands.

    The other half of lag: its own slow minds' answers (it knows what it is
    working on), and what the World GM has on its way to it: an adaptation
    phasing in, a transmission crossing the light delay, news a people's
    couriers are still carrying (`views.in_flight`).
    """
    found = self.role_entity(entity)
    if found is None:
      return []
    flight = self._deliberating(found)
    flight.extend(views.latest(self.context, found.name, views.IN_FLIGHT, []))
    return sorted(flight, key=lambda f: f['lands'])

  def _deliberating(
      self, found: entity_agent.EntityAgent
  ) -> list[dict[str, Any]]:
    """What a role's own slow minds are working on (their components)."""
    now = self._tick()
    labels = {
        id(mind.publisher): mind.label
        for mind in self.context.minds
        if mind.owner == found.name
    }
    out = []
    for key, component in found.get_all_context_components().items():
      if key == basic_with_async_deliberation.DELIBERATION_COMPONENT_KEY:
        label, about = 'Council plan', "Your council's next plan"
      elif id(component) in labels:
        mind = labels[id(component)]
        label, about = _MIND_WORDS.get(mind, (mind, mind))
      else:
        continue
      state = cast(dict[str, Any], component.get_state())
      tasks = cast(list[dict[str, Any]], state.get('in_flight_tasks', []))
      due = [int(t['due_tick']) for t in tasks if int(t['due_tick']) > now]
      if due:
        out.append(
            {'label': label, 'lands': min(due), 'kind': 'mind', 'about': about}
        )
    return out

  def _acts_in(self, entity: str) -> int:
    """Turns until a role's move acts (Mission Control's cross the void)."""
    actor = self.context.actors.get(entity)
    colony = self.context.planet.colony
    if isinstance(actor, actors_lib.MissionActor) and colony is not None:
      return colony.light_delay
    return 1

  def _moves(self, place: Seat, entity: str) -> list[dict[str, Any]]:
    """A player's last moves, newest first, each with what it did.

    A move is sent on a tick, resolved by the world at the end of that tick
    (its outcome, None until then), and acts from the next.
    """
    actor = self.context.actors.get(entity)
    done = {tick: outcome for tick, _, outcome in actor.log} if actor else {}
    return [
        {**move, 'outcome': done.get(move['tick'])}
        for move in reversed(place.sent[-8:])
    ]

  def _pins(self, seat: str | None) -> list[list[Any]]:
    """Where a player's aimed moves of the last few ticks landed.

    Each is [row (north first), col, tick sent, move].
    """
    if self.mode != 'play' or self._spectating(seat):
      return []
    entity = seat if seat in self.seats else self.entity
    now = self.context.planet.tick
    return [
        [*m['target'], m['tick'], m['move']]
        for m in self.seats[entity].sent[-12:]
        if m['target'] is not None and now - m['tick'] <= 4
    ]

  def _acting(self, seat: str | None) -> list[dict[str, Any]]:
    """A player's moves that act this turn, as the map shows them.

    Each is its look (`motion.look`), the move, where it was aimed (row
    north first, col) and the role's place: a people's name (its capital),
    a species' latitude from its own census, or the herds' kind.
    """
    if self.mode != 'play' or self._spectating(seat):
      return []
    entity = seat if seat in self.seats else self.entity
    place, actor = self.seats.get(entity), self.context.actors.get(entity)
    if place is None or actor is None:
      return []
    planet = self.context.planet
    role, where = 'steward', {}
    if isinstance(actor, actors_lib.PolityActor) and planet.civ is not None:
      role, where = 'people', {'people': planet.civ.polities[actor.polity].name}
    elif isinstance(actor, actors_lib.GrazerActor):
      role, where = 'herds', {}
    elif isinstance(actor, actors_lib.SpeciesActor):
      census = self.context.received.get((entity, 'Census'))
      try:
        lat = (
            crises_lib.lat_of(json.loads(census.payload), actor.species)
            if census is not None
            else None
        )
      except (TypeError, ValueError):
        lat = None
      role, where = 'species', ({'lat': lat} if lat is not None else {})
    elif isinstance(actor, actors_lib.ColonyActor):
      role, where = 'colony', {'site': True}
    elif isinstance(actor, actors_lib.MissionActor):
      role, where = 'mission', {'site': True}
    elif not isinstance(actor, actors_lib.StewardActor):
      return []
    acts = planet.tick - self._acts_in(entity)
    found = []
    for sent in place.sent[-12:]:
      if sent['tick'] != acts:
        continue
      shown = motion_lib.look(planet, role, sent['move'], NEUTRAL_OPTIONS)
      if shown is not None:
        found.append({
            **shown,
            **where,
            'move': sent['move'],
            'target': sent['target'],
        })
    return found

  def _reserves(self) -> list[list[Any]]:
    """Protected woods: [row (north first), col, radius in tiles, turns left].

    One per protected region, at its tiles' centre. Everyone sees them: the
    woodcutters find the woods closed.
    """
    planet = self.context.planet
    tiles = planet.tiles
    found = []
    for region, until in sorted(planet.reserves.items()):
      left = until - planet.tick
      if left <= 0:
        continue
      rows, cols = np.nonzero(tiles.region == region)
      if rows.size == 0:
        continue
      found.append([
          int(tiles.rows - 1 - round(float(rows.mean()))),
          int(round(float(cols.mean()))),
          round(float(np.sqrt(len(rows) / np.pi)), 1),
          int(left),
      ])
    return found

  def _couriers(self, seat: str | None) -> list[list[Any]]:
    """A people's player's news still on the road, as couriers on the map.

    Each is [from row, from col, to row, to col, sent tick, landing tick,
    kind of news], rows north first: the courier rides from where it
    happened to the people's capital, arriving on the turn the people reads
    it. Others see none (the ripples show the news spreading to everyone).
    """
    if self.mode != 'play' or self.finished or self._spectating(seat):
      return []
    settlements = self.context.planet.settlements
    if settlements is None or not settlements.config.get('couriers', True):
      return []
    entity = seat if seat in self.seats else self.entity
    actor = self.context.actors.get(entity)
    if not isinstance(actor, actors_lib.PolityActor):
      return []
    if self.context.planet.civ is None:
      return []
    return [
        [*f['from'], *f['to'], f['sent'], f['lands'], f['topic']]
        for f in self.in_flight(entity)
        if f['kind'] == 'news'
        if f['to'] is not None
    ]

  def readings(self, entity: str | None = None) -> list[dict[str, Any]]:
    """What a role's own pub/sub subscribers hold now (for watchers)."""
    found = self.role_entity(entity)
    if found is None:
      return []
    return [
        {'label': key, 'text': component.get_current_value()}
        for key, component in found.get_all_context_components().items()
        if isinstance(component, pubsub.Subscriber)
        and key != 'Advisor briefing'
    ]

  def goal(self, entity: str | None = None) -> str:
    """A role's own goal (the Steward's, a people's, ...): its entity's."""
    found = self.role_entity(entity)
    goal = ''
    if found is not None:
      component = found.get_all_context_components().get(
          minimal.DEFAULT_GOAL_COMPONENT_KEY
      )
      if isinstance(component, constant.Constant):
        goal = str(component.get_state().get('state', ''))
    if not goal:
      scenario = data.scenario(self.scenario)
      goal = str(scenario.get('steward_goal') or scenario.get('premise', ''))
    return ' '.join(goal.split())

  def _species_look(
      self, key: str, keys: Sequence[str] = ()
  ) -> dict[str, list[int]]:
    """A species' colours, light and dark, the same everywhere all game."""
    if key not in self._colours:
      colours = data.load('tilelife.yaml').get('colours', {})
      self._colours[key] = _species_colours(
          key,
          self.context.planet.daisies.traits(key).albedo,
          colours,
          [c['light'] for c in self._colours.values()]
          + [c['light'] for k, c in colours.items() if k in keys],
      )
    return self._colours[key]

  def _seen_life(self, seat: str | None) -> tuple[int, dict[str, Any]] | None:
    """The life a seat's own census last saw, and the tick it was taken.

    A role that reads a census (the Steward, a species, the herds) sees the
    map's life as that census saw it, as late as it arrives: a
    sowing shows where it was made once a census taken after it is read.
    """
    if self.mode != 'play' or self._spectating(seat):
      return None
    entity = seat if seat in self.seats else self.entity
    message = self.context.received.get((entity, 'Census'))
    if message is None or message.time is None:
      return None
    planet, taken = self.context.planet, int(message.time)
    covers = planet.life_seen.get(taken)
    if covers is None and taken == planet.tick:
      covers = dict(planet.tile_life.covers)
    return None if covers is None else (taken, covers)

  def _band_temps(self, seat: str | None) -> list[list[float]] | None:
    """Each latitude band's temperature in the seat's last census."""
    entity = seat if seat in self.seats else self.entity
    message = self.context.received.get((entity, 'Census'))
    try:
      bands = json.loads(message.payload).get('bands') if message else None
    except (TypeError, ValueError, AttributeError):
      bands = None
    found = []
    for band in bands or ():
      lat = re.match(r'(\d+(?:\.\d+)?)°\s*([NS]?)', str(band.get('band', '')))
      if lat and band.get('temp_c') is not None:
        sign = -1.0 if lat.group(2) == 'S' else 1.0
        found.append(
            [sign * float(lat.group(1)), round(float(band['temp_c']), 1)]
        )
    return sorted(found) or None

  def _plantings(self, seat: str | None, seen: int) -> list[dict[str, Any]]:
    """A Steward's aimed sowings its census has seen, lately."""
    entity = seat if seat in self.seats else self.entity
    if not isinstance(self.context.actors.get(entity), actors_lib.StewardActor):
      return []
    planet = self.context.planet
    return [
        {
            'sown': int(p['tick']) - 1,
            'species': p['species'],
            'name': planet.species_name(p['species']),
            'tiles': [
                ''.join('#' if v else '.' for v in row)
                for row in p['tiles'][::-1]
            ],
        }
        for p in planet.plantings
        if seen - PLANTINGS_SHOWN < int(p['tick']) <= seen
    ]

  def _life_tiles(
      self, covers: Mapping[str, Any] | None = None
  ) -> dict[str, Any]:
    """Sessile life per tile, north first (now, or as a census saw it).

    `dominant` is the commonest species (one character each, indexing
    `species`), `cover` the total cover in tenths (0-9) and `covers` each
    species' own cover in 36ths (one base-36 character per tile), which the
    page draws as a meadow: each tile a patch of dots, one per share.
    """
    planet = self.context.planet
    if covers is None:
      covers = dict(planet.tile_life.covers)  # One snapshot (game thread).
    keys, dominant, total = planet.tile_life.dominant(covers)
    for key in keys:  # A species keeps its colour for the whole game.
      self._species_look(key, keys)
    species = [
        {'key': key, 'name': planet.species_name(key), **self._colours[key]}
        for key in keys
    ]
    land = planet.tiles.land
    return {
        'species': species,
        'dominant': [
            ''.join(_BASE36[i] if l else '.' for i, l in zip(row, lrow))
            for row, lrow in zip(dominant[::-1], land[::-1])
        ],
        'cover': [
            ''.join(str(min(9, int(c * 10))) for c in row)
            for row in total[::-1]
        ],
        'covers': [
            [
                ''.join(_BASE36[min(35, int(round(c * 35)))] for c in row)
                for row in covers[key][::-1]
            ]
            for key in keys
        ],
    }

  def _believed(self, entity: str, topic: str, path: str) -> tuple[str, str]:
    """A role's latest reading of a field, and how old it is."""
    message = self.context.beliefs.get((entity, topic))
    value = self.context.believed(entity, topic, path)
    if message is None or value is None:
      return '?', 'no reading'
    if message.time is None:
      return f'{value:.1f}', 'read'
    age = max(0, self.context.planet.tick - int(message.time))
    return f'{value:.1f}', 'now' if age == 0 else f'{age} ticks old'

  def _stats(self, reveal: bool, entity: str) -> list[dict[str, str]]:
    """The strip of numbers: the truth if revealed, else what the role read.

    The Steward reads late, noisy instruments but its Omega meter is exact;
    Mission Control hears the colony years late; the colonists see their own
    sky; species and peoples see what their own beliefs last delivered.
    """
    planet = self.context.planet
    colony = planet.colony
    actor = self.context.actors.get(entity)
    stats: list[dict[str, Any]] = [
        # 'turn', as the dock says it.
        {'label': 'turn', 'value': f'{planet.tick}/{planet.max_ticks}'}
    ]
    omega: dict[str, Any] = {
        'label': 'Ω',
        'value': f'{planet.omega:.1f}',
        'exact': round(float(planet.omega), 4),  # For the moves' costs.
    }
    if colony is not None:
      target = f'{colony.target_temp_c:g}'
      if reveal or isinstance(actor, actors_lib.ColonyActor):
        sky = f'{colony.temp_c:.1f}', f'sky °C (→{target})'
        people = f'{colony.population:.1f}', 'colonists'
      else:
        value, age = self._believed(entity, 'colony.status', 'temp_c')
        sky = value, f'sky °C (→{target}), {age}'
        value, age = self._believed(entity, 'colony.status', 'colonists')
        people = value, f'colonists, {age}'
      # (No Omega: there is no Steward to spend it in Light Delay.)
      return stats + [
          {'label': sky[1], 'value': sky[0]},
          {'label': people[1], 'value': people[0]},
      ]
    if reveal:
      temp = {'label': '°C', 'value': f'{planet.temperature_c():.1f}'}
    else:
      value, age = self._believed(entity, 'climate.global', 'temp_c')
      temp = {'label': f'°C read, {age}', 'value': value}
    # The Omega meter is exact, and it reports biodiversity.
    bio = {'label': 'biodiversity', 'value': f'{planet.biodiversity():.2f}'}
    if reveal or isinstance(actor, actors_lib.StewardActor):
      return stats + [temp, omega, *(self._goal_stats(reveal, entity) or [bio])]
    if isinstance(actor, actors_lib.PolityActor) and planet.civ is not None:
      # A people knows its own storehouse exactly (its sheet says so).
      people = planet.civ.polities[actor.polity]
      grain = planet.civ.ledger.stock(actor.polity, 'grain')
      fed = [
          {'label': 'fed', 'value': f'{min(1.0, people.food_ratio):.0%}'},
          {'label': 'grain', 'value': f'{grain:.0f}'},
      ]
      # The goal all the peoples share, in place of the head count.
      goal = self._goal_stats(reveal, entity)
      if goal:
        return stats + fed + goal
      return stats + [
          {'label': 'people', 'value': f'{people.population:.1f}'},
          *fed,
      ]
    if isinstance(actor, actors_lib.SpeciesActor):
      # A species senses the census exactly and knows its own seed bank.
      stats.append({
          'label': 'cover',
          'value': f'{planet.daisies.cover(actor.species):.0%}',
      })
      if planet.seed_banks is not None:
        bank = planet.seed_banks.level.get(actor.species, 0.0)
        stats.append({'label': 'seed bank', 'value': f'{bank:.0%}'})
      return stats
    if (entity, 'climate.global') in self.context.beliefs:
      return stats + [temp]
    return stats

  def _goal_stats(self, reveal: bool, entity: str) -> list[dict[str, Any]]:
    """The numbers a peoples' goal is judged on, as a role reads them.

    The Steward reads the air through its thermometer (late and noisy) and
    what the peoples made through their census; a people reads both as they
    are published; the truth once revealed. A reading past its limit is
    marked. Empty where the goal has no such numbers.
    """
    planet = self.context.planet
    goal = planet.goal()
    if planet.civ is None or goal['kind'] not in ('industrialize', 'iron_age'):
      return []

    # A people reads the weather and the census as they are published.
    public = isinstance(self.context.actors.get(entity), actors_lib.PolityActor)

    def read(key: tuple[str, str], path: str, truth: float):
      """The truth if revealed, else the reading (or None) and its age."""
      if reveal:
        return truth, ''
      topic = {'Peoples': 'civ.census'}.get(key[1], key[1])
      message = (
          self.context.bus.latest(topic)
          if public
          else self.context.beliefs.get(key) or self.context.received.get(key)
      )
      if message is None:
        return None, ''
      value: Any = json.loads(message.payload)
      for part in path.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
      if not isinstance(value, (int, float)):
        return None, ''
      sent = message.time
      age = 0 if sent is None else max(0, planet.tick - int(sent))
      return value, f', {age} tick{"" if age == 1 else "s"} old' if age else ''

    found = []
    if goal['kind'] == 'industrialize':
      cap = float(goal['max_co2'])
      truth = planet.carbon.relative_co2() if planet.carbon else 1.0
      co2, age = read((entity, 'climate.global'), 'co2', truth)
      found.append({
          'label': f'CO₂ / cap{age}',
          'value': f'{"?" if co2 is None else f"{co2:.2f}"}/{cap:g}',
          'warn': co2 is not None and co2 >= cap,
      })
    good, need = (
        ('energy', goal['min_energy'])
        if goal['kind'] == 'industrialize'
        else ('iron', goal['min_iron'])
    )
    # The peoples' census reaches the Steward at once (its 'Peoples').
    made, age = read((entity, 'Peoples'), f'made.{good}', planet.civ.made(good))
    found.append({
        'label': f'{good} made{age}',
        'value': f'{"?" if made is None else f"{made:.0f}"}/{float(need):g}',
    })
    return found

  def _peoples(self, seat: str | None, reveal: bool) -> dict[str, Any] | None:
    """Towns, roads and routes; a people's planned town only to itself."""
    planet = self.context.planet
    if planet.settlements is None:
      return None
    found = planet.settlements.payload(data.load('settlements.yaml')['colours'])
    if planet.civ is not None:  # How each people eats: hunger shows from afar.
      for p in found['polities']:
        polity = planet.civ.polities.get(p['key'])
        if polity is not None:
          p['fed'] = round(float(polity.food_ratio), 2)
          p['famine'] = bool(polity.famine)
    return found

  def worlds(self) -> list[dict[str, str]]:
    """The worlds this game can map, the one to show first leading."""
    planet = self.context.planet
    if planet.colony_world is None:
      return [{'key': 'home', 'name': 'Home'}]
    return [
        {'key': 'ararat', 'name': planet.colony_world.colony.name},
        {'key': 'home', 'name': 'Home'},
    ]

  def ararat_map(self, seat: str | None, full: bool) -> dict[str, Any]:
    """The colony's own world (Light Delay): tiles, ice, biomes, domes.

    The colonists see it as it is. Everyone else at home (Mission Control)
    sees it as its light shows it, years late, until the reveal.
    """
    planet = self.context.planet
    world = planet.colony_world
    assert world is not None
    tiles, colony = world.tiles, world.colony
    reveal = self.mode == 'watch' or self.finished
    entity = seat if seat in self.seats else self.entity
    on_site = isinstance(
        self.context.actors.get(entity), actors_lib.ColonyActor
    )
    frame = world.frames[-1] if reveal or on_site else world.frames[0]
    late = world.frames[-1]['generation'] - frame['generation']
    north = tiles.rows - 1
    colours = data.load('ararat.yaml')['colour']
    found: dict[str, Any] = {
        'version': 1,
        'game_id': f'{self.game_id}-ararat',
        'world': 'ararat',
        'worlds': self.worlds(),
        'title': (
            f'{colony.name}, four light-years from home: frozen at landfall,'
            ' thawing as the sky warms'
            + (f' (as its light shows it, {late} years ago)' if late else '')
        ),
        'tick': planet.tick,
        'rows': world.grid.bands,
        'cols': world.grid.sectors,
        'tile_rows': tiles.rows,
        'tile_cols': tiles.cols,
        'layers': (
            ['biomes', 'terrain']
            + (['temperature', 'rainfall', 'seasons'] if reveal else [])
        ),
        'regions': [{'row': 0, 'col': 0}],  # The page needs one; none drawn.
        'ice': frame['ice'],
        'biomes': frame['biomes'],
        'tile_temp': (
            [
                [int(round(v)) for v in row]
                for row in world.biomes.temperature[::-1]
            ]
            if reveal
            else None
        ),
        'rain': (
            [
                [int(round(v / 10)) * 10 for v in row]
                for row in world.biomes.rainfall[::-1]
            ]
            if reveal
            else None
        ),
        'peoples': {
            'polities': [{
                'key': 'colony',
                'name': colony.name,
                'unit': 'dome',
                'units': 'domes',
                'label': 'above',  # Domes ring the landing on all sides.
                **colours,
            }],
            'settlements': [
                [0, i + 1, north - r, c, int(i == 0), 0]
                for i, (r, c) in enumerate(frame['domes'])
            ],
            'routes': [],
        },
        'site': (
            [north - colony.site[0], colony.site[1]]
            if colony.site is not None
            else None
        ),
        'ripples': [],
    }
    if full:
      found['terrain'] = tiles.payload()
      found['biome_legend'] = world.biomes.legend()
      found['seasonality'] = _seasonality(world.biomes)
    return found

  def map(
      self, seat: str | None = None, full: bool = False, world: str = ''
  ) -> dict[str, Any]:
    """The map one seat's page draws: tiles over the 48 regions.

    Terrain is public and static, so it is sent only on request (`full`,
    once per game: the page keys it by `game_id`). Climate, ice and life are
    the truth, so a player sees them only once the game is revealed (a
    watcher always does); before that, a player sees the life its own
    census last saw, as late as that census.

    Args:
      seat: The viewing seat (as in `state`).
      full: Include the static terrain (tile classes and elevation).
      world: Which world ('home' or, in Light Delay, 'ararat'); empty for the
        game's first.

    Returns:
      Regions (north first) with their layers, tile ice and, if `full`, the
      terrain.
    """
    world = world or self.worlds()[0]['key']
    if world == 'ararat' and self.context.planet.colony_world is not None:
      return self.ararat_map(seat, full)
    planet = self.context.planet
    grid = planet.grid
    reveal = self.mode == 'watch' or self.finished
    seen = None if reveal else self._seen_life(seat)
    # Fire danger: the truth once revealed, else as the census saw it.
    risk = (
        planet.fire.risk(dict(planet.tile_life.covers))
        if reveal
        else planet.risk_seen.get(seen[0])
        if seen is not None
        else None
    )
    land = planet.daisies.land_fraction()
    temps = planet.daisies.temperatures_c()
    ice = planet.climate.ice_fraction(temps)
    life = np.zeros(grid.size)
    for species in planet.daisies.species():
      life = life + planet.daisies.cover_map(species)
    regions = []
    for r in range(grid.size):
      band, sector = grid.band_of(r), grid.sector_of(r)
      item: dict[str, Any] = {
          'region': r,
          'row': grid.bands - 1 - band,
          'col': sector,
          'name': grid.region_name(r),
          'land': round(float(land[r]), 3),
      }
      if reveal:
        item['temp_c'] = round(float(temps[r]), 1)
        item['ice'] = round(float(ice[r]), 3)
        item['life'] = round(float(life[r]) / max(1e-9, float(land[r])), 3)
      regions.append(item)
    tiles = planet.tiles
    found: dict[str, Any] = {
        'version': 1,
        'game_id': self.game_id,
        'tick': planet.tick,
        # News riding toward a people's player's capital.
        'couriers': self._couriers(seat),
        # Where the player's aimed moves landed.
        'pins': self._pins(seat),
        # The player's moves acting this turn, and how they show.
        'acting': self._acting(seat),
        # Woods the Steward protects from the peoples' axes.
        'reserves': self._reserves(),
        'rows': grid.bands,
        'cols': grid.sectors,
        'tile_rows': tiles.rows,
        'tile_cols': tiles.cols,
        'title': data.scenario(self.scenario).get('map_title'),
        'world': 'home',
        'worlds': self.worlds(),
        'layers': _with_seen_life(
            [
                layer
                for layer in _map_layers(reveal)
                # Fire risk only where something can burn.
                if layer != 'fire'
                or any(
                    planet.fire.flammability(k) > 0
                    for k in planet.daisies.species()
                )
            ]
            + (['pollution'] if planet.pollution is not None else []),
            seen=not reveal and seen is not None,
            # Life first where life is the world's story; after the biomes
            # where peoples are.
            first=planet.civ is None,
            # Where something can burn, its danger as the census saw it.
            burns=any(
                planet.fire.flammability(k) > 0
                for k in planet.daisies.species()
            ),
        ),
        'regions': regions,
        'ice': tiles.ice_rows() if reveal else None,
        'biomes': planet.biomes.rows(),
        'tile_temp': (
            [
                [int(round(v)) for v in row]
                for row in planet.biomes.temperature[::-1]
            ]
            if reveal
            else None
        ),
        'rain': (
            [
                [int(round(v / 10)) * 10 for v in row]
                for row in planet.biomes.rainfall[::-1]
            ]
            if reveal
            else None
        ),
        'life_tiles': (
            self._life_tiles()
            if reveal
            else self._life_tiles(seen[1])
            if seen is not None
            else None
        ),
        # The player's own sowings that census has seen, and where they
        # took root ('#' per tile, north first): what they did, on the map.
        'plantings': (
            [] if reveal or seen is None else self._plantings(seat, seen[0])
        ),
        # When the life shown was seen, for a player: their census's tick
        # and how many turns old it is.
        'life_seen': (
            None
            if reveal or seen is None
            else {'tick': seen[0], 'age': planet.tick - seen[0]}
        ),
        # Fires and their scars are seen from afar, like the biomes ('3'
        # burned this tick, fading to '0'); the risk is a number, the truth.
        'herds': planet.herds.payload(),  # Animals are seen by all.
        # News of places, spreading at the couriers' pace: where it
        # happened, how many ticks ago, what kind, and what it said to those
        # who may read it here (watchers, and the Steward who hears world
        # news); a people's player learns it only as its couriers bring it.
        'ripples': [
            [
                tiles.rows - 1 - r['row'],
                r['col'],
                planet.tick - r['tick'],
                # Biomes shifting (a tally of the planet) ripple softly.
                r['topic'].split('.')[0] if r.get('local', True) else 'shift',
                str(r['text'])
                if self._reads_news(seat, reveal, r['topic'].split('.')[0])
                else None,
            ]
            for r in planet.ripples
        ],
        'news_speed': tiles.per / float(planet.news_config['ticks_per_region']),
        'pollution': (  # Smog and blooms are seen from afar too.
            planet.pollution.payload() if planet.pollution is not None else None
        ),
        'peoples': self._peoples(seat, reveal),  # Towns and roads: public.
        # Clouds, haze and ash in the sky (seen from afar): the albedo pulses
        # over each region, north first; none while the sky is clear.
        'clouds': _clouds(planet),
        # Rain-soaked ground (seen from afar, like the clouds that brought it).
        'wet': (
            [
                ''.join(str(min(9, int(v * 10))) for v in row)
                for row in planet.fire.wet[::-1]
            ]
            if planet.fire.wet.max() > 0.05
            else None
        ),
        'burns': [
            ''.join(str(v) for v in row) for row in planet.fire.scar[::-1]
        ],
        'fire_risk': (
            [
                ''.join(str(min(9, int(v * 10))) for v in row)
                for row in risk[::-1]
            ]
            if risk is not None
            else None
        ),
        # The temperature each latitude band had in that census, for a
        # player's temperature layer: [degrees of latitude, °C].
        'band_temps': (
            None if reveal or seen is None else self._band_temps(seat)
        ),
    }
    if full:
      found['terrain'] = tiles.payload()
      found['biome_legend'] = planet.biomes.legend()
      found['seasonality'] = _seasonality(planet.biomes)
    return found

  def _own_people(self, entity: str) -> dict[str, Any] | None:
    """The seat's own people: numbers, food, where its hands go, stores."""
    actor = self.context.actors.get(entity)
    civ = self.context.planet.civ
    if not isinstance(actor, actors_lib.PolityActor) or civ is None:
      return None
    polity = civ.polities.get(actor.polity)
    if polity is None:
      return None
    stocks = civ.payload()['polities'][actor.polity]['stocks']
    population = float(polity.population)
    eats = population * float(civ.food_per_pop)
    # The goal the peoples share, as their census already tells them, with
    # the pace it asks and the people's numbers against its floor.
    shared = crises_lib.shared_goal(self.context.planet, actor.polity)
    if shared and 'good' in shared:
      shared['label'] = (
          f'{shared["good"]} {"forged" if shared["good"] == "iron" else "made"}'
          ' by all peoples'
      )
    return {
        'name': polity.name,
        'population': round(population, 1),
        'fed': round(float(polity.food_ratio), 2),
        # How many seasons the granary would feed the people.
        'seasons': round(float(stocks.get('grain', 0.0)) / max(eats, 1e-9), 1),
        'shared': shared,
        # The road to what the goal asks for, as the people's sheet says it.
        'aims': [
            civ.aim(actor.polity, str(aim))
            for aim in self.context.scenario['civilization'].get('aims', ())
        ],
        'labor': [
            {
                'key': k,
                'label': civ.recipe(k).label,
                'share': round(float(v), 3),
            }
            for k, v in sorted(polity.allocation.items(), key=lambda kv: -kv[1])
            if v > 0
        ],
        # Every trade's label -> key, so new trades in the moves get colours.
        'trades': {r.label: r.key for r in civ.recipes},
        'stocks': {
            k: round(float(v), 1) for k, v in stocks.items() if float(v) > 0.05
        },
    }

  def _spectating(self, seat: str | None) -> bool:
    """An unknown seat in a game for several: a spectator's view."""
    return (
        self.mode == 'play' and seat not in self.seats and len(self.seats) > 1
    )

  def _recap_marks(self, place: Seat | None) -> list[list[Any]]:
    """A player's moves on the recap's line: [tick it acted, move, way].

    The way is 'warm' or 'cool' by the move's measured profile on this
    world (a turn after it acts), else 'other'. Holding course is no mark.
    """
    if place is None:
      return []
    delay = self._acts_in(place.entity)
    found = []
    for turn in place.played:
      if turn['move'] in NEUTRAL_OPTIONS or turn['how'] == 'held':
        continue
      way = self._way(turn['move'])
      found.append([int(turn['tick']) + delay, short_move(turn['move']), way])
    return found

  def _way(self, move: str) -> str:
    """'warm', 'cool' or 'other': a move's push by its measured profile on

    this world, a turn after it acts.
    """
    planet = self.context.planet
    spec = planet.intervention_for_option(move)
    profile = planet.move_profiles.get(spec.key, []) if spec else []
    push = float(profile[0]) if profile else 0.0
    return 'warm' if push >= 0.5 else 'cool' if push <= -0.5 else 'other'

  def _turning(self, place: Seat | None) -> list[dict[str, Any]]:
    """The game's turning points for a Steward's recap.

    Only a played game's: they tell a person's own moves and readings (a
    watched game's Steward is the mock, whose moves are not kept).
    """
    if (
        self.mode != 'play'
        or place is None
        or not isinstance(
            self.context.actors.get(place.entity), actors_lib.StewardActor
        )
    ):
      return []
    planet = self.context.planet
    low, high = planet.band()
    played = list(place.played)
    return forecast_lib.turning_points(
        planet.history,
        played,
        (low, high) if high - low <= 60 else None,
        self._acts_in(place.entity),
        {str(t['move']): self._way(str(t['move'])) for t in played},
    )

  def _recap(self, place: Seat | None) -> dict[str, Any]:
    """A decided game: the temperature all through, and a player's moves."""
    planet = self.context.planet
    low, high = planet.band()
    moves: dict[str, int] = {}
    held = timed_out = 0
    for choice in place.choices if place is not None else []:
      if choice.startswith('(holding course) '):
        held += 1
      elif choice.startswith('(out of time) '):
        timed_out += 1
      else:
        moves[choice] = moves.get(choice, 0) + 1
    return {
        'outcome': planet.outcome,
        'reason': planet.outcome_reason,
        'ticks': planet.tick,
        'temps': [round(float(r['temp_c']), 1) for r in planet.history],
        'band': [low, high] if high - low <= 60 else None,
        # The player's moves on that line, where each acted, and which way
        # it pushed; and what their thermometer read.
        'marks': self._recap_marks(place),
        # Where the world swung hardest, and the move that came before it
        # with what the player read and what was true then.
        'turning': self._turning(place),
        'seen': (
            [
                [int(r['tick']), round(float(r['believed_temp_c']), 1)]
                for r in planet.history
                if r.get('believed_temp_c') is not None
            ]
            if place is not None
            and isinstance(
                self.context.actors.get(place.entity), actors_lib.StewardActor
            )
            else []
        ),
        'moves': (
            [[m, k] for m, k in sorted(moves.items(), key=lambda kv: -kv[1])][
                :4
            ]
            if place is not None
            else None
        ),
        'held': held,
        'timed_out': timed_out,
        'planned': sum(
            1
            for t in (place.played if place is not None else [])
            if t['how'] == 'planned'
        ),
        # How the player played: turns acted on (a move other
        # than the neutral one), of all turns, and the median seconds a
        # chosen move took.
        'play': self._how_played(place),
        'role': place.role if place is not None else None,
        'rewinds': self.rewinds,
        'minds': self.spend(),
        # The stars earned, and each star's rule.
        'stars': planet.stars(),
        # The side objectives and their verdicts on the true world.
        'side': planet.side_payload(),
        # Whether it can be watched again exactly, and as which game.
        'replayable': bool(
            self.mode == 'play' and self.minds == 'mock' and planet.outcome
        ),
        'game_id': self.game_id,
        # Its chapter of the campaign, best stars and the next.
        'chapter': self.chapter if self.mode == 'play' else None,
        # How far off the Steward's forecasts were, against its raw
        # readings; None where no one read the temperature.
        'forecast': forecast_lib.track_record(
            planet.history, planet.made, planet.move_profiles
        ),
    }

  def _how_played(self, place: 'Seat | None') -> dict[str, Any] | None:
    """Turns, turns acted on, and the median time to choose."""
    if place is None or not place.played:
      return None
    turns = list(place.played)
    seconds = sorted(t['seconds'] for t in turns if t['seconds'] is not None)
    return {
        'turns': len(turns),
        'acted': sum(1 for t in turns if t['move'] not in NEUTRAL_OPTIONS),
        'seconds': seconds[len(seconds) // 2] if seconds else None,
    }

  def _reads_news(self, seat: str | None, reveal: bool, kind: str) -> bool:
    """Whether this seat may read a ripple's news of this kind.

    Watchers and finished games read all; the Steward hears world news and
    the peoples' (and so reads every ripple); a people hears the peoples'
    own news at once (its chronicle), the rest only as its couriers bring it.
    """
    if reveal:
      return True
    if self._spectating(seat):
      return False
    entity = seat if seat in self.seats else self.entity
    actor = self.context.actors.get(entity)
    if isinstance(actor, actors_lib.StewardActor):
      return True
    return isinstance(actor, actors_lib.PolityActor) and kind == 'civ'

  def _trace(self, reveal: bool, ticks: int = 16) -> dict[str, Any] | None:
    """The last ticks' temperatures: as the Steward read them, and the truth.

    The readings are what the Steward believed each tick (late and noisy);
    the truth is shown only once revealed, so a watcher sees the gap.
    """
    planet = self.context.planet
    steward = next(
        (
            name
            for name, actor in self.context.actors.items()
            if isinstance(actor, actors_lib.StewardActor)
        ),
        None,
    )
    if steward is None:  # No Steward reads this planet's temperature.
      return None
    rows = [dict(r) for r in planet.history[-ticks:]]
    low, high = planet.band()
    goal = planet.goal()
    return {
        'reader': steward,
        # A band too wide ever to bind (a thaw's) says nothing: drawn instead
        # is the temperature the goal needs.
        'band': [low, high] if high - low <= 60 else None,
        'target': (
            float(goal['temp_c'])
            if goal.get('kind') == 'thaw' and 'temp_c' in goal
            else None
        ),
        'ticks': [int(r['tick']) for r in rows],
        'seen': [r.get('believed_temp_c') for r in rows],
        'true': (
            [round(float(r['temp_c']), 2) for r in rows] if reveal else None
        ),
        # Its dead reckoning from those readings and its moves.
        'forecast': (
            found.to_dict()
            if (
                found := views.forecast(
                    views.latest(self.context, steward, views.LOGBOOK, None),
                    self._planned_moves(steward),
                )
            )
            else None
        ),
    }

  def state(self, seat: str | None = None) -> dict[str, Any]:
    """Everything one seat's page shows.

    A player sees only their own role's context until the game ends; then
    (and always for a watcher) the true state of the world is revealed too.

    Args:
      seat: The seat (entity name) viewing. Alone, a player may omit it; in a
        game for several, an unknown or missing seat gets a spectator's view (no
        question, prompt, goal or briefing), so a stale link never plays someone
        else's role.

    Returns:
      The view, as JSON-ready data.
    """
    spectator = self._spectating(seat)
    entity = seat if seat in self.seats else self.entity
    place = self.seats[entity]
    planet = self.context.planet
    bus = self.context.bus
    scenario = data.scenario(self.scenario)
    reveal = self.mode == 'watch' or self.finished
    counselled = not spectator and (
        reveal
        or isinstance(self.context.actors.get(entity), actors_lib.StewardActor)
    )

    def rendered(topic: str) -> str:
      message = bus.latest(topic)
      if not reveal or message is None:
        return ''
      return publishing.renderer_for(topic)(message)

    news = []
    if reveal:
      for topic in NEWS_TOPICS:
        for message in bus.history(topic)[-6:]:
          news.append((message.seq, topic.split('.')[0], message.payload))
      news.sort()
    request = place.pending() if self.mode == 'play' and not spectator else None
    question = None
    if request is not None:
      spec = request.action_spec
      actor = self.context.actors.get(entity)
      # A free question's moves (the Steward may also speak an omen).
      options = list(place.offered(request))
      menu = views.menu(self.context, entity)
      saving = list(menu.get('later', ()))
      question = {
          'request_id': request.request_id,
          'call_to_action': spec.call_to_action,
          'options': options,
          # Options that may land where the player taps the map.
          'targets': [o for o in options if self._targetable(entity, o)],
          # Which way each move pushes the world, and about how far.
          'pushes': {
              o: p for o, p in menu.get('pushes', {}).items() if o in options
          },
          # Moves not yet affordable: shown, and plannable.
          'saving': saving,
          'saving_pushes': {
              o: p for o, p in menu.get('pushes', {}).items() if o in saving
          },
          # What the Steward's forecast says each climate move would bring:
          # its own late readings, reckoned with the move.
          'what_if': (
              self._what_if(entity, [*options, *saving])
              if isinstance(actor, actors_lib.StewardActor)
              else {}
          ),
          # The Advisor's pick among these moves, and how old its counsel
          # is: a slow mind, reasoning from older readings.
          'advice': (
              self._advice(entity, [*options, *saving])
              if isinstance(actor, actors_lib.StewardActor)
              else None
          ),
          # What matters most for a people: the move answering its
          # most urgent crisis, and the one strengthening its weakest link.
          'pressing': list(menu.get('pressing', ())),
          'error': request.error or '',
          'context': request.context,
          # The omen the Steward may speak in its own words, if any.
          'omen': self._omen(actor, options),
      }
    briefing = bus.latest('gaia.briefings')
    chronicle = bus.latest('chronicle')
    return {
        'scenario': self.scenario,
        'title': scenario.get('title', self.scenario),
        'mode': self.mode,
        # Who plays the other roles, and what they cost so far.
        'minds': self.spend(),
        'seat': None if spectator else entity,
        'spectator': spectator,
        'seats': (
            [{'name': n, 'role': p.role} for n, p in self.seats.items()]
            if self.mode == 'play'
            else []
        ),
        'role': '' if spectator else place.role,
        # The other players the world is waiting on this turn.
        'waiting_for': [
            p.role
            for n, p in self.seats.items()
            if n != entity and self.mode == 'play' and p.pending() is not None
        ],
        'goal': '' if spectator else self.goal(entity),
        # The scenario's stars, known from the start; met once decided.
        'stars': planet.stars(),
        # The Steward's side objectives: their rules, known from
        # the start; how they stand, it reads on its omega meter.
        'side': (
            [
                {k: o[k] for k in ('key', 'text', 'from', 'until', 'reward')}
                for o in planet.side_payload()
            ]
            if not spectator
            and isinstance(
                self.context.actors.get(entity), actors_lib.StewardActor
            )
            else []
        ),
        'tick': planet.tick,
        'max_ticks': planet.max_ticks,
        'outcome': planet.outcome or '',
        'outcome_reason': planet.outcome_reason or '',
        'finished': self.finished,
        'error': self.error,
        'revealed': reveal,
        # A people's player sees its own people as its rulers do: exactly.
        'people': None if spectator else self._own_people(entity),
        'stats': (
            self._stats(reveal, entity)
            if not spectator
            else self._stats(reveal, entity)[:1]
        ),
        'readings': self.readings(entity) if self.mode == 'watch' else [],
        'climate': rendered('climate.global'),
        'indices': rendered('gaia.indices'),
        'civilization': rendered('civ.census'),
        'gazette': rendered('media.gazette'),
        # What each people made of the last omen it read, in its words.
        'takes': self._takes(),
        'colony': rendered('colony.status'),
        # The Advisor counsels the Steward alone (and every watcher).
        'briefing': briefing.payload if briefing and counselled else '',
        # The temperatures the Steward read (and, once revealed, the truth).
        'trace': self._trace(reveal) if counselled else None,
        # How old each thing this role knows is (none for a spectator).
        'timing': None if spectator else self.timing(entity),
        # Each reading as data, as it reached the role.
        'instruments': None if spectator else self.instruments(entity),
        # Trouble the Steward can see coming, with countdowns.
        'crises': self._crises(
            None if spectator and self.mode == 'play' else entity
        ),
        # The player's last moves and what each did.
        'moves': (
            None
            if spectator or self.mode != 'play'
            else self._moves(place, entity)
        ),
        # What is on its way to this role, and when it lands; and how many
        # turns a move takes to act (a transmission crosses the light delay).
        'in_flight': None if spectator else self.in_flight(entity),
        'acts_in': self._acts_in(entity),
        # The game told once it is decided: its course and your moves.
        'recap': (
            self._recap(None if spectator else place)
            if planet.outcome
            else None
        ),
        'chronicle': chronicle.payload if chronicle and reveal else '',
        # The story so far, as this role may know it.
        'story': self._story(None if spectator else entity, reveal, counselled),
        'news': [{'kind': k, 'text': t} for _, k, t in news[-12:]],
        'question': question,
        # How far the goal has come: the truth, so only once revealed.
        'progress': planet.progress() if reveal else None,
        'speed': self.speed() if self.mode == 'watch' else None,
        'seconds_left': None if spectator else place.seconds_left(),
        'choices': [] if spectator else list(place.choices[-8:]),
        # Holding course: turns left, and why it last stopped by itself.
        'hold': (
            None
            if spectator
            or self.mode != 'play'
            or self.finished
            or planet.outcome
            else {'turns': place.hold, 'stopped': place.stopped}
        ),
        # What came of the course just held, in the player's own readings:
        # shown until they move again.
        'held_run': (
            None if spectator or question is None else self._held_run(entity)
        ),
        # The game this one replays, and whether this one may be.
        'replay_of': self.replay_of,
        'replayable': bool(
            self.mode == 'play' and self.minds == 'mock' and planet.outcome
        ),
        'game_id': self.game_id,
        # Turns that may still be taken back, and whether one can now.
        'rewind': (
            None
            if spectator or self.mode != 'play' or self.finished
            else {
                'left': REWINDS - self.rewinds,
                'can': (
                    self.rewinds < REWINDS
                    and any(t == planet.tick - 2 for t, _ in self._kept)
                ),
                'to': planet.tick - 1,
            }
        ),
        # Moves planned for later turns, soonest first.
        'plans': (
            []
            if spectator or self.mode != 'play'
            else [
                {'tick': t, 'move': m, 'in': t - planet.tick}
                for t, m in sorted(place.plans.items())
            ]
        ),
    }


_BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'
# Daughter species take their parent's colour turned round the colour wheel
# (by these fractions, in order of their number), so a family stays as light
# or dark as its parent while its members stay apart on the map.
_HUE_TURNS = (0.12, 0.88, 0.24, 0.76, 0.36, 0.64)


def _species_colours(
    key: str,
    albedo: float,
    colours: Mapping[str, Any],
    taken: Sequence[Sequence[int]] = (),
) -> dict[str, list[int]]:
  """A species' map colours (light and dark themes).

  Args:
    key: The species key; a daughter's is its parent's plus a number.
    albedo: Its flowers' albedo, for species of unknown lineage.
    colours: Colours set in the data, by key.
    taken: Light-theme colours already on the map; a daughter takes the turn of
      the colour wheel that keeps farthest from them.
  """
  if key in colours:
    return {t: list(colours[key][t]) for t in ('light', 'dark')}
  parent = key.rstrip('0123456789')
  if parent in colours:
    base = colours[parent]['light']
  else:  # Unknown lineage: a green as light as the species' albedo.
    grey = int(40 + 200 * min(1.0, max(0.0, albedo)))
    base = [grey, min(255, grey + 30), grey]
  number = int(key[len(parent) :] or 0)
  hue, lightness, saturation = colorsys.rgb_to_hls(*(x / 255 for x in base))
  if saturation < 0.15:  # White or black: tint it, keeping it pale or dark.
    hue, saturation = 0.1, 0.45
    lightness = min(max(lightness, 0.28), 0.8)
  turns = _HUE_TURNS[number % len(_HUE_TURNS) :] + _HUE_TURNS
  options = [
      [
          int(round(255 * x))
          for x in colorsys.hls_to_rgb((hue + t) % 1.0, lightness, saturation)
      ]
      for t in turns
  ]
  light = max(
      options,
      key=lambda o: min(
          (sum(abs(x - y) for x, y in zip(o, c)) for c in taken),
          default=0,
      ),
  )
  return {'light': light, 'dark': [int(x * 0.78) for x in light]}


def _peoples_now(planet: Any) -> dict[str, Any] | None:
  """Each people's food and busiest trades now (truth, for telemetry)."""
  civ = planet.civ
  if civ is None:
    return None
  return {
      p.name: {
          'fed': round(float(p.food_ratio), 2),
          'hands': {
              k: round(float(v), 2)
              for k, v in sorted(p.allocation.items(), key=lambda kv: -kv[1])[
                  :3
              ]
          },
      }
      for p in civ.polities.values()
  }


def _clouds(planet: Any) -> list[list[float]] | None:
  """Each region's cloud albedo (+ bright cloud or ash, − dark haze)."""
  field = planet.daisies.cloud_field()
  if float(np.abs(field).max()) < 0.005:
    return None
  grid = planet.grid
  rows = field.reshape(grid.bands, grid.sectors)[::-1]  # North first.
  return [[round(float(v), 3) for v in row] for row in rows]


# A news digest's lines: "- tick 6: The desert spreads ...".
_DIGEST_RE = re.compile(r'^- (?:tick (\d+): )?(.+)$', re.MULTILINE)


# The topic in a subscriber's label: "Thermometer (climate.global)".
_TOPIC_RE = re.compile(r'\(([a-z0-9_.,\s]+)\)\s*$')


# Subscriptions of the contrib async-deliberation prefab (peoples, the
# colony) are labelled without their topic: these are their topics.
_PLAIN_TOPICS = {
    'Census of the peoples': 'civ.census',
    'Weather': 'climate.global',
    'Treaties': 'civ.treaties',
    'Gazette': 'media.gazette',
    'Colony status': 'colony.status',
}


# The age a reading ends with: "[tick 3, 2 ticks old]", "[1 tick old]" or
# "[fresh]".
_AGE_RE = re.compile(r'\[(?:tick (\d+), )?(fresh|(\d+) ticks? old)\]\s*$')


def _seasonality(biomes: Any) -> list[list[int]]:
  """Each tile's annual temperature range (°C), north first: geography, so

  static, sent with the terrain.
  """
  return [[int(round(v)) for v in row] for row in biomes.seasonality[::-1]]


def _with_seen_life(
    layers: list[str], seen: bool, first: bool, burns: bool = False
) -> list[str]:
  """The layers, with what a player's census saw.

  Its life (first in worlds whose story is life), the temperature of its
  latitude bands, and, where something can burn, the fire danger.
  """
  if not seen or 'life' in layers:
    return layers
  found = ['life', *layers] if first else [layers[0], 'life', *layers[1:]]
  return found + ['temperature'] + (['fire'] if burns else [])


def _map_layers(reveal: bool) -> list[str]:
  # Terrain and biomes are the planet's visible face; numbers are the truth.
  public = ['biomes', 'terrain']
  return (
      public + ['temperature', 'rainfall', 'seasons', 'life', 'fire']
      if reveal
      else public
  )


def scenario_list() -> list[dict[str, Any]]:
  """The scenarios a player can start, with their titles and premises."""
  found = []
  for key in sorted(scenarios.SCENARIOS):
    spec = data.scenario(key)
    found.append({
        'key': key,
        'title': str(spec.get('title', key)),
        'premise': ' '.join(str(spec.get('premise', '')).split()),
        'roles': [{'name': n, 'label': l} for n, l in roles(key)],
        # Two seats that play this world together, on two phones.
        'coop': spec.get('coop'),
    })
  return found
