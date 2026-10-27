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

"""The World GM's components: the world, the year's gate, questions, answers.

The World GM runs on the stock Asynchronous engine: every person's thread asks
it, on its own, whether the game is over, whether this person may act now,
what the person is asked, what the person observes, and to resolve the
person's answer. A year is a barrier. Each participant answers once a year;
answers are recorded as they come and resolved together, in a fixed order,
when the last is in. So a year's outcome never depends on thread timing.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import threading
import time
from typing import Any, override

from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import player_roster
from concordia.components.game_master import terminate
from concordia.environment import engine as engine_lib
from examples.lifeways.world import world as world_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_WORLD_KEY = 'lifeways_world'
# How many of a player's lives the World GM remembers.
LIVES_KEPT = 100

PASS_ON_QUESTION = (
    "{name}'s child is about to be born. Which of {name}'s customs does"
    ' {name} pass on to the child? Answer "pass on" and the ids of the'
    ' customs, e.g. "pass on c3 c7".'
)
INTERVIEW_QUESTION = (
    'The Experimenter asks {name}: "{question}" What does {name} answer?'
)
# No example answer (small models copy it), and the move asked for exactly
# as written (small models paraphrase it, and a paraphrase names no move).
MOVE_QUESTION = (
    'It is year {year}. What does {name} do this year? Choose one move:\n'
    '{moves}\n'
    'Answer with one of these moves, exactly as written, then the ids of the'
    ' customs {name} follows in doing it (or, to teach, the customs taught),'
    ' if any.'
)


@dataclasses.dataclass
class Lifecycle:
  """How the world's births and deaths reach the simulation.

  The harness that runs the simulation fills these in to read it as it plays:
  `departed` is given each person who died once they have left the game (to
  keep a record of them), `forgotten` receives the variants of customs nobody
  living holds any more (the long record of culture) and `year_closed` is
  told each new year. The born and the dead join and leave the game through
  the GM's `PlayerRoster`, not through these.
  """

  departed: Callable[[entity_lib.Entity], None] = lambda person: None
  forgotten: Callable[[Sequence[Mapping[str, Any]]], None] = (
      lambda records: None
  )
  year_closed: Callable[[int], None] = lambda year: None


class WorldComponent(entity_component.ContextComponent):
  """Holds the world; its state is the world's."""

  def __init__(self, world: world_lib.World):
    super().__init__()
    self.world = world
    self.lock = threading.RLock()

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self.lock:
      return self.world.get_state()

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self.lock:
      self.world.set_state(state)


def _world_component(
    component: entity_component.BaseComponent, key: str
) -> WorldComponent:
  return component.get_entity().get_component(key, type_=WorldComponent)


class TurnGate(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Who may act now, and the close of each year (`__next_acting__`).

  Asked for one person (the asynchronous engine asks per thread), it names
  that person if the person has not yet answered this year. Asked once the
  last answer is in, it first closes the year: it resolves the answers,
  delivers next year's observations, has the born join the game and the dead
  leave it through the GM's `PlayerRoster`, and ends the game at `max_years`
  or when the Experimenter stops the world.
  """

  def __init__(
      self,
      lifecycle: Lifecycle,
      observation_queue: make_observation.ObservationQueue,
      person_prefab: str,
      max_years: int | None = None,
      played: str | None = None,
      played_prefab: str | None = None,
      world_key: str = DEFAULT_WORLD_KEY,
      terminate_key: str = terminate.DEFAULT_TERMINATE_COMPONENT_KEY,
      roster_key: str = player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY,
      free: float | None = None,
  ):
    """Initializes the gate.

    Args:
      lifecycle: What the harness is told as the world plays.
      observation_queue: The queue the GM's `MakeObservation` delivers from.
      person_prefab: The roster's prefab the born join the game as.
      max_years: The year the game ends at; None for never.
      played: The founder someone plays, if any. Once the played person
        dies, someone born of its line lives on as it (`World.heir`), and
        joins the game as `played_prefab`.
      played_prefab: The roster's prefab a played person's heir joins as.
      world_key: The key of the GM's `WorldComponent`.
      terminate_key: The key of the GM's stock `Terminate`.
      roster_key: The key of the GM's stock `PlayerRoster`.
      free: Free mode, for live play: a year also closes after this many
        seconds, whoever has not answered yet doing nothing that year. Runs
        then depend on how fast minds answer. None: tick mode (a year closes
        when every participant has answered; runs are reproducible).
    """
    super().__init__()
    self._world_key = world_key
    self._terminate_key = terminate_key
    self._roster_key = roster_key
    self._person_prefab = person_prefab
    self._played_prefab = played_prefab or person_prefab
    # The lives the player has lived, the latest last (bounded).
    self._lives: list[str] = [played] if played else []
    self._free = free
    self._began = time.monotonic()  # When this year began (free mode).
    self._lifecycle = lifecycle
    self._queue = observation_queue
    self._max_years = max_years
    # Once the game ends nobody acts again: a thread that checked for the end
    # just before it came must not start another year.
    self._ended = False

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    holder = _world_component(self, self._world_key)
    with holder.lock:
      world = holder.world
      late = (
          self._free is not None
          and time.monotonic() - self._began >= self._free
      )
      if (world.year_complete() or late) and not self._ended:
        self._close_year(world)
        self._began = time.monotonic()
      if self._ended:
        return ''
      return ','.join(n for n in action_spec.options if world.may_act(n))

  def _close_year(self, world: world_lib.World) -> None:
    outcome = world.close_year()
    self.deliver(world)
    roster = self.get_entity().get_component(
        self._roster_key, type_=player_roster.PlayerRoster
    )
    heir = ''
    if self._lives and self._lives[-1] not in world.people:
      heir = world.heir(
          self._lives[-1], [params['name'] for params in outcome.births]
      )
    for params in outcome.births:
      roster.join(
          self._played_prefab
          if params['name'] == heir
          else self._person_prefab,
          params,
      )
    if heir:
      self._lives = [*self._lives, heir][-LIVES_KEPT:]
    for name in outcome.deaths:  # The dead hear nothing more.
      self._queue.get_and_clear(name)
    if outcome.deaths and self._free is None:
      # In tick mode no thread takes observations while a year closes, so
      # the dead can leave the queue altogether (bounded over endless runs).
      queued = self._queue.get_all()
      for name in outcome.deaths:
        queued.pop(name, None)
      self._queue.set_all(queued)
    for name in outcome.deaths:
      self._lifecycle.departed(roster.leave(name))
    if outcome.extinct:
      self._lifecycle.forgotten(outcome.extinct)
    self._logging_channel({
        'Key': 'Year',
        'Summary': f'Year {world.year - 1} closed',
        'Value': world.history[-1] if world.history else {},
    })
    if outcome.stopped or (
        self._max_years is not None and world.year >= self._max_years
    ):
      self._ended = True
      self.get_entity().get_component(
          self._terminate_key, type_=terminate.Terminate
      ).terminate()
    self._lifecycle.year_closed(world.year)

  def deliver(self, world: world_lib.World) -> None:
    """Queues the world's pending observations for their people.

    Args:
      world: The world (in its lock).
    """
    for name, texts in world.take_observations().items():
      self._queue.add(name, '\n'.join(texts), ())

  def lives(self) -> list[str]:
    """The lives the player has lived, the latest last ([] if none).

    Read it in the World GM's lock (`Run.lives`).
    """
    return list(self._lives)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'ended': self._ended, 'lives': list(self._lives)}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._ended = bool(state.get('ended', False))
    lives = state.get('lives')  # Older saves keep none.
    if isinstance(lives, Sequence) and not isinstance(lives, str):
      self._lives = [str(name) for name in lives]


class Questions(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """What each person is asked (`__next_action_spec__`)."""

  def __init__(
      self,
      call_to_next_action_spec: str = (
          next_acting.DEFAULT_CALL_TO_NEXT_ACTION_SPEC
      ),
      world_key: str = DEFAULT_WORLD_KEY,
  ):
    """Initializes the questions.

    Args:
      call_to_next_action_spec: The engine's question, with `{name}`.
      world_key: The key of the GM's `WorldComponent`.
    """
    super().__init__()
    self._prefix, self._suffix = call_to_next_action_spec.split('{name}')
    self._world_key = world_key

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    name = action_spec.call_to_action.removeprefix(self._prefix).removesuffix(
        self._suffix
    )
    holder = _world_component(self, self._world_key)
    with holder.lock:
      world = holder.world
      if not world.may_act(name):
        # In free mode the year may have closed (and the person died) since
        # the person was named: it waits for the next year.
        return engine_lib.action_spec_to_string(
            entity_lib.skip_this_step_action_spec()
        )
      if world.interview(name):
        question = INTERVIEW_QUESTION.format(
            name=name, question=world.interview(name)
        )
        tag = world_lib.INTERVIEW
      elif world.asked_to_pass_on(name):
        question = PASS_ON_QUESTION.format(name=name)
        tag = world_lib.PASSING_ON
      else:
        moves = world.moves(name)
        question = MOVE_QUESTION.format(
            year=world.year,
            name=name,
            moves='\n'.join(f'- {m}' for m in moves),
        )
        tag = world_lib.MOVES
      world.mark_asked(name, tag)
    spec = entity_lib.free_action_spec(call_to_action=question, tag=tag)
    self._logging_channel({'Key': 'Question', 'Summary': f'{name}: {tag}'})
    return engine_lib.action_spec_to_string(spec)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class Answers(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Records each answer for the year's close (`__resolution__`).

  The engine hands the GM each answer as a putative event ("Name: answer");
  this component records it as that person's answer for the year. The answer
  takes effect when the year closes (`TurnGate`), with everyone else's.
  """

  def __init__(self, world_key: str = DEFAULT_WORLD_KEY):
    """Initializes the answers.

    Args:
      world_key: The key of the GM's `WorldComponent`.
    """
    super().__init__()
    self._world_key = world_key

  @override
  def pre_observe(self, observation: str) -> str:
    tag = event_resolution.PUTATIVE_EVENT_TAG
    if not observation.startswith(tag):
      return ''
    name, _, answer = observation[len(tag) :].strip().partition(':')
    holder = _world_component(self, self._world_key)
    with holder.lock:
      holder.world.record_action(name.strip(), answer.strip())
    return ''

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    return 'The answer is noted; the year resolves when everyone has answered.'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state
