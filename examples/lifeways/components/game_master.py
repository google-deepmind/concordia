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
from typing import Any, override

from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import terminate
from concordia.environment import engine as engine_lib
from examples.lifeways.world import world as world_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_WORLD_KEY = 'lifeways_world'

PASS_ON_QUESTION = (
    "{name}'s child is about to be born. Which of {name}'s customs does"
    ' {name} pass on to the child? Answer "pass on" and the ids of the'
    ' customs, e.g. "pass on c3 c7".'
)
MOVE_QUESTION = (
    'It is year {year}. What does {name} do this year? Choose one move:\n'
    '{moves}\n'
    'Answer with the move, then the ids of the customs {name} follows in'
    ' doing it (or, to teach, the customs taught), if any, e.g.'
    ' "{example} c3".'
)


@dataclasses.dataclass
class Lifecycle:
  """How the world's births and deaths reach the simulation.

  The harness that builds the simulation fills these in: `spawn` adds an
  entity (`Simulation.add_entity` of an `InstanceConfig`), `retire` removes
  one (keeping a record of it first), `forgotten` receives the variants of
  customs nobody living holds any more (the long record of culture) and
  `year_closed` is told each new year.
  """

  spawn: Callable[[Mapping[str, Any]], None] = lambda params: None
  retire: Callable[[str], None] = lambda name: None
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
  delivers next year's observations, adds the born and removes the dead
  through the `Lifecycle`, and ends the game at `max_years`.
  """

  def __init__(
      self,
      lifecycle: Lifecycle,
      observation_queue: make_observation.ObservationQueue,
      max_years: int | None = None,
      world_key: str = DEFAULT_WORLD_KEY,
      terminate_key: str = terminate.DEFAULT_TERMINATE_COMPONENT_KEY,
  ):
    """Initializes the gate.

    Args:
      lifecycle: How births, deaths and forgotten customs reach the
        simulation.
      observation_queue: The queue the GM's `MakeObservation` delivers from.
      max_years: The year the game ends at; None for never.
      world_key: The key of the GM's `WorldComponent`.
      terminate_key: The key of the GM's stock `Terminate`.
    """
    super().__init__()
    self._world_key = world_key
    self._terminate_key = terminate_key
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
      if world.year_complete() and not self._ended:
        self._close_year(world)
      if self._ended:
        return ''
      return ','.join(n for n in action_spec.options if world.may_act(n))

  def _close_year(self, world: world_lib.World) -> None:
    outcome = world.close_year()
    self.deliver(world)
    for params in outcome.births:
      self._lifecycle.spawn(params)
    if outcome.deaths:  # The dead hear nothing more.
      queued = self._queue.get_all()
      for name in outcome.deaths:
        queued.pop(name, None)
      self._queue.set_all(queued)
    for name in outcome.deaths:
      self._lifecycle.retire(name)
    if outcome.extinct:
      self._lifecycle.forgotten(outcome.extinct)
    self._logging_channel({
        'Key': 'Year',
        'Summary': f'Year {world.year - 1} closed',
        'Value': world.history[-1] if world.history else {},
    })
    if self._max_years is not None and world.year >= self._max_years:
      self._ended = True
      self.get_entity().get_component(
          self._terminate_key, type_=terminate.Terminate
      ).terminate()
    self._lifecycle.year_closed(world.year)

  def deliver(self, world: world_lib.World) -> None:
    """Queues the world's pending observations for their people."""
    for name, texts in world.take_observations().items():
      self._queue.add(name, '\n'.join(texts), ())

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'ended': self._ended}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._ended = bool(state.get('ended', False))


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
      if world.asked_to_pass_on(name):
        question = PASS_ON_QUESTION.format(name=name)
        tag = 'pass_on'
      else:
        moves = world.moves(name)
        question = MOVE_QUESTION.format(
            year=world.year,
            name=name,
            moves='\n'.join(f'- {m}' for m in moves),
            example=moves[0],
        )
        tag = 'move'
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
