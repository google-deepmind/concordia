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

"""A game master that runs any toy, built from standard Concordia pieces.

`switch_act.SwitchAct` sends each engine call to the component under the
matching well-known key:

| Key | Component |
|---|---|
| `__terminate__` | `terminate.Terminate`, told to stop when the toy is done |
| `__make_observation__` | `make_observation.MakeObservation` on a shared queue, no LLM fallback |
| `__next_acting__` | `ToyTurn`: the toy's turn order |
| `__next_action_spec__` | `ToyActionSpec`: the toy's spec for the actor |
| `__resolution__` | `ToyResolution`: the toy resolves the move |

The game master itself never calls a language model; a toy may, to read
what a move's words mean.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any, override

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from examples.crucible.toys import base

GM_PREFAB = 'crucible_toy__GameMaster'


class ToyTurn(entity_component.ContextComponent):
  """`__next_acting__`: whoever the toy says acts next."""

  def __init__(self, toy: base.Toy):
    super().__init__()
    self._toy = toy

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    return self._toy.next_actor()

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class ToyActionSpec(entity_component.ContextComponent):
  """`__next_action_spec__`: the toy's spec for the current actor."""

  def __init__(self, toy: base.Toy):
    super().__init__()
    self._toy = toy

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    actor = self._toy.current_actor or self._toy.actors()[0]
    return engine_lib.action_spec_to_string(self._toy.action_spec(actor))

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class ToyResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """`__resolution__`: the toy resolves each move; holds the toy's state."""

  def __init__(
      self,
      toy: base.Toy,
      queue: make_observation.ObservationQueue,
      terminator: terminate.Terminate,
  ):
    super().__init__()
    self._toy = toy
    self._queue = queue
    self._terminator = terminator
    self._putative_event = ''
    self.receipts: list[base.Receipt] = []

  def deliver(self) -> None:
    """Moves the toy's queued observations onto the shared queue."""
    actors = self._toy.actors()
    for recipient, text in self._toy.drain_outbox():
      self._queue.add(recipient, text, actors)

  @override
  def pre_observe(self, observation: str) -> str:
    if observation.startswith(event_resolution.PUTATIVE_EVENT_TAG):
      self._putative_event = observation[
          len(event_resolution.PUTATIVE_EVENT_TAG) :
      ].strip()
    return ''

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    parsed = base.split_putative_event(self._putative_event, self._toy.actors())
    self._putative_event = ''
    if parsed is None:
      return ''
    receipt = self._toy.resolve(*parsed)
    self.receipts.append(receipt)
    self.deliver()
    if self._toy.done():
      self._terminator.terminate()
    self._logging_channel({
        'Key': 'Resolution',
        'Summary': receipt.summary,
        'Value': receipt.to_dict(),
    })
    return receipt.summary

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'toy': self._toy.get_state()}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._toy.set_state(state['toy'])  # pyrefly: ignore[bad-argument-type]


@dataclasses.dataclass
class ToyGameMaster(prefab_lib.Prefab):
  """A deterministic game master around a toy (params: `name`, `toy`)."""

  description: str = (  # pyrefly: ignore[bad-override]
      'Crucible toy game master: turn order, action specs and resolution all'
      ' come from a toy object; no language model calls of its own.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'Crucible', 'toy': None}
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the game master.

    Args:
      model: Passed to `SwitchAct` and `MakeObservation`, whose language
        model paths are never taken.
      memory_bank: Unused; the game master keeps no associative memory.

    Returns:
      The game master entity.
    """
    del memory_bank
    toy: base.Toy = self.params['toy']
    names = [entity.name for entity in self.entities or ()] or toy.actors()
    queue = make_observation.ObservationQueue()
    terminator = terminate.Terminate()
    resolution = ToyResolution(toy, queue, terminator)
    resolution.deliver()  # The premise and anything else queued at setup.
    components = {
        terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminator,
        make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model=model,
                player_names=names,
                call_to_make_observation=(
                    sequential.DEFAULT_CALL_TO_MAKE_OBSERVATION
                ),
                external_queue=queue,
                allow_llm_fallback=False,
            )
        ),
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: ToyTurn(toy),
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: ToyActionSpec(toy),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: resolution,
    }
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=self.params.get('name', 'Crucible'),
        act_component=switch_act.SwitchAct(model=model, entity_names=names),
        context_components=components,
    )
