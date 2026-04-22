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

"""A prefab for the PersonaInitializer game master.

Wires up the PersonaInitializer component (which runs any PersonaGenerator)
with standard GM infrastructure (memory, observation, termination, etc.).

See: Persona Generators: Generating Diverse Synthetic Personas at Scale
     https://arxiv.org/abs/2602.03545
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import persona_generation
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import measurements as measurements_lib


def build_components(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    generator: persona_generation.PersonaGenerator,
    player_names: Sequence[str],
    next_game_master_name: str,
    initial_context: str,
    diversity_axes: Sequence[str],
    shared_memories: Sequence[str] = (),
) -> dict[str, entity_component.ContextComponent]:
  """Build the components for a PersonaInitializer game master.

  This function is separated from the prefab to allow reuse by extending
  prefabs that need to add additional components.

  Args:
    model: The language model to use.
    memory_bank: The memory bank to use.
    generator: A PersonaGenerator instance (any object satisfying the
      protocol).
    player_names: Names of all player entities.
    next_game_master_name: Name of the game master to transition to.
    initial_context: Shared context passed to Stage 1 generation.
    diversity_axes: Axes along which to encourage diversity.
    shared_memories: Memories shared by all players.

  Returns:
    A dictionary mapping component keys to components.
  """
  memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
  memory_comp = actor_components.memory.AssociativeMemory(
      memory_bank=memory_bank
  )

  instructions_key = 'instructions'
  instructions = gm_components.instructions.Instructions()

  examples_synchronous_key = 'examples'
  examples_synchronous = gm_components.instructions.ExamplesSynchronous()

  player_characters_key = 'player_characters'
  player_characters = gm_components.instructions.PlayerCharacters(
      player_characters=player_names,
  )

  observation_to_memory_key = 'observation_to_memory'
  observation_to_memory = actor_components.observation.ObservationToMemory()

  observation_component_key = (
      actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
  )
  observation = actor_components.observation.LastNObservations(
      history_length=1000,
  )

  next_game_master_key = (
      gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
  )
  next_game_master = persona_generation.PersonaInitializer(
      model=model,
      generator=generator,
      next_game_master_name=next_game_master_name,
      player_names=player_names,
      initial_context=initial_context,
      diversity_axes=diversity_axes,
      shared_memories=shared_memories,
  )

  make_observation_key = (
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  )
  make_observation = gm_components.make_observation.MakeObservation(
      model=model,
      player_names=player_names,
  )

  skip_next_action_spec_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
  )
  skip_next_action_spec = gm_components.next_acting.FixedActionSpec(
      action_spec=entity_lib.skip_this_step_action_spec(),
  )

  terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
  terminate = gm_components.terminate.NeverTerminate()

  return {
      instructions_key: instructions,
      examples_synchronous_key: examples_synchronous,
      player_characters_key: player_characters,
      observation_component_key: observation,
      observation_to_memory_key: observation_to_memory,
      memory_component_key: memory_comp,
      next_game_master_key: next_game_master,
      make_observation_key: make_observation,
      skip_next_action_spec_key: skip_next_action_spec,
      terminate_key: terminate,
  }


def build_game_master(
    model: language_model.LanguageModel,
    name: str,
    player_names: Sequence[str],
    components: dict[str, entity_component.ContextComponent],
    measurements: measurements_lib.Measurements | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build the game master entity from components.

  Args:
    model: The language model to use.
    name: Name of the game master.
    player_names: Names of all player entities.
    components: Dictionary of components to use.
    measurements: Optional measurements instance.

  Returns:
    The constructed game master entity.
  """
  component_order = list(components.keys())

  act_component = gm_components.switch_act.SwitchAct(
      model=model,
      entity_names=player_names,
      component_order=component_order,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=name,
      act_component=act_component,
      context_components=components,
      measurements=measurements,
  )

  return game_master


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """Prefab for a PersonaInitializer game master.

  Generates diverse personas at simulation startup using any generator
  that satisfies the PersonaGenerator protocol. This prefab wires together
  the PersonaInitializer component with standard Concordia GM infrastructure.

  See: https://arxiv.org/abs/2602.03545
  """

  description: str = (
      'An initializer that generates diverse personas from a pluggable '
      'PersonaGenerator and injects their memories into agent memory banks.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'persona initialization',
          'next_game_master_name': 'default rules',
          'initial_context': '',
          'diversity_axes': [],
          'shared_memories': [],
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  # The generator must be set before calling build().
  generator: persona_generation.PersonaGenerator | None = None

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build the PersonaInitializer game master entity.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      A game master entity.

    Raises:
      ValueError: If no generator has been set.
    """
    if self.generator is None:
      raise ValueError(
          'PersonaInitializer prefab requires a generator. '
          'Set prefab.generator = my_generator before calling build().'
      )

    name = self.params.get('name', 'persona initialization')
    next_game_master_name = self.params.get(
        'next_game_master_name', 'default rules'
    )
    player_names = [entity.name for entity in self.entities]

    components = build_components(
        model=model,
        memory_bank=memory_bank,
        generator=self.generator,
        player_names=player_names,
        next_game_master_name=next_game_master_name,
        initial_context=self.params.get('initial_context', ''),
        diversity_axes=self.params.get('diversity_axes', []),
        shared_memories=self.params.get('shared_memories', []),
    )

    return build_game_master(
        model=model,
        name=name,
        player_names=player_names,
        components=components,
        measurements=self.params.get('measurements'),
    )
