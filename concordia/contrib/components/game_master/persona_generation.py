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

r"""Persona generation protocol and initializer GM component.

Defines the PersonaGenerator protocol and PersonaInitializer component for
generating diverse synthetic personas at simulation startup.

See: Persona Generators: Generating Diverse Synthetic Personas at Scale
     https://arxiv.org/abs/2602.03545

All generators follow a two-stage architecture:
  Stage 1: generate_diverse_persona_characteristics() -> List[Dict]
  Stage 2: generate_single_persona_memories(persona) -> List[str]

The PersonaInitializer component runs both stages at initialization time,
injecting the resulting memories into agent memory banks via MakeObservation.
It works like FormativeMemoriesInitializer but accepts a pluggable generator
that satisfies the PersonaGenerator protocol.

Usage:
  from concordia.contrib.components.game_master import persona_generation

  # Any class satisfying the PersonaGenerator protocol
  generator = MyTwoStageGenerator(model)

  initializer = persona_generation.PersonaInitializer(
      model=model,
      generator=generator,
      next_game_master_name='default rules',
      player_names=['Alice', 'Bob'],
      initial_context='A small island community...',
      diversity_axes=['openness', 'extraversion'],
  )
"""

from collections.abc import Sequence
import functools
from typing import Any, Protocol, cast, runtime_checkable

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import concurrency


@runtime_checkable
class PersonaGenerator(Protocol):
  """Protocol for all persona generators.

  Implementations generate diverse synthetic personas in two stages:
    1. Generate a set of diverse persona characteristics (identity, traits).
    2. Generate formative memories or behavioral rules for each persona.

  See: https://arxiv.org/abs/2602.03545
  """

  def __init__(self, model: language_model.LanguageModel) -> None:
    """Initializes the generator with a language model."""
    ...

  def generate_diverse_persona_characteristics(
      self,
      initial_context: str,
      diversity_axes: Sequence[str],
      num_personas: int,
  ) -> list[dict[str, Any]]:
    """Stage 1: Generate diverse persona characteristics.

    Args:
      initial_context: Shared context for all personas (e.g. scenario
        description).
      diversity_axes: Axes along which to encourage diversity (e.g. personality
        traits, demographics).
      num_personas: The number of personas to generate.

    Returns:
      A list of dictionaries, each representing a persona's characteristics.
    """
    ...

  def generate_single_persona_memories(
      self,
      persona_details: dict[str, Any],
  ) -> list[str]:
    """Stage 2: Generate formative memories for one persona.

    Args:
      persona_details: Dictionary containing the persona's characteristics as
        produced by generate_diverse_persona_characteristics().

    Returns:
      A list of strings representing the persona's formative memories,
      backstory episodes, or behavioral rules.
    """
    ...


class PersonaInitializer(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Init GM component that generates personas using any PersonaGenerator.

  See: https://arxiv.org/abs/2602.03545

  Works like FormativeMemoriesInitializer but accepts a pluggable generator
  satisfying the PersonaGenerator protocol. Runs Stage 1 (characteristics)
  then Stage 2 (memories) for each generated persona, and injects the
  resulting memories into agent memory banks via MakeObservation.

  The component runs once during initialization, then passes control to the
  next game master.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      generator: PersonaGenerator,
      next_game_master_name: str,
      player_names: Sequence[str],
      initial_context: str,
      diversity_axes: Sequence[str],
      shared_memories: Sequence[str] = (),
      components: Sequence[str] = (),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = '[Persona Initializer]',
  ):
    """Initializes the PersonaInitializer component.

    Args:
      model: The language model to use.
      generator: A PersonaGenerator instance (any object satisfying the
        protocol). Typically obtained from a registry, e.g.
        ``registry.get_generator('alphaevolve_5', model)``.
      next_game_master_name: Name of the GM to pass control to after
        initialization.
      player_names: Names of all player entities to generate personas for.
      initial_context: Shared context string passed to Stage 1 (e.g. scenario
        description, world setting).
      diversity_axes: Axes along which to encourage diversity in generated
        personas (e.g. personality traits, demographics).
      shared_memories: Memories shared by all players, injected verbatim before
        persona-specific memories.
      components: Keys of GM components (like Time) to condition on.
      memory_component_key: The GM's memory component key.
      make_observation_component_key: The MakeObservation component key.
      pre_act_label: Label for logging.
    """
    super().__init__()
    self._model = model
    self._generator = generator
    self._next_game_master_name = next_game_master_name
    self._player_names = list(player_names)
    self._initial_context = initial_context
    self._diversity_axes = list(diversity_axes)
    self._shared_memories = list(shared_memories)
    self._components = components
    self._memory_component_key = memory_component_key
    self._make_observation_component_key = make_observation_component_key
    self._pre_act_label = pre_act_label

    self._initialized = False
    self._persona_data: list[dict[str, Any]] = []

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def _get_context_string(self) -> str:
    return '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Runs persona generation on first call, then hands off to next GM."""
    if action_spec.output_type != entity_lib.OutputType.NEXT_GAME_MASTER:
      return ''

    if not self._initialized:
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component.Memory
      )
      make_observation = self.get_entity().get_component(
          self._make_observation_component_key,
          type_=make_observation_component.MakeObservation,
      )

      self._run_initialization(memory, make_observation)
      self._initialized = True
      return self.get_entity().name

    # Check if there are still pending observations to deliver.
    make_observation = self.get_entity().get_component(
        self._make_observation_component_key,
        type_=make_observation_component.MakeObservation,
    )
    queue_state = make_observation.get_state().get('queue', {})
    has_pending = any(bool(v) for v in queue_state.values())
    if has_pending:
      return self.get_entity().name

    return self._next_game_master_name

  def _run_initialization(
      self,
      memory: memory_component.Memory,
      make_observation: make_observation_component.MakeObservation,
  ):
    """Runs the full two-stage persona generation pipeline."""
    num_personas = len(self._player_names)

    # ---- Stage 1: Generate characteristics ----
    logging.info(
        '%s Stage 1: Generating characteristics for %d personas.',
        self._pre_act_label,
        num_personas,
    )
    full_context = self._initial_context
    component_context = self._get_context_string()
    if component_context:
      full_context += (
          f'\n\nAdditional Context from Environment:\n{component_context}'
      )

    characteristics = self._generator.generate_diverse_persona_characteristics(
        initial_context=full_context,
        diversity_axes=self._diversity_axes,
        num_personas=num_personas,
    )

    self._logging_channel({
        'Key': f'{self._pre_act_label} Stage 1 Characteristics',
        'Value': str(characteristics),
    })

    if len(characteristics) != num_personas:
      logging.warning(
          '%s Generated %d characteristics but expected %d. '
          'Using min(%d, %d) personas.',
          self._pre_act_label,
          len(characteristics),
          num_personas,
          len(characteristics),
          num_personas,
      )

    # ---- Stage 2: Generate memories (concurrent per persona) ----
    def _generate_memories_for_persona(
        player_name: str,
        persona_details: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, Any]]:
      logging.info(
          '%s Stage 2: Generating memories for %s.',
          self._pre_act_label,
          player_name,
      )
      memories = self._generator.generate_single_persona_memories(
          persona_details
      )
      return (player_name, memories, persona_details)

    # Pair players with characteristics (zip truncates to shorter).
    persona_tasks = {}
    for player_name, persona in zip(self._player_names, characteristics):
      persona_tasks[player_name] = functools.partial(
          _generate_memories_for_persona,
          player_name=player_name,
          persona_details=persona,
      )

    results = concurrency.run_tasks(persona_tasks)

    # ---- Inject memories ----
    for player_name in self._player_names:
      if player_name not in results:
        continue

      _, memories, persona_details = results[player_name]

      # Add shared memories first.
      for shared_mem in self._shared_memories:
        make_observation.add_to_queue(player_name, shared_mem)
        memory.add(f'[shared memory] {shared_mem}')

      # Add persona-specific memories.
      for mem in memories:
        make_observation.add_to_queue(player_name, mem)
        memory.add(f'[persona memory] {player_name}: {mem}')

      self._persona_data.append({
          'player_name': player_name,
          'characteristics': persona_details,
          'memories': memories,
      })

      self._logging_channel({
          'Key': f'{self._pre_act_label} Stage 2 Memories ({player_name})',
          'Memories': memories,
          'Characteristics': str(persona_details),
      })

    logging.info(
        '%s Initialization complete. Generated %d personas.',
        self._pre_act_label,
        len(self._persona_data),
    )

  @property
  def persona_data(self) -> list[dict[str, Any]]:
    """Returns generated persona data (available after initialization)."""
    return list(self._persona_data)

  def get_state(self) -> entity_component.ComponentState:
    return {
        'initialized': self._initialized,
        'persona_data': self._persona_data,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._initialized = state.get('initialized', False)
    self._persona_data = cast(
        list[dict[str, Any]], state.get('persona_data', [])
    )
