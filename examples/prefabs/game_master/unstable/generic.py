# Copyright 2023 DeepMind Technologies Limited.
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

"""A prefab containing a generic game master."""

from collections.abc import Mapping, Sequence
import copy
import dataclasses

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory
from concordia.components.agent import unstable as actor_components
from concordia.components.game_master import unstable as gm_components
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab entity implementing a generic game master."""

  description: str = 'A general purpose game master.'
  params: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: {
          'name': 'default rules',
          # Provide a comma-separated list of thought chain steps to use in the
          # event resolution component.
          'extra_event_resolution_steps': '',
      }
  )
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging]
  ) = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a game master (i.e. a kind of entity).

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      An entity.
    """
    name = self.params.get('name')
    extra_event_resolution_steps = self.params.get(
        'extra_event_resolution_steps', '')
    assert isinstance(extra_event_resolution_steps, str)  # For pytype.

    if ',' in extra_event_resolution_steps:
      extra_event_resolution_steps = [
          step.strip() for step in extra_event_resolution_steps.split(',')
          if step]
    else:
      extra_event_resolution_steps = [extra_event_resolution_steps]

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()

    examples_synchronous_key = 'examples'
    examples_synchronous = gm_components.instructions.ExamplesSynchronous()

    player_names = [entity.name for entity in self.entities]
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

    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label=(
            'Story so far (ordered from oldest to most recent events)'),
    )

    relevant_memories_key = 'relevant_memories'
    relevant_memories = (
        actor_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components={
                display_events_key: display_events.get_pre_act_label(),
            },
            num_memories_to_retrieve=5,
            pre_act_label='Background info',
        )
    )

    world_state_key = 'world_state'
    world_state = gm_components.world_state.WorldState(
        model=model,
        components={
            instructions_key: instructions.get_pre_act_label(),
            player_characters_key: player_characters.get_pre_act_label(),
            relevant_memories_key: relevant_memories.get_pre_act_label(),
            display_events_key: display_events.get_pre_act_label(),
        },
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components={
            instructions_key: instructions.get_pre_act_label(),
            player_characters_key: player_characters.get_pre_act_label(),
            relevant_memories_key: relevant_memories.get_pre_act_label(),
            display_events_key: display_events.get_pre_act_label(),
            world_state_key: world_state.get_pre_act_label(),
        },
        reformat_observations_in_specified_style=(
            'The format to use when describing the '
            'current situation to a player is: '
            '"//date or time//situation description".'
        ),
    )

    next_acting_kwargs = dict(
        model=model,
        components={
            instructions_key: instructions.get_pre_act_label(),
            player_characters_key: player_characters.get_pre_act_label(),
            relevant_memories_key: relevant_memories.get_pre_act_label(),
            display_events_key: display_events.get_pre_act_label(),
            world_state_key: world_state.get_pre_act_label(),
        },
    )
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = gm_components.next_acting.NextActing(
        **next_acting_kwargs,
        player_names=player_names,
    )
    next_action_spec_kwargs = copy.copy(next_acting_kwargs)
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.NextActionSpec(
        **next_action_spec_kwargs,
        player_names=player_names,
    )

    # Define thinking steps for the event resolution component to use whenever
    # it converts putative events like action suggestions into real events in
    # the simulation.
    account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
        model=model, players=self.entities, verbose=False
    )

    event_resolution_steps = [
        thought_chains_lib.maybe_inject_narrative_push,
        account_for_agency_of_others,
        thought_chains_lib.result_to_who_what_where,
    ]
    if extra_event_resolution_steps:
      for step in extra_event_resolution_steps:
        if step:
          event_resolution_steps.append(getattr(thought_chains_lib, step))

    event_resolution_components = {
        instructions_key: instructions.get_pre_act_label(),
        player_characters_key: player_characters.get_pre_act_label(),
        relevant_memories_key: relevant_memories.get_pre_act_label(),
        display_events_key: display_events.get_pre_act_label(),
        world_state_key: world_state.get_pre_act_label(),
    }

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    event_resolution = gm_components.event_resolution.EventResolution(
        model=model,
        event_resolution_steps=event_resolution_steps,
        components=event_resolution_components,
        notify_observers=True,
    )

    components_of_game_master = {
        instructions_key: instructions,
        examples_synchronous_key: examples_synchronous,
        player_characters_key: player_characters,
        relevant_memories_key: relevant_memories,
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        display_events_key: display_events,
        world_state_key: world_state,
        memory_component_key: memory_component,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
    }

    component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )

    return game_master
