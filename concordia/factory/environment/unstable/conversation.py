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

"""A factory to configure game masters specialized for handling conversation."""

from collections.abc import Sequence

from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.components.agent import unstable as actor_components
from concordia.components.game_master import unstable as gm_components
from concordia.language_model import language_model
from concordia.thought_chains.unstable import thought_chains as thought_chains_lib
from concordia.typing.unstable import entity as entity_lib


def build(
    model: language_model.LanguageModel,
    memory_bank: list[str] | associative_memory.AssociativeMemoryBank,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'conversation rules',
    next_game_master_name: str = 'default rules',
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a game master with settings specialized for conversation.

  Args:
    model: The language model to use for game master.
    memory_bank: provide a memory_bank.
    players: A sequence of players (entities).
    name: The name of the game master to build.
    next_game_master_name: The name of the game master to return control to
       after the conversation is over.

  Returns:
    A conversation game master.
  """
  player_names = [player.name for player in players]

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
      actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY)
  observation = actor_components.observation.LastNObservations(
      history_length=1000,
  )

  display_events_key = 'display_events'
  display_events = gm_components.event_resolution.DisplayEvents(
      model=model,
      pre_act_label='Conversation',
  )

  memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
  memory_component = actor_components.memory.AssociativeMemory(
      memory_bank=memory_bank)

  relevant_memories_key = 'relevant_memories'
  relevant_memories = (
      actor_components.all_similar_memories.AllSimilarMemories(
          model=model,
          components={
              display_events_key: display_events.get_pre_act_label(),
          },
          num_memories_to_retrieve=10,
          pre_act_label='Background info',
      )
  )

  make_observation_key = (
      gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY)
  make_observation = (
      gm_components.make_observation.SendComponentPreActValuesToPlayers(
          model=model,
          player_names=player_names,
          components={
              display_events_key: display_events.get_pre_act_label(),
          },
      )
  )

  next_acting_kwargs = dict(
      model=model,
      components={
          instructions_key: instructions.get_pre_act_label(),
          player_characters_key: (
              player_characters.get_pre_act_label()),
          relevant_memories_key: relevant_memories.get_pre_act_label(),
          display_events_key: display_events.get_pre_act_label(),
      },
  )
  next_actor_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY)
  next_actor = gm_components.next_acting.NextActing(
      **next_acting_kwargs,
      player_names=player_names,
  )
  next_action_spec_key = (
      gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY)
  next_action_spec = gm_components.next_acting.FixedActionSpec(
      action_spec=entity_lib.DEFAULT_SPEECH_ACTION_SPEC,
  )

  repetitive_conversations_end_key = 'repetitive_conversations_end'
  repetitive_conversations_end = (
      actor_components.constant.Constant(
          state=('Any conversation that becomes repetitive always ends '
                 'immediately. Long conversations should also end now.'),
          pre_act_label='Note',
      )
  )

  map_game_master_names_to_choices = {
      next_game_master_name: 'Yes, the conversation is over',
      name: 'No, the conversation will continue',
  }

  next_game_master_key = (
      gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY)
  next_game_master = gm_components.next_game_master.NextGameMaster(
      model=model,
      map_game_master_names_to_choices=map_game_master_names_to_choices,
      call_to_action='Is the conversation finished?',
      components={
          instructions_key: instructions.get_pre_act_label(),
          player_characters_key: (
              player_characters.get_pre_act_label()),
          repetitive_conversations_end_key: (
              repetitive_conversations_end.get_pre_act_label()),
          relevant_memories_key: relevant_memories.get_pre_act_label(),
          display_events_key: display_events.get_pre_act_label(),
      },
  )

  identity_without_prefix = thought_chains_lib.RemoveSpecificText(
      substring_to_remove='Putative event to resolve:  '
  )

  event_resolution_key = (
      gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY)
  event_resolution = gm_components.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=(identity_without_prefix,),
      notify_observers=False,
  )

  components_of_game_master = {
      instructions_key: instructions,
      examples_synchronous_key: examples_synchronous,
      player_characters_key: player_characters,
      repetitive_conversations_end_key: repetitive_conversations_end,
      relevant_memories_key: relevant_memories,
      observation_component_key: observation,
      observation_to_memory_key: observation_to_memory,
      display_events_key: display_events,
      memory_component_key: memory_component,
      next_game_master_key: next_game_master,
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

  conversation_game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=name,
      act_component=act_component,
      context_components=components_of_game_master,
  )
  return conversation_game_master
