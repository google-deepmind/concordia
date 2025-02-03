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

"""A generic factory to configure simulations."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as components_lib
from concordia.components.agent import memory_component
from concordia.components.game_master import unstable as gm_components_lib
from concordia.contrib.components.agent import observations_since_last_update
from concordia.contrib.components.agent import situation_representation_via_narrative
from concordia.document import interactive_document
from concordia.environment.scenes.unstable import runner
from concordia.environment.unstable import engine as engine_lib
from concordia.environment.unstable.engines import synchronous
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import entity_component as entity_component_lib
from concordia.typing import scene as scene_lib
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib
import numpy as np


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    importance_model: importance_function.ImportanceModel,
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    memory: associative_memory.AssociativeMemory | None = None,
    supporting_players_at_fixed_locations: Sequence[str] | None = None,
    nonplayer_entities: Sequence[
        entity_component_lib.EntityWithComponents] = tuple([]),
    event_resolution_steps: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
) -> tuple[engine_lib.Engine,
           associative_memory.AssociativeMemory,
           entity_agent_with_logging.EntityAgentWithLogging]:
  """Build a simulation (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    importance_model: The importance model to use for game master memories.
    clock: The simulation clock.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    supporting_players_at_fixed_locations: The locations where supporting
      characters who never move are located.
    nonplayer_entities: The non-player entities.
    event_resolution_steps: thinking steps for the event resolution component
      to use whenever it converts putative events like action suggestions into
      real events in the simulation.

  Returns:
    A tuple consisting of a game master and its memory.
  """
  measurements = measurements_lib.Measurements()

  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=embedder,
        importance=importance_model.importance,
        clock=clock.now,
    )

  gm_memory_bank = legacy_associative_memory.AssociativeMemoryBank(
      game_master_memory)

  instructions = gm_components_lib.instructions.Instructions(
      # logging_channel=measurements.get_channel('Instructions').on_next,
  )

  player_names = [player.name for player in players]
  player_characters = gm_components_lib.instructions.PlayerCharacters(
      player_characters=player_names,
      # logging_channel=measurements.get_channel('PlayerCharacters').on_next,
  )

  scenario_knowledge = components_lib.constant.Constant(
      state='\n'.join(shared_memories),
      pre_act_key='\nBackground:\n',
      # logging_channel=measurements.get_channel('ScenarioKnowledge').on_next,
  )

  nonplayer_entities_list = components_lib.constant.Constant(
      state='\n'.join([entity.name for entity in nonplayer_entities]),
      pre_act_key='\nNon-player entities:\n',
      # logging_channel=measurements.get_channel('NonPlayerEntitiesList').on_next,
  )

  if supporting_players_at_fixed_locations is not None:
    supporting_character_locations_if_any = components_lib.constant.Constant(
        state='\n'.join(supporting_players_at_fixed_locations),
        pre_act_key='\nNotes:\n',
        # logging_channel=measurements.get_channel(
        #     'SupportingCharacterLocationsIfAny').on_next,
    )
  else:
    supporting_character_locations_if_any = components_lib.constant.Constant(
        state='',
        pre_act_key='\nNotes:\n',
        # logging_channel=measurements.get_channel(
        #     'SupportingCharacterLocationsIfAny').on_next,
    )

  observation_label = '\nObservation'
  observation = observations_since_last_update.ObservationsSinceLastUpdate(
      model=model,
      clock_now=clock.now,
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  situation_representation_label = '\nSituation description'
  situation_representation = (
      situation_representation_via_narrative.SituationRepresentation(
          model=model,
          clock_now=clock.now,
          components={
              _get_class_name(instructions): _get_class_name(instructions),
              _get_class_name(scenario_knowledge): _get_class_name(
                  scenario_knowledge
              ),
              _get_class_name(player_characters): _get_class_name(
                  player_characters
              ),
              _get_class_name(nonplayer_entities_list): _get_class_name(
                  nonplayer_entities_list
              ),
              _get_class_name(
                  supporting_character_locations_if_any
              ): _get_class_name(supporting_character_locations_if_any),
          },
          declare_entity_as_protagonist=False,
          pre_act_key=situation_representation_label,
          logging_channel=measurements.get_channel(
              'SituationRepresentation').on_next,
      )
  )

  entity_components = (
      instructions,
      player_characters,
      scenario_knowledge,
      supporting_character_locations_if_any,
      nonplayer_entities_list,
      observation,
      situation_representation,
  )
  components_of_game_master = {_get_class_name(component): component
                               for component in entity_components}
  components_of_game_master[
      components_lib.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          components_lib.memory_component.MemoryComponent(gm_memory_bank))

  # Define thinking steps for the event resolution component to use whenever it
  # converts putative events like action suggestions into real events in the
  # simulation.
  account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
      model=model, players=players, verbose=False
  )
  event_resolution_steps = event_resolution_steps or [
      thought_chains_lib.extract_direct_quote,
      thought_chains_lib.attempt_to_most_likely_outcome,
      thought_chains_lib.result_to_effect_caused_by_active_player,
      account_for_agency_of_others,
      thought_chains_lib.restore_direct_quote,
  ]

  event_resolution = gm_components_lib.event_resolution.EventResolution(
      model=model,
      event_resolution_steps=event_resolution_steps,
      components={
          _get_class_name(instructions): _get_class_name(instructions),
          _get_class_name(player_characters): _get_class_name(
              player_characters
          ),
          _get_class_name(scenario_knowledge): _get_class_name(
              scenario_knowledge
          ),
          _get_class_name(
              supporting_character_locations_if_any
          ): _get_class_name(supporting_character_locations_if_any),
          _get_class_name(nonplayer_entities_list): _get_class_name(
              nonplayer_entities_list
          ),
          _get_class_name(observation): _get_class_name(observation),
          _get_class_name(situation_representation): _get_class_name(
              situation_representation),
      },
      logging_channel=measurements.get_channel('EventResolution').on_next,
  )
  components_of_game_master[
      gm_components_lib.switch_act.DEFAULT_RESOLUTION_COMPONENT_NAME] = (
          event_resolution)
  component_order = list(components_of_game_master.keys())

  act_component = gm_components_lib.switch_act.SwitchAct(
      model=model,
      clock=clock,
      entity_names=player_names,
      component_order=component_order,
      logging_channel=measurements.get_channel('SwitchAct').on_next,
  )

  game_master = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='GM',
      act_component=act_component,
      context_components=components_of_game_master,
      component_logging=measurements,
  )

  # Create the game master object
  env = synchronous.Synchronous()

  return env, game_master_memory, game_master


def create_log(
    *,
    model: language_model.LanguageModel,
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    raw_log: list[Mapping[str, Any]],
    summarize_entire_episode: bool = True,
) -> str:
  """Create an HTML log of the simulation.

  Args:
    model: The language model to use.
    scenes: Sequence of scenes.
    raw_log: The raw log of the simulation, not yet formatted for display.
    summarize_entire_episode: Optionally, summarize the entire episode. This may
      load a lot of tokens into a language model all at once and in some cases
      exceed the model's context window and cause it to crash.

  Returns:
    An HTML string log of the simulation.
  """
  memories_per_scene = []
  for scene in scenes:
    scene_type = scene.scene_type
    scene_game_master = scene_type.game_master
    scene_memories = scene_game_master.get_component(
        memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        type_=memory_component.MemoryComponent).get_all_memories_as_text()
    memories_per_scene.append('\n'.join(scene_memories))

  if summarize_entire_episode:
    detailed_story = '\n'.join(memories_per_scene)
    print(f'detailed_story:\n{detailed_story}\n[end]\n')
    episode_summary = model.sample_text(
        f'Sequence of events:\n{detailed_story}'
        + '\nNarratively summarize the above temporally ordered '
        + 'sequence of events. Write it as a news report. Summary:\n',
        max_tokens=3500,
        terminators=(),
    )
  else:
    episode_summary = ''

  print(f'episode_summary:\n{episode_summary}\n[end]\n')

  history_sources = []
  for scene in scenes:
    scene_type = scene.scene_type
    scene_game_master = scene_type.game_master
    history_sources.append(scene_game_master)

  results_log = html_lib.PythonObjectToHTMLConverter(raw_log).convert()
  tabbed_html = html_lib.combine_html_pages(
      [results_log],
      ['GM'],
      summary=episode_summary,
      title='Simulation Log',
  )
  tabbed_html = html_lib.finalise_html(tabbed_html)

  return tabbed_html


def run_simulation(
    model: language_model.LanguageModel,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    scenes: Sequence[scene_lib.ExperimentalSceneSpec],
    verbose: bool = False,
    summarize_entire_episode_in_log: bool = True,
    compute_metrics: Callable[[Mapping[str, str]], None] | None = None,
) -> str:
  """Run a simulation.

  Args:
    model: The language model to use.
    players: The players.
    clock: The clock of the run.
    scenes: Sequence of scenes to simulate.
    verbose: Whether or not to print verbose debug information.
    summarize_entire_episode_in_log: Optionally, include summaries of the full
      episode in the log.
    compute_metrics: Optionally, a function to compute metrics.

  Returns:
    string of the log of the simulation.
  """
  raw_log = []
  # Run the simulation.
  runner.run_scenes(
      scenes=scenes,
      players=players,
      clock=clock,
      verbose=verbose,
      compute_metrics=compute_metrics,
      log=raw_log,
  )
  result_log = create_log(
      model=model,
      scenes=scenes,
      raw_log=raw_log,
      summarize_entire_episode=summarize_entire_episode_in_log,
  )
  return result_log
