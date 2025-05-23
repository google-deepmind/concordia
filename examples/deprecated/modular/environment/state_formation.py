# Copyright 2024 DeepMind Technologies Limited.
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

"""A Concordia Environment Configuration."""

from collections.abc import Callable, Mapping, Sequence
import datetime
import functools
import math
import random
import types

from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import formative_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components import deprecated as generic_components
from concordia.components.agent import deprecated as agent_components
from concordia.contrib.components.game_master.deprecated import agreement_tracker
from concordia.contrib.components.game_master.deprecated import daily_activities
from concordia.deprecated.factory.agent import basic_agent
from concordia.deprecated.factory.environment import basic_game_master
from concordia.document import interactive_document
from concordia.environment.deprecated import game_master
from concordia.environment.deprecated.scenes import conversation
from examples.deprecated.modular.environment.modules import player_traits_and_styles
from examples.deprecated.modular.environment.supporting_agent_factory import basic_agent as basic_agent_supporting
from examples.deprecated.modular.environment.utils import helper_functions
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.utils import logging_types as logging_lib
from concordia.language_model import language_model
from concordia.thought_chains.deprecated import thought_chains as thought_chains_lib
from concordia.typing.deprecated import entity as entity_lib
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils.deprecated import measurements as measurements_lib
import immutabledict
from ml_collections import config_dict
import numpy as np


DEFAULT_TIME_AND_PLACE_MODULES = (
    'pre_state_villages',
)

ActivityConfig = daily_activities.ActivityConfig

MAJOR_TIME_STEP = datetime.timedelta(minutes=20)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

RESOLUTION_SCENE_TYPE = 'decision'
DEBRIEF_SCENE_TYPE = 'debrief'
NEGOTIATION_SCENE_TYPE = 'negotiation'
POST_NEGOTIATION_SCENE_TYPE = 'post_negotiation'


def _get_conjunction_of_names_string(
    player_names: Sequence[str], and_or: str = 'or'
) -> str:
  """Get a string listing the players [a, b, c] like 'a, b, or c'."""
  player_names_str = ''
  if len(player_names) > 1:
    player_names_str = ', '.join(player_names[:-1])
    player_names_str += f', {and_or} {player_names[-1]}'
  elif len(player_names) == 1:
    player_names_str = player_names[0]
  return player_names_str


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
    config: config_dict.ConfigDict,
) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = config.barbarian_raid_info

  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion.\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


def configure_players(
    rng: random.Random,
    config: config_dict.ConfigDict,
) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
    dict[str, formative_memories.AgentConfig],
]:
  """Configure the players.

  Args:
    rng: the random number generator to use.
    config: the time and place configuration to use.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
  """
  supporting_player_names_village_a = [
      player[0] for player in config.supporting_characters.a
  ]
  supporting_player_names_village_b = [
      player[0] for player in config.supporting_characters.b
  ]
  and_stakeholders_village_a = _get_conjunction_of_names_string(
      supporting_player_names_village_a, and_or='and'
  )
  and_stakeholders_village_b = _get_conjunction_of_names_string(
      supporting_player_names_village_b, and_or='and'
  )
  if len(supporting_player_names_village_a) > 1:
    is_or_are = 'are'
  else:
    is_or_are = 'is'
  elder_village_a_stakeholder_knowledge = (
      f'{and_stakeholders_village_a} are respected and influential in '
      f'{config.village_a_name}. It is critical to gain their support for any '
      f'agreement to be made with {config.village_b_name}. '
      f'{and_stakeholders_village_a} {is_or_are} '
      "generally pretty reasonable. It's a good idea for "
      f'{config.main_characters.a.name} to try to represent '
      'their interests and concerns in the negotiation. There are no '
      f'other influential individuals in {config.village_a_name}.'
  )
  elder_village_b_stakeholder_knowledge = (
      f'{and_stakeholders_village_b} are respected and influential in '
      f'{config.village_b_name}. It is critical to gain their support for any '
      f'agreement to be made with {config.village_a_name}. '
      f'{and_stakeholders_village_b} {is_or_are} '
      "generally pretty reasonable. It's a good idea for "
      f'{config.main_characters.b.name} to try to represent '
      'their interests and concerns in the negotiation. There are no '
      f'other influential individuals in {config.village_b_name}.'
  )

  player_configs = [
      # Main characters:
      formative_memories.AgentConfig(
          name=config.main_characters.a.name,
          gender=config.main_characters.a.gender,
          date_of_birth=datetime.datetime(year=1700, month=9, day=13),
          goal=(
              f'{config.main_characters.a.name} wants to be a good leader and '
              f'do what is best for {config.village_a_name}.'
          ),
          context=' '.join(config.villages.a) + ' ' + config.basic_setting,
          traits=(
              f'Personality: {player_traits_and_styles.get_trait()} and '
              f'{player_traits_and_styles.get_trait()}'
          ),
          extras={
              'player_specific_memories': [
                  (
                      f'{config.main_characters.a.name} is an elder of '
                      f'{config.village_a_name}.'
                  ),
                  (
                      f'{config.main_characters.a.name} represents'
                      f' {config.village_a_name} at the diplomatic meeting at'
                      f' {config.meeting_location}.'
                  ),
                  (
                      f'The center of {config.village_a_name} is a good place'
                      ' to meet other villagers.'
                  ),
                  elder_village_a_stakeholder_knowledge,
                  config.negotiation_objective_thought,
                  *config.villages.a,
              ],
              'home_village': config.village_a_name,
              'main_character': True,
          },
      ),
      formative_memories.AgentConfig(
          name=config.main_characters.b.name,
          gender=config.main_characters.b.gender,
          date_of_birth=datetime.datetime(year=1700, month=2, day=12),
          goal=(
              f'{config.main_characters.b.name} wants to be a good leader and '
              f'do what is best for {config.village_b_name}.'
          ),
          context=' '.join(config.villages.b) + ' ' + config.basic_setting,
          traits=(
              f'Personality: {player_traits_and_styles.get_trait()} and '
              f'{player_traits_and_styles.get_trait()}'
          ),
          extras={
              'player_specific_memories': [
                  (
                      f'{config.main_characters.b.name} is an elder of '
                      f'{config.village_b_name}.'
                  ),
                  (
                      f'{config.main_characters.b.name} represents'
                      f' {config.village_b_name} at the diplomatic meeting at'
                      f' {config.meeting_location}.'
                  ),
                  (
                      f'The center of {config.village_b_name} is a good place'
                      ' to meet other villagers.'
                  ),
                  elder_village_b_stakeholder_knowledge,
                  config.negotiation_objective_thought,
                  *config.villages.b,
              ],
              'home_village': config.village_b_name,
              'main_character': True,
          },
      ),
  ]

  for idx in range(len(config.supporting_characters.a)):
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    gender = config.supporting_characters.a[idx][1]
    him_or_her = 'her'
    his_or_her = 'her'
    if gender == 'male':
      him_or_her = 'him'
      his_or_her = 'his'
    supporting_player_config = formative_memories.AgentConfig(
        name=config.supporting_characters.a[idx][0],
        gender=gender,
        date_of_birth=datetime.datetime(
            year=1725, month=birth_month, day=birth_day
        ),
        goal=(
            f'{config.supporting_characters.a[0][0]} wants to secure '
            f'prosperity for {him_or_her}self and {his_or_her} family.'
        ),
        context=' '.join(config.villages.a) + ' ' + config.basic_setting,
        traits=(
            f'Personality: {player_traits_and_styles.get_trait()} and '
            f'{player_traits_and_styles.get_trait()}'
        ),
        extras={
            'player_specific_memories': [
                (
                    f'{config.supporting_characters.a[0][0]} is from '
                    f'{config.village_a_name}.'
                ),
                (
                    f'{config.supporting_characters.a[0][0]} knows that '
                    f'{config.main_characters.a.name} will represent '
                    f'{config.village_a_name} in the meeting at '
                    f'{config.meeting_location}.'
                ),
                *config.villages.a,
            ],
            'home_village': config.village_a_name,
            'main_character': False,
            'prior_year_activity_distribution': {
                config.farming_activity: 0.5,
                config.free_time_activity: 0.5,
                config.warrior_training_activity: 0.0,
            },
        },
    )
    player_configs.append(supporting_player_config)

  for idx in range(len(config.supporting_characters.b)):
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    gender = config.supporting_characters.b[idx][1]
    him_or_her = 'her'
    his_or_her = 'her'
    if gender == 'male':
      him_or_her = 'him'
      his_or_her = 'his'
    supporting_player_config = formative_memories.AgentConfig(
        name=config.supporting_characters.b[idx][0],
        gender=gender,
        date_of_birth=datetime.datetime(
            year=1725, month=birth_month, day=birth_day
        ),
        goal=(
            f'{config.supporting_characters.a[0][0]} wants to secure '
            f'prosperity for {him_or_her}self and {his_or_her} family.'
        ),
        context=' '.join(config.villages.b) + ' ' + config.basic_setting,
        traits=(
            f'Personality: {player_traits_and_styles.get_trait()} and '
            f'{player_traits_and_styles.get_trait()}'
        ),
        extras={
            'player_specific_memories': [
                (
                    f'{config.supporting_characters.b[0][0]} is from '
                    f'{config.village_b_name}.'
                ),
                (
                    f'{config.supporting_characters.b[0][0]} knows that '
                    f'{config.main_characters.b.name} will represent '
                    f'{config.village_b_name} in the meeting at '
                    f'{config.meeting_location}.'
                ),
                *config.villages.b,
            ],
            'home_village': config.village_b_name,
            'main_character': False,
            'prior_year_activity_distribution': {
                config.farming_activity: 0.5,
                config.free_time_activity: 0.3,
                config.warrior_training_activity: 0.2,
            },
        },
    )
    player_configs.append(supporting_player_config)

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  player_configs_dict = {player.name: player for player in player_configs}

  return main_player_configs, supporting_player_configs, player_configs_dict


def configure_scenes(
    config: config_dict.ConfigDict,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    player_configs_dict: dict[str, formative_memories.AgentConfig],
    no_conversation_game_master: game_master.GameMaster,
    negotiation_game_master: game_master.GameMaster,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    config: the time and place configuration to use
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
    no_conversation_game_master: secondary game master that does not include
      conversations
    negotiation_game_master: secondary game master for the negotiation scene

  Returns:
    scenes: the scenes to use
  """
  main_player_configs = list(main_player_configs)
  supporting_player_configs = list(supporting_player_configs)

  year_increment = datetime.timedelta(days=365)

  main_player_names = [config.name for config in main_player_configs]
  supporting_player_names = [
      config.name for config in supporting_player_configs
  ]
  supporting_player_names_village_a = [
      player[0] for player in config.supporting_characters.a
  ]
  supporting_player_names_village_b = [
      player[0] for player in config.supporting_characters.b
  ]
  and_stakeholders_village_a = _get_conjunction_of_names_string(
      supporting_player_names_village_a, and_or='and'
  )
  and_stakeholders_village_b = _get_conjunction_of_names_string(
      supporting_player_names_village_b, and_or='and'
  )

  home_scene_premises = {}
  for name in main_player_names:
    village_name = player_configs_dict[name].extras['home_village']
    if village_name == config.village_a_name:
      and_stakeholders = and_stakeholders_village_a
    elif village_name == config.village_b_name:
      and_stakeholders = and_stakeholders_village_b
    else:
      raise ValueError(f'Unknown village name: {village_name}')
    home_scene_premises[name] = (
        config.home_phase_premise.format(
            player_name=name,
            village_name=village_name,
            and_supporting_characters=and_stakeholders
        ),
    )
  for name in supporting_player_names:
    home_scene_premises[name] = (
        config.supporting_character_home_phase_premise.format(
            player_name=name,
            village_name=player_configs_dict[name].extras['home_village'],
        ),
    )

  negotiation_phase_premises = {}
  for name in main_player_names:
    negotiation_phase_premises[name] = (
        config.negotiation_phase_premise.format(
            player_name=name,
            village_name=player_configs_dict[name].extras['home_village'],
        ),
        config.negotiation_phase_extra_premise,
        config.negotiation_phase_premise_addendum,
    )

  scene_types = {}
  scene_types['home'] = scene_lib.SceneTypeSpec(
      name='home',
      premise=home_scene_premises,
  )
  scene_types[NEGOTIATION_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=NEGOTIATION_SCENE_TYPE,
      premise=negotiation_phase_premises,
      override_game_master=negotiation_game_master,
  )
  scene_types[POST_NEGOTIATION_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=POST_NEGOTIATION_SCENE_TYPE,
      premise={},
      action_spec=entity_lib.free_action_spec(
          call_to_action=(
              'In {name}\'s view, was there an agreement to pool '
              'agricultural products between villages such that a '
              'village with less food can be resupplied by a village '
              'with more food?'),
          tag='announcement',
      ),
      override_game_master=no_conversation_game_master,
  )
  activities_str = (', '.join(config.activities[:-1]) +
                    f', and {config.activities[-1]}')
  scene_types[RESOLUTION_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=RESOLUTION_SCENE_TYPE,
      premise={},
      action_spec=entity_lib.free_action_spec(
          call_to_action=('How does {name} intend to spend the rest of the '
                          'year? What daily activities will they devote '
                          'their time to? Why? Respond by giving the '
                          'average proportion of their time that they '
                          'intend to spend on each of the '
                          f'following activities: {activities_str}. Note that '
                          'proportions of time should sum to 1. Sleep counts '
                          f'as {config.free_time_activity} and, '
                          'all else equal, '
                          'more leisure time is better.'),
          tag='announcement',
      ),
      override_game_master=no_conversation_game_master,
  )
  scene_types[DEBRIEF_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=DEBRIEF_SCENE_TYPE,
      premise={},
      action_spec=entity_lib.free_action_spec(
          call_to_action=('How does {name} feel about how this year went? '
                          'What went well? What did not go well? What does '
                          '{name} feel could be improved for next year?'),
          tag='reflection',
      ),
      override_game_master=no_conversation_game_master,
  )

  scenes = [
      # Year 1
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=config.times.start,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[NEGOTIATION_SCENE_TYPE],
          start_time=config.times.meeting,
          participant_configs=main_player_configs,
          num_rounds=3,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[POST_NEGOTIATION_SCENE_TYPE],
          start_time=config.times.post_meeting,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=config.times.return_home,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[RESOLUTION_SCENE_TYPE],
          start_time=config.times.decision,
          participant_configs=supporting_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[DEBRIEF_SCENE_TYPE],
          start_time=config.times.debrief,
          participant_configs=main_player_configs + supporting_player_configs,
          num_rounds=1,
      ),
  ]

  for i in range(1, config.num_years):
    year = i * year_increment
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_types[NEGOTIATION_SCENE_TYPE],
            start_time=config.times.meeting + year,
            participant_configs=main_player_configs,
            num_rounds=3,
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_types[POST_NEGOTIATION_SCENE_TYPE],
            start_time=config.times.post_meeting + year,
            participant_configs=main_player_configs,
            num_rounds=1,
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_types['home'],
            start_time=config.times.return_home + year,
            participant_configs=main_player_configs,
            num_rounds=1,
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_types[RESOLUTION_SCENE_TYPE],
            start_time=config.times.decision + year,
            participant_configs=supporting_player_configs,
            num_rounds=1,
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_types[DEBRIEF_SCENE_TYPE],
            start_time=config.times.debrief + year,
            participant_configs=[],
            num_rounds=1,
        )
    )

  return scenes


# The following two functions are used to calculate the production functions
# of villagers used to calculate the amount of product produced by their
# activities e.g. farming, training as a warrior, or free time.
# The specific parameters of these functions were chosen so that
# _sigmoid_like_fn maps 0.0 to 0.0 and 1.0 to 1.0, and it grows supralinearly
# around 0.5, i,e. sigmoidlike_fn(0.5) = 0.73.
def _sigmoid(x):
  return (1 / (1 + math.exp(-(x / 0.3))))


def _sigmoidlike_fn(x):
  return (_sigmoid(x) - 0.5) / (_sigmoid(1.0) - 0.5)


def get_daily_activities_component(
    model: language_model.LanguageModel,
    config: config_dict.ConfigDict,
    memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent.EntityAgent],
    supporting_players: Sequence[entity_agent.EntityAgent],
    player_configs: Sequence[formative_memories.AgentConfig],
    clock_now: Callable[[], datetime.datetime] = datetime.datetime.now,
    basic_setting: str = '',
) -> tuple[
    daily_activities.DailyActivities,
    daily_activities.Payoffs,
]:
  """Get the daily activities tracking component for the game master."""

  village_by_representative = {
      config.main_characters.a.name: config.village_a_name,
      config.main_characters.b.name: config.village_b_name,
  }

  supporting_player_names = [player.name for player in supporting_players]

  activity_configs = tuple(
      [ActivityConfig(name=activity) for activity in config.activities])
  player_initial_activity_distribution = {
      config.name: config.extras['prior_year_activity_distribution']
      for config in player_configs if config.name in supporting_player_names
  }
  activities_component = daily_activities.DailyActivities(
      model=model,
      memory=memory,
      activity_configs=activity_configs,
      resolution_scene=RESOLUTION_SCENE_TYPE,
      players=players,
      player_initial_activity_distribution=player_initial_activity_distribution,
      clock_now=clock_now,
      num_to_retrieve=1,
      basic_setting=basic_setting,
      name='Daily activities\n',
      verbose=False,
  )

  village_a_denizens = [name for name, _ in config.supporting_characters.a]
  village_b_denizens = [name for name, _ in config.supporting_characters.b]

  def _get_overall_activity_product_per_village(
      activity_proportions: Mapping[str, Mapping[str, float]],
      activity_name: str,
  ) -> Mapping[str, float]:
    activity_of_village_a = 0.0
    activity_of_village_b = 0.0
    for player_name, activities in activity_proportions.items():
      if player_name in village_a_denizens:
        activity_of_village_a += _sigmoidlike_fn(activities[activity_name])
      if player_name in village_b_denizens:
        activity_of_village_b += _sigmoidlike_fn(activities[activity_name])
    overall_proportion_activity_of_village_a = (
        activity_of_village_a / len(village_a_denizens))
    overall_proportion_activity_of_village_b = (
        activity_of_village_b / len(village_b_denizens))
    return {config.village_a_name: overall_proportion_activity_of_village_a,
            config.village_b_name: overall_proportion_activity_of_village_b}

  def _are_agricultural_resources_pooled(
      game_master_memory: associative_memory.AssociativeMemory,
      timepoint: str) -> bool:
    """Check if villagers have agreed to pool agricultural resources."""
    retrieved = game_master_memory.retrieve_by_regex(
        regex=r'\[' + timepoint + r'\].*',
        sort_by_time=True,
    )
    result = False
    if retrieved:
      retrieved_str = '\n'.join(retrieved)
      villager_names_str = ', '.join(supporting_player_names)
      chain_of_thought = interactive_document.InteractiveDocument(model)
      chain_of_thought.statement(f'Record of events:\n{retrieved_str}')
      result = chain_of_thought.yes_no_question(
          question=('Is there evidence in the above record of events that '
                    'there is an agreement in place for pooling '
                    'agricultural products between villages such that a '
                    'village with less food can be resupplied by a village '
                    'with more food? Only consider the following villagers: '
                    f'{villager_names_str}. If there is no evidence of a '
                    'particular villager\'s opinion, then assume that villager '
                    'is against sharing agricultural resources.')
      )
    return result

  def player_score_fn(
      current_scene_type: str,
      activity_proportions: Mapping[str, Mapping[str, float]],
      player_name: str,
      game_master_memory: associative_memory.AssociativeMemory,
      timepoint: str,
  ) -> tuple[float, Sequence[str]]:
    """Get the individual part of the score."""
    score = 0.0
    events = []
    if current_scene_type != DEBRIEF_SCENE_TYPE:
      return score, events

    # Shared part of the score (common defense) is calculated first.
    defense_per_village = _get_overall_activity_product_per_village(
        activity_proportions, config.warrior_training_activity)
    # Overall defense is the mean defense over villages because barbarians
    # may attack randomly near any village equiprobably.
    raw_overall_defense = np.mean(list(defense_per_village.values()))
    if raw_overall_defense < config.defense_threshold:
      defense = 0.0
      events.append(config.sample_event_of_failing_to_repel_barbarians())
    else:
      defense = 1.0
      events.append(config.sample_event_of_success_repelling_barbarians())

    # Individual part of the score (farming and free time) is calculated next.
    if player_name in village_by_representative:
      this_village = village_by_representative[player_name]
    else:
      if player_name in  [name for name, _ in config.supporting_characters.a]:
        this_village = config.village_a_name
      elif player_name in [name for name, _ in config.supporting_characters.b]:
        this_village = config.village_b_name
      else:
        raise ValueError(f'Player {player_name} not found in any '
                         f'village: {village_by_representative} -- '
                         f'{config.supporting_characters.a} -- '
                         f'{config.supporting_characters.b}')

    farming_per_village = _get_overall_activity_product_per_village(
        activity_proportions, config.farming_activity)
    if _are_agricultural_resources_pooled(game_master_memory,
                                          timepoint):
      # If villagers have agreed to resupply whichever village has less food
      # then all villages receive the max of their agricultural production, so
      # starvation is unlikely unless all villages starve together.
      raw_agriculture = np.max(list(farming_per_village.values()))
      events.append(config.sample_event_treaty_in_effect())
    else:
      # If villagers did not agree to pool agricultural resources then each
      # village is on its own and villages that do not farm enough starve.
      raw_agriculture = farming_per_village[this_village]
      events.append(config.sample_event_no_treaty_in_effect())

    if raw_agriculture < config.starvation_threshold:
      agriculture = 0.0
      events.append(config.sample_event_of_failing_to_grow_food())
    else:
      agriculture = 1.0
      events.append(config.sample_event_of_success_growing_food())

    # Villager free time contributes positively to the overall score.
    free_time_per_village = _get_overall_activity_product_per_village(
        activity_proportions, config.free_time_activity)
    free_time = free_time_per_village[this_village]

    # The overall score is the product of the activity scores. Note that two of
    # them are constrained to be binary (defense and agriculture) so the only
    # effect they can have is to gate the free time score.
    score = defense * agriculture * free_time

    return score, events

  payoffs = daily_activities.Payoffs(
      memory=memory,
      daily_activities=activities_component,
      players=players,
      player_score_fn=player_score_fn,
      get_timepoint_fn=lambda: str(clock_now().year),
      clock_now=clock_now,
      name='Payoffs',
  )
  return activities_component, payoffs


class Simulation(scenarios_lib.RunnableSimulationWithMemories):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_agent,
      override_agent_model: language_model.LanguageModel | None = None,
      resident_visitor_modules: Sequence[types.ModuleType] | None = None,
      supporting_agent_module: types.ModuleType = basic_agent_supporting,
      time_and_place_module: str | None = None,
      seed: int | None = None,
  ):
    """Initialize the simulation object.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
      override_agent_model: optionally, override the model for all agents. The
        model will be copied for every agent.
      resident_visitor_modules: optionally, use different modules for majority
        and minority parts of the focal population.
      supporting_agent_module: agent module to use for all supporting players. A
        supporting player is a non-player character with a persistent memory
        that plays a specific role in defining the task environment. Their role
        is not incidental but rather is critcal to making the task what it is.
        Supporting players are not necessarily interchangeable with focal or
        background players.
      time_and_place_module: optionally, specify a module containing settings
        that create a sense of setting in a specific time and place. If not
        specified, a random module will be chosen from the default options.
      seed: optionally, specify a seed for the random number generator.
    """
    if resident_visitor_modules is None:
      self._resident_visitor_mode = False
      self._agent_module = agent_module
    else:
      self._resident_visitor_mode = True
      self._resident_agent_module, self._visitor_agent_module = (
          resident_visitor_modules
      )
    self._agent_model = model
    if override_agent_model:
      self._agent_model = override_agent_model

    self._supporting_agent_module = supporting_agent_module

    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._time_and_place_module = time_and_place_module
    _, config = helper_functions.load_time_and_place_module(
        time_and_place_module=time_and_place_module,
        default_time_and_place_modules=DEFAULT_TIME_AND_PLACE_MODULES,
        seed=seed,
    )
    self._rng = random.Random(seed)

    self._clock = game_clock.MultiIntervalClock(
        start=config.times.setup, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.ConstantImportanceModel()
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, shared_context = get_shared_memories_and_context(
        model=model, config=config)
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=config.times.setup,
    )

    main_player_configs, supporting_player_configs, player_configs_dict = (
        configure_players(
            rng=self._rng,
            config=config,
        )
    )

    tasks = {
        config.name: functools.partial(
            self._make_player_memories, player_config=config
        )
        for config in main_player_configs + supporting_player_configs
    }
    self._all_memories = concurrency.run_tasks(tasks)

    main_players = []
    self._resident_names = []
    self._visitor_names = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = {
          'config': player_config,
          'model': self._model,
          'memory': self._all_memories[player_config.name],
          'clock': self._clock,
          'update_time_interval': MAJOR_TIME_STEP,
      }
      if self._resident_visitor_mode:
        if idx == 0:
          player = self._visitor_agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
          self._visitor_names.append(player.name)
        else:
          player = self._resident_agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
          self._resident_names.append(player.name)
      else:
        player = self._agent_module.build_agent(**kwargs)  # pylint: disable=attribute-error
        self._resident_names.append(player.name)

      main_players.append(player)

    supporter_extra_components = {}
    for player_config in supporting_player_configs:
      village_name = player_config.extras['home_village']
      which_village = agent_components.constant.Constant(
          pre_act_key='\nHome village',
          state=(f'{player_config.name} lives in {village_name} and never '
                 'travels elsewhere.'),
      )
      if village_name == config.village_a_name:
        how_things_are_string = (
            config.villager_how_things_are_constant.village_a)
      elif village_name == config.village_b_name:
        how_things_are_string = (
            config.villager_how_things_are_constant.village_b)
      else:
        raise ValueError(f'Unknown village name: {village_name}')
      how_things_are = agent_components.constant.Constant(
          pre_act_key='\nHow things are',
          state=how_things_are_string.format(
              name=player_config.name,
              village_name=village_name,
          ),
      )
      conversation_style = agent_components.constant.Constant(
          pre_act_key='\nGuiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      supporter_extra_components[player_config.name] = {
          'Home village': which_village,
          'How things are': how_things_are,
          'Guiding principle of good conversation': conversation_style,
      }

    supporting_players = []
    for player_config in supporting_player_configs:
      player = self._supporting_agent_module.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components=supporter_extra_components[player_config.name],
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )
    daily_activities_component, self._score = get_daily_activities_component(
        model=model,
        config=config,
        memory=game_master_memory,
        players=self._all_players,
        supporting_players=supporting_players,
        player_configs=main_player_configs + supporting_player_configs,
        clock_now=self._clock.now,
        basic_setting=config.basic_setting,
    )

    setting = generic_components.constant.ConstantComponent(
        state=config.basic_setting,
        name='\nSetting',
    )
    magic_is_not_real = generic_components.constant.ConstantComponent(
        state='Magic is not real. Supernatural events are impossible.',
        name='\nImportant fact',
    )
    barbarians_never_nice = generic_components.constant.ConstantComponent(
        state=(
            'It is not possible to communicate with the barbarian raiders '
            'from the sea. They do not speak any language in common with '
            'the villagers. They cannot be reasoned with. They will '
            'always attempt to invade and plunder the villages.'
        ),
        name='\nCritical premise',
    )

    supporting_player_names_village_a_str = _get_conjunction_of_names_string(
        [name for name, _ in config.supporting_characters.a], and_or='or'
    )
    supporting_player_names_village_b_str = _get_conjunction_of_names_string(
        [name for name, _ in config.supporting_characters.b], and_or='or'
    )
    stakeholders_easy_to_find = generic_components.constant.ConstantComponent(
        state=(
            f'Anyone in {config.village_a_name} looking for '
            f'{supporting_player_names_village_a_str} easily finds them in the '
            'village center. They are influential and respected stakeholders '
            f'in {config.village_a_name} society.\n'
            f'Anyone in {config.village_b_name} looking for '
            f'{supporting_player_names_village_b_str} easily finds them in the '
            'village center. They are influential and respected stakeholders '
            f'in {config.village_b_name} society.'
        ),
        name='\nFact 1'
    )

    no_unofficial_communication = generic_components.constant.ConstantComponent(
        state=(
            'Direct conversation between villagers of'
            f' {config.village_a_name} and {config.village_b_name} is not'
            ' possible. The only possible way to communicate is through the'
            f' diplomatic meeting at {config.meeting_location}. Therefore, the'
            ' two elder representatives who meet at'
            f' {config.meeting_location} constitute the only channel of'
            ' communication between villages. The two elder representatives'
            ' can speak directly to each other, but they can only do so while'
            f' they are physically at {config.meeting_location}. Villagers'
            ' cannot participate in conversations at'
            f' {config.meeting_location} since it is far away. And villagers'
            ' never travel away from their home village.'
        ),
        name='\nFact 2',
    )

    additional_gm_components = [
        setting,
        magic_is_not_real,
        barbarians_never_nice,
        stakeholders_easy_to_find,
        no_unofficial_communication,
        daily_activities_component,
        self._score,
    ]

    supporting_player_locations = []
    supporting_player_locations.extend([
        (
            f'{player_data[0]} waits for {config.main_characters.a.name} in '
            f'the center of {config.village_a_name}.'
        )
        for player_data in config.supporting_characters.a
    ])
    supporting_player_locations.extend([
        (
            f'{player_data[0]} waits for {config.main_characters.b.name} in '
            f'the center of {config.village_b_name}.'
        )
        for player_data in config.supporting_characters.b
    ])

    self._primary_environment, self._game_master_memory = (
        basic_game_master.build_game_master(
            model=self._model,
            embedder=self._embedder,
            importance_model=importance_model_gm,
            clock=self._clock,
            players=self._all_players,
            shared_memories=shared_memories,
            shared_context=shared_context,
            blank_memory_factory=self._blank_memory_factory,
            max_conversation_length=2,
            cap_nonplayer_characters_in_conversation=0,
            memory=game_master_memory,
            supporting_players_at_fixed_locations=supporting_player_locations,
            additional_components=additional_gm_components,
            seed=seed,
        )
    )

    agreement_component = agreement_tracker.AgreementTracker(
        model=model,
        memory=self._game_master_memory,
        negotiating_players=main_players,
        informed_players=supporting_players,
        clock_now=self._clock.now,
        resolution_scenes=(NEGOTIATION_SCENE_TYPE, POST_NEGOTIATION_SCENE_TYPE),
        chain_of_thought_prefix='Negotiation scene',
        basic_setting=config.basic_setting,
        name='Agreement tracker',
        seed=seed,
        verbose=True,
    )

    self._no_conversation_game_master = game_master.GameMaster(
        model=model,
        memory=self._game_master_memory,
        clock=self._clock,
        name='Decision Environment',
        players=self._all_players,
        components=[agreement_component, *additional_gm_components],
        update_thought_chain=[thought_chains_lib.identity],
        randomise_initiative=True,
        player_observes_event=True,
        concurrent_externalities=False,
        seed=seed,
    )
    self._negotiation_game_master = conversation.make_conversation_game_master(
        players=main_players,
        clock=self._clock,
        model=self._model,
        memory_factory=self._blank_memory_factory,  # Unused, we pass memory
        check_for_termination=False,
        randomise_initiative=True,
        name='Negotiation scene',
        review_participants=False,
        max_steps=4,
        memory=self._game_master_memory,
        additional_components=[agreement_component],
        verbose=True,
    )

    self._scenes = configure_scenes(
        config=config,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        player_configs_dict=player_configs_dict,
        no_conversation_game_master=self._no_conversation_game_master,
        negotiation_game_master=self._negotiation_game_master,
    )
    self._secondary_environments = [self._negotiation_game_master,
                                    self._no_conversation_game_master]

    self._init_premise_memories(
        config=config,
        setup_time=config.times.setup,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=(),
        scenario_premise=[
            (
                f'{config.main_characters.a.name} and '
                f'{config.main_characters.b.name} meet periodically at '
                f'{config.meeting_location} to discuss current events and '
                'conduct diplomacy on behalf of the villages they represent.'
            ),
        ],
    )

  def _make_player_memories(
      self, player_config: formative_memories.AgentConfig
  ):
    """Make memories for a player."""
    mem = self._formative_memory_factory.make_memories(player_config)
    # Inject player-specific memories declared in the agent config.
    for extra_memory in player_config.extras['player_specific_memories']:
      mem.add(f'{extra_memory}', tags=['initial_player_specific_memory'])
    return mem

  def _init_premise_memories(
      self,
      config: config_dict.ConfigDict,
      setup_time: datetime.datetime,
      main_player_configs: list[formative_memories.AgentConfig],
      supporting_player_configs: list[formative_memories.AgentConfig],
      shared_memories: Sequence[str],
      scenario_premise: Sequence[str],
  ) -> None:
    """Initialize player memories.

    Args:
      config: the specific time and place configuration for this simulation
      setup_time: the time to set the clock to before initializing memories
      main_player_configs: configs for the main characters
      supporting_player_configs: configs for the supporting characters
      shared_memories: memories shared by all players, the game master, and NPCs
      scenario_premise: premise observation shared by all players and the game
        master.
    """
    player_configs = main_player_configs + supporting_player_configs
    main_players = [
        player
        for player in self._all_players
        if player.name
        in [player_config.name for player_config in main_player_configs]
    ]
    supporting_players = [
        player
        for player in self._all_players
        if player.name
        in [player_config.name for player_config in supporting_player_configs]
    ]

    self._clock.set(setup_time)

    for premise in scenario_premise:
      self._game_master_memory.add(premise)
      for player in self._all_players:
        player.observe(premise)

    for shared_memory in shared_memories:
      self._game_master_memory.add(shared_memory)
      for player in self._all_players:
        player.observe(shared_memory)

    # The game master also observes all the player-specific memories.
    for player_config in player_configs:
      extra_memories = player_config.extras['player_specific_memories']
      for extra_memory in extra_memories:
        self._game_master_memory.add(extra_memory)

    # Generate memory of each elder being at home in their home village.
    for idx, player in enumerate(main_players):
      village = player_configs[idx].extras['home_village']
      scene_premise = config.home_scene_premise.format(
          name=player.name, village=village)

      # Add memory to both player and GM.
      player.observe(scene_premise)
      self._game_master_memory.add(scene_premise)

    for idx, player in enumerate(supporting_players):
      village = player_configs[idx].extras['home_village']
      teleport = (
          f'{player.name} is currently in {village} and has no '
          + 'intention of leaving today.'
      )
      player.observe(teleport)
      self._game_master_memory.add(teleport)

  def get_all_player_memories(self):
    return self._all_memories

  def __call__(self)-> tuple[logging_lib.SimulationOutcome, str]:
    """Run the simulation.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    html_results_log = basic_game_master.run_simulation(
        model=self._model,
        players=self._all_players,
        primary_environment=self._primary_environment,
        secondary_environments=self._secondary_environments,
        clock=self._clock,
        scenes=self._scenes,
        summarize_entire_episode_in_log=False,
    )

    player_scores = self._score.get_scores()
    simulation_outcome = logging_lib.SimulationOutcome(
        resident_scores=immutabledict.immutabledict(
            {name: player_scores[name] for name in self._resident_names}
        ),
        visitor_scores=immutabledict.immutabledict(
            {name: player_scores[name] for name in self._visitor_names}
        ),
        metadata=immutabledict.immutabledict({
            'wallclock_time': datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            'environment': __file__,
            'time_and_place_module': self._time_and_place_module,
        }),
    )
    print('Overall scores per player:')
    if self._resident_visitor_mode:
      idx = 0
      for player_name, score in player_scores.items():
        if idx == 0:
          print('Visitor')
        else:
          print('Resident')
        print(f'  {player_name}: {score}')
        idx += 1
    else:
      for player_name, score in player_scores.items():
        print(f'{player_name}: {score}')

    return simulation_outcome, html_results_log
