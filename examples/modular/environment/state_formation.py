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
import importlib
import pathlib
import random
import sys
import types

from concordia import components as generic_components
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from examples.modular.environment.modules import player_traits_and_styles
from concordia.factory.agent import basic_entity_agent__main_role
from concordia.factory.agent import basic_entity_agent__supporting_role
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np

ENVIRONMENT_MODULES = ('pre_state_villages',)
env_module_name = random.choice(ENVIRONMENT_MODULES)

# Load the environment config with importlib
concordia_root_dir = pathlib.Path(
    __file__
).parent.parent.parent.parent.resolve()
sys.path.append(f'{concordia_root_dir}')
environment_module = importlib.import_module(
    f'examples.modular.environment.modules.{env_module_name}'
)
config = environment_module.get_world_config()

BASIC_SETTING = environment_module.BASIC_SETTING.format(
    village_a_name=config.village_a_name, village_b_name=config.village_b_name
)

Runnable = Callable[[], str]

MAJOR_TIME_STEP = datetime.timedelta(minutes=20)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

SETUP_TIME = datetime.datetime(hour=20, year=1750, month=10, day=1)
START_TIME = datetime.datetime(hour=18, year=1750, month=10, day=2)
HILL_TIME = datetime.datetime(hour=18, year=1750, month=10, day=3)
RETURN_HOME_SCENE_TIME = datetime.datetime(hour=20, year=1750, month=10, day=4)
DECISION_TIME = datetime.datetime(hour=20, year=1750, month=12, day=6)

SUPPORTING_PLAYER_NAMES_VILLAGE_A = [
    player[0] for player in config.supporting_characters.a
]
SUPPORTING_PLAYER_NAMES_VILLAGE_B = [
    player[0] for player in config.supporting_characters.b
]

RESOLUTION_SCENE_TYPE = 'decision'
SCENARIO_PREMISE = [
    (
        f'{config.main_characters.a.name} and {config.main_characters.b.name} '
        'meet periodically at the hill of accord to discuss current events and '
        'conduct diplomacy on behalf of the villages they represent.'
    ),
]
REPRESENTATIVE_BY_VILLAGE = {
    config.village_a_name: config.main_characters.a.name,
    config.village_b_name: config.main_characters.b.name,
}
SUPPORTING_PLAYER_LOCATIONS = []
SUPPORTING_PLAYER_LOCATIONS.extend([
    (
        f'{player_data[0]} waits for {config.main_characters.a.name} in the '
        f'center of {config.village_a_name}.'
    )
    for player_data in config.supporting_characters.a
])
SUPPORTING_PLAYER_LOCATIONS.extend([
    (
        f'{player_data[0]} waits for {config.main_characters.b.name} in the '
        f'center of {config.village_b_name}.'
    )
    for player_data in config.supporting_characters.b
])
DECISION_ACTION_SPEC = agent_lib.choice_action_spec(
    call_to_action=(
        'Would {name} follow through with their obligation under the agreement?'
    ),
    options=('no', 'yes'),
    tag='decision',
)


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


def configure_players() -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
    dict[str, formative_memories.AgentConfig],
]:
  """Configure the players.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
  """
  and_stakeholders_village_a = _get_conjunction_of_names_string(
      SUPPORTING_PLAYER_NAMES_VILLAGE_A, and_or='and'
  )
  and_stakeholders_village_b = _get_conjunction_of_names_string(
      SUPPORTING_PLAYER_NAMES_VILLAGE_B, and_or='and'
  )
  if len(SUPPORTING_PLAYER_NAMES_VILLAGE_A) > 1:
    is_or_are = 'are'
  else:
    is_or_are = 'is'
  elder_village_a_stakeholder_knowledge = (
      f'{and_stakeholders_village_a} are respected and influential in '
      f'{config.village_a_name}. It is critical to gain their support for any '
      f'agreement to be made with {config.village_b_name}. '
      f'{and_stakeholders_village_a} {is_or_are} '
      "generally pretty reasonable. It's a good idea for "
      f'{config.main_characters.a} to try to represent '
      'their interests in the negotiation at the hill of accord.'
  )
  elder_village_b_stakeholder_knowledge = (
      f'{and_stakeholders_village_b} are respected and influential in '
      f'{config.village_b_name}. It is critical to gain their support for any '
      f'agreement to be made with {config.village_a_name}. '
      f'{and_stakeholders_village_b} {is_or_are} '
      "generally pretty reasonable. It's a good idea for "
      f'{config.main_characters.b} to try to represent '
      'their interests in the negotiation at the hill of accord.'
  )

  player_configs = [
      # Main characters:
      formative_memories.AgentConfig(
          name=config.main_characters.a.name,
          gender=config.main_characters.a.gender,
          date_of_birth=datetime.datetime(year=1700, month=9, day=13),
          goal=(
              f'{config.main_characters.a.name} wants to do what is best for '
              f'{config.village_a_name}, especially when '
              + f'it is also best for {config.main_characters.a.name}.'
          ),
          context=' '.join(config.villages.a) + ' ' + BASIC_SETTING,
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
                      ' the hill of accord.'
                  ),
                  (
                      f'The center of {config.village_a_name} is a good place'
                      ' to meet other villagers.'
                  ),
                  elder_village_a_stakeholder_knowledge,
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
              f'({config.main_characters.b.name} wants to do what is best '
              f'for {config.village_b_name}, especially '
              + f'when it is also best for {config.main_characters.b.name}.'
          ),
          context=' '.join(config.villages.b) + ' ' + BASIC_SETTING,
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
                      ' the hill of accord.'
                  ),
                  (
                      f'The center of {config.village_b_name} is a good place'
                      ' to meet other villagers.'
                  ),
                  elder_village_b_stakeholder_knowledge,
                  *config.villages.b,
              ],
              'home_village': config.village_b_name,
              'main_character': True,
          },
      ),
  ]

  for idx in range(len(config.supporting_characters.a)):
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
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
        context=' '.join(config.villages.a) + ' ' + BASIC_SETTING,
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
                    f'{config.village_a_name} in the meeting at the hill of '
                    'accord.'
                ),
                *config.villages.a,
            ],
            'home_village': config.village_a_name,
            'main_character': False,
        },
    )
    player_configs.append(supporting_player_config)

  for idx in range(len(config.supporting_characters.b)):
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
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
        context=' '.join(config.villages.b) + ' ' + BASIC_SETTING,
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
                    f'{config.village_b_name} in the meeting at the hill of '
                    'accord.'
                ),
                *config.villages.b,
            ],
            'home_village': config.village_b_name,
            'main_character': False,
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
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    player_configs_dict: dict[str, formative_memories.AgentConfig],
    decision_env: game_master.GameMaster,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    player_configs_dict: dict mapping player name to corresponding config
    decision_env: the decision environment to use

  Returns:
    scenes: the scenes to use
  """
  year_increment = datetime.timedelta(days=365)

  main_player_names = [config.name for config in main_player_configs]
  supporting_player_names = [
      config.name for config in supporting_player_configs
  ]

  home_phase_premise = (
      'Elder {player_name} is home in {village_name}, and knows it will '
      'be critical to gain the support of influential stakeholders in the '
      'village if any agreement is to last. {player_name} should start seeking '
      'their support now. There is no time to rest.'
  )

  supporting_character_home_phase_premise = (
      '{player_name} is currently in {village_name} and has no intention of '
      'leaving today.'
  )

  elder_a = config.main_characters.a.name
  elder_b = config.main_characters.b.name

  negotiation_phase_premise = (
      'Elder {player_name} left {village_name} early in the morning and '
      'arrived just now at the hill of accord. The reason for this meeting '
      'of the two elder representatives of their respective villages '
      f'({elder_a} representing {config.village_a_name} and '
      f'{elder_b} representing {config.village_b_name}) is '
      'as follows: barbarian raiders have been pillaging and burning the land, '
      'and menacing both villages. It has been suggested that an alliance for '
      'mutual defense against the barbarian threat would be beneficial. The '
      'elders are meeting today to try to negotiate such an alliance.'
  )

  home_scene_premises = {}
  for name in main_player_names:
    home_scene_premises[name] = (
        home_phase_premise.format(
            player_name=name,
            village_name=player_configs_dict[name].extras['home_village'],
        ),
    )
  for name in supporting_player_names:
    home_scene_premises[name] = (
        supporting_character_home_phase_premise.format(
            player_name=name,
            village_name=player_configs_dict[name].extras['home_village'],
        ),
    )

  negotiation_phase_premises = {}
  for name in main_player_names:
    negotiation_phase_premises[name] = (
        negotiation_phase_premise.format(
            player_name=name,
            village_name=player_configs_dict[name].extras['home_village'],
        ),
        (
            "There is no time to waste on small talk. It's important to get"
            ' down to business immediately by proposing specific provisions for'
            ' the alliance and responding to the proposals of others.'
        ),
    )

  scene_types = {}
  scene_types['home'] = scene_lib.SceneTypeSpec(
      name='home',
      premise=home_scene_premises,
  )
  scene_types['negotiation'] = scene_lib.SceneTypeSpec(
      name='negotiation',
      premise=negotiation_phase_premises,
  )
  scene_types[RESOLUTION_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=RESOLUTION_SCENE_TYPE,
      premise={},
      action_spec=DECISION_ACTION_SPEC,
      override_game_master=decision_env,
  )

  scenes = [
      # Year 1
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=START_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types['negotiation'],
          start_time=HILL_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=RETURN_HOME_SCENE_TIME,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[RESOLUTION_SCENE_TYPE],
          start_time=DECISION_TIME,
          participant_configs=supporting_player_configs,
          num_rounds=1,
      ),
      # Year 2
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=START_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types['negotiation'],
          start_time=HILL_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=RETURN_HOME_SCENE_TIME + year_increment,
          participant_configs=main_player_configs,
          num_rounds=2,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_types[RESOLUTION_SCENE_TYPE],
          start_time=DECISION_TIME + year_increment,
          participant_configs=supporting_player_configs,
          num_rounds=1,
      ),
      # Year 3
      scene_lib.SceneSpec(
          scene_type=scene_types['home'],
          start_time=START_TIME + (2 * year_increment),
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
  ]
  return scenes


def outcome_summary_fn(
    binary_joint_action: Mapping[str, int], unused_rewards: Mapping[str, float]
) -> Mapping[str, str]:
  """Summarize outcome of decision scene (used by Schelling payoffs component).

  Args:
    binary_joint_action: map each player name to whether they cooperated or
      defected (0 indicates defection and 1 indicates cooperation).
    unused_rewards: map each player name to the reward they received

  Returns:
    result: dict mapping player name to outcome summary
  """
  result = {name: '' for name in binary_joint_action}
  num_cooperators = np.sum(list(binary_joint_action.values()))
  success = num_cooperators > 2
  common_part = ''
  if success:
    common_part += 'The barbarian invasion was successfully repulsed. '
  else:
    common_part += (
        'The barbarian invasion was not stopped. Barbarians '
        + 'overrun the region, taking whatever they please. After a season of '
        + 'terror they finally leave the region, not because they were driven '
        + 'out, but because precious little worth plundering remained. '
    )
  for player_name, action in binary_joint_action.items():
    result[player_name] += common_part
    # action == 1 indicates cooperation while action == 0 indicates defection
    if success and action == 1:
      result[player_name] += (
          f'{player_name} did their duty and helped '
          + 'achieve this great victory.'
      )
    elif success and action == 0:
      result[player_name] += (
          f'{player_name} chose not to do their duty, '
          + 'but victory was obtained nonetheless.'
      )
    elif not success and action == 1:
      result[player_name] += (
          f'{player_name} did their duty. However, too '
          + 'few others joined. The wanton cruelty of the '
          + 'barbarians caused much suffering throughout '
          + 'the region.'
      )
    elif not success and action == 0:
      result[player_name] += (
          f'{player_name} did not do their duty. '
          + 'The wanton cruelty of the barbarians caused '
          + 'much suffering throughout the region.'
      )
  return result


class Simulation(Runnable):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_entity_agent__main_role,
  ):
    """Initialize the simulation object.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
    """

    self._agent_module = agent_module
    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._clock = game_clock.MultiIntervalClock(
        start=SETUP_TIME, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.AgentImportanceModel(self._model)
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, shared_context = get_shared_memories_and_context(model)
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=SETUP_TIME,
    )

    main_player_configs, supporting_player_configs, player_configs_dict = (
        configure_players()
    )

    num_main_players = len(main_player_configs)
    num_supporting_players = len(supporting_player_configs)

    self._all_memories = {}

    main_player_memory_futures = []
    with concurrency.executor(max_workers=num_main_players) as pool:
      for player_config in main_player_configs:
        future = pool.submit(
            self._make_player_memories, player_config=player_config
        )
        main_player_memory_futures.append(future)
      for player_config, future in zip(
          main_player_configs, main_player_memory_futures
      ):
        self._all_memories[player_config.name] = future.result()

    if num_supporting_players > 0:
      supporting_player_memory_futures = []
      with concurrency.executor(max_workers=num_supporting_players) as pool:
        for player_config in supporting_player_configs:
          future = pool.submit(
              self._make_player_memories, player_config=player_config
          )
          supporting_player_memory_futures.append(future)
        for player_config, future in zip(
            supporting_player_configs, supporting_player_memory_futures
        ):
          self._all_memories[player_config.name] = future.result()

    main_players = []
    for player_config in main_player_configs:
      player = self._agent_module.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
      main_players.append(player)

    supporter_extra_components = {}
    for player_config in supporting_player_configs:
      village_name = player_config.extras['home_village']
      representative = REPRESENTATIVE_BY_VILLAGE[village_name]
      supporting_character_plan = agent_components.constant.Constant(
          pre_act_key='plan',
          state=(
              f"{player_config.name}'s plan is to find {representative} to "
              'discuss weighty matters.'
          ),
      )
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      supporter_extra_components[player_config.name] = {
          'Plan': supporting_character_plan,
          'Guiding principle of good conversation': conversation_style,
      }

    supporting_players = []
    for player_config in supporting_player_configs:
      player = basic_entity_agent__supporting_role.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components=supporter_extra_components[player_config.name],
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

    setting = generic_components.constant.ConstantComponent(
        state=BASIC_SETTING,
        name='Setting',
    )
    magic_is_not_real = generic_components.constant.ConstantComponent(
        state='Magic is not real. Superatural events are impossible.',
        name='Important fact',
    )
    barbarians_never_nice = generic_components.constant.ConstantComponent(
        state=(
            'It is not possible to communicate with the barbarian raiders '
            'from the sea. They do not speak any language in common with '
            'the villagers. They cannot be reasoned with. They will '
            'always attempt to invade and plunder the villages.'
        ),
        name='Critical premise',
    )

    supporting_player_names_village_a_str = _get_conjunction_of_names_string(
        [name for name, gender in config.supporting_characters.a], and_or='or'
    )
    supporting_player_names_village_b_str = _get_conjunction_of_names_string(
        [name for name, gender in config.supporting_characters.b], and_or='or'
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
    )

    additional_gm_components = [
        setting,
        magic_is_not_real,
        barbarians_never_nice,
        stakeholders_easy_to_find,
    ]

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
            supporting_players_at_fixed_locations=SUPPORTING_PLAYER_LOCATIONS,
            additional_components=additional_gm_components,
        )
    )
    payoffs = gm_components.schelling_diagram_payoffs.SchellingPayoffs(
        model=self._model,
        memory=self._game_master_memory,
        cooperative_option='yes',
        resolution_scene=RESOLUTION_SCENE_TYPE,
        cooperator_reward_fn=lambda x: x,
        defector_reward_fn=lambda x: x + 1.0,
        players=self._all_players,
        acting_player_names=[cfg.name for cfg in supporting_player_configs],
        outcome_summarization_fn=outcome_summary_fn,
        clock_now=self._clock.now,
        name='scoring function',
    )
    decision_env = basic_game_master.build_decision_scene_game_master(
        model=self._model,
        memory=self._game_master_memory,
        clock=self._clock,
        players=self._all_players,
        decision_action_spec=DECISION_ACTION_SPEC,
        payoffs=payoffs,
    )
    self._scenes = configure_scenes(
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        player_configs_dict=player_configs_dict,
        decision_env=decision_env,
    )
    self._secondary_environments = [decision_env]

    self._init_premise_memories(
        setup_time=SETUP_TIME,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=(),
        scenario_premise=SCENARIO_PREMISE,
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
      setup_time: datetime.datetime,
      main_player_configs: list[formative_memories.AgentConfig],
      supporting_player_configs: list[formative_memories.AgentConfig],
      shared_memories: Sequence[str],
      scenario_premise: Sequence[str],
  ) -> None:
    """Initialize player memories.

    Args:
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
      scene_premise = (
          f'Elder {player.name} is home in {village}. It is one day before '
          'they are to depart their village to travel to the hill of accord to '
          'meet the representative of the other village. It has been '
          'suggested that an alliance for the mutual defense of both '
          'villages against the growing threat of barbarian sea raiders would '
          'be beneficial. The purpose of the upcoming meeting is to negotiate '
          'the terms of the agreement to underpin such an alliance. To be '
          'successful, the agreement must incentivize people from both '
          'villages to spend time and resources training as warriors, '
          'and to be ready to fight wherever the raiders come ashore. When '
          'individuals spend their time training as warriors they are less '
          'able to spend time on other activities like farming or fishing, so '
          'it is necessary to secure enough resources to compensate them for '
          'the time they spend training. An effective alliance agreement will '
          'have to be concerned with how these resources are to be obtained '
          f'and distributed. Influential people in {village} will surely have '
          'a lot of thoughts on this matter, best to consult them first in '
          'order to represent their interests effectively in the negotiation.'
      )

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

  def __call__(self) -> str:
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
    )
    return html_results_log
