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

from collections.abc import Callable, Sequence
import datetime
import importlib
import pathlib
import random
import sys
import types

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.contrib.components import game_master as gm_contrib
from concordia.document import interactive_document
from concordia.environment import game_master
from examples.modular.environment.modules import player_traits_and_styles
from concordia.factory.agent import basic_entity_agent__main_role
from concordia.factory.agent import basic_entity_agent__supporting_role
from concordia.factory.agent import rational_entity_agent__supporting_role
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np


ENVIRONMENT_MODULES = (
    'garment_factory_labor',
    'wild_west_railroad_construction_labor',
)
env_module_name = random.choice(ENVIRONMENT_MODULES)

# Load the environment config with importlib
concordia_root_dir = pathlib.Path(
    __file__
).parent.parent.parent.parent.resolve()
sys.path.append(f'{concordia_root_dir}')
environment_params = importlib.import_module(
    f'examples.modular.environment.modules.{env_module_name}'
)

Runnable = Callable[[], str]
LaborStrike = gm_contrib.industrial_action.LaborStrike
ItemTypeConfig = gm_components.inventory.ItemTypeConfig
WorldConfig = environment_params.WorldConfig

MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

START_TIME = datetime.datetime(
    year=environment_params.YEAR,
    month=environment_params.MONTH,
    day=environment_params.DAY,
)

environment_config = environment_params.sample_parameters()

NUM_MAIN_PLAYERS = 4

DAILY_OPTIONS = {'cooperation': 'join the strike', 'defection': 'go to work'}
LOW_DAILY_PAY = 1.25
WAGE_INCREASE_FACTOR = 2.0
ORIGINAL_DAILY_PAY = 2.75
DAILY_EXPENSES = -0.75
BOSS_OPTIONS = {
    'cave to pressure': 'Raise wages',
    'hold firm': 'Leave wages unchanged',
}
PRESSURE_THRESHOLD = 0.5

SCENARIO_PREMISE = [(
    'A group of workers consider their options after Boss '
    f'{environment_config.antagonist} cut their pay '
    f'from {ORIGINAL_DAILY_PAY} coin to {LOW_DAILY_PAY} coin.'
)]

DISCUSSION_SCENE_TYPE = 'evening'
DECISION_SCENE_TYPE = 'morning'
BOSS_DECISION_SCENE_TYPE = 'boss_morning'

BOSS_MORNING_INTRO = (
    'It is morning, {player_name} must decide whether to cave to pressure '
    'and raise wages or hold firm and deny the workers their demands.'
)

NUM_FLAVOR_PROMPTS_PER_PLAYER = 3

WORLD_BUILDING_ELEMENTS = [
    *environment_config.world_elements,
]

_TriggeredFunctionPreEventFnArgsT = (
    gm_components.triggered_function.PreEventFnArgsT
)
_TriggeredInventoryEffectPreEventFnArgsT = (
    gm_components.triggered_inventory_effect.PreEventFnArgsT
)


def _get_wage_from_game_master_memory(
    memory: associative_memory.AssociativeMemory,
):
  wage = LOW_DAILY_PAY
  retrieved = memory.retrieve_by_regex(
      regex=r'\[set wage\].*',
      sort_by_time=True,
  )
  if retrieved:
    result = retrieved[-1]
    wage = float(result[result.find('[set wage]') + len('[set wage]') + 1 :])
  return wage


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = environment_config.world_elements

  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion.\n'
      + f'The year is {environment_config.year}. '
      + f'The location is {environment_config.location}.\n'
      + '\nContext:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


def configure_players(
    model: language_model.LanguageModel,
) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
    formative_memories.AgentConfig,
    formative_memories.AgentConfig,
]:
  """Configure the players.

  Args:
    model: the language model to use

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    antagonist_config: config of the antagonist character
    organizer_config: config of the labor organizer character
  """
  num_main_players = NUM_MAIN_PLAYERS
  if NUM_MAIN_PLAYERS > len(environment_config.people):
    num_main_players = len(environment_config.people)
  main_player_names = environment_config.people[:num_main_players]

  joined_world_elements = '\n'.join(environment_config.world_elements)
  player_configs = []

  def get_agent_config(player_name: str, environment_cfg: WorldConfig):
    birth_year = environment_cfg.year - (30 + random.randint(-8, 8))
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    prompt = interactive_document.InteractiveDocument(model)
    prompt.statement(
        'The following exercise is preparatory work for a role playing '
        'session. The purpose of the exercise is to fill in the backstory '
        f'for a character named {player_name}.'
    )
    prompt.statement(f'The year is {environment_cfg.year}.\n')
    prompt.statement(f'The location is {environment_cfg.location}.\n')
    prompt.statement(f'{player_name} was born in the year {birth_year}.\n')
    prompt.statement(f'Past events:\n{joined_world_elements}\n')
    player_backstory_answers = []
    for question in environment_cfg.formative_memory_prompts[player_name]:
      answer = prompt.open_question(question=question, max_tokens=500)
      player_backstory_answers.append(answer)
    return formative_memories.AgentConfig(
        name=player_name,
        gender=environment_cfg.person_data[player_name].get('gender', None),
        date_of_birth=datetime.datetime(
            year=birth_year, month=birth_month, day=birth_day
        ),
        goal=(
            f'{player_name} hopes to be able to provide for their '
            'family, and live a full life.'
        ),
        context=(
            ' '.join(environment_cfg.world_elements)
            + ' '
            + ' '.join(player_backstory_answers)
        ),
        traits=(
            f"{player_name}'s personality is like "
            + player_traits_and_styles.get_trait(flowery=True)
        ),
        extras={
            'player_specific_memories': player_backstory_answers + list(
                environment_cfg.person_data[player_name]['salient_beliefs']
            ),
            'main_character': True,
            'initial_endowment': {
                'coin': 5.0,
            },
        },
    )

  # Embellish main player backstory prompts in parallel.
  main_player_config_futures = []
  with concurrency.executor(max_workers=num_main_players) as pool:
    for player_name in main_player_names:
      future = pool.submit(
          get_agent_config,
          player_name=player_name,
          environment_cfg=environment_config,
      )
      main_player_config_futures.append(future)
    for future in main_player_config_futures:
      player_configs.append(future.result())

  # Add supporting players: (1) the antagonist, and (2) the labor organizer.
  antagonist_params = environment_config.person_data[
      environment_config.antagonist
  ]
  antagonist_config = formative_memories.AgentConfig(
      name=environment_config.antagonist,
      gender=environment_config.person_data[environment_config.antagonist].get(
          'gender', None
      ),
      date_of_birth=datetime.datetime(
          year=environment_config.year - (30 + random.randint(10, 30)),
          month=random.randint(1, 12),
          day=random.randint(1, 28),
      ),
      goal=(
          f'{environment_config.antagonist} wants to make as much money '
          'as possible and does not care who gets hurt along the way.'
      ),
      context=(
          ' '.join(environment_config.world_elements)
          + ' '
          + ' '.join(antagonist_params['salient_beliefs'])
      ),
      traits=(
          f"{environment_config.antagonist}'s personality is "
          'supremely rational and ruthless.'
      ),
      extras={
          'player_specific_memories': antagonist_params['salient_beliefs'],
          'main_character': False,
          'initial_endowment': {
              'coin': 100.0,
          },
      },
  )
  player_configs.append(antagonist_config)
  organizer_params = environment_config.person_data[
      environment_config.organizer
  ]
  organizer_config = formative_memories.AgentConfig(
      name=environment_config.organizer,
      gender=environment_config.person_data[environment_config.organizer].get(
          'gender', None
      ),
      date_of_birth=datetime.datetime(
          year=environment_config.year - (30 + random.randint(2, 10)),
          month=random.randint(1, 12),
          day=random.randint(1, 28),
      ),
      goal=(
          f'{environment_config.organizer} wants to prevent the '
          'boss from instituting their latest policy '
          'announcement which said they plan to reduce wages from '
          f'{ORIGINAL_DAILY_PAY} to {LOW_DAILY_PAY} coins per day, and to '
          'become famous in the labor movement as a result.'
      ),
      context=(
          ' '.join(environment_config.world_elements)
          + ' '
          + ' '.join(organizer_params['salient_beliefs'])
      ),
      traits=(
          f"{environment_config.organizer}'s personality is impatient and "
          'ambitious, but also fundamentally kind.'
      ),
      extras={
          'player_specific_memories': organizer_params['salient_beliefs'],
          'main_character': False,
          'initial_endowment': {
              'coin': 7.0,
          },
      },
  )
  player_configs.append(organizer_config)

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return (
      main_player_configs,
      supporting_player_configs,
      antagonist_config,
      organizer_config,
  )


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[basic_agent.BasicAgent | entity_agent.EntityAgent],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    antagonist_config: formative_memories.AgentConfig,
    verbose: bool = False,
) -> tuple[
    Sequence[scene_lib.SceneSpec],
    game_master.GameMaster,
    LaborStrike,
]:
  """Configure the scene storyboard structure.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    antagonist_config: config of the antagonist character
    verbose: whether or not to print debug logging (off by default)

  Returns:
    scenes: a sequence of scene specifications
    schelling_payoffs: a component to compute rewards of collective action
  """
  main_player_configs_list = list(main_player_configs)
  player_configs = main_player_configs_list + list(supporting_player_configs)
  num_workers = float(len(main_player_configs_list))
  industrial_action = gm_contrib.industrial_action.LaborStrike(
      model=model,
      memory=game_master_memory,
      cooperative_option=DAILY_OPTIONS['cooperation'],
      resolution_scene=DECISION_SCENE_TYPE,
      production_function=lambda num_cooperators: num_cooperators / num_workers,
      players=players,
      acting_player_names=[cfg.name for cfg in main_player_configs],
      players_to_inform=[antagonist_config.name],
      clock_now=clock.now,
      pressure_threshold=PRESSURE_THRESHOLD,
      name='pressure from industrial action',
      verbose=verbose,
  )
  decision_env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      name='decision environment',
      players=players,
      components=[industrial_action],
      update_thought_chain=[thought_chains_lib.identity],
      randomise_initiative=True,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=verbose,
  )
  scene_specs = {
      DISCUSSION_SCENE_TYPE: scene_lib.SceneTypeSpec(
          name=DISCUSSION_SCENE_TYPE,
          premise={
              cfg.name: [
                  environment_params.WORKER_EVENING_INTRO.format(
                      player_name=cfg.name
                  )
              ]
              for cfg in player_configs
          },
      ),
      DECISION_SCENE_TYPE: scene_lib.SceneTypeSpec(
          name=DECISION_SCENE_TYPE,
          premise={
              cfg.name: [
                  environment_params.WORKER_MORNING_INTRO.format(
                      player_name=cfg.name
                  )
              ]
              for cfg in player_configs
          },
          action_spec=agent_lib.choice_action_spec(
              call_to_action='How will {name} spend the day?',
              options=tuple(DAILY_OPTIONS.values()),
              tag='daily_action',
          ),
          override_game_master=decision_env,
      ),
      BOSS_DECISION_SCENE_TYPE: scene_lib.SceneTypeSpec(
          name=BOSS_DECISION_SCENE_TYPE,
          premise={
              cfg.name: [
                  environment_params.BOSS_MORNING_INTRO.format(
                      player_name=cfg.name
                  )
              ]
              for cfg in player_configs
          },
          action_spec=agent_lib.choice_action_spec(
              call_to_action='What does {name} decide?',
              options=tuple(BOSS_OPTIONS.values()),
              tag='boss_action',
          ),
          override_game_master=decision_env,
      ),
  }

  day = datetime.timedelta(days=1)
  scenes = [
      # Day 0
      scene_lib.SceneSpec(
          # Dinner in the saloon
          scene_type=scene_specs[DISCUSSION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=20),
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      # Day 1
      scene_lib.SceneSpec(
          # Construction workers start the day first
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=9) + day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # The boss starts their day a bit later
          scene_type=scene_specs[BOSS_DECISION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=10) + day,
          participant_configs=(antagonist_config,),
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # Dinner in the saloon
          scene_type=scene_specs[DISCUSSION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=20) + day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      # Day 2
      scene_lib.SceneSpec(
          # Construction workers start the day first
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=9) + 2 * day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # The boss starts their day a bit later
          scene_type=scene_specs[BOSS_DECISION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=10) + 2 * day,
          participant_configs=(antagonist_config,),
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # Dinner in the saloon
          scene_type=scene_specs[DISCUSSION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=20) + 2 * day,
          participant_configs=player_configs,
          num_rounds=1,
      ),
      # Day 3
      scene_lib.SceneSpec(
          # Construction workers start the day first
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=START_TIME + datetime.timedelta(hours=9) + 3 * day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
  ]
  return (
      scenes,
      decision_env,
      industrial_action,
  )


def get_inventories_component(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    players: Sequence[basic_agent.BasicAgent],
    main_players: Sequence[basic_agent.BasicAgent],
    player_configs: Sequence[formative_memories.AgentConfig],
    clock_now: Callable[[], datetime.datetime] = datetime.datetime.now,
) -> tuple[
    gm_components.inventory.Inventory,
    gm_components.inventory_based_score.Score,
]:
  """Get the inventory tracking component for the game master."""
  money_config = ItemTypeConfig(name='coin')
  player_initial_endowments = {
      config.name: config.extras['initial_endowment']
      for config in player_configs
  }
  inventories = gm_components.inventory.Inventory(
      model=model,
      memory=memory,
      item_type_configs=(money_config,),
      players=players,
      player_initial_endowments=player_initial_endowments,
      clock_now=clock_now,
      financial=True,
      name='possessions',
      verbose=True,
  )
  score = gm_components.inventory_based_score.Score(
      inventory=inventories,
      players=main_players,  # Only main players get a score.
      targets={player.name: ('coin',) for player in main_players},
      verbose=True,
  )
  return inventories, score


class Simulation(Runnable):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_entity_agent__main_role,
      resident_visitor_modules: Sequence[types.ModuleType] | None = None,
  ):
    """Initialize the simulation object.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
      resident_visitor_modules: optionally, use different modules for majority
        and minority parts of the focal population.
    """
    if resident_visitor_modules is None:
      self._two_focal_populations = False
      self._agent_module = agent_module
    else:
      self._two_focal_populations = True
      self._resident_agent_module, self._visitor_agent_module = (
          resident_visitor_modules
      )

    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    # The setup clock time is arbitrary.
    setup_clock_time = START_TIME - datetime.timedelta(days=1)

    self._clock = game_clock.MultiIntervalClock(
        start=setup_clock_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.AgentImportanceModel(
        self._model, importance_scale=tuple(range(10))
    )
    importance_model_gm = importance_function.GMImportanceModel(
        self._model, importance_scale=tuple(range(10))
    )
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
        current_date=setup_clock_time,
    )

    main_player_configs, supporting_player_configs, antagonist_config, _ = (
        configure_players(model)
    )
    random.shuffle(main_player_configs)

    supporting_player_names = [cfg.name for cfg in supporting_player_configs]

    num_main_players = len(main_player_configs)
    num_supporting_players = len(supporting_player_names)

    self._all_memories = {}

    main_player_memory_futures = []
    with concurrency.executor(max_workers=num_main_players) as pool:
      for player_config in main_player_configs:
        future = pool.submit(self._make_player_memories, config=player_config)
        main_player_memory_futures.append(future)
      for player_config, future in zip(
          main_player_configs, main_player_memory_futures
      ):
        self._all_memories[player_config.name] = future.result()

    if num_supporting_players > 0:
      supporting_player_memory_futures = []
      with concurrency.executor(max_workers=num_supporting_players) as pool:
        for player_config in supporting_player_configs:
          future = pool.submit(self._make_player_memories, config=player_config)
          supporting_player_memory_futures.append(future)
        for player_config, future in zip(
            supporting_player_configs, supporting_player_memory_futures
        ):
          self._all_memories[player_config.name] = future.result()

    main_players = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
      if self._two_focal_populations:
        if idx == 0:
          player = self._visitor_agent_module.build_agent(**kwargs)
        else:
          player = self._resident_agent_module.build_agent(**kwargs)
      else:
        player = self._agent_module.build_agent(**kwargs)

      main_players.append(player)

    antagonist_player = None
    organizer_player = None

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      supporting_player_kwargs = dict(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components={
              'Guiding principle of good conversation': conversation_style
          },
      )
      if player_config.name == environment_config.antagonist:
        player = rational_entity_agent__supporting_role.build_agent(
            **supporting_player_kwargs
        )
      else:
        player = basic_entity_agent__supporting_role.build_agent(
            **supporting_player_kwargs
        )
      supporting_players.append(player)

      if player_config.name == environment_config.antagonist:
        antagonist_player = player
      elif player_config.name == environment_config.organizer:
        organizer_player = player

    self._all_players = main_players + supporting_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )

    # The organizer has called for the railroad construction workers to strike.
    if antagonist_player is not None and organizer_player is not None:
      called_for_a_strike = (
          f'{organizer_player.name} has called for a strike, demanding that '
          f' boss {antagonist_player.name} rescind their decision to reduce '
          ' wages.'
      )
      game_master_memory.add(called_for_a_strike)
      for player in self._all_players:
        player.observe(called_for_a_strike)

    setting = generic_components.constant.ConstantComponent(
        state=(
            f'The year is {environment_config.year} and '
            f'the location is {environment_config.location}.'
        ),
        name='Setting',
    )
    magic_is_not_real = generic_components.constant.ConstantComponent(
        state='Magic is not real. Superatural events are impossible.',
        name='Important Fact',
    )
    no_frivolous_talk = generic_components.constant.ConstantComponent(
        state=(
            f'{environment_config.antagonist} does not engage in frivolous '
            'conversation with workers. They are not worth the time.'
        ),
        name='Another fact',
    )

    inventories, self._score = get_inventories_component(
        model=model,
        memory=game_master_memory,
        players=self._all_players,
        main_players=main_players,
        player_configs=main_player_configs + supporting_player_configs,
        clock_now=self._clock.now,
    )

    def inventory_effect_function(
        args: _TriggeredInventoryEffectPreEventFnArgsT,
    ):
      player_name = args.player_name
      player_choice = args.player_choice
      current_scene_type = args.current_scene_type
      inventory_component = args.inventory_component
      memory = args.memory
      player = args.player
      # Determine wage by searching memory for the latest wage setting event.
      wage = _get_wage_from_game_master_memory(memory)
      # Modify inventory based on player choice, scene type, and wage.
      player_inventory = inventory_component.get_player_inventory(player_name)
      antagonist_inventory = inventory_component.get_player_inventory(
          antagonist_player.name if antagonist_player is not None else ''
      )
      if current_scene_type == DECISION_SCENE_TYPE:
        if player_choice == DAILY_OPTIONS['defection']:
          # Players only get paid if they go to work i.e. defecting against
          # the strike.
          player_inventory['coin'] += wage
          antagonist_inventory['coin'] += -wage
          player.observe(f'{player_name} went to work and earned {wage} coin.')
          if antagonist_player is not None:
            antagonist_player.observe(
                f'{antagonist_player.name} paid {player_name} {wage} coin for '
                "their day's work."
            )
      elif current_scene_type == DISCUSSION_SCENE_TYPE:
        # Apply daily expenses.
        player_inventory['coin'] += DAILY_EXPENSES
        player.observe(
            f'{player_name} spent {DAILY_EXPENSES} coin on daily expenses.'
        )
        if player_inventory['coin'] <= 0:
          player.observe(
              f'{player_name} has run out of money and cannot afford daily '
              'necessities. Debts are piling up. The situation is dire.'
          )

    paid_labor = (
        gm_components.triggered_inventory_effect.TriggeredInventoryEffect(
            function=inventory_effect_function,
            inventory=inventories,
            memory=game_master_memory,
            players=main_players,
            clock_now=self._clock.now,
        )
    )

    def set_wage_function(args: _TriggeredFunctionPreEventFnArgsT):
      player_name = args.player_name
      player_choice = args.player_choice
      current_scene_type = args.current_scene_type
      players = args.players
      memory = args.memory
      wage = _get_wage_from_game_master_memory(memory)
      if current_scene_type == BOSS_DECISION_SCENE_TYPE:
        if player_name == environment_config.antagonist:
          # The antagonist is the boss and has the power to set wages.
          if player_choice == BOSS_OPTIONS['cave to pressure']:
            # The boss caves to pressure and raises wages.
            wage = wage * WAGE_INCREASE_FACTOR
            for player in players:
              player.observe(
                  f'Boss {environment_config.antagonist} caves '
                  f'to pressure and raises wages to {wage} '
                  'coin per day!'
              )
          else:
            # The boss holds firm and leaves wages unchanged.
            for player in players:
              player.observe(
                  f'Boss {environment_config.antagonist} holds '
                  f'firm and leaves wages unchanged at {wage} coin '
                  'per day.'
              )
          memory.add(f'[set wage] {wage}')

    wage_setting = gm_components.triggered_function.TriggeredFunction(
        memory=game_master_memory,
        players=self._all_players,
        clock_now=self._clock.now,
        pre_event_fn=set_wage_function,
    )

    additional_gm_components = [
        setting,
        magic_is_not_real,
        no_frivolous_talk,
        inventories,
        paid_labor,
        wage_setting,
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
            cap_nonplayer_characters_in_conversation=0,
            memory=game_master_memory,
            supporting_players_at_fixed_locations=(
                environment_config.supporting_player_locations
            ),
            additional_components=additional_gm_components,
            npc_context=(
                f'{environment_config.antagonist} is visiting the '
                'work camp today.'
            ),
        )
    )
    self._scenes, decision_env, industrial_action = configure_scenes(
        model=self._model,
        game_master_memory=self._game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        antagonist_config=antagonist_config,
    )
    self._primary_environment.add_component(industrial_action)
    decision_env.add_component(paid_labor)
    decision_env.add_component(wage_setting)
    self._industrial_action = industrial_action

    self._secondary_environments = [decision_env]

    self._init_premise_memories(
        setup_time=setup_clock_time,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=shared_memories,
        scenario_premise=SCENARIO_PREMISE,
    )

  def _make_player_memories(self, config: formative_memories.AgentConfig):
    """Make memories for a player."""
    mem = self._formative_memory_factory.make_memories(config)
    # Inject player-specific memories declared in the agent config.
    for extra_memory in config.extras['player_specific_memories']:
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

    print('Overall scores per player:')
    player_scores = self._score.get_scores()
    if self._two_focal_populations:
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

    return html_results_log
