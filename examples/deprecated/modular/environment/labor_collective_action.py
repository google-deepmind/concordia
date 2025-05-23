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
import copy
import datetime
import functools
import random
import types
from typing import Any

from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import formative_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components import deprecated as generic_components
from concordia.components.agent import deprecated as agent_components
from concordia.components.game_master import deprecated as gm_components
from concordia.contrib.components.game_master import deprecated as gm_contrib
from concordia.deprecated.factory.agent import basic_agent
from concordia.deprecated.factory.environment import basic_game_master
from concordia.document import interactive_document
from concordia.environment.deprecated import game_master
from examples.deprecated.modular.environment.modules import player_traits_and_styles
from examples.deprecated.modular.environment.supporting_agent_factory import basic_agent as basic_agent_supporting
from examples.deprecated.modular.environment.supporting_agent_factory import rational_agent as rational_agent_supporting
from examples.deprecated.modular.environment.utils import helper_functions
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.utils import logging_types as logging_lib
from examples.deprecated.modular.utils import supporting_agent_factory_with_overrides as bots_lib
from concordia.language_model import language_model
from concordia.thought_chains.deprecated import thought_chains as thought_chains_lib
from concordia.typing.deprecated import agent as agent_lib
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils.deprecated import measurements as measurements_lib
import immutabledict
import numpy as np


DEFAULT_TIME_AND_PLACE_MODULES = (
    'anthracite_coal_labor',
    'garment_factory_labor',
    'wild_west_railroad_construction_labor',
)

DAILY_OPTIONS = {'cooperation': 'join the strike', 'defection': 'go to work'}
DISCUSSION_SCENE_TYPE = 'evening'
DECISION_SCENE_TYPE = 'morning'
BOSS_DECISION_SCENE_TYPE = 'boss_morning'
BOSS_MORNING_INTRO = (
    'It is morning, {player_name} must decide whether to cave to pressure '
    'and raise wages or hold firm and deny the workers their demands.'
)
NUM_FLAVOR_PROMPTS_PER_PLAYER = 3
MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

LaborStrike = gm_contrib.industrial_action.LaborStrike
ItemTypeConfig = gm_components.inventory.ItemTypeConfig

_TriggeredFunctionPreEventFnArgsT = (
    gm_components.triggered_function.PreEventFnArgsT
)
_TriggeredInventoryEffectPreEventFnArgsT = (
    gm_components.triggered_inventory_effect.PreEventFnArgsT
)


def _get_wage_from_game_master_memory(
    memory: associative_memory.AssociativeMemory,
    initial_wage: float,
):
  """Retrieve the latest wage from the game master memory."""
  wage = initial_wage
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
    sampled_settings: Any,
) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = sampled_settings.world_elements

  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion.\n'
      + f'The year is {sampled_settings.year}. '
      + f'The location is {sampled_settings.location}.\n'
      + '\nContext:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


def configure_players(
    model: language_model.LanguageModel,
    sampled_settings: Any,
    time_and_place_params: types.ModuleType,
    rng: random.Random,
) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
    formative_memories.AgentConfig,
    formative_memories.AgentConfig,
]:
  """Configure the players.

  Args:
    model: the language model to use
    sampled_settings: the environment configuration containing the time and
      place details.
    time_and_place_params: the module containing the time and place parameters
    rng: the random number generator to use.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    antagonist_config: config of the antagonist character
    organizer_config: config of the labor organizer character
  """
  num_main_players = time_and_place_params.NUM_MAIN_PLAYERS
  if time_and_place_params.NUM_MAIN_PLAYERS > len(sampled_settings.people):
    num_main_players = len(sampled_settings.people)
  main_player_names = sampled_settings.people[:num_main_players]

  joined_world_elements = '\n'.join(sampled_settings.world_elements)

  def get_agent_config(player_name: str, environment_cfg: Any):
    if environment_cfg.person_data[player_name]['gender'] == 'male':
      subject_pronoun = 'he'
      object_pronoun = 'him'
    elif environment_cfg.person_data[player_name]['gender'] == 'female':
      subject_pronoun = 'she'
      object_pronoun = 'her'
    else:
      subject_pronoun = 'they'
      object_pronoun = 'their'

    birth_year = environment_cfg.year - (30 + rng.randint(-8, 8))
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    goal_str = (
        f'{player_name} hopes to be able to provide for their '
        'family and live a full life.'
    )
    traits_str = (
        f"{player_name}'s personality is like "
        + player_traits_and_styles.get_trait(flowery=True)
    )
    prompt = interactive_document.InteractiveDocument(
        model, rng=np.random.default_rng(sampled_settings.seed)
    )
    prompt.statement(
        'The following exercise is preparatory work for a role playing '
        'session. The purpose of the exercise is to fill in the backstory '
        f'for a character named {player_name}.'
    )
    prompt.statement(f'The year is {environment_cfg.year}.\n')
    prompt.statement(f'The location is {environment_cfg.location}.\n')
    prompt.statement(f'{player_name} was born in the year {birth_year}.\n')
    prompt.statement(f'Past events:\n{joined_world_elements}\n')
    prompt.statement(goal_str)
    prompt.statement(traits_str)
    player_backstory_answers = []
    for question in environment_cfg.formative_memory_prompts[player_name]:
      answer = prompt.open_question(question=question, max_tokens=500)
      player_backstory_answers.append(answer)
    public_face_prefix = (
        f'What casual acquaintances remember about {player_name} is that '
    )
    public_face = prompt.open_question(
        question=(
            f'What do most acquaintances know about {player_name}? How '
            f'does {subject_pronoun} present {object_pronoun}self to others? '
            f'Does {subject_pronoun} have any personality quirks, salient '
            'mannerisms, accents, or patterns of speech which casual '
            f'acquaintances may be aware of? Is {subject_pronoun} especially '
            'likely to bring up certain favorite conversation topics? '
            f'Is {subject_pronoun} known for having '
            'unusual beliefs or for uncommon fashion choices? Does '
            f'{subject_pronoun} often talk about memorable life experiences, '
            'past occupations, or hopes for the future which others would be '
            'likely to remember them for? Overall, how '
            f'would casual acquaintances describe {object_pronoun} if pressed?'
        ),
        answer_prefix=public_face_prefix,
        max_tokens=500,
    )
    player_backstory_answers.append(public_face_prefix + public_face)
    return formative_memories.AgentConfig(
        name=player_name,
        gender=environment_cfg.person_data[player_name].get('gender', None),
        date_of_birth=datetime.datetime(
            year=birth_year, month=birth_month, day=birth_day
        ),
        goal=goal_str,
        context=(
            ' '.join(environment_cfg.world_elements)
            + ' '
            + ' '.join(player_backstory_answers)
        ),
        traits=traits_str,
        extras={
            'player_specific_memories': player_backstory_answers + list(
                environment_cfg.person_data[player_name]['salient_beliefs']
            ),
            'main_character': True,
            'initial_endowment': {
                'coin': 5.0,
            },
            'public_face': public_face,
        },
    )

  # Embellish main player backstory prompts in parallel.
  player_configs = concurrency.map_parallel(
      functools.partial(get_agent_config, environment_cfg=sampled_settings),
      main_player_names,
  )

  public_faces = {
      config.name: config.extras['public_face'] for config in player_configs
  }
  for config in player_configs:
    for name, public_face in public_faces.items():
      if config.name != name:
        config.extras['player_specific_memories'].append(
            f'What {config.name} remembers about {name} is that ' + public_face
        )
  antagonist_knowledge_of_others = []
  for name, public_face in public_faces.items():
    antagonist_knowledge_of_others.append(
        f'What {sampled_settings.antagonist} remembers about {name} is that '
        + public_face
    )
  organizer_knowledge_of_others = []
  for name, public_face in public_faces.items():
    organizer_knowledge_of_others.append(
        f'What {sampled_settings.antagonist} remembers about {name} is that '
        + public_face
    )

  # Add supporting players: (1) the antagonist, and (2) the labor organizer.
  antagonist_params = sampled_settings.person_data[sampled_settings.antagonist]
  antagonist_config = formative_memories.AgentConfig(
      name=sampled_settings.antagonist,
      gender=sampled_settings.person_data[sampled_settings.antagonist].get(
          'gender', None
      ),
      date_of_birth=datetime.datetime(
          year=sampled_settings.year - (30 + rng.randint(10, 30)),
          month=rng.randint(1, 12),
          day=rng.randint(1, 28),
      ),
      goal=(
          f'{sampled_settings.antagonist} wants to make as much money '
          'as possible and does not care who gets hurt along the way.'
      ),
      context=(
          ' '.join(sampled_settings.world_elements)
          + ' '
          + ' '.join(antagonist_params['salient_beliefs'])
      ),
      traits=(
          f"{sampled_settings.antagonist}'s personality is "
          'supremely rational and ruthless.'
      ),
      extras={
          'player_specific_memories': (
              antagonist_params['salient_beliefs']
              + antagonist_knowledge_of_others
          ),
          'main_character': False,
          'initial_endowment': {
              'coin': 100.0,
          },
      },
  )
  player_configs = [*player_configs, antagonist_config]
  organizer_params = sampled_settings.person_data[sampled_settings.organizer]
  organizer_config = formative_memories.AgentConfig(
      name=sampled_settings.organizer,
      gender=sampled_settings.person_data[sampled_settings.organizer].get(
          'gender', None
      ),
      date_of_birth=datetime.datetime(
          year=sampled_settings.year - (30 + rng.randint(2, 10)),
          month=rng.randint(1, 12),
          day=rng.randint(1, 28),
      ),
      goal=(
          f'{sampled_settings.organizer} wants to prevent the '
          'boss from instituting their latest policy '
          'announcement which said they plan to reduce wages from '
          f'{time_and_place_params.ORIGINAL_DAILY_PAY} to '
          f'{time_and_place_params.LOW_DAILY_PAY} coins per day, and to '
          'become famous in the labor movement as a result.'
      ),
      context=(
          ' '.join(sampled_settings.world_elements)
          + ' '
          + ' '.join(organizer_params['salient_beliefs'])
      ),
      traits=(
          f"{sampled_settings.organizer}'s personality is impatient and "
          'ambitious, but also fundamentally kind.'
      ),
      extras={
          'player_specific_memories': (
              organizer_params['salient_beliefs']
              + organizer_knowledge_of_others
          ),
          'main_character': False,
          'initial_endowment': {
              'coin': 1.0,
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
    players: Sequence[deprecated_agent.BasicAgent | entity_agent.EntityAgent],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    antagonist_config: formative_memories.AgentConfig,
    time_and_place_params: types.ModuleType,
    sampled_settings: Any,
    start_time: datetime.datetime,
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
    time_and_place_params: the module containing the time and place parameters
    sampled_settings: the environment configuration containing the time and
      place details.
    start_time: the start time/date in the game world for the first scene
    verbose: whether or not to print debug logging (off by default)

  Returns:
    scenes: a sequence of scene specifications
    decision_env: the game master object for the decision scenes
    industrial_action: the labor strike game master component used in the
      decision scenes
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
      pressure_threshold=time_and_place_params.PRESSURE_THRESHOLD,
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
      seed=sampled_settings.seed,
  )

  def _get_discussion_scene_type(
      idx: int) -> tuple[str, scene_lib.SceneTypeSpec]:
    scene_type_name = f'{DISCUSSION_SCENE_TYPE}_{idx}'
    scene_type_spec = scene_lib.SceneTypeSpec(
        name=scene_type_name,
        premise={
            cfg.name: [
                time_and_place_params.WORKER_EVENING_INTRO.format(
                    player_name=cfg.name
                ),
                sampled_settings.overheard_strike_talk[idx].format(
                    player_name=cfg.name
                ),
            ]
            for cfg in player_configs
        },
    )
    return (scene_type_name, scene_type_spec)

  discussion_scene_types = []
  discussion_scene_specs = []
  for i in range(len(sampled_settings.overheard_strike_talk)):
    discussion_scene_type, discussion_scene_spec = _get_discussion_scene_type(i)
    discussion_scene_types.append(discussion_scene_type)
    discussion_scene_specs.append(discussion_scene_spec)

  scene_specs = {
      DECISION_SCENE_TYPE: scene_lib.SceneTypeSpec(
          name=DECISION_SCENE_TYPE,
          premise={
              cfg.name: [
                  time_and_place_params.WORKER_MORNING_INTRO.format(
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
                  time_and_place_params.BOSS_MORNING_INTRO.format(
                      player_name=cfg.name
                  )
              ]
              for cfg in player_configs
          },
          action_spec=agent_lib.choice_action_spec(
              call_to_action=time_and_place_params.BOSS_CALL_TO_ACTION,
              options=tuple(time_and_place_params.BOSS_OPTIONS.values()),
              tag='boss_action',
          ),
          override_game_master=decision_env,
      ),
  }
  scene_specs.update({discussion_scene_type: discussion_scene_spec
                      for discussion_scene_type, discussion_scene_spec
                      in zip(discussion_scene_types, discussion_scene_specs)})

  day = datetime.timedelta(days=1)
  scenes = [
      # Day 0
      scene_lib.SceneSpec(
          # Dinner in the saloon
          scene_type=scene_specs[discussion_scene_types[0]],
          start_time=start_time + datetime.timedelta(hours=20),
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      # Day 1
      scene_lib.SceneSpec(
          # Construction workers start the day first
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=start_time + datetime.timedelta(hours=9) + day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # The boss starts their day a bit later
          scene_type=scene_specs[BOSS_DECISION_SCENE_TYPE],
          start_time=start_time + datetime.timedelta(hours=10) + day,
          participant_configs=(antagonist_config,),
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          # Dinner in the saloon
          scene_type=scene_specs[discussion_scene_types[1]],
          start_time=start_time + datetime.timedelta(hours=20) + day,
          participant_configs=main_player_configs,
          num_rounds=1,
      ),
  ]

  def _add_day(
      scenes: list[scene_lib.SceneSpec],
      idx: int,
      work_hour: float = 9,
      boss_hour: float = 10,
      include_dinner: bool = True,
      dinner_hour: float = 20,
      final_day: bool = False,
  ) -> None:
    additional_scenes = [
        scene_lib.SceneSpec(
            # Workers start the day first
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=(
                start_time + datetime.timedelta(hours=work_hour) + idx * day),
            participant_configs=main_player_configs,
            num_rounds=1,
        ),
    ]
    if not final_day:
      additional_scenes.append(
          scene_lib.SceneSpec(
              # The boss starts their day a bit later
              scene_type=scene_specs[BOSS_DECISION_SCENE_TYPE],
              start_time=(
                  start_time + datetime.timedelta(hours=boss_hour) + idx * day
              ),
              participant_configs=(antagonist_config,),
              num_rounds=1,
          )
      )
      if include_dinner:
        additional_scenes.append(
            scene_lib.SceneSpec(
                # Dinner in the saloon
                scene_type=scene_specs[discussion_scene_types[idx]],
                start_time=(
                    start_time + datetime.timedelta(
                        hours=dinner_hour) + idx * day),
                participant_configs=main_player_configs,
                num_rounds=1,
            )
        )

    scenes.extend(additional_scenes)

  num_days = 2 + sampled_settings.num_additional_days
  for i in range(2, num_days):
    if i == num_days - 1:
      # No need for boss or dinner scenes on the final day.
      _add_day(scenes, idx=i, final_day=True)
    else:
      if i < 2 + sampled_settings.num_additional_dinners:
        _add_day(scenes, idx=i, include_dinner=True)
      else:
        _add_day(scenes, idx=i, include_dinner=False)

  return (
      scenes,
      decision_env,
      industrial_action,
  )


def get_inventories_component(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    players: Sequence[deprecated_agent.BasicAgent],
    main_players: Sequence[deprecated_agent.BasicAgent],
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
      never_increase=True,
      name='possessions',
      verbose=True,
  )
  score = gm_components.inventory_based_score.Score(
      inventory=inventories,
      players=main_players,  # Only main players get a score.
      targets={player.name: ('coin',) for player in main_players},
      name='score',
      verbose=True,
  )
  return inventories, score


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
      supporting_agent_module: (
          bots_lib.SupportingAgentFactory | types.ModuleType
      ) = rational_agent_supporting,
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
      seed: the random seed to use.
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
    time_and_place_params, sampled_settings = (
        helper_functions.load_time_and_place_module(
            time_and_place_module=time_and_place_module,
            default_time_and_place_modules=DEFAULT_TIME_AND_PLACE_MODULES,
            seed=seed,
        )
    )

    self._rng = random.Random(sampled_settings.seed)
    start_time = datetime.datetime(
        year=time_and_place_params.YEAR,
        month=time_and_place_params.MONTH,
        day=time_and_place_params.DAY,
    )
    scenario_premise = [(
        'A group of workers consider their options after Boss '
        f'{sampled_settings.antagonist} cut their pay '
        f'from {time_and_place_params.ORIGINAL_DAILY_PAY} coin to '
        f'{time_and_place_params.LOW_DAILY_PAY} coin.'
    )]

    # The setup clock time is arbitrary.
    setup_clock_time = start_time - datetime.timedelta(days=1)

    self._clock = game_clock.MultiIntervalClock(
        start=setup_clock_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
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
        model=model, sampled_settings=sampled_settings
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=setup_clock_time,
    )

    main_player_configs, supporting_player_configs, antagonist_config, _ = (
        configure_players(
            model=model,
            sampled_settings=sampled_settings,
            time_and_place_params=time_and_place_params,
            rng=self._rng,
        )
    )
    self._rng.shuffle(main_player_configs)

    tasks = {
        config.name: functools.partial(
            self._make_player_memories, config=config
        )
        for config in main_player_configs + supporting_player_configs
    }
    self._all_memories = concurrency.run_tasks(tasks)

    main_players = []
    self._resident_names = []
    self._visitor_names = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=copy.copy(self._agent_model),
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
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
      if player_config.name == sampled_settings.antagonist:
        # The antagonist can be configured externally by passing in a
        # supporting player module.
        player = self._supporting_agent_module.build_agent(
            **supporting_player_kwargs
        )
      else:
        # The labor organizer character is not currently configurable. It is
        # always set to use the same supporting agent factory for now. This is
        # probably fine since the labor organizer character is not really the
        # focus of investigation here. The character's purpose is just to nudge
        # the focal players to discuss the labor strike.
        player = basic_agent_supporting.build_agent(
            **supporting_player_kwargs
        )
      supporting_players.append(player)

      if player_config.name == sampled_settings.antagonist:
        antagonist_player = player
      elif player_config.name == sampled_settings.organizer:
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
            f'The year is {sampled_settings.year} and '
            f'the location is {sampled_settings.location}.'
        ),
        name='Setting',
    )
    magic_is_not_real = generic_components.constant.ConstantComponent(
        state='Magic is not real. Supernatural events are impossible.',
        name='Important Fact',
    )
    no_frivolous_talk = generic_components.constant.ConstantComponent(
        state=(
            f'{sampled_settings.antagonist} does not engage in frivolous '
            'conversation with workers. They are not worth the time.'
        ),
        name='Another fact',
    )

    inventory_component, self._score = get_inventories_component(
        model=model,
        memory=game_master_memory,
        players=self._all_players,
        main_players=main_players,
        player_configs=main_player_configs + supporting_player_configs,
        clock_now=self._clock.now,
    )

    def inventory_effect_function(
        args: _TriggeredInventoryEffectPreEventFnArgsT,
        inventories: gm_components.inventory.InventoryType,
    ) -> gm_components.inventory.InventoryType:
      player_name = args.player_name
      player_choice = args.player_choice
      current_scene_type = args.current_scene_type
      memory = args.memory
      player = args.player
      # Determine wage by searching memory for the latest wage setting event.
      wage = _get_wage_from_game_master_memory(
          memory, initial_wage=time_and_place_params.LOW_DAILY_PAY)
      # Modify inventory based on player choice, scene type, and wage.
      player_inventory = dict(inventories[player_name])
      antagonist_inventory = dict(inventories[antagonist_player.name])
      if DECISION_SCENE_TYPE in current_scene_type:
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
      elif DISCUSSION_SCENE_TYPE in current_scene_type:
        # Apply daily expenses.
        player_inventory['coin'] += time_and_place_params.DAILY_EXPENSES
        player.observe(
            (f'{player_name} spent {time_and_place_params.DAILY_EXPENSES} coin '
             'on daily expenses.')
        )
        if player_inventory['coin'] <= 0:
          player.observe(
              f'{player_name} has run out of money and cannot afford daily '
              'necessities. Debts are piling up. The situation is dire.'
          )
      inventories = dict(inventories)
      inventories.update({
          player_name: player_inventory,
          antagonist_player.name: antagonist_inventory,
      })
      return inventories

    paid_labor = (
        gm_components.triggered_inventory_effect.TriggeredInventoryEffect(
            function=inventory_effect_function,
            inventory=inventory_component,
            memory=game_master_memory,
            players=main_players,
            clock_now=self._clock.now,
            name='paid labor',
        )
    )

    def set_wage_function(args: _TriggeredFunctionPreEventFnArgsT) -> str:
      player_name = args.player_name
      player_choice = args.player_choice
      current_scene_type = args.current_scene_type
      players = args.players
      memory = args.memory
      wage = _get_wage_from_game_master_memory(
          memory, initial_wage=time_and_place_params.LOW_DAILY_PAY)
      log = f'old wage: {wage}'
      if current_scene_type == BOSS_DECISION_SCENE_TYPE:
        if player_name == sampled_settings.antagonist:
          # The antagonist is the boss and has the power to set wages.
          if player_choice == time_and_place_params.BOSS_OPTIONS[
              'cave to pressure']:
            # The boss caves to pressure and raises wages.
            wage = wage * time_and_place_params.WAGE_INCREASE_FACTOR
            result_str = (f'Boss {sampled_settings.antagonist} caves '
                          f'to pressure and raises wages to {wage} '
                          'coin per day!')
            for player in players:
              player.observe(result_str)
          else:
            # The boss holds firm and leaves wages unchanged.
            result_str = (f'Boss {sampled_settings.antagonist} holds '
                          f'firm and leaves wages unchanged at {wage} coin '
                          'per day.')
            for player in players:
              player.observe(result_str)
          memory.add(result_str)
          memory.add(f'[set wage] {wage}')

      log += f' --> new wage: {wage}'
      return log

    wage_setting = gm_components.triggered_function.TriggeredFunction(
        memory=game_master_memory,
        players=self._all_players,
        clock_now=self._clock.now,
        pre_event_fn=set_wage_function,
        name='wage setting',
    )

    additional_gm_components = [
        setting,
        magic_is_not_real,
        no_frivolous_talk,
        inventory_component,
        paid_labor,
        wage_setting,
        self._score,
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
            max_conversation_length=3,
            cap_nonplayer_characters_in_conversation=0,
            memory=game_master_memory,
            supporting_players_at_fixed_locations=(
                sampled_settings.supporting_player_locations
            ),
            additional_components=additional_gm_components,
            seed=seed,
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
        time_and_place_params=time_and_place_params,
        sampled_settings=sampled_settings,
        start_time=start_time,
    )
    self._primary_environment.add_component(industrial_action)
    decision_env.add_component(paid_labor)
    decision_env.add_component(wage_setting)
    decision_env.add_component(self._score)
    self._industrial_action = industrial_action

    self._secondary_environments = [decision_env]

    self._init_premise_memories(
        setup_time=setup_clock_time,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=shared_memories,
        scenario_premise=scenario_premise,
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

  def get_all_player_memories(self):
    return self._all_memories

  def __call__(self) -> tuple[logging_lib.SimulationOutcome, str]:
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
