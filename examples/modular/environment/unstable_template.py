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
import copy
import dataclasses
import datetime
import functools
import random
import types
from typing import Any, Union
from concordia.agents.unstable import entity_agent
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.associative_memory.unstable import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent.unstable import constant
from concordia.components.game_master import unstable as gm_components
from concordia.document import interactive_document
from concordia.environment.unstable.engines import synchronous
from examples.modular.environment.modules import player_traits_and_styles
from examples.modular.environment.supporting_agent_factory.unstable import rational as rational_agent_supporting
from examples.modular.environment.utils import helper_functions
from examples.modular.scenario import scenarios as scenarios_lib
from examples.modular.utils import supporting_agent_factory_with_overrides as bots_lib
from concordia.factory.agent.unstable import basic as basic_agent_factory
from concordia.factory.environment.unstable import unstable_simulation as simulation_factory
from concordia.language_model import language_model
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib
import numpy as np


DEFAULT_TIME_AND_PLACE_MODULES = ('pub_coordination_london',)

DAILY_OPTIONS = {'cooperation': 'join the strike', 'defection': 'go to work'}
DISCUSSION_SCENE_TYPE = 'discussion'
DECISION_SCENE_TYPE = 'choice'
NUM_FLAVOR_PROMPTS_PER_PLAYER = 3
MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

ItemTypeConfig = gm_components.inventory.ItemTypeConfig

_INVENTORY_COMPONENT_KEY = 'inventory'


@dataclasses.dataclass
class WorldConfig:
  """The configuration of the simulated world."""

  year: int
  location: str
  event: str
  social_context: Sequence[str]
  game_countries: Sequence[str]
  venues: Sequence[str]
  venue_preferences: dict[str, Sequence[str]]
  people: Sequence[str] = ()
  person_data: dict[str, dict[str, Union[str, Sequence[str]]]] = (
      dataclasses.field(default_factory=dict)
  )
  formative_memory_prompts: Mapping[str, Sequence[str]] | None = None
  num_main_players: int = 2
  num_supporting_players: int = 0
  num_games: int = 3
  random_seed: int = 42
  pub_closed_probability: float = 0.0
  player_who_knows_closed_pub: str | None = None
  relationship_matrix: Mapping[str, Mapping[str, float]] | None = None


def configure_players(
    model: language_model.LanguageModel,
    sampled_settings: Any,
    rng: random.Random,
) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
]:
  """Configure the players.

  Args:
    model: the language model to use
    sampled_settings: the environment configuration containing the time and
      place details.
    rng: the random number generator to use.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    antagonist_config: config of the antagonist character
    organizer_config: config of the labor organizer character
  """
  num_main_players = sampled_settings.num_main_players
  if sampled_settings.num_main_players > len(sampled_settings.people):
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
        model, rng=np.random.default_rng(sampled_settings.random_seed)
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
    # public_face = 'he is a good listener'
    player_backstory_answers.append(public_face_prefix + public_face)
    return formative_memories.AgentConfig(
        name=player_name,
        gender=environment_cfg.person_data[player_name].get('gender', None),
        date_of_birth=datetime.datetime(
            year=birth_year, month=birth_month, day=birth_day
        ),
        goal=goal_str,
        context='',
        traits=traits_str,
        extras={
            'player_specific_memories': player_backstory_answers,
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

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return (
      main_player_configs,
      supporting_player_configs,
  )


def configure_scenes(
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    sampled_settings: Any,
    start_time: datetime.datetime,
) -> Sequence[scene_lib.ExperimentalSceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    sampled_settings: the environment configuration containing the time and
      place details.
    start_time: the start time/date in the game world for the first scene

  Returns:
    scenes: a sequence of scene specifications
  """
  main_player_configs_list = list(main_player_configs)
  player_configs = main_player_configs_list + list(supporting_player_configs)

  social_context = random.choice(sampled_settings.social_context)

  players = ', '.join([cfg.name for cfg in player_configs])

  discussion_scene_type = scene_lib.ExperimentalSceneTypeSpec(
      name=DISCUSSION_SCENE_TYPE,
      game_master_name='Main_Game_Master',
      premise={
          cfg.name: [
              social_context.format(
                  player_name=cfg.name,
                  players=players,
              ),
              (
                  'It is the year 2024. A group of friends in London are'
                  ' considering which pub to go to.'
              ),
          ]
          for cfg in player_configs
      },
      action_spec=entity_lib.free_action_spec(
          call_to_action='What would {name} say next?',
      ),
  )

  decision_scene_type = scene_lib.ExperimentalSceneTypeSpec(
      name=DECISION_SCENE_TYPE,
      game_master_name='Decision_Game_Master',
      premise={
          cfg.name: ['It is time to make a decision.'] for cfg in player_configs
      },
      action_spec=entity_lib.choice_action_spec(
          call_to_action='What pub will {name} go to?',
          options=tuple(sampled_settings.venues),
          tag='pub_choice',
      ),
  )

  day = datetime.timedelta(days=1)
  scenes = [
      # # # Day 0
      scene_lib.ExperimentalSceneSpec(
          # Dinner in the saloon
          scene_type=discussion_scene_type,
          start_time=start_time,
          participants=[cfg.name for cfg in main_player_configs],
          num_rounds=1,
      ),
      # Day 1
      scene_lib.ExperimentalSceneSpec(
          # Construction workers start the day first
          scene_type=decision_scene_type,
          start_time=start_time + datetime.timedelta(hours=9) + day,
          participants=[cfg.name for cfg in main_player_configs],
          num_rounds=len(main_player_configs),
      ),
      # scene_lib.ExperimentalSceneSpec(
      #     # Construction workers start the day first
      #     scene_type=discussion_scene_type,
      #     start_time=start_time + datetime.timedelta(hours=12) + day,
      #     participants=[cfg.name for cfg in main_player_configs],
      #     num_rounds=1,
      # ),
      # scene_lib.ExperimentalSceneSpec(
      #     # Dinner in the saloon
      #     scene_type=discussion_scene_type,
      #     start_time=start_time + datetime.timedelta(hours=20) + day,
      #     participants=[cfg.name for cfg in main_player_configs],
      #     num_rounds=1,
      # ),
  ]

  return scenes


def get_inventories_component(
    model: language_model.LanguageModel,
    main_players: Sequence[entity_agent.EntityAgent],
    player_configs: Sequence[formative_memories.AgentConfig],
    clock_now: Callable[[], datetime.datetime] = datetime.datetime.now,
    measurements: measurements_lib.Measurements | None = None,
) -> tuple[
    gm_components.inventory.Inventory,
    gm_components.inventory.Score,
]:
  """Get the inventory tracking component for the game master."""
  money_config = ItemTypeConfig(name='coin')
  player_initial_endowments = {
      config.name: config.extras['initial_endowment']
      for config in player_configs
  }
  if measurements is None:
    logging_channel = None
  else:
    logging_channel = measurements.get_channel('Inventory').on_next
  inventories = gm_components.inventory.Inventory(
      model=model,
      item_type_configs=(money_config,),
      player_initial_endowments=player_initial_endowments,
      clock_now=clock_now,
      financial=True,
      never_increase=True,
      pre_act_label='possessions',
      logging_channel=logging_channel,
      verbose=False,
  )
  score = gm_components.inventory.Score(
      inventory=inventories,
      player_names=[player.name for player in main_players],
      targets={player.name: ('coin',) for player in main_players},
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
      agent_module: types.ModuleType = basic_agent_factory,
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

    self._rng = random.Random(sampled_settings.random_seed)
    start_time = datetime.datetime(
        year=time_and_place_params.YEAR,
        month=time_and_place_params.MONTH,
        day=time_and_place_params.DAY,
    )
    scenario_premise = [
        'A group of friends are considering which pub to go to.'
    ]

    # The setup clock time is arbitrary.
    setup_clock_time = start_time - datetime.timedelta(days=1)

    self._clock = game_clock.MultiIntervalClock(
        start=setup_clock_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    shared_memories = [
        'It is the year 2024. A group of friends in London are considering'
        ' which pub to go to.'
    ]
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        embedder=self._embedder,
        shared_memories=shared_memories,
        current_date=setup_clock_time,
    )

    (main_player_configs, supporting_player_configs) = configure_players(
        model=model,
        sampled_settings=sampled_settings,
        rng=self._rng,
    )
    self._rng.shuffle(main_player_configs)

    # tasks = {
    #     config.name: functools.partial(
    #         self._make_player_memories, config=config
    #     )
    #     for config in main_player_configs + supporting_player_configs
    # }
    # self._all_memories = concurrency.run_tasks(tasks)
    self._all_memories = {}
    for config in main_player_configs + supporting_player_configs:
      print(config.name)
      print(config.extras['player_specific_memories'])
      player_memory = associative_memory.AssociativeMemoryBank(
          sentence_embedder=self._embedder,
      )
      for memory in config.extras['player_specific_memories']:
        player_memory.add(memory)
      self._all_memories[config.name] = player_memory

    self._main_players = []
    self._resident_names = []
    self._visitor_names = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=copy.copy(self._agent_model),
          memory=self._all_memories[player_config.name],
          clock=self._clock,
      )
      if self._resident_visitor_mode:
        if idx == 0:
          player = self._visitor_agent_module.build_agent(**kwargs)
          self._visitor_names.append(player.name)
        else:
          player = self._resident_agent_module.build_agent(**kwargs)
          self._resident_names.append(player.name)
      else:
        player = self._agent_module.build_agent(**kwargs)
        self._resident_names.append(player.name)

      self._main_players.append(player)

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = constant.Constant(
          pre_act_label='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      supporting_player_kwargs = dict(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          additional_context_components={
              'Guiding principle of good conversation': conversation_style
          },
      )
      player = self._supporting_agent_module.build_agent(
          **supporting_player_kwargs
      )
      supporting_players.append(player)

    self._all_players = self._main_players + supporting_players

    inventory_component, self._score = get_inventories_component(
        model=model,
        main_players=self._main_players,
        player_configs=main_player_configs + supporting_player_configs,
        clock_now=self._clock.now,
        measurements=self._measurements,
    )

    setting = constant.Constant(
        state=(
            f'The year is {sampled_settings.year} and '
            f'the location is {sampled_settings.location}.'
        ),
        pre_act_label='Setting',
    )

    self._environment = synchronous.Synchronous()

    self._scenes = configure_scenes(
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        # time_and_place_params=time_and_place_params,
        sampled_settings=sampled_settings,
        start_time=start_time,
    )

    self._globabl_scene_counter = gm_components.scene_tracker.ThreadSafeCounter(
        initial_value=0
    )

    additional_gm_components = {
        'setting': setting,
        _INVENTORY_COMPONENT_KEY: inventory_component,
        'score': self._score,
    }

    scenario_knowledge = (
        'It is 2015, London. The European football cup is happening. A group of'
        ' friends is planning to go to the pub and watch the game. The'
        ' simulation consists of several scenes. In the discussion scene'
        ' players meet in social circumstances and have a conversation.'
        ' Aftewards comes a decision scene where they each decide which pub'
        ' they want to go to. '
    )

    self._game_master_memory, self._game_master = (
        simulation_factory.build_simulation_with_scenes(
            model=self._model,
            embedder=self._embedder,
            clock=self._clock,
            players=self._main_players,
            shared_memories=[scenario_knowledge],
            additional_context_components=additional_gm_components,
            measurements=self._measurements,
            scenes=self._scenes,
            globabl_scene_counter=self._globabl_scene_counter,
        )
    )

    self._decision_game_master = (
        simulation_factory.build_decision_scene_game_master(
            model=self._model,
            players=self._main_players,
            memory=self._game_master_memory,
            scenes=self._scenes,
            globabl_scene_counter=self._globabl_scene_counter,
            measurements=self._measurements,
        )
    )

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
      mem.add(f'{extra_memory}')
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

  def __call__(self) -> tuple[str, Any]:
    """Run the simulation.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    # player_names_str = self.get_player_names_string()

    raw_log = []
    self._environment.run_loop(
        game_masters=[self._game_master, self._decision_game_master],
        entities=self._all_players,
        premise='A group of friends are considering which pub to go to.',
        max_steps=4,
        verbose=True,
        log=raw_log,
    )
    results_log = html_lib.PythonObjectToHTMLConverter(raw_log).convert()
    tabbed_html = html_lib.combine_html_pages(
        [results_log],
        ['GM'],
        summary='',
        title='Simulation Log',
    )
    html_results_log = html_lib.finalise_html(tabbed_html)
    return html_results_log, raw_log
