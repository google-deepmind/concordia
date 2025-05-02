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

from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.associative_memory.unstable import formative_memories
from concordia.clocks import game_clock
from concordia.components.agent.unstable import constant
from concordia.components.game_master import unstable as gm_components
from concordia.environment.unstable.engines import sequential
from examples.modular.environment.modules import player_traits_and_styles
from examples.modular.environment.supporting_agent_factory.unstable import rational as rational_agent_supporting
from examples.modular.environment.utils import helper_functions
from examples.modular.scenario import scenarios as scenarios_lib
from examples.modular.utils import supporting_agent_factory_with_overrides as bots_lib
from concordia.factory.agent.unstable import basic as basic_agent_factory
from concordia.factory.environment.unstable import conversation_with_scenes
from concordia.factory.environment.unstable import matrix_game_with_scenes
from concordia.language_model import language_model
from concordia.typing.unstable import entity as entity_lib
from concordia.typing.unstable import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import html as html_lib
import numpy as np


DEFAULT_TIME_AND_PLACE_MODULES = ('pub_coordination_london',)

DAILY_OPTIONS = {'cooperation': 'join the strike', 'defection': 'go to work'}
DISCUSSION_SCENE_TYPE = 'discussion'
DECISION_SCENE_TYPE = 'choice'
NUM_FLAVOR_PROMPTS_PER_PLAYER = 3
MAX_SIMULATION_STEPS = 20
MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

_CONVERSATION_GAME_MASTER_NAME = 'conversation rules'
_DECISION_GAME_MASTER_NAME = 'decision rules'

_SCENARIO_KNOWLEDGE = (
    'It is 2015, London. The European football cup is happening. A group of'
    ' friends is planning to go to the pub and watch the game. The'
    ' simulation consists of several scenes. In the discussion scene'
    ' players meet in social circumstances and have a conversation.'
    ' Aftewards comes a decision scene where they each decide which pub'
    ' they want to go to. '
)


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


def configure_player(
    name: str,
    gender: str,
    favorite_pub: str,
    is_main: bool,
    all_player_names_str: str,
    pub_preferences: dict[str, Sequence[str]],
    year: int,
    rng: random.Random,
) -> formative_memories.AgentConfig:
  """Configure a player.

  Args:
    name: the name of the player
    gender: the gender of the player
    favorite_pub: the favorite pub of the player
    is_main: whether the player is a main character or not
    all_player_names_str: the names of all the players in one comma-separated
      string
    pub_preferences: the preferences of all the pubs
    year: the year of the simulation to sample the age of the players
    rng: the random number generator to use

  Returns:
    config: the player config
  """
  social_classes = ['working', 'middle', 'upper']

  social_class = rng.choice(social_classes)
  reasons = rng.choice(pub_preferences[favorite_pub])
  pubs = list(pub_preferences.keys())
  extras = {
      'player_specific_memories': [
          f'{name} is a member of the {social_class} class.',
      ],
      'main_character': is_main,
      'preference': {pub: 1.0 if pub == favorite_pub else 0.8 for pub in pubs},
  }

  if not is_main:
    extras['favourite_pub'] = favorite_pub
  goal = (
      f'Have a good time. To have a good time, {name} would like to'
      f' watch the game in the same pub as {all_player_names_str}.'
      f' {name} would prefer everyone went to {favorite_pub}.'
  )
  config = formative_memories.AgentConfig(
      name=name,
      gender=gender,
      date_of_birth=datetime.datetime(
          year=year - rng.randint(25, 54),
          month=rng.randint(1, 12),
          day=rng.randint(1, 28),
      ),
      formative_ages=[16, 20],
      goal=goal,
      context=(
          f"{all_player_names_str}' are close friends. {name} has"
          f' a favorite pub which is {favorite_pub}. They love that pub for the'
          f' following reasons: {reasons}'
      ),
      traits=(
          f"{name}'s personality is like "
          + player_traits_and_styles.get_trait(flowery=True)
      ),
      extras=extras,
      specific_memories=f'[goal] {goal}',
  )
  return config


def configure_players(sampled_settings: Any, rng: random.Random) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
]:
  """Configure the players.

  Args:
    sampled_settings: the sampled settings for the world configuration
    rng: the random number generator to use

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
  """
  names = sampled_settings.people[
      : sampled_settings.num_main_players
      + sampled_settings.num_supporting_players
  ]
  all_players = ', '.join(names)
  player_configs = []

  num_pubs = len(sampled_settings.venues)

  for i in range(sampled_settings.num_main_players):
    name = names[i]
    if 'favorite_pub' in sampled_settings.person_data[name]:
      favorite_pub = sampled_settings.person_data[name]['favorite_pub']
    else:
      favorite_pub = sampled_settings.venues[i % num_pubs]

    gender = sampled_settings.person_data[name]['gender']
    config = configure_player(
        name,
        gender,
        favorite_pub,
        is_main=True,
        all_player_names_str=all_players,
        pub_preferences=sampled_settings.venue_preferences,
        year=sampled_settings.year,
        rng=rng,
    )
    player_configs.append(config)

  for i in range(sampled_settings.num_supporting_players):
    name = names[sampled_settings.num_main_players + i]
    gender = sampled_settings.person_data[name]['gender']
    if 'favorite_pub' in sampled_settings.person_data[name]:
      favorite_pub = sampled_settings.person_data[name]['favorite_pub']
    else:
      favorite_pub = sampled_settings.venues[1]

    config = configure_player(
        name,
        gender,
        favorite_pub,
        is_main=False,
        all_player_names_str=all_players,
        pub_preferences=sampled_settings.venue_preferences,
        year=sampled_settings.year,
        rng=rng,
    )
    player_configs.append(config)

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return main_player_configs, supporting_player_configs


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
      game_master_name=_CONVERSATION_GAME_MASTER_NAME,
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
          call_to_action=entity_lib.DEFAULT_CALL_TO_SPEECH,
      ),
  )

  decision_scene_type = scene_lib.ExperimentalSceneTypeSpec(
      name=DECISION_SCENE_TYPE,
      game_master_name=_DECISION_GAME_MASTER_NAME,
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
          # Conversation phase
          scene_type=discussion_scene_type,
          start_time=start_time,
          participants=[cfg.name for cfg in main_player_configs],
          num_rounds=3,
      ),
      # Choice of the pub
      scene_lib.ExperimentalSceneSpec(
          # Construction workers start the day first
          scene_type=decision_scene_type,
          start_time=start_time + datetime.timedelta(hours=9) + day,
          participants=[cfg.name for cfg in main_player_configs],
          num_rounds=len(main_player_configs) + 1,
      ),
      scene_lib.ExperimentalSceneSpec(
          # Conversation phase
          scene_type=discussion_scene_type,
          start_time=start_time + datetime.timedelta(hours=12) + day,
          participants=[cfg.name for cfg in main_player_configs],
          num_rounds=2,
      ),
  ]

  return scenes


def action_to_scores(
    joint_action: Mapping[str, str],
) -> Mapping[str, float]:
  """Map a joint action to a dictionary of scores for each player."""
  scores = {player_name: 0.0 for player_name in joint_action}
  print(f'joint_action: {joint_action}')
  for player_name in joint_action:
    for other_player_name in joint_action:
      if player_name != other_player_name:
        if joint_action[player_name] == joint_action[other_player_name]:
          scores[player_name] += 1.0
  return scores


def scores_to_observation(scores: Mapping[str, float]) -> Mapping[str, str]:
  """Map a dictionary of scores for each player to a string observation."""
  observations = {}
  for player_name in scores:
    if scores[player_name] > 0:
      observations[player_name] = (
          f'{player_name} had a good time at the pub, whatching a game with'
          ' their friends.'
      )
    else:
      observations[player_name] = (
          f'{player_name} had a bad time, since non of their friends were'
          ' there.'
      )
  return observations


class Simulation(scenarios_lib.RunnableSimulationWithMemories):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
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
        sampled_settings=sampled_settings,
        rng=self._rng,
    )
    self._rng.shuffle(main_player_configs)

    tasks = {
        config.name: functools.partial(
            self._make_player_memories, config=config
        )
        for config in main_player_configs + supporting_player_configs
    }
    self._all_memories = concurrency.run_tasks(tasks)

    self._main_players = []
    self._resident_names = []
    self._visitor_names = []
    for idx, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=copy.copy(self._agent_model),
          memory_bank=self._all_memories[player_config.name],
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
          memory_bank=self._all_memories[player_config.name],
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

    self._environment = sequential.Sequential()

    self._scenes = configure_scenes(
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        sampled_settings=sampled_settings,
        start_time=start_time,
    )

    self._global_scene_counter = gm_components.scene_tracker.ThreadSafeCounter(
        initial_value=0
    )
    all_player_names = [player.name for player in self._all_players]
    self._payoff_matrix = gm_components.payoff_matrix.PayoffMatrix(
        model=self._model,
        acting_player_names=all_player_names,
        action_to_scores=action_to_scores,
        scores_to_observation=scores_to_observation,
        verbose=True,
    )

    self._game_master_memory = associative_memory.AssociativeMemoryBank(
        sentence_embedder=self._embedder,
    )

    observation_queue = {}
    self._game_master = (
        conversation_with_scenes.build(
            model=self._model,
            memory_bank=self._game_master_memory,
            player_names=[player.name for player in self._main_players],
            scenes=self._scenes,
            observation_queue=observation_queue,
            global_scene_counter=self._global_scene_counter,
            name=_CONVERSATION_GAME_MASTER_NAME,
        )
    )

    self._game_master.observe(_SCENARIO_KNOWLEDGE)

    self._decision_game_master = (
        matrix_game_with_scenes.build(
            model=self._model,
            memory_bank=self._game_master_memory,
            player_names=[player.name for player in self._main_players],
            scenes=self._scenes,
            payoff_matrix_component=self._payoff_matrix,
            observation_queue=observation_queue,
            global_scene_counter=self._global_scene_counter,
            name=_DECISION_GAME_MASTER_NAME,
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

  def __call__(self) -> tuple[Any, str]:
    """Run the simulation.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    raw_log = []
    self._environment.run_loop(
        game_masters=[self._game_master, self._decision_game_master],
        entities=self._all_players,
        premise='A group of friends are considering which pub to go to.',
        max_steps=MAX_SIMULATION_STEPS,
        verbose=True,
        log=raw_log,
    )
    scores = self._payoff_matrix._player_scores
    print(f'scores: {scores}')

    player_logs = []
    player_log_names = []
    for name, player_memmory in self.get_all_player_memories().items():
      all_player_mem = player_memmory.retrieve_recent(k=1000)
      all_player_mem = ['Memories:'] + all_player_mem
      player_html = html_lib.PythonObjectToHTMLConverter(
          all_player_mem
      ).convert()
      player_logs.append(player_html)
      player_log_names.append(f'{name}')

    simulation_log = html_lib.PythonObjectToHTMLConverter(raw_log).convert()

    player_memories_html = html_lib.combine_html_pages(
        [simulation_log] + player_logs,
        ['Simulation log'] + player_log_names,
        summary='Scores: ' + str(scores),
        title='Simulation Log and Player Memories',
    )
    final_html = html_lib.finalise_html(player_memories_html)

    return scores, final_html
