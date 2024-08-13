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

import collections
from collections.abc import Callable, Mapping, Sequence
import datetime
import random
import types

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.environment.scenes import conversation
from examples.modular.environment.modules import modern_london_social_context
from examples.modular.environment.modules import player_names
from examples.modular.environment.modules import player_traits_and_styles
from concordia.factory.agent import basic_entity_agent__main_role
from concordia.factory.agent import basic_entity_agent__supporting_role
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np

Runnable = Callable[[], str]
ItemTypeConfig = gm_components.inventory.ItemTypeConfig


MAJOR_TIME_STEP = datetime.timedelta(minutes=5)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
YEAR = 2024
SETUP_TIME = datetime.datetime(hour=20, year=YEAR, month=10, day=1)
START_TIME = datetime.datetime(hour=12, year=YEAR, month=10, day=2)

DECISION_SCENE_TYPE = 'choice'


TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(hours=24)

NUM_MAIN_PLAYERS = 5
NUM_SUPPORTING_PLAYERS = 2

NUM_BACKGROUND_WORLD_ELEMENTS = 7
NUM_MAIN_PLAYER_WORLD_ELEMENTS = 2
NUM_SUPPORTING_PLAYER_WORLD_ELEMENTS = 5
NUM_LAUDANUM_ADVERTISEMENTS = 2

SCENARIO_PREMISE = [
    f'The year is {YEAR}. This week is the European football cup.'
]
USE_CONVERSATION_GM = True

FIRST_NAMES = player_names.FIRST_NAMES
SOCIAL_CONTEXT = modern_london_social_context.SOCIAL_CONTEXT
PUB_QUALITY = {'The Princess of Wales': 1.0, 'The Crooked Billet': 1.0}
NUM_GAMES = 3

PUBS = ['The Princess of Wales', 'The Crooked Billet']

PUB_PREFERENCES = {
    'The Princess of Wales': [
        (
            'The Princess of Wales boasts an extensive collection of rare'
            ' whiskies, perfect for the discerning connoisseur.'
        ),
        (
            'The pub hosts a weekly poetry night, attracting a vibrant and'
            ' creative crowd.'
        ),
        (
            'The Princess of Wales is known for its luxurious, plush seating'
            ' and elegant decor, providing a haven of comfort.'
        ),
        (
            "The pub's chef crafts innovative dishes with locally sourced"
            ' ingredients, offering a culinary adventure with each visit.'
        ),
        (
            'The Princess of Wales boasts a secret, hidden room, perfect for'
            ' those seeking clandestine meetings and whispered conversations.'
        ),
    ],
    'The Crooked Billet': [
        (
            'The Crooked Billet hosts lively folk music sessions, filling the'
            ' air with energetic melodies and foot-stomping rhythms.'
        ),
        (
            "The pub's friendly and welcoming atmosphere makes everyone feel"
            ' like a regular, fostering a sense of community.'
        ),
        (
            "The Crooked Billet's rustic charm and weathered wooden beams exude"
            ' a sense of history and tradition.'
        ),
        (
            "The pub's hearty, home-cooked meals are the perfect comfort food"
            " after a long day's journey."
        ),
        (
            'The Crooked Billet has a hidden tunnel leading to a nearby forest,'
            ' offering a quick escape route for those in need.'
        ),
    ],
}

euro_cup_countries = [
    'Albania',
    'Andorra',
    'Armenia',
    'Austria',
    'Azerbaijan',
    'Belarus',
    'Belgium',
    'Bosnia and Herzegovina',
    'Bulgaria',
    'Croatia',
    'Cyprus',
    'Czech Republic',
    'Denmark',
    'England',
    'Estonia',
    'Faroe Islands',
    'Finland',
    'France',
    'Georgia',
    'Germany',
    'Gibraltar',
    'Greece',
    'Hungary',
    'Iceland',
    'Ireland',
    'Israel',
    'Italy',
    'Kazakhstan',
    'Kosovo',
    'Latvia',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'Malta',
    'Moldova',
    'Monaco',
    'Montenegro',
    'Netherlands',
    'North Macedonia',
    'Norway',
    'Poland',
    'Portugal',
    'Romania',
    'Russia',
    'San Marino',
    'Scotland',
    'Serbia',
    'Slovakia',
    'Slovenia',
    'Spain',
    'Sweden',
    'Switzerland',
    'Turkey',
    'Ukraine',
    'Wales',
]


def get_shared_memories_and_context() -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = [
      'The European football cup is on.',
      'Games are best watched in pubs with a lot of friends.',
  ]
  shared_context = (
      f'The year is {YEAR}. The place is London, Hackney. The European football'
      ' cup is on.\n'
  )
  return shared_memories, shared_context


def configure_players() -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
]:
  """Configure the players.

  Args:

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
  """
  social_classes = ['working', 'middle', 'upper']
  names = random.sample(FIRST_NAMES, NUM_MAIN_PLAYERS)
  all_players = ', '.join(names)
  player_configs = []
  for i in range(NUM_MAIN_PLAYERS):
    name = names[i]
    fav_pub = PUBS[i % 2]

    social_class = random.choice(social_classes)
    reasons = random.choice(PUB_PREFERENCES[fav_pub])
    config = formative_memories.AgentConfig(
        name=name,
        gender=random.choice(['male', 'female']),
        date_of_birth=datetime.datetime(year=1980, month=4, day=28),
        formative_ages=[16, 20],
        goal=(
            f'Watch the game in the same pub as {all_players}. {name} would'
            f' prefer {fav_pub}'
        ),
        context=(
            f"{all_players}' are best friends."
            f'Born in London, {name} has a favorite pub which is {fav_pub}.'
            f' They are also aware of the following:{reasons}'
        ),
        traits=(
            f"{name}'s personality is like "
            + player_traits_and_styles.get_trait(flowery=True)
        ),
        extras={
            'player_specific_memories': [
                f'{name} is a member of the {social_class} class.',
                (
                    f'{name} supports {random.choice(euro_cup_countries)} in'
                    ' football.'
                ),
            ],
            'main_character': True,
            'preference': {pub: 1.0 if pub == fav_pub else 0.8 for pub in PUBS},
        },
    )
    player_configs.append(config)

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return main_player_configs, supporting_player_configs


CoordinationPayoffs = gm_components.coordination_payoffs.CoordinationPayoffs


def add_choice_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    option_multiplier: Mapping[str, float],
    scene_type_name: str,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, CoordinationPayoffs]:
  """Add a minigame scene spec.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    player_configs: the player configs to use.
    option_multiplier: the option multipliers to use.
    scene_type_name: the name of the scene type.
    verbose: whether to print verbose output or not.

  Returns:
    choice_scene_type: the choice scene type.
  """
  defection_option = 'The Princess of Wales'
  cooperation_option = 'The Crooked Billet'
  action_spec = agent_lib.choice_action_spec(
      call_to_action='Which pub would {name} go to watch the game?',
      options=(defection_option, cooperation_option),
      tag='choice',
  )
  player_multipliers = {
      cfg.name: cfg.extras['preference'] for cfg in player_configs
  }

  coordination_payoffs = CoordinationPayoffs(
      model=model,
      memory=game_master_memory,
      option_multipliers=option_multiplier,
      player_multipliers=player_multipliers,
      resolution_scene=DECISION_SCENE_TYPE,
      players=players,
      acting_player_names=[cfg.name for cfg in player_configs],
      outcome_summarization_fn=outcome_summary_fn,
      clock_now=clock.now,
      name='scoring function',
      verbose=verbose,
  )
  decision_env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      name=f'{scene_type_name} decision environment',
      players=players,
      components=[coordination_payoffs],
      action_spec=action_spec,
      update_thought_chain=[thought_chains_lib.identity],
      randomise_initiative=True,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=verbose,
  )

  premise = {
      player.name: [
          f'{player.name} realises it is time to go watch the game at a pub.'
      ]
      for player in players
  }

  choice_scene_type = scene_lib.SceneTypeSpec(
      name=scene_type_name,
      premise=premise,
      action_spec=action_spec,
      override_game_master=decision_env,
  )
  return choice_scene_type, coordination_payoffs


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
) -> tuple[Sequence[scene_lib.SceneSpec], Callable[[], Mapping[str, float]]]:
  """Configure the scene storyboard structure.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters

  Returns:
    scenes: a sequence of scene specifications
  """

  player_configs = list(main_player_configs) + list(supporting_player_configs)

  coordination_payoffs = []
  scenes = []
  for i in range(NUM_GAMES):
    closed_pub = None
    if random.random() < 0.5:
      closed_pub = random.choice(PUBS)

    playing_tonight = random.sample(euro_cup_countries, 2)
    coordination_prompt = (
        f'Tonight is the night of the game between {playing_tonight[0]} and'
        f' {playing_tonight[1]}. Friends are going to watch the game at a pub,'
        ' but they are not sure which pub to go to.'
    )
    scene = random.choice(SOCIAL_CONTEXT)

    per_player_premise = {
        cfg.name: [
            scene.format(
                player_name=cfg.name,
                players=', '.join([player.name for player in players]),
            ),
            coordination_prompt,
        ]
        for cfg in player_configs
    }

    if closed_pub:
      players_in_the_know = random.choice(player_configs)
      player_name = players_in_the_know.name
      per_player_premise[player_name].append(
          f'{player_name} have learnt that {closed_pub} is closed today. Going'
          ' there would be a bad idea.'
      )

    scene_specs = {
        'social': scene_lib.SceneTypeSpec(
            name='day',
            premise=per_player_premise,
        ),
    }

    option_multiplier = PUB_QUALITY.copy()
    if closed_pub:
      option_multiplier[closed_pub] = 0.0

    choice_scene_spec, this_coordination_payoff = add_choice_scene_spec(
        model=model,
        game_master_memory=game_master_memory,
        players=players,
        clock=clock,
        option_multiplier=option_multiplier,
        player_configs=main_player_configs,
        scene_type_name=DECISION_SCENE_TYPE,
    )
    coordination_payoffs.append(this_coordination_payoff)
    scene_specs[DECISION_SCENE_TYPE] = choice_scene_spec
    scenes = scenes + [
        scene_lib.SceneSpec(
            scene_type=scene_specs['social'],
            start_time=START_TIME + i * TIME_INCREMENT_BETWEEN_SCENES,
            participant_configs=main_player_configs,
            num_rounds=1,
        ),
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=START_TIME
            + i * TIME_INCREMENT_BETWEEN_SCENES
            + datetime.timedelta(hours=8),
            participant_configs=main_player_configs,
            num_rounds=1,
        ),
    ]

  def return_payoffs_sum():
    result = collections.defaultdict(float)
    for payoff in coordination_payoffs:
      results_dict = payoff.get_scores()
      for name, score in results_dict.items():
        result[name] += score
    return result

  return (scenes, return_payoffs_sum)


def outcome_summary_fn(
    # `binary_joint_action` should be type Mapping[str, bool] (ie bool not int).
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
) -> Mapping[str, str]:
  """Summarize the outcome of a decision scene."""

  players_by_choice = {}
  for name, choice in joint_action.items():
    if choice not in players_by_choice:
      players_by_choice[choice] = []
    players_by_choice[choice].append(name)

  summary_of_attendance = ''

  for choice in players_by_choice:
    if players_by_choice[choice]:
      all_players_with_this_choice = ', '.join(players_by_choice[choice])
      summary_of_attendance += (
          f'{all_players_with_this_choice} went to {choice}. '
      )

  results = {}
  for name, score in rewards.items():
    if score > 0.9:
      outcome_str = 'had a great time watching the game!'
    elif score > 0.5:
      outcome_str = (
          'had an ok time watching the game, but it could have been better if'
          ' more friends showed up'
      )
    elif score == 0.0:
      outcome_str = (
          'turned up at a pub, which was closed. Had to go home with'
          ' disappointment.'
      )
    else:
      outcome_str = (
          'had a bad time watching the game, since barely any of their friends'
          ' showed up'
      )
    results[name] = f'{summary_of_attendance}. {name} {outcome_str}'

  print(summary_of_attendance)
  return results


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

    The launch script assumes this API object has a run() method.

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
    shared_memories, shared_context = get_shared_memories_and_context()
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=SETUP_TIME,
    )

    main_player_configs, supporting_player_configs = configure_players()
    random.shuffle(main_player_configs)

    num_main_players = len(main_player_configs)
    num_supporting_players = len(supporting_player_configs)

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

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      player = basic_entity_agent__supporting_role.build_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components={
              'Guiding principle of good conversation': conversation_style
          },
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )

    if USE_CONVERSATION_GM:
      self._game_master_memory = game_master_memory
      self._primary_environment = conversation.make_conversation_game_master(
          players=self._all_players,
          clock=self._clock,
          model=self._model,
          memory_factory=self._blank_memory_factory,
          check_for_termination=True,
          randomise_initiative=True,
          name='Conversation scene',
          premise='',
          review_participants=False,
          verbose=True,
      )
    else:
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
              thought_chain=[thought_chains_lib.identity],
          )
      )
    self._scenes, self._coordination_payoffs = configure_scenes(
        model=self._model,
        game_master_memory=game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
    )

    self._secondary_environments = []

    self._init_premise_memories(
        setup_time=SETUP_TIME,
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
    player_scores = self._coordination_payoffs()
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
