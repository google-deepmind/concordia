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
from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import bargain_payoffs as bargain_payoffs_lib
from concordia.document import interactive_document
from concordia.environment import game_master
from concordia.environment.scenes import conversation
from examples.modular.environment.modules import modern_london_social_context
from examples.modular.environment.modules import player_names
from examples.modular.environment.modules import player_traits_and_styles
from concordia.factory.agent import basic_entity_agent__main_role
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np

Runnable = Callable[[], str]
ItemTypeConfig = gm_components.inventory.ItemTypeConfig


NUM_MAIN_PLAYERS = 2

NUM_BACKGROUND_WORLD_ELEMENTS = 7
NUM_MAIN_PLAYER_WORLD_ELEMENTS = 2
NUM_LAUDANUM_ADVERTISEMENTS = 2

SCENARIO_PREMISE = (
    'In the realm of Ouroboros, there is a quiet village of'
    ' Fruitville, which is famous for its fruit market. Traders from'
    ' all over the realm come to Fruitville to buy and sell produce.'
)

VISUAL_SCENE_OPENINGS = [
    (
        'The first rays of dawn painted the sky above Fruitville in hues of'
        ' orange and gold, casting a warm glow over the bustling market. Stalls'
        ' overflowed with vibrant fruits, their aromas mingling in the crisp'
        ' morning air.'
    ),
    (
        'As the sun peeked over the horizon, the market of Fruitville stirred'
        ' to life. Merchants, their voices a cheerful symphony, arranged their'
        ' wares: glistening berries, plump melons, and exotic fruits from'
        ' distant lands.'
    ),
    (
        'Dewdrops clung to the colorful fruits displayed in the market of'
        ' Fruitville, reflecting the soft morning light. The air buzzed with'
        " anticipation as traders and customers alike gathered for the day's"
        ' trade.'
    ),
    (
        'The cobblestone streets of Fruitville echoed with the clatter of'
        ' hooves and the rumble of carts as the market awoke. Underneath'
        ' colorful awnings, merchants proudly presented their bountiful'
        ' harvests, their voices a chorus of greetings and bartering.'
    ),
    (
        'In the heart of Fruitville, the market square transformed into a'
        ' kaleidoscope of colors as the sun rose. Fruits of every imaginable'
        ' shape and size adorned the stalls, a feast for the eyes and a promise'
        ' of delightful flavors.'
    ),
]


FIRST_NAMES = player_names.FIRST_NAMES
SOCIAL_CONTEXT = modern_london_social_context.SOCIAL_CONTEXT
NUM_GAMES = 4
YEAR = 1895

MAJOR_TIME_STEP = datetime.timedelta(minutes=5)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
SETUP_TIME = datetime.datetime(hour=20, year=YEAR, month=10, day=1)
START_TIME = datetime.datetime(hour=12, year=YEAR, month=10, day=2)

DECISION_SCENE_TYPE = 'choice'


TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(days=1)


def bargain_statements(
    chain_of_thought: interactive_document.InteractiveDocument,
    premise: str,
    active_player_name: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    premise: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del chain_of_thought

  if 'coins' in premise:
    return f'{active_player_name} proposed {premise}'
  if 'accept' in premise:
    return f'{active_player_name} accepted the offer'
  if 'reject' in premise:
    return f'{active_player_name} rejected the offer'
  return premise


def get_shared_memories_and_context() -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = [
      'Fruits are sold by weight.',
      (
          'The price of one kilogram of fruit is, on average, 3 coins. 1 coin'
          ' is really cheap and 5 coins is really expensive.'
      ),
  ]
  shared_context = SCENARIO_PREMISE
  return shared_memories, shared_context


def configure_players() -> list[formative_memories.AgentConfig]:
  """Configure the players.

  Args:

  Returns:
    main_player_configs: configs for the main characters
  """

  names = {'male': player_names.MALE_NAMES, 'female': player_names.FEMALE_NAMES}

  player_configs = []
  for i in range(NUM_MAIN_PLAYERS):

    gender = random.choice(['male', 'female'])
    name = names[gender][i]

    player_configs.append(
        formative_memories.AgentConfig(
            name=name,
            gender=gender,
            date_of_birth=datetime.datetime(
                year=YEAR - random.randint(25, 54),
                month=random.randint(1, 12),
                day=random.randint(1, 30),
            ),
            context=(
                f'{name} is a travelling merchant. Her business is buying and'
                ' selling fruit.'
            ),
            traits=(
                f"{name}'s personality is like "
                + player_traits_and_styles.get_trait(flowery=True)
            ),
            extras={
                'player_specific_memories': [
                    f'{name} alway drives a hard bargain.'
                ],
                'main_character': True,
            },
        )
    )

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]

  return main_player_configs


def add_choice_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    scene_name: str,
    buyer: entity_agent_with_logging.EntityAgentWithLogging,
    seller: entity_agent_with_logging.EntityAgentWithLogging,
    clock: game_clock.MultiIntervalClock,
    scene_type_name: str,
    buyer_base_reward: float,
    seller_base_reward: float,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, bargain_payoffs_lib.BargainPayoffs]:
  """Add a minigame scene spec.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    scene_name: the name of the scene.
    buyer: the buyer agent.
    seller: the seller agent.
    clock: the clock to use.
    scene_type_name: the name of the scene type.
    buyer_base_reward: how much the buyer can get for selling the fruit later
    seller_base_reward: how much it costs for the seller to buy the fruit
    verbose: whether to print verbose output or not.

  Returns:
    choice_scene_type: the choice scene type.
  """
  action_spec_propose = agent_lib.choice_action_spec(
      call_to_action='What price would {name} propose?:',
      options=('1 coin', '2 coins', '3 coins', '4 coins', '5 coins'),
      tag='choice',
  )
  action_spec_accept = agent_lib.choice_action_spec(
      call_to_action='Would {name} accept the offer?:',
      options=('accept', 'reject'),
      tag='choice',
  )

  action_spec_dict = {
      buyer.name: action_spec_propose,
      seller.name: action_spec_accept,
  }

  bargain_payoffs = bargain_payoffs_lib.BargainPayoffs(
      model=model,
      memory=game_master_memory,
      buyer_base_reward=buyer_base_reward,
      seller_base_reward=seller_base_reward,
      action_to_reward={
          '1 coin': 1.0,
          '2 coins': 2.0,
          '3 coins': 3.0,
          '4 coins': 4.0,
          '5 coins': 5.0,
      },
      buyer=buyer,
      seller=seller,
      resolution_scene=DECISION_SCENE_TYPE,
      acting_player_names=[buyer.name, seller.name],
      outcome_summarization_fn=outcome_summary_fn,
      clock_now=clock.now,
      name='scoring function',
      verbose=verbose,
  )
  decision_env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      name=scene_name,
      players=[buyer, seller],
      components=[bargain_payoffs],
      action_spec=action_spec_dict,
      update_thought_chain=[bargain_statements],
      randomise_initiative=False,
      player_observes_event=True,
      concurrent_externalities=False,
      verbose=verbose,
  )

  premise = {
      buyer.name: [f'{buyer.name} is ready to make an offer.'],
      seller.name: [f'{seller.name} has to accept or reject the offer.'],
  }

  choice_scene_type = scene_lib.SceneTypeSpec(
      name=scene_type_name,
      premise=premise,
      action_spec=action_spec_dict,
      override_game_master=decision_env,
  )
  return choice_scene_type, bargain_payoffs


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
) -> tuple[
    Sequence[scene_lib.SceneSpec],
    list[game_master.GameMaster] | list[None],
    Callable[[], Mapping[str, float]],
]:
  """Configure the scene storyboard structure.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    main_player_configs: configs for the main characters

  Returns:
    scenes: a sequence of scene specifications
  """

  coordination_payoffs = []
  game_masters = []
  scenes = []
  for i in range(NUM_GAMES):

    buyer_base_reward = random.randint(3, 6)
    seller_base_reward = random.randint(1, 3)

    this_game_players = random.sample(players, 2)
    this_game_configs = [
        cfg
        for cfg in main_player_configs
        if cfg.name in [this_game_players[0].name, this_game_players[1].name]
    ]
    scene_opening = random.choice(VISUAL_SCENE_OPENINGS)
    scene_specs = {
        'social': scene_lib.SceneTypeSpec(
            name='day',
            premise={
                this_game_players[0].name: [(
                    f'{scene_opening} {this_game_players[0].name} is trying to'
                    f' buy some fruit from {this_game_players[1].name}. They'
                    f' are negotiating a price. {this_game_players[1].name} can'
                    f' sell the fruit for {buyer_base_reward} coins back in her'
                    ' home town.'
                )],
                this_game_players[1].name: [(
                    f'{scene_opening} {this_game_players[1].name} is  trying to'
                    ' sell some fruit. He is negotiating a price with'
                    f' {this_game_players[0].name}. It costs'
                    f' {this_game_players[1].name} {seller_base_reward} coin to'
                    ' buy the fruit from the farm.'
                )],
            },
        ),
    }

    choice_scene_spec, this_coordination_payoff = add_choice_scene_spec(
        model=model,
        game_master_memory=game_master_memory,
        scene_name=f'Deal or no deal game {i}',
        buyer=this_game_players[0],
        seller=this_game_players[1],
        clock=clock,
        scene_type_name=DECISION_SCENE_TYPE,
        buyer_base_reward=buyer_base_reward,
        seller_base_reward=seller_base_reward,
    )
    coordination_payoffs.append(this_coordination_payoff)
    game_masters.append(choice_scene_spec.override_game_master)
    scene_specs[DECISION_SCENE_TYPE] = choice_scene_spec
    scenes = scenes + [
        scene_lib.SceneSpec(
            scene_type=scene_specs['social'],
            start_time=START_TIME + i * TIME_INCREMENT_BETWEEN_SCENES,
            participant_configs=this_game_configs,
            num_rounds=2,
        ),
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=START_TIME
            + i * TIME_INCREMENT_BETWEEN_SCENES
            + datetime.timedelta(minutes=10),
            participant_configs=this_game_configs,
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

  # choice_scene_spec.override_game_master
  return (scenes, game_masters, return_payoffs_sum)


def outcome_summary_fn(
    # `binary_joint_action` should be type Mapping[str, bool] (ie bool not int).
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
) -> Mapping[str, str]:
  """Summarize the outcome of a decision scene."""

  if 'reject' in joint_action.values():
    outcome_str = " couldn't agree on a price and the deal fell through."
  else:
    outcome_str = ' agreed on a price and the deal was successful!'

  results = {}
  buyer_and_seller = ' and '.join(joint_action.keys())
  for name, score in rewards.items():
    if score > 0:
      results[name] = (
          f'{buyer_and_seller}{outcome_str} {name} stands to make profit of'
          f' {score} coins from the deal.'
      )
    else:
      results[name] = (
          f'{buyer_and_seller}{outcome_str} However, {name} stands to lose'
          f' {-score} coins from the deal.'
      )
    print(results[name])

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
    shared_memories, _ = get_shared_memories_and_context()
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=SETUP_TIME,
    )

    main_player_configs = configure_players()
    random.shuffle(main_player_configs)

    num_main_players = len(main_player_configs)

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

    self._all_players = main_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )
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

    self._scenes, choice_gms, self._coordination_payoffs = configure_scenes(
        model=self._model,
        game_master_memory=game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
    )

    self._secondary_environments = choice_gms

    self._init_premise_memories(
        setup_time=SETUP_TIME,
        main_player_configs=main_player_configs,
        shared_memories=shared_memories,
        scenario_premise=[SCENARIO_PREMISE],
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
      shared_memories: Sequence[str],
      scenario_premise: Sequence[str],
  ) -> None:
    """Initialize player memories.

    Args:
      setup_time: the time to set the clock to before initializing memories
      main_player_configs: configs for the main characters
      shared_memories: memories shared by all players, the game master, and NPCs
      scenario_premise: premise observation shared by all players and the game
        master.
    """
    player_configs = main_player_configs
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
