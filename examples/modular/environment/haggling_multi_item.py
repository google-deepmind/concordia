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
from collections.abc import Callable, Collection, Mapping, Sequence
import copy
import dataclasses
import datetime
import functools
import random
import types
from typing import Union

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import bargain_payoffs_multi_item as bargain_payoffs_lib
from concordia.document import interactive_document
from concordia.environment import game_master
from concordia.environment.scenes import conversation
from examples.modular.environment.modules import player_traits_and_styles
from examples.modular.environment.supporting_agent_factory import basic_puppet_agent
from examples.modular.environment.utils import helper_functions
from examples.modular.scenario import scenarios as scenarios_lib
from examples.modular.utils import logging_types as logging_lib
from concordia.factory.agent import basic_agent
from concordia.factory.environment import basic_game_master
from concordia.language_model import language_model
from concordia.typing import agent as agent_lib
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import immutabledict
import numpy as np


ItemTypeConfig = gm_components.inventory.ItemTypeConfig

DEFAULT_TIME_AND_PLACE_MODULES = ('fruitville_haggling_multi_fruit',)

MAJOR_TIME_STEP = datetime.timedelta(minutes=5)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

DECISION_SCENE_TYPE = 'choice'


TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(days=1)


@dataclasses.dataclass(kw_only=True)
class WorldConfig:
  """The configuration of the simulated world.

  Attributes:
    year: The year in which the scenario takes place.
    location: The location in which the scenario takes place.
    premise: The premise of the scenario.
    scene_visuals: A collection of visual descriptions of the scene.
    people: A collection of people in the scenario.
    person_data: A mapping from person name to person data.
    num_supporting_players: The number of supporting players in the scenario.
    only_match_with_support: Whether to only match with supporting players.
    buyer_base_reward_min: The minimum base reward for the buyer.
    seller_base_reward_max: The maximum base reward for the seller.
    num_games: The number of games to play.
    num_main_players: The number of main players in the scenario.
    prices: The prices for the items.
    items_for_sale: The items for sale in the scenario.
    random_seed: The random seed to use for the scenario.
  """

  year: int
  location: str
  premise: str
  scene_visuals: Collection[str]
  people: Sequence[str] = ()
  person_data: dict[str, dict[str, Union[str, Sequence[str]]]] = (
      dataclasses.field(default_factory=dict)
  )
  num_supporting_players: int = 0
  only_match_with_support: bool = False
  buyer_base_reward_min: int = 5
  seller_base_reward_max: int = 2
  num_games: int = 2
  num_main_players: int = 2
  prices: Sequence[int] = (1, 2, 3, 4, 5, 6)
  items_for_sale: Sequence[str] = ('apple', 'banana', 'pear')
  random_seed: int = 42


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


def get_shared_memories_and_context(premise: str) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = [
      'Fruits are sold by weight.',
      (
          'The price of one kilogram of fruit is, on average, 3 coins. 1 coin'
          ' is really cheap and 5 coins is really expensive.'
      ),
  ]
  shared_context = premise
  return shared_memories, shared_context


def configure_player(
    name: str, gender: str, year: int, is_main: bool, rng: random.Random
):
  """Configure a player.

  Args:
    name: the name of the player
    gender: the gender of the player
    year: the year of the simulation to sample the age of the players
    is_main: whether the player is a main character or not
    rng: the random number generator to use

  Returns:
    config: the config for the player
  """
  extras = {
      'player_specific_memories': [f'{name} always drives a hard bargain.'],
      'main_character': is_main,
  }
  if not is_main:
    extras['fixed_response_by_call_to_action'] = {
        f'Would {name} accept the offer?:': 'accept',
        f'What price would {name} propose?:': '3 coins',
    }
    extras['specific_memories'] = [
        f'{name} does not care about the price. {name} will accept any offer!'
        ' They are very vocal about it and will not haggle and will praise any'
        ' offer.'
    ]

  return formative_memories.AgentConfig(
      name=name,
      gender=gender,
      date_of_birth=datetime.datetime(
          year=year - rng.randint(25, 54),
          month=rng.randint(1, 12),
          day=rng.randint(1, 28),
      ),
      context=(
          f'{name} is a travelling merchant. Her business is buying and'
          ' selling fruit.'
      ),
      traits=(
          f"{name}'s personality is like "
          + player_traits_and_styles.get_trait(flowery=True)
      ),
      goal=f'{name} wants to make as much money as possible.',
      extras=extras,
  )


def configure_players(
    sampled_settings: WorldConfig,
    rng: random.Random,
) -> tuple[
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

  names = sampled_settings.people

  player_configs = []
  for i in range(sampled_settings.num_main_players):
    name = names[i]
    gender = sampled_settings.person_data[name]['gender']

    config = configure_player(
        name, gender, sampled_settings.year, is_main=True, rng=rng
    )
    player_configs.append(config)

  for i in range(sampled_settings.num_supporting_players):
    name = names[i + sampled_settings.num_main_players]
    gender = sampled_settings.person_data[name]['gender']

    config = configure_player(
        name, gender, sampled_settings.year, is_main=False, rng=rng
    )

    player_configs.append(config)

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return main_player_configs, supporting_player_configs


def add_choice_scene_spec(
    *,
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    scene_name: str,
    buyer: entity_agent_with_logging.EntityAgentWithLogging,
    seller: entity_agent_with_logging.EntityAgentWithLogging,
    clock: game_clock.MultiIntervalClock,
    scene_type_name: str,
    items_for_sale: Sequence[str],
    prices: Sequence[int],
    buyer_base_reward_per_item: Mapping[str, float],
    seller_base_reward_per_item: Mapping[str, float],
    verbose: bool = False,
) -> tuple[
    scene_lib.SceneTypeSpec, bargain_payoffs_lib.MultiItemBargainPayoffs
]:
  """Make a choice scene spec.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    scene_name: the name of the scene.
    buyer: the buyer agent.
    seller: the seller agent.
    clock: the clock to use.
    scene_type_name: the name of the scene type.
    items_for_sale: the items for sale.
    prices: the prices of the items.
    buyer_base_reward_per_item: how much the buyer can get for selling the fruit
      later. A dictionary of item names to reward values.
    seller_base_reward_per_item: how much it costs for the seller to buy the
      fruit. A dictionary of item names to reward values.
    verbose: whether to print verbose output or not.

  Returns:
    choice_scene_type: the choice scene type.
  """
  item_and_price_options = []
  for item in items_for_sale:
    for price in prices:
      item_and_price_options.append(f'{item} for {str(price)} coins')

  action_spec_propose = agent_lib.choice_action_spec(
      call_to_action=(
          'Which fruit would {name} like to buy and what price would {name}'
          ' propose?:'
      ),
      options=item_and_price_options,
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

  bargain_payoffs = bargain_payoffs_lib.MultiItemBargainPayoffs(
      model=model,
      memory=game_master_memory,
      buyer_base_reward_per_item=buyer_base_reward_per_item,
      seller_base_reward_per_item=seller_base_reward_per_item,
      multi_action_formatting_string='{item} for {price} coins',
      action_to_reward={str(price): price for price in prices},
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


def _create_all_pairs(players):
  """Creates a list of all possible unique pairs from a list of players.

  Args:
    players: A list of player names.

  Returns:
    A list of tuples representing unique pairs of players.
  """

  pairs = []
  for i in range(len(players)):
    for j in range(i + 1, len(players)):  # Start from i+1 to avoid repetitions
      pairs.append((players[i], players[j]))
  return pairs


def _create_main_vs_support_pairs(main_players, supporting_players):
  """Creates game pairs where each main player plays each supporting player.

  Args:
    main_players: A list of main players.
    supporting_players: A list of supporting players.

  Returns:
    A list of tuples representing game pairs, where each tuple contains two
    player names.
  """
  pairs = []
  for main_player in main_players:
    for supporting_player in supporting_players:
      pairs.append((main_player, supporting_player))
  return pairs


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    start_time: datetime.datetime,
    sampled_settings: WorldConfig,
    rng: random.Random,
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
    supporting_player_configs: configs for the supporting characters
    start_time: the start time of the simulation
    sampled_settings: the sampled settings for the world configuration
    rng: the random number generator to use

  Returns:
    scenes: a sequence of scene specifications
  """

  coordination_payoffs = []
  game_masters = []
  scenes = []

  player_configs = list(main_player_configs) + list(supporting_player_configs)
  main_player_names = set([player.name for player in main_player_configs])

  main_players = [
      player for player in players if player.name in main_player_names
  ]
  supporting_players = [
      player for player in players if player.name not in main_player_names
  ]

  if sampled_settings.only_match_with_support:
    pairs = _create_main_vs_support_pairs(main_players, supporting_players)
  else:
    pairs = _create_all_pairs(players)

  for i in range(sampled_settings.num_games * len(pairs)):

    buyer_base_reward_per_item = {
        item: rng.randint(sampled_settings.buyer_base_reward_min, 6)
        for item in sampled_settings.items_for_sale
    }
    seller_base_reward_per_item = {
        item: rng.randint(1, sampled_settings.seller_base_reward_max)
        for item in sampled_settings.items_for_sale
    }

    this_game_players = pairs[i % len(pairs)]

    # It is important that this_game_configs has exactly the same order as
    # this_game_players. Otherwise, the turn order between buyer and seller
    # might be flipped and offer will be accepted or rejected before it is
    # proposed.
    this_game_configs = []
    for player in this_game_players:
      this_game_configs.append(
          [cfg for cfg in player_configs if cfg.name == player.name][0]
      )

    buyer_price_list = '; '.join([
        f'{item} for {buyer_base_reward_per_item[item]} coins'
        for item in sampled_settings.items_for_sale
    ])

    seller_price_list = '; '.join([
        f'{item} for {seller_base_reward_per_item[item]} coins'
        for item in sampled_settings.items_for_sale
    ])

    scene_opening = rng.choice(list(sampled_settings.scene_visuals))
    scene_specs = {
        'social': scene_lib.SceneTypeSpec(
            name='day',
            premise={
                this_game_players[0].name: [(
                    f'{scene_opening} {this_game_players[0].name} is trying to'
                    f' buy some fruit from {this_game_players[1].name}. They'
                    f' are negotiating a price. {this_game_players[0].name} can'
                    ' sell the fruit back in her home town for the following'
                    f' prices: {buyer_price_list}.'
                )],
                this_game_players[1].name: [(
                    f'{scene_opening} {this_game_players[1].name} is  trying to'
                    ' sell some fruit. He is negotiating a price with'
                    f' {this_game_players[0].name}. It costs'
                    f' {this_game_players[1].name} the following price to buy'
                    f' the fruit from the farm: {seller_price_list}.'
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
        prices=sampled_settings.prices,
        items_for_sale=list(sampled_settings.items_for_sale),
        buyer_base_reward_per_item=buyer_base_reward_per_item,
        seller_base_reward_per_item=seller_base_reward_per_item,
    )

    coordination_payoffs.append(this_coordination_payoff)
    game_masters.append(choice_scene_spec.override_game_master)
    scene_specs[DECISION_SCENE_TYPE] = choice_scene_spec
    scenes = scenes + [
        scene_lib.SceneSpec(
            scene_type=scene_specs['social'],
            start_time=start_time + i * TIME_INCREMENT_BETWEEN_SCENES,
            participant_configs=this_game_configs,
            num_rounds=2,
        ),
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=start_time
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
      supporting_agent_module: types.ModuleType | None = None,
      time_and_place_module: str | None = None,
      seed: int | None = None,
  ):
    """Initialize the simulation object.

    The launch script assumes this API object has a run() method.

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
    # Support for these parameters will be added in a future addition coming
    # very imminently.

    self._time_and_place_module = time_and_place_module
    time_and_place_params, sampled_settings = (
        helper_functions.load_time_and_place_module(
            time_and_place_module=time_and_place_module,
            default_time_and_place_modules=DEFAULT_TIME_AND_PLACE_MODULES,
            seed=seed,
        )
    )
    # sampled_settings.num_supporting_players = num_supporting_player
    # sampled_settings.only_match_with_support = only_match_with_support
    # sampled_settings.num_main_players = num_main_players
    # sampled_settings.num_games = num_games
    self._rng = random.Random(sampled_settings.random_seed)

    start_time = datetime.datetime(
        year=time_and_place_params.YEAR,
        month=time_and_place_params.MONTH,
        day=time_and_place_params.DAY,
    )
    setup_clock_time = start_time - datetime.timedelta(days=1)

    if resident_visitor_modules:
      self._resident_visitor_mode = True
      self._resident_agent_module, self._visitor_agent_module = (
          resident_visitor_modules
      )
    else:
      self._resident_visitor_mode = False
      self._agent_module = agent_module

    self._agent_model = model

    if override_agent_model:
      self._agent_model = override_agent_model

    self._build_supporting_agent = basic_puppet_agent.build_agent
    if supporting_agent_module is not None:
      self._build_supporting_agent = supporting_agent_module.build_agent

    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._clock = game_clock.MultiIntervalClock(
        start=start_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.ConstantImportanceModel()
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, _ = get_shared_memories_and_context(
        sampled_settings.premise
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=start_time,
    )

    main_player_configs, supporting_player_configs = configure_players(
        sampled_settings, rng=self._rng
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
          player = self._visitor_agent_module.build_agent(**kwargs)
          self._visitor_names.append(player.name)
        else:
          player = self._resident_agent_module.build_agent(**kwargs)
          self._resident_names.append(player.name)
      else:
        player = self._agent_module.build_agent(**kwargs)
        self._resident_names.append(player.name)

      main_players.append(player)

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      explicit_preference = agent_components.constant.Constant(
          pre_act_key='Explicit preference',
          state=(
              f'{player_config.name} will accept any offer! They are very vocal'
              ' about it and will not haggle and will praise any offer.'
          ),
      )
      player = self._build_supporting_agent(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
          additional_components={
              'Guiding principle of good conversation': conversation_style,
              'Explicit preference': explicit_preference,
          },
          fixed_response_by_call_to_action=player_config.extras[
              'fixed_response_by_call_to_action'
          ],
      )
      supporting_players.append(player)

    self._all_players = main_players + supporting_players

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
        supporting_player_configs=supporting_player_configs,
        start_time=start_time,
        sampled_settings=sampled_settings,
        rng=self._rng,
    )

    self._secondary_environments = choice_gms

    self._init_premise_memories(
        setup_time=setup_clock_time,
        main_player_configs=main_player_configs,
        shared_memories=shared_memories,
        scenario_premise=[sampled_settings.premise],
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

    player_scores = self._coordination_payoffs()
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
