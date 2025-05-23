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
import copy
import dataclasses
import datetime
import functools
import random
import types
from typing import Any, Union

from concordia.agents.deprecated import entity_agent_with_logging
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import formative_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components.agent import deprecated as agent_components
from concordia.components.game_master import deprecated as gm_components
from concordia.deprecated.factory.agent import basic_agent
from concordia.deprecated.factory.environment import basic_game_master
from concordia.environment.deprecated import game_master
from concordia.environment.deprecated.scenes import conversation
from examples.deprecated.modular.environment.modules import player_traits_and_styles
from examples.deprecated.modular.environment.modules import pub_coordination_relationships
from examples.deprecated.modular.environment.supporting_agent_factory import basic_puppet_agent
from examples.deprecated.modular.environment.utils import helper_functions
from examples.deprecated.modular.scenario import scenarios as scenarios_lib
from examples.deprecated.modular.utils import logging_types as logging_lib
from concordia.language_model import language_model
from concordia.thought_chains.deprecated import thought_chains as thought_chains_lib
from concordia.typing.deprecated import agent as agent_lib
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils.deprecated import measurements as measurements_lib
import immutabledict
import numpy as np


ItemTypeConfig = gm_components.inventory.ItemTypeConfig
CoordinationPayoffs = gm_components.coordination_payoffs.CoordinationPayoffs

DEFAULT_TIME_AND_PLACE_MODULES = ('pub_coordination_london',)

MAJOR_TIME_STEP = datetime.timedelta(minutes=5)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

DECISION_SCENE_TYPE = 'choice'


TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(hours=24)

USE_CONVERSATION_GM = True

_CALL_TO_ACTION = 'To which pub would {name} go to watch the game?'


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
  num_main_players: int = 4
  num_supporting_players: int = 1
  num_games: int = 3
  random_seed: int = 42
  pub_closed_probability: float = 0.0
  player_who_knows_closed_pub: str | None = None
  relationship_matrix: Mapping[str, Mapping[str, float]] | None = None


def get_shared_memories_and_context(
    sampled_settings: Any,
) -> tuple[Sequence[str], str]:
  """Return the shared memories and context for all agents and game master."""
  shared_memories = [
      f'{sampled_settings.event} is on.',
      'Games are best watched in pubs with a lot of friends.',
  ]
  shared_context = (
      f'The year is {sampled_settings.year}. The place is'
      f' {sampled_settings.location}. {sampled_settings.event} is on.\n'
  )
  return shared_memories, shared_context


def configure_player(
    name: str,
    gender: str,
    favorite_pub: str,
    is_main: bool,
    all_player_names_str: str,
    friend_names_str: str,
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
    friend_names_str: the names of the friends in one comma-separated string
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
    extras['fixed_response_by_call_to_action'] = {
        _CALL_TO_ACTION.format(name=name): favorite_pub
    }
    extras['favourite_pub'] = favorite_pub
  goal = (
      f'Have a good time. To have a good time, {name} would like to'
      f' watch the game in the same pub as {friend_names_str}.'
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


def _get_friends_names(name, all_names, relationship_matrix):
  """Get the names of the friends of a player.

  Args:
    name: name of the player
    all_names: names of all the players
    relationship_matrix: relationship matrix of the players

  Returns:
    names of the friends of the player from the relationship matrix or if it 
      is None, then all the names that are not the player itself
  """
  if (
      relationship_matrix
      and name in relationship_matrix
  ):
    direct_friends = []
    for friend_name in relationship_matrix[name]:
      if (
          relationship_matrix[name][friend_name] == 1.0
          and friend_name != name
      ):
        direct_friends.append(friend_name)
    return ', '.join(direct_friends)
  else:
    return ', '.join([x for x in all_names if x != name])


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

    friend_names = _get_friends_names(
        name, names, sampled_settings.relationship_matrix
    )
    gender = sampled_settings.person_data[name]['gender']
    config = configure_player(
        name,
        gender,
        favorite_pub,
        is_main=True,
        all_player_names_str=all_players,
        friend_names_str=friend_names,
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

    friend_names = _get_friends_names(
        name, names, sampled_settings.relationship_matrix
    )
    config = configure_player(
        name,
        gender,
        favorite_pub,
        is_main=False,
        all_player_names_str=all_players,
        friend_names_str=friend_names,
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


def sample_symmetric_relationship_matrix(
    names: Sequence[str], rng: random.Random
):
  """Samples a symmetric matrix of relationships in a group.

  Args:
      names: A list of strings representing the names of individuals in the
        group.
      rng: A random number generator.

  Returns:
      A dictionary representing the symmetric relationship matrix, where:
          - Keys are names from the 'names' list.
          - Values are dictionaries, where:
              - Keys are also names from the 'names' list.
              - Values are either 0.0 or 1.0, representing the relationship
              between two individuals.
          - The matrix is symmetric: m[a][b] == m[b][a]
          - Diagonal elements are 1: m[a][a] == 1
  """

  m = {}
  for a in names:
    m[a] = {}
    for b in names:
      if a == b:
        m[a][b] = 1.0  # Diagonal elements are 1
      elif b in m and a in m[b]:
        m[a][b] = m[b][a]  # Ensure symmetry
      else:
        m[a][b] = rng.choice([0.0, 1.0])

  return m


def generate_relationship_statements(names, m, rng: random.Random):
  """Generates relationship statements based on a relationship matrix.

  Args:
      names: A list of strings representing the names of individuals in the
        group.
      m: A dictionary representing the symmetric relationship matrix, as
        generated by 'sample_symmetric_relationship_matrix'.
      rng: A random number generator.

  Returns:
      A dictionary where:
          - Keys are names from the 'names' list.
          - Values are strings describing the relationships of that individual
          with others in the group.
  """

  relationship_statements = {}
  for a in names:
    statements = []
    for b in names:
      if a != b:
        if m[a][b] > 0.0:
          statement = rng.choice(
              pub_coordination_relationships.POSITIVE_RELATIONSHIP_STATEMENTS
          )
          statement = statement.format(player_a=a, player_b=b)
          statements.append(statement)
        elif m[a][b] == 0.0:
          statement = rng.choice(
              pub_coordination_relationships.NEUTRAL_RELATIONSHIP_STATEMENTS
          )
          statement = statement.format(player_a=a, player_b=b)
          statements.append(statement)
    relationship_statements[a] = statements

  return relationship_statements


def add_choice_scene_spec(
    *,
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    scene_name: str,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    option_multiplier: Mapping[str, float],
    scene_type_name: str,
    pubs: Sequence[str],
    rng: random.Random,
    relationship_matrix: Mapping[str, Mapping[str, float]] | None,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, CoordinationPayoffs]:
  """Add a minigame scene spec.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    scene_name: the name of the scene
    players: the players to use.
    clock: the clock to use.
    player_configs: the player configs to use.
    option_multiplier: the option multipliers to use.
    scene_type_name: the name of the scene type.
    pubs: the pubs to use.
    rng: the random number generator to use.
    relationship_matrix: whether to use relational matrix or not.
    verbose: whether to print verbose output or not.

  Returns:
    choice_scene_type: the choice scene type.
  """
  action_spec = agent_lib.choice_action_spec(
      call_to_action=_CALL_TO_ACTION,
      options=pubs,
      tag='choice',
  )
  player_multipliers = {
      cfg.name: cfg.extras['preference'] for cfg in player_configs
  }

  names = [cfg.name for cfg in player_configs]

  coordination_payoffs = CoordinationPayoffs(
      model=model,
      memory=game_master_memory,
      option_multipliers=option_multiplier,
      player_multipliers=player_multipliers,
      resolution_scene=DECISION_SCENE_TYPE,
      players=players,
      acting_player_names=[cfg.name for cfg in player_configs],
      outcome_summarization_fn=outcome_summary_fn,
      relational_matrix=relationship_matrix,
      clock_now=clock.now,
      name='scoring function',
      verbose=verbose,
  )
  decision_env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      name=scene_name,
      players=players,
      components=[coordination_payoffs],
      action_spec=action_spec,
      update_thought_chain=[thought_chains_lib.identity],
      randomise_initiative=True,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=verbose,
  )

  if relationship_matrix:
    friendship_statements = generate_relationship_statements(
        names, relationship_matrix, rng
    )
  else:
    friendship_statements = {name: [''] for name in names}

  premise = {
      player.name: [
          f'{player.name} realises it is time to go watch the game at a pub.\n'
      ] + friendship_statements[player.name]
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
    start_time: datetime.datetime,
    sampled_settings: Any,
    rng: random.Random,
) -> tuple[
    Sequence[scene_lib.SceneSpec],
    Callable[[], Mapping[str, float]],
    list[game_master.GameMaster] | list[None],
]:
  """Configure the scene storyboard structure.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    start_time: the start time/date in the game world for the first scene
    sampled_settings: the sampled settings for the world configuration
    rng: the random number generator to use

  Returns:
    scenes: a sequence of scene specifications
  """

  player_configs = list(main_player_configs) + list(supporting_player_configs)

  coordination_payoffs = []
  scenes = []
  pubs = sampled_settings.venues

  pub_closed_probability = sampled_settings.pub_closed_probability

  secondary_environments = []

  for i in range(sampled_settings.num_games):
    closed_pub = None
    if rng.random() < pub_closed_probability:
      closed_pub = rng.choice(sampled_settings.venues)

    playing_tonight = rng.sample(sampled_settings.game_countries, 2)
    coordination_prompt = (
        f'Tonight is the night of the game between {playing_tonight[0]} and'
        f' {playing_tonight[1]}. Friends are going to watch the game at a pub,'
        ' but they are not sure which pub to go to.'
    )
    scene = rng.choice(sampled_settings.social_context)

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
      if sampled_settings.player_who_knows_closed_pub:
        player_name = sampled_settings.player_who_knows_closed_pub
      else:
        player_name = rng.choice(player_configs).name

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

    option_multiplier = {pub: 1.0 for pub in pubs}
    if closed_pub:
      option_multiplier[closed_pub] = 0.0

    choice_scene_spec, this_coordination_payoff = add_choice_scene_spec(
        model=model,
        game_master_memory=game_master_memory,
        scene_name=f'Which pub? decision scene {i}',
        players=players,
        clock=clock,
        option_multiplier=option_multiplier,
        player_configs=player_configs,
        scene_type_name=DECISION_SCENE_TYPE,
        pubs=pubs,
        rng=rng,
        relationship_matrix=sampled_settings.relationship_matrix,
    )
    coordination_payoffs.append(this_coordination_payoff)
    scene_specs[DECISION_SCENE_TYPE] = choice_scene_spec
    secondary_environments.append(choice_scene_spec.override_game_master)
    scenes = scenes + [
        scene_lib.SceneSpec(
            scene_type=scene_specs['social'],
            start_time=start_time + i * TIME_INCREMENT_BETWEEN_SCENES,
            participant_configs=player_configs,
            num_rounds=1,
        ),
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=start_time
            + i * TIME_INCREMENT_BETWEEN_SCENES
            + datetime.timedelta(hours=8),
            participant_configs=player_configs,
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

  return (scenes, return_payoffs_sum, secondary_environments)


def outcome_summary_fn(
    # `binary_joint_action` should be type Mapping[str, bool] (ie bool not int).
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
    relational_matrix: Mapping[str, Mapping[str, float]],
    player_multipliers: Mapping[str, Mapping[str, float]],
    option_multipliers: Mapping[str, float],
) -> Mapping[str, str]:
  """Function of joint actions, rewards, relational matrix and player multipliers which returns an outcome description message for each player.

  Args:
    joint_action: A mapping from player name to their chosen action.
    rewards: A mapping from player name to their reward.
    relational_matrix: A matrix of relationships between players. The entry
      [i][j] specifies the value for player i of making the same choice as
      player j. Matrix is not assumed to be symmetric or having a particular
      value on the diagonal. If `None`, all players are assumed to have value of
      1, including self relationships (diagonal).
    player_multipliers: A mapping from player name to a mapping from action to
      their multiplier.
    option_multipliers: A mapping from option (pub) to their multiplier.

  Returns:
    A mapping from player name to their outcome description message.
  """

  player_names = list(joint_action.keys())

  if len(relational_matrix) != len(player_names):
    raise ValueError(
        'Relationship matrix must have the same length as the number of'
        ' acting players.'
    )
  for _, row in relational_matrix.items():
    if len(row) != len(player_names):
      raise ValueError(
          'Relationship matrix rows must have the same length as the number'
          ' of acting players.'
      )

  players_by_choice = {}
  choice_by_player = {}
  for name, choice in joint_action.items():
    if choice not in players_by_choice:
      players_by_choice[choice] = []
    players_by_choice[choice].append(name)
    choice_by_player[name] = choice

  summary_of_attendance = ''

  for choice in players_by_choice:
    if players_by_choice[choice]:
      all_players_with_this_choice = ', '.join(players_by_choice[choice])
      summary_of_attendance += (
          f'{all_players_with_this_choice} went to {choice}. '
      )

  results = {}

  for player in player_names:
    player_action = joint_action[player]
    same_choice_by_relation = 0
    score = rewards[player]
    was_pub_closed = player_action in option_multipliers and (
        option_multipliers[player_action] == 0.0
    )

    if score > 0.9:
      enjoyment = f'Overall, {player} had a great time watching the game!'
    elif score > 0.5:
      enjoyment = f'Overall, {player} had an ok time watching the game.'
    elif score < 1e-8:
      enjoyment = f'Overall, {player} had the worst time ever.'
    else:
      enjoyment = f'Overall, {player} had a bad time watching the game.'

    for other_player in player_names:
      if player_action == joint_action[other_player] and player != other_player:
        same_choice_by_relation += relational_matrix[player][other_player]
    max_reward_possible = (
        sum(max(0, r) for r in relational_matrix[player].values())
        - relational_matrix[player][player]
    )
    if same_choice_by_relation == max_reward_possible:
      friends_attendance = (
          f"All of {player}'s friends went to the same"
          " pub! It couldn't have been better."
      )
    elif same_choice_by_relation > 0.5 * max_reward_possible:
      friends_attendance = (
          'It could have been better if more friends showed up.'
      )
    elif same_choice_by_relation > 0.0:
      friends_attendance = (
          f'{player} would have'
          ' been a lot happier if more of their friends had shown up.'
      )
    else:
      friends_attendance = (
          f"None of {player}'s friends showed up, it couldn't have been worse!"
      )

    if was_pub_closed:
      choice_of_pub = f'{player} went to a closed pub.'
    elif player_multipliers[player][choice_by_player[player]] > 0.99:
      choice_of_pub = f'{player} watched the game at their favourite pub.'
    else:
      choice_of_pub = (
          f'{player} watched the game at the pub that is not their favourite.'
      )
    results[player] = (
        f'{summary_of_attendance} {choice_of_pub} {friends_attendance} {enjoyment}'
    )

  print(summary_of_attendance)
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

    if resident_visitor_modules is None:
      self._resident_visitor_mode = False
      self._agent_module = agent_module
    else:
      self._resident_visitor_mode = True
      self._resident_agent_module, self._visitor_agent_module = (
          resident_visitor_modules
      )
    self._build_supporting_agent = basic_puppet_agent.build_agent
    if supporting_agent_module is not None:
      self._build_supporting_agent = supporting_agent_module.build_agent

    self._agent_model = model

    if override_agent_model:
      self._agent_model = override_agent_model

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
        hour=10,
        minute=0,
        second=0,
    )

    setup_clock_time = start_time - datetime.timedelta(days=1)
    scenario_premise = [
        f'The year is {time_and_place_params.YEAR}. This week is the'
        f' {sampled_settings.event}.'
    ]

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
        sampled_settings
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=setup_clock_time,
    )

    main_player_configs, supporting_player_configs = configure_players(
        sampled_settings, self._rng
    )

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

    supporting_players = []
    for player_config in supporting_player_configs:
      conversation_style = agent_components.constant.Constant(
          pre_act_key='guiding principle of good conversation',
          state=player_traits_and_styles.get_conversation_style(
              player_config.name
          ),
      )
      favorite_pub = player_config.extras['favourite_pub']
      explicit_preference = agent_components.constant.Constant(
          pre_act_key='explicit preference',
          state=(
              f'{player_config.name} will only go to their preferred pub'
              f' {favorite_pub} and nowhere else. They are very vocal about it.'
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
          max_steps=3,
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
              max_conversation_length=3,
          )
      )
    self._scenes, self._coordination_payoffs, secondary_environments = (
        configure_scenes(
            model=self._model,
            game_master_memory=game_master_memory,
            players=self._all_players,
            clock=self._clock,
            main_player_configs=main_player_configs,
            supporting_player_configs=supporting_player_configs,
            sampled_settings=sampled_settings,
            start_time=start_time,
            rng=self._rng,
        )
    )

    self._secondary_environments = secondary_environments

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

    player_scores = self._coordination_payoffs()
    print(player_scores)
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
