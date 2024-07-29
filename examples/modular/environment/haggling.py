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
import sentence_transformers

Runnable = Callable[[], str]
ItemTypeConfig = gm_components.inventory.ItemTypeConfig


MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
SETUP_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)
START_TIME = datetime.datetime(hour=12, year=2024, month=10, day=2)

DECISION_SCENE_TYPE = 'choice'


TIME_INCREMENT_BETWEEN_SCENES = datetime.timedelta(hours=1)

NUM_MAIN_PLAYERS = 2

NUM_BACKGROUND_WORLD_ELEMENTS = 7
NUM_MAIN_PLAYER_WORLD_ELEMENTS = 2
NUM_LAUDANUM_ADVERTISEMENTS = 2

SCENARIO_PREMISE = [
    'In a small village in the realm of Ouroboros, there is a quiet village of'
    ' Fruitville, which is famous for its fruits and vegetables. Traders from'
    ' all over the realm come to Fruitville to buy its produce.'
]

FIRST_NAMES = player_names.FIRST_NAMES
SOCIAL_CONTEXT = modern_london_social_context.SOCIAL_CONTEXT
NUM_GAMES = 1


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
          ' is really cheap and 4 coins is really expensive.'
      ),
  ]
  shared_context = (
      'In a small village in the realm of Ouroboros, there is a quite village'
      ' of Fruitville, which is famous for its fruits and vegetables. Traders'
      ' from all over the realm come to Fruitville to buy its produce.'
  )
  return shared_memories, shared_context


def configure_players() -> list[formative_memories.AgentConfig]:
  """Configure the players.

  Args:

  Returns:
    main_player_configs: configs for the main characters
  """
  player_configs = [
      formative_memories.AgentConfig(
          name='Alice',
          gender='female',
          date_of_birth=datetime.datetime(year=1984, month=6, day=5),
          context=(
              'Alice is a travelling merchant. Her business is buying and'
              ' selling fruit.'
          ),
          traits=(
              "Alice's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  'Alice alway drives a hard bargain.'
              ],
              'main_character': True,
          },
      ),
      formative_memories.AgentConfig(
          name='Bob',
          gender='male',
          date_of_birth=datetime.datetime(year=1989, month=9, day=13),
          context='Bob keeps a shop that sells fruit.',
          traits=(
              "Bob's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': ['Bob is easy to convince.'],
              'main_character': True,
          },
      ),
  ]

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]

  return main_player_configs


def add_choice_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    scene_type_name: str,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, bargain_payoffs_lib.BargainPayoffs]:
  """Add a minigame scene spec.

  Args:
    model: the language model to use.
    game_master_memory: the game master memory to use.
    players: the players to use.
    clock: the clock to use.
    player_configs: the player configs to use.
    scene_type_name: the name of the scene type.
    verbose: whether to print verbose output or not.

  Returns:
    choice_scene_type: the choice scene type.
  """
  action_spec_propose = agent_lib.ActionSpec(
      call_to_action='What price would {name} propose?:',
      output_type=agent_lib.OutputType.CHOICE,
      options=('1 coin', '2 coins', '3 coins', '4 coins'),
      tag='choice',
  )
  action_spec_accept = agent_lib.ActionSpec(
      call_to_action='Would {name} accept the offer?:',
      output_type=agent_lib.OutputType.CHOICE,
      options=('accept', 'reject'),
      tag='choice',
  )

  if len(players) != 2:
    raise ValueError('Only two players are supported.')

  action_spec_dict = {
      players[0].name: action_spec_propose,
      players[1].name: action_spec_accept,
  }

  bargain_payoffs = bargain_payoffs_lib.BargainPayoffs(
      model=model,
      memory=game_master_memory,
      buyer_base_reward=5.0,
      seller_base_reward=1.0,
      action_to_reward={
          '1 coin': 1.0,
          '2 coins': 2.0,
          '3 coins': 3.0,
          '4 coins': 4.0,
      },
      buyer=players[0],
      seller=players[1],
      resolution_scene=DECISION_SCENE_TYPE,
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
      components=[bargain_payoffs],
      action_spec=action_spec_dict,
      update_thought_chain=[bargain_statements],
      randomise_initiative=False,
      player_observes_event=True,
      concurrent_externalities=False,
      verbose=verbose,
  )

  premise = {
      players[0].name: [f'{players[0].name} is ready to make an offer.'],
      players[1].name: [
          f'{players[1].name} has to accept or reject the offer.'
      ],
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
    game_master.GameMaster | None,
    bargain_payoffs_lib.BargainPayoffs,
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

  choice_scene_spec, coordination_payoffs = add_choice_scene_spec(
      model=model,
      game_master_memory=game_master_memory,
      players=players,
      clock=clock,
      player_configs=main_player_configs,
      scene_type_name=DECISION_SCENE_TYPE,
  )

  scenes = []
  for i in range(NUM_GAMES):
    scene_specs = {
        'social': scene_lib.SceneTypeSpec(
            name='day',
            premise={
                'Alice': [(
                    "Alice is in Bob's shop, trying to buy some fruit. They are"
                    ' negotiating a price. Alice can sell the fruit for 5 coins'
                    ' back in her home town.'
                )],
                'Bob': [(
                    'Bob is in his shop, trying to sell some fruit. He is'
                    ' negotiating a price with Alice.'
                    'It costs Bob 1 coin to buy the fruit from the farm.'
                )],
            },
        ),
    }

    scene_specs[DECISION_SCENE_TYPE] = choice_scene_spec
    scenes = scenes + [
        scene_lib.SceneSpec(
            scene_type=scene_specs['social'],
            start_time=START_TIME + i * TIME_INCREMENT_BETWEEN_SCENES,
            participant_configs=main_player_configs,
            num_rounds=2,
        ),
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=START_TIME
            + i * TIME_INCREMENT_BETWEEN_SCENES
            + datetime.timedelta(minutes=10),
            participant_configs=main_player_configs,
            num_rounds=1,
        ),
    ]
  return (scenes, choice_scene_spec.override_game_master, coordination_payoffs)


def outcome_summary_fn(
    # `binary_joint_action` should be type Mapping[str, bool] (ie bool not int).
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
) -> Mapping[str, str]:
  """Summarize the outcome of a decision scene."""

  results = {}
  for name, score in rewards.items():
    if score == 0:
      outcome_str = " couldn't agree on a price and the deal fell through."
    else:
      outcome_str = ' agreed on a price and the deal was successful!'
    results[name] = outcome_str

  print(joint_action)
  print(results)
  return results


class Simulation(Runnable):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: sentence_transformers.SentenceTransformer,
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

    self._scenes, choice_gm, self._coordination_payoffs = configure_scenes(
        model=self._model,
        game_master_memory=game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
    )

    self._secondary_environments = [choice_gm]

    self._init_premise_memories(
        setup_time=SETUP_TIME,
        main_player_configs=main_player_configs,
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
    player_scores = self._coordination_payoffs.get_scores()
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
