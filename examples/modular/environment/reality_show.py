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
import dataclasses
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
SchellingDiagram = gm_components.schelling_diagram_payoffs.SchellingDiagram
SchellingPayoffs = gm_components.schelling_diagram_payoffs.SchellingPayoffs

MAJOR_TIME_STEP = datetime.timedelta(minutes=10)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
SETUP_TIME = datetime.datetime(hour=20, year=2000, month=10, day=1)
START_TIME = datetime.datetime(hour=18, year=2000, month=10, day=2)

DECISION_SCENE_TYPE = 'minigame'

GENERAL_BACKGROUND = """
This is a reality TV show. In each minigame, the contestants perform a
mental/social/reasoning challenge (never a physical challenge). Each minigame
corresponds to a specific game theoretic structure. All minigames are iterated
games (in the game theory sense), and the contestants never know the number of
rounds in advance. Each round is always structured with two phases. First,
players have a chance to communicate to one another. Second, players must select
an action (all games are simultaneous move). The players will be told the set of
legal game actions during each action phase.\n
"""

GM_BACKGROUND_KNOWLEDGE = GENERAL_BACKGROUND + """
We focus on minigames that are social dilemmas, collective action problems, or
bargaining problems.

A minigame consists of the following:
1) a game representation such as a Schelling diagram which maps joint actions to
rewards per player.
2) a cover story to tell the players about the setting.
3) a way of understanding how actions described in natural language map to game
actions.
4) specific information to tell all players about the setting at the start.
5) the true number of rounds the minigame will run (unknown to the players).
"""
SCENARIO_PREMISE = []

CANDIDATE_SHOW_TITLES = [
    'Motive Madness',
    'The Motivation Marathon',
    'Motive Mayhem',
    'The Incentive Initiative',
    'Motive Mashup',
    'The Motivation Matrix',
    'Motive Mania',
    'Dilemma Dash',
    'The Dilemma Dome',
    'Dilemma Detours',
    'The Dilemma Decathlon',
    'Dilemma Dungeons',
    'The Dilemma Dynasty',
    'Dilemma Deathmatch',
]
CANDIDATE_SHOW_DESCRIPTIONS = [
    (
        'A multi-episode event where participants engage in a marathon of '
        'minigames, with their motivations tested at every turn.'
    ),
    (
        'A reality show that pushes contestants to their limits with a series '
        'of increasingly challenging minigames, each designed to explore the '
        'depths of human motivation.'
    ),
    (
        'A high-stakes game show where contestants navigate a series of '
        'minigames, each with its own unique twist on motivation and '
        'decision-making.'
    ),
    (
        'Participants face off in a variety of challenges designed to test'
        ' their ability to make choices under pressure, with ever-changing'
        ' incentives and consequences.'
    ),
    (
        'A fast-paced competition where players must quickly adapt to a diverse'
        ' range of minigames, each with its own set of motivational factors.'
    ),
    (
        'A high-energy competition where contestants must master a wide range '
        'of minigames, each with its own unique motivational twist.'
    ),
    (
        'Contestants race through a gauntlet of minigames, each presenting a'
        ' unique moral or ethical dilemma that tests their decision-making'
        ' skills under pressure.'
    ),
    (
        'A reality show where players must navigate a series of complex '
        'minigames, each with its own set of conflicting objectives and moral '
        'quandaries.'
    ),
    (
        'Players are transported to alternate realities, each with its own'
        ' unique set of minigames and moral dilemmas to overcome.'
    ),
    (
        'Contestants are locked inside a high-tech arena, where they must'
        ' conquer a series of mentally and physically challenging minigames,'
        ' each with its own ethical twist.'
    ),
    (
        'A reality show where players must navigate a maze of interconnected'
        ' minigames, with each decision leading them down a different path'
        ' filled with moral dilemmas.'
    ),
    (
        'Players descend into a mysterious underground labyrinth, where they'
        ' must solve a series of puzzle-based minigames, each with its own'
        ' moral dilemma.'
    ),
]

MINIGAME_INTRO_PREMISE = (
    "The show's host arrived to explain the next minigame. They "
    'said the following:\n'
)


@dataclasses.dataclass(frozen=True)
class MiniGameSpec:
  """A Specification for a minigame.

  Attributes:
    name: The name of the minigame.
    public_premise: Communicate this string to the players at the start of each
      minigame.
    schelling_diagram: A representation of the game mapping joint actions to
      rewards for each player. Schelling diagrams are described in the following
      paper: Schelling, T.C., 1973. Hockey helmets, concealed weapons, and
        daylight saving: A study of binary choices with externalities. Journal
        of Conflict resolution, 17(3), pp.381-428.
    map_external_actions_to_schelling_diagram: Map cooperation and defection
      actions to the names they take in the game's cover story. These strings
      must match the options given in the action spec.
    action_spec: The action specification.
  """

  name: str
  public_premise: str
  schelling_diagram: SchellingDiagram
  map_external_actions_to_schelling_diagram: Mapping[str, str]
  action_spec: agent_lib.ActionSpec


MINIGAMES = [
    MiniGameSpec(
        name='Carpooling',
        public_premise=MINIGAME_INTRO_PREMISE
        + (
            'The next minigame is called Carpooling. Three coworkers can '
            'carpool, cutting commute costs for all, or drive individually. '
            'The commute happens daily, creating repeated decisions.'
        ),
        schelling_diagram=SchellingDiagram(
            # A fear+greed-type (Prisoners' Dilemma-like) dilemma
            cooperation=lambda num_cooperators: num_cooperators - 1.0,
            defection=lambda num_cooperators: num_cooperators + 2.0,
        ),
        map_external_actions_to_schelling_diagram=dict(
            cooperation='try to carpool with others',
            defection='drive individually',
        ),
        action_spec=agent_lib.choice_action_spec(
            call_to_action='Which action would {name} choose in the minigame?',
            options=('try to carpool with others', 'drive individually'),
            tag='minigame_action',
        ),
    ),
    MiniGameSpec(
        name='Home Appliance Sharing',
        public_premise=MINIGAME_INTRO_PREMISE
        + (
            'Three neighbors share a tool/appliance infrequently. Each can '
            'maintain it for shared use, or let others handle '
            'upkeep and risk it being unavailable. Repeated use '
            'creates dilemmas each time the tool/appliance is needed.'
        ),
        schelling_diagram=SchellingDiagram(
            # A greed-type (Chicken-like) dilemma
            cooperation=lambda num_cooperators: 4.0 * num_cooperators,
            defection=lambda num_cooperators: 5.5 * num_cooperators - 2.0,
        ),
        map_external_actions_to_schelling_diagram=dict(
            cooperation='maintain the appliance',
            defection='let others handle upkeep of the appliance',
        ),
        action_spec=agent_lib.choice_action_spec(
            call_to_action='Which action would {name} choose in the minigame?',
            options=(
                'maintain the appliance',
                'let others handle upkeep of the appliance',
            ),
            tag='minigame_action',
        ),
    ),
    MiniGameSpec(
        name='Boat Race',
        public_premise=MINIGAME_INTRO_PREMISE
        + (
            'Three teammates are on a row boat racing team together. Each has '
            'the option to give the race their all and really row '
            'vigorously, but this option is very fatiguing and only '
            'effective when all choose it simultaneously. Alternatively, each '
            'teammate has the option of rowing less vigorously, this gets '
            'them to their goal more slowly, but is less fatiguing and does '
            'not require coordination with the others. The race is repeated '
            'many times, going back and forth across the lake.'
        ),
        schelling_diagram=SchellingDiagram(
            # A fear-type (Stag Hunt-like) dilemma
            cooperation=lambda num_cooperators: (4.0 * num_cooperators) - 1.0,
            defection=lambda num_cooperators: num_cooperators + 4.0,
        ),
        map_external_actions_to_schelling_diagram=dict(
            cooperation='row vigorously',
            defection='row less vigorously',
        ),
        action_spec=agent_lib.choice_action_spec(
            call_to_action='Which action would {name} choose in the minigame?',
            options=('row vigorously', 'row less vigorously'),
            tag='minigame_action',
        ),
    ),
]


def get_random_minigame() -> MiniGameSpec:
  return np.random.choice(MINIGAMES)


def get_random_show_with_description() -> tuple[str, str]:
  """Randomly sample a show title and description from a prewritten list."""
  title = np.random.choice(CANDIDATE_SHOW_TITLES)
  description = np.random.choice(CANDIDATE_SHOW_DESCRIPTIONS)
  return f'"{title}" is a reality TV show described as: "{description}"', title


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
) -> tuple[Sequence[str], str, str]:
  """Return the shared memories and context for all agents and game master."""
  show_title_and_description, show_title = get_random_show_with_description()

  shared_memories = [
      (
          'Alice, Bob, and Charlie are contestants on a reality show: '
          f'{show_title}. There are no other contestants besides Alice, Bob, '
          'and Charlie.'
      ),
      show_title_and_description,
      GENERAL_BACKGROUND,
  ]

  # The shared context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion. It '
      'is OK to omit details that seem less important:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context, show_title


def configure_players(
    show_title: str,
) -> tuple[
    list[formative_memories.AgentConfig], list[formative_memories.AgentConfig]
]:
  """Configure the players.

  Args:
    show_title: the name of the reality show.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
  """
  player_configs = [
      formative_memories.AgentConfig(
          name='Alice',
          gender='female',
          date_of_birth=datetime.datetime(year=1962, month=4, day=28),
          goal='make as much money as possible',
          context=(
              'Alice signed up to be a contestant on a reality TV show, '
              'and hopes to win it since she needs the prize '
              'money.'
          ),
          traits=(
              "Alice's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  f'Alice is a contestant on {show_title}.',
              ],
              'main_character': True,
              'initial_endowment': {'money': 0.0},
          },
      ),
      formative_memories.AgentConfig(
          name='Bob',
          gender='male',
          goal='make as much money as possible',
          date_of_birth=datetime.datetime(year=1940, month=9, day=13),
          context=(
              'Bob signed up to be a contestant on a reality TV show, '
              'and hopes to win it since he needs the prize '
              'money.'
          ),
          traits=(
              "Bob's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  f'Bob is a contestant on {show_title}.'
              ],
              'main_character': True,
              'initial_endowment': {'money': 0.0},
          },
      ),
      formative_memories.AgentConfig(
          name='Charlie',
          gender='male',
          date_of_birth=datetime.datetime(year=1978, month=2, day=11),
          goal='make as much money as possible',
          context=(
              'Charlie signed up to be a contestant on a reality TV show, '
              'and hopes to win it since he needs the prize '
              'money.'
          ),
          traits=(
              "Charlie's personality is like "
              + player_traits_and_styles.get_trait(flowery=True)
          ),
          extras={
              'player_specific_memories': [
                  f'Charlie is a contestant on {show_title}.',
              ],
              'main_character': True,
              'initial_endowment': {'money': 0.0},
          },
      ),
  ]

  main_player_configs = [
      player for player in player_configs if player.extras['main_character']
  ]
  supporting_player_configs = [
      player for player in player_configs if not player.extras['main_character']
  ]

  return main_player_configs, supporting_player_configs


def add_minigame_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    scene_type_name: str,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, SchellingPayoffs]:
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
    minigame_scene_type: the minigame scene type.
  """
  # Pick a minigame at random.
  selected_minigame = get_random_minigame()
  cooperation_option = (
      selected_minigame.map_external_actions_to_schelling_diagram['cooperation']
  )

  schelling_payoffs = gm_components.schelling_diagram_payoffs.SchellingPayoffs(
      model=model,
      memory=game_master_memory,
      cooperative_option=cooperation_option,
      resolution_scene=DECISION_SCENE_TYPE,
      cooperator_reward_fn=selected_minigame.schelling_diagram.cooperation,
      defector_reward_fn=selected_minigame.schelling_diagram.defection,
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
      components=[schelling_payoffs],
      action_spec=selected_minigame.action_spec,
      update_thought_chain=[thought_chains_lib.identity],
      randomise_initiative=True,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=verbose,
  )
  public_conclusion = 'Everyone returned to the break room after the game.'
  minigame_scene_type = scene_lib.SceneTypeSpec(
      name=scene_type_name,
      premise={
          'Alice': [selected_minigame.public_premise],
          'Bob': [selected_minigame.public_premise],
          'Charlie': [selected_minigame.public_premise],
      },
      conclusion={
          'Alice': [public_conclusion],
          'Bob': [public_conclusion],
          'Charlie': [public_conclusion],
      },
      action_spec=selected_minigame.action_spec,
      override_game_master=decision_env,
  )
  return minigame_scene_type, schelling_payoffs


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
) -> tuple[
    Sequence[scene_lib.SceneSpec],
    game_master.GameMaster | None,
    SchellingPayoffs,
]:
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
    decision_env: a game master to handle choice scenes
    schelling_payoffs: a component to compute rewards of collective action
  """

  player_configs = list(main_player_configs) + list(supporting_player_configs)

  conversation_phase_premise = (
      'Alice, Bob, and Charlie are in the break room. Here '
      'they can chat with one another in small groups or all '
      'together at once. Everyone may choose for themself '
      'how they want to spend this free time.'
  )

  scene_specs = {
      'conversation': scene_lib.SceneTypeSpec(
          name='conversation',
          premise={
              'Alice': [conversation_phase_premise],
              'Bob': [conversation_phase_premise],
              'Charlie': [conversation_phase_premise],
          },
      ),
  }
  scene_specs[DECISION_SCENE_TYPE], schelling_payoffs = add_minigame_scene_spec(
      model=model,
      game_master_memory=game_master_memory,
      players=players,
      clock=clock,
      player_configs=player_configs,
      scene_type_name=DECISION_SCENE_TYPE,
  )

  scenes = [
      scene_lib.SceneSpec(
          scene_type=scene_specs['conversation'],
          start_time=START_TIME + 0 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=START_TIME + 1 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=3,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['conversation'],
          start_time=START_TIME + 2 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=START_TIME + 3 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=2,
      ),
  ]
  return (
      scenes,
      scene_specs[DECISION_SCENE_TYPE].override_game_master,
      schelling_payoffs,
  )


def outcome_summary_fn(
    # `binary_joint_action` should be type Mapping[str, bool] (ie bool not int).
    unused_binary_joint_action: Mapping[str, int],
    rewards: Mapping[str, float],
) -> Mapping[str, str]:
  """Summarize the outcome of a decision scene."""
  result = {
      name: f'{name} got a score of {score}' for name, score in rewards.items()
  }
  return result


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
    shared_memories, shared_context, show_title = (
        get_shared_memories_and_context(model)
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
    )

    main_player_configs, supporting_player_configs = configure_players(
        show_title=show_title
    )
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
      print(self._all_memories[player_config.name].get_data_frame()['text'])

    self._all_players = main_players + supporting_players

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
        )
    )
    self._scenes, decision_env, schelling_payoffs = configure_scenes(
        model=self._model,
        game_master_memory=self._game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
    )
    self._schelling_payoffs = schelling_payoffs

    self._secondary_environments = [decision_env]

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
    player_scores = self._schelling_payoffs.get_scores()
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
