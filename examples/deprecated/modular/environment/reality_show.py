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
from typing import Any

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
from concordia.document import interactive_document
from concordia.environment.deprecated import game_master
from examples.deprecated.modular.environment.modules import player_traits_and_styles
from examples.deprecated.modular.environment.supporting_agent_factory import basic_agent as basic_agent_supporting
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


SchellingDiagram = gm_components.schelling_diagram_payoffs.SchellingDiagram
SchellingPayoffs = gm_components.schelling_diagram_payoffs.SchellingPayoffs

DEFAULT_TIME_AND_PLACE_MODULES = (
    'circa_2015_british_reality_show__chicken_3_players',
    'circa_2015_british_reality_show__prisoners_dilemma_3_players',
    'circa_2003_american_reality_show__chicken_3_players',
    'circa_2003_american_reality_show__chicken_4_players',
    'circa_2003_american_reality_show__prisoners_dilemma_3_players',
    'circa_2003_american_reality_show__prisoners_dilemma_4_players',
    'circa_2003_american_reality_show__stag_hunt_3_players',
    'circa_2003_american_reality_show__stag_hunt_4_players',
)

MAJOR_TIME_STEP = datetime.timedelta(minutes=10)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

DECISION_SCENE_TYPE = 'minigame'
DEBRIEF_SCENE_TYPE = 'debrief'

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


@dataclasses.dataclass
class WorldConfig:
  """The configuration of the simulated world."""

  minigame_name: str
  minigame: MiniGameSpec
  year: int
  month: int
  day: int
  num_players: int
  num_additional_minigame_scenes: int
  contestants: Mapping[str, Mapping[str, Any]]
  num_minigame_reps_per_scene: tuple[int, ...]
  num_minigame_reps_per_extra_scene: tuple[int, ...]
  seed: int


def get_random_show_with_description(rng: random.Random) -> tuple[str, str]:
  """Randomly sample a show title and description from a prewritten list."""
  title = rng.choice(CANDIDATE_SHOW_TITLES)
  description = rng.choice(CANDIDATE_SHOW_DESCRIPTIONS)
  return f'"{title}" is a reality TV show described as: "{description}"', title


def _get_all_contestant_names_string(contestant_names: Sequence[str]):
  r"""Returns names [a,b,c] in the string format: \'a, b, and c\'."""
  all_contestants_string = ', '.join([name for name in contestant_names[:-1]])
  all_contestants_string = (
      f'{all_contestants_string}, and {contestant_names[-1]}'
  )
  return all_contestants_string


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
    contestant_names: Sequence[str],
    rng: random.Random,
) -> tuple[Sequence[str], str, str]:
  """Return the shared memories and context for all agents and game master."""
  show_title_and_description, show_title = get_random_show_with_description(rng)
  all_contestants_string = _get_all_contestant_names_string(contestant_names)

  shared_memories = [
      (
          f'{all_contestants_string} are contestants on a reality show: '
          f'{show_title}. There are no other contestants besides '
          f'{all_contestants_string}.'
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
    model: language_model.LanguageModel,
    show_title: str,
    sampled_settings: Any,
    rng: random.Random,
) -> tuple[
    list[formative_memories.AgentConfig], list[formative_memories.AgentConfig]
]:
  """Configure the players.

  Args:
    model: the language model to use
    show_title: the name of the reality show.
    sampled_settings: the environment configuration containing the time and
      place details.
      rng: the random number generator to use.

  Returns:
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
  """

  def get_agent_config(player_name: str, sampled_settings: Any):
    birth_year = sampled_settings.year - (25 + rng.randint(-3, 3))
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    traits_str = sampled_settings.contestants[player_name]['traits']
    catchphrase = sampled_settings.contestants[player_name]['catchphrase']
    subject_pronoun = sampled_settings.contestants[player_name][
        'subject_pronoun'
    ]
    object_pronoun = sampled_settings.contestants[player_name][
        'subject_pronoun'
    ]
    prompt = interactive_document.InteractiveDocument(model)
    prompt.statement(
        'The following exercise is preparatory work for a role playing '
        'session. The purpose of the exercise is to fill in the backstory '
        f'for a character named {player_name}.'
    )
    prompt.statement(f'The year is {sampled_settings.year}.\n')
    age = sampled_settings.year - birth_year
    prompt.statement(
        f'{player_name} was born in the year {birth_year} so '
        f'{subject_pronoun} is currently {age} years old.\n'
    )
    prompt.statement(
        f'{player_name} is currently a contestant on a reality show called '
        f'{show_title}.\n'
    )
    prompt.statement(f'{player_name} is {traits_str}')
    prompt.statement(f"{player_name}'s catchphrase is: {catchphrase}")
    prompt.statement(
        'The following is the transcript of a '
        'confessional-style introductory interview '
        "conversation between the show's host and "
        f'{player_name}.'
    )
    interview = []
    for interview_question in sampled_settings.contestants[player_name][
        'interview_questions'
    ]:
      answer = prompt.open_question(question=interview_question, max_tokens=500)
      combined = f'Host -- "{interview_question}"\n{player_name} -- "{answer}"'
      interview.append(combined)
    hometown_question = 'Where are you from?'
    hometown = prompt.open_question(question=hometown_question, max_tokens=100)
    interview.append(
        f'Host -- "{hometown_question}"\n{player_name} -- "{hometown}"'
    )
    interview_str = '\n'.join(interview)
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
    print(interview_str)
    return formative_memories.AgentConfig(
        name=player_name,
        gender=sampled_settings.contestants[player_name]['gender'],
        date_of_birth=datetime.datetime(
            year=birth_year, month=birth_month, day=birth_day
        ),
        goal='make as much money as possible by winning the reality show',
        context=(
            f'{player_name} is a contestant on a reality TV show, '
            f'and hopes to win it since {subject_pronoun} needs the prize '
            f'money. {subject_pronoun} gave the following confessional-style '
            f'interview at the start of the show:\n{interview_str}\n'
            f'What casual acquaintances remember about {player_name} is that '
            f'{public_face}'
        ),
        traits=f'{player_name} is {traits_str}',
        extras={
            'player_specific_memories': [
                f'{player_name} is a contestant on {show_title}.',
                (
                    f'{player_name} gave the following confessional-style '
                    f'interview at the start of the show:\n{interview_str}'
                ),
                (
                    f'Some memorable things about {player_name} are that '
                    f'{public_face}'
                ),
            ],
            'main_character': True,
            'initial_endowment': {'money': 0.0},
            'public_face': public_face,
        },
    )

  # Embellish main player backstory prompts in parallel.
  player_configs = concurrency.map_parallel(
      functools.partial(get_agent_config, sampled_settings=sampled_settings),
      sampled_settings.contestants,
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

  return main_player_configs, supporting_player_configs


def add_minigame_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    scene_type_name: str,
    sampled_settings: Any,
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
    sampled_settings: the environment configuration containing the time and
      place details.
    verbose: whether to print verbose output or not.

  Returns:
    minigame_scene_type: the minigame scene type.
  """
  selected_minigame = sampled_settings.minigame
  cooperation_option = (
      selected_minigame.map_external_actions_to_schelling_diagram['cooperation']
  )
  contestant_names = [player_config.name for player_config in player_configs]
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
      active_players_observe_joint_action_and_outcome=True,
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
          name: [selected_minigame.public_premise] for name in contestant_names
      },
      conclusion={name: [public_conclusion] for name in contestant_names},
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
    start_time: datetime.datetime,
    sampled_settings: Any,
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
    start_time: the start time/date in the game world for the first scene
    sampled_settings: the environment configuration containing the time and
      place details

  Returns:
    scenes: a sequence of scene specifications
    decision_env: a game master to handle choice scenes
    schelling_payoffs: a component to compute rewards of collective action
  """
  player_configs = list(main_player_configs) + list(supporting_player_configs)

  contestant_names = [
      player_config.name for player_config in main_player_configs
  ]
  all_contestants_string = _get_all_contestant_names_string(contestant_names)

  conversation_phase_premise = (
      f'{all_contestants_string} are in the break room. Here '
      'they can chat with one another in small groups or all '
      'together at once. Everyone may choose for themself '
      'how they want to spend this free time.'
  )
  debrief_phase_premise = (
      'Host: -- "We have reached the end of the show! I would like to take a '
      'moment to thank you all for participating. I hope this was as much fun '
      'for you as it was for me!"'
  )

  scene_specs = {
      'conversation': scene_lib.SceneTypeSpec(
          name='conversation',
          premise={
              name: [conversation_phase_premise] for name in contestant_names
          },
      ),
  }
  scene_specs[DECISION_SCENE_TYPE], schelling_payoffs = add_minigame_scene_spec(
      model=model,
      game_master_memory=game_master_memory,
      players=players,
      clock=clock,
      player_configs=main_player_configs,
      scene_type_name=DECISION_SCENE_TYPE,
      sampled_settings=sampled_settings,
  )
  decision_env = scene_specs[DECISION_SCENE_TYPE].override_game_master
  scene_specs[DEBRIEF_SCENE_TYPE] = scene_lib.SceneTypeSpec(
      name=DEBRIEF_SCENE_TYPE,
      premise={name: [debrief_phase_premise] for name in contestant_names},
      action_spec=agent_lib.choice_action_spec(
          call_to_action='Host: -- "{name}, did you enjoy being on the show?"',
          options=('yes', 'no'),
          tag='debrief_action',
      ),
      override_game_master=decision_env,
  )

  scenes = [
      scene_lib.SceneSpec(
          scene_type=scene_specs['conversation'],
          start_time=start_time + 0 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=start_time + 1 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=sampled_settings.num_minigame_reps_per_scene[0],
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs['conversation'],
          start_time=start_time + 2 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      ),
      scene_lib.SceneSpec(
          scene_type=scene_specs[DECISION_SCENE_TYPE],
          start_time=start_time + 3 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=sampled_settings.num_minigame_reps_per_scene[1],
      ),
  ]

  for i in range(sampled_settings.num_additional_minigame_scenes):
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_specs['conversation'],
            start_time=start_time + len(scenes) * datetime.timedelta(hours=2),
            participant_configs=player_configs,
            num_rounds=1,
        )
    )
    scenes.append(
        scene_lib.SceneSpec(
            scene_type=scene_specs[DECISION_SCENE_TYPE],
            start_time=start_time + len(scenes) * datetime.timedelta(hours=2),
            participant_configs=player_configs,
            num_rounds=sampled_settings.num_minigame_reps_per_extra_scene[i],
        )
    )

  scenes.append(
      # The purpose of the debrief scene is to make it so the players receive
      # the observation containing their scores after the last minigame.
      scene_lib.SceneSpec(
          scene_type=scene_specs[DEBRIEF_SCENE_TYPE],
          start_time=start_time + len(scenes) * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      )
  )

  return (
      scenes,
      scene_specs[DECISION_SCENE_TYPE].override_game_master,
      schelling_payoffs,
  )


def _get_comparative(score_a: float, score_b: float):
  """Return a word to describe whether `score_a` is above or below `score_b`."""
  if score_a > score_b:
    return 'above'
  elif score_a == score_b:
    return 'equal to'
  elif score_a < score_b:
    return 'below'


def outcome_summary_fn(
    unused_binary_joint_action: Mapping[str, int],
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
    cumulative_rewards: Mapping[str, float],
) -> Mapping[str, str]:
  """Summarize the outcome of a decision scene."""
  result = {}
  mean_score = np.mean(list(rewards.values()))
  mean_cumulative_score = np.mean(list(cumulative_rewards.values()))
  for name, score in rewards.items():
    choice = joint_action[name]
    cumulative_score = cumulative_rewards[name]
    reward_comparative = _get_comparative(score, mean_score)
    cumulative_comparative = _get_comparative(
        cumulative_score, mean_cumulative_score
    )
    result[name] = (
        f'[minigame round outcome] {name} chose "{choice}" and got a score of '
        f'{score:.3g}, which was {reward_comparative} the average score of '
        f'{mean_score:.3g}. Cumulatively, {name} currently has a total score '
        f'of {cumulative_score:.3g}, which is {cumulative_comparative} the '
        f'average cumulative score of {mean_cumulative_score:.3g}.'
    )
  return result


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
    # No need for supporting agents in this environment.
    del supporting_agent_module

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

    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._time_and_place_module = time_and_place_module
    _, sampled_settings = helper_functions.load_time_and_place_module(
        time_and_place_module=time_and_place_module,
        default_time_and_place_modules=DEFAULT_TIME_AND_PLACE_MODULES,
        seed=seed,
    )
    self._rng = random.Random(sampled_settings.seed)
    contestant_names = list(sampled_settings.contestants.keys())

    start_time = datetime.datetime(
        year=sampled_settings.year,
        month=sampled_settings.month,
        day=sampled_settings.day,
    )
    setup_time = start_time - datetime.timedelta(days=1)
    self._clock = game_clock.MultiIntervalClock(
        start=setup_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.ConstantImportanceModel()
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    shared_memories, shared_context, show_title = (
        get_shared_memories_and_context(model, contestant_names, rng=self._rng)
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
    )

    main_player_configs, supporting_player_configs = configure_players(
        model=model,
        show_title=show_title,
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
      player = basic_agent_supporting.build_agent(
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
            max_conversation_length=2,
        )
    )
    self._scenes, decision_env, schelling_payoffs = configure_scenes(
        model=self._model,
        game_master_memory=self._game_master_memory,
        players=self._all_players,
        clock=self._clock,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        start_time=start_time,
        sampled_settings=sampled_settings,
    )
    self._schelling_payoffs = schelling_payoffs

    self._secondary_environments = [decision_env]

    self._init_premise_memories(
        setup_time=setup_time,
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

    player_scores = self._schelling_payoffs.get_scores()
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
