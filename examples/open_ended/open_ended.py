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
import datetime
import random
import types

from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from examples.modular.environment.modules import modern_london_social_context
from examples.modular.environment.modules import player_names
from concordia.factory.agent import basic_entity_agent__main_role
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


MAJOR_TIME_STEP = datetime.timedelta(minutes=2)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

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


def configure_scenes(
    player_configs: Sequence[formative_memories.AgentConfig],
    start_time: datetime.datetime,
    per_player_premise,
    num_rounds: int = 1,
) -> Sequence[scene_lib.SceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    player_configs: configs for the main characters
    start_time: the time to start the first scene
    per_player_premise: the premise for the scene for each player
    num_rounds: the number of rounds to run the scene for

  Returns:
    scenes: a sequence of scene specifications
  """

  action_spec = agent_lib.free_action_spec(
      call_to_action=(
          'Generate what {name} would do next. '
          'Give a specific activity. '
          'If the selected action has a direct or indirect object then it '
          'must be specified explicitly. For example, it is valid to respond '
          'with "{name} votes for Caroline because..." but not '
          'valid to respond with "{name} votes because...".'
      ),
      tag='action',
  )

  scene_specs = {
      'social': scene_lib.SceneTypeSpec(
          name='Main scene', premise=per_player_premise, action_spec=action_spec
      ),
  }

  scenes = [
      scene_lib.SceneSpec(
          scene_type=scene_specs['social'],
          start_time=start_time,
          participant_configs=player_configs,
          num_rounds=num_rounds,
      ),
  ]
  return scenes


class Simulation(Runnable):
  """Define the simulation API object for the launch script to interact with."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      setup_time: datetime.datetime,
      scenes: Sequence[scene_lib.SceneSpec],
      player_configs: list[formative_memories.AgentConfig],
      agent_module: types.ModuleType = basic_entity_agent__main_role,
      premise: str | None = None,
      fast_gm: bool = False,
  ):
    """Initialize the simulation object.

    The launch script assumes this API object has a run() method.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      setup_time: the time to start the simulation at.
      scenes: the scene storyboard structure.
      player_configs: configs for the main characters
      agent_module: the agent module to use for all main characters.
      premise: the premise for the scene
      fast_gm: if set to True then use a faster debug game master. This is not
        recommended for real experiments. The debug game master uses a thought
        chain which allows for unrealistic events. In particular, it allows
        agent hallucination to force other agents to take voluntary actions they
        would not take on their own. It also allows game master hallucination to
        override quotes from the players, substituting what the agents wanted to
        say for arbitrary other text generated by the game master without
        reference to the memories or components of the agents involved.
    """

    self._agent_module = agent_module
    self._model = model
    self._embedder = embedder
    self._measurements = measurements

    self._clock = game_clock.MultiIntervalClock(
        start=setup_time, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
    )

    importance_model = importance_function.AgentImportanceModel(self._model)
    importance_model_gm = importance_function.ConstantImportanceModel()
    self._blank_memory_factory = blank_memories.MemoryFactory(
        model=self._model,
        embedder=self._embedder,
        importance=importance_model.importance,
        clock_now=self._clock.now,
    )
    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=[],
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        current_date=setup_time,
    )

    main_player_configs = player_configs
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
    for _, player_config in enumerate(main_player_configs):
      kwargs = dict(
          config=player_config,
          model=self._model,
          memory=self._all_memories[player_config.name],
          clock=self._clock,
          update_time_interval=MAJOR_TIME_STEP,
      )
      player = self._agent_module.build_agent(**kwargs)

      main_players.append(player)

    self._all_players = main_players

    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=self._embedder,
        importance=importance_model_gm.importance,
        clock=self._clock.now,
    )
    self._game_master_memory = game_master_memory

    if fast_gm:
      thought_chain = (
          thought_chains_lib.attempt_to_result,
          thought_chains_lib.result_to_who_what_where,
      )
    else:
      thought_chain = None

    self._primary_environment, self._game_master_memory = (
        basic_game_master.build_game_master(
            model=self._model,
            embedder=self._embedder,
            importance_model=importance_model_gm,
            clock=self._clock,
            players=self._all_players,
            shared_memories=[],
            shared_context=premise,
            blank_memory_factory=self._blank_memory_factory,
            cap_nonplayer_characters_in_conversation=2,
            memory=game_master_memory,
            thought_chain=thought_chain,
            verbose=False,
        )
    )

    self._scenes = scenes

    self._secondary_environments = []

    self._init_premise_memories(
        setup_time=setup_time,
        main_player_configs=main_player_configs,
        shared_memories=[premise],
        scenario_premise=[],
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
      main_player_configs: Sequence[formative_memories.AgentConfig],
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

    return html_results_log
