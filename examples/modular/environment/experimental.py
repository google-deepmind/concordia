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
import copy
import datetime
import functools
import random
import types

from concordia.agents import entity_agent
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.environment.experimental import engine as engine_lib
from examples.modular.environment.modules import player_traits_and_styles
from examples.modular.environment.supporting_agent_factory import basic_agent as basic_agent_supporting
from examples.modular.scenario import scenarios as scenarios_lib
from concordia.factory.agent import basic_agent
from concordia.factory.environment.experimental import simulation as simulation_factory
from concordia.language_model import language_model
from concordia.typing import scene as scene_lib
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
import numpy as np


MAJOR_TIME_STEP = datetime.timedelta(minutes=10)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)


class SundropSaloon(entity_agent.EntityAgent):
  """A pub in Riverbend."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock.MultiIntervalClock,
      name: str = 'Sundrop Saloon',
  ):
    self._act_component = (
        agent_components.concat_act_component.ConcatActComponent(
            model=model,
            clock=clock,
        )
    )
    super().__init__(
        agent_name=name,
        act_component=self._act_component)

    background_constant = agent_components.constant.Constant(
        state='Sundrop Saloon is a pub in Riverbend.',
        pre_act_key='\nLocation: ',
    )
    self._context_components = (background_constant,)


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
) -> tuple[Sequence[str], str]:
  """Get shared memories and context for the simulation."""
  shared_memories = [
      'There is a hamlet named Riverbend.',
      'Riverbend is an idyllic rural town.',
      'The river Solripple runs through the village of Riverbend.',
      'The Solripple is a mighty river.',
      'Riverbend has a temperate climate.',
      'Riverbend has a main street.',
      'There is a guitar store on Main street Riverbend.',
      'There is a grocery store on Main street Riverbend.',
      'There is a school on Main street Riverbend.',
      'There is a library on Main street Riverbend.',
      'Riverbend has only one pub.',
      'There is a pub on Main street Riverbend called The Sundrop Saloon.',
      'Town hall meetings often take place at The Sundrop Saloon.',
      'Riverbend does not have a park',
      'The main crop grown on the farms near Riverbend is alfalfa.',
      'Farms near Riverbend depend on water from the Solripple river.',
      (
          'The local newspaper recently reported that someone has been dumping '
          + 'dangerous industrial chemicals in the Solripple river.'
      ),
      'All named characters are citizens. ',
      # 'All citizens are automatically candidates in all elections. ',
      'There is no need to register in advance to be on the ballot.',
  ]

  # The generic context will be used for the NPC context. It reflects general
  # knowledge and is possessed by all characters.
  shared_context = model.sample_text(
      'Summarize the following passage in a concise and insightful fashion:\n'
      + '\n'.join(shared_memories)
      + '\n'
      + 'Summary:'
  )
  return shared_memories, shared_context


def configure_players() -> tuple[list[formative_memories.AgentConfig],
                                 list[formative_memories.AgentConfig]]:
  """Get player configs for the simulation."""
  main_player_configs = []
  supporting_player_configs = []
  main_player_configs.append(
      formative_memories.AgentConfig(
          name='Alice',
          gender='female',
          date_of_birth=datetime.datetime(year=1990, month=1, day=19),
          goal=('Become the mayor of Riverbend.'),
          context='Alice is a running for mayor of Riverbend.',
          traits='Alice is hyper aggressive and driven to succeed.',
          extras={
              'player_specific_memories': [
                  (
                      'Alice knows that Mayor Bob has been dumping dangerous '
                      'industrial chemicals in the Solripple river.'
                  ),
              ],
              'main_character': True,
          },
      )
  )
  return main_player_configs, supporting_player_configs


def configure_scenes(
    engine: engine_lib.Engine,
    game_master: entity_agent_with_logging.EntityAgentWithLogging,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    supporting_player_configs: Sequence[formative_memories.AgentConfig],
    start_time: datetime.datetime,
) -> Sequence[scene_lib.ExperimentalSceneSpec]:
  """Configure the scene storyboard structure.

  Args:
    engine: the engine to use in all scenes
    game_master: the game master to use in all scenes
    main_player_configs: configs for the main characters
    supporting_player_configs: configs for the supporting characters
    start_time: the start time/date in the game world for the first scene

  Returns:
    scenes: a sequence of scene specifications
  """
  player_configs = list(main_player_configs) + list(supporting_player_configs)

  player_names = [
      player_config.name for player_config in player_configs
  ]

  free_phase_premise = (
      'It\'s springtime in Riverbend.'
  )

  scene_specs = {
      'free': scene_lib.ExperimentalSceneTypeSpec(
          name='free',
          engine=engine,
          game_master=game_master,
          premise={
              name: [free_phase_premise] for name in player_names
          },
      ),
  }

  scenes = [
      scene_lib.ExperimentalSceneSpec(
          scene_type=scene_specs['free'],
          start_time=start_time + 0 * datetime.timedelta(hours=2),
          participant_configs=player_configs,
          num_rounds=1,
      ),
  ]

  return scenes


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
      seed: random seed for the simulation.
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

    if seed is None:
      self._rng = random.Random(1)
    else:
      self._rng = random.Random(seed)

    start_time = datetime.datetime(
        year=2025,
        month=1,
        day=13,
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
    shared_memories, unused_shared_context = get_shared_memories_and_context(
        model)

    self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self._model,
        shared_memories=shared_memories,
        blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
    )

    main_player_configs, supporting_player_configs = configure_players()
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

    sundrop_saloon = SundropSaloon(
        model=self._model,
        clock=self._clock,
    )
    nonplayer_entities = [
        sundrop_saloon,
    ]

    self._primary_environment, self._game_master_memory, self._game_master = (
        simulation_factory.build_simulation(
            model=self._model,
            embedder=self._embedder,
            importance_model=importance_model_gm,
            clock=self._clock,
            players=self._all_players,
            shared_memories=shared_memories,
            nonplayer_entities=nonplayer_entities,
        )
    )

    self._scenes = configure_scenes(
        engine=self._primary_environment,
        game_master=self._game_master,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        start_time=start_time,
    )

    self._init_premise_memories(
        setup_time=setup_time,
        main_player_configs=main_player_configs,
        supporting_player_configs=supporting_player_configs,
        shared_memories=shared_memories,
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

  def __call__(self) -> None:
    """Run the simulation.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    simulation_factory.run_simulation(
        model=self._model,
        players=self._all_players,
        clock=self._clock,
        scenes=self._scenes,
        verbose=True,
    )
