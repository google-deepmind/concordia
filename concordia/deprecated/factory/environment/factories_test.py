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

"""Test environment (game master) factories.
"""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents.deprecated import entity_agent_with_logging
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import formative_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components.agent import deprecated as agent_components
from concordia.deprecated.factory.environment import basic_game_master
from concordia.environment.deprecated import game_master
from concordia.language_model import no_language_model
from concordia.typing.deprecated import scene as scene_lib
import numpy as np


ENVIRONMENT_FACTORIES = {
    'basic_game_master': basic_game_master,
}


def _embedder(text: str):
  del text
  return np.random.rand(16)


class EnvironmentFactoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='basic_game_master',
           environment_name='basic_game_master'),
  )
  def test_give_me_a_name(self, environment_name: str):
    environment_factory = ENVIRONMENT_FACTORIES[environment_name]
    model = no_language_model.NoLanguageModel()
    importance_model_gm = importance_function.ConstantImportanceModel()
    setup_time = datetime.datetime.now()
    clock = game_clock.MultiIntervalClock(
        start=setup_time,
        step_sizes=[datetime.timedelta(hours=1),
                    datetime.timedelta(minutes=10)])
    blank_memory_factory = blank_memories.MemoryFactory(
        model=model,
        embedder=_embedder,
        importance=importance_model_gm.importance,
        clock_now=clock.now,
    )
    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        clock=clock,
        component_order=[],
    )
    player_a = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='Rakshit',
        act_component=act_component,
        context_components={},
    )

    players = [player_a]

    env, mem = environment_factory.build_game_master(
        model=model,
        embedder=_embedder,
        importance_model=importance_model_gm,
        clock=clock,
        players=players,
        shared_memories=[],
        shared_context='',
        blank_memory_factory=blank_memory_factory,
    )
    self.assertIsInstance(env, game_master.GameMaster)
    self.assertIsInstance(mem, associative_memory.AssociativeMemory)

    scenes = [
        scene_lib.SceneSpec(
            scene_type=scene_lib.SceneTypeSpec(name='day'),
            start_time=setup_time,
            participant_configs=[
                formative_memories.AgentConfig(name='Rakshit')],
            num_rounds=1,
        ),
    ]

    html_results_log = basic_game_master.run_simulation(
        model=model,
        players=players,
        primary_environment=env,
        secondary_environments=[],
        clock=clock,
        scenes=scenes,
    )
    self.assertIsInstance(html_results_log, str)


if __name__ == '__main__':
  absltest.main()
