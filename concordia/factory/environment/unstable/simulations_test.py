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
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.environment.unstable.engines import synchronous
from concordia.factory.environment.unstable import simulation
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import scene as scene_lib
import numpy as np


ENVIRONMENT_FACTORIES = {
    'simulation': simulation,
}


def _embedder(text: str):
  del text
  return np.random.rand(16)


class EnvironmentFactoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='simulation',
           simulation_factory_name='simulation'),
  )
  def test_simulation_factory(self, simulation_factory_name: str):
    simulation_factory = ENVIRONMENT_FACTORIES[simulation_factory_name]
    model = no_language_model.NoLanguageModel()
    importance_model_gm = importance_function.ConstantImportanceModel()
    setup_time = datetime.datetime.now()
    clock = game_clock.MultiIntervalClock(
        start=setup_time,
        step_sizes=[datetime.timedelta(hours=1),
                    datetime.timedelta(minutes=10)])
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

    env, mem, game_master = simulation_factory.build_simulation(
        model=model,
        embedder=_embedder,
        importance_model=importance_model_gm,
        clock=clock,
        players=players,
        shared_memories=[],
    )
    self.assertIsInstance(env, synchronous.Synchronous)
    self.assertIsInstance(mem, associative_memory.AssociativeMemory)
    self.assertIsInstance(game_master,
                          entity_agent_with_logging.EntityAgentWithLogging)

    free_scenes = [
        scene_lib.ExperimentalSceneSpec(
            scene_type=scene_lib.ExperimentalSceneTypeSpec(
                name='day',
                game_master=game_master,
                engine=env),
            start_time=setup_time,
            participants=['Rakshit'],
            num_rounds=1,
        ),
    ]

    free_results_log = simulation_factory.run_simulation(
        model=model,
        players=players,
        clock=clock,
        scenes=free_scenes,
    )
    self.assertIsInstance(free_results_log, str)

    choice_scenes = [
        scene_lib.ExperimentalSceneSpec(
            scene_type=scene_lib.ExperimentalSceneTypeSpec(
                name='night',
                game_master=game_master,
                engine=env,
                action_spec=entity_lib.choice_action_spec(
                    call_to_action='Pick x or y',
                    options=['x', 'y']),
            ),
            start_time=setup_time,
            participants=['Rakshit'],
            num_rounds=1,
        ),
    ]

    choice_results_log = simulation_factory.run_simulation(
        model=model,
        players=players,
        clock=clock,
        scenes=choice_scenes,
    )
    self.assertIsInstance(choice_results_log, str)

if __name__ == '__main__':
  absltest.main()
