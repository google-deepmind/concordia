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

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents.unstable import entity_agent_with_logging
from concordia.associative_memory.unstable import basic_associative_memory as associative_memory
from concordia.components.agent import unstable as agent_components
from concordia.environment.unstable.engines import synchronous
from concordia.factory.environment.unstable import conversation
from concordia.factory.environment.unstable import default
from concordia.language_model import no_language_model
import numpy as np


ENVIRONMENT_FACTORIES = {
    'conversation': conversation,
    'default': default,
}


def _embedder(text: str):
  del text
  return np.random.rand(16)


class EnvironmentFactoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='conversation',
           simulation_factory_name='conversation'),
      dict(testcase_name='default',
           simulation_factory_name='default'),
  )
  def test_simulation_factory(self, simulation_factory_name: str):
    simulation_factory = ENVIRONMENT_FACTORIES[simulation_factory_name]
    model = no_language_model.NoLanguageModel()

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
    )
    player_a = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='Rakshit',
        act_component=act_component,
        context_components={},
    )

    players = [player_a]
    environment = synchronous.Synchronous()

    memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )

    game_master = simulation_factory.build(
        model=model,
        memory_bank=memory_bank,
        players=players,
    )

    self.assertIsInstance(game_master,
                          entity_agent_with_logging.EntityAgentWithLogging)

    environment.run_loop(
        game_masters=[game_master],
        entities=players,
    )

    environment.run_loop(
        game_masters=[game_master],
        entities=players,
    )

if __name__ == '__main__':
  absltest.main()
