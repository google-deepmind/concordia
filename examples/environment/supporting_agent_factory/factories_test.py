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

"""Test agent factories.
"""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from examples.environment.supporting_agent_factory import basic
from examples.environment.supporting_agent_factory import rational
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
import numpy as np


OPTIONS = ('x', 'y')
DECISION_ACTION_SPEC = entity_lib.choice_action_spec(
    call_to_action='Does {name} prefer x or y?',
    options=OPTIONS,
    tag='decision',
)
SPEECH_ACTION_SPEC = entity_lib.DEFAULT_SPEECH_ACTION_SPEC
AGENT_NAME = 'Rakshit'

AGENT_FACTORIES = {
    'basic': basic,
    'rational': rational,
}


def _embedder(text: str):
  del text
  return np.random.rand(3)


class AgentFactoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          agent_name='basic',
          main_role=False
      ),
      dict(
          testcase_name='rational',
          agent_name='rational',
          main_role=False,
      ),
  )
  def test_output_in_right_format(self, agent_name: str, main_role: bool):
    agent_factory = AGENT_FACTORIES[agent_name]
    model = no_language_model.NoLanguageModel()
    setup_time = datetime.datetime.now()
    clock = game_clock.MultiIntervalClock(
        start=setup_time,
        step_sizes=[datetime.timedelta(hours=1),
                    datetime.timedelta(minutes=10)])
    config = formative_memories.AgentConfig(
        name=AGENT_NAME,
        extras={'main_character': main_role})
    agent = agent_factory.build_agent(
        config=config,
        model=model,
        memory_bank=basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=_embedder),
        clock=clock,
    )

    self.assertEqual(agent.name, AGENT_NAME)
    self.assertIsInstance(agent, entity_agent.EntityAgent)

    agent.observe('foo')
    agent.observe('bar')

    # Free action
    action = agent.act()
    self.assertIsInstance(action, str)

    # Choice action
    action = agent.act(action_spec=DECISION_ACTION_SPEC)
    self.assertIn(action, OPTIONS)

    # Speech action
    action = agent.act(action_spec=SPEECH_ACTION_SPEC)
    self.assertIsInstance(action, str)

if __name__ == '__main__':
  absltest.main()
