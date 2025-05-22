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
from concordia.agents.deprecated import deprecated_agent
from concordia.agents.deprecated import entity_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import formative_memories
from concordia.clocks import game_clock
from concordia.deprecated.factory.agent import alternative_basic_agent
from concordia.deprecated.factory.agent import alternative_rational_agent
from concordia.deprecated.factory.agent import basic_agent
from concordia.deprecated.factory.agent import basic_agent_without_plan
from concordia.deprecated.factory.agent import observe_and_summarize_agent
from concordia.deprecated.factory.agent import observe_recall_prompt_agent
from concordia.deprecated.factory.agent import paranoid_agent
from concordia.deprecated.factory.agent import parochial_universalization_agent
from concordia.deprecated.factory.agent import rational_agent
from concordia.deprecated.factory.agent import synthetic_user
from concordia.language_model import no_language_model
from concordia.typing.deprecated import agent as agent_lib
import numpy as np


OPTIONS = ('x', 'y')
DECISION_ACTION_SPEC = agent_lib.choice_action_spec(
    call_to_action='Does {name} prefer x or y?',
    options=OPTIONS,
    tag='decision',
)
SPEECH_ACTION_SPEC = agent_lib.DEFAULT_SPEECH_ACTION_SPEC
AGENT_NAME = 'Rakshit'

AGENT_FACTORIES = {
    'alternative_basic_agent': alternative_basic_agent,
    'alternative_rational_agent': alternative_rational_agent,
    'basic_agent': basic_agent,
    'basic_agent_without_plan': basic_agent_without_plan,
    'observe_recall_prompt_agent': observe_recall_prompt_agent,
    'observe_and_summarize_agent': observe_and_summarize_agent,
    'paranoid_agent': paranoid_agent,
    'parochial_universalization_agent': parochial_universalization_agent,
    'rational_agent': rational_agent,
    'synthetic_user': synthetic_user,
}


def _embedder(text: str):
  del text
  return np.random.rand(3)


class AgentFactoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='alternative_basic_agent',
          agent_name='alternative_basic_agent',
          main_role=True
      ),
      dict(
          testcase_name='alternative_rational_agent',
          agent_name='alternative_rational_agent',
          main_role=True
      ),
      dict(
          testcase_name='basic_agent',
          agent_name='basic_agent',
          main_role=True
      ),
      dict(
          testcase_name='basic_agent_without_plan',
          agent_name='basic_agent_without_plan',
          main_role=True,
      ),
      dict(
          testcase_name='observe_and_summarize_agent',
          agent_name='observe_and_summarize_agent',
          main_role=True,
      ),
      dict(
          testcase_name='observe_recall_prompt_agent',
          agent_name='observe_recall_prompt_agent',
          main_role=True,
      ),
      dict(
          testcase_name='paranoid_agent',
          agent_name='paranoid_agent',
          main_role=True,
      ),
      dict(
          testcase_name='parochial_universalization_agent',
          agent_name='parochial_universalization_agent',
          main_role=True,
      ),
      dict(
          testcase_name='rational_agent',
          agent_name='rational_agent',
          main_role=True,
      ),
      dict(
          testcase_name='synthetic_user',
          agent_name='synthetic_user',
          main_role=True,
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
        memory=associative_memory.AssociativeMemory(
            sentence_embedder=_embedder),
        clock=clock,
        update_time_interval=datetime.timedelta(hours=1))

    self.assertEqual(agent.name, AGENT_NAME)
    self.assertIsInstance(
        agent, deprecated_agent.BasicAgent | entity_agent.EntityAgent
    )

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
