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

"""Test entity prefabs."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import basic_with_plan
from concordia.prefabs.entity import fake_assistant_with_configurable_system_prompt
from concordia.prefabs.entity import minimal
from concordia.typing import entity as entity_lib
import numpy as np


OPTIONS = ('x', 'y')
DECISION_ACTION_SPEC = entity_lib.choice_action_spec(
    call_to_action='Does {name} prefer x or y?',
    options=OPTIONS,
    tag='decision',
)
SPEECH_ACTION_SPEC = entity_lib.DEFAULT_SPEECH_ACTION_SPEC
ENTITY_NAME = 'Rakshit'

AGENT_FACTORIES = {
    'basic': basic,
    'basic_with_plan': basic_with_plan,
    'fake_assistant_with_configurable_system_prompt': (
        fake_assistant_with_configurable_system_prompt
    ),
    'minimal': minimal,
}


def _embedder(text: str):
  del text
  return np.random.rand(3)


class EntityPrefabsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='basic', entity_name='basic'),
      dict(testcase_name='basic_with_plan', entity_name='basic_with_plan'),
      dict(testcase_name='fake_assistant_with_configurable_system_prompt',
           entity_name='fake_assistant_with_configurable_system_prompt'),
      dict(testcase_name='minimal', entity_name='minimal'),
  )
  def test_output_in_right_format(self, entity_name: str):
    entity_config_module = AGENT_FACTORIES[entity_name]
    model = no_language_model.NoLanguageModel()
    entity_config = entity_config_module.Entity(
        params=dict(name=ENTITY_NAME,
                    goal='learn to play a game.'),
    )
    entity = entity_config.build(
        model=model,
        memory_bank=basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=_embedder
        ),
    )
    self.assertEqual(entity.name, ENTITY_NAME)
    self.assertIsInstance(
        entity, entity_agent.EntityAgent
    )

    entity.observe('foo')
    entity.observe('bar')

    # Choice action
    action = entity.act(action_spec=DECISION_ACTION_SPEC)
    self.assertIn(action, OPTIONS)

    # Speech action
    action = entity.act(action_spec=SPEECH_ACTION_SPEC)
    self.assertIsInstance(action, str)


if __name__ == '__main__':
  absltest.main()
