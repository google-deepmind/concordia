# Copyright 2026 DeepMind Technologies Limited.
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

"""Runtime policy injection retains prefab context, memory and default behavior."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import scripted_act
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
import numpy as np


def _memory():
  return basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda _: np.ones(8)
  )


class ActingPolicyInjectionTest(parameterized.TestCase):

  @parameterized.parameters(basic, minimal)
  def test_factory_preserves_order_context_memory_and_lifecycle(self, prefab):
    config = prefab.Entity(params={'name': 'Player', 'goal': 'Find the door.'})
    normal_model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    normal_model.sample_text.return_value = 'considering the door.'
    script_model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    script_model.sample_text.return_value = (
        normal_model.sample_text.return_value
    )
    normal = config.build(normal_model, _memory())
    factory = mock.Mock(
        side_effect=lambda order: scripted_act.ScriptedActComponent(
            script_model,
            [
                {'name': 'Player', 'line': 'LOOK'},
                {'name': 'Player', 'line': 'WAIT'},
            ],
            component_order=order,
            prefix_entity_name=False,
        )
    )
    with mock.patch.object(concat_act_component, 'ConcatActComponent') as ctor:
      scripted = config.build(
          script_model, _memory(), act_component_factory=factory
      )
    ctor.assert_not_called()
    factory.assert_called_once_with(
        tuple(normal.get_act_component().get_context_concat_order())
    )
    normal_context = normal.get_all_context_components()
    script_context = scripted.get_all_context_components()
    self.assertEqual(list(normal_context), list(script_context))
    for key, component in normal_context.items():
      self.assertIs(type(script_context[key]), type(component))
      self.assertEqual(script_context[key].get_state(), component.get_state())
    for observation, response in [
        ('A silver door.', 'LOOK'),
        ('The bell tolls.', 'WAIT'),
    ]:
      normal.observe(observation)
      scripted.observe(observation)
      normal.act()
      self.assertEqual(scripted.act(), response)
      # Only the final LLM acting call is replaced; basic perception prompts
      # and their dependency chain, and minimal's non-LLM context, are retained.
      self.assertEqual(
          script_model.sample_text.call_args_list,
          normal_model.sample_text.call_args_list[:-1],
      )
      self.assertIn(
          observation, '\n'.join(scripted.get_last_log()['__act__']['Prompt'])
      )
      self.assertEqual(
          scripted.get_component('__memory__').get_all_memories_as_text(),
          normal.get_component('__memory__').get_all_memories_as_text(),
      )
      normal_model.reset_mock()
      script_model.reset_mock()

  @parameterized.parameters(basic, minimal)
  def test_prebuilt_policy_identity_and_mutually_exclusive_arguments(
      self, prefab
  ):
    model = no_language_model.NoLanguageModel()
    config = prefab.Entity(params={'name': 'Player'})
    policy = scripted_act.ScriptedActComponent(
        model, [{'name': 'Player', 'line': 'LOOK'}]
    )
    entity = config.build(model, _memory(), act_component=policy)
    self.assertIs(entity.get_act_component(), policy)
    self.assertEqual(entity.act(), 'Player LOOK')
    factory = mock.Mock()
    with self.assertRaisesRegex(ValueError, 'not both'):
      config.build(
          model, _memory(), act_component=policy, act_component_factory=factory
      )
    factory.assert_not_called()


if __name__ == '__main__':
  absltest.main()
