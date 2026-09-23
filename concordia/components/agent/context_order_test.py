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

"""Human and LLM acting share Concat's exact established context semantics."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.components.agent import human_act_component
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class ContextOrderTest(parameterized.TestCase):

  def policies(self, order):
    reader = mock.Mock(return_value='LOOK')
    model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    model.sample_text.return_value = 'LOOK'
    human = human_act_component.HumanActComponent(reader, component_order=order)
    llm = concat_act_component.ConcatActComponent(model, component_order=order)
    for policy in (human, llm):
      entity_agent_with_logging.EntityAgentWithLogging('Ilyra', policy)
    return human, llm, reader, model

  @parameterized.named_parameters(
      ('mapping_order', None, {'z': 'Z', 'a': 'A'}, 'Z\nA'),
      ('empty_order_sorts', [], {'z': 'Z', 'a': 'A'}, 'A\nZ'),
      (
          'partial_order_sorts_remainder',
          ['b'],
          {'z': 'Z', 'b': 'B', 'a': 'A'},
          'B\nA\nZ',
      ),
      ('explicit_order', ['b', 'a'], {'a': 'A', 'b': 'B'}, 'B\nA'),
      (
          'preserve_labels_and_whitespace',
          ['z', 'empty'],
          {'empty': '', 'z': '\nLabel:\n Z\n', 'a': ' A '},
          '\nLabel:\n Z\n\n A ',
      ),
      ('empty_context', None, {}, ''),
  )
  def test_exact_context_matches_concat_prompt(self, order, contexts, expected):
    human, llm, reader, model = self.policies(order)
    spec = entity_lib.free_action_spec(call_to_action='Your move, {name}?')
    human.get_action_attempt(contexts, spec)
    llm.get_action_attempt(contexts, spec)
    request = reader.call_args.args[0]
    self.assertEqual(request.context, expected)
    self.assertEqual(llm._context_for_action(contexts), expected)
    self.assertEqual(
        human.get_context_concat_order(), llm.get_context_concat_order()
    )
    prompt = model.sample_text.call_args.kwargs['prompt']
    self.assertTrue(prompt.startswith(expected + '\n'), repr(prompt))
    self.assertEqual(
        human.get_entity().get_last_log()['__act__']['Context'], expected
    )

  @parameterized.parameters(
      human_act_component.HumanActComponent,
      concat_act_component.ConcatActComponent,
  )
  def test_duplicate_order_is_rejected(self, policy):
    with self.assertRaisesRegex(ValueError, 'duplicate components: x, x'):
      policy(mock.Mock(), component_order=['x', 'x'])

  def test_explicit_missing_key_raises_without_asking_human_or_llm(self):
    human, llm, reader, model = self.policies(['missing'])
    for policy in (human, llm):
      with self.assertRaises(KeyError):
        policy.get_action_attempt(
            {'present': 'value'}, entity_lib.DEFAULT_ACTION_SPEC
        )
    reader.assert_not_called()
    model.sample_text.assert_not_called()

  def test_order_and_request_context_are_snapshots_and_retries_keep_string(
      self,
  ):
    order = ['b', 'a']
    human, llm, reader, _ = self.policies(order)
    order.reverse()
    self.assertEqual(human.get_context_concat_order(), ('b', 'a'))
    self.assertEqual(llm.get_context_concat_order(), ('b', 'a'))
    contexts = {'a': 'A', 'b': 'B'}
    requests = []

    def read(request):
      requests.append(request)
      contexts['b'] = 'changed'
      return '' if len(requests) == 1 else 'LOOK'

    reader.side_effect = read
    human.get_action_attempt(contexts, entity_lib.DEFAULT_ACTION_SPEC)
    self.assertEqual([r.context for r in requests], ['B\nA', 'B\nA'])
    self.assertEqual(requests[1].contexts['b'], 'B')

  @parameterized.parameters((None,), ([],), (['b', 'a'],))
  def test_order_state_format_and_partial_restore_match_concat(self, order):
    human, llm, _, _ = self.policies(order)
    self.assertEqual(
        human.get_state()['component_order'], llm.get_state()['component_order']
    )
    # Retain Concat's legacy empty-sequence serialization/restore semantics.
    for state in (
        human.get_state(),
        {},
        {'component_order': []},
        {'component_order': ['a']},
    ):
      human.set_state(state)
      llm.set_state(state)
      self.assertEqual(
          human.get_context_concat_order(), llm.get_context_concat_order()
      )
    self.assertNotIn('input_reader', human.get_state())


if __name__ == '__main__':
  absltest.main()
