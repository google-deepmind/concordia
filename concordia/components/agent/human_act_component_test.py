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

"""Human input validation, entity lifecycle, and sequential GM contract tests."""

import json
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent_with_logging
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.environment import engine
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

OutputType = entity_lib.OutputType


def _make_human(reader, name='Wayfarer', context_components=None):
  return entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=name,
      act_component=human_act_component.HumanActComponent(reader),
      context_components=context_components or {},
  )


class HumanActComponentTest(parameterized.TestCase):

  def test_retries_without_reentering_context_lifecycle(self):
    requests = []
    answers = iter(['', 'open the silver door'])
    context = constant.Constant('A silver door bars your path.')
    original_pre_act = context.pre_act
    with mock.patch.object(
        context, 'pre_act', wraps=original_pre_act
    ) as pre_act:

      def read(request):
        requests.append(request)
        return next(answers)

      human = _make_human(read, context_components={'Scene': context})
      result = human.act(
          entity_lib.free_action_spec(
              call_to_action='What does {name} do? Literal JSON: {"door": 1}'
          )
      )
    self.assertEqual(result, 'open the silver door')
    self.assertEqual(pre_act.call_count, 1)
    self.assertEqual(human.get_phase(), entity_component.Phase.READY)
    self.assertEqual(requests[0].request_id, requests[1].request_id)
    self.assertIsNone(requests[0].error)
    self.assertIn('non-empty', requests[1].error)
    self.assertEqual(requests[1].previous_response, '')
    self.assertEqual(requests[0].entity_name, 'Wayfarer')
    self.assertEqual(
        requests[0].action_spec.call_to_action,
        'What does Wayfarer do? Literal JSON: {"door": 1}',
    )
    self.assertIn('silver door', requests[0].contexts['Scene'])
    with self.assertRaises(TypeError):
      requests[0].contexts['Scene'] = 'Changed'
    self.assertEqual(human.get_last_log()['__act__']['Value'], result)
    self.assertEqual(human.get_last_log()['__act__']['Source'], 'human')

  @parameterized.parameters(*entity_lib.CHOICE_ACTION_TYPES)
  def test_player_and_gm_choices_are_exact_and_retry(self, output_type):
    reader = mock.Mock(side_effect=['0', 'north', ' North ', 'North'])
    human = _make_human(reader)
    self.assertEqual(
        human.act(
            entity_lib.ActionSpec(
                call_to_action='Choose',
                output_type=output_type,
                options=('North', 'South'),
            )
        ),
        'North',
    )
    self.assertEqual(reader.call_count, 4)
    self.assertIn('not one of', reader.call_args.args[0].error)

  def test_options_with_whitespace_are_not_rewritten(self):
    human = _make_human(lambda request: ' North ')
    self.assertEqual(
        human.act(
            entity_lib.choice_action_spec(
                call_to_action='Choose', options=(' North ', 'South')
            )
        ),
        ' North ',
    )

  @parameterized.parameters('nan', 'NaN', 'inf', '-Infinity', '1e9999', 'three')
  def test_float_rejects_non_finite_and_invalid_input(self, bad):
    reader = mock.Mock(side_effect=[bad, '-1.25e2'])
    self.assertEqual(
        _make_human(reader).act(
            entity_lib.float_action_spec(call_to_action='Enter a number')
        ),
        '-1.25e2',
    )
    self.assertIsNotNone(reader.call_args.args[0].error)

  @parameterized.parameters(
      OutputType.FREE, OutputType.MAKE_OBSERVATION, OutputType.RESOLVE
  )
  def test_free_types_reject_blank_and_preserve_human_wording(
      self, output_type
  ):
    reader = mock.Mock(side_effect=[' \n ', '  A door opens.\n'])
    result = _make_human(reader).act(
        entity_lib.ActionSpec(
            call_to_action='Describe', output_type=output_type
        )
    )
    self.assertEqual(result, '  A door opens.\n')

  @parameterized.parameters(None, 123, (['answer'],))
  def test_non_text_input_gets_feedback(self, bad):
    reader = mock.Mock(side_effect=[bad, 'look'])
    self.assertEqual(_make_human(reader).act(), 'look')
    self.assertIsNone(reader.call_args.args[0].previous_response)
    self.assertIn('text', reader.call_args.args[0].error)

  def test_skip_does_not_ask_for_input(self):
    reader = mock.Mock()
    human = _make_human(reader)
    self.assertEqual(human.act(entity_lib.skip_this_step_action_spec()), '')
    reader.assert_not_called()
    self.assertEqual(human.get_phase(), entity_component.Phase.READY)

  def test_unsupported_type_fails_before_input(self):
    reader = mock.Mock()
    human = _make_human(reader)
    with self.assertRaises(NotImplementedError):
      human.act(
          entity_lib.ActionSpec(
              call_to_action='Unsupported',
              output_type=cast(entity_lib.OutputType, 'future'),
          )
      )
    reader.assert_not_called()

  @parameterized.parameters(EOFError, TimeoutError, KeyboardInterrupt)
  def test_reader_failure_is_not_an_action(self, failure):
    reader = mock.Mock(side_effect=failure)
    with self.assertRaises(failure):
      _make_human(reader).act()
    reader.assert_called_once()

  def test_requests_are_distinct_and_entity_routed(self):
    requests = []

    def read(request):
      requests.append(request)
      return 'look'

    one = _make_human(read, 'One')
    two = _make_human(read, 'Two')
    one.act()
    two.act()
    one.act()
    self.assertEqual([r.entity_name for r in requests], ['One', 'Two', 'One'])
    self.assertLen({r.request_id for r in requests}, 3)

  def test_entity_state_restores_without_serializing_transport(self):
    reader = mock.Mock(return_value='look')
    human = _make_human(reader)
    state = human.get_state()
    json.dumps(state)
    restored = _make_human(reader)
    restored.set_state(state)
    self.assertEqual(restored.act(), 'look')

  def test_other_entities_keep_normal_act_components(self):
    reader = mock.Mock(return_value='South')
    human = _make_human(reader)
    npc = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name='Companion',
        act_component=concat_act_component.ConcatActComponent(
            model=no_language_model.NoLanguageModel(),
            randomize_choices=False,
        ),
    )
    spec = entity_lib.choice_action_spec(
        call_to_action='Where next?', options=('North', 'South')
    )
    self.assertEqual(human.act(spec), 'South')
    self.assertEqual(npc.act(spec), 'North')
    reader.assert_called_once()


class HumanGameMasterTest(parameterized.TestCase):

  @parameterized.parameters(
      'not JSON',
      'null',
      '[]',
      '{}',
      '{"call_to_action": 1, "output_type": "free"}',
      '{"call_to_action": "Act", "output_type": 1}',
      '{"call_to_action": "Act", "output_type": "unknown"}',
      '{"call_to_action": "Act", "output_type": "choice", "options": []}',
      '{"call_to_action": "Act", "output_type": "choice", "options": ["A",'
      ' "A"]}',
      '{"call_to_action": "Act", "output_type": "choice", "options": "AB"}',
      '{"call_to_action": "Act", "output_type": "choice", "options": [1]}',
      '{"call_to_action": "Act", "output_type": "free", "options": ["A"]}',
      '{"call_to_action": "Act", "output_type": "free", "tag": 1}',
      '{"call_to_action": "Act", "output_type": "resolve"}',
      '{"call_to_action": "Act", "output_type": "free", "unexpected": true}',
      '{"call_to_action": "  ", "output_type": "free"}',
  )
  def test_next_action_spec_rejects_invalid_payloads(self, response):
    with self.assertRaises(ValueError):
      human_act_component.validate_response(
          response,
          entity_lib.ActionSpec(
              call_to_action='Supply the player action spec as JSON.',
              output_type=OutputType.NEXT_ACTION_SPEC,
          ),
      )

  @parameterized.parameters(
      entity_lib.free_action_spec(call_to_action='What does {name} do?'),
      entity_lib.choice_action_spec(
          call_to_action='Which door?', options=('silver, blue', 'gold')
      ),
      entity_lib.float_action_spec(call_to_action='How much?'),
      entity_lib.skip_this_step_action_spec(),
  )
  def test_sequential_engine_accepts_human_next_action_spec(self, player_spec):
    reader = mock.Mock(
        side_effect=[
            'Wayfarer',
            '{}',
            engine.action_spec_to_string(player_spec),
        ]
    )
    gm = _make_human(reader, 'GM')
    player = _make_human(mock.Mock())
    selected, actual_spec = sequential.Sequential().next_acting(gm, [player])
    self.assertIs(selected, player)
    self.assertEqual(actual_spec, player_spec)
    self.assertIsNotNone(reader.call_args.args[0].error)
    self.assertEqual(gm.get_phase(), entity_component.Phase.READY)

  def test_sequential_observation_resolution_termination_and_gm_selection(self):
    reader = mock.Mock(
        side_effect=[
            'You stand beneath two moons.',
            'The silver door opens.',
            'no',
            'No',
            'Yes',
            'Second GM',
        ]
    )
    gm = _make_human(reader, 'GM')
    player = _make_human(mock.Mock())
    second_gm = _make_human(mock.Mock(), 'Second GM')
    env = sequential.Sequential()
    self.assertEqual(
        env.make_observation(gm, player), 'You stand beneath two moons.'
    )
    env.resolve(gm, 'Wayfarer opens the silver door.')
    self.assertFalse(env.terminate(gm))
    self.assertTrue(env.terminate(gm))
    self.assertIs(env.next_game_master(gm, [gm, second_gm]), second_gm)
    self.assertEqual(
        [
            call.args[0].action_spec.output_type
            for call in reader.call_args_list
        ],
        [
            OutputType.MAKE_OBSERVATION,
            OutputType.RESOLVE,
            OutputType.TERMINATE,
            OutputType.TERMINATE,
            OutputType.TERMINATE,
            OutputType.NEXT_GAME_MASTER,
        ],
    )


if __name__ == '__main__':
  absltest.main()
