# Copyright 2025 DeepMind Technologies Limited.
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

"""Additional tests for action spec validation and edge cases."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.typing import entity as entity_lib


class ActionSpecValidationTest(parameterized.TestCase):
  """Tests for ActionSpec validation edge cases."""

  def test_validate_free_action_accepts_any_string(self):
    """FREE output type should accept any string."""
    spec = entity_lib.ActionSpec(
        call_to_action='What do you do?',
        output_type=entity_lib.OutputType.FREE,
    )
    # Should not raise
    spec.validate('any action')
    spec.validate('')
    spec.validate('Multi\nline\naction')
    spec.validate('Action with special chars: !@#$%^&*()')

  def test_validate_choice_action_rejects_invalid_choice(self):
    """CHOICE output type should reject actions not in options."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick one:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['apple', 'banana', 'cherry'],
    )
    with self.assertRaisesRegex(ValueError, 'not one of'):
      spec.validate('grape')

  def test_validate_choice_action_accepts_valid_choice(self):
    """CHOICE output type should accept valid options."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick one:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['apple', 'banana', 'cherry'],
    )
    spec.validate('apple')
    spec.validate('banana')
    spec.validate('cherry')

  def test_validate_choice_action_case_sensitive(self):
    """CHOICE validation should be case-sensitive."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick one:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['Apple', 'Banana'],
    )
    with self.assertRaisesRegex(ValueError, 'not one of'):
      spec.validate('apple')  # lowercase, should fail

  def test_validate_float_action_accepts_valid_float(self):
    """FLOAT output type should accept valid floats."""
    spec = entity_lib.ActionSpec(
        call_to_action='Enter a number:',
        output_type=entity_lib.OutputType.FLOAT,
    )
    spec.validate('3.14')
    spec.validate('0')
    spec.validate('-42.5')
    spec.validate('1e-10')

  def test_validate_float_action_rejects_non_float(self):
    """FLOAT output type should reject non-float strings."""
    spec = entity_lib.ActionSpec(
        call_to_action='Enter a number:',
        output_type=entity_lib.OutputType.FLOAT,
    )
    with self.assertRaisesRegex(ValueError, 'not a valid float'):
      spec.validate('not_a_number')

    with self.assertRaisesRegex(ValueError, 'not a valid float'):
      spec.validate('3.14.15')  # malformed

    with self.assertRaisesRegex(ValueError, 'not a valid float'):
      spec.validate('abc3.14')

  def test_validate_free_game_master_action_types(self):
    """FREE-type GM action types should accept any string."""
    # MAKE_OBSERVATION is a FREE type
    spec = entity_lib.ActionSpec(
        call_to_action='What happens?',
        output_type=entity_lib.OutputType.MAKE_OBSERVATION,
    )
    # FREE types accept anything, but validate() raises NotImplementedError
    # for unsupported types - this is a bug/limitation we're testing
    with self.assertRaisesRegex(NotImplementedError, 'Unsupported output type'):
      spec.validate('Something interesting happens')

    # RESOLVE is a FREE type
    spec = entity_lib.ActionSpec(
        call_to_action='Resolve action',
        output_type=entity_lib.OutputType.RESOLVE,
    )
    with self.assertRaisesRegex(NotImplementedError, 'Unsupported output type'):
      spec.validate('The action succeeds')

  @parameterized.parameters(
      entity_lib.OutputType.NEXT_ACTING,
      entity_lib.OutputType.NEXT_GAME_MASTER,
      entity_lib.OutputType.TERMINATE,
  )
  def test_validate_choice_game_master_types_unsupported(self, output_type):
    """Choice-based GM action types raise NotImplementedError (known limitation)."""
    # Note: This tests a current limitation - validate() doesn't handle
    # all output types. This should be fixed in a future PR.
    spec = entity_lib.ActionSpec(
        call_to_action='Choose:',
        output_type=output_type,
        options=['option1', 'option2'],
    )
    # Currently raises NotImplementedError instead of validating
    with self.assertRaisesRegex(NotImplementedError, 'Unsupported output type'):
      spec.validate('option1')


class ActionSpecCreationEdgeCasesTest(parameterized.TestCase):
  """Tests for ActionSpec creation edge cases."""

  def test_create_choice_without_options_raises_error(self):
    """Creating a CHOICE spec without options should raise ValueError."""
    with self.assertRaisesRegex(ValueError, 'Options must be provided'):
      entity_lib.ActionSpec(
          call_to_action='Pick one:',
          output_type=entity_lib.OutputType.CHOICE,
          options=(),
      )

  def test_create_choice_with_duplicate_options_raises_error(self):
    """Creating a CHOICE spec with duplicate options should raise ValueError."""
    with self.assertRaisesRegex(ValueError, 'must not contain duplicate'):
      entity_lib.ActionSpec(
          call_to_action='Pick one:',
          output_type=entity_lib.OutputType.CHOICE,
          options=['apple', 'banana', 'apple'],
      )

  def test_create_free_with_options_raises_error(self):
    """Creating a FREE spec with options should raise ValueError."""
    with self.assertRaisesRegex(ValueError, 'Options not supported'):
      entity_lib.ActionSpec(
          call_to_action='What do you do?',
          output_type=entity_lib.OutputType.FREE,
          options=['a', 'b'],
      )

  def test_create_float_with_options_raises_error(self):
    """Creating a FLOAT spec with options should raise ValueError."""
    with self.assertRaisesRegex(ValueError, 'Options not supported'):
      entity_lib.ActionSpec(
          call_to_action='Enter a number:',
          output_type=entity_lib.OutputType.FLOAT,
          options=['1', '2', '3'],
      )

  def test_options_converted_to_tuple(self):
    """ActionSpec should convert options list to tuple."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['a', 'b', 'c'],
    )
    self.assertIsInstance(spec.options, tuple)
    self.assertEqual(spec.options, ('a', 'b', 'c'))

  def test_create_choice_with_special_characters_in_options(self):
    """Options can contain special characters."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['option-1', 'option_2', 'option (3)', 'option.4'],
    )
    spec.validate('option-1')
    spec.validate('option_2')
    spec.validate('option (3)')
    spec.validate('option.4')

  def test_create_choice_with_whitespace_in_options(self):
    """Options can contain whitespace."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['option one', 'option two', 'option\tthree'],
    )
    spec.validate('option one')
    spec.validate('option\tthree')

  def test_create_with_empty_call_to_action(self):
    """call_to_action can be empty (e.g., for SKIP_THIS_STEP)."""
    spec = entity_lib.ActionSpec(
        call_to_action='',
        output_type=entity_lib.OutputType.SKIP_THIS_STEP,
    )
    self.assertEqual(spec.call_to_action, '')


class ActionSpecToAndFromDictTest(parameterized.TestCase):
  """Tests for ActionSpec serialization/deserialization."""

  def test_to_dict_free_action(self):
    """to_dict should properly serialize FREE action spec."""
    spec = entity_lib.ActionSpec(
        call_to_action='What do you do?',
        output_type=entity_lib.OutputType.FREE,
        tag='action',
    )
    result = spec.to_dict()
    self.assertEqual(result['call_to_action'], 'What do you do?')
    self.assertEqual(result['output_type'], 'free')
    self.assertEqual(result['options'], [])
    self.assertEqual(result['tag'], 'action')

  def test_to_dict_choice_action(self):
    """to_dict should properly serialize CHOICE action spec."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick one:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['a', 'b', 'c'],
        tag='decision',
    )
    result = spec.to_dict()
    self.assertEqual(result['call_to_action'], 'Pick one:')
    self.assertEqual(result['output_type'], 'choice')
    self.assertEqual(result['options'], ['a', 'b', 'c'])
    self.assertEqual(result['tag'], 'decision')

  def test_from_dict_free_action(self):
    """from_dict should properly deserialize FREE action spec."""
    action_dict = {
        'call_to_action': 'What do you do?',
        'output_type': 'free',
        'options': [],
        'tag': 'action',
    }
    spec = entity_lib.action_spec_from_dict(action_dict)
    self.assertEqual(spec.call_to_action, 'What do you do?')
    self.assertEqual(spec.output_type, entity_lib.OutputType.FREE)
    self.assertEqual(spec.options, ())
    self.assertEqual(spec.tag, 'action')

  def test_from_dict_choice_action(self):
    """from_dict should properly deserialize CHOICE action spec."""
    action_dict = {
        'call_to_action': 'Pick one:',
        'output_type': 'choice',
        'options': ['x', 'y', 'z'],
        'tag': 'decision',
    }
    spec = entity_lib.action_spec_from_dict(action_dict)
    self.assertEqual(spec.call_to_action, 'Pick one:')
    self.assertEqual(spec.output_type, entity_lib.OutputType.CHOICE)
    self.assertEqual(spec.options, ('x', 'y', 'z'))
    self.assertEqual(spec.tag, 'decision')

  def test_roundtrip_to_dict_from_dict(self):
    """Specs should survive to_dict -> from_dict roundtrip."""
    original = entity_lib.ActionSpec(
        call_to_action='Test action',
        output_type=entity_lib.OutputType.CHOICE,
        options=['opt1', 'opt2'],
        tag='test_tag',
    )
    spec_dict = original.to_dict()
    restored = entity_lib.action_spec_from_dict(spec_dict)
    self.assertEqual(original, restored)

  def test_from_dict_with_output_type_enum(self):
    """from_dict should handle OutputType as enum or string."""
    action_dict_with_enum = {
        'call_to_action': 'Test',
        'output_type': entity_lib.OutputType.FREE,
        'options': [],
        'tag': None,
    }
    spec = entity_lib.action_spec_from_dict(action_dict_with_enum)
    self.assertEqual(spec.output_type, entity_lib.OutputType.FREE)


class ActionSpecHelperFunctionsTest(absltest.TestCase):
  """Tests for ActionSpec helper functions."""

  def test_free_action_spec(self):
    """free_action_spec should create FREE action spec."""
    spec = entity_lib.free_action_spec(
        call_to_action='Test',
        tag='action',
    )
    self.assertEqual(spec.output_type, entity_lib.OutputType.FREE)
    self.assertEqual(spec.call_to_action, 'Test')
    self.assertEqual(spec.tag, 'action')

  def test_choice_action_spec(self):
    """choice_action_spec should create CHOICE action spec."""
    spec = entity_lib.choice_action_spec(
        call_to_action='Pick:',
        options=['a', 'b'],
        tag='choice',
    )
    self.assertEqual(spec.output_type, entity_lib.OutputType.CHOICE)
    self.assertEqual(spec.options, ('a', 'b'))
    self.assertEqual(spec.tag, 'choice')

  def test_float_action_spec(self):
    """float_action_spec should create FLOAT action spec."""
    spec = entity_lib.float_action_spec(
        call_to_action='Enter number:',
        tag='number',
    )
    self.assertEqual(spec.output_type, entity_lib.OutputType.FLOAT)
    self.assertEqual(spec.tag, 'number')

  def test_skip_this_step_action_spec(self):
    """skip_this_step_action_spec should create SKIP action spec."""
    spec = entity_lib.skip_this_step_action_spec()
    self.assertEqual(spec.output_type, entity_lib.OutputType.SKIP_THIS_STEP)
    self.assertEqual(spec.call_to_action, '')


class ActionSpecFrozenTest(absltest.TestCase):
  """Tests that ActionSpec is immutable."""

  def test_action_spec_is_frozen(self):
    """ActionSpec should be frozen (immutable)."""
    spec = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FREE,
    )
    with self.assertRaises(AttributeError):
      spec.call_to_action = 'Modified'

    with self.assertRaises(AttributeError):
      spec.output_type = entity_lib.OutputType.CHOICE

  def test_action_spec_options_immutable(self):
    """ActionSpec options should be immutable (tuple)."""
    spec = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['a', 'b'],
    )
    with self.assertRaises(TypeError):
      spec.options[0] = 'c'


class ActionSpecEqualityTest(parameterized.TestCase):
  """Tests for ActionSpec equality and hashing."""

  def test_identical_specs_are_equal(self):
    """Two specs with same values should be equal."""
    spec1 = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FREE,
        tag='action',
    )
    spec2 = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FREE,
        tag='action',
    )
    self.assertEqual(spec1, spec2)

  def test_different_call_to_action_not_equal(self):
    """Specs with different call_to_action should not be equal."""
    spec1 = entity_lib.ActionSpec(
        call_to_action='Test1',
        output_type=entity_lib.OutputType.FREE,
    )
    spec2 = entity_lib.ActionSpec(
        call_to_action='Test2',
        output_type=entity_lib.OutputType.FREE,
    )
    self.assertNotEqual(spec1, spec2)

  def test_different_output_type_not_equal(self):
    """Specs with different output_type should not be equal."""
    spec1 = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FREE,
    )
    spec2 = entity_lib.ActionSpec(
        call_to_action='Test',
        output_type=entity_lib.OutputType.FLOAT,
    )
    self.assertNotEqual(spec1, spec2)

  def test_different_options_not_equal(self):
    """Specs with different options should not be equal."""
    spec1 = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['a', 'b'],
    )
    spec2 = entity_lib.ActionSpec(
        call_to_action='Pick:',
        output_type=entity_lib.OutputType.CHOICE,
        options=['a', 'c'],
    )
    self.assertNotEqual(spec1, spec2)


if __name__ == '__main__':
  absltest.main()
