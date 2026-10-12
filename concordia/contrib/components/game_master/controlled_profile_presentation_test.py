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

"""Unit tests for ControlledProfilePresentation component."""

import hashlib
import json
import unittest

from concordia.contrib.components.game_master import controlled_profile_presentation


class ControlledProfilePresentationTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.profiles = {
        'Author-A': {'title': 'Senior Engineer', 'affiliation': 'Lab One'},
        'Author-B': {'title': 'Junior Engineer', 'affiliation': 'Lab Two'},
    }
    self.aliases = {
        'Author-A': 'Contributor 1',
        'Author-B': 'Contributor 2',
    }

  def test_initialization_validation(self):
    # Empty profiles
    with self.assertRaises(ValueError):
      controlled_profile_presentation.ControlledProfilePresentation(profiles={})

    # Incomplete aliases
    with self.assertRaises(ValueError):
      controlled_profile_presentation.ControlledProfilePresentation(
          profiles=self.profiles,
          aliases={'Author-A': 'Contributor 1'},
      )

    # Empty alias string
    with self.assertRaises(ValueError):
      controlled_profile_presentation.ControlledProfilePresentation(
          profiles=self.profiles,
          aliases={'Author-A': ' ', 'Author-B': 'Contributor 2'},
      )

    # Duplicate aliases
    with self.assertRaises(ValueError):
      controlled_profile_presentation.ControlledProfilePresentation(
          profiles=self.profiles,
          aliases={'Author-A': 'Same Name', 'Author-B': 'Same Name'},
      )

    # Missing default visible field in profiles
    with self.assertRaises(ValueError):
      controlled_profile_presentation.ControlledProfilePresentation(
          profiles=self.profiles,
          default_visible_fields=['non_existent_field'],
      )

  def test_blinded_condition_excludes_profile_attributes_and_internal_names(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=())
    rendered = cpp.render_for('Reviewer')

    self.assertIn('Contributor 1', rendered)
    self.assertIn('Contributor 2', rendered)
    # Ensure internal names and profile fields are absent
    self.assertNotIn('Author-A', rendered)
    self.assertNotIn('Author-B', rendered)
    self.assertNotIn('Senior Engineer', rendered)
    self.assertNotIn('Junior Engineer', rendered)
    self.assertNotIn('Lab One', rendered)
    self.assertNotIn('Lab Two', rendered)

    self.assertTrue(cpp.verify_field_not_disclosed('Reviewer', 'title'))
    self.assertTrue(cpp.verify_field_not_disclosed('Reviewer', 'affiliation'))

  def test_visible_condition_includes_only_selected_fields(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=('title',))
    rendered = cpp.render_for('Reviewer')

    self.assertIn('Senior Engineer', rendered)
    self.assertIn('Junior Engineer', rendered)
    # Affiliation should remain hidden
    self.assertNotIn('Lab One', rendered)
    self.assertNotIn('Lab Two', rendered)

    self.assertFalse(cpp.verify_field_not_disclosed('Reviewer', 'title'))
    self.assertTrue(cpp.verify_field_not_disclosed('Reviewer', 'affiliation'))

  def test_swapped_counterbalancing_condition(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    # Swap titles between Author-A and Author-B
    cpp.set_condition_for_observer(
        'Reviewer',
        visible_fields=('title',),
        swaps={'Author-A': 'Author-B', 'Author-B': 'Author-A'},
    )
    rendered = cpp.render_for('Reviewer')

    # Contributor 1 (Author-A) should now show Junior Engineer
    self.assertIn('Contributor 1: title: Junior Engineer', rendered)
    # Contributor 2 (Author-B) should now show Senior Engineer
    self.assertIn('Contributor 2: title: Senior Engineer', rendered)

    # Canonical profiles must NOT be mutated
    self.assertEqual(self.profiles['Author-A']['title'], 'Senior Engineer')
    self.assertEqual(self.profiles['Author-B']['title'], 'Junior Engineer')

  def test_explicit_overrides(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer(
        'Reviewer',
        visible_fields=('title',),
        overrides={'Author-A': {'title': 'Principal Scientist'}},
    )
    rendered = cpp.render_for('Reviewer')
    self.assertIn('Contributor 1: title: Principal Scientist', rendered)
    self.assertIn('Contributor 2: title: Junior Engineer', rendered)

    # Overriding a hidden field must fail
    with self.assertRaises(ValueError):
      cpp.set_condition_for_observer(
          'Reviewer',
          visible_fields=(),
          overrides={'Author-A': {'title': 'Principal Scientist'}},
      )

    # Overriding an unknown subject must fail
    with self.assertRaises(ValueError):
      cpp.set_condition_for_observer(
          'Reviewer',
          visible_fields=('title',),
          overrides={'Unknown-Subject': {'title': 'Principal Scientist'}},
      )

    # Swapping unknown subject must fail
    with self.assertRaises(ValueError):
      cpp.set_condition_for_observer(
          'Reviewer',
          visible_fields=('title',),
          swaps={'Author-A': 'Unknown-Subject'},
      )


  def test_json_formatting_style(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=('title',))
    rendered = cpp.render_for('Reviewer', format_style='json')

    parsed = json.loads(rendered)
    self.assertEqual(len(parsed), 2)
    self.assertEqual(parsed[0], {'participant': 'Contributor 1', 'profile': {'title': 'Senior Engineer'}})
    self.assertEqual(parsed[1], {'participant': 'Contributor 2', 'profile': {'title': 'Junior Engineer'}})

  def test_audit_trail_and_cryptographic_fingerprint(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=('title',))
    rendered = cpp.render_for('Reviewer')

    audit = cpp.get_audit_trail()
    self.assertEqual(len(audit), 1)
    record = audit[0]
    self.assertEqual(record['observer'], 'Reviewer')
    self.assertEqual(record['displayed_names'], ['Contributor 1', 'Contributor 2'])
    self.assertEqual(record['visible_fields'], ['title'])
    self.assertEqual(record['rendered_text'], rendered)
    self.assertEqual(
        record['sha256'],
        hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
    )

  def test_queue_presentation_to_observation_component(self):
    class MockMakeObservation:
      def __init__(self):
        self.queue = {}
      def add_to_queue(self, entity_name, event):
        self.queue[entity_name] = event

    mock_obs = MockMakeObservation()
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=())
    event = cpp.queue_presentation(
        observer='Reviewer',
        observation_component=mock_obs,
        prefix='Notice for reviewer:',
        suffix='Please review the work.',
    )

    self.assertEqual(mock_obs.queue['Reviewer'], event)
    self.assertTrue(event.startswith('Notice for reviewer:'))
    self.assertTrue(event.endswith('Please review the work.'))
    self.assertIn('Contributor 1', event)

  def test_state_serialization(self):
    cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    cpp.set_condition_for_observer('Reviewer', visible_fields=('title',))
    cpp.render_for('Reviewer')

    state = cpp.get_state()

    new_cpp = controlled_profile_presentation.ControlledProfilePresentation(
        profiles=self.profiles,
        aliases=self.aliases,
    )
    new_cpp.set_state(state)

    self.assertEqual(new_cpp.get_audit_trail(), cpp.get_audit_trail())
    self.assertEqual(
        new_cpp.render_for('Reviewer'),
        cpp.render_for('Reviewer'),
    )


if __name__ == '__main__':
  unittest.main()
