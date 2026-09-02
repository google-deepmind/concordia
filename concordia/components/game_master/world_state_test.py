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

"""Tests for the world_state component."""

from absl.testing import absltest
from concordia.components.game_master import world_state
from concordia.language_model import no_language_model
from concordia.testing import mock_model
from concordia.typing import entity as entity_lib


class WorldStateTest(absltest.TestCase):
  """Tests for the WorldState component."""

  def setUp(self):
    super().setUp()
    self._model = no_language_model.NoLanguageModel()

  def test_get_pre_act_value_empty(self):
    component = world_state.WorldState(model=self._model)
    self.assertEqual(component.get_pre_act_value(), '\n')

  def test_get_pre_act_value_formats_entries(self):
    component = world_state.WorldState(model=self._model)
    component.set_state({
        'state': {'weather': 'sunny', 'time': 'noon'},
        'latest_action_spec': None,
    })
    self.assertEqual(
        component.get_pre_act_value(), 'weather: sunny\ntime: noon\n'
    )

  def test_get_pre_act_label(self):
    component = world_state.WorldState(
        model=self._model, pre_act_label='\nState'
    )
    self.assertEqual(component.get_pre_act_label(), '\nState')

  def test_state_round_trip(self):
    component = world_state.WorldState(model=self._model)
    original = {'state': {'a': '1'}, 'latest_action_spec': None}
    component.set_state(original)
    self.assertEqual(component.get_state(), original)

  def test_pre_act_records_action_spec_and_returns_state(self):
    component = world_state.WorldState(model=self._model)
    component.set_state({'state': {'x': 'y'}, 'latest_action_spec': None})
    action_spec = entity_lib.ActionSpec(
        call_to_action='test', output_type=entity_lib.OutputType.RESOLVE
    )
    self.assertEqual(component.pre_act(action_spec), 'x: y\n')
    self.assertEqual(
        component.get_state()['latest_action_spec'], action_spec.to_dict()
    )

  def test_action_spec_round_trip(self):
    component = world_state.WorldState(model=self._model)
    action_spec = entity_lib.ActionSpec(
        call_to_action='act', output_type=entity_lib.OutputType.FREE
    )
    component.pre_act(action_spec)
    restored = world_state.WorldState(model=self._model)
    restored.set_state(component.get_state())
    self.assertEqual(
        restored.get_state()['latest_action_spec'], action_spec.to_dict()
    )


class LocationsNormalizeTest(absltest.TestCase):
  """Tests for the Locations._normalize_location helper."""

  def _make_locations(self, valid_locations):
    return world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob'],
        prompt='a prompt',
        valid_locations=valid_locations,
    )

  def test_empty_location_normalizes_to_empty(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location(''), '')

  def test_free_form_when_no_valid_locations(self):
    locations = self._make_locations(None)
    self.assertEqual(locations._normalize_location('  Home.  '), 'Home')

  def test_exact_match(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location('Home'), 'Home')

  def test_case_insensitive_match(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location('home'), 'Home')

  def test_substring_match(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location('the Home base'), 'Home')

  def test_unknown_location_normalizes_to_empty(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location('School'), '')

  def test_trailing_period_is_stripped(self):
    locations = self._make_locations(['Home', 'Work'])
    self.assertEqual(locations._normalize_location('Home.'), 'Home')


class LocationsStateTest(absltest.TestCase):
  """Tests for the Locations component state handling."""

  def test_initial_locations_default_to_empty(self):
    locations = world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob'],
        prompt='p',
    )
    self.assertEqual(locations._entity_locations, {'Alice': '', 'Bob': ''})

  def test_initial_locations_override_defaults(self):
    locations = world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob', 'Carol'],
        prompt='p',
        initial_locations={'Alice': 'Home', 'Bob': 'Work'},
    )
    self.assertEqual(
        locations._entity_locations,
        {'Alice': 'Home', 'Bob': 'Work', 'Carol': ''},
    )

  def test_get_pre_act_value_filters_empty_locations(self):
    locations = world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob'],
        prompt='p',
    )
    self.assertEqual(locations.get_pre_act_value(), '\n')
    locations._entity_locations['Alice'] = 'Home'
    self.assertEqual(locations.get_pre_act_value(), 'Alice: Home\n')

  def test_state_round_trip(self):
    locations = world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob'],
        prompt='p',
    )
    locations._entity_locations['Alice'] = 'Home'
    state = locations.get_state()
    restored = world_state.Locations(
        model=mock_model.MockModel(''),
        entity_names=['Alice', 'Bob'],
        prompt='p',
    )
    restored.set_state(state)
    self.assertEqual(restored._entity_locations, locations._entity_locations)
    self.assertEqual(restored._locations, locations._locations)


if __name__ == '__main__':
  absltest.main()
