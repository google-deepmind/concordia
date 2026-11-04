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

"""Known location labels are matched as complete literal labels."""

import json

from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.components.game_master import world_state
from concordia.language_model import no_language_model
from concordia.typing import entity
from concordia.typing import entity_component


class ScriptedModel(no_language_model.NoLanguageModel):

  def __init__(self, destination):
    self.destination = destination
    self.calls = 0

  def sample_text(self, prompt, **kwargs):
    self.calls += 1
    if self.calls == 1:
      return 'Hall|A hall,Cafe|A cafe,Town Hall|A civic building'
    return 'Alice|' + self.destination


class ResolveAct(entity_component.ActingComponent):

  def get_action_attempt(self, contexts, action_spec):
    return 'Alice walked to the stated destination.'

  def get_state(self):
    return {}

  def set_state(self, state):
    pass


class LocationLabelMatchingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('substring_is_not_label', 'Shallow Lake', ['Hall', 'Cafe'], 'Cafe'),
      ('unicode_word_is_not_label', 'éHallé', ['Hall', 'Cafe'], 'Cafe'),
      ('longest_label', 'at Town Hall', ['Hall', 'Town Hall'], 'Town Hall'),
      (
          'case_insensitive_longest',
          'at TOWN HALL',
          ['Hall', 'Town Hall'],
          'Town Hall',
      ),
      ('canonical_period', 'D.C.', ['D.C.', 'Cafe'], 'D.C.'),
      ('case_insensitive_period', 'd.c.', ['D.C.', 'Cafe'], 'D.C.'),
      ('sentence_period', 'Hall.', ['Hall', 'Cafe'], 'Hall'),
      ('exact_short_label', 'Hall', ['Hall', 'Town Hall'], 'Hall'),
      ('case_insensitive_exact', 'hall', ['Hall', 'Town Hall'], 'Hall'),
      ('punctuated_phrase', '(Hall)', ['Hall', 'Cafe'], 'Hall'),
      (
          'regex_literal',
          'at Hall (East)',
          ['Hall', 'Hall (East)'],
          'Hall (East)',
      ),
      ('unicode_label', 'at Café', ['Café', 'Cafe'], 'Café'),
      ('unknown_retains_previous', 'Somewhere else', ['Hall', 'Cafe'], 'Cafe'),
      ('same_retains_previous', 'SAME', ['Hall', 'Cafe'], 'Cafe'),
      ('freeform', 'Shallow Lake.', None, 'Shallow Lake.'),
  )
  def test_public_resolve_persists_canonical_location(
      self, raw, valid_locations, expected
  ):
    model = ScriptedModel(raw)
    locations = world_state.Locations(
        model,
        ['Alice'],
        'Owned local fixture',
        initial_locations={'Alice': 'Cafe'},
        valid_locations=valid_locations,
    )
    gm = entity_agent.EntityAgent('GM', ResolveAct(), {'locations': locations})
    gm.act(
        entity.ActionSpec(
            call_to_action='Resolve', output_type=entity.OutputType.RESOLVE
        )
    )
    saved = json.loads(json.dumps(gm.get_state()))
    self.assertEqual(
        saved['context_components']['locations']['entity_locations']['Alice'],
        expected,
    )
    self.assertEqual(locations.get_pre_act_value(), f'Alice: {expected}\n')
    self.assertEqual(model.calls, 2)
