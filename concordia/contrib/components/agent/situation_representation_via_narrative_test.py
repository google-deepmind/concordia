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

"""Tests component key sequences and legacy labels in narrative situations."""

from collections.abc import Mapping, Sequence
from typing import Any
import types
import unittest

from concordia.agents import entity_agent
from concordia.components.agent import constant
from concordia.contrib.components.agent import situation_representation_via_narrative as narrative
from concordia.language_model import no_language_model
from concordia.typing import entity
from concordia.typing import entity_component


class _RecordingModel(no_language_model.NoLanguageModel):

  def __init__(self):
    super().__init__()
    self.prompts: list[str] = []

  def sample_text(self, prompt: str, **kwargs: Any) -> str:
    del kwargs
    self.prompts.append(prompt)
    return 'current narrative'


class _Act(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity.ActionSpec,
  ) -> str:
    del action_spec
    return context['situation']

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _make(
    components: Sequence[str] | Mapping[str, str] | None,
    sources: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[entity_agent.EntityAgent, _RecordingModel]:
  model = _RecordingModel()
  if components is None:
    situation = narrative.SituationRepresentation(model, 'obs')
  else:
    situation = narrative.SituationRepresentation(
        model, 'obs', components=components
    )
  contexts: dict[str, entity_component.ContextComponent] = {
      'obs': constant.Constant('OBSERVATION'),
      'situation': situation,
  }
  if sources is None:
    sources = {'Instructions': ('FIRST', 'Visible source')}
  for key, (value, label) in sources.items():
    contexts[key] = constant.Constant(value, pre_act_label=label)
  return entity_agent.EntityAgent('Alice', _Act(), contexts), model


class SituationRepresentationTest(unittest.TestCase):

  def test_sequence_keys_use_component_labels(self):
    agent, model = _make(['Instructions'])
    self.assertEqual(agent.act(), 'The current situation:\ncurrent narrative\n')
    self.assertEqual(len(model.prompts), 2)
    for prompt in model.prompts:
      self.assertIn('Visible source:\nFIRST', prompt)

  def test_two_character_key_is_not_split(self):
    agent, model = _make(['ab'], {'ab': ('FIRST', 'Visible source')})
    agent.act()
    for prompt in model.prompts:
      self.assertIn('Visible source:\nFIRST', prompt)

  def test_tuple_of_keys(self):
    agent, model = _make(('Instructions',))
    agent.act()
    self.assertIn('Visible source:\nFIRST', model.prompts[0])

  def test_sequence_order_is_preserved(self):
    agent, model = _make(
        ['Second', 'Instructions'],
        {
            'Instructions': ('FIRST', 'First label'),
            'Second': ('SECOND', 'Second label'),
        },
    )
    agent.act()
    for prompt in model.prompts:
      self.assertIn('Second label:\nSECOND\nFirst label:\nFIRST', prompt)

  def test_repeated_keys_are_not_deduplicated(self):
    agent, model = _make(['Instructions', 'Instructions'])
    agent.act()
    for prompt in model.prompts:
      self.assertEqual(prompt.count('Visible source:\nFIRST'), 2)

  def test_supplied_sequence_is_owned(self):
    keys = ['Instructions']
    agent, model = _make(keys)
    keys.append('not registered')
    keys[0] = 'also not registered'
    agent.act()
    self.assertIn('Visible source:\nFIRST', model.prompts[0])

  def test_mapping_keeps_custom_labels_and_owns_container(self):
    labels = {'Instructions': 'Legacy label'}
    agent, model = _make(labels)
    labels['Instructions'] = 'changed'
    labels['not registered'] = 'new'
    agent.act()
    for prompt in model.prompts:
      self.assertIn('Legacy label:\nFIRST', prompt)
      self.assertNotIn('Visible source:', prompt)
      self.assertNotIn('changed', prompt)

  def test_read_only_mapping_keeps_its_custom_label(self):
    agent, model = _make(
        types.MappingProxyType({'Instructions': 'Legacy label'})
    )
    agent.act()
    self.assertIn('Legacy label:\nFIRST', model.prompts[0])

  def test_mapping_keeps_an_empty_custom_label(self):
    agent, model = _make({'Instructions': ''})
    agent.act()
    for prompt in model.prompts:
      self.assertIn('Considerations:\n:\nFIRST', prompt)
      self.assertNotIn('Visible source:', prompt)

  def test_empty_default_adds_no_considerations(self):
    agent, model = _make(None)
    agent.act()
    for prompt in model.prompts:
      self.assertNotIn('Considerations:', prompt)
      self.assertNotIn('FIRST', prompt)


if __name__ == '__main__':
  unittest.main()
