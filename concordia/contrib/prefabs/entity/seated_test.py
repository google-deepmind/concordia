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

"""Tests for the seated entity prefab."""

from absl.testing import absltest
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import human_act_component
from concordia.contrib.components.agent import scripted_input
from concordia.contrib.prefabs.entity import seated
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
import numpy as np


def _bank() -> basic_associative_memory.AssociativeMemoryBank:
  return basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda text: np.ones(3)
  )


class SeatedEntityTest(absltest.TestCase):

  def test_factory_from_params_seats_the_reader(self):
    reader = scripted_input.ScriptedInput(['hello'])
    prefab = seated.Entity(
        params={
            'name': 'Alice',
            'act_component_factory': (
                lambda order: human_act_component.HumanActComponent(
                    reader, component_order=order
                )
            ),
        }
    )
    entity = prefab.build(no_language_model.NoLanguageModel(), _bank())
    action = entity.act(entity_lib.free_action_spec(call_to_action='Speak?'))
    self.assertEqual(action, 'hello')
    # The reader saw the entity's own context, instructions first.
    self.assertIn('Alice', reader.requests[0].context)

  def test_without_factory_the_model_acts(self):
    prefab = seated.Entity(params={'name': 'Bob'})
    entity = prefab.build(no_language_model.NoLanguageModel(), _bank())
    self.assertIsInstance(
        entity.get_act_component(), concat_act_component.ConcatActComponent
    )

  def test_keyword_overrides_params(self):
    script = scripted_input.ScriptedInput(['from keyword'])
    prefab = seated.Entity(
        params={
            'name': 'Carol',
            'act_component_factory': lambda order: None,
        }
    )
    entity = prefab.build(
        no_language_model.NoLanguageModel(),
        _bank(),
        act_component=human_act_component.HumanActComponent(script),
    )
    action = entity.act(entity_lib.free_action_spec(call_to_action='Speak?'))
    self.assertEqual(action, 'from keyword')


if __name__ == '__main__':
  absltest.main()
