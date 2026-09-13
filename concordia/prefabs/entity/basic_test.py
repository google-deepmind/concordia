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

"""Acting-policy injection preserves the basic prefab's context and lifecycle."""

from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import human_act_component
from concordia.components.agent import memory
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.utils import measurements
import numpy as np


def _memory():
  return basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda _: np.ones(8)
  )


class BasicTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('defaults', {}),
      ('goal', {'name': 'Ilyra', 'goal': 'Repair the loom.'}),
      (
          'custom_history',
          {
              'name': 'Ilyra',
              'goal': 'Repair the loom.',
              'observation_history_length': 2,
              'situation_perception_history_length': 1,
              'self_perception_history_length': 3,
              'person_by_situation_history_length': 2,
          },
      ),
  )
  def test_only_act_changes_and_perception_lifecycle_is_identical(self, params):
    prefab = basic.Entity(params=params)
    model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    model.sample_text.return_value = 'considering the cradle.'
    automated = prefab.build(model=model, memory_bank=_memory())
    human_model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    human_model.sample_text.return_value = model.sample_text.return_value
    reader = mock.Mock(return_value='EXAMINE the tuning fork')
    factory = mock.Mock(
        side_effect=lambda order: human_act_component.HumanActComponent(
            reader, component_order=order
        )
    )
    with mock.patch.object(concat_act_component, 'ConcatActComponent') as ctor:
      human = prefab.build(
          model=human_model,
          memory_bank=_memory(),
          act_component_factory=factory,
      )
    ctor.assert_not_called()
    expected_order = cast(
        concat_act_component.ConcatActComponent, automated.get_act_component()
    ).get_context_concat_order()
    assert expected_order is not None
    factory.assert_called_once_with(tuple(expected_order))
    self.assertIsInstance(
        human.get_act_component(), human_act_component.HumanActComponent
    )
    self.assertIsInstance(
        automated.get_act_component(), concat_act_component.ConcatActComponent
    )
    human_context = human.get_all_context_components()
    automated_context = automated.get_all_context_components()
    self.assertEqual(list(human_context), list(automated_context))
    for key, component in human_context.items():
      self.assertIs(type(component), type(automated_context[key]))
      self.assertEqual(
          component.get_state(), automated_context[key].get_state()
      )

    # Across two turns, all three perception calls and memory updates remain
    # exactly those of the ordinary prefab. Only its fourth (acting) call goes.
    for observation in ('The cradle is cracked.', 'A silver thread is loose.'):
      automated.observe(observation)
      human.observe(observation)
      self.assertEqual(
          automated.act(), automated.name + ' considering the cradle.'
      )
      self.assertEqual(human.act(), 'EXAMINE the tuning fork')
      self.assertEqual(model.sample_text.call_count, 4)
      self.assertEqual(human_model.sample_text.call_count, 3)
      self.assertEqual(
          human_model.sample_text.call_args_list,
          model.sample_text.call_args_list[:3],
      )
      human_model.sample_choice.assert_not_called()
      request = reader.call_args.args[0]
      self.assertEqual(set(request.contexts), set(human_context))
      self.assertEqual(
          request.context,
          cast(
              concat_act_component.ConcatActComponent,
              automated.get_act_component(),
          )._context_for_action(request.contexts),
      )
      for key in ('SituationPerception', 'SelfPerception', 'PersonBySituation'):
        self.assertIn('considering the cradle.', request.contexts[key])
        self.assertEqual(
            human.get_last_log()[key], automated.get_last_log()[key]
        )
      self.assertIn(observation, request.contexts['__observation__'])
      self.assertEqual(
          human.get_component(
              '__memory__', type_=memory.AssociativeMemory
          ).get_all_memories_as_text(),
          automated.get_component(
              '__memory__', type_=memory.AssociativeMemory
          ).get_all_memories_as_text(),
      )
      self.assertEqual(human.get_last_log()['__act__']['Source'], 'human')
      model.reset_mock()
      human_model.reset_mock()
    self.assertEqual(reader.call_count, 2)

  @parameterized.parameters(True, False)
  def test_default_concat_flags_order_and_measurements_unchanged(self, flags):
    model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    model.sample_text.return_value = 'LOOK'
    channels = measurements.Measurements()
    with mock.patch.object(
        concat_act_component,
        'ConcatActComponent',
        wraps=concat_act_component.ConcatActComponent,
    ) as constructor:
      entity = basic.Entity(
          params={
              'name': 'Ilyra',
              'goal': 'Repair the loom.',
              'randomize_choices': flags,
              'prefix_entity_name': flags,
              'measurements': cast(Any, channels),
          }
      ).build(model=model, memory_bank=_memory())
    self.assertEqual(constructor.call_count, 1)
    self.assertIs(constructor.call_args.kwargs['randomize_choices'], flags)
    self.assertIs(constructor.call_args.kwargs['prefix_entity_name'], flags)
    order = cast(
        concat_act_component.ConcatActComponent, entity.get_act_component()
    ).get_context_concat_order()
    assert order is not None
    self.assertEqual(order[:2], ('Instructions', 'Goal'))
    self.assertCountEqual(order, entity.get_all_context_components())
    self.assertIs(entity.measurements, channels)
    self.assertEqual(entity.act(), 'Ilyra LOOK' if flags else 'LOOK')
    self.assertEqual(model.sample_text.call_count, 4)

  @parameterized.parameters(basic, minimal)
  def test_factory_receives_prefab_order_and_cannot_mix_with_instance(
      self, prefab
  ):
    model = no_language_model.NoLanguageModel()
    config = prefab.Entity(params={'name': 'Ilyra', 'goal': 'Repair the loom.'})
    default = config.build(model, _memory())
    factory = mock.Mock(
        side_effect=lambda order: human_act_component.HumanActComponent(
            lambda request: 'LOOK', component_order=order
        )
    )
    human = config.build(model, _memory(), act_component_factory=factory)
    factory.assert_called_once_with(
        tuple(default.get_act_component().get_context_concat_order())
    )
    self.assertEqual(
        human.get_act_component().get_context_concat_order(),
        default.get_act_component().get_context_concat_order(),
    )
    with self.assertRaisesRegex(ValueError, 'not both'):
      config.build(
          model,
          _memory(),
          act_component=human.get_act_component(),
          act_component_factory=factory,
      )


if __name__ == '__main__':
  absltest.main()
