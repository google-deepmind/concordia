"""Tests for situation_representation_via_narrative module."""

import datetime
from collections.abc import Mapping
from typing import Any
from unittest import mock

from absl.testing import absltest

from concordia.contrib.components.agent import situation_representation_via_narrative
from concordia.language_model import no_language_model
from concordia.typing import entity_component


class SituationRepresentationTest(absltest.TestCase):

  def _make_entity(self, observation_component, named_components=None):
    named_components = named_components or {}
    entity = mock.MagicMock()
    entity.name = 'Alice'
    entity.get_phase.return_value = entity_component.Phase.PRE_ACT

    def get_component(key, type_=None):
      del type_
      if key == 'observations':
        return observation_component
      return named_components[key]

    entity.get_component.side_effect = get_component
    return entity

  def test_get_pre_act_value_builds_situation_and_logs(self):
    observation_component = mock.MagicMock()
    observation_component.get_pre_act_value.return_value = (
        'Alice arrived at the station.'
    )

    context_component = mock.MagicMock()
    context_component.get_pre_act_value.return_value = 'Weather: rainy.'

    logged_payloads = []

    def logging_channel(payload: Mapping[str, Any]) -> None:
      logged_payloads.append(payload)

    component = (
        situation_representation_via_narrative.SituationRepresentation(
            model=no_language_model.NoLanguageModel(),
            observation_component_key='observations',
            components=(('weather', 'Weather Context'),),
            clock_now=lambda: datetime.datetime(2026, 2, 16, 10, 30),
            logging_channel=logging_channel,
        )
    )
    component.set_entity(
        self._make_entity(
            observation_component=observation_component,
            named_components={'weather': context_component},
        )
    )

    with mock.patch.object(
        situation_representation_via_narrative.interactive_document,
        'InteractiveDocument',
    ) as interactive_document_cls:
      first_prompt = mock.MagicMock()
      first_prompt.open_question.return_value = 'Initial world summary.'
      first_prompt.view.return_value.text.return_value = 'first chain'

      second_prompt = mock.MagicMock()
      second_prompt.open_question.return_value = 'Updated situation summary.'
      second_prompt.view.return_value.text.return_value = 'second chain'

      interactive_document_cls.side_effect = [first_prompt, second_prompt]

      result = component.get_pre_act_value()

    self.assertEqual(result, 'Updated situation summary.')
    self.assertEqual(observation_component.get_pre_act_value.call_count, 2)
    context_component.get_pre_act_value.assert_called()
    self.assertLen(logged_payloads, 1)
    logged_payload = logged_payloads[0]
    self.assertEqual(logged_payload['Value'], 'Updated situation summary.')
    self.assertIn('***', logged_payload['Chain of thought'])

  def test_get_pre_act_value_is_cached_until_update(self):
    observation_component = mock.MagicMock()
    observation_component.get_pre_act_value.return_value = 'obs'

    component = (
        situation_representation_via_narrative.SituationRepresentation(
            model=no_language_model.NoLanguageModel(),
            observation_component_key='observations',
            declare_entity_as_protagonist=False,
        )
    )
    component.set_entity(self._make_entity(observation_component))

    with mock.patch.object(
        situation_representation_via_narrative.interactive_document,
        'InteractiveDocument',
    ) as interactive_document_cls:
      first_prompt = mock.MagicMock()
      first_prompt.open_question.return_value = 'Initial summary'
      first_prompt.view.return_value.text.return_value = 'initial chain'

      second_prompt = mock.MagicMock()
      second_prompt.open_question.return_value = 'Computed summary'
      second_prompt.view.return_value.text.return_value = 'update chain'

      interactive_document_cls.side_effect = [
          first_prompt,
          second_prompt,
          mock.MagicMock(
            open_question=mock.MagicMock(
              return_value='New summary after update'
            ),
            view=mock.MagicMock(
              return_value=mock.MagicMock(
                text=mock.MagicMock(return_value='new chain')
              )
            ),
          ),
      ]

      first_result = component.get_pre_act_value()
      second_result = component.get_pre_act_value()

      self.assertEqual(first_result, 'Computed summary')
      self.assertEqual(second_result, 'Computed summary')
      self.assertEqual(interactive_document_cls.call_count, 2)

      component.update()
      third_result = component.get_pre_act_value()

    self.assertEqual(third_result, 'New summary after update')
    self.assertEqual(interactive_document_cls.call_count, 3)

  def test_get_and_set_state_round_trip(self):
    component = situation_representation_via_narrative.SituationRepresentation(
        model=no_language_model.NoLanguageModel(),
        observation_component_key='observations',
    )
    component.set_state({'situation_thus_far': 'Known context'})

    self.assertEqual(
        component.get_state(),
        {'situation_thus_far': 'Known context'},
    )


if __name__ == '__main__':
  absltest.main()
