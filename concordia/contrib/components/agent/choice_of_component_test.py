"""Tests for choice_of_component module."""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.components.agent import choice_of_component
from concordia.language_model import no_language_model
from concordia.typing import entity_component


class ChoiceOfComponentTest(absltest.TestCase):
  """Tests selection and delegation behavior for choice components."""

  def _make_entity(self, component_map):
    entity = mock.MagicMock()
    entity.name = 'Alice'
    entity.get_phase.return_value = entity_component.Phase.PRE_ACT
    entity.get_component.side_effect = lambda key, type_=None: component_map[key]
    return entity

  def test_selects_component_from_menu_and_returns_pre_act_value(self):
    obs_component = mock.MagicMock()
    obs_component.get_pre_act_value.return_value = 'observation text'

    selected_component = mock.MagicMock()
    selected_component.get_pre_act_value.return_value = 'selected value'

    unselected_component = mock.MagicMock()
    unselected_component.get_pre_act_value.return_value = 'other value'

    component = choice_of_component.ChoiceOfComponent(
        model=no_language_model.NoLanguageModel(),
        observations_component_key='observations',
        menu_of_components=('comp_a', 'comp_b'),
    )
    entity = self._make_entity({
        'observations': obs_component,
        'comp_a': unselected_component,
        'comp_b': selected_component,
    })
    component.set_entity(entity)

    with mock.patch.object(
        choice_of_component.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.multiple_choice_question.return_value = 1

      result = component.get_pre_act_value()

    self.assertEqual(result, 'selected value')
    selected_component.get_pre_act_value.assert_called_once()
    unselected_component.get_pre_act_value.assert_not_called()

  def test_choice_without_pre_act_suppresses_output_and_delegates_state(self):
    obs_component = mock.MagicMock()
    obs_component.get_pre_act_value.return_value = 'observation text'

    selected_component = mock.MagicMock()
    selected_component.get_pre_act_value.return_value = 'selected value'

    wrapper = choice_of_component.ChoiceOfComponentWithoutPreAct(
        model=no_language_model.NoLanguageModel(),
        observations_component_key='observations',
        menu_of_components=('comp_a',),
    )

    entity = self._make_entity({
        'observations': obs_component,
        'comp_a': selected_component,
    })
    wrapper.set_entity(entity)

    with mock.patch.object(
        choice_of_component.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.multiple_choice_question.return_value = 0

      self.assertEqual(wrapper.pre_act(mock.MagicMock()), '')
      self.assertEqual(wrapper.get_pre_act_value(), 'selected value')


if __name__ == '__main__':
  absltest.main()
