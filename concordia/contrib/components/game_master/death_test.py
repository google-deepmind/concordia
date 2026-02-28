"""Tests for death component."""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.components.game_master import death
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class DeathTest(absltest.TestCase):
  """Tests death resolution and non-resolution behavior."""

  def _make_component(self):
    return death.Death(
        model=no_language_model.NoLanguageModel(),
        pre_act_label='Death',
        actor_names=['Alice', 'Bob'],
    )

  def _make_entity(
      self,
      next_acting_component,
      observation_component,
      memory,
      terminator=None,
  ):
    entity = mock.MagicMock()
    terminator = terminator or mock.MagicMock()

    def get_component(key, type_=None):
      del type_
      if key == '__next_acting__':
        return next_acting_component
      if key == '__make_observation__':
        return observation_component
      if key == '__memory__':
        return memory
      if key == '__terminate__':
        return terminator
      return mock.MagicMock()

    entity.get_component.side_effect = get_component
    return entity

  def test_post_act_non_resolve_is_noop(self):
    component = self._make_component()
    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='free action',
            output_type=entity_lib.OutputType.FREE,
        )
    )

    with mock.patch.object(
        death.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      result = component.post_act('An event happened.')

    self.assertEqual(result, '')
    self.assertFalse(interactive_document_cls.called)

  def test_post_act_resolve_kills_actor_and_updates_components(self):
    component = self._make_component()

    next_acting_component = mock.MagicMock()
    observation_component = mock.MagicMock()
    memory = mock.MagicMock()
    entity = self._make_entity(
        next_acting_component=next_acting_component,
        observation_component=observation_component,
        memory=memory,
    )
    component.set_entity(entity)

    component.pre_act(
      entity_lib.ActionSpec(
        call_to_action='resolve',
        output_type=entity_lib.OutputType.RESOLVE,
      )
    )

    with mock.patch.object(
        death.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = True
      prompt.open_question.return_value = 'Alice'

      result = component.post_act('A dangerous event happened.')

    self.assertEqual(result, '')
    next_acting_component.remove_actor_from_sequence.assert_called_once_with(
        'Alice'
    )
    observation_component.add_to_queue.assert_called_once_with(
        'Alice', 'Alice has died.'
    )
    memory.add.assert_called_once_with('Alice has died.')
    self.assertEqual(component.get_state()['actors_names'], ['Bob'])

  def test_post_act_resolve_no_death_keyword_keeps_state(self):
    component = self._make_component()
    next_acting_component = mock.MagicMock()
    observation_component = mock.MagicMock()
    memory = mock.MagicMock()
    component.set_entity(
        self._make_entity(next_acting_component, observation_component, memory)
    )

    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='resolve',
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )

    with mock.patch.object(
        death.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = True
      prompt.open_question.return_value = 'NO_DEATH'

      result = component.post_act('A risky event happened.')

    self.assertEqual(result, '')
    next_acting_component.remove_actor_from_sequence.assert_not_called()
    observation_component.add_to_queue.assert_not_called()
    memory.add.assert_not_called()
    self.assertEqual(component.get_state()['actors_names'], ['Alice', 'Bob'])

  def test_post_act_ignores_unknown_names_in_who_died(self):
    component = self._make_component()
    next_acting_component = mock.MagicMock()
    observation_component = mock.MagicMock()
    memory = mock.MagicMock()
    component.set_entity(
        self._make_entity(next_acting_component, observation_component, memory)
    )

    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='resolve',
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )

    with mock.patch.object(
        death.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = True
      prompt.open_question.return_value = 'Charlie, Alice.'

      component.post_act('A dangerous event happened.')

    next_acting_component.remove_actor_from_sequence.assert_called_once_with(
        'Alice'
    )
    self.assertEqual(component.get_state()['actors_names'], ['Bob'])

  def test_post_act_no_death_and_no_actors_terminates(self):
    component = self._make_component()
    component.set_state(
        {
            'actors_names': [],
            'step_counter': 0,
            'last_action_spec': None,
        }
    )
    next_acting_component = mock.MagicMock()
    observation_component = mock.MagicMock()
    memory = mock.MagicMock()
    terminator = mock.MagicMock()
    component.set_entity(
        self._make_entity(
            next_acting_component,
            observation_component,
            memory,
            terminator=terminator,
        )
    )

    component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='resolve',
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )

    with mock.patch.object(
        death.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = False

      component.post_act('No one died this turn.')

    terminator.terminate.assert_called_once()


if __name__ == '__main__':
  absltest.main()
