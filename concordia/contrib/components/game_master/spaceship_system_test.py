"""Tests for spaceship_system module."""

from unittest import mock

from absl.testing import absltest

from concordia.contrib.components.game_master import spaceship_system
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class SpaceshipSystemTest(absltest.TestCase):

  def _make_component(self):
    return spaceship_system.SpaceshipSystem(
        model=no_language_model.NoLanguageModel(),
        system_name='Engine Core',
        system_max_health=3,
        system_failure_probability=1.0,
        warning_message='Warning: Engine Core is failing.',
        pre_act_label='Spaceship System',
    )

  def _make_entity(self, memory, observation, terminator, extra_components=None):
    entity = mock.MagicMock()
    extra_components = extra_components or {}

    def get_component(key, type_=None):
      del type_
      if key == '__memory__':
        return memory
      if key == '__make_observation__':
        return observation
      if key == '__terminate__':
        return terminator
      return extra_components[key]

    entity.get_component.side_effect = get_component
    return entity

  def test_pre_act_resolve_triggers_failure_and_decrements_health(self):
    component = self._make_component()
    memory = mock.MagicMock()
    observation = mock.MagicMock()
    terminator = mock.MagicMock()
    component.set_entity(self._make_entity(memory, observation, terminator))

    with mock.patch.object(
        spaceship_system.random, 'random', return_value=0.0
    ):
      result = component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='resolve',
              output_type=entity_lib.OutputType.RESOLVE,
          )
      )

    self.assertIn('System name: Engine Core', result)
    observation.add_to_queue.assert_called_once_with(
        'All', 'Warning: Engine Core is failing.'
    )
    memory.add.assert_called_once_with('Warning: Engine Core is failing.')
    self.assertEqual(component.get_state()['current_health'], 2)
    self.assertEqual(component.get_state()['step_counter'], 1)

  def test_pre_act_non_resolve_is_noop(self):
    component = self._make_component()
    memory = mock.MagicMock()
    observation = mock.MagicMock()
    terminator = mock.MagicMock()
    component.set_entity(self._make_entity(memory, observation, terminator))

    result = component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='free action',
            output_type=entity_lib.OutputType.FREE,
        )
    )

    self.assertEqual(result, '')
    observation.add_to_queue.assert_not_called()
    memory.add.assert_not_called()

  def test_post_act_when_fixed_resets_health_and_notifies(self):
    component = self._make_component()
    component.set_state({
        'system_name': 'Engine Core',
        'system_max_health': 3,
        'system_failure_probability': 1.0,
        'current_health': 1,
        'is_failing': True,
        'verbose': False,
        'step_counter': 2,
    })

    memory = mock.MagicMock()
    observation = mock.MagicMock()
    terminator = mock.MagicMock()
    component.set_entity(self._make_entity(memory, observation, terminator))

    with mock.patch.object(
        spaceship_system.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = True

      result = component.post_act('Crew repaired the subsystem.')

    self.assertEqual(result, '')
    observation.add_to_queue.assert_called_once_with(
        'All', 'The Engine Core was fixed.'
    )
    memory.add.assert_called_once_with('The Engine Core was fixed.')
    self.assertEqual(component.get_state()['current_health'], 3)
    self.assertFalse(component.get_state()['is_failing'])
    terminator.terminate.assert_not_called()

  def test_post_act_unfixed_and_zero_health_terminates(self):
    component = self._make_component()
    component.set_state({
        'system_name': 'Engine Core',
        'system_max_health': 3,
        'system_failure_probability': 1.0,
        'current_health': 0,
        'is_failing': True,
        'verbose': False,
        'step_counter': 2,
    })

    memory = mock.MagicMock()
    observation = mock.MagicMock()
    terminator = mock.MagicMock()
    component.set_entity(self._make_entity(memory, observation, terminator))

    with mock.patch.object(
        spaceship_system.interactive_document, 'InteractiveDocument'
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.yes_no_question.return_value = False

      component.post_act('Crew failed to repair the subsystem.')

    terminator.terminate.assert_called_once()


if __name__ == '__main__':
  absltest.main()
