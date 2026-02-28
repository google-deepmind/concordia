"""Tests for day_in_the_life_initializer component."""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.components.game_master import day_in_the_life_initializer
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class DayInTheLifeInitializerTest(absltest.TestCase):

  def _make_component(self, **kwargs):
    base_kwargs = {
        'model': no_language_model.NoLanguageModel(),
        'next_game_master_name': 'DialogueGM',
        'player_names': ('Alice', 'Bob'),
        'scenario_type': 'first_date',
        'player_specific_memories': {'Alice': ['m1'], 'Bob': ['m2']},
        'player_specific_context': {
            'Alice': {'wearing': 'blue jacket', 'eating': 'sandwich'},
            'Bob': {'wearing': 'green shirt', 'eating': 'salad'},
        },
        'components': (),
        'delimiter_symbol': '***',
        'num_personal_events': 2,
    }
    base_kwargs.update(kwargs)
    return day_in_the_life_initializer.DayInTheLifeInitializer(**base_kwargs)

  def _make_entity(self, memory_component=None, observation_component=None):
    memory_component = memory_component or mock.MagicMock()
    observation_component = observation_component or mock.MagicMock()
    entity = mock.MagicMock()
    entity.name = 'InitializerGM'

    def get_component(key, type_=None):
      del type_
      if key == '__memory__':
        return memory_component
      if key == '__make_observation__':
        return observation_component
      return mock.MagicMock()

    entity.get_component.side_effect = get_component
    return entity, memory_component, observation_component

  def test_get_player_background_formats_list_memories(self):
    component = self._make_component(
        player_specific_memories={'Alice': ['first memory', 'second memory']}
    )

    background = component._get_player_background('Alice')

    self.assertIn('Relevant Memories:', background)
    self.assertIn('- first memory', background)
    self.assertIn('- second memory', background)

  def test_pre_act_non_next_game_master_is_noop(self):
    component = self._make_component()

    result = component.pre_act(
        entity_lib.ActionSpec(
            call_to_action='ignore',
            output_type=entity_lib.OutputType.FREE,
        )
    )

    self.assertEqual(result, '')

  def test_pre_act_initializes_once_then_hands_off(self):
    component = self._make_component()
    entity, _, _ = self._make_entity()
    component.set_entity(entity)

    with mock.patch.object(component, '_process_dyad') as process_dyad:
      first = component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='next',
              output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=('InitializerGM', 'DialogueGM'),
          )
      )
      second = component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='next',
              output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=('InitializerGM', 'DialogueGM'),
          )
      )

    self.assertEqual(first, 'InitializerGM')
    self.assertEqual(second, 'DialogueGM')
    process_dyad.assert_called_once()

  def test_generate_personal_events_splits_by_delimiter(self):
    component = self._make_component(num_personal_events=2)

    with mock.patch.object(
        day_in_the_life_initializer.interactive_document,
        'InteractiveDocument',
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.open_question.return_value = 'Wake up***Have lunch***'
      prompt.view.return_value.text.return_value = 'prompt text'

      events = component.generate_personal_events(
          player_name='Alice',
          context='Today is Monday.',
          shared_event='Meet Bob at cafe.',
      )

    self.assertEqual(events, ['Wake up', 'Have lunch'])

  def test_get_and_set_state_round_trip(self):
    component = self._make_component()
    component.set_state({'initialized': True})

    self.assertEqual(component.get_state(), {'initialized': True})

  def test_get_player_background_uses_string_memory(self):
    component = self._make_component(
        player_specific_memories={'Alice': 'single memory text'}
    )

    background = component._get_player_background('Alice')

    self.assertIn('Relevant Memories:', background)
    self.assertIn('single memory text', background)

  def test_generate_shared_setup_unknown_scenario_type_raises(self):
    component = self._make_component(scenario_type='unknown_type')

    with self.assertRaisesRegex(ValueError, 'Unknown scenario_type'):
      component.generate_shared_setup('Alice', 'Bob', 'context')

  def test_generate_shared_setup_single_rumination_logs_and_returns_scene(self):
    component = self._make_component(scenario_type='single_rumination')
    logging_channel = mock.MagicMock()
    component.set_logging_channel(logging_channel)

    with mock.patch.object(
        day_in_the_life_initializer.interactive_document,
        'InteractiveDocument',
    ) as interactive_document_cls:
      prompt = interactive_document_cls.return_value
      prompt.open_question.return_value = 'Alice sits quietly at a cafe.'
      prompt.view.return_value.text.return_value = 'prompt trace'

      result = component.generate_shared_setup('Alice', None, 'Evening context')

    self.assertEqual(result, 'Alice sits quietly at a cafe.')
    logging_channel.assert_called_once()
    logged = logging_channel.call_args[0][0]
    self.assertIn('Internal Scene', logged['Key'])

  def test_process_dyad_single_rumination_adds_events_for_both_players(self):
    component = self._make_component(scenario_type='single_rumination')
    memory = mock.MagicMock()
    observation = mock.MagicMock()

    with mock.patch.object(
        component,
        'generate_shared_setup',
        return_value='internal scene',
    ), mock.patch.object(
        component,
        'generate_personal_events',
        return_value=['event one', 'event two'],
    ):
      component._process_dyad('context', memory, observation)

    observation.add_to_queue.assert_any_call('Alice', '[Daily Personal Event 1] "event one"')
    observation.add_to_queue.assert_any_call('Bob', '[Daily Personal Event 2] "event two"')
    observation.add_to_queue.assert_any_call('Alice', '[Internal Scene] "internal scene"')
    observation.add_to_queue.assert_any_call('Bob', '[Internal Scene] "internal scene"')
    memory.add.assert_any_call('[DITL Internal Scene] Alice: "internal scene"')

  def test_process_dyad_two_player_injects_personal_and_shared_events(self):
    component = self._make_component(scenario_type='first_date')
    memory = mock.MagicMock()
    observation = mock.MagicMock()

    with mock.patch.object(
        component,
        'generate_shared_setup',
        return_value='shared setup scene',
    ), mock.patch.object(
        component,
        'generate_personal_events',
        side_effect=[['alice event 1', 'alice event 2'], ['bob event 1']],
    ):
      component._process_dyad('context', memory, observation)

    observation.add_to_queue.assert_any_call(
        'Alice', '[Daily Personal Event 1] "alice event 1"'
    )
    observation.add_to_queue.assert_any_call(
        'Alice', '[Daily Personal Event 2] "alice event 2"'
    )
    observation.add_to_queue.assert_any_call(
        'Bob', '[Daily Personal Event 1] "bob event 1"'
    )
    observation.add_to_queue.assert_any_call(
        'Alice', '[Daily Shared Setup] "shared setup scene"'
    )
    observation.add_to_queue.assert_any_call(
        'Bob', '[Daily Shared Setup] "shared setup scene"'
    )
    memory.add.assert_any_call('[DITL Personal Event] Alice: "alice event 1"')
    memory.add.assert_any_call('[DITL Personal Event] Bob: "bob event 1"')
    memory.add.assert_any_call(
        '[DITL Shared Setup] Alice and Bob: "shared setup scene"'
    )

  def test_process_dyad_two_player_handles_empty_personal_events(self):
    component = self._make_component(scenario_type='friend_meetup')
    memory = mock.MagicMock()
    observation = mock.MagicMock()

    with mock.patch.object(
        component,
        'generate_shared_setup',
        return_value='shared meetup scene',
    ), mock.patch.object(
        component,
        'generate_personal_events',
        side_effect=[[], []],
    ):
      component._process_dyad('context', memory, observation)

    self.assertEqual(observation.add_to_queue.call_count, 2)
    observation.add_to_queue.assert_any_call(
        'Alice', '[Daily Shared Setup] "shared meetup scene"'
    )
    observation.add_to_queue.assert_any_call(
        'Bob', '[Daily Shared Setup] "shared meetup scene"'
    )
    memory.add.assert_called_once_with(
        '[DITL Shared Setup] Alice and Bob: "shared meetup scene"'
    )

  def test_process_dyad_two_player_invalid_participant_propagates_error(self):
    component = self._make_component(player_names=('Alice', 'Eve'))
    memory = mock.MagicMock()
    observation = mock.MagicMock()

    with mock.patch.object(
        component,
        'generate_shared_setup',
        return_value='shared setup scene',
    ), mock.patch.object(
        component,
        'generate_personal_events',
        side_effect=[['alice event'], KeyError('Eve')],
    ):
      with self.assertRaises(KeyError):
        component._process_dyad('context', memory, observation)

  def test_generate_personal_events_asserts_when_eating_statement_empty(self):
    component = self._make_component(
        player_specific_context={
            'Alice': {'wearing': 'blue jacket', 'eating': '  '},
            'Bob': {'wearing': 'green shirt', 'eating': 'salad'},
        }
    )

    with self.assertRaisesRegex(AssertionError, 'eating statement is empty'):
      component.generate_personal_events('Alice', 'context', 'shared')


if __name__ == '__main__':
  absltest.main()
