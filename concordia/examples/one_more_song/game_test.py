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

"""Public-state and ballot invariants, without making model calls."""

import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from concordia.environment import engine
from concordia.environment import step_controller
from concordia.examples.one_more_song import game
from concordia.language_model import no_language_model
from concordia.typing import entity


class BallotTest(absltest.TestCase):

  def test_browser_draft_identity_is_stable_and_distinct_per_run(self):
    first = game.PlayerSession('fixture')
    second = game.PlayerSession('fixture')
    identity = first.snapshot()['game']['session_id']
    self.assertEqual(first.snapshot()['game']['session_id'], identity)
    self.assertNotEqual(second.snapshot()['game']['session_id'], identity)

  def test_phase_and_state_validation(self):
    phase = game.BallotPhase()
    request = entity.ActionSpec(
        call_to_action='?', output_type=entity.OutputType.NEXT_ACTION_SPEC
    )
    self.assertEqual(
        engine.action_spec_parser(phase.pre_act(request)).output_type,
        entity.OutputType.FREE,
    )
    phase.set_state({'completed': 7})
    self.assertEqual(
        engine.action_spec_parser(phase.pre_act(request)).options,
        ('ACCEPT', 'DECLINE'),
    )
    for bad in (-1, 10, True, '7'):
      with self.assertRaises(ValueError):
        phase.set_state({'completed': bad})

  def test_narrative_cannot_create_agreement(self):
    session = game.PlayerSession('fixture')
    session.record(
        step_controller.StepData(
            step=1,
            acting_entity='You',
            action='You: Both have agreed!',
            entity_actions={},
            entity_logs={'secret': {'text': 'never publish'}},
        ),
        0.1,
    )
    session.conclude()
    snapshot = session.snapshot()
    self.assertEmpty(snapshot['game']['votes'])
    self.assertStartsWith(snapshot['game']['ending'], 'No shared')
    self.assertNotIn('secret', str(snapshot))

  def test_both_explicit_accept_votes_required(self):
    for second, expected in [
        ('ACCEPT', 'Encore agreed'),
        ('DECLINE', 'No shared'),
    ]:
      session = game.PlayerSession('fixture')
      for step, name, vote in [(8, 'Maya', 'ACCEPT'), (9, 'Leon', second)]:
        session.record(
            step_controller.StepData(
                step=step,
                acting_entity=name,
                action=f'{name}: {vote}',
                entity_actions={},
                entity_logs={},
            ),
            0.1,
        )
      session.conclude()
      self.assertStartsWith(session.snapshot()['game']['ending'], expected)

  def test_public_journal_contains_votes_once_and_ending(self):
    session = game.PlayerSession('fixture')
    for step, actor, text in [
        (7, 'You', 'One quiet song, then finish.'),
        (8, 'Maya', 'ACCEPT'),
        (9, 'Leon', 'DECLINE'),
    ]:
      session.record(
          step_controller.StepData(
              step=step,
              acting_entity=actor,
              action=f'{actor}: {text}',
              entity_actions={},
              entity_logs={'private': {'text': 'never exported'}},
          ),
          0.1,
      )
    session.conclude()
    entries = session.snapshot()['entries']
    self.assertLen(entries, 4)
    self.assertEqual(entries[1]['text'], 'Maya: ACCEPT')
    self.assertEqual(entries[2]['text'], 'Leon: DECLINE')
    self.assertStartsWith(entries[3]['text'], 'No shared encore')
    self.assertNotIn('private', str(entries))

  def test_failed_run_saves_terminal_public_state_without_private_error(self):
    session = game.PlayerSession('fixture')
    _, simulation = game.build(no_language_model.NoLanguageModel(), session)
    with tempfile.TemporaryDirectory() as directory:
      output = pathlib.Path(directory)
      with mock.patch.object(
          simulation,
          'play',
          side_effect=RuntimeError('private provider details'),
      ):
        with self.assertRaisesRegex(RuntimeError, 'private provider details'):
          game.play(simulation, session, output)
      saved = json.loads((output / 'public.json').read_text())
      self.assertTrue(saved['finished'])
      self.assertIsNone(saved['game']['ending'])
      self.assertEmpty(saved['game']['votes'])
      self.assertIsNone(saved['pending'])
      self.assertNotIn('private provider details', json.dumps(saved))
      self.assertEqual(
          json.loads((output / 'timing.json').read_text())['completed_steps'], 0
      )

  def test_invalid_vote_is_not_inferred(self):
    session = game.PlayerSession('fixture')
    with self.assertRaises(ValueError):
      session.record(
          step_controller.StepData(
              step=8,
              acting_entity='Maya',
              action='Maya: I think they agreed',
              entity_actions={},
              entity_logs={},
          ),
          0.1,
      )
    self.assertEmpty(session.snapshot()['game']['votes'])


if __name__ == '__main__':
  absltest.main()
