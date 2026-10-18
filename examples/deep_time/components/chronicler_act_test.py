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

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.agent import pubsub
from examples.deep_time import actors as actors_lib
from examples.deep_time.components import chronicler_act
from concordia.typing import entity as entity_lib
from reactivex import subject

_BLANK = 'This is the first chapter.'
_SPEC = entity_lib.free_action_spec(
    call_to_action=actors_lib.CHRONICLER_CALL_TO_ACTION
)


def _chronicler(
    act: chronicler_act.ChroniclerAct, drafts: subject.Subject, initial: str
) -> entity_agent.EntityAgent:
  return entity_agent.EntityAgent(
      'Chronicler',
      act_component=act,
      context_components={
          'Previous chapter': pubsub.Subscriber(
              None, initial_value=initial, subject=drafts
          )
      },
  )


class ChroniclerActTest(absltest.TestCase):

  def test_it_hands_in_each_chapter_its_mind_writes_once(self):
    drafts = subject.Subject()
    act = chronicler_act.ChroniclerAct('Previous chapter', blank=_BLANK)
    chronicler = _chronicler(act, drafts, _BLANK)
    self.assertEqual(chronicler.act(_SPEC), actors_lib.CHRONICLER_RESPONSE)
    drafts.on_next('The Quiet Years (ticks 2-5).')
    self.assertEqual(
        chronicler.act(_SPEC),
        f'{actors_lib.HAND_IN_CHAPTER}: The Quiet Years (ticks 2-5).',
    )
    self.assertEqual(chronicler.act(_SPEC), actors_lib.CHRONICLER_RESPONSE)
    # Restored from a checkpoint, it does not hand the chapter in again.
    restored = chronicler_act.ChroniclerAct('Previous chapter', blank=_BLANK)
    restored.set_state(act.get_state())
    again = _chronicler(
        restored, subject.Subject(), 'The Quiet Years (ticks 2-5).'
    )
    self.assertEqual(again.act(_SPEC), actors_lib.CHRONICLER_RESPONSE)


if __name__ == '__main__':
  absltest.main()
