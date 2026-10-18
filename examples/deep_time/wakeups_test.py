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

"""Tests for event-driven wake-ups."""

import json

from absl.testing import absltest
from concordia.components.game_master import interrupt_scheduling
from concordia.contrib.pubsub import topics
from examples.deep_time import game as game_lib
from examples.deep_time import model_setup
from examples.deep_time import simulation
from examples.deep_time import wakeups as wakeups_lib

CONFIG = {
    'prefixes': ['geo.events', 'diplomacy.{key}.', 'bio.alerts.{key}'],
    'timer_ticks': 4,
    'alert_topic': 'bio.alerts.{key}',
    'alarms': [
        {
            'topic': 'bio.census',
            'path': 'local_temp_c.{key}',
            'threshold': 5.0,
            'text': 'The {name} meadows are now at {value:.0f} C.',
        },
        {
            'topic': 'bio.census',
            'path': 'cover.{key}',
            'threshold': 0.04,
            'text': 'The {name} now cover {value:.0%} of the planet.',
        },
    ],
}


def _census(white_c: float, black_c: float, white_cover: float = 0.3) -> str:
  return json.dumps({
      'local_temp_c': {'white': white_c, 'black': black_c},
      'cover': {'white': white_cover, 'black': 0.2},
  })


class WakeUpsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tick = [0]
    self.bus = topics.TopicBus(clock=lambda: self.tick[0])
    self.wakeups = wakeups_lib.WakeUps(self.bus)
    self.wakeups.add('White', 'white', CONFIG)
    self.wakeups.add('Black', 'black', CONFIG)
    self.bus.publish('bio.census', _census(20, 20))

  def step(self) -> set[str]:
    self.tick[0] += 1
    return self.wakeups.step(self.tick[0])

  def test_initial_timer_then_sleep_then_events(self):
    self.assertEqual(self.step(), {'White', 'Black'})
    self.assertEqual(self.step(), set())
    self.bus.publish('geo.events', 'A volcano erupts.', source='world')
    self.assertEqual(self.step(), {'White', 'Black'})
    self.bus.publish('diplomacy.white.black', 'An offer.', source='Black')
    self.assertEqual(self.step(), {'White'})
    self.assertTrue(self.wakeups.is_awake('White'))
    self.assertFalse(self.wakeups.is_awake('Black'))
    self.assertTrue(self.wakeups.is_awake('Gaia'))  # Not event-driven.

  def test_alarms_are_deadband_reflexes_on_the_census(self):
    self.step()
    self.bus.publish('bio.census', _census(23, 20))
    self.assertEqual(self.step(), set())
    self.bus.publish('bio.census', _census(26, 20))
    self.assertEqual(self.step(), {'White'})
    self.bus.publish('bio.census', _census(26, 20, white_cover=0.2))
    self.assertEqual(self.step(), {'White'})
    # Black only ever got each alarm's first reading (temperature, cover).
    self.assertEqual(self.bus.message_count('bio.alerts.black'), 2)

  def test_own_messages_do_not_wake_and_timers_do(self):
    self.step()
    self.bus.publish('diplomacy.white.black', 'Echo.', source='White')
    self.assertEqual(self.step(), set())
    for _ in range(2):
      self.step()
    self.assertEqual(self.step(), {'White', 'Black'})
    self.assertEqual(self.wakeups.woken_by['White'], {'timer': 2, 'event': 0})

  def test_state_round_trip(self):
    self.step()
    state = self.wakeups.get_state()
    timers = self.wakeups.scheduler.get_state()
    other = wakeups_lib.WakeUps(self.bus)
    other.add('White', 'white', CONFIG)
    other.add('Black', 'black', CONFIG)
    other.set_state(state)
    other.scheduler.set_state(timers)
    self.assertEqual(other.get_state(), state)
    self.assertEqual(other.scheduler.get_state(), timers)

  def test_the_world_gm_keeps_the_wake_ups(self):
    sim, context = simulation.build_simulation(
        'daisyworld', model_setup.make_models('mock')
    )
    assert context.wakeups is not None
    world = sim.game_masters[0]
    self.assertIs(
        world.get_component(
            interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY
        ),
        context.wakeups.scheduler,
    )
    self.assertIs(
        world.get_component(wakeups_lib.DEFAULT_WAKEUPS_COMPONENT_KEY),
        context.wakeups,
    )

  def test_context_skips_sleeping_species(self):
    context = game_lib.new_game('daisyworld')
    wakeups = context.wakeups
    assert wakeups is not None
    wakeups.add('White daisies', 'white', CONFIG)
    wakeups.step(1)
    wakeups.step(2)
    self.assertNotIn('White daisies', context.active_entities())


if __name__ == '__main__':
  absltest.main()
