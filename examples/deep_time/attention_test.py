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

"""Tests for attention slots."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import attention as attention_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception

INSTRUMENTS = ('Thermometer', 'Census', 'Air gauge', 'Tide gauge')


def _held(context, spec, initial=''):
  """An instrument the World GM runs, as its entity holds it."""
  perception.deliver(context, spec)
  return perception.subscribe(
      context,
      perception.reading(spec.owner, spec.label, initial, topic=spec.topics[0]),
  )


class AttentionTest(absltest.TestCase):

  def test_watches_the_first_instruments_and_follows_the_slots(self):
    slots = [2]
    attention = attention_lib.Attention(INSTRUMENTS, lambda: slots[0])
    self.assertEqual(attention.watched(), ['Thermometer', 'Census'])
    self.assertFalse(attention.watching('Tide gauge'))
    self.assertTrue(attention.watching('Omega meter'))  # Not limited.
    slots[0] = 3
    self.assertEqual(
        attention.watched(), ['Thermometer', 'Census', 'Air gauge']
    )
    slots[0] = 1
    self.assertEqual(attention.watched(), ['Thermometer'])
    unlimited = attention_lib.Attention(INSTRUMENTS, lambda: None)
    self.assertEqual(unlimited.watched(), list(INSTRUMENTS))
    self.assertEqual(unlimited.options(), {})

  def test_swap_and_state(self):
    attention = attention_lib.Attention(INSTRUMENTS, lambda: 2)
    options = attention.options()
    self.assertLen(options, 4)
    self.assertEqual(
        options['watch the tide gauge instead of the census'],
        ('Tide gauge', 'Census'),
    )
    self.assertTrue(attention.swap('Tide gauge', 'Census'))
    self.assertEqual(attention.watched(), ['Thermometer', 'Tide gauge'])
    self.assertFalse(attention.swap('Thermometer', 'Tide gauge'))
    self.assertFalse(attention.swap('Census', 'Air gauge'))
    resumed = attention_lib.Attention(INSTRUMENTS, lambda: 2)
    resumed.set_state(attention.get_state())
    self.assertEqual(resumed.watched(), attention.watched())
    self.assertEqual(resumed.refocused, 1)

  def test_steward_refocuses_without_spending_omega(self):
    context = game_lib.new_game('great_oxidation')
    attention = attention_lib.Attention(INSTRUMENTS, context.attention_slots)
    steward = actors_lib.StewardActor('Gaia', context.planet, attention)
    option = 'watch the air gauge instead of the census'
    self.assertIn(option, steward.options())
    omega = context.planet.omega
    self.assertIn('now watches the air gauge', steward.resolve(option))
    self.assertEqual(context.planet.omega, omega)
    self.assertTrue(attention.watching('Air gauge'))

  def test_unwatched_instrument_receives_nothing(self):
    context = game_lib.new_game('great_oxidation')
    context.planet.instrument_tier = len(context.planet.instrument_tiers) - 1
    slots = [1]
    attention = attention_lib.Attention(
        ('Thermometer', 'Air gauge'), lambda: slots[0]
    )
    gauge = _held(
        context,
        perception.instrument(
            context,
            'cycles.carbon',
            owner='Gaia',
            label='Air gauge',
            transported=True,
            attention=attention,
        ),
        initial='none',
    )
    context.clock.advance()
    context.publish_state()
    self.assertEqual(gauge.get_current_value(), 'none')
    self.assertNotIn(('Gaia', 'cycles.carbon'), context.beliefs)
    slots[0] = 2
    context.clock.advance()
    context.publish_state()
    self.assertStartsWith(gauge.get_current_value(), 'CO2 x')
    self.assertIn('Air: oxygen', gauge.get_current_value())
    slots[0] = 1
    context.clock.advance()
    context.publish_state()
    self.assertStartsWith(gauge.get_current_value(), 'Not watched')


if __name__ == '__main__':
  absltest.main()
