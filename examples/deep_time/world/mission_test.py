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

"""Tests for Mission Control and the light-delay link to the colony."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time import perception
from examples.deep_time.prefabs import mission


def _game():
  context = game_lib.new_game('light_delay')
  colony = context.planet.colony
  assert colony is not None
  return context, context.planet, colony


def _held(context, spec):
  """An instrument the World GM runs, as its entity holds it."""
  perception.deliver(context, spec)
  return perception.subscribe(context, spec)


class MissionTest(absltest.TestCase):

  def test_designs_arrive_after_the_light_delay(self):
    _, planet, colony = _game()
    control = actors_lib.MissionActor('Mission Control', planet)
    option = 'transmit the design for fabricating solar panels'
    self.assertIn(option, control.options())
    text = control.resolve(option)
    self.assertIn('reaches New Ararat in year 4', text)
    self.assertEqual(
        control.drain_outbox(),
        [(
            'mission.uplink',
            'The design for fabricating solar panels, sent in year 0.',
        )],
    )
    self.assertNotIn(option, control.options())  # Already on its way.
    self.assertEqual(colony.incoming, [(4, 'design', 'fabrication')])
    for _ in range(3):
      planet.advance()
      self.assertNotIn('fabrication', colony.designs)
    report = planet.advance()  # Tick 4: exactly the light delay.
    self.assertEqual(planet.tick, 4)
    self.assertIn('fabrication', colony.designs)
    self.assertIn(
        'the design for fabricating solar panels', report.events[0].text
    )

  def test_directives_reach_the_colonists_sheet(self):
    _, planet, colony = _game()
    control = actors_lib.MissionActor('Mission Control', planet)
    control.resolve('tell the colonists: put more labor into mine ice')
    for _ in range(5):
      planet.advance()
    self.assertIn(
        'The latest directive from home: put more labor into mine ice.',
        colony.sheet(),
    )

  def test_home_reads_the_colony_years_late(self):
    context, planet, _ = _game()
    relay = _held(
        context,
        perception.instrument(
            context,
            'colony.status',
            owner='Mission Control',
            label='Colony relay',
            initial='nothing yet',
            transported=True,
            channel=lambda: mission.light_delay(context),
        ),
    )
    for _ in range(6):
      context.clock.advance()
      context.wire.advance()
      planet.advance()
      context.publish_state()
    self.assertIn('New Ararat (generation 2)', relay.get_current_value())
    self.assertIn('4 ticks old', relay.get_current_value())

  def test_mock_mission_sends_designs_first_then_one_directive(self):
    _, planet, _ = _game()
    control = actors_lib.MissionActor('Mission Control', planet)
    options = {chr(97 + i): o for i, o in enumerate(control.options())}
    policy = mock_language_model.mission_policy
    self.assertEqual(
        policy('', options), 'transmit the design for fabricating solar panels'
    )
    without = {k: v for k, v in options.items() if 'design' not in v}
    relay = 'Colony relay: New Ararat (generation 3). Short of: air (40%).'
    self.assertEqual(
        policy(relay, without),
        'tell the colonists: put more labor into split water for air',
    )
    busy = relay + ' On its way: directive "put more labor into mine ice."'
    self.assertEqual(policy(busy, without), actors_lib.LISTEN)


if __name__ == '__main__':
  absltest.main()
