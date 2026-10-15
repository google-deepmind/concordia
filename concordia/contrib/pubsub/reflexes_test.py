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

"""Tests for declarative reflexes."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from concordia.contrib.pubsub import reflexes
from concordia.contrib.pubsub import topics

SELL_ORE = {
    'name': 'sell_ore',
    'input': 'census',
    'pipeline': [
        {'op': 'json_field', 'path': 'ore'},
        {'op': 'threshold', 'above': 6},
    ],
    'emit': {'topic': 'orders', 'template': 'sell ore ({value})'},
    'cooldown': 2,
}


class _World:

  def __init__(self):
    self.tick = 0
    self.bus = topics.TopicBus(clock=lambda: self.tick)
    self.orders: list[str] = []
    self.bus.observable('orders').subscribe(
        lambda m: self.orders.append(m.payload)
    )

  def census(self, **values) -> None:
    self.bus.publish('census', json.dumps(values))


class ReflexSpecTest(parameterized.TestCase):

  def test_round_trip(self):
    spec = reflexes.ReflexSpec.from_dict(SELL_ORE)
    self.assertEqual(reflexes.ReflexSpec.from_dict(spec.to_dict()), spec)

  @parameterized.parameters(
      ({'pipeline': [{'op': 'eval', 'code': '1'}]}, 'unknown operator'),
      ({'pipeline': [{'op': 'threshold'}]}, 'above'),
      ({'pipeline': [{'op': 'scale', 'factor': 2, 'x': 1}]}, 'unexpected'),
      ({'pipeline': [{'op': 'moving_average', 'n': 0}]}, 'n >= 1'),
      ({'emit': {}}, 'emit.topic'),
      ({'name': ''}, 'name'),
      ({'cooldown': -1}, 'non-negative'),
      ({'colour': 'red'}, 'unknown reflex fields'),
  )
  def test_rejects_malformed_reflexes(self, change, message):
    with self.assertRaisesRegex(ValueError, message):
      reflexes.ReflexSpec.from_dict({**SELL_ORE, **change})

  def test_allowed_operators(self):
    with self.assertRaisesRegex(ValueError, 'not allowed'):
      reflexes.ReflexSpec.from_dict(SELL_ORE, allowed_ops={'json_field'})
    reflexes.ReflexSpec.from_dict(
        SELL_ORE, allowed_ops={'json_field', 'threshold'}
    )

  def test_parse_specs_from_prose(self):
    text = (
        'Here are my standing orders: ['
        + json.dumps(SELL_ORE)
        + ', {"name": "bad", "input": "x", "pipeline": [{"op": "sql"}],'
        ' "emit": {"topic": "y"}}] Hope that helps.'
    )
    specs, errors = reflexes.parse_specs(text)
    self.assertEqual([s.name for s in specs], ['sell_ore'])
    self.assertLen(errors, 1)
    self.assertEqual(reflexes.parse_specs('no json'), ([], ['no JSON found']))


class ReflexesTest(absltest.TestCase):

  def test_threshold_fires_with_cooldown_and_stats(self):
    world = _World()
    engine = reflexes.Reflexes(world.bus)
    engine.install(reflexes.ReflexSpec.from_dict(SELL_ORE), owner='Basalt')
    for tick, ore in enumerate([3, 7, 8, 9, 2, 10]):
      world.tick = tick
      world.census(ore=ore)
    # Fires at tick 1 (7) and tick 3 (9); tick 2 is cooling down, tick 5
    # (10) fires again.
    self.assertEqual(
        world.orders, ['sell ore (7)', 'sell ore (9)', 'sell ore (10)']
    )
    stats = engine.stats('sell_ore')
    self.assertEqual((stats['fires'], stats['last_fired']), (3, 5))
    self.assertEqual(stats['owner'], 'Basalt')
    self.assertIn(
        'sell_ore: on census -> orders (fired 3 times)',
        engine.describe('Basalt'),
    )
    self.assertEqual(world.bus.topology()['orders']['publishers'], ['Basalt'])

  def test_numeric_operators(self):
    world = _World()
    engine = reflexes.Reflexes(world.bus)
    engine.install(
        reflexes.ReflexSpec.from_dict({
            'name': 'trend',
            'input': 'census',
            'pipeline': [
                {'op': 'json_field', 'path': 'grain'},
                {'op': 'moving_average', 'n': 2},
                {'op': 'change'},
                {'op': 'scale', 'factor': 10},
                {'op': 'threshold', 'below': 0},
                {'op': 'every', 'n': 1},
            ],
            'emit': {'topic': 'orders', 'template': '{name}: falling {value}'},
        })
    )
    for grain in (10, 10, 6, 2):
      world.census(grain=grain)
    self.assertEqual(world.orders, ['trend: falling -20', 'trend: falling -40'])

  def test_remove_and_replace(self):
    world = _World()
    engine = reflexes.Reflexes(world.bus)
    spec = reflexes.ReflexSpec.from_dict(SELL_ORE)
    engine.install(spec)
    engine.install(spec)  # Replaces, does not duplicate.
    world.census(ore=9)
    self.assertLen(world.orders, 1)
    self.assertTrue(engine.remove('sell_ore'))
    self.assertFalse(engine.remove('sell_ore'))
    world.tick = 10
    world.census(ore=9)
    self.assertLen(world.orders, 1)
    self.assertEqual(engine.describe(), 'No reflexes.')

  def test_state_reinstalls(self):
    world = _World()
    engine = reflexes.Reflexes(world.bus)
    engine.install(reflexes.ReflexSpec.from_dict(SELL_ORE), owner='Basalt')
    world.census(ore=9)
    state = engine.get_state()
    engine.close()
    restored = reflexes.Reflexes(world.bus)
    restored.set_state(state)
    self.assertEqual(restored.stats('sell_ore')['fires'], 1)
    world.tick = 5
    world.census(ore=9)
    self.assertEqual(restored.stats('sell_ore')['fires'], 2)
    self.assertLen(world.orders, 2)

  def test_state_preserves_pipeline_windows_and_bus_checkpoint(self):
    world1 = _World()
    engine1 = reflexes.Reflexes(world1.bus)
    spec = reflexes.ReflexSpec.from_dict({
        'name': 'avg_drop',
        'input': 'census',
        'pipeline': [
            {'op': 'json_field', 'path': 'grain'},
            {'op': 'moving_average', 'n': 2},
            {'op': 'change'},
            {'op': 'every', 'n': 2},
        ],
        'emit': {'topic': 'orders', 'template': 'delta={value}'},
    })
    engine1.install(spec, owner='Granary')
    # Feed two messages: moving_average emits 10 then 8; change emits -2 (item 1
    # for `every(n=2)`, so it has not fired yet).
    world1.census(grain=10)
    world1.census(grain=6)
    self.assertEmpty(world1.orders)

    bus_state = json.loads(json.dumps(world1.bus.get_state()))
    world2 = _World()
    engine2 = reflexes.Reflexes(world2.bus)
    world2.bus.set_state(bus_state)

    for w in (world1, world2):
      w.census(grain=2)
    self.assertEqual(world1.orders, ['delta=-4'])
    self.assertEqual(world2.orders, world1.orders)
    self.assertEqual(engine2.stats('avg_drop'), engine1.stats('avg_drop'))


if __name__ == '__main__':
  absltest.main()
