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

"""Tests for the pub/sub belief operators."""

import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import sim_time
from concordia.contrib.pubsub import topics
from reactivex import operators as ops
from reactivex import subject as reactivex_subject


def _collect(observable) -> list[Any]:
  out = []
  observable.subscribe(out.append)
  return out


class JsonFieldTest(absltest.TestCase):

  def test_extracts_nested_fields_and_skips_or_defaults(self):
    source = reactivex_subject.Subject()
    skipped = _collect(source.pipe(operators.json_field('cover.white')))
    defaulted = _collect(source.pipe(operators.json_field('a.1', default=-1)))
    source.on_next('{"cover": {"white": 0.4}, "a": [1, 2]}')
    source.on_next('not json')
    source.on_next(topics.Message('t', '{"cover": {"white": 0.5}}', seq=1))
    self.assertEqual(skipped, [0.4, 0.5])
    self.assertEqual(defaulted, [2, -1, -1])

  def test_lookup_errors(self):
    with self.assertRaises(KeyError):
      operators.lookup({'a': [1]}, 'a.5')
    with self.assertRaises(KeyError):
      operators.lookup({'a': 1}, 'b')
    self.assertEqual(operators.lookup({'a': 1}, ''), {'a': 1})


class WithAgeTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, 'age unknown'),
      (0, 'fresh'),
      (1, '1 tick old'),
      (3, '3 ticks old'),
      (2.5, '2.5 ticks old'),
  )
  def test_age_text(self, age, expected):
    self.assertEqual(operators.age_text(age), expected)

  def test_belief_ages_as_the_clock_advances(self):
    now = reactivex_subject.Subject()
    readings = reactivex_subject.Subject()
    beliefs = _collect(readings.pipe(operators.with_age(now)))
    readings.on_next(topics.Message('climate', '21 C', seq=1, time=5))
    now.on_next(5)
    now.on_next(6)
    now.on_next(8)
    readings.on_next(topics.Message('climate', '22 C', seq=2, time=8))
    self.assertEqual(
        beliefs,
        [
            '21 C [age unknown]',
            '21 C [fresh]',
            '21 C [1 tick old]',
            '21 C [3 ticks old]',
            '22 C [fresh]',
        ],
    )

  def test_template_and_render(self):
    now = reactivex_subject.BehaviorSubject(10)
    readings = reactivex_subject.Subject()
    beliefs = _collect(
        readings.pipe(
            operators.with_age(
                now,
                render=lambda m: m.payload.upper(),
                template='{text} (tick {time}, {age_value} old)',
            )
        )
    )
    readings.on_next(topics.Message('x', 'calm', seq=1, time=7))
    self.assertEqual(beliefs, ['CALM (tick 7, 3 old)'])

  def test_with_topic_bus_subscriber(self):
    tick = [0]
    bus = topics.TopicBus(clock=lambda: tick[0])
    ticks = bus.observable('world.tick').pipe(ops.map(lambda m: m.time))
    belief = bus.make_subscriber(
        'climate.global',
        transform=operators.with_age(ticks),
        initial_value='no reading',
    )
    self.assertEqual(belief.get_current_value(), 'no reading')
    bus.publish('world.tick', '0')
    bus.publish('climate.global', '20 C')
    tick[0] = 2
    bus.publish('world.tick', '2')
    self.assertEqual(belief.get_current_value(), '20 C [2 ticks old]')


class DigestDeadbandCombineTest(absltest.TestCase):

  def test_rolling_digest_keeps_last_n(self):
    source = reactivex_subject.Subject()
    digests = _collect(source.pipe(operators.rolling_digest(2, bullet='* ')))
    for text in ('a', 'b', 'c'):
      source.on_next(text)
    self.assertEqual(digests, ['* a', '* a\n* b', '* b\n* c'])
    with self.assertRaises(ValueError):
      operators.rolling_digest(0)

  def test_rolling_digest_renders_messages(self):
    source = reactivex_subject.Subject()
    digests = _collect(
        source.pipe(
            operators.rolling_digest(
                3, render=lambda m: f'{m.time:g}: {m.payload}'
            )
        )
    )
    source.on_next(topics.Message('e', 'blight', seq=1, time=9))
    self.assertEqual(digests, ['- 9: blight'])

  def test_deadband_needs_cumulative_change(self):
    source = reactivex_subject.Subject()
    emitted = _collect(source.pipe(operators.deadband(1.0)))
    for value in (20.0, 20.4, 20.8, 21.1, 21.5, 19.9, 20.5):
      source.on_next(value)
    self.assertEqual(emitted, [20.0, 21.1, 19.9])

  def test_deadband_with_key(self):
    source = reactivex_subject.Subject()
    emitted = _collect(
        source.pipe(operators.deadband(5, key=lambda item: item['t']))
    )
    for t in (0, 3, 6, 7, 12):
      source.on_next({'t': t})
    self.assertEqual([item['t'] for item in emitted], [0, 6, 12])

  def test_combine_labelled(self):
    a = reactivex_subject.Subject()
    b = reactivex_subject.Subject()
    lines = _collect(operators.combine_labelled({'temp': a, 'cover': b}))
    a.on_next('21 C')
    b.on_next('40%')
    a.on_next('22 C')
    self.assertEqual(
        lines,
        [
            'temp: ?; cover: ?',
            'temp: 21 C; cover: ?',
            'temp: 21 C; cover: 40%',
            'temp: 22 C; cover: 40%',
        ],
    )


def _message(payload: str, seq: int = 0, time: float = 0.0) -> topics.Message:
  return topics.Message('climate', payload, seq=seq, time=time)


class TransportTest(absltest.TestCase):

  def test_perfect_channel_passes_messages_through(self):
    source = reactivex_subject.Subject()
    out = _collect(source.pipe(operators.transport(operators.Transport())))
    message = _message('{"t": 1}')
    source.on_next(message)
    self.assertEqual(out, [message])

  def test_drops_are_seeded_and_reproducible(self):
    spec = operators.Transport(drop_rate=0.5)
    source = reactivex_subject.Subject()
    first = _collect(source.pipe(operators.transport(spec, seed=3)))
    second = _collect(source.pipe(operators.transport(spec, seed=3)))
    other = _collect(source.pipe(operators.transport(spec, seed=4)))
    for i in range(40):
      source.on_next(_message('{}', seq=i))
    self.assertEqual(first, second)
    self.assertNotEqual(first, other)
    self.assertBetween(len(first), 5, 35)
    lost = _collect(
        source.pipe(operators.transport(operators.Transport(drop_rate=1.0)))
    )
    source.on_next(_message('{}'))
    self.assertEmpty(lost)

  def test_bias_and_noise_touch_only_numeric_fields(self):
    source = reactivex_subject.Subject()
    biased = _collect(
        source.pipe(
            operators.transport(
                operators.Transport(
                    bias=2.0, fields=('temp', 'bands.1', 'name', 'missing')
                )
            )
        )
    )
    noisy = _collect(
        source.pipe(
            operators.transport(
                operators.Transport(noise_sigma=1.0, fields=('temp',))
            )
        )
    )
    source.on_next(
        _message('{"temp": 20, "bands": [1, 2], "name": "x", "ok": true}', 1, 7)
    )
    source.on_next(_message('not json', 2))
    data = json.loads(biased[0].payload)
    self.assertEqual(
        data, {'temp': 22.0, 'bands': [1, 4.0], 'name': 'x', 'ok': True}
    )
    self.assertEqual((biased[0].seq, biased[0].time), (1, 7))
    self.assertEqual(biased[1].payload, 'not json')
    self.assertNotEqual(json.loads(noisy[0].payload)['temp'], 20)
    self.assertLess(abs(json.loads(noisy[0].payload)['temp'] - 20), 6)

  def test_latency_delays_by_ticks_on_sim_time(self):
    scheduler = sim_time.SimTimeScheduler()
    source = reactivex_subject.Subject()
    out = _collect(
        source.pipe(
            operators.transport(
                operators.Transport(latency=2), scheduler=scheduler
            )
        )
    )
    source.on_next(_message('{"t": 0}'))
    scheduler.advance()
    source.on_next(_message('{"t": 1}', seq=1, time=1))
    self.assertEmpty(out)
    scheduler.advance()
    self.assertEqual([m.payload for m in out], ['{"t": 0}'])
    scheduler.advance()
    self.assertEqual([m.time for m in out], [0, 1])

  def test_spec_can_change_and_stale_arrivals_are_dropped(self):
    scheduler = sim_time.SimTimeScheduler()
    channel = [operators.Transport(latency=3)]
    source = reactivex_subject.Subject()
    out = _collect(
        source.pipe(
            operators.transport(lambda: channel[0], scheduler=scheduler)
        )
    )
    source.on_next(_message('{"t": 0}', seq=0))
    channel[0] = operators.Transport(latency=0)
    source.on_next(_message('{"t": 1}', seq=1))
    self.assertEqual([m.seq for m in out], [1])
    scheduler.advance(3)
    self.assertEqual([m.seq for m in out], [1])

  def test_per_message_spec_and_late_news_kept(self):
    # News that takes longer to arrive the farther away it happened.
    scheduler = sim_time.SimTimeScheduler()
    source = reactivex_subject.Subject()

    def by_distance(message):
      away = json.loads(message.payload)['away']
      return operators.Transport(latency=away, keep_late=True)

    out = _collect(
        source.pipe(
            operators.transport(
                by_distance, per_message=True, scheduler=scheduler
            )
        )
    )
    source.on_next(_message('{"away": 3}', seq=0))  # Far, published first.
    source.on_next(_message('{"away": 1}', seq=1))  # Near, published next.
    scheduler.advance()
    self.assertEqual([m.seq for m in out], [1])
    scheduler.advance(2)
    self.assertEqual([m.seq for m in out], [1, 0])  # Late, but kept.
    with self.assertRaises(ValueError):
      operators.transport(operators.Transport(), per_message=True)

  def test_per_message_spec_still_drops_late_beliefs_by_default(self):
    scheduler = sim_time.SimTimeScheduler()
    source = reactivex_subject.Subject()
    out = _collect(
        source.pipe(
            operators.transport(
                lambda m: operators.Transport(
                    latency=json.loads(m.payload)['away']
                ),
                per_message=True,
                scheduler=scheduler,
            )
        )
    )
    source.on_next(_message('{"away": 3}', seq=0))
    source.on_next(_message('{"away": 1}', seq=1))
    scheduler.advance(3)
    self.assertEqual([m.seq for m in out], [1])  # The older one arrived late.

  def test_completion_waits_for_pending_deliveries(self):
    scheduler = sim_time.SimTimeScheduler()
    source = reactivex_subject.Subject()
    events = []
    source.pipe(
        operators.transport(operators.Transport(latency=1), scheduler=scheduler)
    ).subscribe(
        lambda m: events.append(m.seq),
        on_completed=lambda: events.append('done'),
    )
    source.on_next(_message('{}', seq=5))
    source.on_completed()
    self.assertEmpty(events)
    scheduler.advance()
    self.assertEqual(events, [5, 'done'])

  def test_validation(self):
    with self.assertRaises(ValueError):
      operators.Transport(latency=-1)
    with self.assertRaises(ValueError):
      operators.Transport(drop_rate=1.5)
    with self.assertRaises(ValueError):
      operators.Transport(noise_sigma=-0.1)
    with self.assertRaises(ValueError):
      operators.transport(operators.Transport(latency=1))

  def test_stateful_operators_checkpoint_through_subscriber(self):
    tick = [0]
    bus = topics.TopicBus(
        clock=lambda: tick[0], default_retention=topics.Retention.NONE
    )
    scheduler = sim_time.SimTimeScheduler()
    ticks = bus.observable('world.tick').pipe(ops.map(lambda m: m.time))

    digest_sub = bus.make_subscriber(
        'news', transform=operators.rolling_digest(3, bullet='* ')
    )
    deadband_sub = bus.make_subscriber(
        'temp',
        transform=lambda s: s.pipe(
            operators.json_field('v'),
            operators.deadband(1.0),
            ops.map(str),
        ),
    )
    transport_sub = bus.make_subscriber(
        'climate',
        transform=lambda s: s.pipe(
            operators.transport(
                operators.Transport(latency=2, noise_sigma=0.5, fields=('t',)),
                seed=11,
                scheduler=scheduler,
            ),
            operators.with_age(ticks),
        ),
        initial_value='none',
    )

    bus.publish('world.tick', '0')
    bus.publish('news', 'first')
    bus.publish('news', 'second')
    bus.publish('temp', '{"v": 20.0}')
    bus.publish('temp', '{"v": 20.4}')  # Within deadband of 20.0.
    bus.publish(
        'climate', '{"t": 15.0}'
    )  # In flight on scheduler (due tick 2).
    tick[0] = 1
    scheduler.advance()
    bus.publish('world.tick', '1')

    # Save JSON-serialized states mid-flight at tick 1.
    bus_state = json.loads(json.dumps(bus.get_state()))
    digest_state = json.loads(json.dumps(digest_sub.get_state()))
    deadband_state = json.loads(json.dumps(deadband_sub.get_state()))
    transport_state = json.loads(json.dumps(transport_sub.get_state()))

    # Restore into a fresh bus and scheduler at tick 1.
    tick2 = [1]
    bus2 = topics.TopicBus(
        clock=lambda: tick2[0], default_retention=topics.Retention.NONE
    )
    scheduler2 = sim_time.SimTimeScheduler()
    ticks2 = bus2.observable('world.tick').pipe(ops.map(lambda m: m.time))
    digest_sub2 = bus2.make_subscriber(
        'news', transform=operators.rolling_digest(3, bullet='* ')
    )
    deadband_sub2 = bus2.make_subscriber(
        'temp',
        transform=lambda s: s.pipe(
            operators.json_field('v'),
            operators.deadband(1.0),
            ops.map(str),
        ),
    )
    transport_sub2 = bus2.make_subscriber(
        'climate',
        transform=lambda s: s.pipe(
            operators.transport(
                operators.Transport(latency=2, noise_sigma=0.5, fields=('t',)),
                seed=11,
                scheduler=scheduler2,
            ),
            operators.with_age(ticks2),
        ),
        initial_value='none',
    )
    bus2.set_state(bus_state)
    digest_sub2.set_state(digest_state)
    deadband_sub2.set_state(deadband_state)
    transport_sub2.set_state(transport_state)

    # Advance both worlds identically and verify identical outputs.
    for bus_i, sched_i, t_ref in (
        (bus, scheduler, tick),
        (bus2, scheduler2, tick2),
    ):
      bus_i.publish('news', 'third')
      bus_i.publish('temp', '{"v": 21.1}')  # Crosses 1.0 from 20.0, not 20.4.
      t_ref[0] = 2
      sched_i.advance()
      bus_i.publish('world.tick', '2')
      bus_i.publish('climate', '{"t": 18.0}')
      t_ref[0] = 4
      sched_i.advance(2)
      bus_i.publish('world.tick', '4')

    self.assertEqual(
        digest_sub2.get_current_value(),
        '* first\n* second\n* third',
    )
    self.assertEqual(
        digest_sub2.get_current_value(), digest_sub.get_current_value()
    )
    self.assertEqual(deadband_sub2.get_current_value(), '21.1')
    self.assertEqual(
        transport_sub2.get_current_value(), transport_sub.get_current_value()
    )


if __name__ == '__main__':
  absltest.main()
