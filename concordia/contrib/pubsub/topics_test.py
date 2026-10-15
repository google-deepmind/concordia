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

"""Tests for the TopicBus."""

import threading

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.agent import pubsub
from concordia.components.game_master import interrupt_scheduling
from concordia.contrib.pubsub import topics
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
from reactivex import operators as ops


class _NoOpAct(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del contexts, action_spec
    return 'ok'

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _latest(bus: topics.TopicBus, topic: str) -> topics.Message:
  message = bus.latest(topic)
  assert message is not None
  return message


class TopicBusTest(absltest.TestCase):

  def test_publish_stamps_sequence_time_and_source(self):
    now = [3.0]
    bus = topics.TopicBus(clock=lambda: now[0])
    first = bus.publish('climate.global', 'warm', source='climate')
    now[0] = 4.0
    second = bus.publish('climate.global', 'hot', source='climate')
    self.assertEqual(first.seq, 1)
    self.assertEqual(second.seq, 2)
    self.assertEqual(first.time, 3.0)
    self.assertEqual(second.time, 4.0)
    self.assertEqual(second.source, 'climate')
    self.assertEqual(_latest(bus, 'climate.global'), second)
    self.assertEqual(bus.message_count('climate.global'), 2)

  def test_latest_retention_replays_last_message_to_late_subscriber(self):
    bus = topics.TopicBus()
    bus.publish('a', 'one')
    bus.publish('a', 'two')
    received = []
    bus.values('a').subscribe(received.append)
    self.assertEqual(received, ['two'])

  def test_replay_retention_replays_buffer(self):
    bus = topics.TopicBus(
        topics=[
            topics.TopicSpec(
                'geo.events', retention=topics.Retention.REPLAY, replay_size=2
            )
        ]
    )
    for text in ('one', 'two', 'three'):
      bus.publish('geo.events', text)
    received = []
    bus.values('geo.events').subscribe(received.append)
    self.assertEqual(received, ['two', 'three'])
    self.assertEqual(
        [m.payload for m in bus.history('geo.events')], ['two', 'three']
    )

  def test_none_retention_is_live_only(self):
    bus = topics.TopicBus(default_retention=topics.Retention.NONE)
    bus.publish('a', 'missed')
    received = []
    bus.values('a').subscribe(received.append)
    bus.publish('a', 'seen')
    self.assertEqual(received, ['seen'])
    self.assertEqual(_latest(bus, 'a').payload, 'seen')

  def test_matching_filters_by_prefix(self):
    bus = topics.TopicBus()
    received = []
    bus.matching('climate.').subscribe(lambda m: received.append(m.topic))
    bus.publish('climate.region.1', 'x')
    bus.publish('bio.census', 'y')
    bus.publish('climate.global', 'z')
    self.assertEqual(received, ['climate.region.1', 'climate.global'])

  def test_invalid_topic_name_raises(self):
    bus = topics.TopicBus()
    with self.assertRaises(ValueError):
      bus.declare('has space')
    with self.assertRaises(ValueError):
      bus.declare('')

  def test_redeclare_keeps_spec_and_fills_description(self):
    bus = topics.TopicBus()
    bus.declare('a', retention=topics.Retention.NONE)
    spec = bus.declare('a', description='later')
    self.assertEqual(spec.retention, topics.Retention.NONE)
    self.assertEqual(spec.description, 'later')

  def test_make_subscriber_tracks_topic_with_initial_value(self):
    bus = topics.TopicBus()
    subscriber = bus.make_subscriber(
        'climate.global', subscriber_name='Alice', initial_value='unknown'
    )
    self.assertIsInstance(subscriber, pubsub.Subscriber)
    self.assertEqual(subscriber.get_current_value(), 'unknown')
    self.assertEqual(subscriber.get_pre_act_label(), 'climate.global')
    bus.publish('climate.global', '21.5')
    self.assertEqual(subscriber.get_current_value(), '21.5')
    self.assertEqual(bus.topology()['climate.global']['subscribers'], ['Alice'])

  def test_make_subscriber_render_and_pipeline(self):
    bus = topics.TopicBus(clock=lambda: 7)
    seen = []
    subscriber = bus.make_subscriber(
        't',
        render=lambda m: f'{m.payload}@{m.time:g}',
        pipeline=[ops.distinct_until_changed(), ops.do_action(seen.append)],
    )
    bus.publish('t', 'a')
    bus.publish('t', 'a')
    bus.publish('t', 'b')
    self.assertEqual(subscriber.get_current_value(), 'b@7')
    self.assertEqual(seen, ['a@7', 'b@7'])

  def test_make_subscriber_merges_several_topics(self):
    bus = topics.TopicBus()
    bus.publish('geo.events', 'old quake')
    digest = []
    subscriber = bus.make_subscriber(
        ['geo.events', 'bio.events'],
        subscriber_name='Gaia',
        pipeline=[ops.do_action(digest.append)],
    )
    self.assertEqual(subscriber.get_pre_act_label(), 'geo.events, bio.events')
    bus.publish('bio.events', 'blight')
    bus.publish('geo.events', 'eruption')
    self.assertEqual(digest, ['old quake', 'blight', 'eruption'])
    self.assertEqual(subscriber.get_current_value(), 'eruption')
    self.assertEqual(bus.topology()['bio.events']['subscribers'], ['Gaia'])
    with self.assertRaises(ValueError):
      bus.make_subscriber([])

  def test_make_publisher_forwards_completed_values(self):
    bus = topics.TopicBus()
    done = threading.Event()
    publisher = bus.make_publisher(
        'chronicle', lambda: 'Chapter 1', source='Chronicler'
    )
    bus.values('chronicle').subscribe(lambda _: done.set())
    agent = entity_agent.EntityAgent(
        agent_name='Chronicler',
        act_component=_NoOpAct(),
        context_components={'chronicle': publisher},
    )
    agent.act()
    self.assertTrue(done.wait(timeout=5))
    self.assertEqual(_latest(bus, 'chronicle').payload, 'Chapter 1')
    self.assertEqual(_latest(bus, 'chronicle').source, 'Chronicler')
    self.assertEqual(bus.topology()['chronicle']['publishers'], ['Chronicler'])
    publisher.close()

  def test_subscriber_can_publish_reentrantly(self):
    bus = topics.TopicBus()

    def echo(value: str) -> None:
      bus.publish('out', value.upper())

    bus.values('in').subscribe(echo)
    bus.publish('in', 'echo')
    self.assertEqual(_latest(bus, 'out').payload, 'ECHO')
    self.assertLess(_latest(bus, 'in').seq, _latest(bus, 'out').seq)

  def test_measurements_tap(self):
    measurements = measurements_lib.Measurements()
    bus = topics.TopicBus(measurements=measurements)
    bus.publish('a.b', 'x', source='s')
    datum = measurements.get_last_datum('topic/a.b')
    self.assertEqual(datum['payload'], 'x')
    self.assertEqual(datum['source'], 's')

  def test_topology_dumps(self):
    bus = topics.TopicBus()
    bus.register_publisher('climate.global', 'climate model')
    bus.register_subscriber('climate.global', 'Daisy "white"')
    mermaid = bus.to_mermaid()
    self.assertIn('flowchart LR', mermaid)
    self.assertIn('climate.global', mermaid)
    self.assertIn("Daisy 'white'", mermaid)
    dot = bus.to_dot()
    self.assertIn('"climate model" -> "climate.global";', dot)
    self.assertEqual(bus.stats(), {'climate.global': 0})

  def test_state_round_trip(self):
    bus = topics.TopicBus(clock=lambda: 1.5)
    bus.declare('r', retention=topics.Retention.REPLAY, replay_size=3)
    for text in ('a', 'b'):
      bus.publish('r', text)
    bus.publish('l', 'z')
    state = bus.get_state()

    restored = topics.TopicBus()
    firehose = []
    restored.matching().subscribe(firehose.append)
    restored.set_state(state)
    self.assertEqual([m.payload for m in restored.history('r')], ['a', 'b'])
    self.assertEqual(_latest(restored, 'l').payload, 'z')
    self.assertEqual(restored.message_count('r'), 2)
    self.assertEqual(restored.spec('r').retention, topics.Retention.REPLAY)
    self.assertEqual(firehose, [])
    self.assertEqual(restored.publish('l', 'next').seq, 4)

  def test_as_interrupt_event_matches_masks(self):
    bus = topics.TopicBus(clock=lambda: 12.0)
    message = bus.publish('geo.events', 'eruption', source='lithosphere')
    event = topics.as_interrupt_event(message)
    self.assertIsInstance(event, interrupt_scheduling.Event)
    self.assertEqual(event.timestamp, 12.0)
    self.assertEqual(event.tag, 'geo.events')
    self.assertEqual(event.source, 'lithosphere')
    self.assertEqual(event.description, 'eruption')
    self.assertTrue(
        interrupt_scheduling.InterruptMask(prefixes=('geo.',)).matches(event)
    )
    self.assertFalse(
        interrupt_scheduling.InterruptMask(prefixes=('bio.',)).matches(event)
    )

  def test_close_completes_streams_and_rejects_publish(self):
    bus = topics.TopicBus()
    completed = []
    bus.observable('a').subscribe(on_completed=lambda: completed.append('a'))
    bus.close()
    bus.close()
    self.assertEqual(completed, ['a'])
    with self.assertRaises(RuntimeError):
      bus.publish('a', 'x')

  def test_concurrent_publishers_get_unique_sequence_numbers(self):
    bus = topics.TopicBus()

    def worker(i):
      for j in range(50):
        bus.publish(f't.{i}', str(j))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    self.assertEqual(sum(bus.stats().values()), 200)
    self.assertEqual(bus.publish('t.0', 'last').seq, 201)


if __name__ == '__main__':
  absltest.main()
