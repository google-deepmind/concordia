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

"""Tests for topic observations."""

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.contrib.components.game_master import topic_observations
from concordia.contrib.pubsub import topics
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class _EchoAct(entity_component.ActingComponent):

  def get_action_attempt(self, contexts, action_spec):
    del action_spec
    return '\n'.join(v for v in contexts.values() if v)

  def get_state(self):
    return {}

  def set_state(self, state):
    del state


def _setup(**kwargs):
  bus = topics.TopicBus(clock=lambda: 3)
  queue = make_observation.ObservationQueue()
  routes = topic_observations.TopicObservations(bus, queue, **kwargs)
  return bus, queue, routes


class TopicObservationsTest(absltest.TestCase):

  def test_routes_by_access_list(self):
    bus, queue, routes = _setup(
        access={'Alice': ('weather.',), 'Bob': ('weather.', 'market.')}
    )
    bus.publish('weather.rain', 'It rains.')
    bus.publish('market.grain', 'Grain is dear.')
    bus.publish('secret.plans', 'Nobody hears this.')
    self.assertEqual(queue.get_and_clear('Alice'), ['It rains.'])
    self.assertEqual(
        queue.get_and_clear('Bob'), ['It rains.', 'Grain is dear.']
    )
    self.assertEqual(routes.delivered(), {'Alice': 1, 'Bob': 2})
    self.assertEqual(routes.listeners('weather.rain'), ['Alice', 'Bob'])

  def test_everything_nothing_grant_and_revoke(self):
    bus, queue, routes = _setup(access={'Chronicler': ('',), 'Hermit': ()})
    routes.grant('Spy', 'secret.')
    bus.publish('secret.plans', 'Attack at dawn.')
    self.assertEqual(queue.get_and_clear('Chronicler'), ['Attack at dawn.'])
    self.assertEqual(queue.get_and_clear('Spy'), ['Attack at dawn.'])
    self.assertEqual(queue.get_and_clear('Hermit'), [])
    routes.revoke('Spy', 'secret.')
    routes.grant('Spy', 'secret.')
    routes.grant('Spy', 'secret.')
    self.assertEqual(routes.access('Spy'), ('secret.',))
    routes.revoke('Spy', 'secret.')
    bus.publish('secret.plans', 'Retreat.')
    self.assertEqual(queue.get_and_clear('Spy'), [])

  def test_renderers_by_longest_prefix(self):
    bus, queue, _ = _setup(
        access={'A': ('',)},
        render={
            'geo.': lambda m: f'[geo] {m.payload}',
            'geo.events': lambda m: f'[event at {m.time:g}] {m.payload}',
        },
    )
    bus.publish('geo.events', 'Eruption!')
    bus.publish('geo.plates', 'Drift.')
    bus.publish('bio.census', '40%')
    self.assertEqual(
        queue.get_and_clear('A'),
        ['[event at 3] Eruption!', '[geo] Drift.', '40%'],
    )

  def test_empty_renders_are_skipped(self):
    bus, queue, _ = _setup(access={'A': ('',)}, render=lambda m: '')
    bus.publish('x', 'y')
    self.assertEqual(queue.get_and_clear('A'), [])

  def test_game_master_events_are_published_and_shown(self):
    bus, queue, routes = _setup(
        access={'Alice': ('events.',)}, event_topic='events.resolved'
    )
    gm = entity_agent.EntityAgent(
        agent_name='World',
        act_component=_EchoAct(),
        context_components={
            topic_observations.DEFAULT_TOPIC_OBSERVATIONS_COMPONENT_KEY: routes
        },
    )
    gm.observe(f'{event_resolution.PUTATIVE_EVENT_TAG} Alice: waves')
    gm.observe(f'{event_resolution.EVENT_TAG} Alice waved at the crowd.')
    latest = bus.latest('events.resolved')
    assert latest is not None
    self.assertEqual(latest.payload, 'Alice waved at the crowd.')
    self.assertEqual(latest.source, 'World')
    self.assertEqual(
        queue.get_and_clear('Alice'), ['Alice waved at the crowd.']
    )
    context = gm.act(entity_lib.free_action_spec(call_to_action='?'))
    self.assertIn('Alice hears: events.', context)

  def test_feeds_make_observation(self):
    bus, queue, _ = _setup(access={'Alice': ('weather.',)})
    observation = make_observation.MakeObservation(
        model=no_language_model.NoLanguageModel(),
        player_names=['Alice'],
        external_queue=queue,
        allow_llm_fallback=False,
    )
    gm = entity_agent.EntityAgent(
        agent_name='GM',
        act_component=_EchoAct(),
        context_components={'__make_observation__': observation},
    )
    bus.publish('weather.rain', 'It rains.')
    result = gm.act(
        entity_lib.ActionSpec(
            call_to_action=make_observation.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(
                name='Alice'
            ),
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    self.assertIn('It rains.', result)

  def test_state_round_trip_and_close(self):
    bus, queue, routes = _setup(access={'A': ('x.',)})
    bus.publish('x.1', 'one')
    restored = topic_observations.TopicObservations(bus, queue)
    restored.set_state(routes.get_state())
    self.assertEqual(restored.access('A'), ('x.',))
    self.assertEqual(restored.delivered(), {'A': 1})
    routes.close()
    queue.get_and_clear('A')
    bus.publish('x.2', 'two')
    self.assertEqual(queue.get_and_clear('A'), ['two'])
    restored.close()


if __name__ == '__main__':
  absltest.main()
