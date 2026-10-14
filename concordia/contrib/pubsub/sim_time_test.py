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

"""Tests for the sim-time scheduler."""

import datetime
import threading

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.components.agent import pubsub
from concordia.contrib.pubsub import sim_time
from concordia.contrib.pubsub import topics
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import reactivex
from reactivex import operators as ops


class _NoOpAct(entity_component.ActingComponent):

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return ' | '.join(f'{k}={v}' for k, v in sorted(contexts.items()) if v)

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _record(log: list[str], text: str):
  def action(scheduler, state):
    del scheduler, state
    log.append(text)

  return action


class SimTimeSchedulerTest(absltest.TestCase):

  def test_work_runs_only_when_advanced(self):
    sim = sim_time.SimTimeScheduler()
    log = []
    sim.schedule(_record(log, 'now'))
    sim.schedule_relative(2, _record(log, 'in two'))
    self.assertEqual(log, [])
    self.assertEqual(sim.pending(), 2)
    self.assertEqual(sim.advance(), 1)
    self.assertEqual(log, ['now'])
    self.assertEqual(sim.tick, 1)
    sim.advance()
    self.assertEqual(log, ['now', 'in two'])
    self.assertEqual(sim.pending(), 0)
    self.assertEqual(sim.executed, 2)

  def test_ties_are_ordered_by_key_then_submission(self):
    sim = sim_time.SimTimeScheduler()
    log = []
    sim.schedule_at(1, _record(log, 'zeta-1'), key='zeta')
    sim.schedule_at(1, _record(log, 'alpha-1'), key='alpha')
    sim.schedule_at(1, _record(log, 'zeta-2'), key='zeta')
    sim.schedule_at(0, _record(log, 'early'), key='zzz')
    sim.advance()
    self.assertEqual(log, ['early', 'alpha-1', 'zeta-1', 'zeta-2'])

  def test_now_inside_action_is_the_due_tick(self):
    sim = sim_time.SimTimeScheduler(start_tick=10)
    seen = []
    sim.schedule_relative(
        3, lambda scheduler, state: seen.append(scheduler.now)
    )
    sim.advance(5)
    self.assertEqual(seen, [sim.to_datetime(13.0)])
    self.assertEqual(sim.tick, 15)

  def test_absolute_relative_and_timedelta_times(self):
    sim = sim_time.SimTimeScheduler()
    log = []
    sim.schedule_absolute(4.0, _record(log, 'abs'))
    sim.schedule_relative(datetime.timedelta(seconds=2.5), _record(log, 'td'))
    sim.advance(2)
    self.assertEqual(log, [])
    sim.advance(1)
    self.assertEqual(log, ['td'])
    sim.advance_to(4)
    self.assertEqual(log, ['td', 'abs'])
    self.assertEqual(sim_time.ticks_of(0.2), 1)
    self.assertEqual(sim_time.ticks_of(-3), 0)

  def test_cancel_and_next_due(self):
    sim = sim_time.SimTimeScheduler()
    log = []
    disposable = sim.schedule_relative(1, _record(log, 'cancelled'))
    sim.schedule_relative(3, _record(log, 'kept'))
    self.assertEqual(sim.next_due(), 1)
    disposable.dispose()
    self.assertEqual(sim.next_due(), 3)
    self.assertEqual(sim.pending(), 1)
    sim.advance(3)
    self.assertEqual(log, ['kept'])

  def test_actions_scheduled_while_advancing_run_if_due(self):
    sim = sim_time.SimTimeScheduler()
    log = []

    def chain(scheduler, state):
      del state
      log.append(f'chain@{sim.tick}')
      if len(log) < 3:
        scheduler.schedule_relative(1, chain)

    sim.schedule(chain)
    sim.advance(5)
    self.assertEqual(log, ['chain@0', 'chain@1', 'chain@2'])

  def test_reentrant_advance_and_backwards_time_raise(self):
    sim = sim_time.SimTimeScheduler(start_tick=2)
    errors = []

    def reenter(scheduler, state):
      del scheduler, state
      try:
        sim.advance()
      except RuntimeError as e:
        errors.append(e)

    sim.schedule(reenter)
    sim.advance()
    self.assertLen(errors, 1)
    with self.assertRaises(ValueError):
      sim.advance_to(0)
    with self.assertRaises(ValueError):
      sim.advance(-1)

  def test_periodic_and_time_operators_follow_ticks(self):
    sim = sim_time.SimTimeScheduler()
    values = []
    reactivex.interval(2.0, scheduler=sim).pipe(ops.take(3)).subscribe(
        values.append
    )
    sim.advance(5)
    self.assertEqual(values, [0, 1])
    sim.advance(2)
    self.assertEqual(values, [0, 1, 2])

  def test_threads_can_schedule_concurrently(self):
    sim = sim_time.SimTimeScheduler()
    log = []
    lock = threading.Lock()

    def worker(name):
      for i in range(25):

        def action(scheduler, state, i=i):
          del scheduler, state
          with lock:
            log.append((name, i))

        sim.schedule_at(1, action, key=name)

    threads = [
        threading.Thread(target=worker, args=(n,)) for n in ('b', 'a', 'c')
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    sim.advance()
    expected = []
    for n in ('a', 'b', 'c'):
      for i in range(25):
        expected.append((n, i))
    self.assertEqual(log, expected)

  def test_state_round_trip(self):
    sim = sim_time.SimTimeScheduler()
    sim.schedule(_record([], 'x'))
    sim.advance(4)
    restored = sim_time.SimTimeScheduler()
    restored.set_state(sim.get_state())
    self.assertEqual(restored.tick, 4)
    self.assertEqual(restored.executed, 1)


class LatencySchedulerTest(absltest.TestCase):

  def test_latency_delays_and_can_change(self):
    sim = sim_time.SimTimeScheduler()
    view = sim.latency(3, key='strategist')
    log = []
    view.schedule(_record(log, 'slow'))
    sim.advance(2)
    self.assertEqual(log, [])
    sim.advance()
    self.assertEqual(log, ['slow'])
    view.set_latency(1)
    self.assertEqual(view.latency_ticks, 1)
    view.schedule_relative(1, _record(log, 'faster'))
    sim.advance(2)
    self.assertEqual(log, ['slow', 'faster'])
    self.assertEqual(view.now, sim.now)
    with self.assertRaises(ValueError):
      view.set_latency(-1)

  def test_publisher_result_lands_exactly_latency_ticks_later(self):
    sim = sim_time.SimTimeScheduler()
    calls = []

    def deliberate() -> str:
      calls.append(sim.tick)
      return f'plan made at tick {sim.tick}'

    publisher = pubsub.Publisher(
        function=deliberate,
        skip_if_busy=True,
        scheduler=sim.latency(3, key='Alice'),
    )
    subscriber = pubsub.Subscriber(
        publisher_component_key='__publisher__', initial_value='instinct'
    )
    agent = entity_agent.EntityAgent(
        agent_name='Alice',
        act_component=_NoOpAct(),
        context_components={
            '__publisher__': publisher,
            'belief': subscriber,
        },
    )
    beliefs = []
    for _ in range(7):
      agent.act()
      beliefs.append(subscriber.get_current_value())
      sim.advance()
    # Triggered at tick 0 (lands at 3); triggers at ticks 1-2 are skipped
    # while busy; re-triggered at tick 3 (lands at 6).
    self.assertEqual(calls, [3, 6])
    self.assertEqual(
        beliefs,
        ['instinct'] * 3
        + ['plan made at tick 3'] * 3
        + ['plan made at tick 6'],
    )
    self.assertEqual(publisher.compute_count, 2)
    publisher.close()

  def test_publishers_due_together_publish_in_key_order(self):
    sim = sim_time.SimTimeScheduler()
    bus = topics.TopicBus(clock=lambda: sim.tick)
    order = []
    bus.matching('minds.').subscribe(lambda m: order.append(m.source))
    publishers = {
        name: bus.make_publisher(
            f'minds.{name}',
            lambda name=name: f'{name} thought',
            source=name,
            scheduler=sim.latency(1, key=name),
        )
        for name in ('Zed', 'Amy', 'Kim')
    }
    agents = [
        entity_agent.EntityAgent(
            agent_name=name,
            act_component=_NoOpAct(),
            context_components={'mind': publisher},
        )
        for name, publisher in publishers.items()
    ]
    threads = [threading.Thread(target=agent.act) for agent in agents]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    sim.advance()
    self.assertEqual(order, ['Amy', 'Kim', 'Zed'])
    for publisher in publishers.values():
      publisher.close()

  def test_publisher_scheduler_helper(self):
    self.assertIsNone(sim_time.publisher_scheduler(None, 3, 'x'))
    sim = sim_time.SimTimeScheduler()
    view = sim_time.publisher_scheduler(sim, 3, 'x')
    assert isinstance(view, sim_time.LatencyScheduler)
    self.assertEqual(view.latency_ticks, 3)
    self.assertEqual(view.key, 'x')
    self.assertIs(view.base, sim)


if __name__ == '__main__':
  absltest.main()
