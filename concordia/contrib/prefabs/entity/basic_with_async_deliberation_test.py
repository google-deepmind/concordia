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

"""Tests for the async-deliberation entity prefab."""

from collections.abc import Collection, Mapping, Sequence
import copy
import json
import threading
from typing import Any

from absl.testing import absltest
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.agent import pubsub
from concordia.contrib.prefabs.entity import basic_with_async_deliberation
from concordia.contrib.pubsub import sim_time
from concordia.contrib.pubsub import topics
from concordia.language_model import language_model
from concordia.prefabs.game_master import generic as generic_gm
from concordia.prefabs.simulation import generic as generic_simulation
from concordia.typing import entity as entity_lib
from concordia.typing import human_input
from concordia.typing import prefab as prefab_lib
import numpy as np

PLAN = 'Gather shells at dawn, then mend the nets.'


class _FakeModel(language_model.LanguageModel):
  """Answers every question with the same plan and picks the first choice."""

  def __init__(self):
    self.prompts: list[str] = []
    self._lock = threading.Lock()

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    with self._lock:
      self.prompts.append(prompt)
    return PLAN

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    with self._lock:
      self.prompts.append(prompt)
    return 0, responses[0], {}


def _memory_bank() -> basic_associative_memory.AssociativeMemoryBank:
  return basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=lambda text: np.ones(3)
  )


def _act(entity) -> str:
  return entity.act(
      entity_lib.free_action_spec(call_to_action='What does {name} do?')
  )


class BasicWithAsyncDeliberationTest(absltest.TestCase):

  def test_beliefs_and_a_plan_that_lands_ticks_later(self):
    tick = [0]
    bus = topics.TopicBus(clock=lambda: tick[0])
    scheduler = sim_time.SimTimeScheduler()
    model = _FakeModel()
    prefab = basic_with_async_deliberation.Entity(
        params={
            'name': 'Ada',
            'goal': 'Keep the harbour supplied.',
            'bus': bus,
            'subscriptions': {'Tide table': 'harbour.tides'},
            'clock_topic': 'world.tick',
            'sim_scheduler': scheduler,
            'deliberation_latency_ticks': 2,
        }
    )
    entity = prefab.build(model, _memory_bank())
    bus.publish('world.tick', '0')
    bus.publish('harbour.tides', 'High tide at noon.')
    self.assertEqual(
        entity.get_component(
            'Tide table', type_=pubsub.Subscriber
        ).get_current_value(),
        'High tide at noon. [fresh]',
    )
    _act(entity)
    self.assertIn('No plan yet', model.prompts[-1])
    self.assertIn('High tide at noon. [fresh]', model.prompts[-1])

    for t in (1, 2):
      tick[0] = t
      scheduler.advance()
      bus.publish('world.tick', str(t))
    self.assertEqual(bus.message_count('Ada.plan'), 1)
    plan = entity.get_component(
        'Current plan', type_=pubsub.Subscriber
    ).get_current_value()
    self.assertIn(PLAN, plan)
    self.assertIn('[fresh]', plan)
    _act(entity)
    self.assertIn(PLAN, model.prompts[-1])
    self.assertIn('High tide at noon. [2 ticks old]', model.prompts[-1])
    self.assertIn('Ada', bus.topology()['harbour.tides']['subscribers'])

  def test_renderers_turn_payloads_into_text(self):
    bus = topics.TopicBus()
    entity = basic_with_async_deliberation.Entity(
        params={
            'name': 'Fay',
            'bus': bus,
            'subscriptions': {'Prices': 'market'},
            'renderers': {'market': lambda m: f'Fish cost {m.payload} coins.'},
        }
    ).build(_FakeModel(), _memory_bank())
    bus.publish('market', '3')
    self.assertEqual(
        entity.get_component(
            'Prices', type_=pubsub.Subscriber
        ).get_current_value(),
        'Fish cost 3 coins.',
    )

  def test_without_a_bus_the_plan_reaches_the_entity_directly(self):
    scheduler = sim_time.SimTimeScheduler()
    model = _FakeModel()
    entity = basic_with_async_deliberation.Entity(
        params={'name': 'Bo', 'sim_scheduler': scheduler}
    ).build(model, _memory_bank())
    _act(entity)
    scheduler.advance(2)
    self.assertIn(
        PLAN,
        entity.get_component(
            'Current plan', type_=pubsub.Subscriber
        ).get_current_value(),
    )

  def test_deliberation_can_use_its_own_model(self):
    scheduler = sim_time.SimTimeScheduler()
    actor, planner = _FakeModel(), _FakeModel()
    entity = basic_with_async_deliberation.Entity(
        params={
            'name': 'Eli',
            'sim_scheduler': scheduler,
            'deliberation_model': planner,
        }
    ).build(actor, _memory_bank())
    _act(entity)
    acted = len(actor.prompts)
    scheduler.advance(2)
    self.assertNotEmpty(planner.prompts)
    self.assertLen(actor.prompts, acted)

  def test_extra_components_join_the_prompt(self):
    model = _FakeModel()
    entity = basic_with_async_deliberation.Entity(
        params={
            'name': 'Di',
            'extra_components': {'Motto': constant.Constant('Waste nothing.')},
        }
    ).build(model, _memory_bank())
    _act(entity)
    # The plan runs on a background thread here, so its prompt may come last.
    self.assertTrue(any('Waste nothing.' in p for p in model.prompts))

  def test_the_plan_can_read_chosen_extra_components(self):
    scheduler = sim_time.SimTimeScheduler()
    actor, planner = _FakeModel(), _FakeModel()
    entity = basic_with_async_deliberation.Entity(
        params={
            'name': 'Bo',
            'sim_scheduler': scheduler,
            'deliberation_model': planner,
            'extra_components': {
                'Ledger': constant.Constant('Grain: 4 sacks.'),
                'Motto': constant.Constant('Waste nothing.'),
            },
            'plan_components': ['Ledger'],
        }
    ).build(actor, _memory_bank())
    _act(entity)
    scheduler.advance(2)
    planned = ' '.join(planner.prompts)
    self.assertIn('Grain: 4 sacks.', planned)
    self.assertNotIn('Waste nothing.', planned)
    with self.assertRaisesRegex(ValueError, 'plan_components'):
      basic_with_async_deliberation.Entity(
          params={'name': 'Bo', 'plan_components': ['Nope']}
      ).build(_FakeModel(), _memory_bank())

  def test_a_person_can_act_while_deliberation_runs_in_the_background(self):
    scheduler = sim_time.SimTimeScheduler()
    actor, planner = _FakeModel(), _FakeModel()
    requests: list[human_input.HumanInputRequest] = []

    def person(request: human_input.HumanInputRequest) -> str:
      requests.append(request)
      return 'Mend the nets.'

    entity = basic_with_async_deliberation.Entity(
        params={
            'name': 'Ada',
            'goal': 'Keep the harbour supplied.',
            'sim_scheduler': scheduler,
            'deliberation_model': planner,
        }
    ).build(
        actor,
        _memory_bank(),
        act_component=human_act_component.HumanActComponent(person),
    )
    self.assertEqual(_act(entity), 'Mend the nets.')
    self.assertEmpty(actor.prompts)  # The person acted, not the model.
    self.assertIn('Keep the harbour supplied.', requests[0].context)
    self.assertIn('No plan yet', requests[0].context)
    scheduler.advance(2)
    self.assertNotEmpty(planner.prompts)  # Deliberation still ran.
    _act(entity)
    self.assertIn(PLAN, requests[-1].context)

  def test_an_acting_policy_factory_gets_the_default_order(self):
    def params():  # Fresh components each time: one entity owns each.
      return {
          'name': 'Di',
          'goal': 'Waste nothing.',
          'extra_components': {'Motto': constant.Constant('Mend, do not buy.')},
      }

    default = basic_with_async_deliberation.Entity(params=params()).build(
        _FakeModel(), _memory_bank()
    )
    default_act = default.get_act_component()
    assert isinstance(default_act, concat_act_component.ConcatActComponent)
    orders = []

    def factory(order):
      orders.append(tuple(order))
      return human_act_component.HumanActComponent(
          lambda request: '', component_order=order
      )

    basic_with_async_deliberation.Entity(params=params()).build(
        _FakeModel(), _memory_bank(), act_component_factory=factory
    )
    self.assertEqual(
        orders, [tuple(default_act.get_context_concat_order() or ())]
    )
    self.assertIn('Motto', orders[0])
    with self.assertRaisesRegex(ValueError, 'not both'):
      basic_with_async_deliberation.Entity(params=params()).build(
          _FakeModel(),
          _memory_bank(),
          act_component=human_act_component.HumanActComponent(
              lambda request: ''
          ),
          act_component_factory=factory,
      )

  def test_subscriptions_need_a_bus(self):
    with self.assertRaises(ValueError):
      basic_with_async_deliberation.Entity(
          params={'name': 'Cy', 'subscriptions': {'News': 'news'}}
      ).build(_FakeModel(), _memory_bank())

  def test_mid_deliberation_simulation_checkpoint_and_deepcopy(self):
    tick = [0]
    bus = topics.TopicBus(clock=lambda: tick[0])
    scheduler = sim_time.SimTimeScheduler()
    model = _FakeModel()
    prefabs = {
        'async_entity': basic_with_async_deliberation.Entity(),
        'gm': generic_gm.GameMaster(),
    }
    instances = [
        prefab_lib.InstanceConfig(
            prefab='async_entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Ada',
                'goal': 'Keep the harbour supplied.',
                'bus': bus,
                'subscriptions': {'Tide table': 'harbour.tides'},
                'renderers': {
                    'harbour.tides': lambda m: f'Tide: {m.payload}',
                },
                'clock_topic': 'world.tick',
                'sim_scheduler': scheduler,
                'deliberation_latency_ticks': 2,
                'extra_components': {
                    'Motto': constant.Constant('Waste nothing.'),
                },
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='gm',
            role=prefab_lib.Role.GAME_MASTER,
            params={'name': 'Rules', 'acting_order': 'fixed'},
        ),
    ]
    config = copy.deepcopy(
        prefab_lib.Config(prefabs=prefabs, instances=instances)
    )
    sim1 = generic_simulation.Simulation(
        config=config, model=model, embedder=lambda text: np.ones(3)
    )
    bus.publish('world.tick', '0')
    bus.publish('harbour.tides', 'High tide')
    # Ada acts at tick 0, scheduling a deliberation due at tick 2.
    _act(sim1.entities[0])
    tick[0] = 1
    scheduler.advance()
    bus.publish('world.tick', '1')
    # Checkpoint at tick 1 while Ada's deliberation is still in flight!
    checkpoint = json.loads(json.dumps(sim1.make_checkpoint_data()))

    # Advance sim1 further to tick 4 so bus and scheduler move ahead, then
    # restore into a new simulation sim2 and verify tick 1 state is recovered.
    tick[0] = 4
    scheduler.advance(3)
    bus.publish('world.tick', '4')
    bus.publish('harbour.tides', 'Future tide')

    sim2 = generic_simulation.Simulation(
        config=config, model=model, embedder=lambda text: np.ones(3)
    )
    tick[0] = 1
    sim2.load_from_checkpoint(checkpoint)
    ada2 = sim2.entities[0]
    self.assertEqual(scheduler.tick, 1)
    self.assertTrue(
        ada2.get_component('Deliberation', type_=pubsub.Publisher).is_busy()
    )
    self.assertEqual(
        ada2.get_component(
            'Tide table', type_=pubsub.Subscriber
        ).get_current_value(),
        'Tide: High tide [1 tick old]',
    )
    # Advancing 1 more tick (to tick 2) completes the restored deliberation.
    tick[0] = 2
    scheduler.advance()
    bus.publish('world.tick', '2')
    self.assertIn(
        PLAN,
        ada2.get_component(
            'Current plan', type_=pubsub.Subscriber
        ).get_current_value(),
    )


if __name__ == '__main__':
  absltest.main()
