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

"""Lifeways end to end on the stock asynchronous engine (MOCK)."""

import contextlib
import io
import json
from typing import Any

from absl.testing import absltest
from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import person as person_lib
from concordia.associative_memory import basic_associative_memory
from examples.lifeways.world import world as world_lib


def _play(run: simulation.Run) -> None:
  with contextlib.redirect_stdout(io.StringIO()):
    run.play()


def _unordered_sets(data: Any) -> Any:
  """The data with memory banks' `stored_hashes` (a set) in a fixed order."""
  if isinstance(data, dict):
    return {
        k: sorted(v) if k == 'stored_hashes' else _unordered_sets(v)
        for k, v in data.items()
    }
  if isinstance(data, list):
    return [_unordered_sets(v) for v in data]
  return data


def _snapshot(run: simulation.Run) -> str:
  """The world and every living entity's state, as JSON."""
  state = json.loads(
      json.dumps(
          {
              'world': run.world().get_state(),
              'entities': {
                  e.name: e.get_state()  # pyrefly: ignore[missing-attribute]
                  for e in run.sim.entities
              },
          },
          default=str,
      )
  )
  return json.dumps(_unordered_sets(state), sort_keys=True)


class SandboxTest(absltest.TestCase):

  def test_a_year_resolves_whatever_order_the_threads_answered_in(self):
    runs = []
    for _ in range(2):
      run = simulation.build_simulation(scenarios.settings(max_years=30))
      _play(run)
      runs.append(_snapshot(run))
    self.assertEqual(runs[0], runs[1])

  def test_people_are_born_and_die_and_the_dead_leave(self):
    run = simulation.build_simulation(
        scenarios.settings(max_years=40), archive=simulation.Archive(keep=1000)
    )
    _play(run)
    world = run.world()
    names = {e.name for e in run.sim.entities}

    self.assertEqual(names, {*world.people, world_lib.EXPERIMENTER})
    self.assertEqual([r['name'] for r in run.archive.records], world.dead)
    births = sum(row['births'] for row in world.history)
    self.assertGreater(births, 5)
    self.assertNotEmpty(world.dead)

  def test_children_start_with_what_their_parents_passed_on(self):
    run = simulation.build_simulation(scenarios.settings(max_years=12))
    _play(run)
    world = run.world()
    child = next(p for p in world.people.values() if p.parents)
    entity = next(e for e in run.sim.entities if e.name == child.name)
    held = {
        c.id
        for c in entity.get_all_context_components().values()  # pyrefly: ignore[missing-attribute]
        if isinstance(c, culture.Custom)
    }
    self.assertEqual(held, set(child.given))
    self.assertNotEmpty(held)
    for custom_id in held:  # Each a recorded variant (perhaps a mutation).
      self.assertIn(custom_id, world.culture)

  def test_a_restored_year_plays_on_exactly(self):
    saved: dict[str, Any] = {}
    live = simulation.build_simulation(scenarios.settings(max_years=24))

    def save(year: int) -> None:
      if year == 12:
        saved['checkpoint'] = json.loads(
            json.dumps(live.sim.make_checkpoint_data(), default=str)
        )

    live.lifecycle.year_closed = save
    _play(live)

    restored = simulation.build_simulation(scenarios.settings(max_years=24))
    simulation.load_checkpoint(restored, saved['checkpoint'])
    self.assertEqual(restored.world().year, 12)
    _play(restored)

    self.assertEqual(_snapshot(restored), _snapshot(live))


class LearnerTest(absltest.TestCase):

  def _person(self, **params):
    prefab = person_lib.Person()
    prefab.params = {
        'name': 'Ada',
        'born': 0,
        'seed': 0,
        'customs': [],
        **params,
    }
    bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=simulation.hash_embedder
    )
    from examples.lifeways import mock_language_model  # pylint: disable=g-import-not-at-top

    return prefab.build(mock_language_model.MockModel(), bank)

  def _customs(self, entity) -> dict[str, culture.Custom]:
    return {
        c.id: c
        for c in entity.get_all_context_components().values()
        if isinstance(c, culture.Custom)
    }

  def test_an_offer_taken_up_becomes_a_component(self):
    entity = self._person(uptake={'apprenticeship': 1.0})
    spec = {
        'id': 'c9',
        'kind': 'skill',
        'domain': 'craft',
        'text': 'Knapping flint into blades.',
        'params': {'level': 3},
    }

    entity.observe(culture.offer_line(spec, 'apprenticeship'))

    custom = self._customs(entity)['c9']
    self.assertEqual(custom.spec, spec)
    restored = self._person(uptake={'apprenticeship': 1.0})
    restored.set_state(json.loads(json.dumps(entity.get_state())))
    self.assertEqual(self._customs(restored)['c9'].spec, spec)

  def test_a_full_domain_drops_its_oldest_custom(self):
    old = [
        {
            'id': f'c{i}',
            'kind': 'norm',
            'domain': 'sharing',
            'text': f'Old way {i}.',
            'params': {},
        }
        for i in (1, 2)
    ]
    entity = self._person(customs=old, uptake={'rite': 1.0}, per_domain=2)
    new = {
        'id': 'c3',
        'kind': 'norm',
        'domain': 'sharing',
        'text': 'A new way.',
        'params': {},
    }

    entity.observe(culture.offer_line(new, 'rite'))

    self.assertEqual(sorted(self._customs(entity)), ['c2', 'c3'])

  def test_an_offer_by_an_unknown_channel_is_not_taken_up(self):
    entity = self._person(uptake={})
    spec = {
        'id': 'c9',
        'kind': 'belief',
        'domain': 'sacred',
        'text': 'The river is a god.',
        'params': {},
    }
    entity.observe(culture.offer_line(spec, 'rumour'))
    self.assertEmpty(self._customs(entity))


if __name__ == '__main__':
  absltest.main()
