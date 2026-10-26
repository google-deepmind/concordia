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

import json
from typing import Any

from absl.testing import absltest
from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import person as person_lib
from concordia.associative_memory import basic_associative_memory
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import world as world_lib


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
      run.play()
      runs.append(_snapshot(run))
    self.assertEqual(runs[0], runs[1])

  def test_people_are_born_and_die_and_the_dead_leave(self):
    run = simulation.build_simulation(
        scenarios.settings(max_years=40), archive=simulation.Archive(keep=1000)
    )
    run.play()
    world = run.world()
    names = {e.name for e in run.sim.entities}

    self.assertEqual(names, {*world.people, world_lib.EXPERIMENTER})
    self.assertEqual([r['name'] for r in run.archive.records], world.dead)
    births = sum(row['births'] for row in world.history)
    self.assertGreater(births, 5)
    self.assertNotEmpty(world.dead)

  def test_children_start_with_what_their_parents_passed_on(self):
    run = simulation.build_simulation(scenarios.settings(max_years=12))
    run.play()
    world = run.world()
    # The youngest child, too young yet to have been offered anything else.
    child = max(
        (p for p in world.people.values() if p.parents), key=lambda p: p.born
    )
    self.assertLess(child.age(world.year), world_lib.CHILD_ACTS_AGE)
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

  def test_every_channel_moves_customs(self):
    run = simulation.build_simulation(scenarios.settings(max_years=60))
    run.play()

    census = run.customs_census()

    for channel in ('birth', 'upbringing', 'observation', 'apprenticeship'):
      self.assertGreater(census['by_channel'].get(channel, 0), 0, channel)
    self.assertGreater(census['distinct'], 7)  # Variants have arisen.

  def test_a_restored_year_plays_on_exactly(self):
    saved: dict[str, Any] = {}
    live = simulation.build_simulation(scenarios.settings(max_years=24))

    def save(year: int) -> None:
      if year == 12:
        saved['checkpoint'] = json.loads(
            json.dumps(live.sim.make_checkpoint_data(), default=str)
        )

    live.lifecycle.year_closed = save
    live.play()

    restored = simulation.build_simulation(scenarios.settings(max_years=24))
    simulation.load_checkpoint(restored, saved['checkpoint'])
    self.assertEqual(restored.world().year, 12)
    restored.play()

    self.assertEqual(_snapshot(restored), _snapshot(live))


def _spec(
    custom_id: str, kind: str, domain: str, level: int | None = None
) -> dict[str, Any]:
  return {
      'id': custom_id,
      'kind': kind,
      'domain': domain,
      'text': f'The way of {custom_id}.',
      'params': {} if level is None else {'level': level},
  }


class ArchiveTest(absltest.TestCase):

  def test_it_keeps_only_the_most_recent_and_sinks_all(self):
    sunk = []
    for keep in (0, 2):
      archive = simulation.Archive(keep=keep, sink=sunk.append)
      for i in range(5):
        archive.add({'name': f'P{i}'})
      self.assertEqual(
          [r['name'] for r in archive.records],
          [f'P{i}' for i in range(5 - keep, 5)],
      )
    self.assertLen(sunk, 10)


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

    entity.observe(culture_lib.offer_line(spec, 'apprenticeship'))

    custom = self._customs(entity)['c9']
    self.assertEqual(custom.spec, spec)
    restored = self._person(uptake={'apprenticeship': 1.0})
    restored.set_state(json.loads(json.dumps(entity.get_state())))
    self.assertEqual(self._customs(restored)['c9'].spec, spec)

  def test_a_new_custom_replaces_the_one_of_its_kind_and_domain(self):
    entity = self._person(
        customs=[
            _spec('c1', 'norm', 'sharing'),
            _spec('c2', 'belief', 'sacred'),
        ],
        uptake={'rite': 1.0},
    )

    entity.observe(
        culture_lib.offer_line(_spec('c3', 'norm', 'sharing'), 'rite')
    )

    self.assertEqual(sorted(self._customs(entity)), ['c2', 'c3'])

  def test_a_full_domain_drops_its_oldest_custom(self):
    entity = self._person(
        customs=[
            _spec('c1', 'norm', 'sharing'),
            _spec('c2', 'value', 'sharing'),
        ],
        uptake={'rite': 1.0},
        per_domain=2,
    )

    entity.observe(
        culture_lib.offer_line(_spec('c3', 'belief', 'sharing'), 'rite')
    )

    self.assertEqual(sorted(self._customs(entity)), ['c2', 'c3'])

  def test_only_a_better_skill_replaces_the_one_held(self):
    entity = self._person(
        customs=[_spec('c1', 'skill', 'food', level=3)],
        uptake={'apprenticeship': 1.0},
    )

    entity.observe(
        culture_lib.offer_line(
            _spec('c2', 'skill', 'food', level=3), 'apprenticeship'
        )
    )
    self.assertEqual(sorted(self._customs(entity)), ['c1'])
    entity.observe(
        culture_lib.offer_line(
            _spec('c3', 'skill', 'food', level=4), 'apprenticeship'
        )
    )
    self.assertEqual(sorted(self._customs(entity)), ['c3'])

  def test_the_well_fed_are_copied_more_by_the_success_biased(self):
    spec = _spec('c9', 'habit', 'food')

    def chance(stores: float, biases: str) -> float:
      offer = culture_lib.OFFER.search(
          culture_lib.offer_line(spec, 'observation', 'Bo', stores)
      )
      assert offer is not None
      return culture.offer_chance(offer, {'observation': 0.2}, biases)

    biased = 'learn from those whose stores are full.'
    self.assertGreater(chance(12.0, biased), chance(1.0, biased))
    self.assertEqual(chance(12.0, ''), chance(1.0, ''))

  def test_an_offer_by_an_unknown_channel_is_not_taken_up(self):
    entity = self._person(uptake={})
    spec = {
        'id': 'c9',
        'kind': 'belief',
        'domain': 'sacred',
        'text': 'The river is a god.',
        'params': {},
    }
    entity.observe(culture_lib.offer_line(spec, 'rumour'))
    self.assertEmpty(self._customs(entity))


if __name__ == '__main__':
  absltest.main()
