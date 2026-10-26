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
import sys
import threading
import time
from typing import Any

from absl.testing import absltest
from examples.lifeways import mock_language_model
from examples.lifeways import plot
from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways.components import culture
from examples.lifeways.prefabs import experimenter as experimenter_lib
from examples.lifeways.prefabs import person as person_lib
from concordia.associative_memory import basic_associative_memory
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import world as world_lib
from concordia.typing import entity as entity_lib

_QUESTION = entity_lib.free_action_spec(call_to_action='What now?')


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
    run = simulation.build_simulation(scenarios.settings(max_years=20))
    newborns = []

    def look(year: int) -> None:
      world = run.world()
      for person in world.people.values():
        if person.parents and person.born == year - 1:
          entity = next(e for e in run.sim.entities if e.name == person.name)
          held = {
              c.id
              for c in entity.get_all_context_components().values()  # pyrefly: ignore[missing-attribute]
              if isinstance(c, culture.Custom)
          }
          newborns.append((held, set(person.given)))
          for (
              custom_id
          ) in held:  # Each a recorded variant (perhaps a mutation).
            self.assertIn(custom_id, world.culture)

    run.lifecycle.year_closed = look
    run.play()

    self.assertNotEmpty(newborns)
    for held, given in newborns:
      self.assertNotEmpty(held)
      self.assertEqual(held, given)

  def test_many_channels_move_customs_in_one_run(self):
    run = simulation.build_simulation(scenarios.settings(max_years=80))
    taken_up = set()
    run.lifecycle.year_closed = lambda year: taken_up.update(
        run.customs_census()['by_channel']
    )

    run.play()

    self.assertContainsSubset({'birth', 'upbringing', 'observation'}, taken_up)
    # Many channels carry customs in one run; which of the rarer ones show in
    # a given 80 years varies, and each has its own test in world_test.
    offers = {c for row in run.world().history for c in row['offers']}
    self.assertGreaterEqual(len(offers), 7, offers)

  def test_the_experimenter_releases_what_its_plan_says(self):
    run = simulation.build_simulation(
        scenarios.settings(
            max_years=8,
            plan={'5': 'release belief sacred as a song at 8,8: Sun.'},
        )
    )
    forgotten = []
    run.lifecycle.forgotten = forgotten.extend

    run.play()

    released = [row['released'] for row in run.world().history]
    self.assertEqual(released, [0, 0, 0, 0, 0, 1, 0, 0])
    # The song is a new variant in the world's record (kept while anyone
    # who heard it lives, then in the long record).
    culture = run.world().culture
    recorded = [c.to_dict() for c in map(culture.get, _ids(culture))]
    self.assertIn(
        ('Sun.', 'release'),
        [(c['text'], c['origin']) for c in recorded + forgotten],
    )

  def test_the_experimenter_interviews_a_person(self):
    settings = scenarios.settings(max_years=4)
    name = scenarios.founders(settings)[0]['params']['name']
    settings['plan'] = {'1': f'ask {name}: What do you live by?'}
    run = simulation.build_simulation(settings)

    run.play()

    (transcript,) = run.world().transcripts
    self.assertEqual(
        (transcript['year'], transcript['name'], transcript['question']),
        (2, name, 'What do you live by?'),
    )
    self.assertStartsWith(transcript['answer'], 'We live by these:')

  def test_peoples_are_those_whose_marriage_rules_allow_each_other(self):
    def tongue(words):
      return {
          'kind': 'dialect',
          'domain': 'language',
          'text': f'A tongue: {" ".join(words)}.',
          'params': {'words': words},
          'held_by': 0.5,
      }

    run = simulation.build_simulation(
        scenarios.settings(
            max_years=1,
            arrivals=None,
            founding_customs=[
                tongue(['ka', 'ri', 'to', 'ma']),
                tongue(['bo', 'ha', 'yo', 'lu']),
                {
                    'kind': 'marriage',
                    'domain': 'kinship',
                    'text': 'We marry one who speaks as we do.',
                    'params': {'within': 'tongue'},
                },
            ],
        )
    )
    run.play()

    self.assertEqual(run.peoples(), {'peoples': 2, 'sizes': [6, 6]})

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


def _ids(culture) -> list[str]:
  return [c['id'] for c in culture.get_state()['customs']]


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


class PresetTest(absltest.TestCase):

  def test_two_valleys_are_two_peoples_parted_by_a_ridge(self):
    settings = scenarios.settings('two_valleys', max_years=2)
    run = simulation.build_simulation(settings)
    land = run.world().land
    ridge = settings['ridge']
    col, (low, high) = ridge['col'], ridge['gap']
    kinds = [land.kind((row, col)) for row in range(land.rows)]
    self.assertEqual(
        [row for row, kind in enumerate(kinds) if kind != 'hills'],
        list(range(low, high + 1)),
    )

    run.play()

    self.assertEqual(run.peoples()['peoples'], 2)
    descents = {p.descent for p in run.world().people.values()}
    self.assertEqual(descents, {'West people', 'East people'})

  def test_in_plasmid_soup_people_bud_alone(self):
    run = simulation.build_simulation(
        scenarios.settings('plasmid_soup', max_years=30)
    )
    run.play()

    world = run.world()
    self.assertGreater(sum(row['births'] for row in world.history), 0)
    self.assertEqual({p.sex for p in world.people.values()}, {''})
    self.assertEqual(
        [row['released'] for row in world.history if row['year'] == 20], [1]
    )


class LeverTest(absltest.TestCase):

  def test_a_twin_pulls_one_lever_and_leaves_its_world_alone(self):
    world = scenarios.settings()

    richer = scenarios.with_lever(world, 'richness', 0.5)
    quieter = scenarios.with_lever(world, scenarios.WITHOUT, 'story')

    self.assertEqual(richer['richness'], 0.5)
    self.assertEqual({k for k in richer if richer[k] != world[k]}, {'richness'})
    self.assertNotIn('story', quieter['channels'])
    self.assertIn('story', world['channels'])
    self.assertEqual(world, scenarios.settings())

  def test_a_lever_has_bounds(self):
    world = scenarios.settings()
    for lever, value in (
        ('seed', 1),
        ('mutation', -0.1),
        ('mutation', True),
        (scenarios.WITHOUT, 'telepathy'),
    ):
      with self.assertRaises(ValueError):
        scenarios.with_lever(world, lever, value)


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


class ExperimenterTest(absltest.TestCase):

  def test_its_memory_keeps_only_the_latest(self):
    run = simulation.build_simulation(scenarios.settings(max_years=50))
    run.play()

    experimenter = next(
        e for e in run.sim.entities if e.name == world_lib.EXPERIMENTER
    )
    bank = json.loads(
        experimenter.get_state()['context_components']['__memory__'][  # pyrefly: ignore[missing-attribute]
            'memory_bank'
        ]
    )
    self.assertLessEqual(len(bank), 32)
    self.assertIn('Year 49', bank[-1])

  def test_a_plan_reads_the_year_from_the_world_alone(self):
    plan = experimenter_lib.Plan(
        {5: 'release belief sacred as a song at 1,1: Hm.'}
    )
    status = world_lib.EXPERIMENTER_STATUS_FORM.format(year=5, people=3)
    question = entity_lib.free_action_spec(call_to_action='What now?')

    self.assertEqual(
        plan(question, {'obs': status}),
        'release belief sacred as a song at 1,1: Hm.',
    )
    self.assertEqual(
        plan(question, {'obs': status + ' Ada says: "Year 40. We wait."'}),
        'release belief sacred as a song at 1,1: Hm.',
    )


class QuietTest(absltest.TestCase):

  def test_two_runs_at_once_share_one_redirect(self):
    stdout = sys.stdout
    runs = [
        simulation.build_simulation(scenarios.settings(max_years=years))
        for years in (3, 8)
    ]
    threads = [threading.Thread(target=run.play) for run in runs]
    for thread in threads:
      thread.start()
      time.sleep(0.05)
    for thread in threads:
      thread.join(timeout=120)

    self.assertIs(sys.stdout, stdout)


class PlotTest(absltest.TestCase):

  def test_it_draws_a_census(self):
    run = simulation.build_simulation(scenarios.settings(max_years=3))
    rows = []
    run.lifecycle.year_closed = lambda year: rows.append(
        {**run.world().history[-1], 'held': run.customs_census()}
    )
    run.play()
    out = self.create_tempfile('census.png').full_path

    plot.plot(rows, out)

    with open(out, 'rb') as f:
      self.assertEqual(f.read(4), b'\x89PNG')


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

  def test_an_offer_with_garbled_params_is_passed_over(self):
    entity = self._person(uptake={'release': 1.0})
    line = culture_lib.offer_line(_spec('c7', 'belief', 'sacred'), 'release')

    entity.observe(line + ' |params={are near}')
    self.assertEmpty(self._customs(entity))

    entity.observe(line)  # The person goes on taking up what it can read.
    self.assertEqual(list(self._customs(entity)), ['c7'])

  def test_an_offer_carries_the_whole_custom(self):
    entity = self._person(uptake={'upbringing': 1.0})
    rule = _spec('c7', 'marriage', 'kinship')
    rule['params'] = {'within': 'descent'}
    tongue = _spec('c8', 'dialect', 'language')
    tongue['params'] = {'words': ['ka', 'ri']}

    entity.observe(
        culture_lib.offer_line(rule, 'upbringing', 'Mum', 2.0)
        + '\n'
        + culture_lib.offer_line(tongue, 'upbringing', 'Mum', 2.0)
    )

    customs = self._customs(entity)
    self.assertEqual(customs['c7'].spec, rule)
    self.assertEqual(customs['c8'].spec, tongue)

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

  def test_inherited_customs_keep_within_capacity(self):
    customs = [
        _spec(f'c{i}', 'belief', domain)
        for i, domain in enumerate(('sacred', 'craft', 'identity', 'authority'))
    ]

    entity = self._person(customs=customs, capacity=3)

    self.assertEqual(sorted(self._customs(entity)), ['c1', 'c2', 'c3'])

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

  def test_a_norm_only_expected_does_not_displace_ones_own(self):
    own = _spec('c1', 'norm', 'sharing')
    entity = self._person(customs=[own], uptake={'observation': 1.0})
    expected = _spec('c2', 'norm', 'sharing')
    expected['params']['form'] = culture_lib.KNOWN

    entity.observe(culture_lib.offer_line(expected, 'observation'))
    self.assertEqual(sorted(self._customs(entity)), ['c1'])
    custom = self._customs(entity)['c1']
    self.assertStartsWith(custom.pre_act(_QUESTION), 'A custom of ours')

    entity = self._person(uptake={'observation': 1.0})
    entity.observe(culture_lib.offer_line(expected, 'observation'))
    custom = self._customs(entity)['c2']
    self.assertEqual(custom.spec, expected)
    self.assertStartsWith(custom.pre_act(_QUESTION), 'A custom others expect')

  def test_sermons_move_beliefs_more_and_skills_not_at_all(self):
    def chance(kind: str) -> float:
      offer = culture_lib.OFFER.search(
          culture_lib.offer_line(_spec('c9', kind, 'sacred'), 'preaching')
      )
      assert offer is not None
      return culture.offer_chance(offer, {'preaching': 0.2}, '')

    self.assertGreater(chance('belief'), chance('norm'))
    self.assertEqual(chance('skill'), 0.0)

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
