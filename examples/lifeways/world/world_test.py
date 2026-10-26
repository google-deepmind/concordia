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

"""How the world passes customs on: by what people do, never by minds."""

from absl.testing import absltest
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import world as world_lib

_FAITHFUL = {
    'upbringing': {'mutation': 0.0},
    'observation': {'radius': 2, 'witnessed': 3, 'mutation': 0.0},
    'apprenticeship': {
        'radius': 4,
        'per_master': 2,
        'mutation': 0.0,
        'improve': 0.0,
    },
}


class TransmissionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.world = world_lib.World(
        {'seed': 0, 'founders': 0, 'channels': _FAITHFUL}
    )
    culture = self.world.culture
    self.skill = culture.add('skill', 'food', 'Smoking fish.', {'level': 2}).id
    self.norm = culture.add('norm', 'sharing', 'Share the first catch.').id

  def _person(self, name, age, tile, parents=(), given=None):
    person = world_lib.Person(
        name=name,
        born=self.world.year - age,
        sex='f',
        tile=tile,
        food=5.0,
        parents=list(parents),
        given=[self.skill, self.norm] if given is None else list(given),
    )
    self.world.people[name] = person
    for parent in parents:
      self.world.people[parent].children.append(name)
    return person

  def _offers(self, name):
    """The offers queued for `name`: (custom id, channel, from)."""
    text = '\n'.join(self.world.queued.get(name, ()))
    return [
        (m.group('id'), m.group('via'), m.group('source'))
        for m in culture_lib.OFFER.finditer(text)
    ]

  def test_those_near_see_what_a_person_does_it_by(self):
    self._person('Ada', 30, (5, 5))
    self._person('Bo', 30, (5, 6), given=())
    self._person('Cy', 30, (5, 10), given=())
    self.world.actions = {
        'Ada': f'gather food {self.skill}',
        'Bo': 'rest',
        'Cy': 'rest',
    }

    self.world.close_year()

    self.assertEqual(self._offers('Bo'), [(self.skill, 'observation', 'Ada')])
    self.assertEmpty(self._offers('Cy'))
    self.assertIn(self.skill, self.world.people['Bo'].given)

  def test_what_a_person_was_never_given_is_not_seen(self):
    self._person('Ada', 30, (5, 5), given=[self.norm])
    self._person('Bo', 30, (5, 6), given=())
    self.world.actions = {'Ada': f'gather food {self.skill}', 'Bo': 'rest'}

    self.world.close_year()

    self.assertEmpty(self._offers('Bo'))

  def test_children_see_their_parents_wherever_they_work(self):
    self._person('Ada', 30, (0, 0))
    self._person('Kit', 8, (8, 12), parents=['Ada'], given=())
    self.world.actions = {'Ada': f'gather food {self.skill}', 'Kit': 'play'}

    self.world.close_year()

    self.assertEqual(self._offers('Kit'), [(self.skill, 'upbringing', 'Ada')])

  def test_a_parent_teaches_its_children_and_nobody_else(self):
    self._person('Ada', 30, (5, 5))
    self._person('Kit', 8, (5, 5), parents=['Ada'], given=())
    self._person('Bo', 30, (5, 6), given=())
    self.assertIn(world_lib.TEACH, self.world.moves('Ada'))
    self.world.actions = {
        'Ada': f'teach {self.skill} {self.norm}',
        'Kit': 'play',
        'Bo': 'rest',
    }

    self.world.close_year()

    self.assertEqual(
        sorted(self._offers('Kit')),
        sorted([
            (self.skill, 'upbringing', 'Ada'),
            (self.norm, 'upbringing', 'Ada'),
        ]),
    )
    self.assertEmpty(self._offers('Bo'))

  def test_a_youth_learns_a_craft_beside_a_skilled_worker(self):
    self.world.channels['apprenticeship']['improve'] = 1.0
    self._person('Ada', 30, (5, 5))
    self._person('Jo', 13, (5, 7), given=())
    self.assertIn(world_lib.APPRENTICE, self.world.moves('Jo'))
    self.world.actions = {
        'Ada': f'gather food {self.skill}',
        'Jo': world_lib.APPRENTICE,
    }

    self.world.close_year()

    learned = [o for o in self._offers('Jo') if o[1] == 'apprenticeship']
    self.assertLen(learned, 1)
    better = self.world.culture.get(learned[0][0])
    self.assertEqual(better.parent, self.skill)
    self.assertEqual(better.origin, 'improvement')
    self.assertEqual(better.params['level'], 3)

  def test_variants_nobody_holds_are_forgotten(self):
    self._person('Ada', 30, (5, 5), given=[self.skill])
    self.world.actions = {'Ada': 'rest'}

    outcome = self.world.close_year()

    self.assertEqual([c['id'] for c in outcome.extinct], [self.norm])
    self.assertNotIn(self.norm, self.world.culture)
    self.assertIn(self.skill, self.world.culture)

  def test_every_name_is_new(self):
    names = [self.world._new_name() for _ in range(5000)]  # pylint: disable=protected-access
    self.assertLen(set(names), len(names))


if __name__ == '__main__':
  absltest.main()
