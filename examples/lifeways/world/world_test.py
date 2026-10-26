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
from examples.lifeways import scenarios
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import land as land_lib
from examples.lifeways.world import places as places_lib
from examples.lifeways.world import world as world_lib
import numpy as np

# The default channels, copying without error.
_FAITHFUL = {
    name: {**channel, 'mutation': 0.0, 'improve': 0.0}
    for name, channel in world_lib.DEFAULT_CHANNELS.items()
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

  def test_a_skill_dies_with_the_last_who_worked_by_it(self):
    ada = self._person('Ada', 30, (5, 5), given=[self.skill])
    ada.food, ada.starving = 0.0, 1
    self.world.land.food[:] = 0.0  # A famine: gathering finds nothing.
    self.world.actions = {'Ada': f'gather food {self.skill}'}

    outcome = self.world.close_year()

    self.assertEqual(outcome.deaths, ['Ada'])
    self.assertIn(self.skill, [c['id'] for c in outcome.extinct])
    self.assertEqual(self.world.history[-1]['skill_worked'], 2.0)

  def test_the_needy_within_sight_may_be_given_food(self):
    ada = self._person('Ada', 30, (5, 5))
    bo = self._person('Bo', 30, (5, 6), given=())
    bo.food = 0.5
    self._person('Cy', 30, (6, 6), given=())
    self.assertIn(f'{world_lib.SHARE} Bo', self.world.moves('Ada'))
    self.world.actions = {
        'Ada': f'{world_lib.SHARE} Bo {self.norm}',
        'Bo': 'rest',
        'Cy': 'rest',
    }

    self.world.close_year()

    self.assertEqual(ada.food, 5.0 - self.world.share - 1.0)  # Then she ate.
    self.assertEqual(bo.food, 0.5 + self.world.share - 1.0)
    known = self.world.culture.in_form(self.norm, culture_lib.KNOWN, 0).id
    self.assertIn((known, 'observation', 'Ada'), self._offers('Cy'))
    self.assertEmpty(self.world.breaches)

  def test_keeping_full_stores_from_the_needy_may_be_shamed(self):
    ada = self._person('Ada', 30, (5, 5), given=())
    ada.food = 9.0
    self._person('Bo', 30, (5, 6), given=()).food = 0.5
    self._person('Cy', 30, (6, 5))
    self.world.actions = {'Ada': 'rest', 'Bo': 'rest', 'Cy': 'rest'}

    self.world.close_year()

    self.assertEqual(self.world.breaches, {'Ada': ['Bo']})
    self.assertIn('Ada kept full stores', ' '.join(self.world.queued['Cy']))
    self.assertIn(f'{world_lib.SHAME} Ada', self.world.moves('Cy'))
    self.world.queued.clear()
    self.world.actions = {
        'Ada': 'rest',
        'Bo': 'rest',
        'Cy': f'{world_lib.SHAME} Ada {self.norm}',
    }

    self.world.close_year()

    self.assertIn((self.norm, 'sanction', 'Cy'), self._offers('Ada'))

  def test_keeping_a_norm_one_knows_makes_it_ones_own(self):
    known = self.world.culture.in_form(self.norm, culture_lib.KNOWN, 0).id
    self._person('Ada', 30, (5, 5), given=[known])
    self._person('Bo', 30, (5, 6), given=()).food = 0.5
    self.world.actions = {'Ada': f'{world_lib.SHARE} Bo {known}', 'Bo': 'rest'}

    self.world.close_year()

    self.assertIn((self.norm, 'practice', 'Ada'), self._offers('Ada'))

  def test_one_looking_for_a_partner_looks_until_paired(self):
    self._person('Ada', 20, (5, 5))
    bo = self._person('Bo', 20, (5, 9))
    bo.sex = 'm'
    self.world.actions = {'Ada': world_lib.COURT, 'Bo': 'rest'}
    self.world.close_year()
    self.assertEqual(sorted(self.world.courting), ['Ada'])

    self.world.actions = {'Ada': 'rest', 'Bo': world_lib.COURT}
    self.world.close_year()

    self.assertEqual(self.world.people['Ada'].partner, 'Bo')
    self.assertEmpty(sorted(self.world.courting))

  def test_two_who_court_each_other_by_name_pair(self):
    self._person('Ada', 20, (5, 5))
    self._person('Bo', 20, (5, 6)).sex = 'm'
    self._person('Cy', 20, (5, 7)).sex = 'm'
    self.assertIn(f'{world_lib.COURT_ONE} Bo', self.world.moves('Ada'))
    self.assertIn('Unpartnered near you: Bo', self.world.status('Ada'))
    self.world.actions = {
        'Ada': f'{world_lib.COURT_ONE} Cy',
        'Bo': world_lib.COURT,  # Anyone; but Ada wants Cy.
        'Cy': 'rest',
    }
    self.world.close_year()
    self.assertEqual(self.world.people['Ada'].partner, '')

    self.world.actions = {
        'Ada': 'rest',
        'Bo': 'rest',
        'Cy': f'{world_lib.COURT_ONE} Ada',
    }
    self.world.close_year()

    self.assertEqual(self.world.people['Ada'].partner, 'Cy')
    self.assertEqual(sorted(self.world.courting), ['Bo'])

  def test_one_courted_sees_who_courts_them_whatever_the_distance(self):
    self._person('Ada', 20, (5, 5))
    for i, name in enumerate(('Bo', 'Cy', 'Di', 'Ed')):
      self._person(name, 20, (5, 6 + i)).sex = 'm'
    far = self._person('Far', 20, (12, 20))
    far.sex = 'm'
    self.world.courting = {'Far': 'Ada'}

    self.assertIn(f'{world_lib.COURT_ONE} Far', self.world.moves('Ada'))
    self.assertIn('Far (courting you,', self.world.status('Ada'))

  def test_siblings_never_pair(self):
    self._person('Mum', 40, (5, 5))
    self._person('Ada', 20, (5, 5), parents=['Mum'])
    self._person('Bo', 20, (5, 6), parents=['Mum']).sex = 'm'
    self.world.actions = {
        'Mum': 'rest',
        'Ada': world_lib.COURT,
        'Bo': world_lib.COURT,
    }

    self.world.close_year()

    self.assertEqual(sorted(self.world.courting), ['Ada', 'Bo'])

  def test_strangers_come_to_a_nearly_empty_valley(self):
    self.world.settings['arrivals'] = {
        'below': 3,
        'chance': 1.0,
        'group': 2,
        'people': 'Hill people',
        'customs': [{'kind': 'belief', 'domain': 'sacred', 'text': 'Hills.'}],
    }
    self._person('Ada', 30, (5, 5))
    self.world.actions = {'Ada': 'rest'}

    outcome = self.world.close_year()

    self.assertLen(outcome.births, 2)
    self.assertEqual(self.world.history[-1]['arrivals'], 2)
    self.assertEqual(self.world.history[-1]['births'], 0)
    for params in outcome.births:
      self.assertEqual(params['arrived'], 0)
      self.assertEqual([c['text'] for c in params['customs']], ['Hills.'])
      self.assertIn('Hill people', self.world.queued[params['name']][0])
    self.world.actions = {name: 'rest' for name in self.world.people}

    self.assertEmpty(self.world.close_year().births)  # Three live now.

  def test_founders_hold_one_custom_of_a_kind_and_domain(self):
    norm = {'kind': 'norm', 'domain': 'sharing', 'text': 'Share.'}
    settings = {
        'seed': 0,
        'founders': 8,
        'founding_customs': [
            {**norm, 'held_by': 0.25},
            {**norm, 'params': {'form': culture_lib.KNOWN}, 'held_by': 0.5},
        ],
    }

    founders = scenarios.founders(settings)
    world = world_lib.World(settings)
    world.found(founders)

    forms = [
        [culture_lib.form_of(c) for c in founder['params']['customs']]
        for founder in founders
    ]
    self.assertEqual(
        sorted(world.people), sorted(f['params']['name'] for f in founders)
    )
    self.assertEqual(len(world.culture), 2)
    self.assertEqual(
        sorted(f[0] if f else '' for f in forms),
        [''] * 2 + [culture_lib.INTERNALIZED] * 2 + [culture_lib.KNOWN] * 4,
    )

  def test_passing_on_teaches_no_onlookers(self):
    self._person('Ada', 30, (5, 5))
    self._person('Cy', 30, (5, 6), given=())
    self.world.expecting = {'Ada': ['Ada']}
    self.world.actions = {
        'Ada': f'{world_lib.PASS_ON} {self.skill} {self.norm}',
        'Cy': 'rest',
    }

    outcome = self.world.close_year()

    self.assertLen(outcome.births, 1)
    self.assertEmpty(self._offers('Cy'))

  def test_a_parent_passes_on_only_what_it_was_given(self):
    self._person('Ada', 30, (5, 5), given=[self.norm])
    self.world.expecting = {'Ada': ['Ada']}
    self.world.actions = {
        'Ada': f'{world_lib.PASS_ON} {self.skill} {self.norm}'
    }

    (child,) = self.world.close_year().births

    self.assertEqual([c['id'] for c in child['customs']], [self.norm])

  def test_a_child_does_not_court(self):
    self._person('Kit', 8, (5, 5))
    self._person('Bo', 40, (5, 6)).sex = 'm'
    self.world.actions = {'Kit': world_lib.COURT, 'Bo': world_lib.COURT}

    self.world.close_year()

    self.assertEqual(self.world.people['Kit'].partner, '')
    self.assertEqual(sorted(self.world.courting), ['Bo'])

  def test_a_move_towards_one_is_not_towards_another_named_alike(self):
    self._person('Ada', 30, (5, 5))
    self._person('Kari', 30, (5, 6), given=()).food = 0.5
    self._person('Karito', 30, (6, 5), given=()).food = 2.0
    self.assertIn(f'{world_lib.SHARE} Kari', self.world.moves('Ada'))
    self.assertNotIn(f'{world_lib.SHARE} Karito', self.world.moves('Ada'))
    self.world.actions = {
        'Ada': f'{world_lib.SHARE} Karito',
        'Kari': 'rest',
        'Karito': 'rest',
    }

    self.world.close_year()

    self.assertEqual(self.world.people['Kari'].food, 0.0)
    self.assertEqual(self.world.people['Ada'].food, 4.0)

  def test_a_history_of_none_keeps_none(self):
    self.world.settings['history_kept'] = 0
    self.world.actions = {}
    self.world.close_year()
    self.assertEmpty(self.world.history)

  def test_an_elder_tells_a_story_to_all_within_sight(self):
    self._person('Old', 60, (5, 5))
    self._person('Kit', 6, (5, 6), given=())
    self._person('Far', 30, (9, 9), given=())
    self.assertIn(world_lib.TELL, self.world.moves('Old'))
    self.assertNotIn(world_lib.TELL, self.world.moves('Kit'))
    self.world.actions = {
        'Old': f'{world_lib.TELL} {self.norm} {self.skill}',
        'Kit': 'play',
        'Far': 'rest',
    }

    self.world.close_year()

    self.assertEqual(
        sorted(self._offers('Kit')),
        sorted([(self.norm, 'story', 'Old'), (self.skill, 'story', 'Old')]),
    )
    self.assertEmpty(self._offers('Far'))

  def test_a_preacher_preaches_one_custom(self):
    self._person('Ada', 30, (5, 5))
    self._person('Bo', 30, (5, 6), given=())
    self.world.actions = {
        'Ada': f'{world_lib.PREACH} {self.norm} {self.skill}',
        'Bo': 'rest',
    }

    self.world.close_year()

    self.assertEqual(self._offers('Bo'), [(self.norm, 'preaching', 'Ada')])

  def test_talk_between_dialects_garbles(self):
    culture = self.world.culture
    ours = culture.add(
        'dialect', 'language', 'Ours: ka ri.', {'words': ['ka', 'ri']}
    ).id
    theirs = culture.add(
        'dialect', 'language', 'Theirs: bo ha.', {'words': ['bo', 'ha']}
    ).id
    self.world.settings['garble'] = 1.0
    self._person('Ada', 30, (5, 5), given=[self.norm, ours]).speech = ours
    self._person('Bo', 30, (5, 6), given=[theirs]).speech = theirs
    self._person('Cy', 30, (6, 5), given=[ours]).speech = ours
    self.world.actions = {
        'Ada': f'{world_lib.PREACH} {self.norm}',
        'Bo': 'rest',
        'Cy': 'rest',
    }

    self.world.close_year()

    self.assertEqual(self._offers('Cy'), [(self.norm, 'preaching', 'Ada')])
    (garbled,) = self._offers('Bo')
    self.assertNotEqual(garbled[0], self.norm)
    self.assertEqual(self.world.culture.get(garbled[0]).parent, self.norm)
    self.assertEqual(self.world.history[-1]['tongues'], 2)

  def test_a_child_speaks_as_its_mother_with_drift(self):
    tongue = self.world.culture.add(
        'dialect',
        'language',
        'Ours: ka ri to ma.',
        {'words': ['ka', 'ri', 'to', 'ma']},
    ).id
    self._person('Ada', 30, (5, 5), given=[tongue]).speech = tongue
    self.world.settings['dialect_drift'] = 1.0
    self.world.expecting = {'Ada': ['Ada']}
    self.world.actions = {'Ada': world_lib.PASS_ON}

    (child,) = self.world.close_year().births

    speech = self.world.people[child['name']].speech
    self.assertEqual(self.world.culture.get(speech).parent, tongue)
    self.assertIn(speech, [c['id'] for c in child['customs']])

  def test_the_move_an_answer_names_first_is_the_move_made(self):
    allowed = [world_lib.GATHER, world_lib.REST, f'{world_lib.SHARE} Kari']
    self.assertEqual(
        world_lib.first_move('This year I will gather food, by c3.', allowed),
        world_lib.GATHER,
    )
    self.assertEqual(
        world_lib.first_move('I rest; then I gather food.', allowed),
        world_lib.REST,
    )
    self.assertEqual(
        world_lib.first_move('Share food with Kari!', allowed),
        f'{world_lib.SHARE} Kari',
    )
    self.assertEqual(
        world_lib.first_move('share food with Karito', allowed), ''
    )
    self.assertEqual(world_lib.first_move('I forage.', allowed), '')

  def test_a_person_asked_answers_first_then_moves(self):
    self._person('Ada', 30, (5, 5))
    self.assertIn(world_lib.ASK_FORM, self.world.moves(world_lib.EXPERIMENTER))
    self.world.actions = {
        'Ada': 'rest',
        world_lib.EXPERIMENTER: 'ask Ada: Why do you share?',
    }
    self.world.close_year()
    self.assertEqual(self.world.interview('Ada'), 'Why do you share?')

    self.world.record_action(world_lib.EXPERIMENTER, 'watch')
    self.assertTrue(self.world.may_act('Ada'))
    self.world.record_action('Ada', 'Because the old taught us.')
    self.assertTrue(self.world.may_act('Ada'))  # Now its move.
    self.assertFalse(self.world.year_complete())
    self.world.record_action('Ada', 'gather food')

    self.assertTrue(self.world.year_complete())
    self.assertEqual(
        self.world.transcripts,
        [{
            'year': 1,
            'name': 'Ada',
            'question': 'Why do you share?',
            'answer': 'Because the old taught us.',
        }],
    )
    self.assertIn(
        'Because the old', ' '.join(self.world.queued['Experimenter'])
    )

  def test_a_late_answer_does_nothing_and_never_answers_an_interview(self):
    self._person('Ada', 30, (5, 5))
    self.world.mark_asked('Ada', world_lib.MOVES)
    # The year closes at its deadline before Ada answers, and the
    # Experimenter asks her something for next year.
    self.world.actions = {world_lib.EXPERIMENTER: 'ask Ada: Why?'}
    self.world.close_year()

    self.world.record_action('Ada', 'gather food')  # Last year's answer.

    self.assertEqual(self.world.interview('Ada'), 'Why?')
    self.assertNotIn('Ada', self.world.actions)
    self.assertEmpty(self.world.transcripts)
    self.world.mark_asked('Ada', world_lib.INTERVIEW)
    self.world.record_action('Ada', 'Because.')
    self.assertEqual(self.world.transcripts[-1]['answer'], 'Because.')

  def test_the_experimenter_hears_of_a_question_gone_wrong(self):
    ada = self._person('Ada', 30, (5, 5))
    self.world.actions = {world_lib.EXPERIMENTER: 'ask Ada why'}
    self.world.close_year()
    self.assertIn('not understood', ' '.join(self.world.queued['Experimenter']))

    self.world.queued.clear()
    self.world.actions = {world_lib.EXPERIMENTER: 'ask Ada: Why?'}
    ada.food, ada.starving = 0.0, 1
    self.world.land.food[:] = 0.0
    self.world.close_year()  # Asked, then dead of hunger at the same close.

    self.assertIn(
        'Ada died before answering', ' '.join(self.world.queued['Experimenter'])
    )
    self.assertEmpty(self.world.interviews)

  def test_every_name_is_new(self):
    names = [self.world._new_name() for _ in range(5000)]  # pylint: disable=protected-access
    self.assertLen(set(names), len(names))


class LibraryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.world = world_lib.World({
        'seed': 0,
        'founders': 0,
        'channels': _FAITHFUL,
        'library': {'cost': 3.0, 'spacing': 5, 'shelf': 2, 'ruin_after': 3},
    })
    culture = self.world.culture
    self.reading = culture.add('skill', 'learning', 'Reading.', {'level': 1}).id
    self.norm = culture.add('norm', 'sharing', 'Share.').id
    self.belief = culture.add('belief', 'sacred', 'Hills.').id

  def _person(self, name, tile, given, age=30):
    person = world_lib.Person(
        name=name,
        born=self.world.year - age,
        sex='f',
        tile=tile,
        food=8.0,
        given=list(given),
    )
    self.world.people[name] = person
    return person

  def _year(self, **actions):
    self.world.queued.clear()
    self.world.actions = dict(actions)
    return self.world.close_year()

  def test_the_literate_found_a_library_and_write_on_its_shelf(self):
    ada = self._person('Ada', (5, 5), [self.reading, self.norm, self.belief])
    self._person('Bo', (5, 6), [self.norm])
    self.assertIn(world_lib.FOUND_LIBRARY, self.world.moves('Ada'))

    self._year(
        Ada=f'{world_lib.FOUND_LIBRARY} {self.reading} {self.norm}',
        Bo='rest',
    )

    (library,) = self.world.places.all()
    self.assertEqual((library.tile, library.shelf), ((5, 5), [self.norm]))
    self.assertEqual(ada.food, 8.0 - 3.0 - 1.0)  # Founding, then a year.
    self.assertIn('founds a library', ' '.join(self.world.queued['Bo']))
    self.assertEmpty(
        self.world.culture.prune(  # The shelf keeps its text.
            {self.reading, self.belief} | self.world.places.texts()
        )
    )
    self.assertNotIn(world_lib.FOUND_LIBRARY, self.world.moves('Bo'))

    self._year(Ada=f'{world_lib.WRITE} {self.reading} {self.belief}', Bo='rest')
    self.assertEqual(library.shelf, [self.norm, self.belief])

  def test_a_literate_reader_is_offered_the_shelf_and_nobody_watching(self):
    self._person('Ada', (5, 5), [self.reading, self.belief])
    self._year(Ada=f'{world_lib.FOUND_LIBRARY} {self.reading} {self.belief}')
    del self.world.people['Ada']  # The writer dies; the text stays.
    self._person('Cy', (5, 6), [self.reading])
    self._person('Bo', (6, 6), [])
    self.assertIn(world_lib.READ, self.world.moves('Cy'))

    self._year(Cy=f'{world_lib.READ} {self.reading}', Bo='rest')

    text = '\n'.join(self.world.queued['Cy'])
    offer = culture_lib.OFFER.search(text)
    assert offer is not None
    self.assertEqual(
        (offer.group('id'), offer.group('via')), (self.belief, 'reading')
    )
    self.assertEqual(self.world.history[-1]['rediscovered'], 1)
    self.assertNotIn('[custom', ' '.join(self.world.queued.get('Bo', [])))

  def test_one_who_cannot_read_reads_nothing(self):
    self._person('Ada', (5, 5), [self.reading, self.belief])
    self._year(Ada=f'{world_lib.FOUND_LIBRARY} {self.reading} {self.belief}')
    self._person('Bo', (5, 6), [self.norm])

    self._year(Ada='rest', Bo=f'{world_lib.READ} {self.norm}')

    self.assertIn('cannot make out', ' '.join(self.world.queued['Bo']))
    self.assertNotIn(self.belief, self.world.people['Bo'].given)

  def test_an_unused_library_falls_to_ruin_and_its_texts_crumble(self):
    self._person('Ada', (5, 5), [self.reading, self.belief])
    self._year(Ada=f'{world_lib.FOUND_LIBRARY} {self.reading} {self.belief}')
    (library,) = self.world.places.all()
    del self.world.people['Ada']

    years = 0
    while self.world.places.all() and years < 200:
      self._year()
      years += 1

    self.assertTrue(self.world.places.ruined(library, self.world.year))
    self.assertEmpty(self.world.places.all())
    self.assertGreater(years, 3)


class _PlaceTest(absltest.TestCase):
  """A world with a school and a shrine setting, and two customs."""

  def setUp(self):
    super().setUp()
    self.world = world_lib.World({
        'seed': 0,
        'founders': 0,
        'channels': _FAITHFUL,
        'school': {'cost': 3.0, 'spacing': 5, 'term': 2},
        'shrine': {'cost': 3.0, 'spacing': 5, 'every': 2},
    })
    self.norm = self.world.culture.add('norm', 'sharing', 'Share.').id
    self.belief = self.world.culture.add('belief', 'sacred', 'Hills.').id

  def _person(self, name, age, tile, given=()):
    person = world_lib.Person(
        name=name,
        born=self.world.year - age,
        sex='f',
        tile=tile,
        food=8.0,
        given=list(given),
    )
    self.world.people[name] = person
    return person

  def _year(self, **actions):
    self.world.queued.clear()
    self.world.actions = dict(actions)
    return self.world.close_year()

  def _offers(self, name):
    text = '\n'.join(self.world.queued.get(name, ()))
    return sorted(
        (m.group('id'), m.group('via'), m.group('source'))
        for m in culture_lib.OFFER.finditer(text)
    )


class SchoolTest(_PlaceTest):

  def test_a_school_teaches_any_child_nearby_for_a_term(self):
    self._person('Ada', 30, (5, 5), [self.norm, self.belief])
    self._person('Kit', 8, (5, 7))
    self.assertIn(world_lib.FOUND_SCHOOL, self.world.moves('Ada'))
    self.assertNotIn(world_lib.GO_TO_SCHOOL, self.world.moves('Kit'))

    self._year(
        Ada=f'{world_lib.FOUND_SCHOOL} {self.norm} {self.belief}', Kit='play'
    )
    self.assertIn(world_lib.GO_TO_SCHOOL, self.world.moves('Kit'))
    self.assertIn(world_lib.TEACH_AT_SCHOOL, self.world.moves('Ada'))
    self._year(Ada='rest', Kit=world_lib.GO_TO_SCHOOL)  # Within the term.

    self.assertEqual(
        self._offers('Kit'),
        sorted([(self.norm, 'school', 'Ada'), (self.belief, 'school', 'Ada')]),
    )

    self._year(Ada='rest', Kit='play')
    self._year(Ada='rest', Kit=world_lib.GO_TO_SCHOOL)  # The term is over.

    self.assertEmpty(self._offers('Kit'))
    self.assertIn('Nobody teaches', ' '.join(self.world.queued['Kit']))


class ShrineTest(_PlaceTest):

  def test_a_rite_falls_on_its_calendar_and_makes_norms_ones_own(self):
    known = self.world.culture.in_form(self.norm, culture_lib.KNOWN, 0).id
    self._person('Ada', 30, (5, 5), [self.norm, self.belief])
    self._person('Bo', 30, (5, 6), [known])
    self.assertIn(world_lib.BUILD_SHRINE, self.world.moves('Ada'))
    self._year(Ada=f'{world_lib.BUILD_SHRINE} {self.belief}', Bo='rest')
    self.assertNotIn(world_lib.JOIN_RITE, self.world.moves('Bo'))  # Year 1.
    self._year(Ada='rest', Bo='rest')
    self.assertIn(world_lib.JOIN_RITE, self.world.moves('Bo'))  # Year 2.
    self.assertIn('rite falls this year', self.world.status('Bo'))

    self._year(
        Ada=f'{world_lib.LEAD_RITE} {self.norm} {self.belief}',
        Bo=world_lib.JOIN_RITE,
    )

    self.assertEqual(
        self._offers('Bo'),
        sorted([(self.norm, 'rite', 'Ada'), (self.belief, 'rite', 'Ada')]),
    )
    self.assertEmpty(self._offers('Ada'))

  def test_joining_a_rite_teaches_no_onlookers(self):
    self._person('Ada', 30, (5, 5), [self.belief])
    self._person('Bo', 30, (5, 6), [self.norm])
    self._person('Cy', 30, (6, 6))
    self._year(Ada=f'{world_lib.BUILD_SHRINE} {self.belief}', Bo='rest')
    self._year(Ada='rest', Bo='rest', Cy='rest')

    self._year(
        Ada=f'{world_lib.LEAD_RITE} {self.belief}',
        Bo=f'{world_lib.JOIN_RITE} {self.norm}',
        Cy='rest',
    )

    self.assertEqual(self._offers('Cy'), [])

  def test_a_rite_nobody_leads_offers_nothing(self):
    self._person('Ada', 30, (5, 5), [self.belief])
    self._person('Bo', 30, (5, 6))
    self._year(Ada=f'{world_lib.BUILD_SHRINE} {self.belief}', Bo='rest')
    self._year(Ada='rest', Bo='rest')

    self._year(Ada='rest', Bo=world_lib.JOIN_RITE)

    self.assertEmpty(self._offers('Bo'))
    self.assertIn('Nobody leads the rite', ' '.join(self.world.queued['Bo']))


class ReleaseTest(_PlaceTest):

  def test_a_song_is_offered_to_all_within_sight_of_its_place(self):
    self._person('Ada', 30, (5, 5))
    self._person('Far', 30, (9, 9))
    self.assertIn(
        world_lib.RELEASE_FORM, self.world.moves(world_lib.EXPERIMENTER)
    )

    self._year(
        Ada='rest',
        Far='rest',
        Experimenter='release belief sacred as a song at 5,6: The sun sings.',
    )

    (offer,) = self._offers('Ada')
    custom = self.world.culture.get(offer[0])
    self.assertEqual((offer[1], offer[2]), ('release', None))
    self.assertEqual(
        (custom.kind, custom.text, custom.origin),
        ('belief', 'The sun sings.', 'release'),
    )
    self.assertEmpty(self._offers('Far'))
    self.assertEqual(self.world.history[-1]['released'], 1)
    # The year's record keeps the release and what the Experimenter was told.
    self.assertEqual(
        self.world.history[-1]['releases'],
        [{'id': custom.id, 'how': 'song', 'tile': [5, 6], 'heard': 1}],
    )
    self.assertEqual(
        self.world.history[-1]['reports'],
        [f'{custom.id} is sung at 5,6; 1 hear it.'],
    )
    self._year(Ada='rest', Far='rest', Experimenter='watch')
    self.assertEqual(self.world.history[-1]['reports'], [])

  def test_a_drought_withers_only_the_land_within_reach(self):
    self._person('Ada', 30, (5, 5))
    food = self.world.land.food.copy()

    self._year(Ada='rest', Experimenter='send a drought at 5,5')

    record = self.world.history[-1]['disasters'][0]
    self.assertEqual(
        (record['kind'], record['tile'], record['by'], record['near']),
        ('drought', [5, 5], world_lib.EXPERIMENTER, 1),
    )
    self.assertLess(self.world.land.food[5, 7], food[5, 7])  # Two steps.
    self.assertGreater(self.world.land.food[5, 9], food[5, 9] * 0.5)  # Four.
    self.assertIn(
        'A drought strikes at 5,5: 1 lived within reach, 0 died',
        self.world.history[-1]['reports'][0],
    )

  def test_a_plague_takes_lives_within_reach_only(self):
    for i in range(12):
      self._person(f'Near{i}', 60, (5, 5))
    self._person('Far', 60, (12, 12))

    self._year(Experimenter='send a plague at 5,5')

    record = self.world.history[-1]['disasters'][0]
    self.assertGreater(record['dead'], 0)
    self.assertIn('Far', self.world.people)
    self.assertLen(
        [n for n in self.world.people if n.startswith('Near')],
        12 - record['dead'],
    )

  def test_raiders_take_the_stores_within_reach(self):
    self._person('Ada', 30, (5, 5))
    self._person('Far', 30, (12, 12))

    self._year(Ada='rest', Far='rest', Experimenter='send a raid at 5,6')

    self.assertGreater(self.world.history[-1]['disasters'][0]['food_lost'], 0)
    if 'Ada' in self.world.people:
      self.assertEqual(self.world.people['Ada'].food, 0.0)
    self.assertGreater(self.world.people['Far'].food, 0.0)

  def test_a_fire_burns_a_library_and_its_texts(self):
    library = self.world.places.found(
        places_lib.LIBRARY, (5, 5), 'Ada', self.world.year
    )
    self.world.places.write(library, self.norm, self.world.year)
    self.world.land.terrain[5, 5] = land_lib.TERRAIN.index('forest')

    self._year(Experimenter='send a fire at 5,5')

    self.assertEqual(library.shelf, [])
    record = self.world.history[-1]['disasters'][0]
    self.assertEqual(record['texts_lost'], 1)
    self.assertGreater(record['food_lost'], 0)
    # Burned to nothing; only the seed bank regrew before the year closed.
    land = self.world.land
    self.assertAlmostEqual(
        float(land.food[5, 5]), land.seed_bank * land.capacity((5, 5))
    )

  def test_the_experimenter_brings_migrants_to_a_place(self):
    self.world.settings['arrivals'] = {
        'people': 'Hill people',
        'customs': [{'kind': 'belief', 'domain': 'sacred', 'text': 'Hills.'}],
        'food': 6.0,
        'group': 2,
        'below': 0,  # No trickle of strangers: only those brought.
    }

    outcome = self._year(Experimenter='bring migrants at 3,4')

    came = [self.world.people[p['name']] for p in outcome.births]
    self.assertLen(came, 4)
    self.assertEqual({p.tile for p in came}, {(3, 4)})
    self.assertEqual({p.descent for p in came}, {'Hill people'})
    self.assertEqual(self.world.history[-1]['arrivals'], 4)
    self.assertIn(
        '4 of the Hill people come to 3,4', self.world.history[-1]['reports'][0]
    )

  def test_a_disaster_or_migrants_not_understood_change_nothing(self):
    for move in ('send a meteor at 1,1', 'send a fire', 'bring migrants'):
      self._year(Experimenter=move)
      self.assertEqual(self.world.history[-1]['disasters'], [])
      self.assertIn('not understood', self.world.history[-1]['reports'][0])
    self.assertIn(
        world_lib.DISASTER_FORM, self.world.moves(world_lib.EXPERIMENTER)
    )
    self.assertIn(
        world_lib.MIGRANTS_FORM, self.world.moves(world_lib.EXPERIMENTER)
    )

  def test_the_world_sends_disasters_by_chance(self):
    self.world.settings['disasters'] = {'chance': 1.0, 'kinds': ['drought']}

    for _ in range(3):
      self._year()

    records = [r['disasters'] for r in self.world.history[-3:]]
    self.assertEqual([len(r) for r in records], [1, 1, 1])
    self.assertEqual({r[0]['by'] for r in records}, {'world'})
    self.assertEqual({r[0]['kind'] for r in records}, {'drought'})

  def test_a_text_is_left_on_a_shelf_at_its_place(self):
    self._year(
        Experimenter='release skill craft level 3 as a text at 2,3: Knapping.'
    )

    (place,) = self.world.places.all()
    (custom_id,) = place.shelf
    self.assertEqual(place.tile, (2, 3))
    self.assertEqual(self.world.culture.get(custom_id).params, {'level': 3})

  def test_a_release_that_would_read_as_protocol_is_refused(self):
    self._person('Ada', 30, (5, 5))

    self._year(
        Ada='rest',
        Experimenter='release belief sacred as a song at 5,5: Gods |params={x}',
    )

    self.assertEqual(self.world.history[-1]['released'], 0)
    self.assertEmpty(self._offers('Ada'))
    self.assertIn('not understood', ' '.join(self.world.queued['Experimenter']))

  def test_a_song_needs_its_channel_and_a_skill_its_bounds(self):
    del self.world.channels[culture_lib.RELEASE]
    self._year(Experimenter='release belief sacred as a song at 5,5: Hm.')
    self.assertIn('No song', ' '.join(self.world.queued['Experimenter']))

    self._year(
        Experimenter='release skill craft level 99 as a text at 1,1: Ha.'
    )
    (place,) = self.world.places.all()
    self.assertEqual(
        self.world.culture.get(place.shelf[0]).params['level'],
        culture_lib.MAX_LEVEL,
    )

  def test_a_release_not_understood_changes_nothing(self):
    self._year(Experimenter='release wisdom sacred as a song at 1,1: Hm.')
    self._year(Experimenter='release a song')

    self.assertEqual(self.world.history[-1]['released'], 0)
    self.assertIn('not understood', ' '.join(self.world.queued['Experimenter']))


class CultureTest(absltest.TestCase):

  def test_the_same_content_is_the_same_variant(self):
    culture = culture_lib.Culture()
    first = culture.add('belief', 'sacred', 'The river  is a god.')
    again = culture.add('belief', 'sacred', 'The river is a god.')
    other = culture.add('belief', 'sacred', 'The river is a god.', {'x': 1})

    self.assertEqual(again.id, first.id)
    self.assertNotEqual(other.id, first.id)
    self.assertEqual(culture.created(), 2)

  def test_a_norm_in_each_form(self):
    culture = culture_lib.Culture()
    own = culture.add('norm', 'sharing', 'Share.')

    known = culture.in_form(own.id, culture_lib.KNOWN, 3)

    self.assertEqual(culture_lib.form_of(own.spec()), culture_lib.INTERNALIZED)
    self.assertEqual(culture_lib.form_of(known.spec()), culture_lib.KNOWN)
    self.assertEqual((known.parent, known.text), (own.id, own.text))
    self.assertEqual(culture.in_form(known.id, culture_lib.KNOWN, 4), known)
    self.assertEqual(
        culture.in_form(known.id, culture_lib.INTERNALIZED, 4).id,
        culture.in_form(own.id, culture_lib.INTERNALIZED, 4).id,
    )

  def test_copying_errors_lose_know_how_and_care_adds_it(self):
    culture = culture_lib.Culture()
    skill = culture.add('skill', 'craft', 'Knapping.', {'level': 4})
    rng = np.random.default_rng(0)

    erred = {
        culture.copy(skill.id, rng, 1.0, 1).params['level'] for _ in range(20)
    }
    improved = culture.copy(skill.id, rng, 0.0, 1, improve=1.0)

    self.assertEqual(erred, {3})
    self.assertEqual(improved.params['level'], 5)

  def test_a_dialect_drifts_a_word_at_a_time(self):
    culture = culture_lib.Culture()
    words = ['ka', 'ri', 'to', 'ma']
    tongue = culture.add(
        'dialect', 'language', 'The tongue: ka ri to ma.', {'words': words}
    )

    drifted = culture.copy(tongue.id, np.random.default_rng(1), 1.0, 1)

    self.assertEqual(
        sum(a != b for a, b in zip(words, drifted.params['words'])), 1
    )
    self.assertEqual(
        drifted.text, f'The tongue: {" ".join(drifted.params["words"])}.'
    )
    self.assertEqual(culture_lib.intelligibility(tongue, drifted), 0.75)

  def test_marriage_rules(self):
    tongue = {'within': culture_lib.TONGUE}
    people = {'within': culture_lib.DESCENT}
    self.assertTrue(culture_lib.accepts(None, 0.0, False))
    self.assertTrue(culture_lib.accepts(tongue, 0.75, False))
    self.assertFalse(culture_lib.accepts(tongue, 0.5, True))
    self.assertTrue(culture_lib.accepts(people, 0.0, True))
    self.assertFalse(culture_lib.accepts(people, 1.0, False))

  def test_forgotten_content_may_arise_again_under_a_new_id(self):
    culture = culture_lib.Culture()
    first = culture.add('belief', 'sacred', 'The river is a god.')
    culture.prune(keep=())

    again = culture.add('belief', 'sacred', 'The river is a god.')

    self.assertNotEqual(again.id, first.id)
    restored = culture_lib.Culture()
    restored.set_state(culture.get_state())
    self.assertEqual(
        restored.add('belief', 'sacred', 'The river is a god.').id, again.id
    )


if __name__ == '__main__':
  absltest.main()
