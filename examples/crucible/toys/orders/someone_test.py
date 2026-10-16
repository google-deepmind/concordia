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


"""Arc 4: a clause for "someone" asks one member only."""

import copy

from absl.testing import absltest
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.orders import mock as mock_lib
from examples.crucible.toys.orders import rules
from examples.crucible.toys.orders import toy as toy_lib

SOMEONE_WATCH = 'Someone keep watch; everyone else build.'


def _toy(scenario: str = 'voyage') -> toy_lib.OrdersToy:
  toy = toy_lib.OrdersToy(
      model_setup.mock_model(), toys.load_data('orders'), scenario
  )
  toy.world.shift = 0
  return toy


def _watched_turns(order: str, turns: int) -> list[list[str]]:
  toy = _toy('night_watch')
  watchers = []
  for turn in range(turns):
    toy.resolve(toy.commander, order if turn == 0 else toy_lib.HOLD)
    actions = toy.world.log[-1]['actions']
    watchers.append([m for m, a in actions.items() if a == 'watch'])
  return watchers


class SomeoneTest(absltest.TestCase):

  def test_someone_goes_to_the_freshest_willing(self):
    toy = _toy()
    toy.resolve(toy.commander, SOMEONE_WATCH)
    last = toy.world.log[-1]
    self.assertEqual(last['actions']['Ash'], 'watch')  # first on a tie
    self.assertEqual(last['actions']['Bea'], 'build')
    self.assertEqual(last['why']['Ash'], 'Someone keep watch')
    toy = _toy()
    toy.world.fatigue['Ash'] = 2
    toy.resolve(toy.commander, SOMEONE_WATCH)
    self.assertEqual(toy.world.log[-1]['actions']['Bea'], 'watch')
    self.assertEqual(toy.world.log[-1]['actions']['Ash'], 'build')

  def test_someone_keeps_the_watch_where_a_name_lapses(self):
    # Night watch: one turn on watch wears anyone out.
    someone = _watched_turns(SOMEONE_WATCH, 6)
    named = _watched_turns('Bea, keep watch; everyone else build.', 6)
    self.assertTrue(all(len(w) == 1 for w in someone))
    self.assertEqual(someone[:2], [['Ash'], ['Bea']])  # passed on as they tire
    self.assertIn([], named)  # worn out, Bea rests and the watch lapses

  def test_no_one_willing_means_no_one_does_it(self):
    crew = rules.CrewRule(['Ash', 'Bea', 'Cy', 'Dex'], {'tired_at': 3}, {})
    ship = rules.Ship(fatigue={'Ash': 4, 'Bea': 3, 'Cy': 0, 'Dex': 0})
    takes = {'Ash': 'always', 'Bea': 'fresh', 'Cy': 'never', 'Dex': 'never'}
    self.assertEqual(crew.volunteer(ship, takes), 'Ash')  # worn, but bold
    self.assertIsNone(crew.volunteer(ship, takes, excluded=['Ash']))
    ship.fatigue['Bea'] = 1
    self.assertEqual(crew.volunteer(ship, takes), 'Bea')  # the freshest
    toy = _toy('contrary')  # Dex alone: he never takes it up
    toy.world.order = 'Someone keep watch.'
    self.assertIsNone(toy.volunteer(toy.world, 1))

  def test_only_orders_for_someone_carry_its_law_and_who_takes_it(self):
    toy = _toy()
    self.assertFalse(toy.asks_someone('Bea, keep watch; everyone else build.'))
    self.assertTrue(toy.asks_someone('If a leak opens, one of you repair it.'))
    toy.world.order = 'Ash, build; whoever is fresh, keep watch.'
    self.assertEqual(toy.volunteer(toy.world, 1), 'Bea')  # Ash is named


class RotaTest(absltest.TestCase):

  def _toy(self, off: dict[str, list[int]]) -> toy_lib.OrdersToy:
    data = copy.deepcopy(toys.load_data('orders'))
    data['scenarios']['voyage']['off_duty'] = off
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    return toy

  def test_off_duty_rests_and_reads_nothing(self):
    toy = self._toy({'Ash': [1]})
    toy.world.fatigue['Ash'] = 3
    receipt = toy.resolve(toy.commander, 'Ash, build; everyone else build.')
    last = toy.world.log[-1]
    self.assertEqual(last['actions']['Ash'], 'rest')
    self.assertEqual(last['why']['Ash'], toy_lib.OFF_DUTY)
    self.assertEqual(toy.world.fatigue['Ash'], 1)
    self.assertIn('Ash is off duty.', str(receipt))
    self.assertEqual(toy.view()['crew'][0]['off'], False)  # back on turn 2

  def test_someone_covers_the_rota_where_a_name_lapses(self):
    def watchers(order):
      toy = self._toy({'Ash': [2, 3]})
      out = []
      for turn in range(4):
        toy.resolve(toy.commander, order if turn == 0 else toy_lib.HOLD)
        acts = toy.world.log[-1]['actions']
        out.append([m for m, a in acts.items() if a == 'watch'])
      return out

    self.assertEqual(
        watchers(SOMEONE_WATCH), [['Ash'], ['Bea'], ['Bea'], ['Ash']]
    )
    self.assertEqual(
        watchers('Ash, keep watch; everyone else build.'),
        [['Ash'], [], [], ['Ash']],
    )

  def test_a_hurt_member_is_off_duty_and_someone_passes_over_them(self):
    data = copy.deepcopy(toys.load_data('orders'))
    voyage = data['scenarios']['voyage']
    voyage['events'] = {1: 'squall', 2: 'squall'}
    voyage['hazards'] = {
        'quiet': {}, 'squall': {'strikes_unwatched': 4, 'injures': 2}
    }  # fmt: skip
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    toy.world.fatigue['Ash'] = 2  # the most worn
    receipt = toy.resolve(toy.commander, SOMEONE_WATCH)
    self.assertIn('Ash is hurt', str(receipt))
    self.assertTrue(toy.rota.off_duty(toy.world, 'Ash', 2))
    self.assertEqual(toy.world.log[-1]['actions']['Bea'], 'watch')
    toy.resolve(toy.commander, toy_lib.HOLD)  # turn 2: watched, no one hurt
    self.assertEqual(toy.world.injured, {'Ash': 2})
    self.assertFalse(toy.rota.off_duty(toy.world, 'Ash', 3))


class HeardTest(absltest.TestCase):

  def _toy(self, off: dict[str, list[int]]) -> toy_lib.OrdersToy:
    data = copy.deepcopy(toys.load_data('orders'))
    data['scenarios']['voyage']['off_duty'] = off
    data['scenarios']['voyage']['events'] = {}
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    return toy

  def test_an_order_given_off_duty_is_not_heard(self):
    toy = self._toy({'Bea': [2]})
    toy.resolve(toy.commander, 'Build.')
    receipt = toy.resolve(toy.commander, 'Rest.')
    self.assertIn('Bea is off duty and misses it.', str(receipt))
    toy.resolve(toy.commander, toy_lib.HOLD)  # turn 3: Bea is back
    last = toy.world.log[-1]
    self.assertEqual(last['actions']['Ash'], 'rest')
    self.assertEqual(last['why']['Bea'], 'Build')  # the order she heard
    self.assertNotEqual(last['actions']['Bea'], 'rest')
    bea = next(m for m in toy.view()['crew'] if m['name'] == 'Bea')
    self.assertEqual(bea['heard'], 'Build.')
    self.assertIn(
        'Still on an earlier order: Bea',
        toy.action_spec(toy.commander).call_to_action,
    )
    toy.resolve(toy.commander, 'Rest.')  # said again, it reaches her
    self.assertEqual(toy.world.log[-1]['actions']['Bea'], 'rest')
    self.assertEqual(toy.world.heard, {})

  def test_someone_needs_to_have_heard_it(self):
    toy = self._toy({'Ash': [2]})
    toy.resolve(toy.commander, 'Build.')
    toy.resolve(toy.commander, SOMEONE_WATCH)  # Ash is off and misses it
    toy.resolve(toy.commander, toy_lib.HOLD)
    last = toy.world.log[-1]
    # Ash is the freshest (he rested off duty) but never heard "someone".
    self.assertEqual(last['why']['Ash'], 'Build')
    self.assertNotEqual(last['actions']['Ash'], 'watch')
    self.assertEqual(last['actions']['Bea'], 'watch')  # heard it; willing


class BosunTest(absltest.TestCase):

  def _toy(self, bosun: str, off=None) -> toy_lib.OrdersToy:
    data = copy.deepcopy(toys.load_data('orders'))
    voyage = data['scenarios']['voyage']
    voyage.pop('crew_present', None)
    voyage.update(bosun=bosun, events={}, off_duty=off or {})
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    return toy

  def test_each_temperament_passes_the_order_on_its_own_way(self):
    order = 'Build; Cy, keep watch.'
    relayed = {}
    for bosun in ('Ash', 'Bea', 'Cy', 'Dex'):
      toy = self._toy(bosun)
      toy.resolve(toy.commander, order)
      relayed[bosun] = toy.world.relayed
    self.assertEqual(relayed['Cy'], order)  # literal: word for word
    self.assertEqual(relayed['Ash'], 'Build.')  # bold: no watch clause
    self.assertEqual(relayed['Bea'], 'Build; Cy, keep watch; repair leaks.')
    self.assertEqual(relayed['Dex'], "don't build; Cy, don't keep watch.")

  def test_a_relay_goes_by_what_a_clause_asks(self):
    # Cycle 83: by vocabulary, "tired" read as a clause to rest, "leak" in a
    # condition as a repair, and "Don't repair" as repairing.
    mock = mock_lib.OrdersMock(toys.load_data('orders'))
    self.assertEqual(
        mock.relay('If tired, build; keep watch.', 'drops_rest'),
        'If tired, build.',
    )
    self.assertEqual(
        mock.relay("Don't keep watch.", 'drops_rest'), "Don't keep watch."
    )
    self.assertEqual(
        mock.relay('Build; if a leak opens, keep watch.', 'adds_repair'),
        'Build; if a leak opens, keep watch; repair leaks.',
    )
    self.assertEqual(
        mock.relay("Build; don't repair.", 'adds_repair'),
        "Build; don't repair.",
    )

  def test_the_crew_follow_the_bosuns_words_the_bosun_the_commanders(self):
    toy = self._toy('Ash')
    receipt = toy.resolve(toy.commander, 'Build; Cy, keep watch.')
    self.assertIn('Ash passes it on: "Build."', str(receipt))
    self.assertEqual(toy.world.log[-1]['actions']['Cy'], 'build')  # no watch
    self.assertEqual(toy.view()['bosun'], 'Ash')
    cards = {m['name']: m for m in toy.view()['crew']}
    self.assertTrue(cards['Cy']['via_bosun'])
    self.assertFalse(cards['Ash']['via_bosun'])
    prompt = toy.action_spec(toy.commander).call_to_action
    self.assertIn('Ash passed it on as: "Build."', prompt)
    self.assertNotIn('Still on an earlier order', prompt)

  def test_with_the_bosun_off_duty_the_order_goes_out_as_given(self):
    toy = self._toy('Ash', off={'Ash': [1]})
    receipt = toy.resolve(toy.commander, 'Build; Cy, keep watch.')
    self.assertIn('the order goes out as you gave it', str(receipt))
    self.assertEqual(toy.world.log[-1]['actions']['Cy'], 'watch')


class WindTest(absltest.TestCase):

  def _toy(self, bosun=None) -> toy_lib.OrdersToy:
    data = copy.deepcopy(toys.load_data('orders'))
    voyage = data['scenarios']['voyage']
    voyage.pop('crew_present', None)
    voyage.update(
        events={1: 'storm'},
        hazards={'quiet': {}, 'storm': {'garbles': True}},
    )
    if bosun:
      voyage['bosun'] = bosun
    toy = toy_lib.OrdersToy(model_setup.mock_model(), data, 'voyage')
    toy.world.shift = 0
    return toy

  def test_an_order_given_in_a_storm_carries_only_its_first_clause(self):
    toy = self._toy()
    receipt = toy.resolve(toy.commander, 'Cy, keep watch; everyone else build.')
    self.assertIn('only "Cy, keep watch." carries', str(receipt))
    last = toy.world.log[-1]
    self.assertEqual(last['actions']['Cy'], 'watch')
    self.assertEqual(last['why']['Ash'], 'none')  # never heard "build"
    self.assertNotEqual(last['actions']['Bea'], 'build')
    toy.resolve(toy.commander, 'Build.')  # quiet now: heard in full
    self.assertEqual(toy.world.log[-1]['actions']['Ash'], 'build')

  def test_the_bosun_relays_what_the_wind_left(self):
    toy = self._toy(bosun='Cy')
    toy.resolve(toy.commander, 'Build; Bea, keep watch.')
    self.assertEqual(toy.world.relayed, 'Build.')


class AppointTest(absltest.TestCase):

  def test_free_play_names_a_bosun_by_order(self):
    toy = toy_lib.OrdersToy(
        model_setup.mock_model(), toys.load_data('orders'), 'free'
    )
    receipt = toy.resolve(toy.commander, 'Ash is bosun; build; Cy, keep watch.')
    self.assertIn('Ash is bosun now.', str(receipt))
    self.assertEqual(toy.view()['bosun'], 'Ash')
    self.assertNotIn('watch', toy.world.relayed)  # the bold drop the watch
    receipt = toy.resolve(toy.commander, 'No bosun; build.')
    self.assertIn('No bosun now.', str(receipt))
    self.assertIsNone(toy.view()['bosun'])

  def test_a_scene_without_it_ignores_the_word(self):
    toy = toy_lib.OrdersToy(
        model_setup.mock_model(), toys.load_data('orders'), 'voyage'
    )
    toy.resolve(toy.commander, 'Ash is bosun; build.')
    self.assertIsNone(toy.view()['bosun'])


class AuditTest(absltest.TestCase):
  """Cases from the cycle 76 rule audit."""

  def _scene(self, name: str) -> toy_lib.OrdersToy:
    toy = toy_lib.OrdersToy(
        model_setup.mock_model(), toys.load_data('orders'), name
    )
    toy.world.shift = 0
    return toy

  def test_one_member_takes_up_someone_through_a_bosun(self):
    toy = self._scene('word_of_mouth')  # Bea relays with "repair leaks"
    toy.resolve(toy.commander, SOMEONE_WATCH)
    actions = toy.world.log[-1]['actions']
    self.assertEqual(sum(a == 'watch' for a in actions.values()), 1)

  def test_a_bosun_named_at_the_start_keeps_his_name(self):
    toy = self._scene('chain_of_command')  # Cy relays word for word
    toy.resolve(toy.commander, 'Cy keeps watch; everyone else build.')
    self.assertEqual(toy.world.relayed, 'Cy keeps watch; everyone else build.')
    actions = toy.world.log[-1]['actions']
    self.assertEqual(actions['Cy'], 'watch')
    self.assertNotEqual(actions['Ash'], 'watch')

  def test_no_stale_relay_once_the_bosun_is_off_duty(self):
    toy = self._scene('chain_of_command')  # Cy off duty on turns 5-7
    toy.resolve(toy.commander, 'Build; Bea, keep watch.')
    for _ in range(3):
      toy.resolve(toy.commander, toy_lib.HOLD)
    toy.resolve(toy.commander, 'Build.')  # for turn 5: goes out as given
    self.assertEqual(toy.world.relayed, '')
    self.assertNotIn(
        'passed it on as', toy.action_spec(toy.commander).call_to_action
    )

  def test_storm_cut_words_are_not_an_earlier_order(self):
    toy = WindTest()._toy()
    toy.resolve(toy.commander, 'Cy, keep watch; everyone else build.')
    prompt = toy.action_spec(toy.commander).call_to_action
    self.assertNotIn('Still on an earlier order', prompt)
    self.assertIn('only "Cy, keep watch." carried', prompt)
    cards = {m['name']: m for m in toy.view()['crew']}
    self.assertTrue(cards['Ash']['carried'])


if __name__ == '__main__':
  absltest.main()
