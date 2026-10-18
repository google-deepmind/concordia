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

"""Tests for coal treaties between peoples."""

from absl.testing import absltest
from concordia.agents import entity_agent
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time import model_setup
from examples.deep_time import simulation
from examples.deep_time.components import pact_mind
from examples.deep_time.components import resolution
from examples.deep_time.prefabs import polity as polity_prefab
from examples.deep_time.world import data
from examples.deep_time.world import publishing
from examples.deep_time.world import treaties as treaties_lib
from concordia.typing import entity as entity_lib

NAMES = {'amber': 'Amber folk', 'basalt': 'Basalt clans', 'cedar': 'Cedar'}


class _Said:
  """What an answer did, shaped like a world event (its text)."""

  def __init__(self, text: str):
    self.text = text


def _answer(planet, key: str, name: str, verb: str = 'accept') -> list[_Said]:
  """The partner answers the offer awaiting it on its own menu."""
  actor = actors_lib.PolityActor(name, key, planet, 0.15)
  option = next(o for o in actor.options() if o.startswith(f'{verb} the '))
  return [_Said(actor.resolve(option))]


def _treaties() -> treaties_lib.Treaties:
  return treaties_lib.Treaties(data.load('civ.yaml')['treaties'])


class TreatiesTest(absltest.TestCase):

  def test_signed_treaties_bind_both_and_lapse(self):
    t = _treaties()
    proposal = t.propose('amber', 'basalt', NAMES)
    assert proposal is not None
    self.assertIn('Amber folk propose a coal treaty', proposal.text)
    self.assertFalse(t.can_propose('cedar', 'basalt'))  # One offer at a time.
    self.assertIsNone(t.propose('amber', 'amber', NAMES))
    self.assertIs(t.answer('basalt', True, tick=3), proposal)
    self.assertEqual(t.bound('amber'), ['basalt'])
    self.assertEqual(t.bound('basalt'), ['amber'])
    self.assertFalse(t.can_propose('basalt', 'amber'))
    self.assertEqual(t.check({'amber': t.coal_cap}, tick=5), [])
    # The generation after signing is grace (its coal was already burning).
    self.assertEqual(t.check({'amber': 99.0}, tick=4), [])
    # Binding for `duration` generations: ticks 5 to 4 + duration.
    t.check({}, tick=4 + t.duration)
    self.assertEqual(t.bound('amber'), ['basalt'])
    t.check({}, tick=5 + t.duration)
    self.assertEqual(t.bound('amber'), [])
    self.assertEqual(t.signed, 1)

  def test_crossed_offers_and_stale_answers_are_refused(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES, tick=0)
    self.assertFalse(t.can_propose('basalt', 'amber'))
    t.propose('cedar', 'amber', NAMES, tick=0)
    # An answer written to the Cedar's offer cannot settle someone else's.
    self.assertIsNone(t.answer('basalt', True, tick=1, proposer='cedar'))
    self.assertIsNotNone(t.answer('basalt', True, tick=1, proposer='amber'))
    # Unanswered offers lapse.
    t.check({}, tick=1 + t.offer_ticks)
    self.assertIsNone(t.pending_for('amber'))

  def test_both_breakers_are_recorded(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES)
    t.answer('basalt', True, tick=0)
    broken = t.check({'amber': 9.0, 'basalt': 9.0}, tick=2)
    self.assertEqual(broken, [('amber', 'basalt'), ('basalt', 'amber')])
    self.assertEqual(t.breaches, {'amber': 1, 'basalt': 1})

  def test_a_refused_offer_waits_before_it_is_repeated(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES, tick=0, kind='fish')
    t.answer('basalt', False, tick=1)
    wait = t.refusal_ticks
    self.assertGreater(wait, 0)
    self.assertIsNone(
        t.propose('amber', 'basalt', NAMES, tick=wait, kind='fish')
    )
    # Other kinds, other peoples, and the refuser's own offer back are not
    # held back.
    self.assertTrue(t.can_propose('amber', 'basalt', 'coal', tick=2))
    self.assertTrue(t.can_propose('cedar', 'basalt', 'fish', tick=2))
    self.assertTrue(t.can_propose('basalt', 'amber', 'fish', tick=2))
    restored = _treaties()
    restored.set_state(t.get_state())
    self.assertFalse(restored.can_propose('amber', 'basalt', 'fish', tick=wait))
    self.assertIsNotNone(
        restored.propose('amber', 'basalt', NAMES, tick=1 + wait, kind='fish')
    )

  def test_declined_offers_bind_nobody(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES)
    t.answer('basalt', False, tick=1)
    self.assertEqual(t.bound('amber'), [])
    self.assertIsNone(t.answer('basalt', True, tick=2))

  def test_breach_ends_treaty_and_embargoes_trade(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES)
    t.answer('basalt', True, tick=0)
    broken = t.check({'amber': 1.0, 'basalt': t.coal_cap + 1}, tick=2)  # Day 1.
    self.assertEqual(broken, [('basalt', 'amber')])
    self.assertEqual(t.breaches, {'basalt': 1})
    self.assertEqual(t.bound('amber'), [])
    self.assertTrue(t.embargoed('amber', 'basalt'))
    self.assertFalse(t.can_propose('amber', 'basalt'))
    payload = t.payload(NAMES, tick=2)
    self.assertEqual(payload['embargoes'], [['Amber folk', 'Basalt clans']])
    self.assertEqual(payload['breaches'], {'Basalt clans': 1})
    t.check({}, tick=2 + t.embargo_ticks)
    self.assertFalse(t.embargoed('amber', 'basalt'))

  def test_state_round_trip(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES)
    t.answer('basalt', True, tick=0)
    t.propose('cedar', 'amber', NAMES)
    other = _treaties()
    other.set_state(t.get_state())
    self.assertEqual(other.get_state(), t.get_state())
    self.assertEqual(other.pending_for('amber'), t.pending_for('amber'))


class TreatyFlowTest(absltest.TestCase):

  def test_peoples_offer_sign_and_break_treaties(self):
    context = game_lib.new_game('great_acceleration')
    planet = context.planet
    civ, trade, treaties = planet.civ, planet.trade, planet.treaties
    assert civ is not None and trade is not None and treaties is not None
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    option = 'propose a coal treaty to the Basalt clans'
    self.assertNotIn(option, amber.options())  # Needs diplomacy, both sides.
    civ.polities['amber'].techs.append('diplomacy')
    self.assertNotIn(option, amber.options())
    civ.polities['basalt'].techs.append('diplomacy')
    self.assertIn(option, amber.options())
    self.assertLessEqual(len(amber.options()), 26)
    text = amber.resolve(option)
    self.assertIn('Amber folk propose a coal treaty', text)
    self.assertEqual(amber.drain_outbox(), [('diplomacy.basalt.amber', text)])
    self.assertNotIn(option, amber.options())
    # The Basalt clans' treaty mind answers (here the mock's rule).
    # The World GM's sheet for its treaty mind: the offer awaiting it.
    self.assertEqual(
        polity_prefab.treaty_offer(context, 'basalt'),
        pact_mind.offer_sheet(
            text, 'Amber folk (coal treaty)', treaties.question
        ),
    )
    self.assertEqual(polity_prefab.treaty_offer(context, 'amber'), '')
    events = _answer(planet, 'basalt', 'Basalt clans')
    self.assertIn('The Basalt clans sign the coal treaty', events[0].text)
    self.assertEqual(treaties.bound('basalt'), ['amber'])
    # The Basalt clans burn far more coal than the cap: the treaty breaks.
    civ.polities['basalt'].emissions['coal'] = 50.0
    self.assertEqual(treaties.check({'basalt': 50.0}, planet.tick + 1), [])
    treaties.check({'basalt': 50.0}, planet.tick + 2)
    self.assertTrue(treaties.embargoed('amber', 'basalt'))
    context.publish_state()
    record = context.bus.latest('civ.treaties')
    assert record is not None
    self.assertIn(
        'Embargoes: Amber folk and Basalt clans. Treaties broken: Basalt'
        ' clans 1.',
        publishing.render_treaties(record),
    )
    civ.ledger.deposit('basalt', 'grain', 10.0, 'test')
    self.assertEqual(
        trade._ship('basalt', 'amber', 'grain', 5, 1.0),  # pylint: disable=protected-access
        [
            'The embargo voids a sale of 5 grain from the Basalt clans to the'
            ' Amber folk.'
        ],
    )
    self.assertNotEmpty(trade._ship('basalt', 'cedar', 'grain', 5, 1.0))  # pylint: disable=protected-access


class FishingQuotaTest(absltest.TestCase):

  def test_quotas_bind_catches_and_breaches_embargo(self):
    t = _treaties()
    proposal = t.propose('amber', 'basalt', NAMES, kind='fish')
    assert proposal is not None
    self.assertIn('propose a fishing quota', proposal.text)
    t.answer('basalt', True, tick=0)
    self.assertEqual(t.bound('amber', 'fish'), ['basalt'])
    self.assertEqual(t.bound('amber'), [])  # No coal treaty.
    # A coal treaty between the same peoples can still be offered.
    self.assertTrue(t.can_propose('amber', 'basalt', 'coal'))
    cap = t.caps['fish']
    self.assertEqual(t.check({'fish': {'amber': cap}, 'coal': {}}, tick=2), [])
    broken = t.check({'fish': {'basalt': cap + 1}, 'coal': {}}, tick=3)
    self.assertEqual(broken, [('basalt', 'amber')])
    self.assertTrue(t.embargoed('amber', 'basalt'))

  def test_a_late_answer_cannot_sign_another_kind(self):
    t = _treaties()
    t.propose('amber', 'basalt', NAMES, tick=0, kind='fish')
    self.assertIsNone(t.answer('basalt', True, tick=1, kind='coal'))
    self.assertIsNotNone(t.answer('basalt', True, tick=1, kind='fish'))

  def test_peoples_offer_quotas_over_shared_seas(self):
    context = game_lib.new_game('great_acceleration')
    planet = context.planet
    civ = planet.civ
    assert civ is not None and planet.treaties is not None
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += ['diplomacy', 'fishing_boats']
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    option = 'propose a fishing quota to the Basalt clans'
    self.assertIn(option, amber.options())
    self.assertIn('fishing quota', amber.resolve(option))
    events = _answer(planet, 'basalt', 'Basalt clans')
    self.assertIn('sign the fishing quota', events[0].text)
    self.assertIn(
        'bound by fishing quotas', polity_prefab.storehouse(context, 'basalt')
    )
    context.publish_state()
    record = context.bus.latest('civ.treaties')
    assert record is not None
    self.assertIn('Fishing quotas', publishing.render_treaties(record))


class RunoffPactTest(absltest.TestCase):

  def test_runoff_pacts_are_a_kind_from_data(self):
    t = _treaties()
    spec = t.kinds['runoff']
    self.assertEqual(spec.name, 'runoff pact')
    self.assertEqual(spec.metric, 'runoff')
    self.assertIn('chemistry', spec.requires)
    self.assertEqual(
        t.terms('runoff'),
        f'washes at most {spec.cap:g} fertiliser runoff to the sea',
    )
    self.assertEqual(t.terms('coal'), f'burns at most {t.coal_cap:g} coal')
    t.propose('amber', 'basalt', NAMES, tick=0, kind='runoff')
    t.answer('basalt', True, tick=0, kind='runoff')
    self.assertEqual(t.bound('basalt', 'runoff'), ['amber'])
    self.assertEqual(t.check({'runoff': {'amber': spec.cap}}, tick=2), [])
    self.assertEqual(
        t.check({'runoff': {'amber': spec.cap + 0.1}}, tick=3),
        [('amber', 'basalt')],
    )

  def test_runoff_pacts_are_listed_once_someone_knows_chemistry(self):
    context = game_lib.new_game('great_acceleration')
    civ = context.planet.civ
    assert civ is not None

    def listed() -> str:
      context.publish_state()
      record = context.bus.latest('civ.treaties')
      assert record is not None
      return publishing.render_treaties(record)

    self.assertIn('Fishing quotas', listed())  # Everyone has boats here.
    self.assertNotIn('Runoff pacts', listed())
    civ.polities['cedar'].techs.append('chemistry')
    self.assertIn('Runoff pacts', listed())

  def test_peoples_offer_sign_and_keep_runoff_pacts(self):
    context = game_lib.new_game('great_acceleration')
    planet = context.planet
    civ = planet.civ
    assert civ is not None and planet.treaties is not None
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    option = 'propose a runoff pact to the Basalt clans'
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += ['diplomacy']
    self.assertNotIn(option, amber.options())  # Nobody knows chemistry.
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += ['chemistry']
    self.assertIn(option, amber.options())
    self.assertIn('runoff pact', amber.resolve(option))
    # The reply names its kind, and settles only the runoff offer.
    events = _answer(planet, 'basalt', 'Basalt clans')
    self.assertIn('sign the runoff pact', events[0].text)
    self.assertIn('washes at most', events[0].text)
    self.assertIn(
        'bound by runoff pacts with the Amber folk: at most',
        polity_prefab.storehouse(context, 'basalt'),
    )
    # Fertilised fields that wash too much break it.
    civ.ledger.deposit('basalt', 'fertiliser', 50.0, 'test')
    civ.set_allocation('basalt', {'fertilised_farm': 1.0})
    for _ in range(3):
      planet.advance()
    self.assertEqual(planet.treaties.bound('basalt', 'runoff'), [])
    self.assertTrue(planet.treaties.embargoed('amber', 'basalt'))
    context.publish_state()
    record = context.bus.latest('civ.treaties')
    assert record is not None
    self.assertIn('Runoff pacts', publishing.render_treaties(record))


class HumanDiplomacyTest(absltest.TestCase):

  def test_a_person_answers_offers_on_their_own_menu(self):
    context = game_lib.new_game('great_acceleration')
    planet = context.planet
    civ = planet.civ
    assert civ is not None and planet.treaties is not None
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += ['diplomacy']
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    self.assertFalse(any(o.startswith('accept') for o in basalt.options()))
    planet.propose_treaty('amber', 'basalt', 'coal')
    options = basalt.options()
    accept = 'accept the coal treaty offered by the Amber folk'
    self.assertEqual(
        options[:2],
        [accept, 'decline the coal treaty offered by the Amber folk'],
    )
    self.assertIn('sign the coal treaty', basalt.resolve(accept))
    self.assertEqual(planet.treaties.bound('basalt'), ['amber'])
    # Announced like a treaty mind's reply: to the proposer, and as news.
    topics = [topic for topic, _ in basalt.drain_outbox()]
    self.assertEqual(topics, ['diplomacy.amber.basalt', 'civ.events'])
    self.assertFalse(any(o.startswith('accept') for o in basalt.options()))
    # A people a model plays answers on its menu too (its treaty mind's
    # counsel is in its context): every answer is an action.
    planet.propose_treaty('basalt', 'amber', 'fish')
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    self.assertTrue(any(o.startswith('accept') for o in amber.options()))

  def test_every_people_has_its_treaty_mind_and_sees_the_terms(self):
    sim, context = simulation.build_simulation(
        'great_acceleration',
        model_setup.make_models('mock'),
        humans={'Basalt clans': lambda request: ''},
    )
    components = {
        e.name: e.get_all_context_components()
        for e in sim.get_entities()
        if isinstance(e, entity_agent.EntityAgent)
    }
    # Whoever plays it: a person or a model.
    self.assertIn('Treaty mind', components['Basalt clans'])
    self.assertIn('Treaty mind', components['Amber folk'])
    civ = context.planet.civ
    assert civ is not None
    for key in ('amber', 'basalt'):
      civ.polities[key].techs += ['diplomacy']
    context.planet.propose_treaty('amber', 'basalt', 'coal')
    self.assertIn(
        'An offer awaits our answer: The Amber folk propose a coal treaty',
        polity_prefab.storehouse(context, 'basalt'),
    )
    self.assertNotIn(
        'An offer awaits', polity_prefab.storehouse(context, 'amber')
    )


class ResolutionOrderTest(absltest.TestCase):

  def test_rival_offers_resolve_the_same_whatever_order_threads_finish(self):
    pending = []
    for joined in (
        (
            'Amber folk: propose a fishing quota to the Basalt clans\n'
            'Cedar league: propose a fishing quota to the Basalt clans'
        ),
        (
            'Cedar league: propose a fishing quota to the Basalt clans\n'
            'Amber folk: propose a fishing quota to the Basalt clans'
        ),
    ):
      context = game_lib.new_game('great_acceleration')
      planet = context.planet
      civ = planet.civ
      assert civ is not None and planet.treaties is not None
      for key in ('amber', 'basalt', 'cedar'):
        civ.polities[key].techs += ['diplomacy']
        context.add_actor(
            actors_lib.PolityActor(civ.polities[key].name, key, planet, 0.15)
        )
      component = resolution.ActorResolution(context)
      component.pre_observe(f'{resolution.PUTATIVE_EVENT_TAG} {joined}')
      component.pre_act(
          entity_lib.ActionSpec(
              call_to_action='?', output_type=entity_lib.OutputType.RESOLVE
          )
      )
      offer = planet.treaties.pending_for('basalt')
      assert offer is not None
      pending.append(offer.proposer)
    self.assertEqual(pending, ['amber', 'amber'])


class MockTreatyTest(absltest.TestCase):

  def test_mock_signs_only_what_it_can_keep(self):
    question = (
        'The Amber folk propose a coal treaty to the Basalt clans: for 10'
        ' generations each burns at most 4 coal a generation, so the air'
        ' stays kind. Should Basalt clans sign it?'
    )
    self.assertStartsWith(
        mock_language_model.treaty_answer(
            'Last generation we burned 2.0 coal. ' + question
        ),
        'accept',
    )
    self.assertStartsWith(
        mock_language_model.treaty_answer(
            'Last generation we burned 9.5 coal. ' + question
        ),
        'decline',
    )
    self.assertIsNone(mock_language_model.treaty_answer('Should we?'))
    quota = (
        'The Amber folk propose a fishing quota to the Basalt clans: for 10'
        ' generations each catches at most 5 fish a generation. Should Basalt'
        ' clans sign it?'
    )
    self.assertStartsWith(
        mock_language_model.treaty_answer(
            'Last generation we caught 8.0 fish. ' + quota
        ),
        'decline',
    )

  def test_mock_signs_a_runoff_pact_only_if_its_fields_keep_it(self):
    question = (
        'The Amber folk propose a runoff pact to the Basalt clans: for 10'
        ' generations each washes at most 0.3 fertiliser runoff to the sea a'
        ' generation, so the coastal seas breathe again. Should Basalt clans'
        ' sign it?'
    )
    self.assertStartsWith(mock_language_model.treaty_answer(question), 'accept')
    heavy = (
        'Last generation 1.2 of our fertiliser washed down the rivers to the'
        ' sea. '
        + question
    )
    self.assertStartsWith(mock_language_model.treaty_answer(heavy), 'decline')
    # Just over the cap is over the cap: runoff is printed to hundredths.
    edge = heavy.replace('1.2 of our', '0.31 of our')
    self.assertStartsWith(mock_language_model.treaty_answer(edge), 'decline')

  def test_mock_offers_runoff_pacts_over_choking_seas(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. Labor: farm the'
        ' land 30%. Dead zones choke {dead}% of the coastal seas, and fish'
        ' neither live nor breed there.'
    )
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'propose a runoff pact to the Basalt clans',
        'c': 'keep the current balance of labor',
    }
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(sheet.format(dead=25), options), options['b'])
    self.assertNotEqual(policy(sheet.format(dead=10), options), options['b'])
    # Never a pact it would break itself.
    heavy = sheet.format(dead=25) + (
        ' Last generation 0.70 of our fertiliser washed down the rivers to'
        ' the sea.'
    )
    self.assertNotEqual(policy(heavy, options), options['b'])

  def test_mock_studies_diplomacy_once_it_burns_coal(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 20.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. '
    )
    options = {
        'a': 'learn masonry (22 research)',
        'b': 'learn diplomacy and treaties (26 research)',
        'c': actors_lib.KEEP_LABOR,
    }
    policy = mock_language_model.polity_policy
    self.assertEqual(policy(sheet, options), options['a'])
    burning = sheet + 'Last generation we burned 2.0 coal.'
    self.assertEqual(policy(burning, options), options['b'])


if __name__ == '__main__':
  absltest.main()
