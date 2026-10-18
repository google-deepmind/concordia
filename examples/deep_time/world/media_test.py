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

"""Tests for the Gazette: broadsheets, credibility and consent."""

from absl.testing import absltest
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time.components import press_mind
from examples.deep_time.world import data
from examples.deep_time.world import media as media_lib
from examples.deep_time.world import publishing
import numpy as np

CONFIG = data.load('media.yaml')
CALM = media_lib.Evidence(
    relative_co2=1.0,
    restless=0.0,
    attacked=frozenset(),
    fed={'amber': 1.0, 'basalt': 1.0, 'cedar': 1.0},
)
SMOKY = media_lib.Evidence(
    relative_co2=1.2, restless=0.0, attacked=frozenset(), fed=CALM.fed
)


def _game():
  context = game_lib.new_game('great_acceleration')
  planet = context.planet
  civ, gazette = planet.civ, planet.gazette
  assert civ is not None and gazette is not None
  return context, planet, civ, gazette


class GazetteTest(absltest.TestCase):

  def test_stance_is_the_first_keyword(self):
    stances = CONFIG['stances']
    self.assertEqual(
        media_lib.parse_stance('Clean air: but coal and progress...', stances),
        'clean',
    )
    self.assertEqual(
        media_lib.parse_stance('We say coal and progress!', stances),
        'progress',
    )
    self.assertIsNone(media_lib.parse_stance('Nothing to say.', stances))

  def test_headlines_drop_markdown_and_labels(self):
    self.assertEqual(
        media_lib.headline(
            '  **Headline:** “Coal Fuels Our Future!”  \nThe plough turns.'
        ),
        'Coal Fuels Our Future',
    )
    self.assertEqual(
        media_lib.headline('Smoke Kills. Burn less.'), 'Smoke Kills'
    )

  def test_true_broadsheets_earn_credibility_and_sway_everyone(self):
    _, _, civ, gazette = _game()
    gazette.commission('cedar', tick=0)
    sheet = gazette.publish(
        'cedar', 'Clean air: Smoke Kills. Burn less.', 'g0', SMOKY, civ
    )
    assert sheet is not None
    self.assertEqual(
        (sheet.stance, sheet.title, sheet.votes), ('clean', 'Smoke Kills', 2)
    )
    # Judged true, it sways with credibility 1.5.
    self.assertAlmostEqual(gazette.credibility('cedar'), 1.5)
    for people in civ.polities.values():
      self.assertAlmostEqual(people.consent, 1.0 - 1.5 * gazette.sway)
    # The same claim under a clean sky is called false.
    gazette.commission('cedar', tick=3)
    sheet = gazette.publish('cedar', 'clean air: Doom!', 'g3', CALM, civ)
    assert sheet is not None
    self.assertEqual(sheet.votes, -2)
    self.assertAlmostEqual(gazette.credibility('cedar'), 1.0)
    self.assertEqual(gazette.forum.get_karma()['Cedar league'], 0)

  def test_progress_rings_false_from_a_struck_people(self):
    _, _, civ, gazette = _game()
    for people in civ.polities.values():
      people.consent = 0.5
    struck = media_lib.Evidence(
        relative_co2=1.0,
        restless=0.0,
        attacked=frozenset({'basalt'}),
        fed=CALM.fed,
    )
    gazette.commission('basalt', tick=0)
    sheet = gazette.publish(
        'basalt', 'coal and progress: Coal Feeds Us.', 'g0', struck, civ
    )
    assert sheet is not None
    self.assertEqual(sheet.votes, -2)
    # Credibility 0.5: the broadsheet moves consent half as far.
    self.assertAlmostEqual(civ.polities['amber'].consent, 0.5 + 0.5 * 0.08)

  def test_presses_need_paper_rest_and_a_commission(self):
    _, _, civ, gazette = _game()
    self.assertFalse(gazette.can_print('amber', 0, paper=0.5))
    self.assertTrue(gazette.can_print('amber', 0, paper=1.0))
    gazette.commission('amber', 0)
    self.assertFalse(gazette.can_print('amber', 1, paper=5.0))
    self.assertIsNone(
        gazette.publish('basalt', 'clean air: x', 'g1', CALM, civ)
    )
    gazette.publish('amber', 'no stance at all', 'g1', CALM, civ)
    self.assertNotIn('amber', gazette.pending)
    self.assertFalse(gazette.can_print('amber', 2, paper=5.0))
    self.assertTrue(gazette.can_print('amber', 3, paper=5.0))

  def test_blame_fades_and_state_round_trips(self):
    context, _, civ, gazette = _game()
    gazette.blame_smoke('amber', civ)
    self.assertAlmostEqual(civ.polities['amber'].consent, 1.0 - gazette.blame)
    gazette.step(civ)
    self.assertAlmostEqual(
        civ.polities['amber'].consent,
        1.0 - gazette.blame * (1.0 - gazette.relax),
    )
    gazette.commission('cedar', 0)
    gazette.publish('cedar', 'clean air: Smoke Kills.', 'g0', SMOKY, civ)
    state = context.get_state()
    other, _, other_civ, other_gazette = _game()
    other.set_state(state)
    self.assertEqual(other_gazette.get_state(), gazette.get_state())
    self.assertEqual(
        other_civ.polities['amber'].consent, civ.polities['amber'].consent
    )

  def test_consent_scales_coal_mining(self):
    _, _, civ, _ = _game()
    people = civ.polities['basalt']
    temps = np.full(48, 15.0)
    forest = np.zeros(48)
    full = civ._yield('coal', people, temps, 1.0, forest)  # pylint: disable=protected-access
    people.consent = 0.5
    half = civ._yield('coal', people, temps, 1.0, forest)  # pylint: disable=protected-access
    self.assertAlmostEqual(half, 0.5 * full)
    self.assertEqual(
        civ._yield('ore', people, temps, 1.0, forest),  # pylint: disable=protected-access
        civ._yield('ore', civ.polities['basalt'], temps, 1.0, forest),  # pylint: disable=protected-access
    )


class PamphletTest(absltest.TestCase):

  def test_coal_burners_buy_pamphlets_that_sway_half(self):
    context, planet, civ, gazette = _game()
    basalt = actors_lib.PolityActor('Basalt clans', 'basalt', planet, 0.15)
    self.assertNotIn(actors_lib.PAY_PAMPHLETS, basalt.options())
    civ.polities['basalt'].emissions['coal'] = 3.0  # Burns coal.
    self.assertIn(actors_lib.PAY_PAMPHLETS, basalt.options())
    silver = civ.stock('basalt', 'silver')
    self.assertIn(
        'pay the coal guilds', basalt.resolve(actors_lib.PAY_PAMPHLETS)
    )
    self.assertAlmostEqual(
        civ.stock('basalt', 'silver'), silver - gazette.pamphlet_silver
    )
    self.assertIn('paid by the coal guilds', gazette.question_for('basalt'))
    for people in civ.polities.values():
      people.consent = 0.5
    # No stance named: the guilds' side is taken.
    sheet = gazette.publish(
        'basalt', 'Coal Feeds Us. Burn on.', 'g0', CALM, civ
    )
    assert sheet is not None
    self.assertEqual(sheet.stance, 'progress')
    credibility = gazette.credibility('basalt')
    self.assertAlmostEqual(
        civ.polities['amber'].consent,
        0.5 + gazette.sway * credibility * gazette.pamphlet_sway,
    )
    del context

  def test_mock_press_writes_for_whoever_pays(self):
    text = mock_language_model.press_answer(
        'Our land is poor in coal. You are the press of Amber folk, paid by'
        ' the coal guilds. Write a pamphlet.'
    )
    assert text is not None
    self.assertStartsWith(text, 'coal and progress:')


class GazetteFlowTest(absltest.TestCase):

  def test_peoples_print_and_everyone_reads_the_gazette(self):
    context, planet, civ, _ = _game()
    amber = actors_lib.PolityActor('Amber folk', 'amber', planet, 0.15)
    press = lambda: context.bus.latest('civ.cedar.press')  # pylint: disable=unnecessary-lambda-assignment
    self.assertNotIn(actors_lib.PRINT_BROADSHEET, amber.options())
    civ.polities['amber'].techs += ['glassmaking', 'printing']
    self.assertNotIn(actors_lib.PRINT_BROADSHEET, amber.options())  # No paper.
    # The menu is curated, but a fresh unlock always shows.
    self.assertIn('put more labor into: make paper', amber.options())
    # The Cedar league start with the press and two sheets of paper.
    cedar = actors_lib.PolityActor(
        'Cedar league', 'cedar', planet, 0.15, press=press
    )
    self.assertIn('paper 2.0', civ.sheet('cedar'))
    self.assertIn(actors_lib.PRINT_BROADSHEET, cedar.options())
    self.assertLessEqual(len(cedar.options()), actors_lib.MAX_OPTIONS)
    self.assertEqual(
        cedar.resolve(actors_lib.PRINT_BROADSHEET),
        'The Cedar league set their presses running.',
    )
    self.assertAlmostEqual(civ.stock('cedar', 'paper'), 1.0)
    self.assertNotIn(actors_lib.PRINT_BROADSHEET, cedar.options())  # Writing.
    mind = press_mind.PressMind(
        mock_language_model.DeepTimeMockModel(), context, 'basalt'
    )
    self.assertEqual(mind._make_pre_act_value(), '')  # pylint: disable=protected-access
    context.bus.publish(
        'civ.cedar.press', 'clean air: Smoke Kills. Burn less coal.'
    )
    # The printers' sheet reaches the Gazette only when the people publishes
    # it (an action).
    self.assertIsNone(context.bus.latest('civ.cedar.propaganda'))
    self.assertIn(actors_lib.PUBLISH_BROADSHEET, cedar.options())
    news = cedar.resolve(actors_lib.PUBLISH_BROADSHEET)
    self.assertIn('The Cedar league print a broadsheet', news)
    self.assertIn(
        ('civ.cedar.propaganda', 'clean air: Smoke Kills. Burn less coal.'),
        cedar.drain_outbox(),
    )
    self.assertNotIn(actors_lib.PUBLISH_BROADSHEET, cedar.options())
    context.publish_state()
    message = context.bus.latest('media.gazette')
    assert message is not None
    text = publishing.render_gazette(message)
    # Under the clear starting sky the claim is judged false (half sway).
    self.assertIn('Cedar league, for clean air: "Smoke Kills" (-2 votes)', text)
    self.assertIn('Consent for coal: Amber folk 96%', text)
    self.assertIn('Cedar league 0.5.', text)

  def test_land_strikes_lower_consent(self):
    _, planet, civ, gazette = _game()
    assert planet.backlash is not None
    planet.backlash.pressure[civ.polities['basalt'].regions[0]] = 5.0
    planet.advance()
    self.assertIn('basalt', planet.struck)
    self.assertLess(civ.polities['basalt'].consent, 1.0)
    self.assertIn('basalt', planet.broadsheet_evidence().attacked)
    del gazette


class MockPressTest(absltest.TestCase):

  def test_mock_press_argues_from_its_land(self):
    question = 'You are the press of Cedar league. Write a broadsheet.'
    clean = mock_language_model.press_answer(
        'Our land is poor in coal and ore. ' + question
    )
    progress = mock_language_model.press_answer(
        'Our land is rich in coal. ' + question
    )
    assert clean is not None and progress is not None
    self.assertStartsWith(clean, 'clean air:')
    self.assertStartsWith(progress, 'coal and progress:')
    self.assertIsNone(mock_language_model.press_answer('Hello.'))


if __name__ == '__main__':
  absltest.main()
