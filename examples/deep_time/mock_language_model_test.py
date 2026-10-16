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

"""Tests for the deterministic mock model."""

from absl.testing import absltest
from concordia.document import interactive_document
from examples.deep_time import mock_language_model

OPTIONS = (
    'wait and save omega',
    'seed white daisies [2 omega]',
    'seed black daisies [2 omega]',
    'seed bright clouds (cools the planet) [4 omega]',
    'spread basalt dust (draws carbon from the air) [3 omega]',
    'release a dark haze (warms the planet) [4 omega]',
)


def _context(temp: float, omega: float) -> str:
  return (
      f'Thermometer: Global mean temperature {temp:.1f} C (habitable band 10'
      ' to 35 C); sunlight 1.50; planetary albedo 0.60. [measured at tick 3]\n'
      f'Omega meter: Omega {omega:.1f}; biodiversity 0.40; stability 0.90.'
  )


def _choose(temp: float, omega: float) -> str:
  model = mock_language_model.DeepTimeMockModel()
  doc = interactive_document.InteractiveDocument(model)
  doc.statement(_context(temp, omega))
  idx = doc.multiple_choice_question('Choose one.', OPTIONS)
  return OPTIONS[idx]


class MockModelTest(absltest.TestCase):

  def test_parses_options_of_last_question(self):
    prompt = 'Question: old\n  (a) x\nQuestion: new\n  (a) yes\n  (b) no\n'
    self.assertEqual(
        mock_language_model.last_question_options(prompt),
        {'a': 'yes', 'b': 'no'},
    )

  def test_steward_cools_when_hot(self):
    self.assertTrue(_choose(33.0, 9.0).startswith('seed bright clouds'))
    self.assertTrue(_choose(33.0, 3.0).startswith('seed white'))

  def test_steward_draws_down_carbon_when_hot_and_co2_high(self):
    model = mock_language_model.DeepTimeMockModel()
    doc = interactive_document.InteractiveDocument(model)
    doc.statement(_context(33.0, 9.0) + ' CO2 x1.20 of the start (greenhouse')
    choice = OPTIONS[doc.multiple_choice_question('Choose one.', OPTIONS)]
    self.assertTrue(choice.startswith('spread basalt dust'))

  def test_steward_rouses_volcanoes_to_thaw(self):
    options = OPTIONS + ('rouse the deep volcanoes (pumps carbon) [5 omega]',)

    def choose(temp: float, omega: float) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          'Overarching goal: Thaw the planet before tick 25.\n'
          + _context(temp, omega)
      )
      return options[doc.multiple_choice_question('Choose.', options)]

    self.assertTrue(choose(-20.0, 6.0).startswith('rouse the deep volcanoes'))
    self.assertEqual(choose(-20.0, 3.0), 'wait and save omega')
    self.assertEqual(choose(5.0, 9.0), 'wait and save omega')

  def test_steward_reads_the_air_and_tips_the_ice_with_a_haze(self):
    swap = 'watch the air gauge instead of the census'
    options = OPTIONS + ('rouse the deep volcanoes (pumps carbon) [5 omega]',)
    tip = (
        ' A dark haze may tip the ice from x1.3 of the start, and almost surely'
        ' does above x2.0 under a clear sky; below x1.3 it only warms a turn.'
    )
    ash = (
        '\n- tick 9: A great eruption at Old Smoky Shield (magnitude 3) veils'
        ' the sky with ash and dusts the bare ground with minerals.'
    )

    def choose(co2, omega, when=9, extra='', offered=options) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      air = '' if co2 is None else f'\nCO2 x{co2:.2f} of the start.{tip}'
      doc.statement(
          'Overarching goal: Thaw the planet before tick 22.\n'
          + _context(-25.0, omega)
          + air
          + extra
          + f'\n[observation] Tick {when} of 22.'
      )
      return offered[doc.multiple_choice_question('Choose.', offered)]

    self.assertEqual(choose(None, 6.0, offered=options + (swap,)), swap)
    self.assertTrue(choose(1.5, 6.0).startswith('rouse'))  # Not sure yet.
    self.assertTrue(choose(2.1, 6.0).startswith('release a dark haze'))
    self.assertEqual(choose(2.1, 3.0), 'wait and save omega')  # Saves for it.
    self.assertEqual(choose(2.1, 6.0, extra=ash), 'wait and save omega')
    self.assertTrue(choose(1.5, 6.0, when=17).startswith('release a dark'))
    # Volcanoes still resting: it waits rather than doing something else.
    self.assertEqual(choose(1.0, 6.0, offered=OPTIONS), 'wait and save omega')

  def test_advisor_briefs_a_frozen_world_on_the_tipping_point(self):
    def brief(co2: float | None, extra: str = '') -> str:
      if co2 is None:
        air = ''
      else:
        air = (
            f' CO2 x{co2:.2f} of the start. A dark haze may tip the ice from'
            ' x1.3 of the start, and almost surely does above x2.0.'
        )
      # Like the real Advisor's prompt: readings and moves, not the goal.
      answer = mock_language_model.advisor_answer(
          'You are the advisor of Gaia. End with "Recommend:" followed by'
          ' exactly one of: wait and save omega, seed bright clouds, release'
          ' a dark haze, spread basalt dust, rouse the deep volcanoes.\n'
          + _context(-25.0, 5.0)
          + ' Ice covers 99% of the planet.'
          + air
          + extra
          + '\nTick 9 of 22.'
      )
      assert answer is not None
      return answer

    self.assertIn('watch the air gauge, not the empty census', brief(None))
    self.assertIn('too thin', brief(1.1))
    self.assertTrue(brief(1.1).endswith('rouse the deep volcanoes.'))
    self.assertIn('may tip', brief(1.6))
    self.assertTrue(brief(2.2).endswith('Recommend: release a dark haze.'))
    ash = '\n- tick 9: A great eruption at Ash Peak veils the sky with ash.'
    self.assertTrue(brief(2.2, ash).endswith('wait and save omega.'))

  def test_steward_warms_when_cold(self):
    self.assertTrue(_choose(11.0, 9.0).startswith('release a dark haze'))
    self.assertTrue(_choose(11.0, 2.5).startswith('seed black'))

  def test_steward_waits_when_comfortable(self):
    self.assertEqual(_choose(22.0, 20.0), 'wait and save omega')

  def test_steward_keeps_a_living_world_living(self):
    protect = 'protect the wild woods (the peoples may not fell them) [4 omega]'
    offered = OPTIONS + (protect,)

    def choose(temp: float, life: float) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          "Overarching goal: a seed ship must leave while the planet's"
          ' biodiversity is at least 0.17.\n'
          f'Thermometer: Global mean temperature {temp:.1f} C (habitable band 5'
          ' to 25 C). [measured at tick 3]\n'
          f'Omega meter: Omega 9.0; biodiversity {life:.2f} (the seas at 100%'
          ' of their life); stability 0.90.'
      )
      return offered[doc.multiple_choice_question('Choose.', offered)]

    self.assertTrue(choose(18.0, 0.18).startswith('seed bright clouds'))
    self.assertEqual(choose(13.0, 0.18), protect)  # Cool enough: the woods.
    self.assertEqual(choose(13.0, 0.25), 'wait and save omega')  # Thriving.

  def test_steward_keeps_the_air_under_a_goals_cap(self):
    swap = 'watch the air gauge instead of the census'

    def choose(temp, co2, offered=OPTIONS + (swap,)) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      air = '' if co2 is None else f'\nCO2 x{co2:.2f} of the start.'
      doc.statement(
          'Overarching goal: the air must hold less than 1.3 times the'
          ' starting CO2.\n'
          + _context(temp, 9.0)
          + air
      )
      return offered[doc.multiple_choice_question('Choose.', offered)]

    self.assertEqual(choose(22.0, None), swap)  # The air gauge first.
    self.assertTrue(choose(22.0, 1.24).startswith('spread basalt dust'))
    self.assertEqual(choose(22.0, 1.1), 'wait and save omega')
    # Too cold to take the dust's cooling: it waits.
    self.assertEqual(choose(16.0, 1.24), 'wait and save omega')

  def test_advisor_answer(self):
    hot = mock_language_model.advisor_answer(
        'You are the advisor of Gaia.\n' + _context(30.0, 9.0)
    )
    assert hot is not None
    self.assertTrue(hot.endswith('Recommend: seed bright clouds.'))
    calm = mock_language_model.advisor_answer(
        'You are the advisor of Gaia.\n' + _context(22.0, 9.0)
    )
    assert calm is not None
    self.assertIn('wait and save omega', calm)
    self.assertIsNone(mock_language_model.advisor_answer(_context(22, 1)))

  def test_advisor_reads_the_trend_in_the_earlier_readings(self):
    def brief(temp: float, earlier: str) -> str:
      out = mock_language_model.advisor_answer(
          'You are the advisor of Gaia.\n'
          + _context(temp, 9.0)
          + f' Earlier readings you acted on: {earlier} C (oldest first).'
      )
      assert out is not None
      return out

    # Cold and still falling: warm it.
    self.assertIn('dark haze', brief(14.0, '22.0, 19.0, 17.0'))
    self.assertIn('falling 3 C a tick', brief(14.0, '22.0, 19.0, 17.0'))
    # Cold but already warming fast: it needs no push.
    warming = brief(14.0, '6.0, 7.0, 9.0')
    self.assertIn('warming on its own', warming)
    self.assertTrue(warming.endswith('Recommend: wait and save omega.'))
    # Readings that swing back and forth are too unsteady to act on early.
    unsteady = brief(15.0, '27.0, 14.0, 26.0')
    self.assertIn('too unsteady', unsteady)
    self.assertTrue(unsteady.endswith('Recommend: wait and save omega.'))
    # Out of the band, act whatever the swing.
    self.assertIn('bright clouds', brief(37.0, '27.0, 14.0, 26.0'))

  def test_steward_acts_early_on_fresh_advice_only(self):
    briefing = (
        '\nAdvisor briefing: Overheating. Recommend: seed bright clouds.'
        ' [tick 5, {age}]'
    )

    def choose(age: str) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(_context(28.5, 9.0) + briefing.format(age=age))
      return OPTIONS[doc.multiple_choice_question('Choose one.', OPTIONS)]

    self.assertTrue(choose('1 tick old').startswith('seed bright clouds'))
    self.assertTrue(choose('fresh').startswith('seed bright clouds'))
    self.assertEqual(choose('5 ticks old'), 'wait and save omega')
    self.assertEqual(_choose(28.5, 9.0), 'wait and save omega')

  def test_chronicle_covers_only_new_ticks(self):
    prompt = (
        'You are the Chronicler of this planet.\n'
        'Previous chapter: The Quiet Years (ticks 1-3): calm. [tick 4, fresh]\n'
        'Tick 3 of 40 (-3 Gyr). Old news blight.\n'
        'Tick 4 of 40 (-2.9 Gyr). A white-petal blight sweeps the equator.\n'
        'Tick 5 of 40 (-2.8 Gyr).\n'
    )
    chapter = mock_language_model.chronicle_answer(prompt)
    assert chapter is not None
    self.assertTrue(chapter.startswith('The Years of Change (ticks 4-5):'))
    self.assertIn('white-petal blight', chapter)
    self.assertNotIn('Old news', chapter)
    self.assertIsNone(mock_language_model.chronicle_answer('Hello'))

  def test_grazer_policies(self):
    model = mock_language_model.DeepTimeMockModel()
    behaviours = ('graze heavily', 'browse lightly', 'wander to new pastures')

    def choose(white: int, black: int) -> str:
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          f'Census: white daisies cover {white}% (local 20 C, growth 0.5);'
          f' black daisies cover {black}% (local 25 C, growth 0.5).'
      )
      return behaviours[doc.multiple_choice_question('Choose.', behaviours)]

    self.assertEqual(choose(40, 20), 'graze heavily')
    self.assertEqual(choose(20, 15), 'browse lightly')
    self.assertEqual(choose(10, 10), 'wander to new pastures')
    answer = mock_language_model.grazer_adaptation_answer(
        'Which heritable shift? The grazers have been eating...\n'
        'Genome: Genome of the grazers: optimum 20.0 C; prefers white petals'
        ' 50% of the time.\nCensus: white daisies cover 45% (local 20 C,'
        ' growth 0.5); black daisies cover 10% (local 25 C, growth 0.5).'
    )
    self.assertTrue((answer or '').startswith('prefer white petals'))

  def test_unknown_questions_ignore_the_order_options_are_shown_in(self):
    model = mock_language_model.DeepTimeMockModel()

    def answer(options):
      doc = interactive_document.InteractiveDocument(model)
      doc.statement('Some context.')
      return options[
          doc.multiple_choice_question('Should we change course?', options)
      ]

    self.assertEqual(answer(('Yes', 'No')), answer(('No', 'Yes')))

  def test_unknown_questions_use_stable_hash(self):
    model = mock_language_model.DeepTimeMockModel()
    first = model.sample_choice('Question: pick\n  (a) x\n  (b) y', ['a', 'b'])
    second = model.sample_choice('Question: pick\n  (a) x\n  (b) y', ['a', 'b'])
    self.assertEqual(first[0], second[0])
    self.assertEqual(model.stats(), {'text_calls': 0, 'choice_calls': 2})

  def test_species_policy(self):
    model = mock_language_model.DeepTimeMockModel()
    behaviours = ('spread aggressively', 'hold steady', 'go dormant')

    def choose(local: int, growth: float) -> str:
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          'Genome: Genome of the black daisies: optimum 22.5 C; petal albedo'
          f' 0.25.\nCensus: black daisies cover 12% (local {local} C, growth'
          f' {growth:.2f}); white daisies cover 50% (local 20 C, growth 0.90).'
      )
      return behaviours[doc.multiple_choice_question('Choose.', behaviours)]

    self.assertEqual(choose(24, 0.9), 'spread aggressively')
    self.assertEqual(choose(40, 0.0), 'go dormant')
    self.assertEqual(choose(28, 0.4), 'hold steady')

  def test_adaptation_answer_follows_pressure(self):
    def answer(local: int) -> str:
      return (
          mock_language_model.adaptation_answer(
              'Genome: Genome of the white daisies: optimum 22.5 C;'
              f' ...\nCensus: white daisies cover 40% (local {local} C, growth'
              ' 0.50).\nWhich single heritable shift would help them most?'
          )
          or ''
      )

    self.assertTrue(answer(30).startswith('warmer optimum'))
    self.assertTrue(answer(15).startswith('cooler optimum'))
    self.assertTrue(answer(23).startswith('no change'))
    self.assertIsNone(mock_language_model.adaptation_answer('Hello?'))

  def test_steward_tends_the_great_oxidation(self):
    options = (
        'wait and save omega',
        'seed oxygen-breathing mats [2 omega]',
        (
            'seed oxygen-making cyanobacteria (hastens the oxygen crisis) [3'
            ' omega]'
        ),
    )

    def choose(oxygen: float, iron: float, aerobes: int) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          _context(20.0, 9.0)
          + f" Air: oxygen {oxygen:.4f} of today's level, methane 0.80;"
          f' iron left to soak up oxygen {iron:.2f}.\nCensus: aerobes mats'
          f' cover {aerobes}% (local 18 C, growth 0.50).\nGoal: aerobes'
          ' should cover 5% of the planet.'
      )
      return options[doc.multiple_choice_question('Choose one.', options)]

    self.assertTrue(choose(0.2, 0.0, 1).startswith('seed oxygen-breathing'))
    self.assertTrue(choose(0.0, 0.01, 0).startswith('seed oxygen-making'))
    self.assertTrue(choose(0.0, 0.5, 0).startswith('wait'))
    self.assertTrue(choose(0.2, 0.0, 12).startswith('wait'))

  def test_steward_buys_instruments_when_calm(self):
    options = (
        'wait and save omega',
        'build better instruments (sharper, faster readings) [3 omega]',
    )

    def choose(noise: float, herds: bool) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      text = _context(22.0, 9.0) + (
          ' Instruments: standing stones and tide marks (temperatures'
          f' ±{noise:g} C, 2 ticks late, 20% of readings lost).'
      )
      if herds:
        text += ' Grazers 0.020 (prefer white petals 40%, optimum 22 C).'
      doc.statement(text)
      return options[doc.multiple_choice_question('Choose one.', options)]

    self.assertTrue(choose(2.5, False).startswith('build better'))
    self.assertTrue(choose(1.0, False).startswith('wait'))
    self.assertTrue(choose(1.0, True).startswith('build better'))
    self.assertTrue(choose(0.2, True).startswith('wait'))

  def test_polity_policy_feeds_first_then_supplies_the_bloomery(self):
    options = (
        'put more labor into: farm the land',
        'put more labor into: cut wood',
        'put more labor into: burn charcoal',
        'put more labor into: mine ore',
        'put more labor into: smelt iron in a bloomery',
        'put more labor into: forge tools',
        'put more labor into: study and teach',
        'keep the current balance of labor',
    )

    def choose(
        fed, grain, wood, charcoal, ore, iron, tools, variant, study=10, farm=60
    ):
      model = mock_language_model.DeepTimeMockModel(
          mock_language_model.VARIANTS[variant]
      )
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          f'Our storehouse and labor: 10.0 people, fed {fed}% last'
          f' generation. Storehouse: grain {grain}, wood {wood}, charcoal'
          f' {charcoal}, ore {ore}, iron {iron}, tools {tools}. Labor: farm'
          f' the land {farm}%, study and teach {study}%.'
      )
      return options[doc.multiple_choice_question('Choose one.', options)]

    self.assertIn('farm', choose(80, 9, 0, 0, 0, 0, 0, 'default'))
    self.assertIn('forge tools', choose(100, 60, 0, 0, 0, 2, 1, 'default'))
    self.assertIn('cut wood', choose(100, 60, 1, 1, 0, 0, 9, 'default'))
    self.assertIn('charcoal', choose(100, 60, 5, 1, 0, 0, 9, 'default'))
    self.assertIn('mine ore', choose(100, 60, 5, 3, 1, 0, 9, 'default'))
    self.assertIn('smelt iron', choose(100, 60, 5, 3, 3, 0, 9, 'default'))
    self.assertIn('keep', choose(80, 9, 0, 0, 0, 0, 0, 'passive'))
    # Still fed from the store, under two seasons' bread and the fields all
    # but abandoned: farm first. With hands in the fields, the chain goes on.
    self.assertIn('farm', choose(100, 15, 5, 3, 3, 0, 9, 'default', farm=4))
    # Any food trade counts; the people's own is grown first.
    model = mock_language_model.DeepTimeMockModel()
    doc = interactive_document.InteractiveDocument(model)
    doc.statement(
        'Our storehouse and labor: 10.0 people, fed 100% last generation.'
        ' Storehouse: grain 12, wood 5, charcoal 3, ore 3, iron 0, tools 9.'
        ' Labor: farm with fertiliser 4%, smelt iron in a bloomery 60%.'
    )
    food = options + ('put more labor into: farm with fertiliser',)
    chosen = food[doc.multiple_choice_question('Choose one.', food)]
    self.assertEqual(chosen, 'put more labor into: farm with fertiliser')
    # Unless the fertiliser is gone: then plain fields, which need nothing.
    doc = interactive_document.InteractiveDocument(model)
    doc.statement(
        'Our storehouse and labor: 10.0 people, fed 100% last generation.'
        ' Storehouse: grain 12, wood 5, charcoal 3, ore 3, iron 0, tools 9,'
        ' fertiliser 0.0. Labor: farm with fertiliser 4%, smelt iron in a'
        ' bloomery 60%.'
    )
    chosen = food[doc.multiple_choice_question('Choose one.', food)]
    self.assertEqual(chosen, 'put more labor into: farm the land')
    self.assertIn('smelt iron', choose(100, 15, 5, 3, 3, 0, 9, 'default'))
    self.assertIn('study', choose(100, 60, 5, 3, 3, 0, 9, 'default', study=2))

  def test_polity_policy_specialises_and_trades_for_what_it_lacks(self):
    options = (
        'put more labor into: mine ore',
        'put more labor into: cut wood',
        'ask to buy charcoal',
        'keep the current balance of labor',
    )

    def choose(land: str, charcoal: float, standing: str = 'none') -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          'Our storehouse and labor: 10.0 people, fed 100% last generation.'
          f' Storehouse: grain 12.0, wood 0.0, charcoal {charcoal}, ore 1.0,'
          ' iron 0.0, tools 9.0. Labor: study and teach 10%. Research 0.0; we'
          f' know: none yet. Our land is {land}. Silver 20.0. Standing'
          f' orders: {standing}.'
      )
      return options[doc.multiple_choice_question('Choose one.', options)]

    self.assertEqual(
        choose('rich in ore, poor in forest', 0.0), 'ask to buy charcoal'
    )
    self.assertEqual(
        choose('rich in ore, poor in forest', 0.0, 'buy charcoal (2 more)'),
        'put more labor into: mine ore',
    )
    self.assertEqual(
        choose('rich in ore', 0.0), 'put more labor into: mine ore'
    )

  def test_polity_policy_learns_the_cheapest_tech_when_fed(self):
    options = (
        'put more labor into: farm the land',
        'learn writing (10 research)',
        'learn irrigation (6 research)',
        'keep the current balance of labor',
    )
    model = mock_language_model.DeepTimeMockModel()
    doc = interactive_document.InteractiveDocument(model)
    doc.statement(
        'Our storehouse and labor: 10.0 people, fed 100% last generation.'
        ' Storehouse: grain 60.0, wood 0.0, charcoal 0.0, ore 0.0, iron 0.0,'
        ' tools 9.0. Labor: farm the land 60%.'
    )
    chosen = options[doc.multiple_choice_question('Choose one.', options)]
    self.assertEqual(chosen, 'learn irrigation (6 research)')

  def test_polity_policy_saves_for_a_better_tech_within_reach(self):
    sheet = (
        '10.0 people, fed 100% last generation. Storehouse: grain 60.0, wood'
        ' 5.0, charcoal 5.0, ore 5.0, iron 5.0, tools 5.0. Labor: farm the'
        ' land 60%. Research 26.0; we know: writing. Within reach: masonry'
        ' (22), steam power (30). '
    )
    options = {
        'a': 'put more labor into: farm the land',
        'b': 'learn masonry (22 research)',
        'c': 'keep the current balance of labor',
    }
    choice = mock_language_model.polity_policy(sheet, options)
    self.assertNotEqual(choice, options['b'])
    without = sheet.replace(', steam power (30)', '')
    self.assertEqual(
        mock_language_model.polity_policy(without, options), options['b']
    )

  def test_sample_text_mentions_temperature(self):
    model = mock_language_model.DeepTimeMockModel()
    self.assertIn('27.5 C', model.sample_text(_context(27.5, 1.0)))

  def test_steward_warms_a_cold_world_whose_people_go_hungry(self):
    def choose(temp: float) -> str:
      model = mock_language_model.DeepTimeMockModel()
      doc = interactive_document.InteractiveDocument(model)
      doc.statement(
          _context(temp, 9.0)
          + '\nPeoples: The peoples: Amber folk 10.0 people, fed 100%;'
          ' Cedar league 9.0 people, fed 62% FAMINE.'
      )
      return OPTIONS[doc.multiple_choice_question('Choose one.', OPTIONS)]

    self.assertIn('dark haze', choose(15.0))  # Near the cold edge: warm it.
    self.assertIn('bright clouds', choose(28.0))  # Near the hot edge: cool it.
    self.assertEqual(choose(22.0), 'wait and save omega')  # Not the weather.

  def test_steward_waits_when_the_seeds_it_wants_are_not_offered(self):
    options = tuple(o for o in OPTIONS if 'black daisies' not in o)
    model = mock_language_model.DeepTimeMockModel()
    doc = interactive_document.InteractiveDocument(model)
    doc.statement(_context(11.0, 3.0))  # Cold, haze unaffordable, no black.
    chosen = options[doc.multiple_choice_question('Choose one.', options)]
    self.assertEqual(chosen, 'wait and save omega')


if __name__ == '__main__':
  absltest.main()
