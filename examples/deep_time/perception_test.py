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

"""Tests for instruments (ageing beliefs and digests)."""

import json

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.components import world_step
from examples.deep_time.prefabs import steward
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing
from concordia.typing import entity as entity_lib


def _world_gm_step(context):
  """One step of the World GM's `WorldStep` (it steps the world, publishes)."""
  world_step.WorldStep(context, player_names=['Amber folk']).pre_act(
      entity_lib.ActionSpec(  # As the engine asks it each step.
          call_to_action='Which entities act next?',
          output_type=entity_lib.OutputType.NEXT_ACTING,
          options=('Amber folk',),
      )
  )


def _held(context, spec, initial=''):
  """An instrument the World GM runs, as its entity holds it."""
  perception.deliver(context, spec)
  return perception.subscribe(
      context,
      perception.reading(spec.owner, spec.label, initial, topic=spec.topics[0]),
  )


class PerceptionTest(absltest.TestCase):

  def test_instrument_shows_tick_and_age(self):
    context = game_lib.new_game('daisyworld')
    belief = _held(
        context,
        perception.instrument(
            context,
            'gaia.briefings',
            owner='Gaia',
            label='Advisor briefing',
        ),
        initial='nothing yet',
    )
    self.assertEqual(belief.get_current_value(), 'nothing yet')
    self.assertEqual(
        belief.get_pre_act_label(), 'Advisor briefing (gaia.briefings)'
    )
    context.bus.publish('gaia.briefings', 'Recommend: wait.')
    self.assertEqual(
        belief.get_current_value(), 'Recommend: wait. [tick 0, fresh]'
    )
    for _ in range(3):
      context.clock.advance()
      context.publish_state()
    self.assertEqual(
        belief.get_current_value(), 'Recommend: wait. [tick 0, 3 ticks old]'
    )
    self.assertIn(
        'Gaia', context.bus.topology()['gaia.briefings']['subscribers']
    )

  def test_an_entity_knows_its_own_state_only_as_the_gm_publishes_it(self):
    context = game_lib.new_game('daisyworld')
    state = {'stores': 'grain 3'}
    # The World GM publishes them (a role's rules); the entity reads them.
    perception.publish(
        context, 'Amber folk', 'storehouse', lambda: state['stores']
    )
    perception.publish(
        context, 'Amber folk', 'offer', lambda: state['stores'] + ' offered'
    )
    sheet = perception.read(
        context, 'Amber folk', 'storehouse', label='Storehouse'
    )
    brief = perception.reader(context, 'Amber folk', 'offer')
    # Nothing is known until the World GM publishes.
    self.assertEqual(sheet.get_current_value(), '')
    self.assertEqual(brief(), '')
    _world_gm_step(context)
    self.assertEqual(sheet.get_current_value(), 'grain 3')
    self.assertEqual(brief(), 'grain 3 offered')
    # The world changes; the entity learns it at the GM's next publication.
    state['stores'] = 'grain 1'
    self.assertEqual(sheet.get_current_value(), 'grain 3')
    _world_gm_step(context)
    self.assertEqual(sheet.get_current_value(), 'grain 1')
    self.assertEqual(
        perception.sheet_topic('Amber folk', 'storehouse'),
        'sheets.amber_folk.storehouse',
    )
    # Sheets are no one's observations.
    self.assertEqual(context.observation_queue.get_all(), {})

  def test_climate_instrument_renders_readings(self):
    context = game_lib.new_game('daisyworld')
    belief = _held(
        context,
        perception.instrument(
            context,
            'climate.global',
            owner='Gaia',
            label='Thermometer',
        ),
        initial='',
    )
    self.assertStartsWith(belief.get_current_value(), 'Global mean temperature')
    self.assertEndsWith(belief.get_current_value(), '[tick 0, fresh]')

  def test_transported_thermometer_is_late_noisy_and_upgradable(self):
    context = game_lib.new_game('daisyworld')
    planet = context.planet
    tier = planet.instrument()
    assert tier is not None
    self.assertEqual(tier['key'], 'stones')
    belief = _held(
        context,
        perception.instrument(
            context,
            'climate.global',
            owner='Gaia',
            label='Thermometer',
            transported=True,
        ),
        initial='none',
    )
    # The tick-0 reading is still on its way (stones are two ticks late):
    # the World GM carries it (its deliveries' state), not the Steward.
    self.assertEqual(belief.get_current_value(), 'none')
    in_flight = context.deliveries.get_state()['Gaia/Thermometer']
    self.assertIn('climate.global', json.dumps(in_flight))
    self.assertNotIn('operator_state', belief.get_state())

    def step():
      _world_gm_step(context)  # The World GM's own order of operations.

    for _ in range(12):
      step()
    reading = context.beliefs[('Gaia', 'climate.global')]
    assert reading.time is not None
    self.assertLessEqual(reading.time, planet.tick - 2)
    gaps = [
        r['epistemic_gap_c'] for r in planet.history if 'epistemic_gap_c' in r
    ]
    self.assertNotEmpty(gaps)
    self.assertGreater(max(gaps), 0.0)
    self.assertIn('ticks old', belief.get_current_value())
    planet.instrument_tier = len(planet.instrument_tiers) - 1
    step()
    step()
    reading = context.beliefs[('Gaia', 'climate.global')]
    self.assertEqual(reading.time, planet.tick)
    self.assertLess(planet.history[-1]['epistemic_gap_c'], 1.5)

  def test_exact_instruments_without_tiers(self):
    context = game_lib.new_game('daisyworld')
    context.planet.instrument_tier = None
    self.assertEqual(
        context.transport_for('climate.global'),
        perception.operators.Transport(),
    )

  def test_news_digest_merges_topics(self):
    context = game_lib.new_game('daisyworld')
    digest = perception.news_digest(
        context,
        ['geo.events', 'bio.events'],
        owner='Gaia',
        label='News',
        initial='quiet',
        size=2,
    )
    context.bus.publish('geo.events', 'ash')
    context.bus.publish('bio.events', 'blight')
    context.bus.publish('geo.events', 'flare')
    self.assertEqual(
        digest.get_current_value(), '- tick 0: blight\n- tick 0: flare'
    )

  def test_local_news_arrives_later_the_farther_away(self):
    context = game_lib.new_game('fire_and_iron')
    planet = context.planet
    assert planet.civ is not None
    home = planet.civ.polities['amber'].home
    news = _held(
        context,
        perception.local_news(context, owner='Amber folk', polity='amber'),
    )
    grid = planet.grid
    by_steps = {}
    for r in range(grid.size):
      by_steps.setdefault(planet.region_steps(home, r), r)
    near, far, beyond = by_steps[0], by_steps[3], by_steps.get(6)
    for region, text in ((far, 'far fire'), (near, 'near town')):
      context.bus.publish(
          f'local.r{region}', json.dumps({'text': text, 'region': region})
      )
    if beyond is not None:  # Out of reach: never heard.
      context.bus.publish(
          f'local.r{beyond}', json.dumps({'text': 'unheard', 'region': beyond})
      )
    self.assertIn('near town (here)', news.get_current_value())
    self.assertNotIn('far fire', news.get_current_value())
    for _ in range(3):
      context.wire.advance()
    value = news.get_current_value()
    self.assertIn('far fire (3 regions away)', value)
    self.assertLess(value.index('near town'), value.index('far fire'))
    self.assertNotIn('unheard', value)

  def test_the_thermometer_lists_the_readings_acted_on_before(self):
    planet = game_lib.new_game('daisyworld').planet
    for _ in range(5):
      planet.advance()
    for r, v in zip(planet.history, (20.0, 21.0, 22.5, 24.0, 25.5, 99.0)):
      r['believed_temp_c'] = v
    render = perception.with_earlier_readings(planet, lambda m: m.payload)
    text = render(topics.Message('climate.global', 'Now 26 C.', seq=0))
    # The last three ticks before this one (this tick's belief is the reading).
    self.assertEqual(
        text,
        'Now 26 C. Earlier readings you acted on: 22.5, 24.0, 25.5 C'
        ' (oldest first).',
    )

  def test_the_thermometer_reckons_its_late_reading_forward(self):
    planet = game_lib.new_game('daisyworld').planet
    (clouds,) = [i for i in planet.interventions() if i.key == 'bright_clouds']
    for _ in range(4):
      planet.advance()
    for r in planet.history:
      r.update(believed_temp_c=30.0, read_tick=r['tick'] - 1, read_sigma=1.0)
    planet.omega = 50
    planet.apply_intervention('Gaia', clouds)  # Acts on the next tick.
    planet.advance()
    render = perception.with_earlier_readings(planet, lambda m: m.payload)
    reading = json.dumps({'temp_c': 30.0})
    text = render(topics.Message('climate.global', reading, seq=0, time=4))
    self.assertIn('Your moves since that reading have cooled the planet', text)
    self.assertRegex(text, r'Your forecast: about 2\d ± \d C now')

  def test_a_planet_wide_tally_ripples_but_is_not_local_news(self):
    context = game_lib.new_game('fire_and_iron')
    planet = context.planet
    where = (10, 10)
    topic = f'local.r{int(planet.tiles.region[where])}'
    tally = planet_lib.WorldEvent(
        'bio.events', 'The taiga spreads.', where=where, local=False
    )
    context.publish_state([tally])
    self.assertIsNone(context.bus.latest(topic))
    raid = planet_lib.WorldEvent('civ.events', 'A raid.', where=where)
    context.publish_state([raid])
    latest = context.bus.latest(topic)
    assert latest is not None
    self.assertIn('A raid.', latest.payload)

  def test_the_steward_sees_the_peoples_where_there_are_any(self):
    planet = game_lib.new_game('sandbox').planet
    assert planet.civ is not None
    text = publishing.render_peoples(
        topics.Message('civ.census', json.dumps(planet.civ.payload()), seq=0)
    )
    self.assertTrue(text.startswith('The peoples: Amber folk'))
    self.assertIn('people, fed 100%', text)
    for name, need in (('sandbox', True), ('daisyworld', False)):
      # A static setting of the Steward's, from its params.
      peoples = steward.steward_config(game_lib.new_game(name)).params[
          'peoples'
      ]
      self.assertEqual(peoples, need)
      self.assertEqual(
          'civ.events' in steward._news_topics(peoples), need  # pylint: disable=protected-access
      )


if __name__ == '__main__':
  absltest.main()
