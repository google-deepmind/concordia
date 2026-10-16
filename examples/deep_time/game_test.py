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

import contextlib
import io
import json
from unittest import mock

from absl.testing import absltest
from concordia.contrib.pubsub import topics
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time import model_setup
from examples.deep_time import simulation
from examples.deep_time.world import publishing


def _haze(context: game_lib.GameContext):
  (haze,) = [i for i in context.planet.interventions() if i.key == 'dark_haze']
  return haze


class PushesTest(absltest.TestCase):

  def test_a_scenario_measures_pushes_on_its_own_world(self):
    snowball = game_lib.new_game('snowball')
    daisyworld = game_lib.new_game('daisyworld')
    self.assertTrue(_haze(snowball).push.startswith('may tip the ice'))
    self.assertNotIn('tip the ice', _haze(daisyworld).push)
    self.assertIn('°C', _haze(daisyworld).push)
    option = _haze(snowball).option
    self.assertEqual(
        snowball.pushes([option, 'wait and save omega']),
        {option: _haze(snowball).push},
    )

  def test_mission_control_moves_land_a_light_delay_later(self):
    _, context = simulation.build_simulation(
        'light_delay', model_setup.make_models('mock'), max_ticks=1
    )
    options = list(context.actors['Mission Control'].action_spec().options)
    pushes = context.pushes(options)
    food = 'tell the colonists: put more labor into grow food under glass'
    self.assertTrue(
        pushes[food].startswith('in 4 years, if they follow it: food +')
    )
    (design,) = [o for o in options if o.startswith('transmit the design')]
    self.assertTrue(pushes[design].startswith('in 4 years: '))
    self.assertNotIn('listen and wait', pushes)
    self.assertLen(pushes, len(options) - 1)  # Every move but listening.
    colonists = list(
        context.actors['Colonists of New Ararat'].action_spec().options
    )
    own = context.pushes(colonists)
    self.assertTrue(
        own['put more labor into: grow food under glass'].startswith('food +')
    )


class AirGaugeTest(absltest.TestCase):

  def test_shows_the_goals_cap_on_the_air(self):
    payload = game_lib.new_game('great_acceleration').planet.cycles_payload()
    self.assertEqual(payload['cap_co2'], 1.3)
    text = publishing.render_carbon(
        topics.Message(
            topic='cycles.carbon', payload=json.dumps(payload), seq=1
        )
    )
    self.assertIn('The goal caps the air at x1.30 of the start.', text)
    daisy = game_lib.new_game('daisyworld').planet.cycles_payload()
    self.assertNotIn('cap_co2', daisy)


class FrozenWorldAdvisorTest(absltest.TestCase):

  def test_the_advisor_of_a_frozen_world_briefs_on_the_carbon(self):
    briefings = []
    real = mock_language_model.advisor_answer

    def spy(prompt):
      answer = real(prompt)
      if answer:
        briefings.append(answer)
      return answer

    sim, context = simulation.build_simulation(
        'snowball', model_setup.make_models('mock'), max_ticks=8
    )
    with mock.patch.object(mock_language_model, 'advisor_answer', spy):
      with contextlib.redirect_stdout(io.StringIO()):
        sim.play(premise='', max_steps=8)
    self.assertNotEmpty(briefings)
    self.assertFalse([b for b in briefings if 'No urgent danger' in b])
    self.assertTrue(any('rouse the deep volcanoes' in b for b in briefings))
    del context


if __name__ == '__main__':
  absltest.main()
