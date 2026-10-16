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

"""Tests for the Advisor's almanac: exact forecasts the Steward cannot see."""

import contextlib
import io
from unittest import mock

from absl.testing import absltest
from examples.deep_time import game as game_lib
from examples.deep_time import mock_language_model
from examples.deep_time import model_setup
from examples.deep_time import simulation


class AlmanacTest(absltest.TestCase):

  def test_eruption_forecasts_are_exact_and_draw_nothing(self):
    context = game_lib.new_game('snowball')
    lithosphere = context.planet.lithosphere
    assert lithosphere is not None
    state = lithosphere.get_state()
    ahead = lithosphere.forecast(8)
    self.assertEqual(lithosphere.get_state(), state)  # Nothing was drawn.
    actual = []
    for tick in range(1, 9):
      for eruption in lithosphere.step():
        actual.append((tick, eruption))
    self.assertEqual(ahead, actual)

  def test_forecasts_never_touch_the_live_stream(self):
    context = game_lib.new_game('snowball')
    lithosphere = context.planet.lithosphere
    assert lithosphere is not None
    state = lithosphere.get_state()
    for _ in range(3):
      lithosphere.forecast(10)
    self.assertEqual(lithosphere.get_state(), state)

  def test_almanac_reads_the_sun_ahead(self):
    context = game_lib.new_game('daisyworld')
    text = context.planet.almanac(5)
    self.assertStartsWith(text, 'Almanac (known only to the advisor)')
    self.assertRegex(text, r'in 5 ticks \(\+\d+\.\d%\)')

  def test_only_the_advisor_reads_the_almanac(self):
    texts, choices = [], []
    original_text = mock_language_model.DeepTimeMockModel.sample_text
    original_choice = mock_language_model.DeepTimeMockModel.sample_choice

    def sample_text(model, prompt, **kwargs):
      texts.append(prompt)
      return original_text(model, prompt, **kwargs)

    def sample_choice(model, prompt, responses, **kwargs):
      choices.append(prompt)
      return original_choice(model, prompt, responses, **kwargs)

    with mock.patch.object(
        mock_language_model.DeepTimeMockModel, 'sample_text', sample_text
    ), mock.patch.object(
        mock_language_model.DeepTimeMockModel, 'sample_choice', sample_choice
    ):
      models = model_setup.make_models('mock')
      sim, _ = simulation.build_simulation('daisyworld', models, 4)
      with contextlib.redirect_stdout(io.StringIO()):
        sim.play(max_steps=4)
    briefings = [t for t in texts if 'advisor of Gaia' in t]
    self.assertNotEmpty(briefings)
    self.assertTrue(all('Almanac (known only to' in t for t in briefings))
    steward = [c for c in choices if 'wait and save omega' in c]
    self.assertNotEmpty(steward)
    self.assertFalse(any('Almanac (known only to' in c for c in steward))

  def test_mock_advisor_cools_ahead_of_a_brightening_sun(self):
    prompt = (
        'You are the advisor of Gaia. Global mean temperature 24.0 C'
        ' (habitable band 10 to 35 C). Almanac (known only to the advisor):'
        ' the sun gives 1.000 now and will give 1.060 in 5 ticks (+6.0%).'
    )
    hot = mock_language_model.advisor_answer(prompt)
    assert hot is not None
    self.assertIn('seed bright clouds', hot)
    calm = mock_language_model.advisor_answer(
        prompt.replace('(+6.0%)', '(+1.0%)')
    )
    assert calm is not None
    self.assertIn('wait', calm)


if __name__ == '__main__':
  absltest.main()
