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

"""Tests for the lithosphere."""

from absl.testing import absltest
from examples.deep_time.world import data
from examples.deep_time.world import geology
from examples.deep_time.world import planet as planet_lib


def _lithosphere(seed: int = 0) -> geology.Lithosphere:
  return geology.Lithosphere(data.load('geology.yaml'), seed=seed)


class LithosphereTest(absltest.TestCase):

  def test_same_seed_same_eruptions(self):
    first, second = _lithosphere(3), _lithosphere(3)
    self.assertEqual(
        [first.step() for _ in range(50)], [second.step() for _ in range(50)]
    )

  def test_rate_and_bounds(self):
    lithosphere = _lithosphere(1)
    eruptions = []
    for _ in range(400):
      eruptions.extend(lithosphere.step())
    self.assertBetween(len(eruptions) / 400, 0.08, 0.25)
    self.assertLen(eruptions, lithosphere.count)
    for eruption in eruptions:
      self.assertBetween(eruption.magnitude, 1, 3)
      self.assertAlmostEqual(
          eruption.ash_albedo,
          eruption.magnitude * lithosphere.ash_per_magnitude,
      )
      self.assertIn(eruption.volcano, eruption.describe())

  def test_state_round_trip_continues_the_stream(self):
    lithosphere = _lithosphere(5)
    for _ in range(10):
      lithosphere.step()
    resumed = _lithosphere(99)
    resumed.set_state(lithosphere.get_state())
    self.assertEqual(
        [lithosphere.step() for _ in range(20)],
        [resumed.step() for _ in range(20)],
    )
    self.assertEqual(lithosphere.count, resumed.count)


class PlanetGeologyTest(absltest.TestCase):

  def _planet(self, seed: int) -> planet_lib.Planet:
    return planet_lib.Planet(
        data.scenario('daisyworld'),
        data.load('daisyworld.yaml'),
        planet_lib.interventions_from_config(data.load('interventions.yaml')),
        lithosphere=_lithosphere(seed),
    )

  def test_eruptions_cool_and_fertilise(self):
    planet = self._planet(0)
    reports = [planet.advance() for _ in range(40)]
    texts = []
    for r in reports:
      texts.extend(t for t in r.texts() if 'eruption' in t)
    self.assertNotEmpty(texts)
    self.assertLen(texts, planet.history[-1]['eruptions'])
    peak = max(row['fertility'] for row in planet.history)
    self.assertGreater(peak, 1.0)

  def test_seeds_change_history(self):
    histories = set()
    for seed in range(4):
      planet = self._planet(seed)
      for _ in range(30):
        planet.advance()
      histories.add(planet.history[-1]['eruptions'])
    self.assertGreater(len(histories), 1)


if __name__ == '__main__':
  absltest.main()
