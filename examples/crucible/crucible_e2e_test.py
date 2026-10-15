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

"""End-to-end MOCK games of every toy, through Concordia's Simulation."""

import json
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from examples.crucible import harness
from examples.crucible import model_setup
from examples.crucible import audits
from examples.crucible import toys


class EndToEndTest(parameterized.TestCase):

  @parameterized.product(toy=list(toys.registry()), seat=harness.BOTS)
  def test_game_finishes(self, toy, seat):
    game = harness.build_game(toy, model_setup.mock_model(), seat)
    game.play()
    self.assertTrue(game.toy.done())
    self.assertIn('winner', game.outcome())

  @parameterized.parameters(*[(t,) for t in toys.registry()])
  def test_same_seed_same_game(self, toy):
    outcomes = []
    for _ in range(2):
      game = harness.build_game(toy, model_setup.mock_model(), 'random', 3)
      game.play()
      outcomes.append(json.dumps(game.toy.get_state(), sort_keys=True))
    self.assertEqual(outcomes[0], outcomes[1])

  def test_model_in_the_seat(self):
    game = harness.build_game('claims', model_setup.mock_model(), 'model')
    game.play()
    self.assertTrue(game.toy.done())

  def test_run_folder(self):
    game = harness.build_game('claims', model_setup.mock_model(), 'skilled')
    log = game.play()
    with tempfile.TemporaryDirectory() as out_dir:
      summary = harness.write_run(game, log, out_dir)
    self.assertNotEmpty(summary['receipts'])

  def test_skill_shows_in_claims(self):
    result = audits.audit('claims', 'sandbox', seeds=2)
    self.assertGreaterEqual(result['skilled_minus_random'], 30)
    self.assertGreaterEqual(result['skilled_minus_passive'], 50)


if __name__ == '__main__':
  absltest.main()
