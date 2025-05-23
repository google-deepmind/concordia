# Copyright 2023 DeepMind Technologies Limited.
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

from absl.testing import absltest
from absl.testing import parameterized
from examples.deprecated.modular.scoring import elo
import numpy as np

# pylint: disable=g-one-element-tuple


class EloTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='test 0',
          win_loss=np.array(
              [[0, 2, 0, 1], [3, 0, 5, 0], [0, 3, 0, 1], [4, 0, 3, 0]]
          ),
          expected_elos=np.array([1422, 1507, 1428, 1642]),
      ),
      dict(
          testcase_name='test 1',
          win_loss=np.array([
              [0, 8, 8, 10, 9],
              [2, 0, 6, 6, 6],
              [2, 4, 0, 6, 4],
              [0, 4, 4, 0, 5],
              [1, 4, 6, 5, 0],
          ]),
          expected_elos=np.array([1773, 1491, 1428, 1380, 1428]),
      ),
  )
  def test_calculation_is_correct(
      self,
      win_loss: np.ndarray[tuple[int, int], np.dtype[np.number]],
      expected_elos: np.ndarray[tuple[int], np.dtype[np.number]]
  ):
    elo_ratings = elo.get_elo_ratings(win_loss)
    np.testing.assert_equal(elo_ratings, expected_elos)


if __name__ == '__main__':
  absltest.main()
