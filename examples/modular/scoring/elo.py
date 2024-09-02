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

"""Calculate Elo scores from a win_loss matrix.

This algorithm is from:
Newman, M.E.J., 2023. Efficient computation of rankings from pairwise
comparisons. Journal of Machine Learning Research, 24(238), pp.1-25.
"""

import numpy as np

# pylint: disable=g-one-element-tuple

_DEFAULT_NUM_ITERATIONS = 50


def _update_model_params(
    win_loss: np.ndarray[tuple[int, int], np.dtype[np.number]],
    params: np.ndarray[tuple[int], np.dtype[np.number]]):
  """Update Bradley-Terry model parameters using the Newman 2023 algorithm.

  This function modifies the params array in place.

  Args:
    win_loss: [agents X agents] matrix of win-loss data. The value
      win_loss[i, j] is the number of scenarios where agent i won against agent
      j.
    params: [agents] array of Bradley-Terry model parameters.
  """
  num_agents = params.shape[0]
  for i in range(num_agents):
    numerator_vals_to_sum = []
    denominator_vals_to_sum = []
    for j in range(num_agents):
      if i != j:
        total_rating = params[i] + params[j]
        numerator_vals_to_sum.append(
            win_loss[i, j] * params[j] / total_rating
        )
        denominator_vals_to_sum.append(win_loss[j, i] / total_rating)

    params[i] = sum(numerator_vals_to_sum) / sum(denominator_vals_to_sum)

  geometric_mean = params.prod() ** (1 / num_agents)
  params /= geometric_mean


def _params_to_elo_ratings(
    params: np.ndarray[tuple[int], np.dtype[np.number]]
) -> np.ndarray[tuple[int], np.dtype[np.number]]:
  """Convert model parameters to Elo ratings centered at 1500."""
  elos = 400 * np.log10(params)
  mean_elo = np.mean(elos)
  return np.round(elos - mean_elo + 1500)


def get_elo_ratings(
    win_loss: np.ndarray[tuple[int, int], np.dtype[np.number]],
    num_iterations: int = _DEFAULT_NUM_ITERATIONS,
) -> np.ndarray[tuple[int], np.dtype[np.number]]:
  """Get Elo scores from win_loss matrix of shape [num_scenarios, num_agents].

  Args:
    win_loss: [agents X agents] matrix of win-loss data. The value
      win_loss[i, j] is the number of scenarios where agent i won against agent
      j.
    num_iterations: number of iterations to run the algorithm.

  Returns:
    elo_ratings: [agents] 1-D array of Elo ratings.
  """
  # First initialize parameters since this is an iterative algorithm.
  num_rows, num_columns = win_loss.shape
  if num_rows != num_columns:
    raise ValueError('win_loss matrix must be square.')
  if num_rows < 4:
    raise ValueError('win_loss matrix must contain at least 4 agents.')

  params = np.ones(num_rows)
  # Iteratively update the model parameters.
  for _ in range(num_iterations):
    _update_model_params(win_loss, params)
  # Convert the resulting parameters to Elo ratings.
  elo_ratings = _params_to_elo_ratings(params)
  return elo_ratings
