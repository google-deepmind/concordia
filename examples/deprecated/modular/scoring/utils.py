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

"""Utility functions for scoring submissions to the Concordia challenge."""

import numpy as np


def get_win_loss_matrix(score_matrix):
  """Convert scores [scenarios X agents] into win-loss data [agents X agents].

  Agent i is considered to have won against agent j on scenario s when
  score_matrix[s, i] > score_matrix[s, j].

  Args:
    score_matrix: [scenarios X agents] matrix of raw scores.

  Returns:
    win_loss_matrix: [agents X agents] matrix of win-loss data. The value
    win_loss_matrix[i, j] is the number of scenarios where agent i won against
    agent j.
  """
  num_scenarios, num_evaluated_agents = score_matrix.shape
  win_loss = np.zeros((num_evaluated_agents, num_evaluated_agents))
  for scenario_idx in range(num_scenarios):
    for i in range(num_evaluated_agents):
      for j in range(num_evaluated_agents):
        if score_matrix[scenario_idx, i] > score_matrix[scenario_idx, j]:
          win_loss[i, j] += 1

  return win_loss
