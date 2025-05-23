# Copyright 2024 DeepMind Technologies Limited.
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

"""Settings for an early 2000s american reality show for the prisoners_dilemma.
"""

from examples.deprecated.modular.environment.modules import circa_2003_american_reality_show as parent_module


def sample_parameters(seed: int | None = None):
  """Sample parameters of the setting and the backstory for each player."""
  return parent_module.sample_parameters(
      minigame_name='stag_hunt',
      num_players=4,
      seed=seed,
  )
