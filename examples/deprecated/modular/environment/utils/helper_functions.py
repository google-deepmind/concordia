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

"""Helper functions for loading modules to support environment configuration."""

from collections.abc import Sequence
import importlib
import random
import types
from typing import Any

from examples.deprecated.modular.environment import modules


def load_time_and_place_module(
    default_time_and_place_modules: Sequence[str],
    time_and_place_module: str | None = None,
    seed: int | None = None,
) -> tuple[types.ModuleType, Any]:
  """Load a module that adapts the setting to be a particular time and place.

  Args:
    default_time_and_place_modules: A list of modules to choose from.
    time_and_place_module: The name of the specific module to load. If None, a
      random module from default_time_and_place_modules will be chosen.
    seed: The random seed to use for sampling the parameters.

  Returns:
    time_and_place_params: a module containing the settings for the time and
      place.
    sampled_settings: a dictionary of sampled settings for the selected setting,
      produced by calling time_and_place_params.sample_parameters(). The idea is
      that, while `time_and_place_params` contains the general settings like
      the date and location, `sampled_settings` contains the specific values
      like the names of individual characters, which must be resampled for each
      run of the simulation.
  """
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)
  if time_and_place_module is None:
    time_and_place_module = rng.choice(default_time_and_place_modules)
  # Load the environment config with importlib
  time_and_place_params = importlib.import_module(
      f'{modules.__name__}.{time_and_place_module}'
  )
  sampled_settings = time_and_place_params.sample_parameters(seed=seed)
  return time_and_place_params, sampled_settings
