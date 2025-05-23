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

"""A set of pub names and reasons to like them."""

import random
from examples.deprecated.modular.environment import pub_coordination
from examples.deprecated.modular.environment.modules import pub_coordination_london

YEAR = pub_coordination_london.YEAR
MONTH = pub_coordination_london.MONTH
DAY = pub_coordination_london.DAY

NUM_PUBS = pub_coordination_london.NUM_PUBS


def sample_parameters(seed: int | None = None):
  """Samples a set of parameters for the world configuration."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  pubs = rng.sample(
      list(pub_coordination_london.PUB_PREFERENCES.keys()),
      pub_coordination_london.NUM_PUBS,
  )
  pub_preferences = {
      k: pub_coordination_london.PUB_PREFERENCES[k] for k in pubs
  }

  config = pub_coordination.WorldConfig(
      year=pub_coordination_london.YEAR,
      location="London",
      event="European football cup",
      game_countries=pub_coordination_london.EURO_CUP_COUNTRIES,
      venues=pubs,
      venue_preferences=pub_preferences,
      social_context=pub_coordination_london.SOCIAL_CONTEXT,
      random_seed=seed,
      num_main_players=2,
      num_supporting_players=0,
      num_games=3,
  )

  all_names = list(pub_coordination_london.MALE_NAMES) + list(
      pub_coordination_london.FEMALE_NAMES
  )

  rng.shuffle(all_names)
  config.people = all_names

  for _, name in enumerate(pub_coordination_london.MALE_NAMES):
    config.person_data[name] = {"gender": "male"}
  for _, name in enumerate(pub_coordination_london.FEMALE_NAMES):
    config.person_data[name] = {"gender": "female"}

  return config
