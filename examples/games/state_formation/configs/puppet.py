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

"""Puppet test configuration for deterministic state formation testing.

Returns a ConfigDict from sample_parameters() matching the structure of
pre_state_villages.py, but with minimal deterministic data and fixed puppet
responses. Tests structure matching, not specific numerical outcomes.
"""

import datetime

from ml_collections import config_dict


def sample_parameters(unused_seed: int | None = None):
  """Return a minimal ConfigDict for deterministic puppet testing."""
  config = config_dict.ConfigDict()

  # Village names
  config.village_a_name = "Agartha"
  config.village_b_name = "Bretonnia"

  # Main characters (elders) — environment prepends "Elder " prefix
  config.main_characters = config_dict.ConfigDict()
  config.main_characters.a = config_dict.ConfigDict()
  config.main_characters.a.name = "Aldric"
  config.main_characters.a.gender = "male"
  config.main_characters.b = config_dict.ConfigDict()
  config.main_characters.b.name = "Brynn"
  config.main_characters.b.gender = "female"

  # Supporting characters (villagers) — tuples of (name, gender)
  config.supporting_characters = config_dict.ConfigDict()
  config.supporting_characters.a = (
      ("Greta", "female"),
      ("Ivan", "male"),
  )
  config.supporting_characters.b = (
      ("Klaus", "male"),
      ("Marta", "female"),
  )

  # Activities
  config.activities = (
      "free time",
      "farming",
      "training as a warrior",
  )

  # Simulation parameters
  config.num_years = 1

  # Scoring thresholds
  config.defense_threshold = 0.25
  config.starvation_threshold = 0.1

  # Time settings (used for scenario premise)
  config.times = config_dict.ConfigDict()
  config.times.setup = datetime.datetime(hour=20, year=1750, month=1, day=1)

  # Basic setting
  config.basic_setting = (
      "Agartha and Bretonnia are pre-state societies. They are small "
      "villages on the coast."
  )

  # Barbarian raid info (shared memories)
  config.barbarian_raid_info = [
      "Barbarian raiders are a constant threat.",
      "The barbarian raids have become more frequent of late.",
  ]

  # Village cultural elements (shared memories)
  config.villages = config_dict.ConfigDict()
  config.villages.a = [
      "Agartha is known for its Festival of Tides.",
  ]
  config.villages.b = [
      "Bretonnia is known for its harvest dances.",
  ]

  # Home scene premise — uses {name} and {village} placeholders
  config.home_scene_premise = (
      "Elder {name} is home in {village}. Tomorrow they depart "
      "for the hill of accord to negotiate an alliance."
  )

  # Negotiation phase premise — uses {player_name} and {village_name}
  config.negotiation_phase_premise = (
      "Elder {player_name} left {village_name} early in the morning and "
      "arrived just now at the hill of accord. The reason for this "
      "meeting of the two elder representatives "
      "(Aldric representing Agartha and "
      "Brynn representing Bretonnia) "
      "is to negotiate an alliance against the barbarian raiders."
  )

  # Villager how_things_are constant — uses {name} and {village_name}
  config.villager_how_things_are_constant = config_dict.ConfigDict()
  config.villager_how_things_are_constant.village_a = (
      "{name} has always been a farmer near {village_name}."
  )
  config.villager_how_things_are_constant.village_b = (
      "{name}'s family values strength and trains for war near {village_name}."
  )

  # Event description samplers (lambdas returning strings)
  config.sample_event_of_failing_to_repel_barbarians = (
      lambda: "The barbarian raid could not be repelled."
  )
  config.sample_event_of_success_repelling_barbarians = (
      lambda: "The barbarian raiders were successfully repelled."
  )
  config.sample_event_of_failing_to_grow_food = (
      lambda: "The harvest failed this year."
  )
  config.sample_event_of_success_growing_food = (
      lambda: "The harvest was successful this year."
  )
  config.sample_event_no_treaty_in_effect = (
      lambda: "There is no treaty in effect."
  )
  config.sample_event_treaty_in_effect = (
      lambda: "The treaty to pool agricultural resources is in effect."
  )

  # Fixed responses for puppet agents (deterministic testing).
  # Use shorthand keys ('post_negotiation', 'activity') to avoid dots in
  # ConfigDict keys. environment.py maps these to the actual CTA strings.
  config.player_fixed_responses = {
      # Elders respond to treaty questions
      "Elder Aldric": {
          "post_negotiation": "yes",
      },
      "Elder Brynn": {
          "post_negotiation": "yes",
      },
      # Villagers respond with activity allocations
      "Greta": {
          "activity": '{"farming": 0.6, "warrior": 0.3, "free_time": 0.1}',
      },
      "Ivan": {
          "activity": '{"farming": 0.3, "warrior": 0.5, "free_time": 0.2}',
      },
      "Klaus": {
          "activity": '{"farming": 0.5, "warrior": 0.4, "free_time": 0.1}',
      },
      "Marta": {
          "activity": '{"farming": 0.4, "warrior": 0.4, "free_time": 0.2}',
      },
  }

  return config
