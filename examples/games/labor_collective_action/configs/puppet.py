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

"""Minimal deterministic configuration for puppet agent tests.

Imports economic constants and scene intro templates from the anthracite
coal scenario rather than re-declaring them, so the test stays in sync
with the real config structure.  Workers: Alice+Charlie strike, Bob works.
Boss holds firm.
"""

from examples.games.labor_collective_action.configs import anthracite_coal_labor as _scenario

# ── Re-export constants that environment.py reads via getattr ────────────
YEAR = _scenario.YEAR
MONTH = _scenario.MONTH
DAY = _scenario.DAY

LOW_DAILY_PAY = _scenario.LOW_DAILY_PAY
ORIGINAL_DAILY_PAY = _scenario.ORIGINAL_DAILY_PAY
DAILY_EXPENSES = _scenario.DAILY_EXPENSES
PRESSURE_THRESHOLD = _scenario.PRESSURE_THRESHOLD
WAGE_INCREASE_FACTOR = _scenario.WAGE_INCREASE_FACTOR

WORKER_EVENING_INTRO = _scenario.WORKER_EVENING_INTRO
WORKER_MORNING_INTRO = _scenario.WORKER_MORNING_INTRO
BOSS_MORNING_INTRO = _scenario.BOSS_MORNING_INTRO
BOSS_CALL_TO_ACTION = _scenario.BOSS_CALL_TO_ACTION
BOSS_OPTIONS = _scenario.BOSS_OPTIONS

NUM_MAIN_PLAYERS = 3

# ── Puppet-specific overrides ────────────────────────────────────────────
OVERHEARD_STRIKE_TALK = [
    "{player_name} hears rumors of a strike.",
]


def sample_parameters():
  """Return a minimal WorldConfig for deterministic puppet testing."""
  config = _scenario.WorldConfig(
      year=YEAR,
      location="TestTown",
      background_poor_work_conditions=(),
      seed=42,
      people=("Alice", "Bob", "Charlie"),
      antagonist="Mr. Boss",
      organizer="TestOrganizer",
      overheard_strike_talk=OVERHEARD_STRIKE_TALK,
      formative_memory_prompts={
          "Alice": [],
          "Bob": [],
          "Charlie": [],
      },
      person_data={
          "Alice": {"gender": "female", "salient_beliefs": []},
          "Bob": {"gender": "male", "salient_beliefs": []},
          "Charlie": {"gender": "female", "salient_beliefs": []},
          "Mr. Boss": {"gender": "male", "salient_beliefs": []},
          "TestOrganizer": {"gender": "female", "salient_beliefs": []},
      },
      num_additional_days=0,
      num_additional_dinners=0,
  )
  return config
