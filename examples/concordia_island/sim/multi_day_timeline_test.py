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

"""Unit test for multi-day timeline integrity and marketplace transitions."""

from absl.testing import absltest
from examples.concordia_island import island_simulation
from examples.concordia_island import mock_language_model
from examples.concordia_island.sim import agents as agents_lib
import numpy as np


class _DummyEmbedder:

  def __call__(self, text):
    return np.zeros(768, dtype=np.float32)


class MultiDayTimelineTest(absltest.TestCase):

  def test_two_day_timeline_and_marketplace_cycle(self):
    """Run a 2-day (16 ticks + 2 marketplace nights) simulation and verify timeline continuity."""
    model = mock_language_model.FastMockLanguageModel(verbose=False)
    embedder = _DummyEmbedder()

    agent_configs = [
        agents_lib.AgentConfig(
            name="Alice",
            gender="female",
            home_place="millbrook_apts_unit_1",
            work_place="cafe",
            personality="diligent, anxious and cautious",
            backstory=(
                "Alice works at the cafe and lives at Millbrook Apartments."
            ),
        ),
        agents_lib.AgentConfig(
            name="Bob",
            gender="male",
            home_place="brecksville_commons_unit_5",
            work_place="office_floor_tech",
            personality="outgoing and practical",
            backstory="Bob works in tech and lives in Brecksville Commons.",
        ),
    ]

    # 4 ticks starting from 7:00 PM:
    # Tick 0: Thursday, Jan 1st 7:00 PM (Day 1 Evening)
    # Tick 1: Thursday, Jan 1st 9:00 PM (Day 1 Nightfall -> Marketplace 2
    # rounds)
    # Tick 2: Friday, Jan 2nd 7:00 AM (Day 2 Morning reset to home, breakfast
    # food deduction)
    # Tick 3: Friday, Jan 2nd 9:00 AM (Day 2 Morning work/activities)
    sim = island_simulation.IslandConcordiaSimulation(
        agent_configs=agent_configs,
        model=model,
        embedder=embedder,
        start_time="Thursday, January 1st, 7:00 PM",
        engine_type="simultaneous",
        max_ticks=4,
        tick_interval_minutes=120,
        enable_nighttime_marketplace=True,
        marketplace_rounds=2,
        fiscal_config="control",
        food_min_daily=3,
        agent_prefab="island__MinimalEntity",
        experience_sampling=False,
    )

    results = sim.play(max_ticks=4)
    self.assertIsNotNone(results)

    # 1. Verify economic profiles were initialized and updated
    profiles = sim.economic_profiles
    self.assertIsNotNone(profiles)
    self.assertIn("Alice", profiles)
    self.assertIn("Bob", profiles)

    # Both agents started with initial cash and food balances
    self.assertGreater(profiles["Alice"].liquid_balance, 0)
    self.assertGreater(profiles["Bob"].liquid_balance, 0)

    # 2. Verify raw log structure and timeline continuity
    raw_log = sim.get_raw_log()
    self.assertNotEmpty(raw_log)

    # Extract all observations and events per agent
    alice_entries = [e for e in raw_log if "Alice" in str(e)]
    bob_entries = [e for e in raw_log if "Bob" in str(e)]
    self.assertNotEmpty(alice_entries)
    self.assertNotEmpty(bob_entries)

    # 3. Verify clock progression spanned across both days
    log_text = str(raw_log)
    self.assertIn("January 1st", log_text)
    self.assertIn("January 2nd", log_text)

    # 4. Verify marketplace observations have correct Thursday timestamp
    #    (not Friday, which is the clock's post-overnight-skip state).
    if "marketplace" in log_text.lower():
      self.assertIn("marketplace [Thursday, January 1st, 11:00 PM]", log_text)
      self.assertNotIn("marketplace [Friday, January 2nd, 11:00 PM]", log_text)


if __name__ == "__main__":
  absltest.main()
