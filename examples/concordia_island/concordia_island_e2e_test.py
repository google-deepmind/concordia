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

"""End-to-end verification test for third-party Concordia Island export."""

import collections
import json
import os
import random
import re
import sys
import tempfile
from absl.testing import absltest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
  sys.path.append(_REPO_ROOT)
import concordia  # pylint: disable=g-import-not-at-top
import concordia.contrib  # pylint: disable=g-import-not-at-top
_OPEN_CONCORDIA = os.path.join(_REPO_ROOT, "concordia")
if os.path.isdir(_OPEN_CONCORDIA) and _OPEN_CONCORDIA not in concordia.__path__:
  concordia.__path__.append(_OPEN_CONCORDIA)
_OPEN_CONTRIB = os.path.join(_OPEN_CONCORDIA, "contrib")
if (
    os.path.isdir(_OPEN_CONTRIB)
    and _OPEN_CONTRIB not in concordia.contrib.__path__
):
  concordia.contrib.__path__.append(_OPEN_CONTRIB)

from examples.concordia_island import run  # pylint: disable=g-import-not-at-top
from examples.concordia_island.personas import generator as persona_generator  # pylint: disable=g-import-not-at-top
from examples.concordia_island.ui import atlas_loader  # pylint: disable=g-import-not-at-top
from examples.concordia_island.ui import map_dashboard  # pylint: disable=g-import-not-at-top
from examples.concordia_island.ui import map_data  # pylint: disable=g-import-not-at-top
from examples.concordia_island.ui import simulation_studio  # pylint: disable=g-import-not-at-top


_BRECKSVILLE_PERSONAS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "personas", "populations", "brecksville_ohio",
)


def _require_brecksville_personas(test_case: absltest.TestCase) -> None:
  """Skips a test if the bundled Brecksville population is not checked in."""
  # The 1,000 persona files ship separately from the engine; until they land,
  # tests that need the exact cohort are skipped rather than failed.
  if not os.path.isdir(_BRECKSVILLE_PERSONAS_DIR):
    test_case.skipTest(
        "Brecksville persona population not found at "
        f"{_BRECKSVILLE_PERSONAS_DIR}"
    )


class ConcordiaIslandEndToEndTest(absltest.TestCase):

  def test_bundled_ohio_suburb_personas_reproduce_exact_cohort(self):
    _require_brecksville_personas(self)
    personas = persona_generator.load_personas(
        date_label="brecksville_ohio"
    )
    self.assertLen(personas, 1000)
    names = sorted(personas.keys())
    random.Random(42).shuffle(names)
    first_20 = names[:20]
    self.assertEqual(
        first_20[:5],
        [
            "Olivia Welch",
            "Jon Huang",
            "Shirley Williams",
            "Tara Lee",
            "Allison Lee",
        ],
    )
    self.assertIn("Andrea Rosales", first_20)
    self.assertGreater(len(personas["Olivia Welch"].formative_memories), 5)

  def test_atlases_load_and_validate(self):
    atlases = atlas_loader.available_atlases()
    self.assertIn("brecksville", atlases)
    self.assertIn("concordia_island", atlases)
    self.assertIn("kerala", atlases)
    for aid in ("brecksville", "concordia_island", "kerala"):
      atlas = atlas_loader.load_atlas(aid)
      self.assertEqual(atlas.id, aid)
      self.assertNotEmpty(atlas.places)

  def test_unified_studio_and_map_api_on_default_paper_run(self):
    run_dir = map_data.default_run_dir()
    if not run_dir:
      self.skipTest("data/paper_runs/job_loss/gemini_2_5_flash_esa not found")
    map_dashboard.init_source()
    status_cfg, cfg = map_dashboard.handle_api("/api/config", {})
    self.assertEqual(status_cfg, 200)
    self.assertEqual(cfg["geography"], "concordia_island")

    status_atlas, atlas_json = map_dashboard.handle_api(
        "/api/atlas", {"g": ["concordia_island"]}
    )
    self.assertEqual(status_atlas, 200)
    self.assertEqual(atlas_json["meta"]["id"], "concordia_island")

    # The paper runs saved no location_history.json, so the trajectory is
    # rebuilt from the "// place [time]" tags in the per-agent memory files.
    with open(os.path.join(run_dir, "agent_names.json")) as f:
      agent_names = json.load(f)
    status_hist, hist = map_dashboard.handle_api("/api/location_history", {})
    self.assertEqual(status_hist, 200)
    self.assertIsNone(hist["error"])
    self.assertEqual(hist["ticks"][0], 1)
    self.assertGreater(hist["ticks"][-1], 96)  # Runs into day 13.
    last = hist["snapshots"][str(hist["ticks"][-1])]
    self.assertCountEqual([e["agent"] for e in last], agent_names)
    self.assertTrue(all(e["age"] >= 0 for e in last))

    data = simulation_studio.get_run_data()
    self.assertIsNone(data["error"])
    self.assertLen(agent_names, data["unique_agents"])
    self.assertIsNotNone(data["sim_clock"])
    self.assertIsNotNone(data["clock"])

    cog = simulation_studio.get_cognitive_data()
    self.assertIn("Olivia Welch", cog["agents"])
    self.assertGreater(cog["agents"]["Olivia Welch"]["memory_count"], 50)

  def test_run_esa_pipeline(self):
    _require_brecksville_personas(self)
    with tempfile.TemporaryDirectory() as tmp_dir:
      cfg = run.ExperimentConfig(
          agents=2,
          ticks=2,
          output_dir=tmp_dir,
          start_time="Thursday, January 1st, 7:00 AM",
          embedder_type="dummy",
          embedding_dimension=768,
          engine_type="simultaneous",
          personas_date="brecksville_ohio",
          use_mock=True,
          agent_prefab="esa",
          event_config="job_loss",
          layoff_fraction=0.5,
          first_dates=True,
          poll=False,
      )
      state = run.run_experiment(cfg)
      self.assertEqual(state["agents"], 2)
      for expected_file in (
          "simulation_state.json",
          "location_history.json",
          "entity_memories.json",
          "simulation_structured.json",
          "simulation_log.html",
          "laid_off_agents.json",
          "performance.json",
      ):
        self.assertTrue(
            os.path.isfile(os.path.join(tmp_dir, expected_file)),
            f"Missing {expected_file}",
        )

  def test_async_smoke_day_night_day_with_dates(self):
    """CLI smoke path: a dated day, one X + marketplace night, next morning.

    Guards two regressions: scheduled first dates deadlocking the async engine
    (no conversation driver) and nighttime GM turns advancing the daytime
    clock, which fast-forwarded runs far past --ticks.
    """
    _require_brecksville_personas(self)
    with tempfile.TemporaryDirectory() as tmp_dir:
      cfg = run.ExperimentConfig(
          agents=4,
          ticks=12,
          output_dir=tmp_dir,
          start_time="Thursday, January 1st, 7:00 AM",
          embedder_type="dummy",
          embedding_dimension=768,
          engine_type="async",
          personas_date="brecksville_ohio",
          use_mock=True,
          agent_prefab="esa",
          event_config="job_loss",
          layoff_fraction=0.5,
          first_dates=True,
          nighttime_social=True,
          x_rounds=1,
          enable_nighttime_marketplace=True,
          marketplace_rounds=1,
          poll=False,
      )
      with self.assertLogs(level="INFO") as captured:
        run.run_experiment(cfg)
      with open(os.path.join(tmp_dir, "simulation_state.json")) as f:
        sim_state = json.load(f)
      self.assertEqual(sim_state["final_tick"], 12)
      with open(os.path.join(tmp_dir, "simulation_structured.json")) as f:
        structured = json.load(f)
      by_entity = collections.Counter(
          e.get("entity_name") for e in structured["entries"]
      )
      self.assertGreater(by_entity["x_rules"], 0)
      self.assertGreater(by_entity["marketplace_rules"], 0)
      with open(os.path.join(tmp_dir, "entity_memories.json")) as f:
        memories = json.load(f)
      dialogue = []
      for mems in memories.values():
        for m in mems:
          if m.startswith("[observation] //") and ' -- "' in m:
            dialogue.append(m)
      self.assertNotEmpty(dialogue)

      # Night-switch race: on the morning after a night, no agent may take an
      # island action before it has itself been switched into the night GM.
      tick = 0
      switched_at_tick: dict[int, set[str]] = collections.defaultdict(set)
      early_island = []
      for message in (r.getMessage() for r in captured.records):
        m = re.search(r"FixedIntervalClock: tick (\d+) ->", message)
        if m:
          tick = int(m.group(1))
          continue
        m = re.search(r"switching (.+?) to \S+.* \(day \d+ -> (\d+)\)", message)
        if m:
          switched_at_tick[tick].add(m.group(1))
          continue
        m = re.search(
            r"TickGated.pre_act\(RESOLVE\): (.+?) entering resolve phase at "
            r"tick (\d+)",
            message,
        )
        if m and int(m.group(2)) == 8:
          if m.group(1) not in switched_at_tick[8]:
            early_island.append(m.group(1))
      self.assertLen(switched_at_tick[8], cfg.agents)
      self.assertEmpty(early_island)

  def test_consecutive_gms_island_x_and_marketplace(self):
    _require_brecksville_personas(self)
    from examples.concordia_island import island_simulation  # pylint: disable=g-import-not-at-top
    from examples.concordia_island import mock_language_model  # pylint: disable=g-import-not-at-top
    from examples.concordia_island.sim import agents as agents_lib  # pylint: disable=g-import-not-at-top

    agent_cfgs = [
        agents_lib.AgentConfig(
            name="Maria Santos",
            home_place="sunset_apartments",
            work_place="town_square",
            personality="Outgoing neighbor",
            backstory="Lives in Sunset Apartments.",
        ),
        agents_lib.AgentConfig(
            name="David Chen",
            home_place="sunset_apartments",
            work_place="town_square",
            personality="Thoughtful engineer",
            backstory="Works at Town Square.",
        ),
    ]
    model = mock_language_model.FastMockLanguageModel(verbose=False)
    embedder = run.DummyEmbedder(dimension=768)
    sim = island_simulation.IslandConcordiaSimulation(
        agent_configs=agent_cfgs,
        model=model,
        embedder=embedder,
        engine_type="async",
        max_ticks=9,
        tick_interval_minutes=120,
        experience_sampling=False,
        enable_nighttime_social=True,
        nighttime_social_mode="combined",
        x_rounds=1,
        enable_nighttime_marketplace=True,
        marketplace_rounds=1,
    )
    gm_names = [gm.name for gm in sim.game_masters]
    self.assertEqual(
        gm_names, ["island rules", "x_rules", "marketplace_rules"]
    )
    sim.play(max_ticks=9)

    from examples.concordia_island.sim import internet_forum  # pylint: disable=g-import-not-at-top

    # Verify X GM ('x_rules') processed nighttime social posts on X
    x_gm = sim.game_masters[1]
    forum_comp = x_gm.get_component(internet_forum.DEFAULT_FORUM_COMPONENT_KEY)
    forum_state = forum_comp.get_state()
    self.assertEqual(getattr(forum_comp, "_forum_name", "X"), "X")
    self.assertNotEmpty(forum_state["posts"])

    # Verify Marketplace GM ('marketplace_rules') processed nighttime orders
    mkt_gm = sim.game_masters[2]
    mkt_comp = mkt_gm.get_component("marketplace")
    mkt_state = mkt_comp.get_state()
    self.assertNotEmpty(mkt_state.get("purchase_history", {}))

    # Verify simulation returned to 'island rules' on Day 2 and reached tick 9
    island_gm_entity = sim.game_masters[0]
    clock = island_gm_entity.get_component("clock")
    self.assertGreaterEqual(clock.current_tick, 9)


if __name__ == "__main__":
  absltest.main()
