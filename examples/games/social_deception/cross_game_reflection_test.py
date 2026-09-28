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

"""Tests for cross_game_reflection.py 5-Pillar Elo System."""

from absl.testing import absltest
from examples.games.social_deception import cross_game_reflection


class CrossGameReflectionTest(absltest.TestCase):

  def test_persona_key_equality_and_serialization(self):
    k1 = cross_game_reflection.PersonaKey(
        "models/gemini-3-flash-preview", "neutral"
    )
    k2 = cross_game_reflection.PersonaKey.from_str(
        "models/gemini-3-flash-preview::neutral"
    )
    self.assertEqual(k1, k2)
    self.assertEqual(k1.to_str(), "models/gemini-3-flash-preview::neutral")

    d = k1.to_dict()
    self.assertEqual(cross_game_reflection.PersonaKey.from_dict(d), k1)

  def test_5_pillar_elo_rating_update_and_role_counting(self):
    elo = cross_game_reflection.TriEloRating()
    # 2 Game-Level Elos
    self.assertEqual(elo.deception_game_elo, 1200.0)
    self.assertEqual(elo.detection_game_elo, 1200.0)
    # 3 Belief-Level Elos
    self.assertEqual(elo.deception_belief_elo, 1200.0)
    self.assertEqual(elo.detection_belief_elo, 1200.0)
    self.assertEqual(elo.epistemic_belief_elo, 1200.0)
    self.assertEqual(elo.games_played, 0)
    self.assertEqual(elo.total_seat_runs, 0)

    # Simulate 5-pillar game update
    elo.update(
        delta_dec_game=+16.0,
        delta_det_game=+8.0,
        delta_epi_belief=-4.0,
        delta_dec_belief=+12.0,
        delta_det_belief=+10.0,
        role_contributions=[
            "Demon",
            "Poisoner",
            "Witness",
            "Soldier",
            "Empath",
        ],
    )

    self.assertEqual(elo.deception_game_elo, 1216.0)
    self.assertEqual(elo.detection_game_elo, 1208.0)
    self.assertEqual(elo.epistemic_belief_elo, 1196.0)
    self.assertEqual(elo.deception_belief_elo, 1212.0)
    self.assertEqual(elo.detection_belief_elo, 1210.0)
    # Check backward compatibility aliases
    self.assertEqual(elo.deception_elo, 1216.0)
    self.assertEqual(elo.detection_elo, 1208.0)
    self.assertEqual(elo.epistemic_elo, 1196.0)

    self.assertEqual(elo.games_played, 1)
    self.assertEqual(elo.total_seat_runs, 5)
    self.assertEqual(elo.role_counts["Demon"], 1)
    self.assertEqual(elo.role_counts["Witness"], 1)

  def test_belief_system_format_for_notepad(self):
    bs = cross_game_reflection.BeliefSystem(
        strategic_learnings=["Never spend dead vote before Day 3."],
        common_pitfalls=[
            "Voting YES on Corruptor bluff without checking Outsider count."
        ],
        role_playbooks={
            "Demon": ["Coordinate fake Townsfolk claim with Minion."]
        },
    )
    formatted = bs.format_for_notepad(role="Demon")
    self.assertIn("Never spend dead vote before Day 3.", formatted)
    self.assertIn("Voting YES on Corruptor bluff", formatted)
    self.assertIn("Coordinate fake Townsfolk claim", formatted)

  def test_agent_reflection_5_scores_attributes(self):
    pkey = cross_game_reflection.PersonaKey("test_model", "neutral")
    ref = cross_game_reflection.AgentReflection(
        agent_name="Player_0",
        role="Empath",
        perceived_role="Empath",
        alignment="good",
        persona_key=pkey,
        deception_score=0.5,
        detection_score=0.8,
        deception_belief_score=0.5,
        detection_belief_score=0.85,
        epistemic_score=0.9,
        key_learnings=["Watch out for Poisoner."],
        rules_for_future=["Check sober neighbor."],
        won_game=True,
        was_drunk=False,
        was_poisoned=True,
        is_evil=False,
        is_outsider=False,
        is_townsfolk=True,
        detected_poisoning_or_drunkenness=True,
        epistemic_calibration_notes="Correctly diagnosed poisoned ping.",
    )
    self.assertTrue(ref.won_game)
    self.assertTrue(ref.was_poisoned)
    self.assertTrue(ref.detected_poisoning_or_drunkenness)
    self.assertEqual(ref.detection_belief_score, 0.85)
    self.assertEqual(ref.epistemic_score, 0.9)

  def test_experience_store_5_pillar_serialization(self):
    store = cross_game_reflection.ExperienceStore()
    k = cross_game_reflection.PersonaKey("model_test", "neutral")
    state = store.get_persona_state(k)
    state.elo.deception_game_elo = 1250.0
    state.elo.detection_belief_elo = 1240.0
    state.elo.epistemic_belief_elo = 1230.0
    state.belief_system.strategic_learnings.append("Rule 1")

    serialized = store.to_dict()
    self.assertIn("model_test::neutral", serialized["personas"])
    p_data = serialized["personas"]["model_test::neutral"]
    self.assertEqual(p_data["elo"]["deception_game_elo"], 1250.0)
    self.assertEqual(p_data["elo"]["detection_belief_elo"], 1240.0)
    self.assertEqual(p_data["elo"]["epistemic_belief_elo"], 1230.0)

  def test_experience_store_merge_from(self):
    k = cross_game_reflection.PersonaKey("model_test", "neutral")

    store_1 = cross_game_reflection.ExperienceStore()
    s1 = store_1.get_persona_state(k)
    s1.elo.deception_game_elo = 1300.0
    s1.elo.detection_game_elo = 1250.0
    s1.elo.games_played = 2
    s1.elo.role_counts["Demon"] = 2
    s1.belief_system.strategic_learnings = ["Learn A", "Learn B"]

    s1.belief_system.role_playbooks["Demon"] = ["Playbook A"]

    store_2 = cross_game_reflection.ExperienceStore()
    s2 = store_2.get_persona_state(k)
    s2.elo.deception_game_elo = 1200.0
    s2.elo.detection_game_elo = 1350.0
    s2.elo.games_played = 2
    s2.elo.role_counts["Demon"] = 1
    s2.elo.role_counts["Witness"] = 1
    s2.belief_system.strategic_learnings = ["Learn B", "Learn C"]
    s2.belief_system.role_playbooks["Demon"] = ["Playbook B"]

    store_1.merge_from(store_2)
    merged = store_1.get_persona_state(k)

    self.assertEqual(merged.elo.games_played, 4)
    # Weighted average: (1300*2 + 1200*2)/4 = 1250.0
    self.assertEqual(merged.elo.deception_game_elo, 1250.0)
    # Weighted average: (1250*2 + 1350*2)/4 = 1300.0
    self.assertEqual(merged.elo.detection_game_elo, 1300.0)
    self.assertEqual(merged.elo.role_counts["Demon"], 3)
    self.assertEqual(merged.elo.role_counts["Witness"], 1)
    self.assertEqual(
        merged.belief_system.strategic_learnings,
        ["Learn A", "Learn B", "Learn C"],
    )
    self.assertEqual(
        merged.belief_system.role_playbooks["Demon"],
        ["Playbook A", "Playbook B"],
    )

  def test_experience_store_import_learnings(self):
    temp_dir = self.create_tempdir()
    f1 = temp_dir.create_file("seed_0_store.json")
    f2 = temp_dir.create_file("seed_1_store.json")

    k = cross_game_reflection.PersonaKey("model_test", "neutral")
    s1 = cross_game_reflection.ExperienceStore()
    p1 = s1.get_persona_state(k)
    p1.elo.deception_game_elo = 1260.0
    p1.elo.games_played = 1
    p1.belief_system.strategic_learnings = ["Learn 1"]
    s1.save(f1.full_path)

    s2 = cross_game_reflection.ExperienceStore()
    p2 = s2.get_persona_state(k)
    p2.elo.deception_game_elo = 1240.0
    p2.elo.games_played = 1
    p2.belief_system.strategic_learnings = ["Learn 2"]
    s2.save(f2.full_path)

    # Test importing from directory
    master = cross_game_reflection.ExperienceStore(
        initial_learnings=temp_dir.full_path
    )
    m_state = master.get_persona_state(k)
    self.assertEqual(m_state.elo.games_played, 2)
    self.assertEqual(m_state.elo.deception_game_elo, 1250.0)
    self.assertIn("Learn 1", m_state.belief_system.strategic_learnings)
    self.assertIn("Learn 2", m_state.belief_system.strategic_learnings)

  def test_consolidate_learnings_with_llm_mock(self):
    class MockConsolidator:

      def sample_text(self, *args, **kwargs) -> str:
        del args, kwargs
        return (
            "```json\n"
            "{\n"
            '  "strategic_learnings": [\n'
            '    "On Day 1, execute if Outsider count is odd or a reliable ping'
            ' is present; otherwise pass.",\n'
            '    "Cross-reference total outsider claims against census N mod'
            ' 3."\n'
            "  ],\n"
            '  "common_pitfalls": [\n'
            '    "Over-trusting a single info ping when Poisoner is on'
            ' script.",\n'
            '    "Spending dead vote prematurely on Day 2."\n'
            "  ],\n"
            '  "role_playbooks": {\n'
            '    "Witness": [\n'
            '      "Share Townsfolk ping early to establish town baseline."\n'
            "    ],\n"
            '    "Demon": [\n'
            '      "Starpass to living Minion if Town suspicion becomes'
            ' overwhelming."\n'
            "    ]\n"
            "  }\n"
            "}\n"
            "```"
        )

    store = cross_game_reflection.ExperienceStore()
    current_belief = cross_game_reflection.BeliefSystem(
        strategic_learnings=["Old rule about voting."],
        role_playbooks={"Witness": ["Old witness rule."]},
    )
    raw_learnings = [
        "Player 4 lied about being Witness to protect Player 1 (Demon)."
    ]
    raw_rules_by_role = {
        "Demon": ["If Town suspects Player 1, Player 1 should kill self."],
        "Witness": ["Check Player 2 and Player 3."],
    }

    consolidated = store.consolidate_learnings(
        model=MockConsolidator(),
        current_belief=current_belief,
        new_learnings=raw_learnings,
        new_rules_by_role=raw_rules_by_role,
    )

    # 1. Verify structure is maintained
    self.assertIsInstance(consolidated, cross_game_reflection.BeliefSystem)
    self.assertLen(consolidated.strategic_learnings, 2)
    self.assertLen(consolidated.common_pitfalls, 2)
    self.assertIn("Witness", consolidated.role_playbooks)
    self.assertIn("Demon", consolidated.role_playbooks)

    # 2. Verify de-anecdotalization (no ephemeral player names).
    for s in consolidated.strategic_learnings:
      self.assertNotIn("Player 4", s)
      self.assertNotIn("Player 1", s)
    for p in consolidated.common_pitfalls:
      self.assertNotIn("Player 4", p)
      self.assertNotIn("Player 1", p)

    # 3. Verify strict role segregation (Demon vs Witness playbooks).
    self.assertIn(
        "Starpass to living Minion", consolidated.role_playbooks["Demon"][0]
    )
    self.assertIn(
        "Share Townsfolk ping early",
        consolidated.role_playbooks["Witness"][0],
    )

  def test_consolidate_learnings_fallback_on_invalid_json(self):
    class BrokenLLM:

      def sample_text(self) -> str:
        return "I am sorry, but I cannot format this as JSON."

    store = cross_game_reflection.ExperienceStore()
    current_belief = cross_game_reflection.BeliefSystem(
        strategic_learnings=["Existing rule 1."],
        common_pitfalls=["Existing pitfall 1."],
        role_playbooks={"Empath": ["Existing empath rule."]},
    )
    new_learnings = ["New general learning."]
    new_rules_by_role = {"Empath": ["New empath rule."]}

    # Should not throw exception and should fall back safely to deduplication
    fallback_belief = store.consolidate_learnings(
        model=BrokenLLM(),
        current_belief=current_belief,
        new_learnings=new_learnings,
        new_rules_by_role=new_rules_by_role,
    )

    self.assertIsInstance(fallback_belief, cross_game_reflection.BeliefSystem)
    self.assertIn("New general learning.", fallback_belief.strategic_learnings)
    self.assertIn("Existing rule 1.", fallback_belief.strategic_learnings)
    self.assertIn("Existing pitfall 1.", fallback_belief.common_pitfalls)
    self.assertIn("New empath rule.", fallback_belief.role_playbooks["Empath"])
    self.assertIn(
        "Existing empath rule.", fallback_belief.role_playbooks["Empath"]
    )

  def test_role_conditional_notepad_isolation(self):
    bs = cross_game_reflection.BeliefSystem(
        strategic_learnings=["Universal strategic directive A."],
        common_pitfalls=["Universal endgame pitfall B."],
        role_playbooks={
            "Witness": ["Townsfolk: Coordinate with your ping."],
            "Demon": ["Demon: Starpass to Minion if cornered."],
        },
    )

    # 1. Townsfolk (Witness) should only see Witness playbook
    witness_notepad = bs.format_for_notepad(role="Witness")
    self.assertIn("Universal strategic directive A.", witness_notepad)
    self.assertIn("Universal endgame pitfall B.", witness_notepad)
    self.assertIn("Playbook for Witness:", witness_notepad)
    self.assertIn("Townsfolk: Coordinate with your ping.", witness_notepad)
    self.assertNotIn("Playbook for Demon:", witness_notepad)
    self.assertNotIn("Starpass to Minion", witness_notepad)

    # 2. Demon should only see Demon playbook
    demon_notepad = bs.format_for_notepad(role="Demon")
    self.assertIn("Universal strategic directive A.", demon_notepad)
    self.assertIn("Universal endgame pitfall B.", demon_notepad)
    self.assertIn("Playbook for Demon:", demon_notepad)
    self.assertIn("Demon: Starpass to Minion if cornered.", demon_notepad)
    self.assertNotIn("Playbook for Witness:", demon_notepad)
    self.assertNotIn("Townsfolk: Coordinate with your ping.", demon_notepad)

    # 3. None role should only render generic sections
    generic_notepad = bs.format_for_notepad(role=None)
    self.assertIn("Universal strategic directive A.", generic_notepad)
    self.assertIn("Universal endgame pitfall B.", generic_notepad)
    self.assertNotIn("Playbook for", generic_notepad)


if __name__ == "__main__":
  absltest.main()
