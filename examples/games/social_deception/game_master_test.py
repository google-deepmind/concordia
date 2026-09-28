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

"""Tests for the social deception game master component."""

from absl.testing import absltest
from absl.testing import parameterized
from examples.games.social_deception import game_master
from examples.games.social_deception import game_tracker
from examples.games.social_deception.setup import scripts
from concordia.typing import entity as entity_lib


class MockAgent:

  def __init__(self, name):
    self.name = name
    self.observations = []

  def observe(self, observation: str):
    self.observations.append(observation)


class GameMasterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.players = [
        game_tracker.PlayerState(
            name="Alice",
            role="Servant",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Bob",
            role="Executioner",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Charlie",
            role="Witness",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Dave",
            role="Demon",
            alignment=game_tracker.Alignment.EVIL,
        ),
    ]
    self.grim = game_tracker.GameTracker(self.players)
    self.st = game_master.GameMaster(self.grim)
    self.st._discussion_turns_left = {
        name: 3 for name in self.grim.player_names
    }
    self.st._discussion_order = list(self.grim.player_names)

  def test_invalid_target_count_night_action(self):
    # Setup Dave (Demon) night action
    self.st._phase = game_master.Phase.NIGHT
    self.st._day_number = 2
    self.st._night_actors = ["Dave"]
    self.st._night_actor_index = 0

    # Act 1: Give '0' targets to Demon, expecting 1
    res1 = self.st._resolve_night_action("Dave", "I kill nobody.")
    # Actor index should remain at 0 because it's invalid
    self.assertEqual(self.st._night_actor_index, 0)
    self.assertIn("Invalid action: Please select one player to kill.", res1)

    # Act 2: Give '2' targets to Demon, expecting 1
    res2 = self.st._resolve_night_action("Dave", "I kill Alice and Bob.")
    self.assertEqual(self.st._night_actor_index, 0)
    self.assertIn(
        "Invalid action: Please select exactly one player to kill", res2
    )

    # Act 3: Give exactly '1' target to Demon
    res3 = self.st._resolve_night_action("Dave", "I kill Alice.")
    self.assertEqual(self.st._night_actor_index, 1)  # Directly advanced!
    self.assertIn("Alice", res3)

    # The transition to day happens when all night actors finish
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.DISCUSSION)
    self.assertIn("Rising and shining", self.st._public_log[-1])

  def test_invalid_target_count_daylight_actions(self):
    self.st._phase = game_master.Phase.DISCUSSION
    initial_turns = self.st._discussion_turns_left["Alice"]

    # 1. Invalid Whisper (0 targets)
    res_whisper_0 = self.st._resolve_discussion(
        "Alice", "whisper to no one: hello"
    )
    self.assertIn("Invalid whisper", res_whisper_0)
    # Ensure Alice didn't lose her turn
    self.assertEqual(self.st._discussion_turns_left["Alice"], initial_turns)

    # 2. Invalid Nomination (2 targets)
    res_nom_2 = self.st._resolve_discussion("Alice", "nominate Bob and Charlie")
    self.assertIn("Invalid nomination", res_nom_2)
    self.assertEqual(self.st._discussion_turns_left["Alice"], initial_turns)

    # 3. Invalid Day kill (Executioner, 2 targets)
    res_slay_2 = self.st._resolve_discussion("Bob", "shoot Alice and Charlie")
    self.assertIn("Invalid action", res_slay_2)
    # Ensure Bob didn't lose his turn
    self.assertEqual(
        self.st._discussion_turns_left["Bob"],
        self.st._discussion_turns_left["Bob"],
    )

    # 4. Valid Nomination (1 target)
    res_nom_valid = self.st._resolve_discussion(
        "Alice", "nominate Bob for execution"
    )
    self.assertNotIn("Invalid", res_nom_valid)
    # We transitioned states and used a turn
    self.assertEqual(self.st._discussion_turns_left["Alice"], initial_turns - 1)

    # 5. Invalid Pass (no reason)
    res_pass_invalid = self.st._resolve_discussion("Bob", "pass")
    self.assertIn("Invalid pass", res_pass_invalid)
    # Ensure Bob didn't lose his turn
    self.assertEqual(self.st._discussion_turns_left["Bob"], initial_turns)

    # 6. Valid Pass (with reason)
    res_pass_valid = self.st._resolve_discussion(
        "Bob", "pass because I have nothing to say"
    )
    self.assertNotIn("Invalid pass", res_pass_valid)
    self.assertEqual(self.st._discussion_turns_left["Bob"], initial_turns - 1)

    # Check that the valid pass was logged privately
    bob_whispers = self.st._private_whispers.get("Bob", [])
    self.assertTrue(
        any("You passed for the following reason:" in w for w in bob_whispers)
    )

  def test_servant_vote_restriction(self):
    self.grim.get_player("Alice").servant_master = "Bob"
    # Setup a nomination
    self.st._phase = game_master.Phase.VOTING
    self.st._current_nomination = ("Charlie", "Dave")
    self.st._voter_index = 0

    # Voter order: Alice (0), Bob (1), Charlie (2), Dave (3)
    # Alice (Servant) votes YES
    self.st._resolve_voting("Alice", "yes")

    # Bob (Master) votes NO
    self.st._resolve_voting("Bob", "no")

    # Finalize voting (Charlie and Dave vote NO)
    self.st._resolve_voting("Charlie", "no")
    self.st._resolve_voting("Dave", "no")

    # Check the log for the rejection message
    log_dump = "\n".join(self.st._public_log).lower()
    self.assertIn("not counted", log_dump)
    # Check that Alice's YES vote was NOT counted
    self.assertEqual(self.st._votes_for_current_nominee, 0)

  def test_servant_vote_success(self):
    self.grim.get_player("Alice").servant_master = "Bob"
    # Setup a nomination
    self.st._phase = game_master.Phase.VOTING
    self.st._current_nomination = ("Charlie", "Dave")
    self.st._voter_index = 0

    # Alice (Servant) votes YES
    self.st._resolve_voting("Alice", "yes")

    # Bob (Master) votes YES
    self.st._resolve_voting("Bob", "yes")

    # Charlie and Dave vote NO
    self.st._resolve_voting("Charlie", "no")
    self.st._resolve_voting("Dave", "no")

    # Both votes should be counted
    self.assertEqual(self.st._votes_for_current_nominee, 2)

  def test_executioner_shot_in_discussion(self):
    self.st._phase = game_master.Phase.DISCUSSION
    # Bob (Executioner) shoots Dave (Demon)
    res = self.st._resolve_discussion("Bob", "shoot Dave")
    self.assertIn("die", res)
    self.assertEqual(
        self.grim.get_player("Dave").status, game_tracker.PlayerStatus.DEAD
    )

  def test_innocent_nomination(self):
    self.grim.get_player("Dave").role = "Innocent"  # Set nominee as Innocent
    self.grim.get_player("Bob").role = "Executioner"  # Townsfolk

    self.st._phase = game_master.Phase.TOWN_SQUARE
    # Bob (Townsfolk) nominates Dave (Innocent)
    res = self.st._resolve_town_square("Bob", "nominate Dave")
    self.assertIn("instantly executed", res)
    self.assertEqual(
        self.grim.get_player("Bob").status, game_tracker.PlayerStatus.DEAD
    )

  def test_private_role_introduction(self):
    # Check that Alice (Servant) got her private intro
    alice_whispers = self.st._private_whispers.get("Alice", [])
    self.assertTrue(any("You are the Servant" in w for w in alice_whispers))
    self.assertTrue(any("your master" in w.lower() for w in alice_whispers))

    # Check that Dave (Demon) got his private intro
    dave_whispers = self.st._private_whispers.get("Dave", [])
    self.assertTrue(any("You are the Demon" in w for w in dave_whispers))

  def test_player_cannot_pass(self):
    self.st._player_can_pass = False

    self.st._phase = game_master.Phase.DISCUSSION
    self.st._discussion_turns_left["Bob"] = 3
    # Try passing in DISCUSSION
    res_pass_invalid = self.st._resolve_discussion(
        "Bob", "pass because I want to"
    )
    self.assertIn("Invalid action", res_pass_invalid)
    self.assertEqual(
        self.st._discussion_turns_left["Bob"], 3
    )  # Did not use turn

    # Try passing in TOWN_SQUARE
    self.st._phase = game_master.Phase.TOWN_SQUARE
    res_pass_ts = self.st._resolve_town_square("Bob", "pass")
    self.assertIn("Invalid action", res_pass_ts)
    self.assertNotIn("Bob", self.st._nominators_passed)

  def test_full_setup_flow(self):
    self.st._phase = game_master.Phase.SETUP
    self.st._setup_player_index = 0

    # 1. SETUP phase: should iterate through all 4 players
    for _ in range(4):
      self.assertEqual(self.st._phase, game_master.Phase.SETUP)
      acting = self.st._handle_next_acting()
      spec = self.st._handle_next_action_spec()
      self.assertIn("free", spec)
      self.assertIn(acting, spec)
      self.st._last_event = f"{acting}: Ack"
      self.st._handle_resolve(None)

    # After 4 setup steps, calling handle_next_acting should transition to
    # ROLE_REFLECTION
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.ROLE_REFLECTION)

    # 2. ROLE_REFLECTION phase: should iterate through all 4 players
    for _ in range(4):
      self.assertEqual(self.st._phase, game_master.Phase.ROLE_REFLECTION)
      acting = self.st._handle_next_acting()
      spec = self.st._handle_next_action_spec()
      self.assertIn("free", spec)
      self.assertIn("Reflect on your role", spec)
      self.st._last_event = f"{acting}: I am ready."
      self.st._handle_resolve(None)

    # After 4 reflections, calling handle_next_acting should transition to
    # NIGHT 1
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.NIGHT)
    self.assertEqual(self.st._day_number, 1)

  def test_info_only_night_action(self):
    # 1. Start night 1 properly to populate cache
    self.st._start_night(1)

    # Dave (Demon) acts first to get bluffs in this setup.
    acting_dave = self.st._handle_next_acting()
    self.assertEqual(acting_dave, "Dave")
    self.st._last_event = "Dave: Ack"
    self.st._handle_resolve(None)

    # Charlie is Witness. She should be next to act.
    acting = self.st._handle_next_acting()
    self.assertEqual(acting, "Charlie")

    # 1. Witness (Charlie): Should prompt with ACTUAL INFO
    spec_json = self.st._handle_next_action_spec()
    self.assertIn("The Game Master reveals", spec_json)

    cached_info = self.st._night_result_cache["Charlie"]
    # We must set _last_event so that _handle_resolve can find the actor.
    self.st._last_event = "Charlie: Ack"
    res = self.st._handle_resolve(None)
    self.assertEqual(res, f"Charlie resolves: {cached_info}")

    # 2. Demon (Dave) on Night 2: Not info-only, should prompt for FREE action
    self.st._start_night(2)
    # Find Dave's position in night actors
    self.st._night_actor_index = self.st._night_actors.index("Dave")
    spec_json_dave = self.st._handle_next_action_spec()
    self.assertIn("free", spec_json_dave)
    self.assertIn("select one player to kill", spec_json_dave)

  def test_night_skip_inactive_roles(self):
    players = [
        game_tracker.PlayerState(
            name="Alice",
            role="Gravedigger",
            alignment=game_tracker.Alignment.GOOD,
        ),
        game_tracker.PlayerState(
            name="Bob", role="Witness", alignment=game_tracker.Alignment.GOOD
        ),
        game_tracker.PlayerState(
            name="Charlie", role="Demon", alignment=game_tracker.Alignment.EVIL
        ),
    ]
    grim = game_tracker.GameTracker(players)
    st = game_master.GameMaster(grim)

    # Transition to Night 1
    st._phase = game_master.Phase.NIGHT
    st._day_number = 1
    st._night_actors = ["Alice", "Bob", "Charlie"]
    st._night_actor_index = 0

    # Alice (Gravedigger) should be skipped because resolve_night_action is ""
    # on Night 1. Bob (Witness) should be next to act because she has info.
    # Charlie (Demon) should also be skipped because resolve_night_action is ""
    # on Night 1.

    # We call _get_night_actors_for_day to populate the cache
    st._night_result_cache = {}
    st._night_actors = st._get_night_actors_for_day(1)
    st._night_actor_index = 0

    # On Night 1, active and info roles wake first, followed by passive roles
    self.assertEqual(st._night_actors, ["Charlie", "Bob", "Alice"])

    acting = st._handle_next_acting()
    self.assertEqual(acting, "Charlie")

    # Resolve Charlie's action (receiving bluffs)
    st._last_event = "Charlie: Ack"
    st._handle_resolve(None)

    acting = st._handle_next_acting()
    self.assertEqual(acting, "Bob")

    # Action spec for Bob (Witness) should be her specific role info.
    spec_json = st._handle_next_action_spec()
    self.assertIn("The Game Master reveals", spec_json)

    # Resolve Bob's action. Must set _last_event.
    st._last_event = "Bob: Ack"
    st._handle_resolve(None)

    # Resolve Alice's Night 1 wake
    acting = st._handle_next_acting()
    self.assertEqual(acting, "Alice")
    st._last_event = "Alice: Ack"
    st._handle_resolve(None)

    # After all night actors, it should transition to day.
    self.assertEqual(st._phase, game_master.Phase.DISCUSSION)

    # Verify we are in discussion and some player is acting
    self.assertEqual(st._phase, game_master.Phase.DISCUSSION)
    next_acting = st._handle_next_acting()
    self.assertIn(next_acting, ["Alice", "Bob", "Charlie"])

  def test_agent_observations(self):
    # Clear initial setup observations
    for name in self.grim.player_names:
      self.st._handle_observation(
          entity_lib.ActionSpec(
              call_to_action=f"What is the current situation faced by {name}?",
              output_type=entity_lib.OutputType.MAKE_OBSERVATION,
          )
      )

    # 1. Test public log queues to everyone
    self.st._log_pub("It is a beautiful day.")
    for name in self.grim.player_names:
      obs = self.st._handle_observation(
          entity_lib.ActionSpec(
              call_to_action=f"What is the current situation faced by {name}?",
              output_type=entity_lib.OutputType.MAKE_OBSERVATION,
          )
      )
      self.assertIn("[PUBLIC] It is a beautiful day.", obs)

    # 2. Test private log queues only to target
    self.st._log_priv("Alice", "You are special.")
    alice_obs = self.st._handle_observation(
        entity_lib.ActionSpec(
            call_to_action="What is the current situation faced by Alice?",
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    self.assertIn("[PRIVATE] You are special.", alice_obs)
    # Others should not see it
    for name in ["Bob", "Charlie", "Dave"]:
      other_obs = self.st._handle_observation(
          entity_lib.ActionSpec(
              call_to_action=f"What is the current situation faced by {name}?",
              output_type=entity_lib.OutputType.MAKE_OBSERVATION,
          )
      )
      self.assertNotIn("[PRIVATE] You are special.", other_obs)

    # 3. Test whisper logic: Public notification but private content
    self.st._resolve_discussion(
        "Alice",
        '{"action": "whisper", "target": "Bob", "message": "I am the Demon"}',
    )

    # Alice and Bob should have the private content
    alice_obs = self.st._handle_observation(
        entity_lib.ActionSpec(
            call_to_action="What is the current situation faced by Alice?",
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    bob_obs = self.st._handle_observation(
        entity_lib.ActionSpec(
            call_to_action="What is the current situation faced by Bob?",
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    self.assertIn(
        "[PRIVATE] Alice (whisper to Bob): I am the Demon",
        alice_obs,
    )
    self.assertIn(
        "[PRIVATE] Alice (whisper to Bob): I am the Demon",
        bob_obs,
    )

    # Everyone (including Alice and Bob) should see the public notification
    public_notif = "[PUBLIC] Alice whispers to Bob."
    self.assertIn(public_notif, alice_obs)
    self.assertIn(public_notif, bob_obs)

    # Charlie and Dave should NOT have the private content,
    # but should receive the public notification.
    for name in ["Charlie", "Dave"]:
      other_obs = self.st._handle_observation(
          entity_lib.ActionSpec(
              call_to_action=f"What is the current situation faced by {name}?",
              output_type=entity_lib.OutputType.MAKE_OBSERVATION,
          )
      )
      self.assertIn(public_notif, other_obs)
      self.assertNotIn("I am the Demon", other_obs)

  def test_full_game_progression_day_numbers(self):
    # This test verifies the fix for the day number and phase transitions.
    # 1. Start in SETUP
    self.st._phase = game_master.Phase.SETUP
    self.st._setup_player_index = 0
    for _ in range(4):
      acting = self.st._handle_next_acting()
      self.st._last_event = f"{acting}: Ack"
      self.st._handle_resolve(None)

    # Transition to ROLE_REFLECTION should happen automatically on next acting
    # call
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.ROLE_REFLECTION)

    # 2. ROLE_REFLECTION
    for _ in range(4):
      acting = self.st._handle_next_acting()
      self.st._last_event = f"{acting}: I am ready."
      self.st._handle_resolve(None)

    # 3. Transition to NIGHT 1
    # Next acting call should trigger transition to Night 1
    acting_n1 = self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.NIGHT)
    self.assertEqual(self.st._day_number, 1)

    # Dave (Demon) acts first to get bluffs
    self.assertEqual(acting_n1, "Dave")
    self.st._last_event = "Dave: Ack"
    self.st._handle_resolve(None)

    # Charlie is Witness, she should act next
    # Populate cache (simulates _start_night)
    self.st._night_result_cache["Charlie"] = "The Storyteller shows you that..."

    acting_n2 = self.st._handle_next_acting()
    self.assertEqual(acting_n2, "Charlie")

    # Resolve Charlie's action
    self.st._last_event = "Charlie: Ack"
    self.st._handle_resolve(None)

    # Next actor in Night 1
    # Servant (Alice) wakes every night.
    # Executioner (Bob) doesn't wake.
    # Demon (Dave) doesn't wake on Night 1.
    next_acting = self.st._handle_next_acting()
    self.assertEqual(next_acting, "Alice")
    self.st._last_event = "Alice: I pick Bob as my master."
    self.st._handle_resolve(None)

    # MUST ACK the result for Bob (Executioner Night 1 wake)
    next_acting = self.st._handle_next_acting()
    self.assertEqual(next_acting, "Bob")
    self.st._last_event = "Bob: Ack"
    self.st._handle_resolve(None)

    next_acting = self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.DISCUSSION)
    # CRITICAL: Day number should be 1, NOT 2.
    self.assertEqual(self.st._day_number, 1)

    # 4. Finish DISCUSSION
    # Use up all turns
    for name in self.players:
      self.st._discussion_turns_left[name.name] = 0

    # Transition to REFLECTION
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.REFLECTION)

    # 5. REFLECTION
    for _ in range(4):
      acting = self.st._handle_next_acting()
      self.st._last_event = f"{acting}: Reflecting."
      self.st._handle_resolve(None)

    # 6. TOWN_SQUARE
    next_acting = self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.TOWN_SQUARE)

    # Nominate and execute
    # (Simplified: pass everyone)
    for name in self.players:
      self.st._nominators_passed.add(name.name)

    # Transition to EXECUTION then NIGHT 2
    self.st._handle_next_acting()
    self.assertEqual(self.st._phase, game_master.Phase.NIGHT)
    # SHOULD BE NIGHT 2
    self.assertEqual(self.st._day_number, 2)

  def test_nominator_voting_heuristic_allows_yes_or_no(self):
    self.st._phase = game_master.Phase.VOTING
    self.st._current_nomination = ("Alice", "Bob")
    spec_json = self.st._handle_next_action_spec()
    self.assertIn("Alice has nominated Bob for execution", spec_json)
    self.assertIn("free to vote 'yes' or 'no'", spec_json)

  def test_early_win_condition_no_living_good_players(self):
    # Kill all good players (Alice, Bob, Charlie)
    self.st._grimoire.players["Alice"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Bob"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Charlie"].status = game_tracker.PlayerStatus.DEAD
    # Add 2 alive evil players (so 3 total alive evil players:
    # Dave + Minion1 + Minion2 > 2)
    self.st._grimoire.players["Minion1"] = game_tracker.PlayerState(
        name="Minion1", role="Spy", alignment=game_tracker.Alignment.EVIL
    )
    self.st._grimoire.players["Minion2"] = game_tracker.PlayerState(
        name="Minion2", role="Poisoner", alignment=game_tracker.Alignment.EVIL
    )
    win_res = self.st.check_early_game_over()
    self.assertIsNotNone(win_res)
    self.assertIn("Evil wins! No living Good players remain", win_res)
    self.assertEqual(True, self.st._check_win_conditions())

  def test_early_win_condition_insufficient_good_votes(self):
    # 3 players alive: Bob (Good), Minion (Evil), Dave (Demon, Evil)
    # Alice (Dead, spent dead vote), Charlie (Dead, spent dead vote)
    self.st._grimoire.players["Alice"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Alice"].has_spent_dead_vote = True
    self.st._grimoire.players["Charlie"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Charlie"].has_spent_dead_vote = True

    # Bob's default role in test fixture is Executioner. If Executioner has
    # not used ability, Good still has a win out via Executioner shot!
    self.st._grimoire.players["Minion"] = game_tracker.PlayerState(
        name="Minion", role="Spy", alignment=game_tracker.Alignment.EVIL
    )
    win_res_with_executioner = self.st.check_early_game_over()
    self.assertIsNone(win_res_with_executioner)

    # Once Executioner ability is used (or if Bob is Soldier), Good has no
    # win out
    self.st._grimoire.players["Bob"].has_used_single_use_ability = True
    win_res = self.st.check_early_game_over()
    self.assertIsNotNone(win_res)
    self.assertIn(
        "Evil wins! Good cannot achieve enough votes to execute the Demon.",
        win_res,
    )
    self.assertEqual(True, self.st._check_win_conditions())

  def test_apprentice_promotion_when_5_alive(self):
    # 5 alive players remaining after Demon dies:
    # Alice, Bob, Charlie, Frank (Good), Eve (Apprentice)
    self.st._grimoire.players["Frank"] = game_tracker.PlayerState(
        name="Frank", role="Soldier", alignment=game_tracker.Alignment.GOOD
    )
    self.st._grimoire.players["Eve"] = game_tracker.PlayerState(
        name="Eve", role="Apprentice", alignment=game_tracker.Alignment.EVIL
    )
    # Demon dies
    self.st._grimoire.players["Dave"].status = game_tracker.PlayerStatus.DEAD
    win_res = self.st._check_win_conditions()
    # Game should continue: Apprentice promoted to Demon
    self.assertIsNone(win_res)
    self.assertEqual(self.st._grimoire.players["Eve"].role, "Demon")

  def test_apprentice_fails_to_promote_when_less_than_5_alive(self):
    # 4 alive players: Alice, Bob (Good), Dave (Demon), Eve (Apprentice)
    self.st._grimoire.players["Charlie"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Eve"] = game_tracker.PlayerState(
        name="Eve", role="Apprentice", alignment=game_tracker.Alignment.EVIL
    )
    # Demon dies
    self.st._grimoire.players["Dave"].status = game_tracker.PlayerStatus.DEAD
    win_res = self.st._check_win_conditions()
    self.assertEqual(win_res, "Good wins! The Demon has been slain.")

  def test_saint_executed_causes_evil_win(self):
    self.st._grimoire.players["Alice"].role = "Saint"
    self.st._grimoire.players["Alice"].status = game_tracker.PlayerStatus.DEAD
    win_res = self.st._check_win_conditions(executed_today="Alice")
    self.assertEqual(win_res, "Evil wins! The Saint has been executed.")

  def test_impaired_saint_executed_does_not_cause_evil_win(self):
    self.st._grimoire.players["Alice"].role = "Saint"
    self.st._grimoire.players["Alice"].status = game_tracker.PlayerStatus.DEAD
    self.st._grimoire.players["Alice"].is_poisoned = True
    win_res = self.st._check_win_conditions(executed_today="Alice")
    self.assertNotEqual(win_res, "Evil wins! The Saint has been executed.")

  def test_mayor_three_player_peace_win(self):
    # 3 players alive: Alice (Mayor, Good), Bob (Good), Dave (Demon, Evil)
    self.st._grimoire.players["Alice"].role = "Mayor"
    self.st._grimoire.players["Charlie"].status = game_tracker.PlayerStatus.DEAD
    # No execution today
    win_res = self.st._check_win_conditions(executed_today=None)
    self.assertIsInstance(win_res, str)
    self.assertIn("Good wins! The Mayor has secured peace", str(win_res))

  def test_executioner_shot_when_poisoned_wastes_ability_without_killing_demon(
      self,
  ):
    self.st._phase = game_master.Phase.DISCUSSION
    self.st._grimoire.players["Bob"].is_poisoned = True
    # Bob (Poisoned Executioner) shoots Dave (Demon)
    res = self.st._resolve_discussion("Bob", "shoot Dave")
    self.assertIn("nothing happens", res)
    # Dave is still ALIVE
    self.assertEqual(
        self.st._grimoire.players["Dave"].status,
        game_tracker.PlayerStatus.ALIVE,
    )
    # Bob's ability is spent
    self.assertTrue(
        self.st._grimoire.players["Bob"].has_used_single_use_ability
    )

  def test_innocent_nomination_by_outsider_does_not_execute_nominator(self):
    self.st._grimoire.players["Dave"].role = "Innocent"
    self.st._grimoire.players["Alice"].role = "Servant"  # Outsider
    self.st._phase = game_master.Phase.TOWN_SQUARE
    res = self.st._resolve_town_square("Alice", "nominate Dave")
    self.assertNotIn("instantly executed", res)
    self.assertEqual(
        self.st._grimoire.players["Alice"].status,
        game_tracker.PlayerStatus.ALIVE,
    )

  def test_base_game_multi_demon_win_condition(self):

    script = scripts.BaseGame()
    players = [
        game_tracker.PlayerState(
            name=f"Town_{i}",
            role="BasicTownsfolk",
            alignment=game_tracker.Alignment.GOOD,
        )
        for i in range(5)
    ] + [
        game_tracker.PlayerState(
            name=f"Demon_{i}",
            role="Demon",
            alignment=game_tracker.Alignment.EVIL,
        )
        for i in range(2)
    ]
    tracker = game_tracker.GameTracker(players, script=script)
    gm = game_master.GameMaster(tracker)

    # 1. Initially both Demons are alive -> No win
    self.assertIsNone(gm._check_win_conditions())

    # 2. First Demon is executed -> Demon_2 is still alive -> No win yet
    tracker.players["Demon_0"].status = game_tracker.PlayerStatus.DEAD
    self.assertIsNone(gm._check_win_conditions(executed_today="Demon_0"))

    # 3. Second Demon is executed -> All Demons dead -> Good wins!
    tracker.players["Demon_1"].status = game_tracker.PlayerStatus.DEAD
    win_res = gm._check_win_conditions(executed_today="Demon_1")
    self.assertIsInstance(win_res, str)
    self.assertIn("Good wins", win_res)


if __name__ == "__main__":
  absltest.main()
