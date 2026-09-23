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

"""Game master implementation for Social Deception."""

from collections.abc import Mapping, Sequence
import enum
import json
import logging
import math
import os
import random
import re
from typing import Any

from concordia.components.game_master import event_resolution
from examples.games.social_deception import game_tracker
from examples.games.social_deception.roles import basic_epistemic_town
from examples.games.social_deception.setup import setup_utils
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class Phase(enum.Enum):
  SETUP = "setup"
  ROLE_REFLECTION = "role_reflection"
  NIGHT = "night"
  DAY_START = "day_start"
  DISCUSSION = "discussion"
  REFLECTION = "reflection"
  TOWN_SQUARE = "town_square"
  DEFENSE = "defense"
  VOTING = "voting"
  EXECUTION = "execution"
  GAME_OVER = "game_over"


class DaylightAction(enum.Enum):
  """Possible actions for players during the day phase."""

  BROADCAST = ("broadcast", "broadcast [message]", None)
  WHISPER = ("whisper", "whisper to [player]: [message]", None)
  REPLY = ("reply", "reply to [player]: [message]", None)
  DAY_KILL = (
      "shoot",
      "shoot [player] (Executioner)",
      ["Executioner"],
  )
  NOMINATE = ("nominate", "nominate [player] for execution", None)
  PASS = ("pass", "pass", None)

  def __init__(
      self,
      command_key: str,
      output_string: str,
      role_filter: Sequence[str] | None = None,
  ):
    self.command_key = command_key
    self.output_string = output_string
    self.role_filter = role_filter


def _get_role_instance(role_name: str):
  clean_name = role_name.replace(" ", "")
  if hasattr(basic_epistemic_town, clean_name):
    return getattr(basic_epistemic_town, clean_name)()
  raise ValueError(f"Unknown role: {role_name}")


class GameMaster(entity_component.ContextComponent):
  """Game Master component that manages the Social Deception game loop."""

  def __init__(
      self,
      tracker_obj: game_tracker.GameTracker | None = None,
      grimoire_obj: game_tracker.GameTracker | None = None,
      player_can_pass: bool = True,
      max_day: int = -1,
      clean_log_file: str | None = None,
      turns_per_player: int = 3,
  ):
    super().__init__()
    self._tracker = tracker_obj if tracker_obj is not None else grimoire_obj
    if self._tracker is None:
      raise ValueError("Either tracker_obj or grimoire_obj must be provided.")
    self._grimoire = self._tracker

    self._player_can_pass = player_can_pass
    self._max_day = max_day
    self._agents = {}
    self._logging_in_progress = False
    self._phase = Phase.SETUP
    self._day_number = 0

    if clean_log_file:
      self._clean_log_path = clean_log_file
      dir_name = os.path.dirname(self._clean_log_path)
      try:
        if dir_name:
          os.makedirs(dir_name, exist_ok=True)
        with open(self._clean_log_path, "w") as f:
          f.write("--- Clean Game Log Started ---\n")
      except (OSError, IOError) as e:
        logging.warning(
            "Failed to initialize clean log file %s: %s. Skipping clean log.",
            self._clean_log_path,
            e,
        )
        self._clean_log_path = None
    else:
      self._clean_log_path = None

    self._player_names = sorted(self._grimoire.player_names)
    self._discussion_order = list(self._player_names)
    self._discussion_turns_left = {}
    self._discussion_player_index = 0
    self._setup_player_index = 0
    self._discussion_retries = {}
    self._reflection_player_index = 0
    self._turns_per_player = turns_per_player
    self._outstanding_two_way_whispers = {}
    self._notified_dead_players = set()

    self._nominators_passed = set()
    self._town_square_retries = {}
    self._current_nomination = None  # (nominator, nominee)
    self._highest_vote_count = 0
    self._players_on_the_block = []

    self._voter_index = 0
    self._votes_for_current_nominee = 0
    self._votes_this_nomination = {}  # Dict[str, bool]
    self._current_expected_player = None

    self._night_actors = []
    self._night_actor_index = 0
    self._night_result_cache = {}  # Dict[str, str]
    self._night_retry_count = 0

    self._public_log = []
    self._private_whispers = {}  # Dict[str, List[str]]
    self._observation_queue = {name: [] for name in self._player_names}
    self._last_event = ""

    self._role_instances = {}
    for name, p in self._grimoire.players.items():
      instance = _get_role_instance(p.perceived_role)
      instance.player_name = name
      self._role_instances[name] = instance
      introduction = instance.player_introduction()
      self._log_priv(name, f"You are the {p.perceived_role}. {introduction}")

    # Log overhead introductions for the Game Master
    self._log_priv("Game Master", "Overhead Introductions for this game:")
    for name, instance in self._role_instances.items():
      self._log_priv(
          "Game Master",
          f"{name} ({instance.role_name}): {instance.overhead_introduction()}",
      )

    # Start directly at Night 1
    self._phase = Phase.NIGHT
    self._day_number = 1
    self._start_night(1)

  def pre_observe(self, observation: str) -> str:
    if event_resolution.PUTATIVE_EVENT_TAG in observation:
      self._last_event = observation.split(
          event_resolution.PUTATIVE_EVENT_TAG, 1
      )[1].strip()

      # Prevent private content from persisting in GM memory.
      # During SETUP and ROLE_REFLECTION, putative events contain
      # secret role info (e.g. "I am the Demon"). Returning empty
      # string tells the ObservationToMemory component to skip it.
      if self._phase in (Phase.SETUP, Phase.ROLE_REFLECTION):
        return ""

    return ""

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    self._check_and_notify_new_deaths()
    output_type = action_spec.output_type
    self._log_clean(
        f"DEBUG: GameMaster.pre_act called with output_type={output_type}"
    )

    if output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      return self._handle_observation(action_spec)
    elif output_type == entity_lib.OutputType.RESOLVE:
      return self._handle_resolve(action_spec)
    elif output_type == entity_lib.OutputType.NEXT_ACTING:
      return self._handle_next_acting()
    elif output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return self._handle_next_action_spec()
    elif output_type == entity_lib.OutputType.TERMINATE:
      logging.info("Handling TERMINATE in GameMaster. Phase is %s", self._phase)

      if self._phase == Phase.GAME_OVER:
        logging.info("Terminating due to GAME_OVER")
        return "Yes"
      if self._max_day != -1 and self._day_number > self._max_day:
        logging.info("Terminating due to max day reached")
        return "Yes"
      return "No"

    return ""

  def get_state(self) -> Mapping[str, Any]:
    return {
        "phase": self._phase.value,
        "day_number": self._day_number,
        "public_log": self._public_log,
        "private_whispers": self._private_whispers,
        "highest_vote_count": self._highest_vote_count,
        "players_on_the_block": self._players_on_the_block,
        "setup_player_index": self._setup_player_index,
        "reflection_player_index": self._reflection_player_index,
        "night_actor_index": self._night_actor_index,
        "night_actors": (
            list(self._night_actors) if hasattr(self, "_night_actors") else []
        ),
        "night_result_cache": (
            dict(self._night_result_cache)
            if hasattr(self, "_night_result_cache")
            else {}
        ),
        "grimoire": self._grimoire.get_state(),
        "notified_dead_players": list(self._notified_dead_players),
        "observation_queue": {
            k: list(v) for k, v in self._observation_queue.items()
        },
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._phase = Phase(state["phase"])
    self._day_number = state["day_number"]
    self._public_log = list(state["public_log"])
    self._private_whispers = dict(state["private_whispers"])
    self._highest_vote_count = state["highest_vote_count"]
    self._players_on_the_block = list(state["players_on_the_block"])
    self._setup_player_index = state.get("setup_player_index", 0)
    self._reflection_player_index = state.get("reflection_player_index", 0)
    self._night_actor_index = state.get("night_actor_index", 0)
    self._night_actors = list(state.get("night_actors", []))
    self._night_result_cache = dict(state.get("night_result_cache", {}))
    if "grimoire" in state:
      self._grimoire.set_state(state["grimoire"])
    self._notified_dead_players = set(state.get("notified_dead_players", []))
    self._observation_queue = {
        k: list(v) for k, v in state.get("observation_queue", {}).items()
    }
    for name in self._player_names:
      if name not in self._observation_queue:
        self._observation_queue[name] = []

  def set_agents(self, agents: Mapping[str, Any]) -> None:
    """Registers agent entities to receive observations."""
    self._agents = dict(agents)

  def _find_player(self, target: str) -> str | None:
    """Finds a player name by fuzzy matching spaces and underscores."""
    if not target:
      return None
    target_search = target.replace("_", " ").lower()
    for name in self._grimoire.players:
      if name.replace("_", " ").lower() == target_search:
        return name
    # Try regex as fallback
    for name in self._grimoire.players:
      if re.search(
          rf"\b{re.escape(name.replace('_', ' '))}\b",
          target_search,
          re.IGNORECASE,
      ):
        return name
    return None

  @property
  def phase(self) -> Phase:
    return self._phase

  @property
  def day_number(self) -> int:
    return self._day_number

  @property
  def game_over_reason(self) -> str | None:
    return self._grimoire.game_over_reason

  @property
  def winner(self) -> game_tracker.Alignment | None:
    reason = self._grimoire.game_over_reason or ""
    if "good wins" in reason.lower():
      return game_tracker.Alignment.GOOD
    if "evil wins" in reason.lower():
      return game_tracker.Alignment.EVIL
    alive = {
        n: p
        for n, p in self._grimoire.players.items()
        if p.status == game_tracker.PlayerStatus.ALIVE
    }
    if not any(self._grimoire.is_type(n, "Demon") for n in alive):
      return game_tracker.Alignment.GOOD
    if len(alive) <= 2:
      return game_tracker.Alignment.EVIL
    return None

  @property
  def turns_per_player(self) -> int:
    return self._turns_per_player

  @turns_per_player.setter
  def turns_per_player(self, value: int) -> None:
    self._turns_per_player = value

  def _log_clean(self, message: str) -> None:
    """Logs a message directly to the clean log file."""
    if self._clean_log_path:
      try:
        with open(self._clean_log_path, "a") as f:
          f.write(f"{message}\n")
      except (OSError, IOError) as e:
        logging.warning(
            "Failed to write to clean log file %s: %s", self._clean_log_path, e
        )

  def _log_pub(self, message: str) -> None:
    """Logs a public message and queues it for all players.

    Args:
      message: The public message content to log and broadcast.
    """
    self._public_log.append(f"{message} [PUBLIC]")
    tagged_message = f"[PUBLIC] {message}"
    if self._clean_log_path:
      try:
        with open(self._clean_log_path, "a") as f:
          f.write(f"{message}\n")
      except (OSError, IOError) as e:
        logging.warning(
            "Failed to write to clean log file %s: %s", self._clean_log_path, e
        )
    for name in self._player_names:
      self._observation_queue.setdefault(name, []).append(tagged_message)

  def _log_priv(
      self, player_name: str, message: str, public_notify: bool = False
  ) -> None:
    """Logs a private message and queues it for the targeted player.

    Args:
      player_name: The name of the player to send the private message to.
      message: The private message content.
      public_notify: Whether to publicly notify the town that a whisper
        occurred.
    """
    self._private_whispers.setdefault(player_name, []).append(
        f"{message} [PRIVATE]"
    )
    if self._clean_log_path and player_name not in (
        "Game Master",
        "Storyteller",
    ):
      try:
        with open(self._clean_log_path, "a") as f:
          f.write(f"[PRIVATE] {message}\n")
      except (OSError, IOError) as e:
        logging.warning(
            "Failed to write to clean log file %s: %s", self._clean_log_path, e
        )
    if player_name in self._player_names:
      self._observation_queue.setdefault(player_name, []).append(
          f"[PRIVATE] {message}"
      )

    if public_notify and not self._logging_in_progress:
      self._logging_in_progress = True
      try:
        self._log_pub(f"Game Master whispers to {player_name}.")
      finally:
        self._logging_in_progress = False

  def _check_and_notify_new_deaths(self):
    for name, p in self._grimoire.players.items():
      if p.status == game_tracker.PlayerStatus.DEAD:
        if name not in self._notified_dead_players:
          self._notified_dead_players.add(name)
          self._log_priv(
              name,
              "🚨 STATUS CHANGE: You are now DEAD. Your active ability is"
              " disabled. You have exactly 1 dead vote remaining.",
          )
      else:
        if name in self._notified_dead_players:
          self._notified_dead_players.remove(name)

  def _handle_next_acting(self) -> str:
    self._log_clean(
        f"DEBUG: GameMaster._handle_next_acting called, phase={self._phase}"
    )

    while True:
      if self._phase == Phase.SETUP:
        if self._setup_player_index < len(self._player_names):
          return self._player_names[self._setup_player_index]
        self._phase = Phase.ROLE_REFLECTION
        self._reflection_player_index = 0
        continue

      if self._phase == Phase.ROLE_REFLECTION:
        if self._reflection_player_index < len(self._player_names):
          return self._player_names[self._reflection_player_index]
        self._start_night(1)
        continue

      if self._phase == Phase.NIGHT:
        self._log_clean(
            f"DEBUG: NIGHT index={self._night_actor_index}, "
            f"night_actors={len(self._night_actors)}"
        )
        while self._night_actor_index < len(self._night_actors):
          current_actor = self._night_actors[self._night_actor_index]
          role = self._role_instances.get(current_actor)
          self._log_clean(
              f"DEBUG: NIGHT index={self._night_actor_index}, "
              f"current_actor={current_actor}, role={role}"
          )
          if role:
            if current_actor in self._night_result_cache:
              return current_actor
            else:
              self._log_clean(f"DEBUG: NIGHT cache miss for {current_actor}")
          self._night_actor_index += 1

        self._log_clean("DEBUG: NIGHT calling _transition_to_day")
        self._transition_to_day()
        continue

      if self._phase == Phase.DISCUSSION:
        logging.info("In DISCUSSION phase in _handle_next_acting")
        if all(count <= 0 for count in self._discussion_turns_left.values()):
          logging.info(
              "All players finished discussion, transitioning to reflection"
          )
          self._transition_to_reflection()
          continue

        # Cycle until we find a player who still has turns left
        while (
            self._discussion_turns_left[
                self._discussion_order[self._discussion_player_index]
            ]
            <= 0
        ):
          self._discussion_player_index = (
              self._discussion_player_index + 1
          ) % len(self._discussion_order)

        next_player = self._discussion_order[self._discussion_player_index]
        logging.info("Next acting player in DISCUSSION: %s", next_player)
        return next_player

      if self._phase == Phase.REFLECTION:
        if self._reflection_player_index < len(self._player_names):
          return self._player_names[self._reflection_player_index]
        self._transition_to_town_square()
        continue

      if self._phase == Phase.TOWN_SQUARE:
        # Find next alive player who hasn't nominated today and hasn't passed
        for name in self._player_names:
          if name not in self._nominators_passed:
            p = self._grimoire.players[name]
            if (
                p.status == game_tracker.PlayerStatus.ALIVE
                and not p.has_nominated_today
            ):
              return name
        self._resolve_execution_phase()
        continue

      if self._phase == Phase.DEFENSE:
        return (
            self._current_nomination[1]
            if self._current_nomination
            else self._player_names[0]
        )

      if self._phase == Phase.VOTING:
        while self._voter_index < len(self._player_names):
          voter_name = self._player_names[self._voter_index]
          voter_p = self._grimoire.players[voter_name]
          if (
              voter_p.status == game_tracker.PlayerStatus.ALIVE
              or not voter_p.has_spent_dead_vote
          ):
            return voter_name
          self._voter_index += 1
        # Safety fallback if stuck in Voting or all voters processed
        self._phase = Phase.TOWN_SQUARE
        self._current_nomination = None
        continue

      if self._phase == Phase.EXECUTION:
        self._start_night(self._day_number + 1)
        continue

      if self._phase == Phase.GAME_OVER:
        return self._player_names[0]

      return self._player_names[0]

  def _handle_next_action_spec(self) -> str:
    self._check_and_notify_new_deaths()
    acting_player = self._handle_next_acting()
    self._current_expected_player = acting_player

    if self._phase == Phase.SETUP:
      evil_intel = ""
      if (
          self._grimoire.get_player(acting_player).alignment
          == game_tracker.Alignment.EVIL
          and not self._grimoire.script.is_compact_game
      ):
        evil_intel = self._grimoire.get_evil_intelligence()
      intro = setup_utils.generate_player_intro(
          self._grimoire.get_player(acting_player).perceived_role,
          self._grimoire.script,
          len(self._player_names),
          evil_intel,
      )
      return json.dumps(
          entity_lib.ActionSpec(
              call_to_action=f"[{acting_player}] {intro} (Respond with 'Ack')",
              output_type=entity_lib.OutputType.FREE,
          ).to_dict()
      )

    if self._phase == Phase.ROLE_REFLECTION:
      role = self._grimoire.get_player(acting_player).perceived_role
      spec = entity_lib.ActionSpec(
          call_to_action=(
              f" You are the {role}. "
              " Reflect on your role and initial strategy. "
              " You cannot pass your turn. You must respond "
              " with your reflection on your role."
          ),
          output_type=entity_lib.OutputType.FREE,
      )
      return json.dumps(spec.to_dict())

    p_state = self._grimoire.players[acting_player]
    if p_state.status == game_tracker.PlayerStatus.DEAD:
      if p_state.has_spent_dead_vote:
        dead_vote_status = "SPENT (cannot vote again)"
      else:
        dead_vote_status = (
            "AVAILABLE (you can use your dead vote once, so save it for when"
            " it counts! E.g., vote to execute the opposing team)"
        )
      role_prefix = (
          f"🚨 YOU ARE DEAD! You are the DEAD {p_state.perceived_role}.\n- You"
          " have been killed or executed.\n- You CANNOT use your character's"
          f" active ability anymore.\n- Your dead vote is:"
          f" {dead_vote_status}.\n-"
          " Focus on analyzing others' claims to solve the game and vote"
          " strategically!\n\n"
      )
    else:
      ability_note = ""
      if p_state.has_used_single_use_ability:
        ability_note = " (Your single-use ability has already been used)"
      role_prefix = (
          "💚 YOU ARE ALIVE! You are the"
          f" {p_state.perceived_role}{ability_note}.\n\n"
      )

    if self._phase == Phase.NIGHT:
      role = self._role_instances[acting_player]
      acting_player_state = self._grimoire.players[acting_player]
      acting_player_state.woke_up_tonight = True

      # Define the strategic guide once to avoid redundancy
      town_summary = self._build_town_summary()
      strategic_guardrail = (
          f"\n\n**TOWN STATUS:**\n{town_summary}\n\n**Strategic Guidance:**\n-"
          " **Target LIVING:** Drives the game forward by removing threats or"
          " gathering fresh evidence.\n- **Target DEAD:** No mechanical"
          " effect, but useful for:\n  - *Bluffing:* E.g., faking Guardian"
          " saves or Soldier passives (otherwise, it just wastes a night"
          " action like a Demon kill).\n  - *Role Interpretation:* Knowing"
          " alive/dead status is crucial always, especially for distance-based"
          " roles like Empath (who skip dead neighbors).\n- **Avoid Loops:**"
          " Do not repeatedly target the same player; rotate to gather new"
          " info or eliminate new threats.\n- **Goal-Oriented:** Choose targets"
          " whose status-change (killing) or check (investigating) best helps"
          " your team's objective."
      )

      # 1. Handle BMR: Lunatic (Thinks they are the Demon)
      if acting_player_state.role == "Lunatic":
        demon_role = next(
            (
                self._role_instances[p.name]
                for p in self._grimoire.players.values()
                if self._grimoire.is_type(p.name, "Demon")
                and p.role != "Lunatic"
            ),
            None,
        )
        if demon_role:
          # Get the Demon's standard prompt ("Please select a player to kill")
          demon_prompt = demon_role.resolve_night_action(
              "", self._grimoire, self._day_number
          )
          spec = entity_lib.ActionSpec(
              call_to_action=(
                  f"{role_prefix}You are the Demon."
                  f" {demon_prompt}{strategic_guardrail}"
              ),
              # Use FREE to force a selection
              output_type=entity_lib.OutputType.FREE,
          )
          return json.dumps(spec.to_dict())

      # 2. Handle standard role actions from the cache
      action_result = self._night_result_cache.get(acting_player)
      if action_result:
        # 2a. Handle Error Feedback
        if action_result.startswith("Invalid") or action_result.startswith(
            "ERROR"
        ):
          prompt = role.resolve_night_action(
              "", self._grimoire, self._day_number
          )
          call_to_action = f"ERROR: {action_result} {role_prefix}"
          if action_result not in prompt:
            call_to_action += f" {prompt}"

          spec = entity_lib.ActionSpec(
              call_to_action=call_to_action + strategic_guardrail,
              output_type=entity_lib.OutputType.FREE,
          )
          return json.dumps(spec.to_dict())

        # 2b. Handle Active Role Prompts (e.g., "Please select...")
        elif action_result.startswith("Please"):
          spec = entity_lib.ActionSpec(
              call_to_action=(
                  f"{role_prefix}{action_result}{strategic_guardrail}"
              ),
              output_type=entity_lib.OutputType.FREE,
          )
          return json.dumps(spec.to_dict())

        # 2c. Handle Passive Information Roles & Night 1 Wake
        else:
          spec = entity_lib.ActionSpec(
              call_to_action=(
                  f"{role_prefix}{action_result} (Reflect on your role,"
                  " alignment, and initial strategy)"
              ),
              output_type=entity_lib.OutputType.FREE,
          )
          return json.dumps(spec.to_dict())

      # Default: Skip if no action is needed
      return json.dumps(entity_lib.skip_this_step_action_spec().to_dict())

    if self._phase == Phase.DISCUSSION:
      p_state = self._grimoire.get_player(acting_player)
      is_alive_executioner = (
          p_state.status == game_tracker.PlayerStatus.ALIVE
          and p_state.role == "Executioner"
      )
      reply_target = self._outstanding_two_way_whispers.get(acting_player)

      options_list = [
          "broadcast [message]",
          "whisper to [player]: [message]",
          "nominate [player]",
      ]
      if reply_target:
        options_list.append(f"reply to {reply_target}: [message]")
      if is_alive_executioner:
        options_list.append("shoot [player]")
      if self._player_can_pass:
        options_list.append("pass because [reason]")

      options_str = ", ".join(options_list)

      reply_tip = ""
      if reply_target:
        reply_tip = (
            "\n\n🔔 **UNANSWERED WHISPER ALERT:** You have a pending TWO-WAY"
            f" private whisper from **{reply_target}** that expects a"
            f" reply!\nPlease choose `reply to {reply_target}: [message]` as"
            " your action to whisper back. Formulate your response"
            " strategically:\n  - *Truthful Sharing:* If you believe they are"
            " on your team and you trust them, coordinate and share your real"
            " role/findings.\n  - *Strategic Bluffing:* If you are Evil or"
            " suspect they are Evil, craft a highly convincing bluff matching"
            " your claimed identity."
        )

      town_summary = self._build_town_summary()
      cta = (
          f"{role_prefix}{town_summary}\n\nConsider the claims and accusations"
          " made so far.\n**Strategic Day Actions Guide:**\n- **`broadcast"
          " [message]`**: Share claims, verified findings, or public theories"
          " with the town.\n- **`whisper to [player]: [message]`**: Initiate"
          " secret 1-on-1 coordination once you have complete confidence in a"
          " player's alignment. (If Good: Verify powerful roles and claims"
          " without drawing the Demon's attention to yourself. If Evil:"
          " Coordinate bluffs).\n- **`nominate [player]`**: Put a suspect on"
          " trial for execution.\n"
      )
      if is_alive_executioner:
        cta += (
            "- **`shoot [player]` (Executioner)**: Publicly shoot a suspected"
            " Demon to instantly win.\n"
        )
      if self._player_can_pass:
        cta += (
            "- **`pass because [reason]`**: Pass your discussion turn if you"
            " have no active move.\n"
        )

      cta += f"\nChoose action: {options_str}.{reply_tip}"
      spec = entity_lib.ActionSpec(
          call_to_action=cta,
          output_type=entity_lib.OutputType.FREE,
      )
      return json.dumps(spec.to_dict())

    if self._phase == Phase.REFLECTION:
      spec = entity_lib.ActionSpec(
          call_to_action=(
              f"{role_prefix}\nMANDATORY STRATEGIC REFLECTION DIRECTIVE:\nThis"
              " is your dedicated private reflection and internal journal"
              " phase. You CANNOT pass or return a short phrase.\nCRITICAL"
              " NOTICE: You are recording internal thoughts in your private"
              " journal. You are NOT communicating with anyone.\nDo NOT"
              " include any whisper tags (`[ONE-WAY]`, `[TWO-WAY]`) or address"
              " other players directly. Do NOT output game actions"
              " (nominate/protect).\nInstead, you MUST write an in-depth,"
              " multi-paragraph strategic reflection detailing:\n1. Event &"
              " Information Synthesis: What crucial claims, whispers, and"
              " mechanical outcomes emerged today?\n2. Ledger & GM"
              " Cross-Verification: Based on your Cumulative Game Ledger and"
              " private GM knowledge, who are your primary suspects or trusted"
              " allies? Are there any poisoned or drunk players skewing"
              " information?\n3. Concrete Action Plan: What is your exact plan"
              " for upcoming nominations, voting, night actions, and private"
              " communication?\nProvide your comprehensive strategic"
              " reflection below:"
          ),
          output_type=entity_lib.OutputType.FREE,
      )
      return json.dumps(spec.to_dict())

    if self._phase == Phase.TOWN_SQUARE:
      options = []
      already_nominated = []
      for p_name, p_state in self._grimoire.players.items():
        if not p_state.has_been_nominated_today:
          status_lbl = (
              "ALIVE"
              if p_state.status == game_tracker.PlayerStatus.ALIVE
              else "DEAD"
          )
          options.append(f"nominate {p_name} ({status_lbl})")
        else:
          already_nominated.append(p_name)

      if self._player_can_pass or not options:
        options.append("pass")

      # Calculate the voting threshold for better strategic decision-making
      alive_count = len([
          p
          for p in self._grimoire.players.values()
          if p.status == game_tracker.PlayerStatus.ALIVE
      ])
      threshold = math.ceil(alive_count / 2)

      # Build the strategic context
      block_info = ""
      if self._players_on_the_block:
        block_info = (
            f" Currently on the block: {', '.join(self._players_on_the_block)}"
            f" ({self._highest_vote_count} votes)."
        )

      cta = (
          f"{role_prefix}It is your turn to nominate. **Rules &"
          " Guardrails:**\n1. Do *NOT* nominate yourself.\n2. Do *NOT*"
          " repeatedly nominate the same player from previous days to avoid"
          f" game loops.\n3. {block_info}\n4. Execution today requires at least"
          f" **{threshold}** votes.\n\n"
      )

      if already_nominated:
        cta += (
            "Already nominated today (invalid targets):"
            f" {', '.join(already_nominated)}.\n\n"
        )

      cta += (
          "**Strategic Tip on Nominating:**\n- **Nominating a LIVING Player:**"
          " The primary way to execute the Demon (Good team) or to"
          " frame/eliminate an innocent Townsfolk (Evil team).\n- **Nominating"
          " a DEAD Player:** A valuable tactical option. Executing a dead"
          " player satisfies the requirement of an execution without causing"
          " anyone to die.\n  - *Good Team:* If suspects are unclear, or if you"
          " want to protect key alive allies or the Saint, nominate a dead"
          " player to satisfy execution rules (like under a Vortox) or safely"
          " end the day.\n  - *Evil Team:* Use this to waste the town's daily"
          " execution (keeping your Demon or minions alive) or to blend in as a"
          " cautious Good player.\nBefore nominating, verify the alive/dead"
          " status of the player in the options list. Choose the target that"
          " best advances your team's win condition today.\n\nThink carefully:"
          " Which execution benefits your team's goal, and who do you have the"
          " most evidence against?"
      )

      spec = entity_lib.ActionSpec(
          call_to_action=cta,
          output_type=entity_lib.OutputType.CHOICE,
          options=tuple(options),
      )
      return json.dumps(spec.to_dict())

    if self._phase == Phase.DEFENSE:
      spec = entity_lib.ActionSpec(
          call_to_action=role_prefix
          + "Provide your defense speech to the town.",
          output_type=entity_lib.OutputType.FREE,
      )
      return json.dumps(spec.to_dict())

    if self._phase == Phase.VOTING:
      nominee = (
          self._current_nomination[1] if self._current_nomination else "Unknown"
      )
      nominator = (
          self._current_nomination[0] if self._current_nomination else "Unknown"
      )
      # Build strategic voting prompt
      alive_count = len([
          p
          for p in self._grimoire.players.values()
          if p.status == game_tracker.PlayerStatus.ALIVE
      ])
      threshold = math.ceil(alive_count / 2)
      cta = (
          f"{role_prefix}{nominator} has nominated {nominee} for"
          f" execution.\nExecution requires at least **{threshold}** votes"
          f" ({alive_count} alive players).\n**Strategic Voting"
          f" Heuristics:**\n- **If you are {nominator} (The Nominator):** You"
          " initiated this nomination, but you are free to vote 'yes' or 'no'."
          " Vote 'yes' if you still want them executed, or vote 'no' if you"
          " were testing the town's reaction, changed your mind after their"
          " defense speech, or want to save your vote.\n- **If you are"
          " {nominee} (The Nominee):** You are on trial. Voting 'yes'"
          " contributes to your own execution.\n- **If Good:** Vote 'yes' only"
          " if evidence supports executing this suspect. Track who votes 'yes'"
          " and 'no'—voting patterns reveal evil alliances saving each"
          " other.\n- **If Evil:** Vote 'yes' to eliminate Good players"
          " without exposing yourself, or vote 'no' to save a teammate or to"
          " not get exposed."
      )
      spec = entity_lib.ActionSpec(
          call_to_action=cta,
          output_type=entity_lib.OutputType.CHOICE,
          options=("yes", "no"),
      )
      return json.dumps(spec.to_dict())

    return json.dumps(entity_lib.skip_this_step_action_spec().to_dict())

  def _build_town_summary(self) -> str:
    """Builds a structured summary of the current game state."""
    lines = []
    lines.append(f"--- TOWN STATUS (Day {self._day_number}) ---")
    living = []
    dead = []
    for name in self._player_names:
      if name not in self._grimoire.players:
        continue
      p = self._grimoire.players[name]
      if p.status == game_tracker.PlayerStatus.ALIVE:
        living.append(name)
      else:
        dead_vote_note = (
            " (dead vote spent)"
            if p.has_spent_dead_vote
            else " (dead vote available)"
        )
        dead.append(f"{name}{dead_vote_note}")
    lines.append(f"Living ({len(living)}): {', '.join(living)}")
    if dead:
      lines.append(f"Dead: {', '.join(dead)}")
    else:
      lines.append("Dead: None")

    if self._current_nomination:
      nominator, nominee = self._current_nomination
      lines.append(f"CURRENT NOMINATION: {nominator} has nominated {nominee}")
    if self._players_on_the_block:
      lines.append(
          f"On the block: {', '.join(self._players_on_the_block)}"
          f" ({self._highest_vote_count} votes)"
      )
    lines.append(f"Phase: {self._phase.value}")
    lines.append("--- END TOWN STATUS ---")
    return "\n".join(lines)

  def _handle_observation(self, action_spec: entity_lib.ActionSpec) -> str:
    """Returns queued observations for the player being queried."""
    player_name = ""
    for name in self._player_names:
      if name in action_spec.call_to_action:
        player_name = name
        break
    if not player_name:
      match = re.search(r"'(.*?)'", action_spec.call_to_action)
      if match:
        player_name = match.group(1)

    if player_name and player_name in self._observation_queue:
      queued = self._observation_queue[player_name]
      self._observation_queue[player_name] = []
      return "\n".join(queued)

    return ""

  def _handle_resolve(self, action_spec: entity_lib.ActionSpec | None) -> str:
    event = self._last_event
    acting_player = event.split(":", 1)[0].strip() if ":" in event else ""
    action_content = event.split(":", 1)[1].strip() if ":" in event else event

    if not acting_player and self._current_expected_player:
      acting_player = self._current_expected_player

    if self._phase == Phase.SETUP:
      # Use the known index rather than parsing from the event string,
      # since some models respond without a "Player_X:" prefix.
      setup_player = self._player_names[self._setup_player_index]
      self._setup_player_index += 1
      self._log_priv(setup_player, f"Setup processed for {setup_player}.")
      return f"Setup processed for {setup_player}."

    if self._phase == Phase.ROLE_REFLECTION:
      # Use the known index rather than parsing from the event string,
      # since some models respond without a "Player_X:" prefix.
      reflection_player = self._player_names[self._reflection_player_index]
      reflection_content = action_content if action_content else event
      self._log_priv(reflection_player, f"Reflection: {reflection_content}")
      self._reflection_player_index += 1
      response_message = f"Reflection noted for {reflection_player}."
      self._log_clean(response_message)
      return response_message

    if self._phase == Phase.NIGHT:
      res = self._resolve_night_action(acting_player, action_content)
      return res

    if self._phase == Phase.DISCUSSION:
      res = self._resolve_discussion(acting_player, action_content)
      if res.startswith("Invalid"):
        self._discussion_retries[acting_player] = (
            self._discussion_retries.get(acting_player, 0) + 1
        )
        if self._discussion_retries[acting_player] >= 3:
          self._discussion_retries[acting_player] = 0
          # Force pass: consume turn and advance index
          if acting_player in self._discussion_turns_left:
            self._discussion_turns_left[acting_player] -= 1
          self._discussion_player_index = (
              self._discussion_player_index + 1
          ) % len(self._discussion_order)
          self._log_pub(
              f"Forced pass for {acting_player} after 3 invalid attempts."
          )
          return f"{res} Max retries reached, forcing pass."
      else:
        self._discussion_retries[acting_player] = 0  # Reset on success
      return res

    if self._phase == Phase.REFLECTION:
      return self._resolve_reflection(acting_player, action_content)

    if self._phase == Phase.TOWN_SQUARE:
      return self._resolve_town_square(acting_player, action_content)

    if self._phase == Phase.DEFENSE:
      self._log_pub(f"{acting_player} gives their defense: {action_content}")
      self._phase = Phase.VOTING
      self._voter_index = 0
      self._votes_for_current_nominee = 0
      nominee = (
          self._current_nomination[1] if self._current_nomination else "Unknown"
      )
      return f"Defense complete. Voting begins for {nominee}."

    if self._phase == Phase.VOTING:
      return self._resolve_voting(acting_player, action_content)

    return f"Storyteller acknowledged: {event}"

  def _start_night(self, day_num: int) -> str:
    self._phase = Phase.NIGHT
    self._day_number = day_num
    self._grimoire.reset_daily_flags()
    self._grimoire.reset_dusk_flags()
    self._logging_in_progress = False
    self._players_on_the_block = []
    self._highest_vote_count = 0
    self._nominators_passed = set()
    self._current_nomination = None

    for p in self._grimoire.players.values():
      p.woke_up_tonight = False

    self._night_result_cache = {}
    self._night_actors = self._get_night_actors_for_day(day_num)
    self._night_actor_index = 0
    return f"Night {self._day_number} begins."

  def _get_night_actors_for_day(self, day_num: int) -> list[str]:
    # Wakes everyone who has a night action, sorted by priority
    actors = []
    for name in self._player_names:
      p = self._grimoire.players[name]
      if p.status == game_tracker.PlayerStatus.ALIVE:
        role = self._role_instances[name]

        # Information roles usually need the entire player pool to choose from.
        # Active roles will check if targets are empty and return "Please..."
        targets = []
        if role.role_name in [
            "Witness",
            "Researcher",
            "Investigator",
            "Matchmaker",
        ]:
          targets = self._player_names

        # Evaluate the night action to see if the role wakes
        # We use an empty action string to signify "initial wake check"
        res = role.resolve_night_action(
            "", self._grimoire, day_num, players=targets
        )

        if res:
          # BMR: Godfather wake condition
          if p.role == "Godfather":
            outsider_died = any(
                self._grimoire.players[d].role
                in self._grimoire.script.outsiders
                for d in self._grimoire.deaths_today
            )
            if not outsider_died:
              continue

          actors.append((name, role.night_priority))
          if not res.startswith("Please") and not res.startswith("Invalid"):
            self._log_clean(f"[NIGHT INFO] {name}: {res}")
          self._night_result_cache[name] = res
        elif day_num == 1:
          # On Night 1, all players get woken to receive their role & rules
          # context.
          actors.append((name, 999))
          intro_msg = (
              f"Night 1 has fallen. You are the {p.perceived_role}. "
              "Review your role, alignment, and rules context."
          )
          self._night_result_cache[name] = intro_msg
          self._log_clean(f"[NIGHT INFO] {name}: {intro_msg}")

    # Sort actors by priority
    actors.sort(key=lambda x: x[1])
    return [name for name, priority in actors]

  def _resolve_night_action(
      self, acting_player: str, action_content: str
  ) -> str:
    # Refresh misregistrations before resolving any ability
    self._grimoire.refresh_misregistrations()

    if not acting_player or acting_player not in self._role_instances:
      # If we can't identify the actor, skip to the next one to avoid hanging.
      self._night_actor_index += 1
      return f"Unidentified player '{acting_player}', skipping."

    role = self._role_instances[acting_player]

    # Parsing Fix: Strip acting player's name and role name from the prefix
    # multiple times. Players often respond with "Player_1: Player_2, Player_3"
    # or "Player_1: Ack" or even "Player_1: Seer: Player_2, Player_3".
    cleaned_content = action_content
    prefixes_to_strip = [
        acting_player.lower(),
        role.role_name.lower(),
        role.role_name.lower().replace(" ", ""),
    ]

    changed = True
    while changed:
      changed = False
      for prefix in prefixes_to_strip:
        if cleaned_content.lower().startswith(prefix):
          cleaned_content = cleaned_content[len(prefix) :].strip()
          if cleaned_content.startswith(":"):
            cleaned_content = cleaned_content[1:].strip()
          changed = True

    # Check if this role requires an active target selection prompt
    # ("Please...")
    role_check = role.resolve_night_action(
        "", self._grimoire, self._day_number, players=self._player_names
    )
    if not role_check.startswith("Please"):
      # It's an info or passive wake (not an active target selection)
      if not self._grimoire.players[acting_player].woke_up_tonight:
        self._log_priv(acting_player, f"Night reflection: {cleaned_content}")
        self._grimoire.players[acting_player].woke_up_tonight = True
      self._night_retry_count = 0
      self._night_actor_index += 1
      logging.info(
          "Advanced night actor index to %d/%d",
          self._night_actor_index,
          len(self._night_actors),
      )
      if self._night_actor_index >= len(self._night_actors):
        logging.info("All night actors processed, calling _transition_to_day")
        return self._transition_to_day()
      cached_res = self._night_result_cache.get(acting_player)
      return f"{acting_player} resolves: {cached_res or role_check}"

    # Find players mentioned in the action content to pass as 'players'.
    # We use regex to find whole words that match player names.
    targets = []
    cleaned_content_search = cleaned_content.replace("_", " ")
    for p_name in self._player_names:
      p_name_search = p_name.replace("_", " ")
      if re.search(
          rf"\b{re.escape(p_name_search)}\b",
          cleaned_content_search,
          re.IGNORECASE,
      ):
        targets.append(p_name)
    self._log_clean(
        f"DEBUG: Night action for {acting_player}. Targets: {targets}. Cleaned"
        f" content: '{cleaned_content}'"
    )
    logging.info(
        "Night action for %s. Content: '%s'. Stripped: '%s'. Targets: %s."
        " Player names: %s",
        acting_player,
        action_content,
        cleaned_content,
        targets,
        self._player_names,
    )

    # Information roles (Witness, Researcher, Investigator, Matchmaker)
    # usually need the entire player pool to choose from if targets are empty.
    if not targets and role.role_name in [
        "Witness",
        "Researcher",
        "Investigator",
        "Matchmaker",
    ]:
      targets = self._player_names

    info = role.resolve_night_action(
        cleaned_content, self._grimoire, self._day_number, players=targets
    )
    if info.startswith("Please") or info.startswith("Invalid"):
      self._night_retry_count += 1
      if self._night_retry_count >= 3:
        # Force progress after 3 failed attempts
        logging.error(
            "Player %s failed night action 3 times. Skipping.", acting_player
        )
        self._night_retry_count = 0
        self._night_actor_index += 1
        return f"Max retries reached for {acting_player}. Moving to next actor."

      self._night_result_cache[acting_player] = info
      return f"Invalid action: {info}"

    # Successful night action: cache the result so we can prompt for Ack
    self._night_result_cache[acting_player] = info
    self._grimoire.players[acting_player].woke_up_tonight = True

    self._log_priv(acting_player, info)
    self._night_retry_count = 0
    self._night_actor_index += 1
    logging.info(
        "Advanced night actor index to %d/%d",
        self._night_actor_index,
        len(self._night_actors),
    )
    if self._night_actor_index >= len(self._night_actors):
      logging.info("All night actors processed, calling _transition_to_day")
      return self._transition_to_day()
    return f"Night action for {acting_player} resolved. Result: {info}"

  def _transition_to_day(self) -> str:
    logging.info("Entering _transition_to_day")
    self._phase = Phase.DISCUSSION
    self._grimoire.reset_daily_flags()
    self._discussion_turns_left = {
        name: self._turns_per_player for name in self._player_names
    }
    # Randomize the discussion order each day.
    random.shuffle(self._discussion_order)
    self._discussion_player_index = 0
    deaths = self._grimoire.resolve_deaths()
    announcement = (
        f"Day {self._day_number}: Rising and shining! The following players"
        f" have died: {', '.join(deaths) if deaths else 'None'}"
    )
    starpass_happened = False
    for death in self._grimoire.last_deaths_info:
      if death["source"] == "Demon":
        victim = self._grimoire.players[death["name"]]
        if victim.role == "Demon":
          starpass_happened = True
          break

    if starpass_happened:
      minions = [
          n
          for n, p in self._grimoire.players.items()
          if p.status == game_tracker.PlayerStatus.ALIVE
          and self._grimoire.is_type(n, "Minion")
      ]
      if minions:
        new_demon_name = random.choice(minions)
        self._grimoire.players[new_demon_name].role = "Demon"
        self._role_instances[new_demon_name] = _get_role_instance("Demon")
        self._role_instances[new_demon_name].player_name = new_demon_name
        self._log_pub("The Demon has perished! A new Demon has risen.")

    # Final win check after all night events and morning deaths
    win_result = self._check_win_conditions()
    if win_result:
      self._phase = Phase.GAME_OVER

      win_str = (
          win_result
          if isinstance(win_result, str)
          else "Evil wins! Early win condition met."
      )
      self._grimoire.game_over_reason = win_str
      self._log_pub(win_str)
      announcement += f" {win_str}"

    self._log_pub(self._build_town_summary())
    self._log_pub(announcement)
    logging.info(
        "Leaving _transition_to_day with announcement: %s", announcement
    )
    return announcement

  def _resolve_discussion(self, acting_player: str, action_content: str) -> str:
    try:
      data = json.loads(action_content)
      action_key = data.get("action", "").lower()
    except (json.JSONDecodeError, AttributeError):
      # Fallback to simple string parsing if not valid JSON
      action_lower = action_content.lower()
      if DaylightAction.DAY_KILL.command_key in action_lower:
        action_key = DaylightAction.DAY_KILL.command_key
        parts = re.split(
            rf"\b{DaylightAction.DAY_KILL.command_key}\b",
            action_content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        target = parts[1].strip(": ").strip() if len(parts) > 1 else ""
        data = {"target": target}
      elif DaylightAction.BROADCAST.command_key in action_lower:
        action_key = DaylightAction.BROADCAST.command_key
        parts = re.split(
            rf"\b{DaylightAction.BROADCAST.command_key}\b",
            action_content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        msg = parts[1].strip(": ").strip() if len(parts) > 1 else ""
        data = {"message": msg}
      elif DaylightAction.REPLY.command_key in action_lower:
        action_key = DaylightAction.REPLY.command_key
        match = re.search(r"reply to (.*?):(.*)", action_content, re.IGNORECASE)
        if match:
          data = {
              "target": match.group(1).strip(),
              "message": match.group(2).strip(),
          }
        else:
          parts = re.split(
              r"\breply\s+(?:to\s+)?\b",
              action_content,
              maxsplit=1,
              flags=re.IGNORECASE,
          )
          target = parts[1].split(":")[0].strip() if len(parts) > 1 else ""
          msg = (
              parts[1].split(":")[1].strip()
              if len(parts) > 1 and ":" in parts[1]
              else ""
          )
          data = {"target": target, "message": msg}
      elif DaylightAction.WHISPER.command_key in action_lower:
        action_key = DaylightAction.WHISPER.command_key
        match = re.search(
            r"whisper to (.*?):(.*)", action_content, re.IGNORECASE
        )
        if match:
          data = {
              "target": match.group(1).strip(),
              "message": match.group(2).strip(),
          }
        else:
          parts = re.split(
              rf"\b{DaylightAction.WHISPER.command_key}\b",
              action_content,
              maxsplit=1,
              flags=re.IGNORECASE,
          )
          target = parts[1].split(":")[0].strip() if len(parts) > 1 else ""
          msg = (
              parts[1].split(":")[1].strip()
              if len(parts) > 1 and ":" in parts[1]
              else ""
          )
          data = {"target": target, "message": msg}
      elif DaylightAction.NOMINATE.command_key in action_lower:
        action_key = DaylightAction.NOMINATE.command_key
        parts = re.split(
            rf"\b{DaylightAction.NOMINATE.command_key}\b",
            action_content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        raw_target = parts[1].strip(": ").strip() if len(parts) > 1 else ""
        target = self._find_player(raw_target) or raw_target
        data = {"target": target}
      else:
        action_key = action_lower
        data = {"message": action_content, "target": ""}

    # Validation and Execution
    if action_key in (
        DaylightAction.WHISPER.command_key,
        DaylightAction.REPLY.command_key,
    ):
      target = data.get("target", "").strip()
      found_player = self._find_player(target)
      if not found_player:
        return "Invalid whisper"
      target = found_player

      msg = data.get("message", "").strip()
      whisper_msg = f"{acting_player} (whisper to {target}): {msg}"
      self._log_priv(target, whisper_msg)
      self._log_priv(acting_player, whisper_msg)
      self._log_pub(f"{acting_player} whispers to {target}.")

      if (
          action_key == DaylightAction.WHISPER.command_key
          and "[TWO-WAY]" in msg.upper()
      ):
        self._outstanding_two_way_whispers[target] = acting_player

      if self._outstanding_two_way_whispers.get(acting_player) == target:
        self._outstanding_two_way_whispers.pop(acting_player, None)

      if acting_player in self._discussion_turns_left:
        self._discussion_turns_left[acting_player] -= 1
      self._discussion_player_index = (self._discussion_player_index + 1) % len(
          self._discussion_order
      )
      return f"Whispered to {target}"

    if action_key == DaylightAction.NOMINATE.command_key:
      target = data.get("target", "").strip()
      if re.search(r"\band\b", action_content, re.IGNORECASE):
        return "Invalid nomination"
      if not target or target not in self._grimoire.players:
        return "Invalid nomination"

      if acting_player in self._discussion_turns_left:
        self._discussion_turns_left[acting_player] -= 1
      self._discussion_player_index = (self._discussion_player_index + 1) % len(
          self._discussion_order
      )
      return self._resolve_nomination_attempt(acting_player, target)

    if action_key == DaylightAction.DAY_KILL.command_key:
      target = data.get("target", "").strip()
      if re.search(r"\band\b", action_content, re.IGNORECASE):
        return "Invalid action"
      if not target or target.lower() not in [
          n.lower() for n in self._grimoire.players.keys()
      ]:
        return "Invalid action"

      if (
          self._grimoire.get_player(acting_player).status
          != game_tracker.PlayerStatus.ALIVE
      ):
        return f"Invalid action: {acting_player} is dead and cannot shoot."

      if acting_player in self._discussion_turns_left:
        self._discussion_turns_left[acting_player] -= 1

      self._discussion_player_index = (self._discussion_player_index + 1) % len(
          self._discussion_order
      )
      res = self._grimoire.resolve_executioner_shot(acting_player, target)
      self._log_pub(f"Executioner Shot: {res}")

      # Proactively check if the game is over after an Executioner shot
      win_result = self._check_win_conditions()
      if win_result:
        self._phase = Phase.GAME_OVER
        win_str = (
            win_result
            if isinstance(win_result, str)
            else "Good wins! The Demon has been slain."
        )
        self._grimoire.game_over_reason = win_str
        self._log_pub(win_str)
        return f"{res} {win_str}"

      return res

    if action_content.lower().startswith("pass"):
      if not self._player_can_pass:
        return "Invalid action"
      if " because " not in action_content.lower():
        return "Invalid pass"
      if acting_player in self._discussion_turns_left:
        self._discussion_turns_left[acting_player] -= 1
      self._discussion_player_index = (self._discussion_player_index + 1) % len(
          self._discussion_order
      )
      reason = action_content.lower().split(" because ", 1)[1]
      self._log_priv(
          acting_player, f"You passed for the following reason: {reason}"
      )
      self._log_pub(f"{acting_player} passes.")
      return f"Acknowledged turn for {acting_player}."

    if action_key == DaylightAction.BROADCAST.command_key:
      msg = data.get("message", "").strip()
      self._log_pub(f"{acting_player} broadcasts: {msg}")
      if acting_player in self._discussion_turns_left:
        self._discussion_turns_left[acting_player] -= 1
      self._discussion_player_index = (self._discussion_player_index + 1) % len(
          self._discussion_order
      )
      return msg

    # Fallback/Turn progression
    if acting_player in self._discussion_turns_left:
      self._discussion_turns_left[acting_player] -= 1

    if all(count <= 0 for count in self._discussion_turns_left.values()):
      return self._transition_to_reflection()

    self._discussion_player_index = (self._discussion_player_index + 1) % len(
        self._discussion_order
    )
    return f"Acknowledged turn for {acting_player}."

  def _process_nomination(self, acting_player: str, target: str) -> str:
    """Consolidated logic for handling Witch, Innocent, and Defense phases."""
    p_nom = self._grimoire.players[acting_player]
    witch_msg = ""

    # Check for Witch Curse status before resolving
    is_cursed = False
    if any(r.role_name == "Witch" for r in self._role_instances.values()):
      if getattr(p_nom, "is_witch_cursed", False):
        is_cursed = True

    # Resolve the Nomination mechanics (checks validity and Innocent trap)
    try:
      # This flips 'has_nominated_today' and 'has_been_nominated_today' flags.
      nom_result = self._grimoire.nominate_player(acting_player, target)

      # If the nomination was valid and the player was cursed, they die now.
      # Note: This is a kill, not an execution; nominations continue.
      if is_cursed:
        self._grimoire.kill_player(acting_player)
        witch_msg = f"{acting_player} nominated despite the curse and DIED! "
        self._log_pub(witch_msg)

      self._log_pub(nom_result)

      # Handle the Innocent execution (which ends the day)
      if "instantly executed" in nom_result:
        self._current_nomination = None
        res = (
            f"Nomination of {target} triggered {acting_player} being instantly"
            " executed! The day ends."
        )
        return f"{witch_msg}{res} {self._resolve_execution_phase()}"

      # The nomination counts: Proceed to Defense and Voting
      self._current_nomination = (acting_player, target)
      self._phase = Phase.DEFENSE
      return f"{witch_msg}{target} is nominated. Time for defense."

    except ValueError as e:
      # If the nomination was illegal (e.g. double nomination), the attempt
      # fails and the curse does not trigger.
      raise e

  def _resolve_nomination_attempt(self, acting_player: str, target: str) -> str:
    """Wrapper for consolidated nomination logic used during Discussion."""
    try:
      return self._process_nomination(acting_player, target)
    except ValueError as e:
      self._log_pub(str(e))
      return str(e)

  def _transition_to_reflection(self) -> str:
    self._phase = Phase.REFLECTION
    self._reflection_player_index = 0
    self._log_pub(
        "Discussion ends. Players will now reflect on the events privately."
    )
    return "Discussion has ended. Entering reflection phase."

  def _resolve_reflection(self, acting_player: str, action_content: str) -> str:
    self._log_priv(acting_player, f"Reflection: {action_content}")
    self._reflection_player_index += 1
    if self._reflection_player_index >= len(self._player_names):
      return self._transition_to_town_square()
    response_message = f"Reflection recorded for {acting_player}."
    self._log_clean(response_message)
    return response_message

  def _transition_to_town_square(self) -> str:
    self._phase = Phase.TOWN_SQUARE
    self._nominators_passed = set()
    self._town_square_retries = {}
    self._log_pub("Discussion ends. Gathering in Town Square for nominations.")
    return "Discussion time over. Entering Town Square phase."

  def _resolve_town_square(
      self, acting_player: str, action_content: str
  ) -> str:
    """Consolidated Town Square logic with single retry/completion checks."""
    action_lower = action_content.lower()
    try:
      if "pass" in action_lower:
        if not self._player_can_pass:
          raise ValueError("Invalid action: Passing is not allowed.")
        self._nominators_passed.add(acting_player)
        self._log_pub(f"{acting_player} passes their nomination opportunity.")
      elif "nominate" in action_lower:
        target_raw = action_content.split("nominate", 1)[1].strip(": ").strip()
        target = self._find_player(target_raw)
        if not target:
          raise ValueError(
              f"Player {target_raw} does not exist. Choose a valid player."
          )
        return self._process_nomination(acting_player, target)
    except ValueError as e:
      error_msg = str(e)
      self._log_pub(error_msg)
      self._town_square_retries[acting_player] = (
          self._town_square_retries.get(acting_player, 0) + 1
      )
      if self._town_square_retries[acting_player] >= 3:
        self._log_pub(f"{acting_player} failed to act correctly. Forcing pass.")
        self._nominators_passed.add(acting_player)
      else:
        return f"ERROR: {error_msg}. Please try again."

    all_alive = [
        n
        for n, p in self._grimoire.players.items()
        if p.status == game_tracker.PlayerStatus.ALIVE
    ]
    if all(
        name in self._nominators_passed
        or self._grimoire.players[name].has_nominated_today
        for name in all_alive
    ):
      return self._resolve_execution_phase()

    return f"{acting_player} done in Town Square."

  def _resolve_voting(self, acting_player: str, action_content: str) -> str:
    voter = self._grimoire.players[acting_player]
    vote = action_content.lower()
    can_vote = (
        voter.status == game_tracker.PlayerStatus.ALIVE
        or not voter.has_spent_dead_vote
    )
    if can_vote:
      if voter.status == game_tracker.PlayerStatus.DEAD and vote == "yes":
        voter.has_spent_dead_vote = True

      if vote == "yes":
        self._votes_this_nomination[acting_player] = True
        self._log_pub(f"{acting_player} votes YES.")
      else:
        self._votes_this_nomination[acting_player] = False
        self._log_pub(f"{acting_player} votes NO.")
    else:
      # Inform the agent privately that their vote was invalid
      self._votes_this_nomination[acting_player] = False
      self._log_priv(
          acting_player,
          "You have already spent your dead vote and cannot vote again.",
      )
      self._log_pub(
          f"{acting_player} attempts to vote, but has no votes remaining."
      )

    self._voter_index += 1
    if self._voter_index == len(self._player_names):
      nominee = (
          self._current_nomination[1] if self._current_nomination else "Unknown"
      )

      # Tally votes with Servant restriction
      self._votes_for_current_nominee = 0
      for name, voted_yes in self._votes_this_nomination.items():
        if voted_yes:
          voter_p = self._grimoire.players[name]
          if voter_p.role == "Servant":
            master_name = voter_p.servant_master
            if master_name and not self._votes_this_nomination.get(
                master_name, False
            ):
              self._log_pub(
                  f"{name}'s vote ({voter_p.role}) was not counted because"
                  f" their Master ({master_name}) did not vote YES."
              )
              continue
          self._votes_for_current_nominee += 1

      self._votes_this_nomination = {}
      alive_count = len([
          p
          for p in self._grimoire.players.values()
          if p.status == game_tracker.PlayerStatus.ALIVE
      ])
      threshold = math.ceil(alive_count / 2)
      if self._votes_for_current_nominee >= threshold:
        if self._votes_for_current_nominee > self._highest_vote_count:
          self._highest_vote_count = self._votes_for_current_nominee
          self._players_on_the_block = [nominee]
        elif self._votes_for_current_nominee == self._highest_vote_count:
          self._players_on_the_block.append(nominee)
      self._phase = Phase.TOWN_SQUARE
      self._current_nomination = None
      self._log_clean(
          "DEBUG: Voting Resolution:"
          f" Day: {self._day_number},"
          f" current nominee: {nominee},"
          f" votes: {self._votes_for_current_nominee},"
          f" highest vote count: {self._highest_vote_count}, players on the"
          f" block: {self._players_on_the_block}"
      )
      return (
          f"Voting complete for {nominee}. {nominee} received"
          f" {self._votes_for_current_nominee} votes."
      )
    return f"{acting_player} has voted."

  def _resolve_execution_phase(self) -> str:
    self._phase = Phase.EXECUTION
    result = "Town Square closes. "
    self._log_clean(
        "DEBUG: Execution Phase:"
        f" Day: {self._day_number},"
        f" highest vote count: {self._highest_vote_count}, players on the"
        f" block: {self._players_on_the_block}"
    )
    executed = None
    if len(self._players_on_the_block) == 1:
      executed = self._players_on_the_block[0]
      exec_msg = self._grimoire.execute_player(executed)
      result += exec_msg
    else:
      result += "No execution today."

    win_result = self._check_win_conditions(executed_today=executed)
    if win_result:
      self._phase = Phase.GAME_OVER
      win_str = (
          win_result
          if isinstance(win_result, str)
          else "Evil wins! Early win condition met."
      )
      self._grimoire.game_over_reason = win_str
      self._log_pub(result + " " + win_str)
      return result + " " + win_str

    final_result = result + " Execution phase complete."
    self._log_pub(final_result)
    self._log_pub(self._build_town_summary())
    return final_result

  def _check_win_conditions(
      self, executed_today: str | None = None
  ) -> str | bool | None:
    if executed_today:
      executed_p = self._grimoire.players.get(executed_today)
      if (
          executed_p
          and executed_p.role == "Saint"
          and not self._grimoire.is_impaired(executed_today)
      ):
        return "Evil wins! The Saint has been executed."

    alive_players = {
        n: p
        for n, p in self._grimoire.players.items()
        if p.status == game_tracker.PlayerStatus.ALIVE
    }
    demon_alive = any(
        self._grimoire.is_type(name, "Demon") for name in alive_players
    )
    if not demon_alive:
      apprentice = [
          n
          for n, p in self._grimoire.players.items()
          if p.status == game_tracker.PlayerStatus.ALIVE
          and p.role == "Apprentice"
      ]
      if len(alive_players) >= 5 and apprentice:
        self._grimoire.players[apprentice[0]].role = "Demon"
        return None
      return "Good wins! The Demon has been slain."
    if len(alive_players) <= 2:
      return "Evil wins! Only two players remain."

    if not executed_today and len(alive_players) == 3:
      has_sober_mayor = any(
          p.role == "Mayor"
          and p.status == game_tracker.PlayerStatus.ALIVE
          and not self._grimoire.is_impaired(p.name)
          for p in self._grimoire.players.values()
      )
      if has_sober_mayor:
        return (
            "Good wins! The Mayor has secured peace with no execution among 3"
            " living players."
        )

    if self.check_early_game_over():
      return True  # Identified No-Win Condition

    if self._grimoire.game_over_reason:
      return self._grimoire.game_over_reason

    return None

  def check_early_game_over(self) -> str | None:
    """Checks if Good has no mathematical possibility of winning/executing the Demon.

    Returns:
      A string describing the early win condition if met, or None.
    """
    alive_players = {
        n: p
        for n, p in self._grimoire.players.items()
        if p.status == game_tracker.PlayerStatus.ALIVE
    }
    demon_alive = any(
        self._grimoire.is_type(name, "Demon") for name in alive_players
    )
    if not demon_alive:
      return None

    alive_good = [
        n
        for n, p in alive_players.items()
        if p.alignment == game_tracker.Alignment.GOOD
    ]
    if not alive_good:
      return (
          "Evil wins! No living Good players remain to nominate or execute the"
          " Demon."
      )

    dead_good_with_dead_vote = [
        n
        for n, p in self._grimoire.players.items()
        if p.status == game_tracker.PlayerStatus.DEAD
        and p.alignment == game_tracker.Alignment.GOOD
        and not p.has_spent_dead_vote
    ]
    max_good_votes = len(alive_good) + len(dead_good_with_dead_vote)
    execution_threshold = math.ceil(len(alive_players) / 2)

    has_active_executioner = any(
        p.role == "Executioner"
        and p.status == game_tracker.PlayerStatus.ALIVE
        and not p.has_used_single_use_ability
        for p in self._grimoire.players.values()
    )
    has_active_mayor = len(alive_players) == 3 and any(
        p.role == "Mayor"
        and p.status == game_tracker.PlayerStatus.ALIVE
        and not self._grimoire.is_impaired(p.name)
        for p in self._grimoire.players.values()
    )

    if (
        max_good_votes < execution_threshold
        and not has_active_executioner
        and not has_active_mayor
    ):
      return "Evil wins! Good cannot achieve enough votes to execute the Demon."

    return None


# Aliases
Storyteller = GameMaster
