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

"""Island Game Master prefab for Concordia simulation.

Extends situated_in_time_and_place with island-specific locations and
a fixed-interval clock: 120-minute ticks, waking hours only (7 AM - 11 PM,
8 ticks per day).

Supports async conversations via AsyncConversation* components when used
with the asynchronous.Asynchronous engine.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import re
import threading
from typing import Any

from absl import logging
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import terminate as terminate_components
from concordia.document import interactive_document
from concordia.environment import engine as engine_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements as async_measurements_lib

from examples.concordia_island.prefabs import simulationist_instructions
from examples.concordia_island.sim import conversation as async_conv
from examples.concordia_island.sim import conversation_director as conv_director_lib
from examples.concordia_island.sim import economic_profile
from examples.concordia_island.sim import fiscal_configs
from examples.concordia_island.sim import fiscal_scheduler
from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import food_consumption
from examples.concordia_island.sim import locations as locations_lib
from examples.concordia_island.sim import social_scheduler as social_scheduler_lib
from examples.concordia_island.sim import temporal_nudge as temporal_nudge_lib
from examples.concordia_island.sim import time_based_next_gm as time_based_lib


class LocationAwareMakeObservation(
    gm_components.make_observation.MakeObservation,
):
  """MakeObservation that prepends location and timestamp to observations.

  After generating the observation via the parent class, looks up the
  agent's current location and the current simulation time, then prepends
  them as `// location: [time] observation` so the agent always knows
  where and when it is.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      locations_key: str = "locations",
      clock_key: str = "clock",
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      social_scheduler_key: str = (
          social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY
      ),
      observation_context: dict[str, str] | None = None,
      initial_player_locations: dict[str, str] | None = None,
      filtered_display_events_key: str = "",
      world_state: Any = None,
      enable_conversation_shards: bool = False,
      **kwargs,
  ):
    super().__init__(
        model=model,
        player_names=player_names,
        components=components,
        **kwargs,
    )
    self._locations_key = locations_key
    self._clock_key = clock_key
    self._async_conversation_key = async_conversation_key
    self._social_scheduler_key = social_scheduler_key
    self._enable_conversation_shards = enable_conversation_shards
    self._observation_context = (
        observation_context if observation_context is not None else {}
    )
    self._filtered_display_events_key = filtered_display_events_key
    self._world_state = world_state
    self._last_known_locations: dict[str, str] = dict(
        initial_player_locations or {}
    )

  def _get_entity_location(self, entity_name: str) -> str:
    try:
      locations = self.get_entity().get_component(self._locations_key)
      state = locations.get_state()
      entity_locations = state.get("entity_locations", {})
      gm_location = entity_locations.get(entity_name, "")
      if gm_location:
        self._last_known_locations[entity_name] = gm_location
        return gm_location
    except (AttributeError, KeyError):
      pass
    return self._last_known_locations.get(entity_name, "")

  def _get_current_time(self) -> str:
    try:
      clock = self.get_entity().get_component(
          self._clock_key,
          type_=action_spec_ignored.ActionSpecIgnored,
      )
      return clock.get_pre_act_value().strip()
    except (AttributeError, KeyError):
      return ""

  def _is_past_night(self, entity_name: str) -> bool:
    """Whether `entity_name` may receive island observations right now.

    Delegates to `FixedIntervalClock.is_agent_past_night`, which is always
    True without a nighttime GM and on the first day. Clocks without the
    method (older checkpoints, proxies) never hold anyone.

    Args:
      entity_name: The player the observation is being built for.

    Returns:
      True if the player has completed the nighttime phase for the current day.
    """
    try:
      clock = self.get_entity().get_component(self._clock_key)
    except (AttributeError, KeyError):
      return True
    is_past = getattr(clock, "is_agent_past_night", None)
    if is_past is None:
      return True
    return bool(is_past(entity_name))

  def _get_conversation_state(
      self,
  ) -> async_conv.AsyncConversationState | None:
    try:
      return self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
    except (AttributeError, KeyError):
      return None

  def _get_social_scheduler(
      self,
  ) -> social_scheduler_lib.SocialScheduler | None:
    try:
      comp = self.get_entity().get_component(
          self._social_scheduler_key,
          type_=social_scheduler_lib.SocialScheduler,
      )
      if isinstance(comp, social_scheduler_lib.SocialScheduler):
        return comp
      return None
    except (AttributeError, KeyError):
      return None

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type != entity_lib.OutputType.MAKE_OBSERVATION:
      return super().pre_act(action_spec)

    active_entity_name = self._get_active_entity_name_from_call_to_action(
        action_spec.call_to_action
    )

    # 0. Per-agent night gate. When the entity loop switches this agent to a
    # nighttime GM, the engine first flushes the island GM's observations
    # while the clock already reads 7:00 AM of the new day. Keep everything
    # queued until the agent is back from the night so that, in its memory,
    # the night precedes the morning (breakfast, waking up at home, ...).
    # No-op without a nighttime GM and on the first day.
    if active_entity_name and not self._is_past_night(active_entity_name):
      return ""

    # 1. Fire any scheduled social events for this tick (dates, meetups)
    # BEFORE generating observations. This creates the conversation and
    # sets the shared setup context, suppressing regular daytime observations.
    social_scheduler = self._get_social_scheduler()
    if social_scheduler is not None and hasattr(
        social_scheduler, "fire_pending_events_for_current_tick"
    ):
      social_scheduler.fire_pending_events_for_current_tick()

    # 2. Defensive 7:00 AM morning check: Guarantee all agents wake up at home
    current_time_str = self._get_current_time()
    if (
        current_time_str
        and "7:00 AM" in current_time_str
        and self._last_known_locations
    ):
      home = self._last_known_locations.get(active_entity_name)
      current_loc = self._get_entity_location(active_entity_name)
      if home and current_loc != home:
        try:
          loc_comp = self.get_entity().get_component(self._locations_key)
          if loc_comp is not None:
            loc_state = loc_comp.get_state()
            entity_locs = loc_state.get("entity_locations", {})
            entity_locs[active_entity_name] = home
            loc_comp.set_state(loc_state)
            self._last_known_locations[active_entity_name] = home
            logging.info(
                "LocationAwareMakeObservation: 7:00 AM reset %s to home %s",
                active_entity_name,
                home,
            )
        except (AttributeError, KeyError):
          pass

    # Check conversation state BEFORE draining observations.
    # If the agent is in a conversation, DO NOT drain — leave observations
    # unconsumed in the queue (and in the optional world state, if one is
    # configured) so they are delivered after the conversation ends. Draining
    # here and then discarding is irreversible: get_and_clear() empties the
    # in-memory list, and a world state's drain_observations() marks its
    # entries consumed.
    conv_state = self._get_conversation_state()
    in_conversation = conv_state is not None and conv_state.is_in_conversation(
        active_entity_name
    )

    if in_conversation:
      if self._enable_conversation_shards:
        # Dialogue observations are handled by the dedicated conversation shard.
        # Return empty to suppress the Island GM generating a parallel timeline.
        return ""
      else:
        assert conv_state is not None
        result = conv_state.get_dialogue_context(active_entity_name)
        if result:
          conv = conv_state.get_conversation_for(active_entity_name)
          # Use the conversation's stored location, not the dynamic lookup
          # which may have been corrupted by a stale planning resolve.
          location = conv.location if conv else ""
          current_time = self._get_current_time()
          prefix_parts = []
          if location:
            prefix_parts.append(location)
          if current_time:
            prefix_parts.append(f"[{current_time}]")
          if prefix_parts:
            result = f"// {' '.join(prefix_parts)}: {result}"
        self._logging_channel({
            "Key": "Observation (conversation)",
            "Summary": result[:100] if result else "",
            "Value": result,
        })
        return result

    # Not in a conversation — safe to drain queued observations.
    queued_events = self._queue.get_and_clear(active_entity_name)
    if self._world_state is not None:
      try:
        world_state_obs = self._world_state.drain_observations(
            active_entity_name
        )
        if world_state_obs:
          queued_events.extend(world_state_obs)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Failed to drain observations from world state for %s: %s",
            active_entity_name,
            e,
        )
    queued_str = ""

    if queued_events:
      location = self._get_entity_location(active_entity_name)
      current_time = self._get_current_time()
      prefix_parts = []
      if location:
        prefix_parts.append(location)
      if current_time:
        prefix_parts.append(f"[{current_time}]")
      tagged = []
      for event in queued_events:
        stripped = event.strip()
        if (
            stripped
            and not stripped.startswith("//")
            and not stripped.startswith("[journal]")
            and prefix_parts
        ):
          tagged.append(f"// {' '.join(prefix_parts)}: {stripped}")
        elif stripped:
          tagged.append(stripped)
      queued_str = "\n\n\n".join(tagged)

    if queued_str:
      self._logging_channel({
          "Key": "Observation (queued)",
          "Summary": queued_str[:100] if queued_str else "",
          "Value": queued_str,
      })

    self._observation_context["target"] = active_entity_name
    if self._filtered_display_events_key:
      filtered = self.get_entity().get_component(
          self._filtered_display_events_key
      )
      filtered.update()

    result = super().pre_act(action_spec)

    if result.strip():
      location = self._get_entity_location(active_entity_name)
      current_time = self._get_current_time()
      prefix_parts = []
      if location:
        prefix_parts.append(location)
      if current_time:
        prefix_parts.append(f"[{current_time}]")
      if prefix_parts:
        prefix = f"// {' '.join(prefix_parts)}: "
        events = result.split("\n\n\n")
        tagged_events = []
        for event in events:
          stripped = event.strip()
          if stripped and not stripped.startswith("//"):
            tagged_events.append(f"{prefix}{stripped}")
          elif stripped:
            tagged_events.append(stripped)
        result = "\n\n\n".join(tagged_events)

    # Prepend queued resolution from the prior tick so the agent first
    # learns what happened, then sees the fresh scene observation.
    if queued_str and result.strip():
      result = f"{queued_str}\n\n\n{result}"
    elif queued_str:
      result = queued_str

    return result


PUTATIVE_EVENT_TAG = "[putative_event]"
EVENT_TAG = "[event]"


class IslandEventResolution(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Resolves agent actions into events with location-aware observation.

  This is the SINGLE resolution component for the island GM. It handles
  both planning actions (isolated agents) and speech actions (agents in
  conversations). Having one resolver avoids the race condition where
  SwitchAct._resolve only uses __resolution__ and ignores other resolvers.
  """

  # Minimum turn count before checking for ending conversations.
  _TERMINATE_CHECK_MIN_TURN = 4
  _TERMINATE_NOTE = (
      "Any conversation that becomes repetitive always ends immediately. "
      "When participants have said goodbye, finished their exchange, or "
      "stated an intention to leave, the conversation is over."
  )

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      components: Sequence[str] = (),
      clock_key: str = "clock",
      locations_key: str = "locations",
      display_events_key: str = "display_events",
      next_acting_key: str = "__next_acting__",
      memory_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      make_observation_key: str = (
          gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      terminate_boring: bool = True,
      pre_act_label: str = "\nEvent Resolution",
      enable_conversation_shards: bool = False,
  ):
    super().__init__()
    self._model = model
    self._components = tuple(components)
    self._player_names = set(player_names)
    self._player_names_list = list(player_names)
    self._clock_key = clock_key
    self._locations_key = locations_key
    self._display_events_key = display_events_key
    self._next_acting_key = next_acting_key
    self._memory_key = memory_key
    self._make_observation_key = make_observation_key
    self._async_conversation_key = async_conversation_key
    self._terminate_boring = terminate_boring
    self._pre_act_label = pre_act_label
    self._enable_conversation_shards = enable_conversation_shards
    self._active_entity_name = None
    self._putative_action = None
    self._pending_events: dict[str, str] = {}

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f"{self.get_component_pre_act_label(key)}:\n"
        f"{self.get_named_component_pre_act_value(key)}"
    )

  def get_active_entity_name(self) -> str | None:
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    return self._putative_action

  def pre_observe(self, observation: str) -> str:
    if PUTATIVE_EVENT_TAG in observation:
      tag_end = observation.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
      raw = observation[tag_end:].strip()
      found = False
      for name in self._player_names:
        if re.search(rf"(?:^|\n){re.escape(name)}\b", raw):
          self._pending_events[name] = observation
          found = True
      if not found:
        self._pending_events["__unknown__"] = observation
    return ""

  def _get_recent_events_at_location(
      self, location: str, limit: int = 10
  ) -> str:
    try:
      mem = self.get_entity().get_component(
          self._memory_key, type_=memory_component.Memory
      )
      events = mem.scan(selector_fn=lambda x: EVENT_TAG in x)
      relevant = [
          e.split(EVENT_TAG)[-1].strip()
          for e in events
          if location.lower() in e.lower()
      ]
      return "\n".join(relevant[-limit:]) if relevant else ""
    except (AttributeError, KeyError):
      return ""

  def _extract_speaker_and_utterance(
      self, action_text: str
  ) -> tuple[str | None, str | None]:
    """Extract speaker name and utterance from action text."""
    text = action_text.strip()
    for name in self._player_names_list:
      if text.startswith(f"{name}:"):
        return name, text[len(f"{name}:") :].strip()
      if text.startswith(f"{name} "):
        remainder = text[len(f"{name} ") :]
        if remainder.startswith("--"):
          remainder = remainder[2:]
        elif remainder.startswith("said:"):
          remainder = remainder[5:]
        elif remainder.startswith("says:"):
          remainder = remainder[5:]
        return name, remainder.strip()
    return None, text

  def _has_exit_intent(self, text: str) -> bool:
    return bool(async_conv._EXIT_REGEX.search(text))  # pylint: disable=protected-access

  def _should_terminate_conversation(
      self, conv: async_conv.Conversation
  ) -> bool:
    """Ask the GM whether the conversation has ended or become repetitive.

    Follows the Dialogic Game Master pattern from third_party Concordia:
    Checks if participants have said goodbye / departed, or asks the GM LLM
    whether the conversation is finished.

    Args:
      conv: The conversation to check.

    Returns:
      True if the conversation should terminate.
    """
    if not self._terminate_boring:
      return False
    turn_count = len(conv.utterances)
    if turn_count < 2:
      return False

    _, last_text = conv.utterances[-1]
    has_exit = self._has_exit_intent(last_text)

    # If recent utterances have exit intent, check on every turn.
    # Otherwise check every other turn once turn_count >= min_turns.
    min_turn = getattr(
        conv, "terminate_check_min_turn", self._TERMINATE_CHECK_MIN_TURN
    )
    if not has_exit and (turn_count < min_turn or turn_count % 2 != 0):
      return False

    # Check if both participants in a 2-person conversation expressed exit
    # intent.
    if len(conv.participants) == 2 and len(conv.utterances) >= 2:
      p1_recent = any(
          self._has_exit_intent(txt)
          for spk, txt in conv.utterances[-4:]
          if spk == conv.participants[0]
      )
      p2_recent = any(
          self._has_exit_intent(txt)
          for spk, txt in conv.utterances[-4:]
          if spk == conv.participants[1]
      )
      if p1_recent and p2_recent:
        logging.info(
            "Both participants [%s] expressed exit intent; GM ending conv %d at"
            " turn %d",
            ", ".join(conv.participants),
            conv.conv_id,
            turn_count,
        )
        return True

    if self._model is None:
      return False

    recent = conv.utterances[-6:]
    transcript = "\n".join(f'{speaker}: "{text}"' for speaker, text in recent)

    doc = interactive_document.InteractiveDocument(self._model)
    doc.statement(f"Note: {self._TERMINATE_NOTE}")
    doc.statement(
        "The following is a conversation between"
        f' {", ".join(conv.participants)} at {conv.location}:'
    )
    doc.statement(transcript)
    terminate_options = [
        "Yes, the conversation is over",
        "No, the conversation will continue",
    ]
    choice = doc.multiple_choice_question(
        question="Is the conversation finished?",
        answers=terminate_options,
        randomize_choices=False,
    )

    should_end = choice == 0
    if should_end:
      logging.info(
          "GM terminated conversation %d [%s] at turn %d (dialogic GM check:"
          " finished)",
          conv.conv_id,
          ", ".join(conv.participants),
          turn_count,
      )
    return should_end

  @classmethod
  def _is_marketplace_json(cls, text: str) -> bool:
    """Return True if text contains a structured marketplace JSON action spec.

    Handles both raw JSON and name-prefixed forms like:
      '{"action": "bid", "good": "...", "price": 4.0, "qty": 3}'
      'Allison Lee {"action": "bid", "good": "...", "price": 4.0}'
    """
    if not text:
      return False
    stripped = text.strip().strip('"').strip("'").strip()
    # Look for any JSON-like substring containing a marketplace action.
    # The LLM often prefixes the agent's name before the JSON object.
    brace_start = stripped.find("{")
    if brace_start == -1:
      return False
    json_part = stripped[brace_start:]
    if not json_part.endswith("}"):
      # Try to find the closing brace
      brace_end = json_part.rfind("}")
      if brace_end == -1:
        return False
      json_part = json_part[: brace_end + 1]
    return any(
        f'"action": "{act}"' in json_part or f'"action":"{act}"' in json_part
        for act in ("bid", "save", "withdraw", "pass")
    )

  def _resolve_speech(self, agent: str, raw_action: str) -> str:
    """Resolve a speech action for an agent in a conversation."""
    _, utterance = self._extract_speaker_and_utterance(raw_action)

    # Reject marketplace JSON bids from being stored as conversation
    # utterances.  When the marketplace GM is active but the conversation
    # hasn't been ended yet, the agent may respond with a JSON bid spec
    # (e.g. {"action": "bid", "good": "...", "price": 4.0, "qty": 3}).
    # These must NOT be treated as speech.
    if utterance and self._is_marketplace_json(utterance):
      logging.info(
          "Rejected marketplace JSON as speech for %s — force-ending "
          "conversation: %s",
          agent,
          utterance[:80],
      )
      # Force-end the conversation to prevent an infinite loop: the agent
      # stays "in conversation" if we just return empty, so the engine
      # keeps asking for speech → LLM outputs JSON → rejected → repeat.
      try:
        conv_state = self.get_entity().get_component(
            self._async_conversation_key,
            type_=async_conv.AsyncConversationState,
        )
        conv = conv_state.get_conversation_for(agent)
        if conv is not None and conv.active:
          conv_state.end_conversation(conv.conv_id)
          logging.info(
              "Force-ended conversation %d due to JSON bid from %s",
              conv.conv_id,
              agent,
          )
      except (AttributeError, KeyError):
        pass
      return ""

    try:
      conv_state = self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
    except (AttributeError, KeyError):
      return ""

    if not conv_state.is_in_conversation(agent):
      return ""

    result = ""
    if utterance and conv_state.is_my_turn(agent):
      conv_state.add_utterance(agent, utterance)
      result = f'{agent} said: "{utterance}"'

      # Check if GM wants to end the conversation early
      conv = conv_state.get_conversation_for(agent)
      if conv is not None and conv.active:
        if self._should_terminate_conversation(conv):
          conv_state.end_conversation(conv.conv_id)
    elif utterance:
      result = f"{agent} is waiting for their turn to speak."

    self._logging_channel({
        "Key": self._pre_act_label,
        "Summary": result[:100] if result else "",
        "Value": result,
    })
    return result

  def _resolve_single_agent(
      self,
      agent: str,
      putative_action: str,
  ) -> str:
    """Resolve a single agent's action (speech or planning timeline)."""
    # Check if agent is in conversation
    conv_state = None
    try:
      conv_state = self.get_entity().get_component(
          self._async_conversation_key,
          type_=async_conv.AsyncConversationState,
      )
    except (AttributeError, KeyError):
      pass

    # --- Action Spec Tag based resolution ---
    try:
      tick_gated = self.get_entity().get_component(
          self._next_acting_key, type_=TickGatedNextActing
      )
      tag = tick_gated.get_and_clear_last_tag(agent)
      if tag == "speech":
        if self._enable_conversation_shards:
          logging.warning(
              "IslandEventResolution: unexpected speech action from %s while"
              " conversation sharding is enabled",
              agent,
          )
          return ""
        return self._resolve_speech(agent, f"{agent}: {putative_action}")
      elif tag == "action":
        if conv_state and conv_state.is_in_conversation(agent):
          logging.info(
              "IslandEventResolution: suppressing planning action resolution"
              " for %s who is in a conversation",
              agent,
          )
          return ""
        # Fall through to planning timeline resolution
        pass
    except (AttributeError, KeyError):
      # Fallback to conversation state check if tag not found (backward compat)
      if conv_state and conv_state.is_in_conversation(agent):
        if self._enable_conversation_shards:
          return ""
        return self._resolve_speech(agent, f"{agent}: {putative_action}")

    # --- Planning timeline resolution (isolated agent) ---
    location = ""
    try:
      loc_comp = self.get_entity().get_component(self._locations_key)
      loc_state = loc_comp.get_state()
      entity_locs = loc_state.get("entity_locations", {})
      location = entity_locs.get(agent, "")
    except (AttributeError, KeyError):
      pass

    clock_time = ""
    clock_comp = None
    try:
      clock_comp = self.get_entity().get_component(self._clock_key)
      clock_time = clock_comp.get_pre_act_value().strip()
    except (AttributeError, KeyError):
      pass

    recent_events = (
        self._get_recent_events_at_location(location) if location else ""
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    # Inject context from declared components (simulationist instructions,
    # location descriptions, recent events, etc.) — matching the idiomatic
    # Concordia pattern used by the default EventResolution.
    if self._components:
      component_states = "\n".join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f"{component_states}\n")
    if clock_time:
      prompt.statement(f"Current time: {clock_time}")
    if location:
      prompt.statement(f"{agent} is at {location}.")
    if recent_events:
      prompt.statement(f"Recent events at {location}:\n{recent_events}")
    prompt.statement(
        f"{agent} attempted the following action: {putative_action}"
    )

    # Derive dynamic tick interval and boundaries for resolution prompt.
    tick_interval = (
        getattr(clock_comp, "_tick_interval", None) if clock_comp else None
    )
    if not isinstance(tick_interval, datetime.timedelta):
      tick_interval = datetime.timedelta(hours=2)

    if clock_time:
      try:
        start_dt = fixed_clock.parse_sim_time(clock_time)
      except ValueError:
        start_dt = datetime.datetime(2026, 1, 1, 7, 0, 0)
    else:
      start_dt = datetime.datetime(2026, 1, 1, 7, 0, 0)

    end_dt = start_dt + tick_interval

    def _fmt_time(t: datetime.datetime) -> str:
      h = t.hour % 12 or 12
      return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"

    start_str = _fmt_time(start_dt)
    end_str = _fmt_time(end_dt)

    total_mins = max(1, int(tick_interval.total_seconds() // 60))
    if total_mins % 60 == 0:
      hours = total_mins // 60
      duration_phrase = f"{hours} hour{'s' if hours != 1 else ''}"
    else:
      duration_phrase = f"{total_mins} minutes"

    ex1_dt = start_dt + datetime.timedelta(
        minutes=min(15, max(1, total_mins // 4))
    )
    ex2_dt = start_dt + datetime.timedelta(
        minutes=min(45, max(2, total_mins * 3 // 4))
    )
    ex1_str = _fmt_time(ex1_dt)
    ex2_str = _fmt_time(ex2_dt)

    call_to_action = (
        f"The agent {agent} planned the following activities for the next"
        f" {duration_phrase} ({start_str} to {end_str}):\n{putative_action}\n\n"
        f"Generate a detailed timeline of what actually happened between"
        f" {start_str} and {end_str}. Resolve the plan with mundane, realistic"
        " outcomes, unexpected delays, small successes, or minor frustrations."
        " Break down the resolution chronologically into 2 to 4 distinct,"
        f" different timepoints within {start_str} to {end_str}. All"
        f" timestamps MUST be within the {start_str} to {end_str} window, for"
        f" example:\n{ex1_str}: {agent} did X.\n{ex2_str}: {agent} did Y.\n"
        f"Focus only on what {agent} experienced and observed. Do not describe"
        " other characters' voluntary actions. Do not express uncertainty."
    )

    event_statement = prompt.open_question(
        call_to_action,
        max_tokens=1500,
        terminators=(),
    )

    result = f"{self._pre_act_label}: {EVENT_TAG} {event_statement}\n"

    # Queue the resolved event as an observation for the acting agent.
    try:
      make_obs = self.get_entity().get_component(
          self._make_observation_key,
          type_=gm_components.make_observation.MakeObservation,
      )
      prefix_parts = []
      if location:
        prefix_parts.append(location)
      if clock_time:
        prefix_parts.append(f"[{clock_time}]")
      if prefix_parts:
        tagged_event = f"// {' '.join(prefix_parts)}: {event_statement}"
      else:
        tagged_event = event_statement
      make_obs.add_to_queue(agent, tagged_event)
    except (AttributeError, KeyError):
      logging.warning(
          "Could not queue resolved event for %s: MakeObservation "
          "component '%s' not found.",
          agent,
          self._make_observation_key,
      )

    return result

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    self._active_entity_name = None
    self._putative_action = None

    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      self._logging_channel({
          "Key": self._pre_act_label,
          "Summary": "",
          "Value": "",
          "Prompt": "",
          "Details": {"Observers prompt": ""},
      })
      return ""

    raw_action = None
    active_entity_name = None
    if self._pending_events:
      gm = self.get_entity()
      thread_id = threading.current_thread().ident
      if hasattr(self.get_entity(), "_capture_key_by_thread"):
        active_entity_name = self.get_entity()._capture_key_by_thread.get(  # pylint: disable=protected-access
            thread_id
        )
      if not active_entity_name:
        active_entity_name = getattr(gm, "_active_capture_key", None)

      selected = None
      if active_entity_name and active_entity_name in self._pending_events:
        selected = self._pending_events.pop(active_entity_name)

      if not selected and "__unknown__" in self._pending_events:
        selected = self._pending_events.pop("__unknown__")

      if not selected and self._pending_events:
        # Batch / Simultaneous fallback: take all unique pending event strings
        unique_events = list(dict.fromkeys(self._pending_events.values()))
        self._pending_events.clear()
        raw_parts = []
        for ev in unique_events:
          tag_pos = ev.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
          raw_parts.append(ev[tag_pos:].strip())
        raw_action = "\n".join(raw_parts)
      elif selected:
        tag_pos = selected.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
        raw_action = selected[tag_pos:].strip()

    if not raw_action or not raw_action.strip():
      self._logging_channel({
          "Key": self._pre_act_label,
          "Summary": "",
          "Value": "",
          "Prompt": "",
          "Details": {"Observers prompt": ""},
      })
      return ""

    # Parse all agent actions from raw_action
    agent_actions: list[tuple[str, str]] = []
    matches = []
    for name in self._player_names_list:
      for pattern in [
          rf"(?:^|\n){re.escape(name)}:\s*",
          rf"(?:^|\n){re.escape(name)}\s+--\s*",
          rf"(?:^|\n){re.escape(name)}\s+said:\s*",
          rf"(?:^|\n){re.escape(name)}\s+says:\s*",
      ]:
        for m in re.finditer(pattern, raw_action):
          start_pos = (
              m.start()
              if not raw_action[m.start() :].startswith("\n")
              else m.start() + 1
          )
          matches.append((start_pos, m.end(), name))

    if not matches:
      found_name = None
      for name in self._player_names_list:
        if raw_action.strip().startswith(name):
          found_name = name
          break
      if found_name:
        remainder = raw_action.strip()[len(found_name) :].lstrip(":").strip()
        agent_actions.append((found_name, remainder))
      elif active_entity_name:
        agent_actions.append((active_entity_name, raw_action.strip()))
      else:
        self._putative_action = raw_action.strip()
    else:
      matches.sort(key=lambda x: x[0])
      unique_matches = []
      seen_starts = set()
      for m in matches:
        if m[0] not in seen_starts:
          seen_starts.add(m[0])
          unique_matches.append(m)
      for i, (_, content_start, name) in enumerate(unique_matches):
        next_start = (
            unique_matches[i + 1][0]
            if i + 1 < len(unique_matches)
            else len(raw_action)
        )
        act_text = raw_action[content_start:next_start].strip()
        agent_actions.append((name, act_text))

    if not agent_actions:
      self._logging_channel({
          "Key": self._pre_act_label,
          "Summary": "",
          "Value": "",
          "Prompt": "",
          "Details": {"Observers prompt": ""},
      })
      return ""

    resolved_events = []
    for agent, act_text in agent_actions:
      self._active_entity_name = agent
      self._putative_action = act_text
      res = self._resolve_single_agent(agent, act_text)
      if res:
        resolved_events.append(res.strip())

    combined_result = "\n\n".join(resolved_events) + (
        "\n" if resolved_events else ""
    )
    self._logging_channel({
        "Key": self._pre_act_label,
        "Summary": combined_result[:100] if combined_result else "",
        "Value": combined_result,
    })
    return combined_result

  def get_state(self) -> entity_component.ComponentState:
    return {
        "_active_entity_name": self._active_entity_name,
        "_putative_action": self._putative_action,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._active_entity_name = state.get("_active_entity_name")
    self._putative_action = state.get("_putative_action")


class TickGatedNextActing(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Unified next-acting + action-spec with one-action-per-tick gating.

  Handles both NEXT_ACTING and NEXT_ACTION_SPEC output types:

  NEXT_ACTING — returns entity name if eligible, empty string if not:
    - Isolated + not acted this tick → eligible
    - In conversation + my turn     → eligible
    - In conversation + not my turn → NOT eligible (empty → engine sleeps)
    - Isolated + already acted      → NOT eligible (empty → engine sleeps)

  NEXT_ACTION_SPEC — returns the appropriate action spec:
    - In conversation → SPEECH spec
    - Not in conversation → DEFAULT action spec
  """

  def __init__(
      self,
      player_names: Sequence[str],
      clock_key: str = "clock",
      async_conversation_key: str = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
      memory_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      pre_act_label: str = "\nInitiative",
      enable_conversation_shards: bool = False,
      social_scheduler_key: str = (
          social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY
      ),
      has_nighttime_gm: bool = False,
  ):
    super().__init__()
    self._player_names = player_names
    self._clock_key = clock_key
    self._async_conversation_key = async_conversation_key
    self._memory_key = memory_key
    self._pre_act_label = pre_act_label
    self._enable_conversation_shards = enable_conversation_shards
    self._social_scheduler_key = social_scheduler_key
    self._has_nighttime_gm = has_nighttime_gm
    self._currently_active_player = None
    self._in_resolve_phase: dict[str, bool] = {}
    self._acted_this_tick: set[str] = set()
    self._last_seen_tick: int = -1
    self._last_seen_day: int = -1
    self._pending_events: dict[str, str] = {}
    self._agent_to_last_tag: dict[str, str] = {}

    self._lock = threading.Lock()

  def _get_clock(self) -> fixed_clock.FixedIntervalClock | None:
    try:
      return self.get_entity().get_component(
          self._clock_key, type_=fixed_clock.FixedIntervalClock
      )
    except (AttributeError, KeyError):
      return None

  def _get_social_scheduler(
      self,
  ) -> social_scheduler_lib.SocialScheduler | None:
    try:
      comp = self.get_entity().get_component(
          self._social_scheduler_key,
          type_=social_scheduler_lib.SocialScheduler,
      )
      if isinstance(comp, social_scheduler_lib.SocialScheduler):
        return comp
      return None
    except (AttributeError, KeyError):
      return None

  def _get_conversation_state(self) -> async_conv.AsyncConversationState | None:
    try:
      return self.get_entity().get_component(
          self._async_conversation_key, type_=async_conv.AsyncConversationState
      )
    except (AttributeError, KeyError):
      return None

  def get_and_clear_last_tag(self, agent_name: str) -> str | None:
    with self._lock:
      return self._agent_to_last_tag.pop(agent_name, None)

  def _extract_agent_name_from_call_to_action(
      self, call_to_action: str
  ) -> str | None:
    prefix = "In what action spec format should "
    suffix = " respond?"
    if prefix in call_to_action:
      start_idx = call_to_action.index(prefix) + len(prefix)
      remaining = call_to_action[start_idx:]
      if suffix in remaining:
        end_idx = remaining.index(suffix)
        return remaining[:end_idx]
    return None

  def pre_observe(self, observation: str) -> str:
    if PUTATIVE_EVENT_TAG in observation:
      tag_end = observation.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
      raw = observation[tag_end:].strip()
      for name in self._player_names:
        # Check if it starts with name or has name after a newline
        # (combined actions)
        if raw.startswith(name) or f"\n{name}" in raw:
          with self._lock:
            self._pending_events[name] = observation
          logging.info(
              "TickGated.pre_observe: registered %s for resolution", name
          )

    return ""

  def _get_resolving_entity(self) -> str | None:
    with self._lock:
      if not self._pending_events:
        return None
      thread_id = threading.current_thread().ident
      active_entity_name = None
      if hasattr(self.get_entity(), "_capture_key_by_thread"):
        active_entity_name = self.get_entity()._capture_key_by_thread.get(  # pylint: disable=protected-access
            thread_id
        )
      if not active_entity_name:
        active_entity_name = getattr(
            self.get_entity(), "_active_capture_key", None
        )
      if (
          active_entity_name
          and active_entity_name in self._player_names
          and active_entity_name in self._pending_events
      ):
        self._pending_events.pop(active_entity_name)
        return active_entity_name
    return None

  def _is_eligible(self, agent_name: str) -> bool:
    if self._social_scheduler_key:
      try:
        social_sched = self.get_entity().get_component(
            self._social_scheduler_key,
            type_=social_scheduler_lib.SocialScheduler,
        )
        if social_sched is not None and hasattr(
            social_sched, "is_scheduled_for_current_tick"
        ):
          if social_sched.is_scheduled_for_current_tick(agent_name):
            if self._enable_conversation_shards:
              return False
      except (AttributeError, KeyError):
        pass

    conv_state = self._get_conversation_state()
    if conv_state is not None and conv_state.is_in_conversation(agent_name):
      if self._enable_conversation_shards:
        # Agent is in a conversation handled by a dedicated conversation shard.
        return False
      return conv_state.is_my_turn(agent_name)

    clock = self._get_clock()
    if clock is not None:
      if self._has_nighttime_gm and hasattr(clock, "is_agent_past_night"):
        # Per-agent night gate. The entity loop asks for the next GM before
        # every turn, so a held agent is routed to the nighttime GM on its
        # next iteration (TimeBasedNextGM marks it "entered" on the clock) and
        # returns here only after the night (marked "finished" when the island
        # GM is asked again). This cannot deadlock the tick: the clock only
        # advances when every player has acted in the current tick, and no
        # player can be held before having had the chance to switch.
        if not clock.is_agent_past_night(agent_name):
          return False
      elif hasattr(clock, "is_nighttime_completed") and hasattr(
          clock, "_current_dt"
      ):
        current_day = getattr(clock, "_current_dt").day
        with self._lock:
          if self._last_seen_day == -1:
            self._last_seen_day = current_day
          elif current_day != self._last_seen_day:
            # Day changed — update tracking.  Do NOT block agents here:
            # TimeBasedNextGM handles the switch to the nighttime GM,
            # and blocking agents causes a tick-barrier deadlock (no agents
            # can act → tick never advances → marketplace never runs →
            # nighttime never completes).
            if clock.is_nighttime_completed(
                self._last_seen_day
            ) or clock.is_nighttime_completed(current_day):
              self._last_seen_day = current_day
            elif self._has_nighttime_gm:
              return False

      if hasattr(clock, "has_agent_acted"):
        if clock.has_agent_acted(agent_name):
          return False
      else:
        current_tick = clock.current_tick
        with self._lock:
          if current_tick != self._last_seen_tick:
            logging.info(
                "TickGated: tick changed %d -> %d, clearing acted set %s",
                self._last_seen_tick,
                current_tick,
                self._acted_this_tick,
            )
            self._acted_this_tick.clear()
            self._last_seen_tick = current_tick
          if agent_name in self._acted_this_tick:
            return False

    return True

  def _extract_agent_from_call_to_action(
      self, call_to_action: str
  ) -> str | None:
    """Extract player full name from standard call_to_action format."""
    match = re.search(
        r"(?:what will|what should)\s+([^?]+?)\s+do\s+next",
        call_to_action,
        re.IGNORECASE,
    )
    if match:
      name = match.group(1).strip()
      # Verify the parsed name is actually a registered player
      if name in self._player_names:
        return name
    return None

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      entity_name = self._get_resolving_entity()
      if not entity_name:
        # Robust fallback: parse agent name directly from the call to action
        entity_name = self._extract_agent_from_call_to_action(
            action_spec.call_to_action
        )
      if not entity_name:
        # Thread-local fallback
        candidate = getattr(self.get_entity(), "_active_capture_key", None)
        if candidate in self._player_names:
          entity_name = candidate
      clock = self._get_clock()
      current_tick = clock.current_tick if clock is not None else -1
      if entity_name:
        with self._lock:
          self._in_resolve_phase[entity_name] = current_tick
        logging.info(
            "TickGated.pre_act(RESOLVE): %s entering resolve phase at tick %d",
            entity_name,
            current_tick,
        )
      else:
        with self._lock:
          self._in_resolve_phase["__sequential__"] = current_tick
          for name in self._player_names:
            self._in_resolve_phase[name] = current_tick
        logging.info(
            "TickGated.pre_act(RESOLVE): batch entering resolve phase at"
            " tick %d",
            current_tick,
        )
      return ""

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      if not action_spec.options:
        return ""
      # Fire any scheduled social events for this tick (dates, meetups)
      # BEFORE selecting eligible agents. This creates conversations and moves
      # participants, ensuring they are recognized as in-conversation.
      social_scheduler = self._get_social_scheduler()
      if social_scheduler is not None and hasattr(
          social_scheduler, "fire_pending_events_for_current_tick"
      ):
        social_scheduler.fire_pending_events_for_current_tick()

      eligible_agents = [
          name for name in action_spec.options if self._is_eligible(name)
      ]
      if eligible_agents:
        with self._lock:
          self._currently_active_player = eligible_agents[0]
        if len(action_spec.options) > 1:
          try:
            gm = self.get_entity()
            thread_id = threading.current_thread().ident
            if hasattr(gm, "set_capture_key_for_thread"):
              gm.set_capture_key_for_thread(thread_id, eligible_agents[0])
          except Exception:  # pylint: disable=broad-except
            pass
          return eligible_agents[0]
        logging.info(
            "TickGated.pre_act(NEXT_ACTING): %d/%d eligible (%s) at tick %d",
            len(eligible_agents),
            len(action_spec.options),
            ", ".join(eligible_agents),
            self._last_seen_tick,
        )
        return ", ".join(eligible_agents)
      # Log why not eligible
      conv_state = self._get_conversation_state()
      clock = self._get_clock()
      current_tick = (
          clock.current_tick if clock is not None else self._last_seen_tick
      )
      for agent_name in action_spec.options[:3]:
        in_conv = conv_state is not None and conv_state.is_in_conversation(
            agent_name
        )
        if clock is not None and hasattr(clock, "has_agent_acted"):
          already_acted = clock.has_agent_acted(agent_name)
        else:
          with self._lock:
            already_acted = agent_name in self._acted_this_tick
        logging.info(
            "TickGated.pre_act(NEXT_ACTING): %s -> NOT eligible"
            " (in_conv=%s, acted=%s, tick=%d)",
            agent_name,
            in_conv,
            already_acted,
            current_tick,
        )
      return ""

    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      agent_name = self._extract_agent_name_from_call_to_action(
          action_spec.call_to_action
      )
      if agent_name is None:
        return ""

      social_scheduler = self._get_social_scheduler()
      if social_scheduler is not None and hasattr(
          social_scheduler, "fire_pending_events_for_current_tick"
      ):
        social_scheduler.fire_pending_events_for_current_tick()

      conv_state = self._get_conversation_state()
      if conv_state is not None and conv_state.is_in_conversation(agent_name):
        if self._enable_conversation_shards:
          logging.warning(
              "TickGated.pre_act(NEXT_ACTION_SPEC): %s in conversation on"
              " island rules while sharding enabled",
              agent_name,
          )
          return ""
        if not conv_state.is_my_turn(agent_name):
          logging.info(
              "TickGated.pre_act(NEXT_ACTION_SPEC): %s in conv, not turn",
              agent_name,
          )
          return ""
        speech_call = entity_lib.DEFAULT_CALL_TO_SPEECH.replace(
            "{name}", agent_name
        )
        result_spec = entity_lib.ActionSpec(
            call_to_action=speech_call,
            output_type=entity_lib.OutputType.FREE,
            options=(),
            tag="speech",
        )
        with self._lock:
          self._agent_to_last_tag[agent_name] = result_spec.tag
        logging.info(
            "TickGated.pre_act(NEXT_ACTION_SPEC): %s -> SPEECH spec",
            agent_name,
        )
        return engine_lib.action_spec_to_string(result_spec)

      clock = self._get_clock()
      time_prefix = ""
      if clock is not None and hasattr(clock, "get_pre_act_value"):
        current_time_str = clock.get_pre_act_value().strip()
        if current_time_str:
          time_prefix = f"It is currently {current_time_str}. "

      action_call = (
          f"{time_prefix}Plan your activities for the next 2 hours (120"
          " minutes). Provide a numbered list of steps you intend to take. Be"
          " realistic about what can be accomplished in this time. Consider"
          " staying at the current location or moving to a different one."
      )
      default_spec = entity_lib.ActionSpec(
          call_to_action=action_call,
          output_type=entity_lib.OutputType.FREE,
          options=(),
          tag="action",
      )
      with self._lock:
        self._agent_to_last_tag[agent_name] = default_spec.tag
      logging.info(
          "TickGated.pre_act(NEXT_ACTION_SPEC): %s -> ACTION spec",
          agent_name,
      )
      return engine_lib.action_spec_to_string(default_spec)

    return ""

  def post_act(self, event: str) -> str:
    with self._lock:
      is_seq = self._in_resolve_phase.pop("__sequential__", None)
      if is_seq is not None:
        conv_state = self._get_conversation_state()
        for name in self._player_names:
          self._in_resolve_phase.pop(name, None)
          in_conv = conv_state is not None and conv_state.is_in_conversation(
              name
          )
          if not in_conv:
            self._acted_this_tick.add(name)
        logging.info(
            "TickGated.post_act: sequential batch acted_this_tick=%s",
            self._acted_this_tick,
        )
        return ""

    thread_id = threading.current_thread().ident
    gm = self.get_entity()
    agent = None
    if hasattr(gm, "_capture_key_by_thread"):
      agent = gm._capture_key_by_thread.get(thread_id)  # pylint: disable=protected-access
    if not agent:
      agent = getattr(gm, "_active_capture_key", None)

    if not agent:
      for name in self._player_names:
        if name in event:
          agent = name
          break
    else:
      logging.info(
          "TickGated.post_act: extracted agent from thread capture: %s",
          agent,
      )

    if not agent:
      logging.info(
          "TickGated.post_act: no agent found in event or capture key: %s",
          event[:100],
      )
      return ""

    with self._lock:
      resolve_tick = self._in_resolve_phase.pop(agent, None)
      current_resolve_keys = list(self._in_resolve_phase.keys())

    if resolve_tick is None:
      logging.info(
          "TickGated.post_act: %s not in resolve phase, skipping"
          " (in_resolve_phase=%s)",
          agent,
          current_resolve_keys,
      )
      return ""

    # Check if the resolution belongs to the current simulation tick.
    # Late resolutions from a previous tick must be skipped to prevent
    # cross-tick pollution.
    clock = self._get_clock()
    current_tick = clock.current_tick if clock is not None else -1
    if clock is not None and current_tick != resolve_tick:
      logging.warning(
          "TickGated.post_act: %s late resolution completed (resolve_tick=%d,"
          " current_tick=%d). Skipping acted registration to prevent tick"
          " pollution.",
          agent,
          resolve_tick,
          current_tick,
      )
      return ""
    conv_state = self._get_conversation_state()
    in_conv = conv_state is not None and conv_state.is_in_conversation(agent)
    if not in_conv:
      with self._lock:
        self._acted_this_tick.add(agent)
        logging.info(
            "TickGated.post_act: %s acted, acted_this_tick=%s",
            agent,
            self._acted_this_tick,
        )
      # Notify the clock barrier so it can advance the tick once ALL
      # agents have acted.  Previously only the local _acted_this_tick
      # set was updated — the clock's barrier was never triggered,
      # causing fast entities to see the old tick and slow entities
      # to miss ticks entirely.
      clock = self._get_clock()
      if clock is not None and hasattr(clock, "mark_agent_acted"):
        clock.mark_agent_acted(agent)
    else:
      logging.info(
          "TickGated.post_act: %s in conversation, not counting as acted",
          agent,
      )
    return ""

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player

  def get_state(self) -> entity_component.ComponentState:
    return {
        "player_names": list(self._player_names),
        "currently_active_player": self._currently_active_player,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if "player_names" in state:
      self._player_names = state["player_names"]
    if "currently_active_player" in state:
      with self._lock:
        self._currently_active_player = state["currently_active_player"]


class TickBasedTerminator(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Terminates the simulation when the clock reaches a target tick count.

  Follows the same pattern as terminate.Terminate and SceneBasedTerminator:
  returns 'Yes' from pre_act when output_type is TERMINATE and the clock
  has reached or exceeded the configured max_ticks.

  When max_ticks is None, never terminates (same as NeverTerminate).
  """

  def __init__(
      self,
      clock_key: str = "clock",
      max_ticks: int | None = None,
  ):
    super().__init__()
    self._clock_key = clock_key
    self._max_ticks = max_ticks

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      if self._max_ticks is not None:
        try:
          clock = self.get_entity().get_component(self._clock_key)
          if clock.current_tick >= self._max_ticks:
            return "Yes"
        except (AttributeError, KeyError):
          pass
    return "No"

  def get_state(self) -> entity_component.ComponentState:
    return {"max_ticks": self._max_ticks}

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._max_ticks = state.get("max_ticks", self._max_ticks)


ISLAND_CLOCK_DESCRIPTION = """
Time on the island starts Thursday, January 1st, 2026.
Time advances in 2-hour ticks. Waking hours are 7:00 AM to 11:00 PM
(8 ticks per day). Sleeping hours are skipped.
Time format: "DayOfWeek, Month Dayth, HH:MM AM/PM"
(e.g. "Thursday, January 1st, 9:00 AM", "Friday, January 2nd, 11:00 AM").

Days of the week determine schedules:
- Monday to Friday: Workdays. Residents go to their workplaces.
- Saturday and Sunday: Rest days. Residents relax, socialize, and explore.

Daily schedule:
- Morning (7:00 AM - 1:00 PM): Most residents at work or morning activities
- Afternoon (1:00 PM - 5:00 PM): Peak activity in public spaces
- Evening (5:00 PM - 11:00 PM): Social gatherings, dining, rest
"""


@dataclasses.dataclass
class IslandGameMaster(prefab_lib.Prefab):
  """Game Master for island simulation.

  Based on situated_in_time_and_place pattern with:
  - Island-specific locations
  - 120-minute tick clock
  - Movement resolution
  - Encounter handling for co-located agents

  Params:
    name: GM name (default: "island rules")
    start_time: Starting time string (default: "Thursday, January 1st, 7:00 AM")
    extra_components: Additional GM components
  """

  description: str = "Game master for island simulation"
  entities: tuple[entity_agent_with_logging.EntityAgentWithLogging, ...] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    name = self.params.get("name", "island rules")
    start_time = self.params.get("start_time", "Thursday, January 1st, 7:00 AM")
    extra_components = self.params.get("extra_components", {})
    initial_locations = self.params.get("initial_locations", {})
    use_relevant_memories = self.params.get("use_relevant_memories", False)
    max_ticks = self.params.get("max_ticks", None)
    tick_interval_minutes = self.params.get("tick_interval_minutes", 120)
    setting = self.params.get("setting", None)

    player_names = [entity.name for entity in self.entities]

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    associative_memory = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    instructions_key = "instructions"
    instructions = simulationist_instructions.SimulationistInstructions(
        setting=setting,
    )

    examples_key = "examples"
    examples = simulationist_instructions.SimulationistExamples()

    player_characters_key = "player_characters"
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    display_events_key = "display_events"
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label="Story so far (recent events)",
    )

    observation_context: dict[str, str] = {}
    # Populated after entity_locations is created.
    locations_ref: list[gm_components.world_state.Locations | None] = [None]

    def _location_event_filter(event: str) -> bool:
      observation_target = observation_context.get("target", "")
      if not observation_target:
        return True
      loc_component = locations_ref[0]
      if loc_component is None:
        return True
      try:
        state = loc_component.get_state()
        locs = state.get("entity_locations", {})
        entity_loc = str(locs.get(observation_target, ""))
      except (AttributeError, KeyError):
        return True
      if not entity_loc:
        return True
      normalized_loc = entity_loc.replace("_", " ").lower()
      event_lower = event.lower()
      loc_match = (
          f"// {entity_loc.lower()}" in event_lower
          or f"// {normalized_loc}" in event_lower
      )
      return loc_match

    filtered_display_events_key = "filtered_display_events"
    filtered_display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label="Story so far (recent events)",
        event_filter_fn=_location_event_filter,
    )

    relevant_memories_key = "relevant_memories"
    relevant_memories = None
    if use_relevant_memories:
      relevant_memories = (
          actor_components.all_similar_memories.AllSimilarMemories(
              model=model,
              components=[display_events_key],
              num_memories_to_retrieve=25,
              pre_act_label="Background info",
          )
      )

    locations_constant_key = "locations_constant"
    locations_constant = actor_components.constant.Constant(
        locations_lib.get_island_locations_prompt(setting=setting),
        pre_act_label="Locations",
    )

    clock_constant_key = "clock_constant"
    clock_constant = actor_components.constant.Constant(
        ISLAND_CLOCK_DESCRIPTION,
        pre_act_label="Clock description",
    )

    clock_key = "clock"
    locations_key = "locations"
    async_conversation_state_key = async_conv.DEFAULT_ASYNC_CONVERSATION_KEY
    sim_clock = self.params.get("clock")
    if sim_clock is None:
      sim_clock = fixed_clock.FixedIntervalClock(
          start_time=start_time,
          tick_interval_minutes=tick_interval_minutes,
          waking_hour_start=7,
          waking_hour_end=23,
          player_names=player_names,
          conversation_state_key=async_conversation_state_key,
          max_ticks=max_ticks,
          pre_act_label="\nCurrent time",
          initial_locations=initial_locations
          if isinstance(initial_locations, dict)
          else None,
          locations_key=locations_key,
      )
    # Get all valid canonical location names for constrained location parsing
    valid_locations = locations_lib.get_all_public_location_names(
        setting=setting
    )
    # Add all agent home/work locations if initial_locations is provided
    if isinstance(initial_locations, dict):
      for loc in initial_locations.values():
        if loc and loc not in valid_locations:
          valid_locations.append(loc)

    locations_deps = [
        instructions_key,
        locations_constant_key,
        clock_constant_key,
        player_characters_key,
        display_events_key,
        clock_key,
    ]
    if use_relevant_memories:
      locations_deps.append(relevant_memories_key)

    entity_locations = gm_components.world_state.Locations(
        model=model,
        entity_names=player_names,
        prompt=locations_lib.get_island_locations_prompt(setting=setting),
        initial_locations=initial_locations,
        valid_locations=valid_locations,
        components=locations_deps,
        pre_act_label="\nCurrent locations",
    )
    locations_ref[0] = entity_locations

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_obs_deps = [
        instructions_key,
        player_characters_key,
        locations_constant_key,
        clock_constant_key,
        filtered_display_events_key,
        clock_key,
        locations_key,
    ]
    if use_relevant_memories:
      make_obs_deps.append(relevant_memories_key)

    # Conversations are executed inline by the island GM in the single-process
    # runner. (Dedicated conversation shards are only available in distributed
    # runners that are not part of this release.)
    enable_conversation_shards = False

    make_observation = LocationAwareMakeObservation(
        model=model,
        player_names=player_names,
        components=make_obs_deps,
        locations_key=locations_key,
        social_scheduler_key=social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY,
        observation_context=observation_context,
        initial_player_locations=initial_locations,
        filtered_display_events_key=filtered_display_events_key,
        enable_conversation_shards=enable_conversation_shards,
    )

    enable_nighttime_social = self.params.get("enable_nighttime_social", False)
    enable_nighttime_marketplace = self.params.get(
        "enable_nighttime_marketplace", False
    )
    has_nighttime_gm = bool(
        enable_nighttime_social or enable_nighttime_marketplace
    )
    if has_nighttime_gm and hasattr(sim_clock, "enable_night_gate"):
      # Hold each agent's island action *and* queued island observations
      # until it has been through tonight's nighttime GM(s).
      sim_clock.enable_night_gate()
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    tick_gated = TickGatedNextActing(
        player_names=player_names,
        clock_key=clock_key,
        async_conversation_key=async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
        social_scheduler_key=social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY,
        enable_conversation_shards=enable_conversation_shards,
        has_nighttime_gm=has_nighttime_gm,
    )

    event_resolution_key = (
        gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    event_resolution_deps = [
        instructions_key,
        examples_key,
        player_characters_key,
        locations_constant_key,
        clock_constant_key,
        filtered_display_events_key,
        clock_key,
        locations_key,
    ]
    if use_relevant_memories:
      event_resolution_deps.append(relevant_memories_key)

    terminate_boring = self.params.get("terminate_boring", True)
    event_resolution = IslandEventResolution(
        model=model,
        player_names=player_names,
        components=event_resolution_deps,
        clock_key=clock_key,
        locations_key=locations_key,
        display_events_key=display_events_key,
        next_acting_key=next_actor_key,
        terminate_boring=terminate_boring,
        enable_conversation_shards=enable_conversation_shards,
    )

    max_conv_turns = self.params.get("max_conv_turns", 8)
    async_conversation_state = async_conv.AsyncConversationState(
        player_names=player_names,
        max_turns=max_conv_turns,
        clock_key=clock_key,
        conversation_timeout=10800.0,
    )

    # --- Temporal Nudge (work hours, end-of-day reset, weekends) ---
    temporal_nudge_key = "temporal_nudge"
    agent_configs = self.params.get("agent_configs", [])
    temporal_nudge = temporal_nudge_lib.TemporalNudge(
        agent_configs=agent_configs,
        clock_key=clock_key,
        locations_key=locations_key,
        make_observation_key=make_observation_key,
        memory_key=memory_component_key,
    )

    # --- Conversation Director (replaces broken AsyncConversationTrigger) ---
    conversation_director_key = "conversation_director"
    cooldown_ticks = self.params.get("cooldown_ticks", 4)
    max_conversation_size = self.params.get("max_conversation_size", 4)
    # Build agent descriptions for the LLM from agent configs
    agent_descriptions = {}
    for cfg in agent_configs:
      if hasattr(cfg, "personality") and cfg.personality:
        agent_descriptions[cfg.name] = cfg.personality
    conversation_director = conv_director_lib.ConversationDirector(
        model=model,
        player_names=player_names,
        async_conversation_key=async_conversation_state_key,
        locations_key=locations_key,
        clock_key=clock_key,
        make_observation_key=make_observation_key,
        max_conversation_size=max_conversation_size,
        cooldown_ticks=cooldown_ticks,
        agent_descriptions=agent_descriptions,
    )

    # --- Social Scheduler (for scheduled first dates, friend meetups, etc.) ---
    # Only the PRIMARY island GM (named exactly "island rules") runs the
    # SocialScheduler. Secondary island GMs ("island rules 1", etc.) get a
    # no-op stub. This prevents a race condition where multiple island GMs
    # try to fire the same social events concurrently, causing duplicate or
    # dropped conversations.
    social_scheduler_key = social_scheduler_lib.DEFAULT_SOCIAL_SCHEDULER_KEY
    is_primary_island_gm = name == "island rules"
    social_events = self.params.get("social_events")
    if social_events is None:
      social_events = []
    if is_primary_island_gm:
      social_scheduler = social_scheduler_lib.SocialScheduler(
          model=model,
          player_names=player_names,
          events=social_events,
          async_conversation_key=async_conversation_state_key,
          locations_key=locations_key,
          clock_key=clock_key,
          make_observation_key=make_observation_key,
          agent_descriptions=agent_descriptions,
          home_locations=(
              initial_locations if isinstance(initial_locations, dict) else None
          ),
      )
    else:
      # Secondary island GMs delegate event coordination to the primary.
      # _get_social_scheduler() returns None for stubs, so pre_act() skips
      # event firing on these workers.
      social_scheduler = actor_components.constant.Constant(
          state="",
          pre_act_label="\nSocial Scheduler (delegated to primary)",
      )

    # --- Next Game Master (Multi-GM switching) ---
    next_game_master_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )
    num_marketplace_gms = self.params.get("num_marketplace_gms", 1)
    # The primary daytime GM coordinator is always "island rules".
    # Consecutive nighttime order:
    #   island rules (day) -> x_rules (night stage 1)
    #                      -> marketplace_rules (night stage 2)
    #                      -> island rules (next morning)
    island_coordinator_name = "island rules"
    nighttime_gm = ""
    if enable_nighttime_social:
      nighttime_gm = "x_rules"
    elif enable_nighttime_marketplace:
      nighttime_gm = "marketplace_rules"
    if nighttime_gm:
      # Only the async engine runs one loop per entity; the sequential and
      # simultaneous engines switch every entity at once.
      engine_type = str(self.params.get("engine_type", "async"))
      next_game_master = time_based_lib.TimeBasedNextGM(
          island_gm_name=island_coordinator_name,
          nighttime_gm_name=nighttime_gm,
          clock_key=clock_key,
          async_conversation_key=async_conversation_state_key,
          social_scheduler_key=social_scheduler_key,
          enable_conversation_shards=enable_conversation_shards,
          num_marketplace_gms=num_marketplace_gms,
          player_names=player_names,
          per_agent_switching=(engine_type == "async"),
      )
    else:
      next_game_master = actor_components.constant.Constant(
          state=island_coordinator_name,
          pre_act_label="\nNext Game Master",
      )

    terminate_key = terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY

    components_of_game_master = {
        instructions_key: instructions,
        examples_key: examples,
        player_characters_key: player_characters,
        locations_constant_key: locations_constant,
        clock_constant_key: clock_constant,
        display_events_key: display_events,
        filtered_display_events_key: filtered_display_events,
        clock_key: sim_clock,
        locations_key: entity_locations,
        memory_component_key: associative_memory,
        temporal_nudge_key: temporal_nudge,
        make_observation_key: make_observation,
        next_actor_key: tick_gated,
        next_action_spec_key: tick_gated,
        terminate_key: sim_clock,
        async_conversation_state_key: async_conversation_state,
        # Stub for backward compat with checkpoints that saved state for
        # the now-removed AsyncConversationResolution component.
        "async_conversation_resolution": actor_components.constant.Constant(
            state="",
            pre_act_label="\nConversation (deprecated)",
        ),
        # Stub for backward compat with checkpoints that saved state for
        # the now-removed WorldState component.
        "world_state": actor_components.constant.Constant(
            state="",
            pre_act_label="\nWorld State (deprecated)",
        ),
        # Stub for backward compat with the old trigger.
        "async_conversation_trigger": actor_components.constant.Constant(
            state="",
            pre_act_label="\nConversation Trigger (deprecated)",
        ),
        conversation_director_key: conversation_director,
        social_scheduler_key: social_scheduler,
        event_resolution_key: event_resolution,
        next_game_master_key: next_game_master,
    }

    # --- Economic Components (Food Consumption & Fiscal Scheduler) ---
    economic_profiles = self.params.get("economic_profiles")
    if economic_profiles is None and agent_configs:
      food_min_daily = self.params.get("food_min_daily", 3)
      starting_food_units = self.params.get("starting_food_units", 12)
      laid_off_str = self.params.get("laid_off_agents", "")
      if isinstance(laid_off_str, str):
        laid_off_names = [
            n.strip() for n in laid_off_str.split(",") if n.strip()
        ]
      else:
        laid_off_names = list(laid_off_str)
      economic_profiles = economic_profile.build_economic_profiles(
          agent_configs,
          food_min_daily=food_min_daily,
          laid_off_agents=laid_off_names,
          starting_food_units=starting_food_units,
      )

    if economic_profiles:
      fiscal_config_name = self.params.get("fiscal_config", "control")
      fiscal_events = self.params.get("fiscal_events")
      if fiscal_events is None:
        fiscal_events = fiscal_configs.get_fiscal_events(fiscal_config_name)

      world_state_obj = self.params.get("world_state")
      food_consumption_comp = food_consumption.FoodConsumptionComponent(
          profiles=economic_profiles,
          clock_key=clock_key,
          make_observation_key=make_observation_key,
          world_state=world_state_obj,
      )
      fiscal_scheduler_comp = fiscal_scheduler.FiscalScheduler(
          fiscal_events=fiscal_events,
          profiles=economic_profiles,
          clock_key=clock_key,
          make_observation_key=make_observation_key,
          world_state=world_state_obj,
      )
      components_of_game_master["food_consumption"] = food_consumption_comp
      components_of_game_master["fiscal_scheduler"] = fiscal_scheduler_comp

    if extra_components:
      components_of_game_master.update(extra_components)

    if use_relevant_memories and relevant_memories is not None:
      components_of_game_master[relevant_memories_key] = relevant_memories

    component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
        measurements=async_measurements_lib.ReactiveMeasurements(),
    )

    return game_master
