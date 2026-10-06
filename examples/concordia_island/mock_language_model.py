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

"""Mock language model that returns sensible strings for debugging.

Used for fast debugging of async engine concurrency. Returns detailed,
sensible responses without LLM calls, with location awareness and dialogue.
"""

from collections.abc import Collection, Mapping, Sequence
import re
import threading
import time
from typing import Any, override

from concordia.language_model import language_model


# Location for conversation testing - agents will move here
COMMON_ROOM = "sunset_apartments_common_room"

# All known agent names
KNOWN_AGENTS = [
    "Maria Santos",
    "David Chen",
    "Lisa Martinez",
    "James Wilson",
    "Sophie Brown",
]

# Agent actions to cycle through
AGENT_ACTIONS = [
    "decides to go to the common room for some coffee",
    "walks to the common room to see if anyone is around",
    "heads to the common room for a morning chat",
    "goes to the common room to relax",
    "makes their way to the common room",
]

# Conversation dialogue lines - agents cycle through these
CONVERSATION_LINES = [
    "Good morning! It's such a beautiful day on the island today.",
    "Yes, the weather is perfect. Did you sleep well?",
    "I slept wonderfully. Have you had coffee yet?",
    "Just about to make some. Would you like a cup?",
    "That would be lovely, thank you so much!",
    "Do you have any plans for today?",
    "I was thinking of going to the beach later.",
    "That sounds nice. Maybe I'll join you.",
    "The more the merrier! We could have a picnic.",
    "Great idea! I'll bring some snacks.",
    "It's nice to have neighbors like you.",
    "I agree. This island community is wonderful.",
    "We should do this more often.",
    "Absolutely. Same time tomorrow?",
    "Perfect. See you then!",
]


class MockLanguageModel(language_model.LanguageModel):
  """Mock model returning detailed responses for debugging concurrency."""

  def __init__(self, delay: float = 0.1, verbose: bool = True) -> None:
    """Initialize mock model.

    Args:
      delay: Simulated delay per call (seconds). Set to 0 for instant.
      verbose: Whether to print debug info on each call.
    """
    self._delay = delay
    self._verbose = verbose
    self._call_count = 0
    self._lock = threading.RLock()
    # Track agent locations for conversation triggering
    self._agent_locations: dict[str, str] = {}
    self._action_index = 0
    self._conversation_index = 0

  def _log(self, method: str, prompt_preview: str, response_preview: str):
    if self._verbose:
      thread = threading.current_thread().name
      with self._lock:
        self._call_count += 1
        print(
            f"[MockLLM #{self._call_count}] [{thread}] {method}: "
            f'"{prompt_preview[:50]}..." -> "{response_preview[:80]}..."'
        )

  def _extract_agent_name(self, prompt: str) -> str | None:
    """Extract agent name from the prompt.

    Prioritizes specific patterns to avoid returning wrong agent when
    multiple names appear in the prompt.

    Args:
      prompt: The LLM prompt to search.

    Returns:
      The agent name found, or None.
    """
    # First check for speech prompt pattern - most specific
    match = re.search(
        r"what is ([A-Z][a-z]+ [A-Z][a-z]+) likely to say",
        prompt,
        re.IGNORECASE,
    )
    if match:
      return match.group(1)

    # Check for "What would X do" pattern
    match = re.search(r"What would ([A-Z][a-z]+ [A-Z][a-z]+) do", prompt)
    if match:
      return match.group(1)

    # Check for entity action pattern "Entity X is next to act"
    match = re.search(r"Entity ([A-Z][a-z]+ [A-Z][a-z]+) is next", prompt)
    if match:
      return match.group(1)

    # Check for situation / person question patterns
    match = re.search(
        r"(?:situation is|person is|observations of|traits of|traits:)"
        r" ([A-Z][a-z]+ [A-Z][a-z]+)",
        prompt,
        re.IGNORECASE,
    )
    if match:
      return match.group(1)

    # General regex for any capitalized two-word agent name in prompt
    name_matches = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", prompt)
    for candidate in name_matches:
      if candidate not in (
          "Concordia Island",
          "Sunset Apartments",
          "Common Room",
          "Town Square",
          "Coral Village",
          "Palm Heights",
          "Ocean View",
          "Paradise Point",
          "Game Master",
          "Fixed Interval",
          "Action Spec",
      ):
        return candidate

    # Fallback: return first known agent in prompt
    for agent in KNOWN_AGENTS:
      if agent in prompt:
        return agent
    return "The agent"

  def _get_next_action(self, agent_name: str) -> str:
    """Get the next action for an agent - all go to common room."""
    with self._lock:
      # Update agent location to common room
      self._agent_locations[agent_name] = COMMON_ROOM
      action = AGENT_ACTIONS[self._action_index % len(AGENT_ACTIONS)]
      self._action_index += 1
      return f"{agent_name} {action}."

  def _get_conversation_line(self, agent_name: str) -> str:
    """Get the next line of dialogue for an agent in conversation.

    Uses the format: {name} -- "dialogue"

    Args:
      agent_name: The agent who is speaking.

    Returns:
      Formatted dialogue line.
    """
    with self._lock:
      line = CONVERSATION_LINES[
          self._conversation_index % len(CONVERSATION_LINES)
      ]
      self._conversation_index += 1
      return f'{agent_name} -- "{line}"'

  def _get_location_response(self, prompt: str) -> str:  # pylint: disable=unused-argument
    """Return proper location format for Locations component.

    Args:
      prompt: The LLM prompt (unused).

    Returns:
      Formatted location string.
    """
    with self._lock:
      # Always return all agents in common room for conversation testing
      locations = []
      for agent in KNOWN_AGENTS:
        self._agent_locations[agent] = COMMON_ROOM
        locations.append(f"{agent}|{COMMON_ROOM}")
      return ",".join(locations)

  def _is_conversation_prompt(self, prompt: str) -> bool:
    """Check if this prompt is for an active conversation."""
    prompt_lower = prompt.lower()
    conversation_indicators = [
        "you are having a conversation with",
        "it is your turn to speak",
        "likely to say next",
        "in conversation with",
        "having a conversation",
        "dialogue",
        "what would you like to say",
        "respond to",
        "participating in a conversation",
        "conversation between",
        "talking with",
        "speaking with",
        "recent exchanges:",
    ]
    return any(ind in prompt_lower for ind in conversation_indicators)

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    time.sleep(self._delay)

    # Generate contextual response based on prompt patterns
    prompt_lower = prompt.lower()
    agent_name = self._extract_agent_name(prompt)

    # Check for marketplace first
    if (
        "[available goods & services at tonight's marketplace]" in prompt_lower
        or "return only the json." in prompt_lower
        or (
            "what will" in prompt_lower
            and "do tonight in the marketplace?" in prompt_lower
        )
        or (
            "marketplace" in prompt_lower and "checking account" in prompt_lower
        )
    ):
      # Marketplace prompt - return a valid marketplace action JSON
      response = (
          '{"action": "bid", "good": "Maruchan Ramen Meal", "price": 3.0,'
          ' "qty": 1}'
      )

    # Check for X social media / dating prompt
    elif (
        "do on x?" in prompt_lower
        or '"action": "post"' in prompt_lower
        or '"action": "create_profile"' in prompt_lower
        or '"action": "select_partner"' in prompt_lower
    ):
      author = agent_name or "Agent"
      response = (
          f'{{"action": "post", "author": "{author}",'
          ' "title": "Evening update", "content": "Checking in on X tonight!"}'
      )

    # Check for conversation
    elif self._is_conversation_prompt(prompt) and agent_name:
      response = self._get_conversation_line(agent_name)
      print(f"[MOCK-CONVERSATION] {response}")

    elif (
        "is the game/simulation finished" in prompt_lower
        or "should the simulation terminate" in prompt_lower
        or "terminate: yes or no" in prompt_lower
        or "is the game finished" in prompt_lower
        or prompt_lower.strip().endswith("terminate?")
    ):
      response = "No"

    elif "which entities act next" in prompt_lower:
      # next_acting prompt - return an agent name that needs to act
      response = "Lisa Martinez"

    elif (
        "where are the named people currently located" in prompt_lower
        or "person1|location1" in prompt_lower
    ):
      # This is the Locations component asking about locations
      # Return proper format: person1|location1,person2|location2
      response = self._get_location_response(prompt)
      print(
          f"[MOCK-DEBUG] Location query detected! Returning: {response[:100]}"
      )

    elif "comma-separated list of locations" in prompt_lower:
      # Locations initialization - return island locations
      response = (
          "sunset_apartments_common_room|A cozy common room for residents,"
          "town_square|The central gathering place,"
          "beach|A beautiful sandy beach"
      )

    elif "question: what situation" in prompt_lower:
      response = (
          f"{agent_name} is going about their daily routine on the island."
      )

    elif "question: what kind of person" in prompt_lower:
      response = (
          f"{agent_name} is a friendly and conscientious community member."
      )

    elif "reflect" in prompt_lower or "journal" in prompt_lower:
      response = f"{agent_name} reflects on the day and plans ahead."

    elif "core traits" in prompt_lower and agent_name:
      # This is the agent deciding what to do - return a proper action
      response = self._get_next_action(agent_name)

    elif (
        "action spec" in prompt_lower or "respond in the format" in prompt_lower
    ):
      # Return JSON format action spec
      response = (
          '{"call_to_action": "What do you do next?", "output_type": "free",'
          ' "options": [], "tag": "action"}'
      )

    elif "what does" in prompt_lower and "observe" in prompt_lower:
      # Observation prompt - make it location-aware
      if agent_name:
        # Check if others are in common room
        others_in_common = [
            n
            for n, l in self._agent_locations.items()
            if l == COMMON_ROOM and n != agent_name
        ]
        if others_in_common:
          others_str = " and ".join(others_in_common[:2])
          response = (
              f"{agent_name} is in the common room. They see {others_str} "
              "already here, having coffee."
          )
        else:
          response = (
              f"{agent_name} enters the quiet common room. "
              "Morning light streams through the windows."
          )
      else:
        response = (
            "The agent sees a peaceful scene in the common room. "
            "People are going about their morning routines."
        )

    elif "resolve" in prompt_lower or "putative event" in prompt_lower:
      # Event resolution
      if agent_name:
        response = f"Event: {agent_name} is now in the common room."
      else:
        response = "Event: The agent continues their daily activities."

    elif "location" in prompt_lower:
      # Generic location query
      response = COMMON_ROOM

    elif "time" in prompt_lower or "clock" in prompt_lower:
      response = "Day 1, 8:30 AM"

    elif "conversation" in prompt_lower or "talk" in prompt_lower:
      # Conversation-related prompts - generate dialogue if agent involved
      if agent_name:
        response = self._get_conversation_line(agent_name)
        print(f"[MOCK-CONVERSATION] {response}")
      else:
        response = "Yes, they should have a conversation about the morning."

    elif "state" in prompt_lower or "world" in prompt_lower:
      response = "Everything is normal. The island is peaceful."

    else:
      # Default: for agent action prompts, generate proper actions
      if agent_name:
        response = self._get_next_action(agent_name)
      else:
        response = (
            '{"call_to_action": "Continue with your activity.", "output_type":'
            ' "free", "options": [], "tag": "action"}'
        )

    self._log("sample_text", prompt, response)
    return response

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    time.sleep(self._delay)

    if not responses:
      return 0, "", {}

    # Log responses for debugging
    if self._verbose:
      thread = threading.current_thread().name
      with self._lock:
        print(f"[MockLLM] [{thread}] sample_choice options: {list(responses)}")

    prompt_lower = prompt.lower()

    # For conversation trigger - say YES to start conversations
    if "conversation" in prompt_lower or "should they talk" in prompt_lower:
      # Find the "Yes" option
      for i, r in enumerate(responses):
        if "yes" in r.lower():
          self._log("sample_choice", prompt, f"-> {r} (YES for conversation)")
          return i, r, {}

    # For simulation termination - respect if Clock terminator signaled Yes
    if "terminate" in prompt_lower or "should the simulation" in prompt_lower:
      if (
          "\nyes\n" in prompt_lower
          or "clock: yes" in prompt_lower
          or "clock description\nyes" in prompt_lower
          or prompt_lower.endswith("yes")
      ):
        for i, r in enumerate(responses):
          if "yes" in r.lower():
            self._log("sample_choice", prompt, f"-> {r} (YES for termination)")
            return i, r, {}
      else:
        for i, r in enumerate(responses):
          if "no" in r.lower():
            self._log("sample_choice", prompt, f"-> {r} (NO for termination)")
            return i, r, {}

    # For general choice options: if prompt explicitly mentions No variant
    for i, r in enumerate(responses):
      r_stripped = r.strip().lower()
      # Search prompt for this letter followed by No
      patterns = [
          rf"\b{re.escape(r_stripped)}\)?[:\)]\s*no\b",  # a) No, a: No
          rf"\({re.escape(r_stripped)}\)\s*no\b",  # (a) No
          rf"\b{re.escape(r_stripped)}\s+no\b",  # a No
      ]
      for pattern in patterns:
        if re.search(pattern, prompt_lower):
          self._log("sample_choice", prompt, f"-> {r} (found No in prompt)")
          return i, r, {}

    # Also look for the option itself being a "No" variant
    no_variants = ("no", "no.", "false", "continue")
    for i, r in enumerate(responses):
      if r.lower().strip() in no_variants:
        self._log("sample_choice", prompt, f"-> {r} (No variant)")
        return i, r, {}

    # If responses are single letters, pick the LAST option
    if all(len(r.strip()) <= 2 for r in responses):
      choice = len(responses) - 1
      self._log(
          "sample_choice", prompt, f"-> {responses[choice]} (last letter)"
      )
      return choice, responses[choice], {}

    # Default: pick first option
    choice = 0
    self._log("sample_choice", prompt, f"-> {responses[choice]}")
    return choice, responses[choice], {}

  def get_agent_locations(self) -> dict[str, str]:
    """Return current agent locations for debugging."""
    with self._lock:
      return dict(self._agent_locations)


class FastMockLanguageModel(MockLanguageModel):
  """Zero-delay mock model for maximum speed testing."""

  def __init__(self, verbose: bool = True) -> None:
    super().__init__(delay=0.0, verbose=verbose)
