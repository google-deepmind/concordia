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

"""A component that resolves events by generating a narrative and then individual observations.

This version uses a simplified 3-step, text-only process for maximum
reliability.
"""

from collections.abc import Sequence
import datetime
import functools
import re

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.contrib.components.game_master import gm_working_memory as gm_working_memory_lib
from concordia.contrib.components.game_master import location_based_filter
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import concurrency
import termcolor


PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG


class NarrativeQualityValidator:
  """Validates narrative quality based on configurable metrics."""

  def __init__(
      self,
      min_dialogue_exchanges: int = 3,
      min_words: int = 400,
      required_player_names: list[str] | None = None,
  ):
    self._min_dialogue_exchanges = min_dialogue_exchanges
    self._min_words = min_words
    self._required_player_names = required_player_names or []

  def validate(self, narrative: str) -> tuple[bool, list[str]]:
    """Validate narrative quality.

    Args:
      narrative: The narrative text to validate.

    Returns:
      tuple of (is_valid, list of error messages)
    """
    errors = []

    dialogue_count = len(re.findall(r'"[^"]+"', narrative))
    if dialogue_count < self._min_dialogue_exchanges:
      errors.append(
          f"Insufficient dialogue: found {dialogue_count}, need"
          f" {self._min_dialogue_exchanges}"
      )

    word_count = len(narrative.split())
    if word_count < self._min_words:
      errors.append(
          f"Insufficient length: {word_count} words, need {self._min_words}"
      )

    if self._required_player_names:
      missing_players = [
          player_name
          for player_name in self._required_player_names
          if player_name not in narrative
      ]
      if missing_players:
        errors.append(f'Missing players: {", ".join(missing_players)}')

    return (not errors, errors)

  def get_retry_feedback(self, narrative: str, errors: list[str]) -> str:
    """Generate specific retry feedback based on validation errors."""
    feedback_parts = []

    dialogue_count = len(re.findall(r'"[^"]+"', narrative))
    word_count = len(narrative.split())

    for error in errors:
      if "Insufficient dialogue" in error:
        deficit = self._min_dialogue_exchanges - dialogue_count
        feedback_parts.append(
            f"Add {deficit} more quoted dialogue exchanges. Include"
            " back-and-forth conversations with specific quoted speech, not"
            ' summaries like "they discussed." Example: Alice said, "Where are'
            ' the keys?" Bob replied, "I left them on the counter."'
        )

      if "Insufficient length" in error:
        deficit = self._min_words - word_count
        feedback_parts.append(
            f"Add approximately {deficit} more words. Expand scenes with:"
            '\n  - More detailed physical actions (not "she organized files"'
            ' but "she picked up the red folder, opened the filing cabinet,'
            ' and placed it in the alphabetized section")'
            "\n  - Environmental observations (lighting, sounds, smells,"
            " textures)"
            "\n  - Character reactions and behavioral tells (facial"
            " expressions, body language, tone of voice)"
            "\n  - Specific consequences of actions (what changed in the"
            " environment, how others reacted)"
        )

      if "Missing players" in error:
        missing_match = re.search(r"Missing players: (.+)", error)
        if missing_match:
          missing_players_str = missing_match.group(1)
          feedback_parts.append(
              "The following players are not mentioned:"
              f" {missing_players_str}."
              "\n  You MUST add scenes that show what these players are"
              " doing."
              "\n  Create new scene headings with these players present and"
              " describe their specific actions, dialogue, and the"
              " consequences of what they do."
              "\n  Even if a player is alone, describe their activities in"
              " detail within a scene that includes them."
          )

    if not feedback_parts:
      return "No specific issues found."

    feedback = "**RETRY FEEDBACK - SPECIFIC IMPROVEMENTS NEEDED:**\n\n"
    feedback += "\n\n".join(feedback_parts)

    return feedback

  def get_quality_report(self, narrative: str) -> dict[str, int | float]:
    """Generate a quality report with metrics."""
    dialogue_count = len(re.findall(r'"[^"]+"', narrative))
    word_count = len(narrative.split())
    scene_count = len(re.findall(r"---\s*SCENE:", narrative))
    players_mentioned = sum(
        1 for player in self._required_player_names if player in narrative
    )
    return {
        "dialogue_count": dialogue_count,
        "word_count": word_count,
        "scene_count": scene_count,
        "players_mentioned": players_mentioned,
        "players_required": len(self._required_player_names),
        "dialogue_per_scene": (
            dialogue_count / scene_count if scene_count > 0 else 0
        ),
    }


class NarrativeHistoryManager(
    entity_component.ContextComponent,
):
  """A component that stores narrative history and provides summaries.

  Narratives are chunked by scene on entry.
  """

  def __init__(self):
    self._narratives = []
    super().__init__()

  def get_pre_act_value(self) -> str:
    """Return a summary of the narrative history."""
    return f"Narrative history contains {len(self._narratives)} entries."

  def _chunk_narrative_by_scene(self, narrative: str) -> list[dict[str, str]]:
    """Parses narrative into scenes."""
    scenes = []
    scene_starts = list(
        re.finditer(r"--- SCENE:(.*?)\n---\n", narrative, re.DOTALL)
    )
    for i, start in enumerate(scene_starts):
      scene_header = start.group(1).strip()
      scene_content_start = start.end()
      if i + 1 < len(scene_starts):
        scene_content_end = scene_starts[i + 1].start()
      else:
        scene_content_end = len(narrative)
      scene_content = narrative[scene_content_start:scene_content_end].strip()
      scenes.append({"header": scene_header, "content": scene_content})
    return scenes

  def add_narrative(self, time_str: str, narrative: str) -> None:
    """Add a narrative to the history, chunking it by scene."""
    scenes = self._chunk_narrative_by_scene(narrative)
    if not scenes:
      scenes = [{"header": "Full Narrative", "content": narrative}]
    self._narratives.append({"time": time_str, "scenes": scenes})

  def get_key_elements_for_gm_memory(
      self,
      time_str: str,
      narrative: str,
      model: language_model.LanguageModel,
  ) -> str:
    """Return a summary of the narrative for GM memory."""
    prompt = interactive_document.InteractiveDocument(model)
    prompt.statement(f"The following is a narrative of events:\n{narrative}")
    summary = prompt.open_question(
        "Summarize the key events, interactions, and outcomes of the above"
        " narrative in 2-4 sentences.",
        max_tokens=1000,
    )
    summary = re.sub(
        r"<thinking>.*?</thinking>", "", summary, flags=re.DOTALL
    ).strip()
    sample_match = re.search(r"<sample>(.*)</sample>", summary, re.DOTALL)
    if sample_match:
      summary = sample_match.group(1).strip()

    return f"{EVENT_TAG} Summary of events around {time_str}: {summary}"

  def get_narratives(self) -> list[dict[str, str | list[dict[str, str]]]]:
    """Return all structured narratives."""
    return self._narratives

  def get_state(self) -> entity_component.ComponentState:
    """Return component state."""
    return {"narratives": self._narratives}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    from typing import cast  # pylint: disable=g-import-not-at-top

    self._narratives = cast(
        list[dict[str, str | list[dict[str, str]]]], state.get("narratives", [])
    )


class SimultaneousNarrativeEventResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Resolves events by generating one master narrative, then deriving player observations."""

  _model: language_model.LanguageModel
  _player_names: Sequence[str]
  _memory_component_key: str
  _make_observation_component_key: str
  _world_state_key: str
  _generative_clock_key: str
  _components: Sequence[str]
  _pre_act_label: str
  _narrative: str
  _verbose: bool
  _time_period_minutes: int

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      make_observation_component_key: str = make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY,
      world_state_key: str = "world_state",
      generative_clock_key: str = "generative_clock",
      location_filter_key: str = "location_filter",
      narrative_history_key: str = "narrative_history",
      causal_state_tracker_key: str = "causal_state_tracker",
      ambient_environment_key: str = "ambient_environment",
      gm_working_memory_key: str = "gm_working_memory",
      components: Sequence[str] = (),
      pre_act_label: str = "Narrative Event",
      verbose: bool = False,
      time_period_minutes: int = 15,
      use_good_scene_example: bool = False,
      *args,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._model = model
    self._player_names = player_names
    self._memory_component_key = memory_component_key
    self._make_observation_component_key = make_observation_component_key
    self._world_state_key = world_state_key
    self._generative_clock_key = generative_clock_key
    self._location_filter_key = location_filter_key
    self._narrative_history_key = narrative_history_key
    self._causal_state_tracker_key = causal_state_tracker_key
    self._ambient_environment_key = ambient_environment_key
    self._gm_working_memory_key = gm_working_memory_key
    self._components = components
    self._pre_act_label = pre_act_label
    self._narrative = ""
    self._verbose = verbose
    self._time_period_minutes = time_period_minutes
    self._use_good_scene_example = use_good_scene_example

    # Initialize narrative quality validator
    self._quality_validator = NarrativeQualityValidator(
        min_dialogue_exchanges=max(3, time_period_minutes // 5),
        min_words=max(400, time_period_minutes * 25),
        required_player_names=list(player_names),
    )

  def _generate_event_id(
      self, scene_heading: str, timestamp_str: str, player_name: str
  ) -> str:
    """Generate a unique event ID from scene information.

    Args:
      scene_heading: The scene heading line from the narrative
      timestamp_str: The timestamp string for this narrative window
      player_name: The player this event is for

    Returns:
      A unique event ID string for tagging memories
    """
    # Try to extract location and time from scene heading
    # Format: "--- SCENE: Location Name (Time) ---"
    match = re.search(
        r"SCENE:\s*([^(]+)\(([^)]+)\)", scene_heading, re.IGNORECASE
    )
    if match:
      location = match.group(1).strip().lower().replace(" ", "_")
      time = match.group(2).strip().replace(":", "").replace(" ", "").lower()
      # Include player name for player-specific event grouping
      return f"{player_name.lower().replace(' ', '_')}_{location}_{time}"

    # Fallback: use timestamp hash
    timestamp_hash = hash(f"{player_name}_{timestamp_str}") % 100000
    return f"{player_name.lower().replace(' ', '_')}_event_{timestamp_hash}"

  def _extract_scenes_from_narrative(
      self, narrative: str, player_name: str
  ) -> list[dict[str, str]]:
    """Extract scene information from the narrative for a specific player.

    Args:
      narrative: The filtered narrative text
      player_name: The player to check for presence

    Returns:
      List of dicts with 'heading' and 'content' keys for scenes where player is
      present
    """
    scenes = []
    # Split narrative into scenes based on scene headings
    scene_pattern = r"---\s*SCENE:[^-]+---"
    parts = re.split(f"({scene_pattern})", narrative, flags=re.IGNORECASE)

    current_heading = None
    for part in parts:
      part = part.strip()
      if not part:
        continue

      if re.match(scene_pattern, part, re.IGNORECASE):
        current_heading = part
      elif current_heading:
        # Check if player is present in this scene's content part
        present_match = re.search(
            r"\*\*Present:\*\*\s*([^\n]+)", part, re.IGNORECASE
        )
        if not present_match:
          present_match = re.search(r"Present:\s*([^\n]+)", part, re.IGNORECASE)

        if present_match:
          present_players = present_match.group(1)
          if player_name in present_players:
            scenes.append({"heading": current_heading, "content": part})
        current_heading = None

    return scenes

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_named_component(
      self, component_name: str
  ) -> action_spec_ignored.ActionSpecIgnored:
    return self.get_entity().get_component(
        component_name, type_=action_spec_ignored.ActionSpecIgnored
    )

  def _component_pre_act_display(self, key: str) -> str:
    component = self.get_entity().get_component(
        key, type_=action_spec_ignored.ActionSpecIgnored
    )
    label = component.get_pre_act_label()
    if key == self._gm_working_memory_key:
      working_mem = self.get_entity().get_component(
          key, type_=gm_working_memory_lib.GMWorkingMemory
      )
      value = working_mem.get_narrative()
    else:
      value = component.get_pre_act_value()
    return f"{label}:\n{value}"

  def _identify_missing_players(
      self, narrative_text: str, player_names: Sequence[str]
  ) -> list[str]:
    """Helper to identify players not explicitly mentioned in the narrative."""
    player_names_str = ", ".join(player_names)
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        f"I have a list of players: {player_names_str}.\n"
        "I have a narrative:\n"
        "---\n"
        f"{narrative_text}\n"
        "---"
    )
    missing_players_str = prompt.open_question(
        f"Which of the players from the list [{player_names_str}] are NOT"
        " mentioned or do not seem to participate in the narrative?\nConsider"
        " that players might be referred to by nicknames or pronouns if it's"
        " clear who is being referred to.\nIf all players are mentioned or"
        " participate, answer 'None'.\nOtherwise, list only the names of"
        " players who are missing from the narrative, separated by commas.",
        max_tokens=1000,
        terminators=[],
    )
    missing_players_str = re.sub(
        r"<thinking>.*?</thinking>", "", missing_players_str, flags=re.DOTALL
    ).strip()
    sample_match = re.search(
        r"<sample>(.*)</sample>", missing_players_str, re.DOTALL
    )
    if sample_match:
      missing_players_str = sample_match.group(1).strip()

    if "none" in missing_players_str.lower():
      return []

    explicitly_missing = [
        p.strip() for p in missing_players_str.split(",") if p.strip()
    ]
    missing_players = []
    for player in player_names:
      if player in explicitly_missing:
        missing_players.append(player)
    return missing_players

  def _parse_observation_to_memories(
      self,
      observation_text: str,
      filtered_narrative: str,
      player_name: str,
      current_time_str: str,
  ) -> list[str]:
    """Converts observation into episodic memories using LLM for proper formatting."""
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        "Master Narrative (filtered for"
        f" {player_name}):\n{filtered_narrative}\n\nObservation for"
        f" {player_name}:\n{observation_text}"
    )

    episodic_memory_instructions = f"""
Convert the observation above into a list of STANDALONE EPISODIC MEMORIES for {player_name}.

**CRITICAL REQUIREMENTS FOR EACH MEMORY:**
1. **Standalone Format**: Each memory MUST be self-contained and retrievable via embedding search without context
2. **Standard Structure**: [At TIME], [AGENT] did [ACTION] and observed [CONSEQUENCE]
3. **TEMPORAL ACCURACY - ESSENTIAL**: Use the SPECIFIC TIME each event occurred from the narrative, NOT just the window start time
   - If narrative spans 8:00 AM to 8:30 AM, events at 8:05 should say "At 8:05 AM", events at 8:20 should say "At 8:20 AM"
   - This is CRITICAL for maintaining event sequencing during retrieval
   - Events that happen later MUST have later timestamps (8:20 comes after 8:10 which comes after 8:05)
4. **Dialogue Threading**: For conversations, use "In response to [PERSON] saying '[STATEMENT]', {player_name} said '[RESPONSE]'"
5. **Bidirectional**: IMPORTANT - For each dialogue exchange, create TWO memories:
   - One for {player_name} speaking (formatted as above)
   - One for {player_name} hearing the other person (formatted: "At [TIME], {player_name} heard [PERSON] say '[STATEMENT]'")
6. **Full Names Always**: Never use pronouns - always use full character names
7. **Linked Actions**: If a look/gesture and conversation are linked, combine into one memory
8. **Multi-turn Conversations**: For back-and-forth exchanges, you can include 2-3 turns in one memory if they form a coherent mini-episode

**EXAMPLES OF GOOD EPISODIC MEMORIES:**
(Note: These examples show events from an 8:00 AM - 8:30 AM time window with proper temporal progression)

Action with completion:
"At Monday, January 15, 2024 at 08:05 AM, Alex Chen organized the scattered documents on the desk and observed that all files were now neatly sorted by category, completing the task successfully."

Simple dialogue (create 2 memories):
Memory 1: "At Monday, January 15, 2024 at 08:10 AM, Agent A heard Agent B say 'We need to discuss the missing equipment. Have you seen anything unusual?'"
Memory 2: "At Monday, January 15, 2024 at 08:10 AM, in response to Agent B saying 'We need to discuss the missing equipment', Agent A made eye contact with Agent B and said 'I haven't noticed anything out of the ordinary.'"

Threaded conversation (one memory capturing the exchange):
"At Monday, January 15, 2024 at 08:12 AM, after Sarah Martinez asked 'Where were you last night?', David Kim hesitated before responding 'I was at home, why do you ask?' Sarah Martinez leaned forward and Sarah Martinez's expression became more serious."

Investigating/searching with discovery:
"At Monday, January 15, 2024 at 08:15 AM, Maria Santos opened the locked cabinet in the storage room and observed a hidden compartment containing documents marked 'Secret'."

Searching without finding:
"At Monday, January 15, 2024 at 08:18 AM, James Wilson searched through the entire workspace looking for the missing keycard but observed that it was not there."

Movement with destination:
"At Monday, January 15, 2024 at 08:00 AM, Elena Rodriguez left the main hall and traveled to the east wing laboratory, arriving at 08:05 AM."

Physical task with completion state:
"At Monday, January 15, 2024 at 08:20 AM, Robert Taylor assembled the device components and observed that the mechanism was now fully functional and operational, completing the assembly successfully."

Physical task with failure:
"At Monday, January 15, 2024 at 08:22 AM, Chen Wei attempted to repair the broken lock but observed that the lock remained jammed despite Chen Wei's efforts."

Watching/observing event:
"At Monday, January 15, 2024 at 08:25 AM, Aisha Patel watched as Thomas Green and Maya Johnson had a heated discussion near the entrance, and Aisha Patel observed that Thomas Green appeared defensive while Maya Johnson kept pointing at a document."

Critical dialogue for later retrieval:
"At Monday, January 15, 2024 at 08:28 AM, in response to Katherine Lee saying 'Drop the weapon now!', Marcus Brown said 'Not until you explain what happened to the data!' while holding the device."

Finding/discovering object:
"At Monday, January 15, 2024 at 08:08 AM, Yuki Tanaka found the missing research notes hidden behind the bookshelf and observed that they appeared to have been deliberately concealed."

Environmental observation:
"At Monday, January 15, 2024 at 08:00 AM, Sofia Morales observed that the facility was unusually quiet with minimal activity in the corridors."

**WHAT TO AVOID:**
- Vague: "Agent A did some work" ❌
- Missing context: "She said hello" ❌  
- Incomplete dialogue: Only one side of conversation when both should be captured ❌
- Missing consequences: "Agent A cleaned" without noting completion/result ❌

**YOUR TASK:**
Extract and reformat ALL significant events from the observation into standalone episodic memories.

**CRITICAL - UNIQUE SEQUENTIAL TIMESTAMPS**:
1. Use the scene headings for base times (e.g., "--- SCENE: Location (8:35 AM) ---")
2. **EVERY memory MUST have a UNIQUE timestamp** - no two memories should share the exact same time
3. For multiple events within the same scene, add SECONDS to differentiate (e.g., 8:35:00 AM, 8:35:20 AM, 8:35:40 AM)
4. Events that happen later in the narrative MUST have later timestamps
5. If a scene spans a time range (8:35 AM - 8:45 AM), spread events across that range (8:36 AM, 8:38 AM, 8:40 AM, etc.)

For each dialogue exchange, create memories for BOTH what {player_name} said AND what {player_name} heard.
Return one memory per line, with no bullet points or numbering.
If there are no significant events, return "No memories to extract."
"""

    raw_memories = prompt.open_question(
        episodic_memory_instructions, max_tokens=5000, terminators=[]
    )

    # Clean up LLM response
    raw_memories = re.sub(
        r"<thinking>.*?</thinking>", "", raw_memories, flags=re.DOTALL
    ).strip()
    sample_match = re.search(r"<sample>(.*)</sample>", raw_memories, re.DOTALL)
    if sample_match:
      raw_memories = sample_match.group(1).strip()

    if "no memories" in raw_memories.lower():
      return []

    # Parse memories (one per line)
    memories = []
    for line in raw_memories.splitlines():
      line = line.strip()
      # Remove bullet points or numbering if present
      line = re.sub(r"^[-*•]\s*", "", line)
      line = re.sub(r"^\d+\.\s*", "", line)
      if line and not line.startswith("#") and len(line) > 20:
        memories.append(line)

    # Add event tagging to memories (backward compatible feature)
    # Extract scenes where this player was present
    scenes = self._extract_scenes_from_narrative(
        filtered_narrative, player_name
    )

    if scenes:
      # Tag memories with event IDs based on scenes
      tagged_memories = []
      for memory in memories:
        # Try to match memory to scene based on timing
        # Use the first scene heading as default
        event_id = self._generate_event_id(
            scenes[0]["heading"],
            current_time_str,
            player_name,
        )

        # For sophisticated matching, check if memory timestamp matches scene
        for scene in scenes:
          # Extract time from scene heading if present
          scene_time_match = re.search(r"\(([^)]+)\)", scene["heading"])
          if scene_time_match:
            scene_time = scene_time_match.group(1)
            # Check if memory contains this time
            if scene_time in memory:
              event_id = self._generate_event_id(
                  scene["heading"], current_time_str, player_name
              )
              break

        # Prepend event tag to memory
        tagged_memory = f"[EVENT:{event_id}] {memory}"
        tagged_memories.append(tagged_memory)

      return tagged_memories
    else:
      # Fallback: No scenes found, use timestamp-based event ID
      fallback_event_id = self._generate_event_id(
          "--- SCENE: Unknown Location ---", current_time_str, player_name
      )
      return [f"[EVENT:{fallback_event_id}] {memory}" for memory in memories]

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ""

    # --- PREPARATION: Gather context and plans (ROBUST VERSION) ---
    if self._verbose:
      print(termcolor.colored("Starting simplified event resolution", "cyan"))

    current_time_str = self.get_named_component_pre_act_value(
        self._generative_clock_key
    ).strip()

    # Calculate end time for the prompts
    try:
      start_dt = datetime.datetime.strptime(
          current_time_str, "%A, %B %d, %Y at %I:%M %p"
      )
      end_dt = start_dt + datetime.timedelta(minutes=self._time_period_minutes)
      end_time_str = end_dt.strftime("%A, %B %d, %Y at %I:%M %p")
    except ValueError:
      end_time_str = f"{current_time_str} + {self._time_period_minutes} minutes"

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)

    if not suggestions:
      if self._verbose:
        print(
            termcolor.colored(
                "No putative events found in memory. This might be an issue"
                " with player agents saving their plans.",
                "yellow",
            )
        )
      return ""

    if self._verbose:
      print(
          termcolor.colored(
              f"Found {len(suggestions)} putative events. Parsing them now...",
              "cyan",
          )
      )

    latest_actions = {}
    for suggestion in suggestions:
      found_player = None
      for player_name in self._player_names:
        if player_name in suggestion:
          found_player = player_name
          break

      if found_player:
        try:
          action_start_index = suggestion.find(PUTATIVE_EVENT_TAG) + len(
              PUTATIVE_EVENT_TAG
          )
          action = suggestion[action_start_index:].strip()
          latest_actions[found_player] = f"{found_player}'s plan: {action}"
          if self._verbose:
            print(
                termcolor.colored(
                    f"Successfully parsed plan for {found_player}", "green"
                )
            )
        except (ValueError, TypeError) as e:
          if self._verbose:
            print(
                termcolor.colored(
                    f"Error parsing suggestion for {found_player}: {e}", "red"
                )
            )
            print(
                termcolor.colored(
                    f"Problematic suggestion: {suggestion}", "yellow"
                )
            )

    if not latest_actions:
      if self._verbose:
        print(
            termcolor.colored(
                "CRITICAL: Found putative events but failed to parse any plans."
                " Check player agent memory format and ensure the tag is"
                " present.",
                "red",
            )
        )
      return ""

    if self._verbose:
      print(termcolor.colored("Player plans:", "cyan"))
      for player, plan in latest_actions.items():
        print(termcolor.colored(f"  {player}: {plan}", "white"))

    component_states = "\n".join(
        [self._component_pre_act_display(key) for key in self._components]
    )

    # --- STEP 1: GENERATE THE MASTER NARRATIVE (WITH RETRY LOGIC) ---
    if self._verbose:
      print(termcolor.colored("STEP 1: Generating master narrative", "cyan"))

    min_words = max(400, self._time_period_minutes * 25)  # ~25 words per minute

    min_dialogue = max(
        3, self._time_period_minutes // 5
    )  # ~1 dialogue per 5 min
    # Initial Narrative Prompt (using the strengthened version)
    initial_narrative_instructions = f"""
**CRITICAL MASTER NARRATIVE GENERATION RULES - READ CAREFULLY!**

You are the narrator. Your primary goal is to generate a comprehensive, DETAILED, and COHESIVE narrative for the time period from {current_time_str} to {end_time_str} ({self._time_period_minutes} minutes).

**ABSOLUTELY ESSENTIAL: EVERY PLAYER MUST BE INCLUDED AND ACTIVELY DESCRIBED IN THE NARRATIVE.**
Based on the provided world state and player plans, describe **exactly what happens for ALL players**, ensuring each player's plan is incorporated into their actions.

**CRITICAL FORMATTING RULES - MUST FOLLOW EXACTLY:**
1.  **MANDATORY Scene Headings:** You MUST start EVERY scene with this EXACT format:
    
    --- SCENE: [Location Name] ([Time]) ---
    **Present:** [Player1], [Player2], [Player3]
    
    Example:
    --- SCENE: Store Front (9:05 AM) ---
    **Present:** Alice Pryant, Bob Johnson
    
2.  **ALL players MUST appear:** Every player must be listed as **Present:** in at least one scene.
3.  **Multiple scenes required:** Create separate scenes when players are in different locations or time advances.
4.  **No JSON:** Your entire output must be human-readable prose with scene headings.

**LENGTH AND DETAIL REQUIREMENTS:**
- Write AT LEAST {min_words} words for this {self._time_period_minutes}-minute narrative.
- **DIALOGUE QUOTA: Include AT LEAST {min_dialogue} SEPARATE QUOTED DIALOGUE EXCHANGES** (not just one person speaking, but back-and-forth conversations). Use standard quote marks ("). Examples:
  * Alice said, "Where did you put the keys?"
  * Bob replied, "I left them on the counter."
  * Alice frowned, "I don't see them there."
- For conversations, write out the FULL back and forth dialogue with quotes, not summaries like "they discussed the topic."
- Describe specific physical actions, not summaries (e.g., "Alice picked up the red folder and walked to the filing cabinet" not "Alice organized files"). Given the time period, players will do multiple things and mention them all.
- Include environmental details and character reactions.
- Resolve all events. Some plans may not be fully resolved, or may conflict with other plans or changes in the environment. Use your best judgment to resolve the conflicts.
- Things in the environment can change, include those, and introduce variation in the environment if needed to move the story along.
- **CRITICAL FOR OBSERVATIONS:** Ensure that details are included in the narrative that clearly show:
    *   What a player *tried* to do (their planned actions).
    *   What a player *actually* did (actions taken).
    *   The *consequences* or *immediate outcomes* of their actions.
    *   What *other players* did, their *consequences*, and how those were *perceived* by the observing player (not just their presence, but their specific actions and the results).
    *   If a repetitive task (like cleaning or stacking) is *completed* or *failed*, and the observable state reflects this.

**PACING AND TIME RULES:**
- The narrative covers EXACTLY from {current_time_str} to {end_time_str}.
- Every single minute of the period must be accounted for in the narrative flow. There should be no undocumented gaps.
- If a character performs a long task (e.g., "reading for 20 minutes"), mention their progress or thoughts at least every 5 minutes with specific details.

**GENERAL RULES:**
- Never use first person ("I", "my"). Always use player names.
- Be specific with locations and outcomes.
- Create a rich, detailed story that shows, not tells.
- **VERY IMPORTANT: All players must be mentioned and actively engaged in the narrative. Even if a player is alone, describe their activities in detail within a scene heading that includes their presence.**

**BEFORE YOU BEGIN WRITING THE NARRATIVE, REVIEW THE PLAYER PLANS AGAIN.**
**ENSURE THAT EVERY SINGLE PLAYER NAMED IN 'PLAYER PLANS' HAS THEIR INTENDED ACTIONS AND PRESENCE REFLECTED IN THE NARRATIVE.**
**IF A PLAYER'S PLAN INVOLVES A LONG TASK, BREAK IT DOWN INTO SMALLER, DETAILED ACTIONS SPREAD ACROSS THE {self._time_period_minutes} MINUTES.**

**CHAIN OF THOUGHT DOUBLE-CHECK (INTERNAL MONOLOGUE - DO NOT OUTPUT):**
1.  Have I included every player in at least one scene with active descriptions?
2.  For each player, have I explicitly shown their planned actions, the actions they actually took, and the immediate consequences or outcomes?
3.  Are there clear details about what other players did and the observable consequences of *those* actions?
4.  If a player attempted a repetitive task (e.g., cleaning, stacking), have I clearly indicated if it was completed or if the attempt failed and why, reflecting the current state of the environment?
5.  Does the narrative provide enough detail for an individual player to answer: "What was my plan?", "What did I do?", "What happened because of what I did?", "What did others do and what was the result?", "Is my task completed or not?"

Begin the narrative now. Start with the first scene heading. IMMEDIATELY AFTER THE SCENE HEADING, write a very detailed, multi-paragraph description of what happens in that scene, including all actions, dialogue, and observations. Do not stop after just the heading.
Write a separate multi-paragraph description for each scene heading.

IMPORTANT: This is likely to be a multiple page narrative with enough detail to build the story. Do not over simplify.
IMPORTANT: Players are providing very detailed plans, and you need to turn that into a story. All players must try to do their plans, and so **ALL PLAYERS MUST BE MENTIONED AND THEIR PLANNED ACTIONS DESCRIBED IN THE NARRATIVE.**
"""

    narrative_prompt = interactive_document.InteractiveDocument(self._model)
    narrative_prompt.statement(f"CONTEXT\nWorld State:\n{component_states}")
    narrative_prompt.statement(
        f"All players in simulation: {', '.join(self._player_names)}"
    )
    narrative_prompt.statement(
        "PLAYER PLANS (these are plans for the upcoming time period):"
    )
    for _, action in latest_actions.items():
      narrative_prompt.statement(f"  - {action}")
    if len(latest_actions) < len(self._player_names):
      narrative_prompt.statement(
          "Note: Not all players provided a plan. Players without a plan should"
          " be assumed to continue their previous activities or react to their"
          " environment."
      )

    # Add example of good scene detail
    example_good_scene = """
EXAMPLE OF MINIMAL SCENE DETAIL (15-Minute Narrative with Multiple Scenes. This is a minimal example of what is needed. More is better):

--- SCENE: Community Center Lobby (9:00 AM - 9:05 AM) ---
**Present:** Alex R., Beatrice C.

The morning chill still clung to the air outside, but inside the bustling Community Center lobby, a different kind of energy was building. Alex R., stationed at the reception desk, was diligently reviewing the day's event schedule on her computer, double-checking room bookings for the Zumba class. Her shift had just begun at 9:00 AM, and she was determined to get a head start. She sighed softly, noting a conflict for Room 3, where a yoga session was double-booked with a knitting circle. She made a mental note to call the knitting instructor first. The fluorescent lights overhead hummed a steady tune, and a faint aroma of stale coffee hung in the air from the previous night.

Meanwhile, Beatrice C. was meticulously arranging pamphlets on a display rack near the entrance, her cart filled with brochures for various local services. Her plan was to categorize and restock all informational materials by 9:30 AM, ensuring visitors could easily find what they needed. At 9:02 AM, she knelt to sort a stack of brightly colored flyers for an upcoming charity run, ensuring they were front and center. As she worked, she could faintly hear the distant clatter of Alex's keyboard and the occasional muffled ring of a phone. She straightened up, stretching her back, and noticed a discarded coffee cup left on a bench. She planned to address it after the pamphlet rack was perfect.

At approximately 9:04 AM, Alex finished her initial review. She picked up the phone to call the knitting instructor about the double-booking. After a brief conversation, she managed to reschedule the knitting circle to an alternate, smaller room. "Perfect," she murmured to herself, hanging up the phone with a soft click, a small victory for the start of her day.

--- SCENE: Community Center Maintenance Closet (9:03 AM - 9:08 AM) ---
**Present:** Charlie D.

Just a short walk from the main lobby, Charlie D. was wrestling with a small, persistent leak from a ceiling pipe inside the cramped maintenance closet. His goal was to assess the damage and stop the drip by 9:10 AM, before the first large group of seniors arrived for their morning activities. At 9:03 AM, he grunted, straining to tighten a pipe joint with a wrench. The drip, however, persisted, a slow but steady plink-plonk onto a small bucket he'd placed underneath, already a quarter full. "Stubborn thing," he muttered, wiping a bead of sweat from his brow with the back of his hand. He quickly grabbed a roll of heavy-duty duct tape from a nearby shelf, its adhesive scent strong in the small space, and began to wrap it around the joint, hoping for a temporary fix. He noted the time on his wristwatch – 9:06 AM – realizing this was taking longer than expected. By 9:08 AM, the dripping had slowed considerably, though a tiny seep remained. He decided it was the best he could do for now and closed his toolbox.

--- SCENE: Community Center Lobby (9:06 AM - 9:15 AM) ---
**Present:** Alex R., Beatrice C., (Charlie D. enters at 9:09 AM)

Back in the lobby, at 9:06 AM, a hurried-looking visitor rushed in, heading straight for the information desk. "Excuse me," the visitor asked curtly, "is the city council meeting still in Room 5?" Alex quickly consulted her screen. "Yes, it is, and it starts at 9:30 AM." She provided directions with a practiced smile, pointing towards the hallway. As the visitor hurried off, Beatrice, having completed the main pamphlet rack at 9:08 AM, moved to a smaller display table, carefully arranging brochures for local parks and recreation. She noticed a new visitor, an elderly woman with a cane, slowly making her way towards the reception desk. "Good morning!" Beatrice offered kindly, receiving a gentle smile in return.

At 9:09 AM, Charlie D. emerged from the maintenance closet, stretching his back as he re-entered the lobby. He scanned the area, noting the growing number of early arrivals. He walked over to Alex's desk. "Morning, Alex. The pipe's a temporary fix, but it'll need proper attention later," he reported quietly.

Alex looked up. "Thanks for the update, Charlie. Sounds like a fun start to your day. Anything else brewing?"

"Not yet," Charlie replied, "but I heard a tour bus pulling up outside just now. Get ready for the invasion."

Beatrice, overhearing, chuckled, "Always an adventure. I'm almost done with the park brochures, then I'll tackle the poster frames."

By 9:13 AM, Alex was already helping another visitor find the restrooms, her movements fluid and efficient. Beatrice had neatly organized the entire display table, the brochures looking full and inviting. Charlie walked towards the main doors, preparing to assist with the imminent influx, just as the main doors swung open, admitting the first wave of chattering tourists from the tour bus, their voices filling the lobby. The time was 9:15 AM.
"""
    if self._use_good_scene_example:
      narrative_prompt.statement(example_good_scene)

    self._narrative = narrative_prompt.open_question(
        initial_narrative_instructions, max_tokens=50000, terminators=[]
    )
    if self._verbose:
      print(termcolor.colored("Initial narrative generated:", "blue"))
      print(termcolor.colored(self._narrative, "blue"))

    # --- RETRY LOGIC FOR MASTER NARRATIVE ---
    max_narrative_retries = 2  # Limit retries to prevent infinite loops

    for retry_attempt in range(max_narrative_retries):
      missing_players = self._identify_missing_players(
          self._narrative, self._player_names
      )

      # Validate narrative quality
      is_valid, quality_errors = self._quality_validator.validate(
          self._narrative
      )

      if not missing_players and is_valid:
        if self._verbose:
          print(
              termcolor.colored(
                  "Master narrative generated successfully on attempt"
                  f" {retry_attempt + 1}. All players included, quality checks"
                  " passed.",
                  "green",
              )
          )
        break  # Exit retry loop if all players are found and quality is good

      # Collect all issues
      all_issues = []
      if missing_players:
        all_issues.append(f"Missing players: {', '.join(missing_players)}")
      if not is_valid:
        all_issues.extend(quality_errors)

      if self._verbose:
        issues_str = "; ".join(all_issues)
        print(
            termcolor.colored(
                f"Retry {retry_attempt + 1}: Issues detected: {issues_str}",
                "yellow",
            )
        )

      # Construct a retry prompt
      retry_prompt_doc = interactive_document.InteractiveDocument(self._model)
      retry_prompt_doc.statement(f"CONTEXT\nWorld State:\n{component_states}")
      retry_prompt_doc.statement(
          f"All players in simulation: {', '.join(self._player_names)}"
      )
      retry_prompt_doc.statement(
          "PLAYER PLANS (these are plans for the upcoming time period):"
      )
      for _, action in latest_actions.items():
        retry_prompt_doc.statement(f"  - {action}")
      if len(latest_actions) < len(self._player_names):
        retry_prompt_doc.statement(
            "Note: Not all players provided a plan. Players without a plan"
            " should be assumed to continue their previous activities or react"
            " to their environment."
        )

      retry_prompt_doc.statement(
          "\n---\n**Previous Master Narrative"
          f" (INCOMPLETE):**\n{self._narrative}\n---"
      )

      # Craft instructions to append/integrate missing player details
      retry_instructions = f"""
The previous narrative was incomplete. Some players and their actions were not adequately described.
Your task is to **REVISE AND EXPAND THE PROVIDED MASTER NARRATIVE** to specifically include detailed actions for the following players, integrating their plans from the 'PLAYER PLANS' section:
Missing Players: {', '.join(missing_players)}

**CRITICAL REVISION RULES:**
1.  **Do NOT rewrite the entire narrative.** Build upon the existing narrative provided after "Previous Master Narrative (INCOMPLETE):".
2.  **INTEGRATE missing player details seamlessly.** For each missing player, create new scene headings or expand existing ones (if the player's presence makes sense there) to describe their activities.
3.  **Ensure temporal accuracy:** The new details must fit logically within the timeline of {current_time_str} to {end_time_str}.
4.  **Maintain all previous formatting rules (scene headings, no JSON, detailed actions).**
5.  **Be specific about their actions and location, drawing directly from their plans.**
6.  **The final output should be a complete, revised narrative that includes ALL players and addresses their plans.**
7.  **The revised narrative should be at least {min_words} words and cover all {self._time_period_minutes} minutes.**
8.  **Ensure the narrative includes all necessary details for individual observations:** Clearly depict planned actions, actual actions, consequences, and what other players did/consequences, as well as task completion states (e.g., "the stack of boxes was now neat and complete," or "the spilled coffee remained, despite her cleaning attempt").

**CHAIN OF THOUGHT DOUBLE-CHECK (INTERNAL MONOLOGUE - DO NOT OUTPUT):**
1.  Have I now included all originally missing players with active descriptions?
2.  For these players, have I explicitly shown their planned actions, the actions they actually took, and the immediate consequences or outcomes?
3.  Are there clear details about what other players did and the observable consequences of *those* actions, for both original and newly added scenes?
4.  If a player attempted a repetitive task, have I clearly indicated if it was completed or failed, reflecting the current state?

Provide the **FULL, REVISED, AND COMPLETE NARRATIVE** now, incorporating details for all players.
"""
      self._narrative = retry_prompt_doc.open_question(
          retry_instructions, max_tokens=50000, terminators=[]
      )
      if self._verbose:
        print(
            termcolor.colored(
                f"Revised narrative generated on attempt {retry_attempt + 1}:",
                "yellow",
            )
        )
        print(termcolor.colored(self._narrative, "magenta"))

    # Final check after retries
    if self._identify_missing_players(self._narrative, self._player_names):
      if self._verbose:
        print(
            termcolor.colored(
                "WARNING: Even after retries, some players are still missing"
                " from the master narrative.",
                "red",
            )
        )
        print(
            termcolor.colored(
                "Remaining missing players:"
                f" {', '.join(self._identify_missing_players(self._narrative, self._player_names))}",
                "red",
            )
        )

    if self._verbose:
      print(termcolor.colored("Final Master narrative generated:", "green"))
      print(termcolor.colored(self._narrative, "magenta"))

    # --- STEP 2: GENERATE OBSERVATION FOR EACH PLAYER ---
    if self._verbose:
      print(termcolor.colored("STEP 2: Generating player observations", "cyan"))

    make_observation = self.get_entity().get_component(
        self._make_observation_component_key,
        type_=make_observation_component.MakeObservation,
    )

    # Define the function to generate observation for a single player
    def generate_player_observation(player_name: str) -> list[str]:
      """Generate observation for a single player (threadsafe)."""
      if self._verbose:
        print(termcolor.colored(f"  - Generating for {player_name}", "cyan"))

      observation_prompt = interactive_document.InteractiveDocument(self._model)

      # Get location-filtered narrative for this specific player
      if self._location_filter_key:
        location_filter = self.get_entity().get_component(
            self._location_filter_key,
            type_=location_based_filter.LocationBasedFilter,
        )
        filtered_narrative = location_filter.filter_narrative_for_entity(
            self._narrative, player_name
        )
      else:
        filtered_narrative = self._narrative

      observation_prompt.statement(
          f"**Master Narrative of Events (From {current_time_str} to"
          f" {end_time_str}):**\n{filtered_narrative}"
      )
      player_plan = latest_actions.get(
          player_name, f"{player_name}'s plan: [No plan provided]"
      )
      observation_prompt.statement(
          f"**Original Plan for {player_name}:**\n{player_plan}"
      )

      observation_instructions = f"""You are writing a personalized observation for {player_name}.
The texts above are the master narrative of everything that just happened and {player_name}'s original plan.

**Your Task:**
Based on the "Master Narrative of Events" and critically considering the "Original Plan for {player_name}", create a DETAILED observation of ONLY what {player_name} would have personally seen, heard, said, and done, including any environmental details {player_name} would have perceived.

**CRITICAL RULES:**
1.  **CRITICAL LOCATION CHECK:** The Master Narrative contains scene headings with "Present: [names]".
    - If {player_name} is NOT listed in a scene's "Present:" field, you MUST IGNORE that entire scene completely.
    - Do NOT infer, assume, or hallucinate {player_name}'s presence in scenes where they are not explicitly listed.
    - Only process scenes where {player_name} is explicitly listed as present in the heading.
2.  **COMPLETENESS - CAPTURE ALL PERCEIVABLE EVENTS:** Extract ALL events from scenes where {player_name} was present that {player_name} would realistically have perceived. Do not drop events or observations.
3.  **CRITICAL - REALISTIC OBSERVABILITY & NO MENTAL STATES:**
    - **You MUST NOT include the internal thoughts, feelings, motivations, or intentions of OTHER characters.** For example, instead of "{player_name} saw that Bob was anxious", write "{player_name} saw Bob wringing his hands and glancing around." Describe BEHAVIOR, not internal states.
    - **An agent cannot know *why* another agent did something, only *that* they did it and what the observable result was.**
    - **DESCRIBE ONLY BEHAVIOR, NOT INTERPRETATION:** Do not use adverbs or adjectives that interpret *why* an action was done (e.g., 'nervously,' 'anxiously,' 'angrily,' 'hurriedly'). Just describe the observable action itself.
        - BAD: "Alice fidgeted nervously." -> GOOD: "Alice fidgeted with her hands."
        - BAD: "Bob paced anxiously." -> GOOD: "Bob paced back and forth."
    - **Being present in a scene ≠ perceiving everything.** Use realistic judgment.
    - **Whispered/Quiet Conversations:** If {player_name} is not part of a quiet or whispered conversation, or is across the room, {player_name} CANNOT know what was said. The observation should be "{player_name} saw Alice and Bob whispering in the corner" or "{player_name} saw Maria and David talking quietly by the window", NOT the content of the conversation. Only include dialogue content if {player_name} is explicitly part of the conversation or the narrative states they overheard it.
    - **Out of View:** Actions happening behind {player_name}, in another part of a large room, or obscured from view should not be included unless the narrative indicates they were noticed (e.g., via sound).
    - **Focused Attention:** If {player_name} is concentrating on a task (e.g., reading, talking to someone), they might miss subtle events happening around them.
    - **Deliberate Concealment:** Actions described as subtle or concealed (e.g., "Bob subtly pocketed the key") should likely not be perceived by others unless they were watching Bob closely.
4.  **Perspective:** Write from a close third-person perspective (e.g., "{player_name} saw...", "{player_name} heard Bob say...", "{player_name} continued reading...").
5.  **Pronouns:** NEVER use pronouns when referring to players; always use their full names (e.g., '{player_name}', 'Alice Pryant').
6.  **Format:** You MUST structure your response using the EXACT headings provided below. If a section has no information, state 'None.' under that heading.

**Observation Format:**
    ```
    [TIME]: {current_time_str}

    [PLANNED ACTIONS SUMMARY for {player_name}]:
    - Summarize {player_name}'s "Original Plan" here, as if {player_name} is reflecting on what they intended to do.

    [ACTIONS TAKEN by {player_name}]:
    - Describe in detail *exactly* what {player_name} did during this time period, based on the Master Narrative.
    - If a task like 'cleaning' or 'stacking boxes' was completed, explicitly state that it is now **completed** and how {player_name} perceives this completion (e.g., "The messy stack of boxes was now neatly organized, a task {player_name} completed successfully." or "The kitchen counter gleamed, her cleaning efforts finished.").
    - If {player_name} attempted a task but it failed or was interrupted, state that (e.g., "He tried to fix the leak, but it continued to drip, the repair unsuccessful.").

    [CONSEQUENCES OF {player_name}'s ACTIONS]:
    - Describe the direct and immediate results or impacts of {player_name}'s own actions on the environment or other characters, as observed by {player_name}.
    - Include any changes to the environment or objects {player_name} interacted with.

    [PERCEIVED ACTIONS AND CONSEQUENCES OF OTHERS]:
    - For any other players {player_name} perceived (either directly or by observing their impact on the environment), describe *what they did* and *what the consequence of their action was*.
    - Be realistic about what {player_name} could actually perceive given their position, attention, and the nature of the action (see rule 3 above).
    - Examples of partial perception: "Alice and Bob had a whispered conversation that {player_name} could not overhear", "Maria made a subtle gesture that {player_name} did not notice while focused on the paperwork"
    - Do NOT simply state their presence or absence. State specific actions and their outcomes when perceivable (e.g., "Alice was seen organizing files, and the report she needed was now neatly categorized.").
    - If another player attempted a task (e.g., cleaning) and it was completed or failed, note that observable outcome (e.g., "The floor where John had been mopping was now visibly clean and sparkling.").
    - If {player_name} did not perceive any specific actions or consequences of others, state 'None.'.

    [ENVIRONMENTAL OBSERVATIONS]:
    - Describe any general changes or details in the environment that {player_name} would have observed, not directly related to their own or others' specific actions, but simply the state of the world.
    ```

**CHAIN OF THOUGHT DOUBLE-CHECK (INTERNAL MONOLOGUE - DO NOT OUTPUT):**
1.  Have I used ONLY information from the "Master Narrative of Events" for {player_name}'s perspective?
2.  Is the observation formatted EXACTLY with the requested sections: [TIME], [PLANNED ACTIONS SUMMARY], [ACTIONS TAKEN], [CONSEQUENCES OF {player_name}'s ACTIONS], [PERCEIVED ACTIONS AND CONSEQUENCES OF OTHERS], [ENVIRONMENTAL OBSERVATIONS]?
3.  For [PLANNED ACTIONS SUMMARY], have I accurately summarized {player_name}'s original plan?
4.  For [ACTIONS TAKEN], have I described {player_name}'s specific actions and clearly stated if repetitive tasks were **completed** or **failed**?
5.  For [CONSEQUENCES], have I focused on the direct outcomes of {player_name}'s *own* actions?
6.  For [PERCEIVED ACTIONS AND CONSEQUENCES OF OTHERS]:
    a. Have I captured ALL events from scenes where {player_name} was present that they would realistically perceive (completeness)?
    b. Have I been realistic about observability (whispering, distance, concealment)?
    c. Have I described *what others did* and *the consequence of their action*, rather than just their presence?
7.  Have I included relevant [ENVIRONMENTAL OBSERVATIONS]?
8.  Have I stated 'None.' if a section has no information?

Begin the personalized observation for {player_name} now, using the exact format specified above.
"""
      try:
        player_observation_str_raw = observation_prompt.open_question(
            observation_instructions, max_tokens=15000, terminators=[]
        )

        # Remove thinking tags
        player_observation_str_raw = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            player_observation_str_raw,
            flags=re.DOTALL,
        ).strip()

        # Remove sample tags and keep content
        sample_match = re.search(
            r"<sample>(.*)</sample>", player_observation_str_raw, re.DOTALL
        )
        if sample_match:
          player_observation_str_raw = sample_match.group(1).strip()

        # Observation Filtering Step
        filter_prompt = interactive_document.InteractiveDocument(self._model)
        filter_instructions = f"""You are an observation filter. Your job is to ensure that an agent's observation ONLY contains information they could realistically see, hear, or deduce from observable actions, based on a master narrative.

You MUST remove or rephrase any information that violates realistic observability, such as:
- The internal thoughts, feelings, motivations, or intentions of other characters.
- The content of conversations the agent could not realistically overhear (e.g., whispers across a room, quiet conversations between others).
- Events the agent could not have seen or heard due to their location, focus of attention, or the subtlety of the action.

**Master Narrative Context:**
---
{filtered_narrative}
---

**Agent Observation to Filter:**
---
{player_observation_str_raw}
---

**CRITICAL FILTERING RULES:**
1.  **REMOVE MENTAL STATES & INTERPRETATIONS:** Eliminate all descriptions of other characters' internal states (thoughts, feelings, motivations) AND any adverbs/adjectives that interpret behavior (e.g., 'nervously', 'angrily'). Rephrase to describe observable behavior ONLY.
    - BAD: "Bob was angry." -> GOOD: "Bob clenched his fists and his face turned red."
    - BAD: "Alice wanted to find the key." -> GOOD: "Alice began searching the desk."
    - BAD: "She fidgeted nervously." -> GOOD: "She fidgeted with her hands."
    - BAD: "He slammed the door angrily." -> GOOD: "He slammed the door."
2.  **REMOVE UNHEARD DIALOGUE:** If the observation contains dialogue from others that {player_name} could not have heard (e.g., whispering, private chat), replace the dialogue content with a behavioral description.
    - BAD: "Across the room, Alice whispered to Bob, 'The key is in the drawer.'"
    - GOOD: "Across the room, Alice and Bob whispered to each other."
3.  **MAINTAIN FORMAT:** Preserve the section headings like [ACTIONS TAKEN...], [PERCEIVED ACTIONS...], etc.
4.  **DO NOT ADD NEW INFO:** Only remove or rephrase content from the provided observation based on the master narrative and observability rules.

Now, rewrite the observation for {player_name}, filtering out any information {player_name} could not have perceived.
"""
        filter_prompt.statement(
            f"Master Narrative Context:\n---\n{filtered_narrative}\n---"
        )
        filter_prompt.statement(
            "Agent Observation to Filter for"
            f" {player_name}:\n---\n{player_observation_str_raw}\n---"
        )

        player_observation_str = filter_prompt.open_question(
            filter_instructions, max_tokens=15000, terminators=[]
        )
        player_observation_str = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            player_observation_str,
            flags=re.DOTALL,
        ).strip()
        sample_match = re.search(
            r"<sample>(.*)</sample>", player_observation_str, re.DOTALL
        )
        if sample_match:
          player_observation_str = sample_match.group(1).strip()

        memories = self._parse_observation_to_memories(
            player_observation_str,
            filtered_narrative,
            player_name,
            current_time_str,
        )

        if not memories:
          # Fallback to full observation if parsing yields no memories
          if self._verbose:
            print(
                termcolor.colored(
                    f"    Could not parse memories for {player_name}, using"
                    " full observation.",
                    "yellow",
                )
            )
          return [f"[observation]\n{player_observation_str.strip()}"]
        else:
          if self._verbose:
            print(
                termcolor.colored(
                    f"    Generated {len(memories)} memories for {player_name}",
                    "green",
                )
            )
          return memories
      except (RuntimeError, ValueError, AttributeError, re.error) as e:
        # If LLM call fails, provide a minimal observation
        error_observation = (
            f"[TIME]: {current_time_str}\n"
            f"[ERROR]: Failed to generate observation for {player_name}: {e}"
        )
        return [f"[observation]\n{error_observation.strip()}"]

    # Execute parallel observation generation using Concordia's concurrency
    observation_tasks = {
        player_name: functools.partial(generate_player_observation, player_name)
        for player_name in self._player_names
    }

    observations_dict = concurrency.run_tasks(
        observation_tasks, max_workers=len(self._player_names)
    )

    # Queue all observations
    for player_name, observation_list in observations_dict.items():
      if self._verbose:
        print(termcolor.colored(f"Observations for {player_name}:", "yellow"))
      for observation in observation_list:
        if self._verbose:
          print(termcolor.colored(observation, "yellow"))
        make_observation.add_to_queue(player_name, observation)
        memory.add(f"{player_name} observation: {observation}")
      if self._verbose:
        print(
            termcolor.colored(
                f"    Queued {len(observation_list)} entries for {player_name}",
                "green",
            )
        )

    # --- STEP 3: Recording narrative history for checkpoint/debugging ---
    if self._verbose:
      print(termcolor.colored("STEP 3: Recording narrative history", "cyan"))

    if self._narrative_history_key:
      narrative_history = self.get_entity().get_component(
          self._narrative_history_key,
          type_=NarrativeHistoryManager,
      )

      # Add full narrative to the history manager for checkpoint/debugging
      # Note: GM memory already receives all player observations via
      # ObservationToMemory, so no need to add a summary here.
      narrative_history.add_narrative(current_time_str, self._narrative)

    # Update world state with narrative
    world_state = self.get_entity().get_component(
        self._world_state_key, type_=entity_component.ContextComponent
    )
    world_state.post_act(self._narrative)

    # Also update other components that process narrative in post_act
    if self._location_filter_key:
      try:
        self.get_entity().get_component(
            self._location_filter_key, type_=entity_component.ContextComponent
        ).post_act(self._narrative)
      except (KeyError, ValueError):
        pass
    if self._causal_state_tracker_key:
      try:
        self.get_entity().get_component(
            self._causal_state_tracker_key,
            type_=entity_component.ContextComponent,
        ).post_act(self._narrative)
      except (KeyError, ValueError):
        pass
    if self._ambient_environment_key:
      try:
        self.get_entity().get_component(
            self._ambient_environment_key,
            type_=entity_component.ContextComponent,
        ).post_act(self._narrative)
      except (KeyError, ValueError):
        pass

    # Advance the generative clock at the end of pre_act
    generative_clock = self.get_entity().get_component(
        self._generative_clock_key,
    )
    generative_clock.advance_by_minutes(self._time_period_minutes)

    result_message = (
        f"Narrative for {self._time_period_minutes}m (from {current_time_str})"
        " resolved and observations distributed. Clock advanced."
    )
    self._log(key=self._pre_act_label, value=result_message, prompt="")
    return result_message

  def get_narrative(self) -> str:
    return self._narrative

  def get_state(self) -> entity_component.ComponentState:
    return {"_narrative": self._narrative}

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._narrative = str(state.get("_narrative", ""))

  def _log(self, key: str, value: str, prompt: str):
    self._logging_channel({
        "Key": key,
        "Summary": value,
        "Value": value,
        "Prompt": prompt,
    })
