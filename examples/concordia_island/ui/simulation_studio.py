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

r"""Concordia Simulation Studio & Unified Multi-Scale Map Dashboard.

Single-process command center for Concordia multi-agent simulations:
  0. Multi-Scale Cartographic Map Dashboard (mounted natively at /map)
  1. Cognitive & Memory Stream Inspector (categorized memories, profiles)
  2. Interactive Launch Configurator & Command Generator
  3. GM / Agent Component Wiring & Live Agent Interview Mode

Usage:
  python -m examples.concordia_island.ui.simulation_studio \
    --run_dir=examples/concordia_island/data/paper_runs/job_loss/gemini_2_5_flash_esa \
    --geography=concordia_island \
    --port=9090
"""

# pylint: disable=broad-exception-caught,g-inconsistent-quotes

import datetime
import http.server
import json
import mimetypes
import os
import re
import socketserver
import sys
import threading
import time
import traceback
from typing import Any
import urllib.parse

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
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

# pylint: disable=g-import-not-at-top,g-bad-import-order
from absl import app
from absl import flags
from absl import logging
from concordia.contrib import language_models as contrib_language_models
from examples.concordia_island import mock_language_model
from examples.concordia_island.personas import generator as persona_generator
from examples.concordia_island.ui import map_dashboard
from examples.concordia_island.ui import map_data
# pylint: enable=g-import-not-at-top,g-bad-import-order

FLAGS = flags.FLAGS

if "id" not in flags.FLAGS:
  _RUN_ID = flags.DEFINE_string(
      "id", "", "Run ID to inspect (looked up under ./local_runs/)."
  )
else:
  _RUN_ID = flags.FLAGS["id"]

if "run_dir" not in flags.FLAGS:
  _RUN_DIR = flags.DEFINE_string(
      "run_dir",
      "",
      "Path to local simulation run directory (defaults to the bundled"
      " data/paper_runs/job_loss/gemini_2_5_flash_esa run).",
  )
else:
  _RUN_DIR = flags.FLAGS["run_dir"]

if "personas_dir" not in flags.FLAGS:
  _PERSONAS_DIR = flags.DEFINE_string(
      "personas_dir", "", "Path to personas directory or JSON bundle."
  )
else:
  _PERSONAS_DIR = flags.FLAGS["personas_dir"]

if "geography" not in flags.FLAGS:
  _GEOGRAPHY = flags.DEFINE_string(
      "geography",
      "",
      "Atlas id to render in the Map tab (brecksville, concordia_island,"
      " kerala). Empty picks one from --personas_dir or the run directory.",
  )
else:
  _GEOGRAPHY = flags.FLAGS["geography"]

if "port" not in flags.FLAGS:
  _PORT = flags.DEFINE_integer("port", 9090, "HTTP port for the Studio.")
else:
  _PORT = flags.FLAGS["port"]

if "expected_agents" not in flags.FLAGS:
  _EXPECTED_AGENTS = flags.DEFINE_integer(
      "expected_agents", 100, "Expected number of agents."
  )
else:
  _EXPECTED_AGENTS = flags.FLAGS["expected_agents"]

if "expected_ticks" not in flags.FLAGS:
  _EXPECTED_TICKS = flags.DEFINE_integer(
      "expected_ticks", 100, "Expected number of ticks."
  )
else:
  _EXPECTED_TICKS = flags.FLAGS["expected_ticks"]

if "start_time" not in flags.FLAGS:
  _START_TIME = flags.DEFINE_string(
      "start_time", "Thursday, January 1st, 7:00 AM", "Starting sim time."
  )
else:
  _START_TIME = flags.FLAGS["start_time"]

if "tick_interval" not in flags.FLAGS:
  _TICK_INTERVAL = flags.DEFINE_integer(
      "tick_interval", 120, "Minutes per simulation tick."
  )
else:
  _TICK_INTERVAL = flags.FLAGS["tick_interval"]

if "api_type" not in flags.FLAGS:
  _API_TYPE = flags.DEFINE_string(
      "api_type",
      "google_aistudio",
      "Concordia language_model_setup api_type for live agent interviews.",
  )
else:
  _API_TYPE = flags.FLAGS["api_type"]

_INTERVIEW_MODEL = flags.DEFINE_string(
    "interview_model",
    "gemini-2.5-flash",
    "Model name for live agent interview mode.",
)
if "api_key" not in flags.FLAGS:
  _API_KEY = flags.DEFINE_string(
      "api_key",
      "",
      "Optional API key for live agent interviews (or set $GOOGLE_API_KEY).",
  )
else:
  _API_KEY = flags.FLAGS["api_key"]
_USE_MOCK_INTERVIEW = flags.DEFINE_bool(
    "use_mock_interview",
    False,
    "Explicitly use MockLanguageModel for offline interview testing.",
)
_INTERVIEW_AGENT = flags.DEFINE_string(
    "interview_agent",
    "Olivia Welch",
    "Pre-selected agent for interview mode.",
)
_MAP_URL = flags.DEFINE_string(
    "map_url",
    "/map",
    "URL of the Map Dashboard tab (defaults to built-in /map).",
)


def _read_file(fpath: str) -> str:
  if not fpath or not os.path.isfile(fpath):
    return ""
  try:
    with open(fpath, "r", encoding="utf-8") as f:
      return f.read()
  except Exception:
    return ""


def _read_json(fpath: str) -> Any:
  if not fpath or not os.path.isfile(fpath):
    return None
  try:
    with open(fpath, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception as e:
    logging.warning("Failed to parse JSON for %s: %s", fpath, e)
    return None


def classify_memory(text: str) -> dict[str, Any]:
  """Classifies a memory into category, badge, color, and extracts metadata."""
  raw = text.strip()
  category = "life_event"
  badge = "Event"
  color = "#0284c7"
  icon = "📅"

  if raw.startswith("[formative]") or "psychological profile:" in raw.lower():
    category, badge, color, icon = "formative", "Formative", "#7c3aed", "🧠"
  elif raw.startswith("[self]") or raw.startswith("[background]"):
    category, badge, color, icon = "identity", "Identity", "#4f46e5", "👤"
  elif raw.startswith(("I am feeling", "Based on what I was feeling")):
    # Emotional-state appraisals written by the ESA agent's emotion
    # components; these are the agent's own reflections.
    category, badge, color, icon = "journal", "Journal", "#9333ea", "📓"
  elif "// marketplace [" in raw:
    # Only memories emitted by the marketplace game master carry this prefix.
    category, badge, color, icon = "marketplace", "Marketplace", "#16a34a", "🛒"
  elif "// bank [" in raw or any(
      k in raw
      for k in [
          "PAYROLL DEPOSIT:",
          "RENT PAYMENT:",
          "UBI DEPOSIT:",
          "RENT WARNING:",
      ]
  ):
    # Fiscal events are emitted by the bank with these exact tokens.
    category, badge, color, icon = "fiscal", "Fiscal", "#d97706", "💰"
  elif any(
      k in raw.lower()
      for k in [
          "you are having a conversation",
          "it is your turn to speak",
          "conversation with",
          '-- "',
      ]
  ):
    category, badge, color, icon = (
        "conversation",
        "Conversation",
        "#db2777",
        "💬",
    )
  elif "[journal]" in raw or "reflects:" in raw:
    category, badge, color, icon = "journal", "Journal", "#9333ea", "📓"

  ts_match = re.search(
      r"\[([A-Za-z]+(?:day)?,?\s+[A-Za-z]+\s+\d+(?:st|nd|rd|th)?,?"
      r"\s+\d+:\d+\s+[AP]M)\]",
      raw,
  )
  timestamp = ts_match.group(1) if ts_match else ""
  loc_match = re.match(r"//\s*([a-z_0-9]+)", raw)
  location = loc_match.group(1) if loc_match else ""

  return {
      "raw": raw,
      "category": category,
      "badge": badge,
      "color": color,
      "icon": icon,
      "timestamp": timestamp,
      "location": location,
  }


def _sort_memories_chronologically(
    memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  if not memories:
    return memories
  front = [
      m for m in memories if m.get("category") in ("identity", "formative")
  ]
  rest = [
      m for m in memories if m.get("category") not in ("identity", "formative")
  ]
  return front + rest


def get_run_data() -> dict[str, Any]:
  """Returns unified map and Studio simulation state from LocalFileSource."""
  src = map_dashboard.get_source()
  data = dict(src.data())
  data["start_time"] = _START_TIME.value
  data["tick_interval"] = _TICK_INTERVAL.value
  return data


def _active_run_id() -> str:
  """Returns the run id of the loaded run (the run directory's name)."""
  return str(getattr(map_dashboard.get_source(), "run_id", "") or "")


def _build_cognitive_payload() -> dict[str, Any]:
  """Constructs the complete structured payload for the cognitive debugger."""
  run_dir = _RUN_DIR.value or map_data.default_run_dir()
  personas_dir = _PERSONAS_DIR.value

  bundled_personas = {}
  try:
    bundled_personas = persona_generator.load_personas(
        cns_path=personas_dir or persona_generator.DEFAULT_PERSONAS_BASE_PATH,
        date_label="brecksville_ohio",
    )
  except Exception:
    pass

  payload: dict[str, Any] = {
      "agents": {},
      "agent_names": [],
      "metadata": {
          "run_dir": run_dir,
          "personas_dir": personas_dir or "",
      },
  }

  agent_names: list[str] = []
  sim_state_data = {}
  laid_off_agents = set()
  mugged_agents = set()
  states_data = {}
  structured_entity_memories = {}

  if run_dir and os.path.isdir(run_dir):
    loaded_names = _read_json(os.path.join(run_dir, "agent_names.json"))
    if isinstance(loaded_names, list):
      agent_names.extend(loaded_names)

    sim_state_data = (
        _read_json(os.path.join(run_dir, "simulation_state.json")) or {}
    )
    for name in (sim_state_data.get("locations") or {}).keys():
      if name not in agent_names:
        agent_names.append(name)

    lo_list = _read_json(os.path.join(run_dir, "laid_off_agents.json"))
    if isinstance(lo_list, list):
      laid_off_agents = set(lo_list)
    mugged_list = _read_json(os.path.join(run_dir, "mugged_agents.json"))
    mugged_agents = set(mugged_list) if isinstance(mugged_list, list) else set()

    states_data = _read_json(os.path.join(run_dir, "entity_states.json")) or {}
    for name in states_data.keys():
      if name not in agent_names:
        agent_names.append(name)

    fast_mems = _read_json(os.path.join(run_dir, "entity_memories.json"))
    if isinstance(fast_mems, dict) and fast_mems:
      structured_entity_memories = fast_mems
    else:
      struct_json = _read_json(
          os.path.join(run_dir, "simulation_structured.json")
      )
      if isinstance(struct_json, dict) and "entity_memories" in struct_json:
        structured_entity_memories = struct_json["entity_memories"]
      else:
        for fname in sorted(os.listdir(run_dir)):
          if fname.endswith("_memories.json"):
            mdata = _read_json(os.path.join(run_dir, fname))
            if (
                isinstance(mdata, dict)
                and isinstance(mdata.get("agent"), str)
                and isinstance(mdata.get("memories"), list)
            ):
              structured_entity_memories[mdata["agent"]] = mdata["memories"]

    for name in structured_entity_memories.keys():
      if name not in agent_names:
        agent_names.append(name)

  for name in agent_names:
    sdata = states_data.get(name, {})
    ctx = sdata.get("context_components", {}) if isinstance(sdata, dict) else {}
    instr = ctx.get("Instructions", {}).get("state", "")
    loc = ctx.get("LocationInfo", {}).get("state", "")
    sched = ctx.get("ScheduleAwareness", {})
    self_p = ctx.get("SelfPerception", {}).get("state", "")

    home_p = ""
    if "lives at " in loc:
      home_p = loc.split("lives at ")[1].split(".")[0].strip()

    persona_info: dict[str, Any] = {
        "name": name,
        "home_place": home_p,
        "work_place": (
            sched.get("work_place", "") if isinstance(sched, dict) else ""
        ),
        "instructions": instr,
        "self_perception": self_p,
        "traits": {},
    }
    if name in bundled_personas:
      p_obj = bundled_personas[name]
      for k, v in p_obj.to_dict().items():
        if v or k not in persona_info or not persona_info[k]:
          persona_info[k] = v

    if laid_off_agents:
      persona_info["economic_class"] = (
          "Laid-Off (AI Displacement)"
          if name in laid_off_agents
          else persona_info.get("economic_class") or "Control (Employed)"
      )
    if mugged_agents:
      persona_info["shock_status"] = (
          "Mugged (phone and wallet stolen)"
          if name in mugged_agents
          else "Control (not mugged)"
      )

    raw_mems = structured_entity_memories.get(name, [])
    classified_mems = [classify_memory(str(m)) for m in raw_mems]
    if not classified_mems and persona_info.get("formative_memories"):
      classified_mems = [
          classify_memory(
              f"[formative] {m}" if not m.startswith("[formative]") else m
          )
          for m in persona_info.get("formative_memories", [])
      ]
    classified_mems = _sort_memories_chronologically(classified_mems)
    payload["agents"][name] = {
        "name": name,
        "memories": classified_mems,
        "memory_count": len(classified_mems),
        "persona": persona_info,
        "entity_state": sdata,
    }

  payload["agent_names"] = sorted(set(agent_names))
  return payload


_cognitive_lock = threading.Lock()
_cognitive_cached = None
_cognitive_time = 0.0
_COGNITIVE_TTL = 60


def get_cognitive_data() -> dict[str, Any]:
  """Returns cached or freshly built cognitive debugger payload."""
  global _cognitive_cached, _cognitive_time
  now = time.time()
  with _cognitive_lock:
    if _cognitive_cached and (now - _cognitive_time) < _COGNITIVE_TTL:
      return _cognitive_cached
  data = _build_cognitive_payload()
  with _cognitive_lock:
    _cognitive_cached = data
    _cognitive_time = time.time()
  return data


_interview_model_lock = threading.Lock()
_interview_model_instance = None
_interview_model_init_error = ""


def _get_interview_model():
  """Lazily initializes and returns the language model for live interviews."""
  global _interview_model_instance, _interview_model_init_error
  with _interview_model_lock:
    if _interview_model_instance is not None:
      return _interview_model_instance
    if _USE_MOCK_INTERVIEW.value:
      _interview_model_instance = mock_language_model.MockLanguageModel()
      return _interview_model_instance
    try:
      api_key = (
          _API_KEY.value
          or os.environ.get("GOOGLE_API_KEY", "")
          or os.environ.get("GEMINI_API_KEY", "")
          or os.environ.get("OPENAI_API_KEY", "")
          or None
      )
      _interview_model_instance = contrib_language_models.language_model_setup(
          api_type=_API_TYPE.value,
          model_name=_INTERVIEW_MODEL.value,
          api_key=api_key,
      )
      _interview_model_init_error = ""
    except Exception as e:
      _interview_model_init_error = f"{type(e).__name__}: {e}"
      logging.error("Failed to initialize interview model: %s", e)
      _interview_model_instance = None
    return _interview_model_instance


def _extract_json_response(raw_text: str) -> dict[str, Any]:
  """Extracts JSON or parses structured CoT from model response text."""
  cleaned = raw_text.strip()
  if '```json' in cleaned:
    m = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
    if m:
      cleaned = m.group(1).strip()
  elif '```' in cleaned:
    m = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
    if m:
      cleaned = m.group(1).strip()
  try:
    data = json.loads(cleaned)
    if isinstance(data, dict):
      return data
  except Exception:
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end > start:
      try:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
          return data
      except Exception:
        pass

  # If not valid JSON, parse labeled sections
  res = {}
  sit = re.search(
      r'(?:1\.?\s*)?(?:SituationPerception|Situation'
      r' Perception)[:\s]+(.*?)(?=(?:2\.?\s*)?SelfPerception|$)',
      raw_text,
      re.DOTALL | re.IGNORECASE,
  )
  if sit:
    res['situation_perception'] = sit.group(1).strip()

  self_p = re.search(
      r'(?:2\.?\s*)?(?:SelfPerception|Self'
      r' Perception)[:\s]+(.*?)(?=(?:3\.?\s*)?PersonBySituation|$)',
      raw_text,
      re.DOTALL | re.IGNORECASE,
  )
  if self_p:
    res['self_perception'] = self_p.group(1).strip()

  p_by_s = re.search(
      r'(?:3\.?\s*)?(?:PersonBySituation|Person by'
      r' Situation)[:\s]+(.*?)(?=(?:4\.?\s*)?(?:Interview'
      r' Response|Response|Spoken Response|\w+:)|$)',
      raw_text,
      re.DOTALL | re.IGNORECASE,
  )
  if p_by_s:
    res['person_by_situation'] = p_by_s.group(1).strip()

  resp = re.search(
      r'(?:4\.?\s*)?(?:Interview Response|Response|Spoken'
      r' Response)[:\s]+["\']?(.*?)["\']?$',
      raw_text,
      re.DOTALL | re.IGNORECASE,
  )
  if resp:
    res['interview_response'] = resp.group(1).strip()
  elif not res:
    res['interview_response'] = raw_text.strip()

  return res


def _execute_interview_turn(
    agent_name: str,
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
    cutoff_index: int | None = None,
) -> dict[str, Any]:
  """Replicates Concordia component pipeline and ConcatActComponent for interview."""
  conversation_history = conversation_history or []
  cog = get_cognitive_data()
  agents = cog.get('agents', {})
  agent_data = agents.get(agent_name)
  if not agent_data:
    return {'error': f'Agent "{agent_name}" not found in current simulation.'}

  persona = agent_data.get('persona', {})
  all_memories = agent_data.get('memories', [])
  total_memories = len(all_memories)

  # Apply memory cutoff if specified
  if cutoff_index is not None and 0 <= cutoff_index < total_memories:
    selected_mems = all_memories[: cutoff_index + 1]
    actual_cutoff = cutoff_index
  else:
    selected_mems = all_memories
    actual_cutoff = total_memories - 1 if total_memories > 0 else 0

  last_obs = (
      selected_mems[-1].get('raw', selected_mems[-1].get('text', ''))
      if selected_mems
      else ''
  )

  # Determine current location dynamically from last observation prefix
  # e.g., "[observation] // lighthouse [Saturday, January 10th, 9:00 PM]: ..."
  current_location = persona.get('home_place', 'town_square')
  for m in reversed(selected_mems):
    mtext = m.get('raw', m.get('text', ''))
    match = re.search(r'//\s*([a-z_0-9]+)[\s\[:]', mtext)
    if match:
      current_location = match.group(1)
      break

  # Separate formative memories from recent endogenous memories
  formative_mems = [
      m.get('raw', m.get('text', ''))
      for m in selected_mems
      if m.get('category') in ('formative', 'identity')
  ]
  endogenous_mems = [
      m.get('raw', m.get('text', ''))
      for m in selected_mems
      if m.get('category') not in ('formative', 'identity')
  ]

  # Deduplicate consecutive repeated memories to remove looping artifacts
  deduped_endogenous = []
  for m in endogenous_mems:
    if not deduped_endogenous or deduped_endogenous[-1] != m:
      clean_m = m.strip()
      if len(clean_m) > 280:
        clean_m = clean_m[:280] + '...'
      deduped_endogenous.append(clean_m)
  recent_slice = (
      deduped_endogenous[-18:]
      if len(deduped_endogenous) > 18
      else deduped_endogenous
  )

  # Clean formative memories
  clean_formative = []
  for m in formative_mems[:8]:
    clean_m = m.strip()
    if len(clean_m) > 300:
      clean_m = clean_m[:300] + '...'
    clean_formative.append(clean_m)

  # 1. Component: Instructions (exact Concordia pre_act format)
  personality = persona.get('personality', '')
  backstory = persona.get('backstory', '')
  age = persona.get('age', '')
  rel_status = persona.get('relationship_status', '')
  traits = persona.get('traits', {})

  # Extract fallback persona traits from [self] / [background] if not in persona
  if not personality:
    for m in formative_mems:
      if f'{agent_name} is ' in m and 'years old' not in m:
        personality = m.split(f'{agent_name} is ', 1)[1].rstrip('.')
        break
  if not age:
    for m in formative_mems:
      m_age = re.search(r'(\d+)\s+years old', m)
      if m_age:
        age = m_age.group(1)
        break
  if not backstory:
    for m in formative_mems:
      if m.startswith('[background]'):
        backstory = m[len('[background]') :].strip()
        break
  traits_lines = []
  if age:
    traits_lines.append(f'{agent_name} is {age} years old.')
  if personality:
    traits_lines.append(f'{agent_name} is {personality}.')
  if rel_status:
    traits_lines.append(f'Family / relationship: {rel_status}.')
  if backstory:
    traits_lines.append(f'Background: {backstory}')
  if traits and isinstance(traits, dict):
    traits_lines.append(
        'Psychological profile: '
        + ', '.join(f'{k}: {v}' for k, v in traits.items())
    )
  if not traits_lines:
    traits_lines.append(f'{agent_name} is an island resident.')
  instructions_val = '\n'.join(traits_lines)

  # 2. Component: LocationInfo (exact Concordia pre_act format)
  home_place = persona.get('home_place', 'chippewa_ridge_unit_93')
  work_place = persona.get('work_place', 'community_center')
  loc_lines = [f'{agent_name} lives at {home_place}.']
  if work_place:
    loc_lines.append(
        f'They work at {work_place}. Working hours are 9:00 AM to 5:00 PM,'
        ' Monday through Friday.'
    )
  else:
    loc_lines.append('Currently not employed at a standard workplace.')
  location_info_val = ' '.join(loc_lines)

  # 3. Component: ScheduleAwareness
  schedule_val = (
      f'General schedule: Moves between home ({home_place}), community spaces,'
      ' and island spots. Evenings are spent at home, social areas, or'
      f' reflective places like {current_location}.'
  )

  # 4. Component: Recent events (Observations)
  events_parts = []
  if clean_formative:
    events_parts.append('--- Formative Memories & Identity ---')
    events_parts.extend(clean_formative)
  events_parts.append('--- Recent Observations & Island Experiences ---')
  events_parts.extend(recent_slice)
  recent_events_val = '\n'.join(events_parts)

  # 5. Component: CurrentLocation
  current_location_val = current_location

  # Assemble the base context string (Components 1 through 5)
  base_sections = [
      f"{agent_name}'s core traits:\n{instructions_val}",
      f'Location information:\n{location_info_val}',
      f"{agent_name}'s current schedule:\n{schedule_val}",
      (
          'Recent events (ordered from least recent to most recent):\n'
          f'{recent_events_val}'
      ),
      f"{agent_name}'s current location:\n{current_location_val}",
  ]
  base_context_str = '\n\n'.join(base_sections)

  # Prepare conversation history for interview
  conv_turns_text = ''
  if conversation_history:
    conv_lines = ['Interview history so far:']
    for turn in conversation_history[-6:]:
      r = 'Researcher' if turn.get('role') == 'user' else agent_name
      conv_lines.append(f'{r}: "{turn.get("text", "")}"')
    conv_turns_text = '\n'.join(conv_lines) + '\n\n'

  # The Cognitive Chain-of-Thought prompt asking the component questions
  cot_prompt = f"""{base_context_str}

{conv_turns_text}A researcher is conducting a live, in-depth post-simulation interview with {agent_name}.
The researcher asks: "{user_message}"

You must simulate {agent_name}'s authentic Concordia cognitive components in order before answering:
1. SituationPerception: What situation is {agent_name} in right now? (1-2 sentences resolving current location, immediate events, and the interaction)
2. SelfPerception: What kind of person is {agent_name}? (1-2 sentences resolving core traits, values, and identity)
3. PersonBySituation: What would a person like {agent_name} do in a situation like this? (1-2 sentences on how they approach answering the researcher)
4. Final in-character verbal response to the researcher: {agent_name}'s authentic first-person ("I") spoken response directly answering the researcher's question while remaining strictly true to their life experiences and current state.

Respond with ONLY a valid JSON object in this exact schema:
{{
  "situation_perception": "{agent_name} is currently [1-2 sentences]",
  "self_perception": "{agent_name} is [1-2 sentences]",
  "person_by_situation": "{agent_name} would [1-2 sentences]",
  "interview_response": "[{agent_name}'s authentic first-person response to the researcher, 2-4 sentences]"
}}"""

  # Execute with the configured language model.
  model = _get_interview_model()
  if model is None:
    err_msg = (
        _interview_model_init_error
        or 'Interview model is not configured or unavailable.'
    )
    return {'error': f'Interview model offline: {err_msg}'}

  raw_output = ''
  model_call_error = ''
  is_simulated = False
  res_container = []
  err_container = []

  def _query_model():
    try:
      logging.info(
          'Sending interview prompt to %s (%d chars)...',
          _INTERVIEW_MODEL.value,
          len(cot_prompt),
      )
      out = model.sample_text(cot_prompt, max_tokens=350, temperature=0.6)
      logging.info('Received model response (%d chars)', len(out))
      res_container.append(out)
    except Exception as query_err:
      err_str = f'{type(query_err).__name__}: {query_err}'
      logging.error('Model sample_text failed: %s', err_str)
      err_container.append(err_str)

  th = threading.Thread(target=_query_model, daemon=True)
  th.start()
  th.join(timeout=90.0)

  if res_container:
    raw_output = res_container[0]
  elif err_container:
    return {'error': f'Model inference error: {err_container[0]}'}
  else:
    return {'error': 'TimeoutError: Model did not respond within 90s'}

  parsed = _extract_json_response(raw_output)
  situation_p = parsed.get(
      'situation_perception',
      f'{agent_name} is currently at {current_location} reflecting on recent'
      ' events.',
  )
  self_p = parsed.get(
      'self_perception',
      f'{agent_name} is {personality}.',
  )
  person_by_sit = parsed.get(
      'person_by_situation',
      f'{agent_name} would answer the researcher candidly based on their'
      ' experiences.',
  )
  interview_reply = parsed.get('interview_response', raw_output)

  # Assemble the exact ConcatActComponent context string
  component_order = [
      'Instructions',
      'LocationInfo',
      'ScheduleAwareness',
      'RecentEvents',
      'CurrentLocation',
      'SituationPerception',
      'SelfPerception',
      'PersonBySituation',
  ]
  contexts_dict = {
      'Instructions': f"\n{agent_name}'s core traits:\n{instructions_val}",
      'LocationInfo': f'\nLocation information:\n{location_info_val}',
      'ScheduleAwareness': (
          f"\n{agent_name}'s current schedule:\n{schedule_val}"
      ),
      'RecentEvents': (
          '\nRecent events (ordered from least recent to most recent):\n'
          f'{recent_events_val}'
      ),
      'CurrentLocation': (
          f"\n{agent_name}'s current location:\n{current_location_val}"
      ),
      'SituationPerception': (
          f'\nQuestion: What situation is {agent_name} in right now?\nAnswer:'
          f' {situation_p}'
      ),
      'SelfPerception': (
          f'\nQuestion: What kind of person is {agent_name}?\nAnswer: {self_p}'
      ),
      'PersonBySituation': (
          f'\nQuestion: What would a person like {agent_name} do in a situation'
          f' like this?\nAnswer: {person_by_sit}'
      ),
  }
  concat_context_str = '\n'.join(
      contexts_dict[k] for k in component_order if contexts_dict.get(k)
  )

  return {
      'agent': agent_name,
      'reply': interview_reply,
      'situation_perception': situation_p,
      'self_perception': self_p,
      'person_by_situation': person_by_sit,
      'raw_chain_of_thought': raw_output,
      'concat_context': concat_context_str,
      'component_order': component_order,
      'current_location': current_location,
      'cutoff_index': actual_cutoff,
      'total_memories': total_memories,
      'last_observation': last_obs,
      'is_simulated': is_simulated,
      'model': _INTERVIEW_MODEL.value,
      'model_init_error': _interview_model_init_error,
      'model_call_error': model_call_error,
  }


_INTERVIEWS_DIR = flags.DEFINE_string(
    'interviews_dir',
    os.path.join('local_runs', 'interviews'),
    'Directory where Interview tab transcripts (.md and .json) are saved.',
)


def _save_interview_data(
    agent_name: str,
    history: list[dict[str, Any]],
    markdown_content: str | None = None,
) -> dict[str, Any]:
  """Saves interview transcript and component data to --interviews_dir."""
  try:
    save_dir = _INTERVIEWS_DIR.value
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    agent_slug = re.sub(r'\s+', '_', agent_name.lower().strip())
    run_id_str = _active_run_id() or 'unknown'
    base_name = f'interview_{agent_slug}_{run_id_str}_{ts}'

    md_filename = f'{base_name}.md'
    json_filename = f'{base_name}.json'
    md_path = os.path.join(save_dir, md_filename)
    json_path = os.path.join(save_dir, json_filename)

    payload = {
        'agent': agent_name,
        'run_id': _active_run_id(),
        'timestamp': datetime.datetime.now().isoformat(),
        'model': _INTERVIEW_MODEL.value,
        'turns_count': len(history),
        'history': history,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
      json.dump(payload, f, indent=2)

    if markdown_content:
      md_text = markdown_content
    else:
      md_lines = [
          f'# 🎙️ Interview Transcript: {agent_name}',
          f'- **Timestamp:** {payload["timestamp"]}',
          f'- **Run ID:** {payload["run_id"]}',
          f'- **Model:** {payload["model"]}',
          '',
          '---',
          '',
      ]
      for turn in history:
        role = turn.get('role', 'unknown')
        text = turn.get('text', '')
        if role == 'user':
          md_lines.append(f'### 🧑‍🔬 Researcher\n{text}\n')
        elif role == 'agent':
          md_lines.append(f'### 👤 {agent_name}\n{text}\n')
          sit_p = turn.get('situation_perception')
          self_p = turn.get('self_perception')
          p_by_s = turn.get('person_by_situation')
          loc = turn.get('current_location')
          if sit_p or self_p or p_by_s:
            md_lines.append(
                '<details><summary>🧠 Cognitive Perceptions'
                ' (Components)</summary>\n'
            )
            if loc:
              md_lines.append(f'- **Current Location:** `{loc}`')
            if sit_p:
              md_lines.append(f'- **Situation Perception:** {sit_p}')
            if self_p:
              md_lines.append(f'- **Self Perception:** {self_p}')
            if p_by_s:
              md_lines.append(f'- **Person by Situation:** {p_by_s}')
            md_lines.append('\n</details>\n')
      md_text = '\n'.join(md_lines)

    with open(md_path, 'w', encoding='utf-8') as f:
      f.write(md_text)

    logging.info('Saved interview to %s and %s', md_path, json_path)
    return {
        'success': True,
        'md_path': md_path,
        'json_path': json_path,
        'md_filename': md_filename,
        'json_filename': json_filename,
        'turns_count': len(history),
    }
  except Exception as e:
    logging.error(
        'Failed to save interview data: %s\n%s', e, traceback.format_exc()
    )
    return {'error': str(e)}


# ---------------------------------------------------------------------------
# HTML generation — self-contained, zero CDN dependencies
# ---------------------------------------------------------------------------


def _generate_html() -> str:
  """Generates the complete Simulation Studio HTML application."""
  raw_html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__DOC_TITLE__</title>
<style>
:root {
  --bg-base: #f8fafc; --bg-panel: #ffffff; --bg-card: #f1f5f9;
  --bg-card-hover: #e2e8f0; --border: #e2e8f0;
  --accent-blue: #0284c7; --accent-purple: #7c3aed; --accent-green: #16a34a;
  --accent-amber: #d97706; --accent-rose: #e11d48; --accent-pink: #db2777;
  --text-main: #0f172a; --text-muted: #64748b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg-base); color: var(--text-main); height: 100vh;
  display: flex; flex-direction: column; overflow: hidden; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* Header */
.hdr { background: #ffffff; padding: 10px 20px; border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.hdr-title { display: flex; align-items: center; gap: 10px; }
.hdr-title h1 { font-size: 1.1rem; font-weight: 700;
  background: linear-gradient(to right, var(--accent-blue), var(--accent-purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hdr-pill { font-size: 0.7rem; background: var(--bg-card); padding: 4px 10px;
  border-radius: 12px; border: 1px solid var(--border); color: var(--text-muted); }
.hdr-right { display: flex; gap: 10px; align-items: center; }

/* Tabs */
.tab-bar { display: flex; gap: 4px; padding: 0 16px; background: var(--bg-panel);
  border-bottom: 1px solid var(--border); flex-shrink: 0; overflow-x: auto; }
.tab-btn { padding: 10px 16px; background: transparent; border: none;
  border-bottom: 2px solid transparent; color: var(--text-muted);
  font-size: 0.8rem; font-weight: 600; cursor: pointer;
  display: flex; align-items: center; gap: 6px; white-space: nowrap; transition: all 0.15s; }
.tab-btn:hover { background: #f1f5f933; color: var(--text-main); }
.tab-btn.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue);
  background: #eff6ff; }

/* Panels */
.panel { flex: 1; overflow: hidden; display: none; }
.panel.active { display: flex; }

/* Grid helpers */
.grid-cols-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-cols-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.grid-cols-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }
.grid-cols-5 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 12px; }

/* Cards */
.card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card h3 { font-size: 0.85rem; font-weight: 700; margin-bottom: 12px; color: var(--accent-blue); }
.metric-card { background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; }
.metric-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-muted); margin-bottom: 2px; font-weight: 700; }
.metric-val { font-size: 1.4rem; font-weight: 700; color: var(--text-main); }
.metric-sub { font-size: 0.7rem; color: var(--text-muted); }

/* Sidebar */
.sidebar { width: 280px; background: var(--bg-panel); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
.sidebar-search { padding: 10px; border-bottom: 1px solid var(--border); }
.sidebar-search input { width: 100%; padding: 7px 10px; background: #f1f5f9;
  border: 1px solid var(--border); border-radius: 6px; color: var(--text-main);
  font-size: 0.8rem; }
.sidebar-search input:focus { outline: none; border-color: var(--accent-blue); }
.agent-list { flex: 1; overflow-y: auto; }
.agent-item { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; cursor: pointer;
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.8rem; transition: background 0.15s; }
.agent-item:hover { background: #f1f5f9; }
.agent-item.active { background: #eff6ff; border-left: 3px solid var(--accent-blue); }
.agent-item .name { font-weight: 600; }
.agent-item .meta { font-size: 0.7rem; color: var(--text-muted); }

/* Memory stream */
.mem-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; margin-bottom: 8px; transition: border-color 0.15s; }
.mem-card:hover { border-color: #cbd5e1; }
.mem-badge { font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;
  text-transform: uppercase; letter-spacing: 0.5px; }
.mem-body { font-size: 0.82rem; line-height: 1.5; color: #334155;
  white-space: pre-wrap; margin-top: 6px; }

/* Filter chips */
.filter-bar { display: flex; gap: 6px; flex-wrap: wrap; padding: 10px 16px;
  border-bottom: 1px solid var(--border); }
.filter-chip { padding: 5px 12px; border-radius: 14px; border: 1px solid var(--border);
  background: var(--bg-panel); color: var(--text-muted); font-size: 0.7rem;
  font-weight: 600; cursor: pointer; transition: all 0.15s; }
.filter-chip:hover { background: var(--bg-card); color: var(--text-main); }
.filter-chip.active { background: var(--bg-card); border-color: var(--accent-blue);
  color: var(--text-main); }

/* Profile grid */
.profile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.meta-row { display: flex; justify-content: space-between; padding: 5px 0;
  border-bottom: 1px solid #f1f5f9; font-size: 0.8rem; }
.meta-row span:first-child { color: var(--text-muted); }
.meta-row span:last-child { font-weight: 500; }
.trait-bar { height: 5px; background: #e2e8f0; border-radius: 3px; overflow: hidden; margin-top: 3px; }
.trait-fill { height: 100%; background: linear-gradient(to right, var(--accent-blue), var(--accent-purple)); }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border);
  color: var(--text-muted); font-weight: 700; font-size: 0.65rem;
  text-transform: uppercase; background: #f8fafc; position: sticky; top: 0; }
td { padding: 5px 8px; border-bottom: 1px solid #f1f5f9; color: #334155; }
tr:hover td { background: #f1f5f9; }

.sql-btn { padding: 6px 16px; background: var(--accent-green); color: #000;
  border: none; border-radius: 6px; font-weight: 700; font-size: 0.75rem;
  cursor: pointer; margin-top: 8px; }
.sql-btn:hover { opacity: 0.9; }

/* Command preview */
.cmd-block { background: #f8fafc; border: 1px solid var(--border);
  border-radius: 6px; padding: 12px; font-family: ui-monospace, monospace;
  font-size: 0.75rem; line-height: 1.6; white-space: pre-wrap;
  overflow-x: auto; user-select: all; }
.cmd-block.cyan { color: #0369a1; }
.cmd-block.green { color: #166534; }

/* Config inputs */
.cfg-label { display: block; font-size: 0.7rem; font-weight: 700;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
.cfg-range { width: 100%; accent-color: var(--accent-blue); }
.cfg-select { width: 100%; background: #f8fafc; border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; color: var(--text-main); font-size: 0.8rem; }
.cfg-select:focus { outline: none; border-color: var(--accent-blue); }
.cfg-val { color: var(--accent-blue); font-family: monospace; }

/* Timeline steps */
.tl-step { border-left: 2px solid var(--accent-blue); padding: 8px 0 8px 16px; margin-left: 8px; }
.tl-step .step-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
.tl-step .step-desc { font-size: 0.75rem; color: var(--text-muted); margin-top: 3px; }

/* Copy button */
.copy-btn { padding: 4px 12px; background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text-muted); font-size: 0.7rem; cursor: pointer; }
.copy-btn:hover { background: var(--bg-card-hover); color: var(--text-main); }

/* Content scroll */
.scroll-y { overflow-y: auto; }
.main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.p-16 { padding: 16px; }
.gap-12 { gap: 12px; }
.gap-16 { gap: 16px; }
.mb-12 { margin-bottom: 12px; }
.mb-16 { margin-bottom: 16px; }
.flex-1 { flex: 1; min-height: 0; }

/* Interview Mode */
.interview-chat-wrap { display: flex; flex-direction: column; height: 100%; flex: 1; min-height: 0; background: var(--bg-base); }
.chat-stream { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; }
.chat-msg { display: flex; gap: 12px; max-width: 85%; }
.chat-msg.user { align-self: flex-end; flex-direction: row-reverse; }
.chat-msg.agent { align-self: flex-start; }
.chat-avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.chat-avatar.user { background: linear-gradient(135deg, #0284c7, #38bdf8); color: white; }
.chat-avatar.agent { background: linear-gradient(135deg, #7c3aed, #a855f7); color: white; }
.chat-bubble { padding: 12px 16px; border-radius: 12px; font-size: 0.84rem; line-height: 1.55; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.chat-msg.user .chat-bubble { background: #0284c7; color: white; border-top-right-radius: 2px; }
.chat-msg.agent .chat-bubble { background: var(--bg-panel); color: var(--text-main); border: 1px solid var(--border); border-top-left-radius: 2px; }

.cot-drawer { margin-top: 10px; border-top: 1px solid #f1f5f9; padding-top: 8px; font-size: 0.75rem; }
.cot-toggle { cursor: pointer; color: var(--accent-purple); font-weight: 600; display: flex; align-items: center; gap: 4px; user-select: none; }
.cot-content { display: none; margin-top: 8px; background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; }
.cot-content.open { display: block; }
.cot-item { margin-bottom: 6px; }
.cot-label { font-weight: 700; color: #475569; text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.5px; }
.cot-val { color: #1e293b; margin-top: 2px; }

.quick-pills { display: flex; gap: 6px; overflow-x: auto; padding: 8px 16px; background: var(--bg-panel); border-top: 1px solid var(--border); }
.quick-pill { font-size: 0.72rem; padding: 5px 12px; border-radius: 14px; border: 1px solid var(--border); background: var(--bg-card); color: var(--text-muted); cursor: pointer; white-space: nowrap; transition: all 0.15s; }
.quick-pill:hover { background: #eff6ff; color: var(--accent-blue); border-color: var(--accent-blue); }

.chat-input-area { padding: 12px 16px; background: var(--bg-panel); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: flex-end; }
.chat-input-area textarea { flex: 1; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; resize: none; height: 50px; line-height: 1.4; outline: none; background: #f8fafc; font-family: inherit; }
.chat-input-area textarea:focus { border-color: var(--accent-blue); background: #ffffff; }

.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: none; align-items: center; justify-content: center; z-index: 1000; padding: 20px; }
.modal-overlay.open { display: flex; }
.modal-box { background: var(--bg-panel); border-radius: 10px; width: 100%; max-width: 850px; max-height: 85vh; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <div class="hdr-title">
    <h1>__HEADER_TITLE__</h1>
    <span class="hdr-pill" id="globalStats">Loading...</span>
  </div>
  <div class="hdr-right">
  </div>
</div>

<!-- Tab Bar -->
<div class="tab-bar">
__TAB_MAP__
  <button class="__CLS_TAB_COGNITIVE__" onclick="switchTab('cognitive', this)">🧠 Memory Visualizer</button>
  <button class="tab-btn" onclick="switchTab('launcher', this)">🚀 Launch Studio</button>
  <button class="tab-btn" onclick="switchTab('timeline', this)">🔌 Component Wiring</button>
  <button class="tab-btn" id="tabBtnInterview" onclick="switchTab('interview', this)">🎙️ Agent Interview</button>
</div>

__PANEL_MAP__

<!-- Tab 1: Cognitive Debugger -->
<div class="__CLS_PANEL_COGNITIVE__" id="panel-cognitive" style="flex-direction: row;">
  <div class="sidebar">
    <div class="sidebar-search">
      <input type="text" id="agentSearch" placeholder="Search agents..." oninput="filterAgents()">
    </div>
    <div class="agent-list" id="agentList"></div>
  </div>
  <div class="main-content">
    <div class="filter-bar" id="filterBar">
      <button class="filter-chip active" onclick="filterMems('all', this)">All</button>
      <button class="filter-chip" onclick="filterMems('life_event', this)">📅 Events</button>
      <button class="filter-chip" onclick="filterMems('conversation', this)">💬 Conversations</button>
      <button class="filter-chip" onclick="filterMems('marketplace', this)">🛒 Marketplace</button>
      <button class="filter-chip" onclick="filterMems('fiscal', this)">💰 Fiscal</button>
      <button class="filter-chip" onclick="filterMems('formative', this)">🧠 Formative</button>
      <button class="filter-chip" onclick="filterMems('journal', this)">📓 Journal</button>
      <button class="filter-chip" onclick="filterMems('identity', this)">👤 Identity</button>
    </div>
    <div class="p-16 scroll-y flex-1" id="memoryStream">
      <div style="text-align:center;color:var(--text-muted);padding:40px;">Select an agent to inspect</div>
    </div>
  </div>
</div>

<!-- Tab 4: Launch Studio -->
<div class="panel" id="panel-launcher" style="flex-direction: column;">
  <div class="p-16 scroll-y flex-1">
    <div class="grid-cols-2" style="gap:20px;">
      <div class="card">
        <h3>⚙️ Simulation Parameters</h3>
        <div style="display:flex;flex-direction:column;gap:14px;">
          <div>
            <label class="cfg-label">Preset</label>
            <select class="cfg-select" id="cfgPreset" onchange="applyPreset()">
              <option value="smoke">Quick test: 4 agents × 12 ticks (1 day + 1 night), mock LLM (no API key, ~1 min)</option>
              <option value="brecksville112" selected>Brecksville: 20 agents × 112 ticks (14 days, real LLM)</option>
              <option value="nightMkt">Night GM test 1 · Marketplace only (UBI arm): 20 agents × 40 ticks (4 nights, real LLM)</option>
              <option value="nightX">Night GM test 2 · X only: 20 agents × 40 ticks (4 nights, real LLM)</option>
              <option value="nightBoth">Night GM test 3 · X then Marketplace: 20 agents × 40 ticks (4 nights, real LLM)</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div>
            <label class="cfg-label">Agents: <span class="cfg-val" id="cfgAgentsV">20</span></label>
            <input type="range" class="cfg-range" id="cfgAgents" min="1" max="1000" value="20" oninput="markCustom()">
          </div>
          <div>
            <label class="cfg-label">Ticks: <span class="cfg-val" id="cfgTicksV">112</span> <span style="color:var(--text-muted);font-weight:400;">(8 ticks = 1 day)</span></label>
            <input type="range" class="cfg-range" id="cfgTicks" min="1" max="224" value="112" oninput="markCustom()">
          </div>
          <div>
            <label class="cfg-label">Population / Map</label>
            <select class="cfg-select" id="cfgPopulation" onchange="markCustom()">
              <option value="brecksville">Brecksville, Ohio suburb (1,000 personas)</option>
            </select>
          </div>
          <div>
            <label class="cfg-label">Decision Logic</label>
            <select class="cfg-select" id="cfgPrefab" onchange="markCustom()">
              <option value="esa">Enacted Self Agent — ESA (esa)</option>
              <option value="rational">Rational Choice (rational)</option>
              <option value="minimal">Minimal (minimal)</option>
              <option value="entity">Basic Associative Memory (entity)</option>
            </select>
          </div>
          <div>
            <label class="cfg-label">Exogenous Event Config</label>
            <select class="cfg-select" id="cfgEvent" onchange="markCustom()">
              <option value="">None</option>
__EVENT_OPTIONS__
            </select>
          </div>
          <div>
            <label class="cfg-label">Layoff Fraction: <span class="cfg-val" id="cfgLayoffV">0.00</span></label>
            <input type="range" class="cfg-range" id="cfgLayoff" min="0" max="1" step="0.05" value="0" oninput="markCustom()">
          </div>
          <div>
            <label class="cfg-label">Fiscal Policy Arm</label>
            <select class="cfg-select" id="cfgFiscal" onchange="markCustom()">
              <option value="control" selected>Control (payroll + rent only)</option>
              <option value="ubi">UBI ($125/week transfer)</option>
            </select>
          </div>
          <div>
            <label class="cfg-label">Language Model</label>
            <select class="cfg-select" id="cfgBackend" onchange="markCustom()">
              <option value="mock">Mock (deterministic, offline; pipeline check only)</option>
              <option value="google_aistudio" selected>Google AI Studio (needs $GOOGLE_API_KEY)</option>
              <option value="openai">OpenAI (needs $OPENAI_API_KEY)</option>
              <option value="ollama">Ollama (local server)</option>
              <option value="vllm">vLLM (local server)</option>
              <option value="together_ai">Together AI</option>
            </select>
          </div>
          <div>
            <label class="cfg-label">Model Name</label>
            <input class="cfg-select" id="cfgModel" value="gemini-2.5-flash" oninput="markCustom()">
          </div>
          <div style="display:flex;gap:16px;flex-wrap:wrap;">
            <label class="cfg-label"><input type="checkbox" id="cfgDates" checked onchange="markCustom()"> First dates</label>
            <label class="cfg-label"><input type="checkbox" id="cfgX" checked onchange="markCustom()"> Night GM: X</label>
            <label class="cfg-label"><input type="checkbox" id="cfgMktOn" checked onchange="markCustom()"> Night GM: Marketplace</label>
            <label class="cfg-label"><input type="checkbox" id="cfgPoll" onchange="markCustom()"> Final survey (poll)</label>
          </div>
          <div>
            <label class="cfg-label">X Rounds: <span class="cfg-val" id="cfgXRoundsV">2</span></label>
            <input type="range" class="cfg-range" id="cfgXRounds" min="1" max="5" value="2" oninput="markCustom()">
          </div>
          <div>
            <label class="cfg-label">Marketplace Rounds: <span class="cfg-val" id="cfgMktV">5</span></label>
            <input type="range" class="cfg-range" id="cfgMkt" min="1" max="10" value="5" oninput="markCustom()">
          </div>
          <div>
            <label class="cfg-label">Output Directory</label>
            <input class="cfg-select" id="cfgOut" value="./local_runs/brecksville_20x112" oninput="markCustom()">
          </div>
        </div>
      </div>
      <div class="card" style="display:flex;flex-direction:column;gap:16px;">
        <h3>🚀 Commands (run from the repository root)</h3>
        <div style="font-size:0.75rem;color:var(--text-muted);" id="cmdNote"></div>
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Step 1 · Run the simulation (Python async engine)</span>
            <button class="copy-btn" onclick="copyCmd('cmdPy')">📋 Copy</button>
          </div>
          <div class="cmd-block green" id="cmdPy"></div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Step 2 (optional, 2nd terminal) · Watch progress</span>
            <button class="copy-btn" onclick="copyCmd('cmdProgress')">📋 Copy</button>
          </div>
          <div class="cmd-block cyan" id="cmdProgress"></div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Step 3 · Open the finished run in Simulation Studio</span>
            <button class="copy-btn" onclick="copyCmd('cmdView')">📋 Copy</button>
          </div>
          <div class="cmd-block cyan" id="cmdView"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Tab 5: Interactive Component Wiring -->
<div class="panel" id="panel-timeline" style="flex-direction: column;">
  <div style="display:flex;gap:4px;padding:8px 16px;background:var(--bg-card);border-bottom:1px solid var(--border);flex-shrink:0;">
    <button class="filter-chip active" onclick="switchTimelineView('wiring', this)">🔌 Component Wiring</button>
  </div>

  <!-- Sub-view 1: Interactive Component Wiring Graph -->
  <div class="p-16 scroll-y flex-1" id="tl-wiring" style="position:relative;">
    <div class="card" style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="margin-bottom:0;">🔌 GM ↔ Agent Component Wiring</h3>
        <div style="display:flex;gap:8px;align-items:center;">
          <select id="wiringPhase" onchange="renderWiringGraph()" style="font-size:0.72rem;padding:4px 8px;border:1px solid var(--border);border-radius:4px;background:var(--bg-card);">
            <option value="all">All Phases</option>
            <option value="observe">Phase 1: Make Observation</option>
            <option value="act">Phase 2: Action Generation</option>
            <option value="resolve">Phase 3: Event Resolution</option>
            <option value="night">Phase 4: Nighttime Marketplace</option>
          </select>
          <button class="copy-btn" onclick="resetWiringZoom()">⟳ Reset Zoom</button>
        </div>
      </div>
      <p style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;">
        Interactive graph showing how GM components wire into Entity Agent components.
        <b>Hover</b> nodes for details. <b>Click</b> to highlight data flow. Drag to pan. Scroll to zoom.
      </p>
    </div>
    <div style="background:var(--bg-panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;position:relative;">
      <svg id="wiringGraph" width="100%" height="620" style="cursor:grab;"></svg>
      <div id="wiringTooltip" style="display:none;position:absolute;padding:10px 14px;background:#0f172a;color:#f8fafc;border-radius:6px;font-size:0.72rem;max-width:320px;pointer-events:none;z-index:10;box-shadow:0 4px 12px rgba(0,0,0,0.25);line-height:1.5;"></div>
    </div>
    <div class="card" style="margin-top:12px;">
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.7rem;">
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#0284c7;margin-right:4px;vertical-align:middle;"></span>GM Infrastructure</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#7c3aed;margin-right:4px;vertical-align:middle;"></span>GM Observation</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#d97706;margin-right:4px;vertical-align:middle;"></span>GM Economic</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#16a34a;margin-right:4px;vertical-align:middle;"></span>GM Resolution</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#e11d48;margin-right:4px;vertical-align:middle;"></span>Entity Agent</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#db2777;margin-right:4px;vertical-align:middle;"></span>LLM / External</span>
        <span style="color:var(--text-muted);">━━ data flow &nbsp; ╌╌ conditional</span>
      </div>
    </div>
  </div>
</div>

<!-- Tab 6: Agent Interview Mode -->
<div class="panel" id="panel-interview" style="flex-direction: row; height: calc(100vh - 86px);">
  <!-- Left Column: Agent & Memory Controls -->
  <div class="sidebar" style="width: 340px; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--bg-panel);">
    <div style="padding: 14px; border-bottom: 1px solid var(--border);">
      <h3 style="font-size: 0.9rem; font-weight: 700; color: var(--accent-purple); margin-bottom: 10px;">🎙️ Interview Configuration</h3>
      <label style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 4px;">Choose Agent:</label>
      <select id="interviewAgentSelect" onchange="onInterviewAgentChange()" style="width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.82rem; background: var(--bg-base); font-weight: 600;">
        <option value="">Loading agents...</option>
      </select>
    </div>

    <!-- Agent Card -->
    <div style="padding: 14px; border-bottom: 1px solid var(--border); background: #faf5ff44;">
      <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #7c3aed, #a855f7); color: white; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 700;" id="interviewAgentAvatar">👤</div>
        <div>
          <div id="interviewAgentCardName" style="font-weight: 700; font-size: 0.95rem;">Select Agent</div>
          <div id="interviewAgentStatusBadge" style="font-size: 0.65rem; color: var(--text-muted);">Status: —</div>
        </div>
      </div>
      <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px;">
        <span class="hdr-pill" id="interviewAgentLocBadge">📍 Loc: —</span>
        <span class="hdr-pill" id="interviewAgentMemBadge">🧠 Mems: 0</span>
        <span class="hdr-pill" id="interviewAgentModelBadge" style="color: var(--accent-green);">🟢 LLM</span>
      </div>
    </div>

    <!-- Memory Cutoff Scrubber -->
    <div style="padding: 14px; border-bottom: 1px solid var(--border); flex-shrink: 0;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <label style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted);">Memory Cutoff Point:</label>
        <span id="interviewCutoffDisplay" style="font-size: 0.72rem; font-weight: 700; color: var(--accent-purple);">Latest</span>
      </div>
      <input type="range" id="interviewCutoffSlider" min="0" max="100" value="100" oninput="onInterviewCutoffInput()" style="width: 100%; margin-bottom: 8px; cursor: pointer;">
      <div style="display: flex; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <button class="copy-btn" onclick="setInterviewCutoffPct(0)">⏮️ Start</button>
        <button class="copy-btn" onclick="setInterviewCutoffPct(50)">⏸️ Mid-Sim</button>
        <button class="copy-btn" onclick="setInterviewCutoffPct(100)">⏭️ Latest</button>
      </div>
      <div style="font-size: 0.68rem; color: var(--text-muted); margin-bottom: 4px;">Last Observation Seen:</div>
      <div id="interviewCutoffMemPreview" style="font-size: 0.72rem; line-height: 1.4; color: #334155; background: var(--bg-base); padding: 8px; border-radius: 6px; border: 1px solid var(--border); max-height: 90px; overflow-y: auto;">Select an agent to see cutoff...</div>
    </div>

    <!-- Controls -->
    <div style="padding: 14px; margin-top: auto; display: flex; flex-direction: column; gap: 8px;">
      <button class="copy-btn" id="saveCloudtopBtn" onclick="saveInterviewToCloudtop()" style="text-align: center; padding: 8px 10px; background: #e0e7ff; color: #3730a3; font-weight: 700; border: 1px solid #c7d2fe;">💾 Save to Cloudtop (.md + .json)</button>
      <div id="saveCloudtopStatus" style="font-size: 0.68rem; line-height: 1.3; display: none; white-space: pre-wrap; padding: 6px; background: var(--bg-base); border-radius: 4px; border: 1px solid var(--border);"></div>
      <button class="copy-btn" onclick="exportInterviewTranscript()" style="text-align: center; padding: 6px;">📥 Download Client Markdown</button>
      <button class="copy-btn" onclick="resetInterviewChat()" style="text-align: center; padding: 6px; color: var(--text-muted);">🗑️ Clear Conversation</button>
    </div>
  </div>

  <!-- Right Column: Interactive Chat -->
  <div class="interview-chat-wrap">
    <!-- Chat Header -->
    <div style="padding: 10px 20px; background: var(--bg-panel); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
      <div>
        <span style="font-weight: 700; font-size: 0.9rem;" id="chatHeaderTitle">Agent Interview</span>
        <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 10px;" id="chatHeaderSub">Replicating Concordia Component Pipeline</span>
      </div>
      <div>
        <button class="copy-btn" onclick="openConcatModal()">🔍 View ConcatActContext</button>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="chat-stream" id="interviewChatStream">
      <!-- Welcome message injected by JS -->
    </div>

    <!-- Quick Question Pills -->
    <div class="quick-pills">
      <span style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); align-self: center;">Suggested:</span>
      <button class="quick-pill" onclick="sendQuickPrompt(this)">How do you feel about the UBI policy?</button>
      <button class="quick-pill" onclick="sendQuickPrompt(this)">What are you doing at the lighthouse right now?</button>
      <button class="quick-pill" onclick="sendQuickPrompt(this)">How did job displacement change your daily routine?</button>
      <button class="quick-pill" onclick="sendQuickPrompt(this)">Who on the island do you trust the most?</button>
    </div>

    <!-- Input Bar -->
    <div class="chat-input-area">
      <textarea id="interviewMsgInput" placeholder="Ask a question... (Enter to send, Shift+Enter for newline)" onkeydown="onInterviewInputKey(event)"></textarea>
      <button class="sql-btn" id="interviewSendBtn" onclick="sendInterviewMessage()" style="margin-top: 0; height: 50px; padding: 0 20px;">💬 Send</button>
    </div>
  </div>
</div>

<!-- Modal: ConcatActComponent Context Viewer -->
<div class="modal-overlay" id="concatModalOverlay" onclick="if(event.target===this)closeConcatModal()">
  <div class="modal-box">
    <div style="padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
      <h3 style="font-size: 0.9rem; font-weight: 700; color: var(--accent-blue); margin-bottom: 0;">🔍 ConcatActComponent Replicated Context String</h3>
      <div style="display: flex; gap: 8px;">
        <button class="copy-btn" onclick="copyConcatText()">📋 Copy</button>
        <button class="copy-btn" onclick="closeConcatModal()">✕ Close</button>
      </div>
    </div>
    <div style="padding: 16px; overflow-y: auto; flex: 1;">
      <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 12px;">
        This is the exact full context string assembled by concatenating the <code>pre_act</code> outputs of all components in order (<code>Instructions</code>, <code>LocationInfo</code>, <code>ScheduleAwareness</code>, <code>RecentEvents</code>, <code>CurrentLocation</code>, <code>SituationPerception</code>, <code>SelfPerception</code>, <code>PersonBySituation</code>), exactly as <code>ConcatActComponent._context_for_action()</code> constructs it before prompting the LLM.
      </p>
      <pre id="concatModalText" style="font-family: ui-monospace, monospace; font-size: 0.75rem; line-height: 1.5; color: #1e293b; background: #f8fafc; padding: 14px; border-radius: 6px; border: 1px solid var(--border); white-space: pre-wrap; word-break: break-word;"></pre>
    </div>
  </div>
</div>

<script>
// ---- State ----
let cogData = null;
let selectedAgent = null;
let currentFilter = 'all';

// ---- Tab switching ----
function switchTab(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'timeline') { setTimeout(renderWiringGraph, 50); }
}

// ---- Escape HTML ----
function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

// ---- Cognitive Debugger ----
async function loadCognitive() {
  try {
    const r = await fetch('/api/cognitive');
    cogData = await r.json();
    document.getElementById('globalStats').textContent =
        (cogData.agent_names || []).length + ' Agents Loaded';
    renderAgentList();
    if (cogData.agent_names && cogData.agent_names.length > 0) {
      selectAgent(cogData.agent_names[0]);
    }
    initInterview();
  } catch(e) {
    console.error('Cognitive load failed:', e);
  }
}

function filterAgents() {
  renderAgentList();
}

function renderAgentList() {
  if (!cogData) return;
  const q = (document.getElementById('agentSearch').value || '').toLowerCase();
  const list = document.getElementById('agentList');
  list.innerHTML = '';
  (cogData.agent_names || []).forEach(name => {
    if (q && !name.toLowerCase().includes(q)) return;
    const agent = cogData.agents[name] || {};
    const persona = agent.persona || {};
    const tier = persona.economic_class || '';
    const div = document.createElement('div');
    div.className = 'agent-item' + (name === selectedAgent ? ' active' : '');
    div.onclick = () => selectAgent(name);
    div.innerHTML = '<div><div class="name">' + esc(name) + '</div>' +
        '<div class="meta">' + esc(tier) + ' • ' + (agent.memory_count || 0) + ' mems</div></div>';
    list.appendChild(div);
  });
}

function selectAgent(name) {
  selectedAgent = name;
  renderAgentList();
  renderMemories();
}

function filterMems(cat, btn) {
  currentFilter = cat;
  document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderMemories();
}

function renderMemories() {
  const el = document.getElementById('memoryStream');
  if (!cogData || !selectedAgent) {
    el.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;">Select an agent</div>';
    return;
  }
  const agent = cogData.agents[selectedAgent] || { memories: [] };
  const mems = agent.memories || [];
  const filtered = currentFilter === 'all' ? mems : mems.filter(m => m.category === currentFilter);

  if (filtered.length === 0) {
    el.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;">No memories in this category.</div>';
    return;
  }

  let html = '';
  // Profile card at top
  const persona = agent.persona || {};
  if (persona.name) {
    html += '<div class="card mb-12"><h3>👤 ' + esc(persona.name) + '</h3>';
    html += '<div class="profile-grid">';
    html += '<div>';
    const fields = [['Age', persona.age], ['Gender', persona.gender], ['Ethnicity', persona.ethnicity],
                    ['Economic Tier', persona.economic_class], ['Neighborhood', persona.neighborhood],
                    ['Home', persona.home_place], ['Workplace', persona.work_place]];
    fields.forEach(([k, v]) => {
      if (v) html += '<div class="meta-row"><span>' + k + '</span><span>' + esc(String(v)) + '</span></div>';
    });
    html += '</div>';
    // Traits
    const traits = persona.traits || {};
    if (Object.keys(traits).length > 0) {
      html += '<div>';
      html += '<div style="font-size:0.75rem;font-weight:700;color:var(--accent-purple);margin-bottom:8px;">Psychological Traits</div>';
      Object.entries(traits).forEach(([k, v]) => {
        const pct = typeof v === 'number' ? (v > 1 ? (v/5)*100 : v*100) : 50;
        html += '<div style="margin-bottom:6px;"><div style="display:flex;justify-content:space-between;font-size:0.7rem;"><span style="color:var(--text-muted);">' +
            k.replace(/_/g, ' ') + '</span><span>' + (typeof v === 'number' ? v.toFixed(2) : v) + '</span></div>' +
            '<div class="trait-bar"><div class="trait-fill" style="width:' + pct + '%;"></div></div></div>';
      });
      html += '</div>';
    }
    html += '</div></div>';
  }

  // Memory cards
  filtered.forEach((m, idx) => {
    html += '<div class="mem-card">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;">' +
        '<div style="display:flex;gap:8px;align-items:center;">' +
        '<span class="mem-badge" style="background:' + m.color + '22;color:' + m.color + ';border:1px solid ' + m.color + '44;">' + m.icon + ' ' + m.badge + '</span>' +
        (m.timestamp ? '<span style="font-size:0.7rem;color:var(--text-muted);">🕒 ' + esc(m.timestamp) + '</span>' : '') +
        (m.location ? '<span style="font-size:0.7rem;color:var(--accent-blue);">📍 ' + esc(m.location) + '</span>' : '') +
        '</div>' +
        '<span style="font-size:0.65rem;color:#475569;">#' + (idx+1) + '</span>' +
        '</div>' +
        '<div class="mem-body">' + esc(m.raw) + '</div>' +
        '</div>';
  });

  el.innerHTML = html;
}

// ---- Launch Studio ----
// Generates commands for the third-party runner (run.py). Uses
// fromCharCode for backslash/newline so the Python template needs no escaping.
const LAUNCH_BS = String.fromCharCode(92);
const LAUNCH_NL = String.fromCharCode(10);
const LAUNCH_PRESETS = {
  smoke: {agents: 4, ticks: 12, population: 'brecksville', prefab: 'esa',
          event: 'job_loss', layoff: 0.5, backend: 'mock',
          model: '', dates: true, x: true, mkt: true, poll: false,
          xRounds: 1, mktRounds: 1, out: './local_runs/quick_test'},
  brecksville112: {agents: 20, ticks: 112, population: 'brecksville',
          prefab: 'esa', event: '', layoff: 0, backend: 'google_aistudio',
          model: 'gemini-2.5-flash', dates: true, x: true, mkt: true,
          poll: false, xRounds: 2, mktRounds: 5,
          out: './local_runs/brecksville_20x112'},
  nightMkt: {agents: 20, ticks: 40, population: 'brecksville',
          prefab: 'esa', event: '', layoff: 0,
          fiscal: 'ubi', backend: 'google_aistudio', model: 'gemini-2.5-flash',
          dates: true, x: false, mkt: true, poll: false, xRounds: 2,
          mktRounds: 5, out: './local_runs/night_marketplace'},
  nightX: {agents: 20, ticks: 40, population: 'brecksville',
          prefab: 'esa', event: '', layoff: 0,
          fiscal: 'ubi', backend: 'google_aistudio', model: 'gemini-2.5-flash',
          dates: true, x: true, mkt: false, poll: false, xRounds: 2,
          mktRounds: 5, out: './local_runs/night_x'},
  nightBoth: {agents: 20, ticks: 40, population: 'brecksville',
          prefab: 'esa', event: '', layoff: 0,
          fiscal: 'ubi', backend: 'google_aistudio', model: 'gemini-2.5-flash',
          dates: true, x: true, mkt: true, poll: false, xRounds: 2,
          mktRounds: 5, out: './local_runs/night_x_marketplace'},
};
const LAUNCH_POPULATIONS = {
  brecksville: {personas: 'brecksville_ohio',
                geography: 'brecksville'},
};
const LAUNCH_KEY_ENV = {google_aistudio: 'GOOGLE_API_KEY', openai: 'OPENAI_API_KEY',
                        together_ai: 'TOGETHER_API_KEY'};

function _el(id) { return document.getElementById(id); }

function applyPreset() {
  const p = LAUNCH_PRESETS[_el('cfgPreset').value];
  if (p) {
    _el('cfgAgents').value = p.agents; _el('cfgTicks').value = p.ticks;
    _el('cfgPopulation').value = p.population; _el('cfgPrefab').value = p.prefab;
    _el('cfgEvent').value = p.event; _el('cfgLayoff').value = p.layoff;
    _el('cfgFiscal').value = p.fiscal || 'control';
    _el('cfgBackend').value = p.backend; _el('cfgModel').value = p.model;
    _el('cfgDates').checked = p.dates; _el('cfgX').checked = p.x;
    _el('cfgMktOn').checked = p.mkt; _el('cfgPoll').checked = p.poll;
    _el('cfgXRounds').value = p.xRounds; _el('cfgMkt').value = p.mktRounds;
    _el('cfgOut').value = p.out;
  }
  updateCmd();
}

function markCustom() {
  _el('cfgPreset').value = 'custom';
  updateCmd();
}

function _joinCmd(parts) {
  return parts.join(' ' + LAUNCH_BS + LAUNCH_NL + '  ');
}

function updateCmd() {
  const agents = _el('cfgAgents').value;
  const ticks = _el('cfgTicks').value;
  const layoff = parseFloat(_el('cfgLayoff').value).toFixed(2);
  const backend = _el('cfgBackend').value;
  const model = _el('cfgModel').value.trim();
  const pop = LAUNCH_POPULATIONS[_el('cfgPopulation').value];
  const prefab = _el('cfgPrefab').value;
  const event = _el('cfgEvent').value;
  const out = _el('cfgOut').value.trim() || './local_runs/run';
  const xOn = _el('cfgX').checked;
  const mktOn = _el('cfgMktOn').checked;

  _el('cfgAgentsV').textContent = agents;
  _el('cfgTicksV').textContent = ticks;
  _el('cfgLayoffV').textContent = layoff;
  _el('cfgXRoundsV').textContent = _el('cfgXRounds').value;
  _el('cfgMktV').textContent = _el('cfgMkt').value;
  _el('cfgModel').disabled = (backend === 'mock');

  const parts = ['python -m examples.concordia_island.run',
    '--engine_type=async',
    '--personas_date=' + pop.personas,
    '--agents=' + agents,
    '--ticks=' + ticks,
    '--agent_prefab=' + prefab];
  if (event) parts.push('--event_config=' + event);
  if (parseFloat(layoff) > 0) parts.push('--layoff_fraction=' + layoff);
  const fiscal = _el('cfgFiscal').value;
  if (fiscal !== 'control') parts.push('--fiscal_config=' + fiscal);
  parts.push('--first_dates=' + (_el('cfgDates').checked ? 'True' : 'False'));
  parts.push('--enable_x=' + (xOn ? 'True' : 'False'));
  if (xOn) parts.push('--x_rounds=' + _el('cfgXRounds').value);
  parts.push('--enable_nighttime_marketplace=' + (mktOn ? 'True' : 'False'));
  if (mktOn) parts.push('--marketplace_rounds=' + _el('cfgMkt').value);
  parts.push('--poll=' + (_el('cfgPoll').checked ? 'True' : 'False'));
  if (backend === 'mock') {
    parts.push('--use_mock=True');
  } else {
    parts.push('--api_type=' + backend);
    if (model) parts.push('--model_name=' + model);
  }
  parts.push('--output_dir=' + out);

  let runCmd = 'mkdir -p ' + out + LAUNCH_NL + _joinCmd(parts);
  const keyEnv = LAUNCH_KEY_ENV[backend];
  if (keyEnv) {
    runCmd = 'export ' + keyEnv + '=<your-key>   # once per terminal' +
        LAUNCH_NL + runCmd;
  }
  let note = 'Copy each block into a terminal opened at the repository root ' +
      '(the folder containing examples/ and concordia/).';
  if (backend === 'mock') {
    note += ' Mock mode verifies the pipeline (engine, GMs, artifacts, map) ' +
        'with canned text; it is NOT a real simulation.';
  }
  _el('cmdNote').textContent = note;
  _el('cmdPy').textContent = runCmd;
  _el('cmdProgress').textContent =
      'watch -n 15 cat ' + out + '/progress.json';
  _el('cmdView').textContent = _joinCmd([
      'python -m examples.concordia_island.ui.simulation_studio',
      '--run_dir=' + out,
      '--geography=' + pop.geography,
      '--port=9090']) + LAUNCH_NL + '# then open http://localhost:9090';
}

function copyCmd(id) {
  const text = _el(id).textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event && event.target;
    if (btn) { const t = btn.textContent; btn.textContent = '✅ Copied';
               setTimeout(() => { btn.textContent = t; }, 1200); }
  });
}

// ---- Component Wiring tab: sub-view switching ----
function switchTimelineView(view, btn) {
  ['wiring'].forEach(v => {
    const el = document.getElementById('tl-' + v);
    if (el) el.style.display = (v === view) ? '' : 'none';
  });
  btn.parentElement.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  if (view === 'wiring') renderWiringGraph();
}

// ---- Component Wiring Graph (TF-style) ----
const WIRING_NODES = [
  // GM Infrastructure (blue) - Layer 0
  {id:'clock', label:'FixedIntervalClock', cat:'infra', phase:['all','observe','act','resolve'],
   desc:'120-min tick clock (7AM-11PM). Manages tick counter, waking hours, acted-per-tick tracking. 8 daytime ticks/day.',
   x:80, y:30},
  {id:'clock_const', label:'Clock Description', cat:'infra', phase:['all','observe','act'],
   desc:'Constant prompt: "Time advances in 2-hour ticks. Waking hours 7AM-11PM..."', x:250, y:30},
  {id:'instructions', label:'SimulationistInstr.', cat:'infra', phase:['all','observe','act','resolve'],
   desc:'Top-level GM instructions: setting description, simulationist persona for the GM LLM.', x:420, y:30},
  {id:'player_chars', label:'PlayerCharacters', cat:'infra', phase:['all','observe','act','resolve'],
   desc:'Registry of all player names. Used by observation, location, and resolution components.', x:600, y:30},
  {id:'locations_const', label:'Locations Prompt', cat:'infra', phase:['all','observe','act'],
   desc:'Static list of all island locations: medical_clinic, restaurant, community_center, park, etc.', x:780, y:30},

  // GM Observation (purple) - Layer 1
  {id:'display_events', label:'DisplayEvents', cat:'observe', phase:['all','observe','resolve'],
   desc:'Maintains rolling window of recent events. "Story so far" — feeds into observation and resolution.', x:80, y:130},
  {id:'filtered_events', label:'FilteredDisplayEvents', cat:'observe', phase:['all','observe','resolve'],
   desc:'Location-filtered view of DisplayEvents. Only shows events at the agent current location.', x:290, y:130},
  {id:'locations', label:'Locations (WorldState)', cat:'observe', phase:['all','observe','act','resolve'],
   desc:'Tracks entity_locations map. LLM-resolved from movement actions. Feeds location into observations.', x:530, y:130},
  {id:'temporal_nudge', label:'TemporalNudge', cat:'observe', phase:['all','observe'],
   desc:'Injects work-hour reminders, end-of-day resets, weekend cues based on schedule.', x:750, y:130},

  // GM Social & Economic (amber) - Layer 2
  {id:'conv_director', label:'ConversationDirector', cat:'economic', phase:['all','observe'],
   desc:'LLM-driven: selects co-located agents for conversation. Manages cooldown_ticks and max group size.', x:60, y:230},
  {id:'social_sched', label:'SocialScheduler', cat:'economic', phase:['all','observe'],
   desc:'Fires pre-scheduled social events (dates, meetups) at specific ticks. Creates conversations with setup context.', x:270, y:230},
  {id:'food_consumption', label:'FoodConsumption', cat:'economic', phase:['all','observe'],
   desc:'Deducts daily food units at 7AM tick. Emits "You consumed N food units" observation. Checks food >= minimum.', x:490, y:230},
  {id:'fiscal_sched', label:'FiscalScheduler', cat:'economic', phase:['all','observe'],
   desc:'Emits fiscal events from sim/fiscal_configs.py: payroll (Mon 7AM), rent (Fri 3PM), and UBI transfers (Mon 7AM, ubi arm only).', x:710, y:230},
  {id:'async_conv', label:'AsyncConversationState', cat:'economic', phase:['all','observe','act','resolve'],
   desc:'Manages multi-turn conversations: participant lists, turn order, dialogue context, termination.', x:910, y:230},

  // GM Make Observation (purple) - Layer 3
  {id:'make_obs', label:'LocationAwareMakeObs', cat:'observe', phase:['all','observe'],
   desc:'CORE: Generates observations for each agent. Prepends "// location [time]: obs". Fires social events, drains queued events, handles conversation context.', x:350, y:330},

  // GM Acting (green) - Layer 4
  {id:'tick_gated', label:'TickGatedNextActing', cat:'resolve', phase:['all','act'],
   desc:'Controls turn order. NEXT_ACTING: returns eligible agents (not acted + not waiting). NEXT_ACTION_SPEC: returns SPEECH or ACTION spec.', x:150, y:430},
  {id:'event_resolution', label:'IslandEventResolution', cat:'resolve', phase:['all','resolve'],
   desc:'Resolves agent actions via LLM: movement resolution, conversation progression/termination, memory writes.', x:550, y:430},
  {id:'next_gm', label:'TimeBasedNextGM', cat:'resolve', phase:['all','night'],
   desc:'Switches GM at nighttime: island_rules (day) ↔ marketplace_rules (night, 9PM+).', x:850, y:430},

  // Entity Agent Components (rose) - Layer 5
  {id:'e_instructions', label:'Instructions', cat:'entity', phase:['all','act'],
   desc:'Agent Instructions: "{name} core traits" — personality, backstory loaded from persona.', x:40, y:540},
  {id:'e_location', label:'LocationInfo', cat:'entity', phase:['all','act'],
   desc:'Static location info: "lives at X, works at Y, places they can visit: ..."', x:180, y:540},
  {id:'e_schedule', label:'ScheduleAwareness', cat:'entity', phase:['all','act'],
   desc:'Dynamic schedule: "It is Monday 9AM, workday morning. You should be at your workplace."', x:330, y:540},
  {id:'e_observation', label:'ImportantMemories', cat:'entity', phase:['all','act'],
   desc:'Recent events buffer (last 50). Receives observations from GM MakeObservation.', x:490, y:540},
  {id:'e_situation', label:'SituationPerception', cat:'entity', phase:['all','act'],
   desc:'LLM Q: "What situation is {name} in right now?" Uses 10 recent memories.', x:640, y:540},
  {id:'e_self', label:'SelfPerception', cat:'entity', phase:['all','act'],
   desc:'Daily LLM Q: "What kind of person is {name}?" Depends on SituationPerception.', x:790, y:540},
  {id:'e_movement', label:'MovementDecision', cat:'entity', phase:['all','act'],
   desc:'LLM Q: "Should {name} stay or move?" Uses situation + self perception.', x:350, y:620},
  {id:'e_person', label:'PersonBySituation', cat:'entity', phase:['all','act'],
   desc:'LLM Q: "What would {name} do in this situation?" Final context before action.', x:560, y:620},

  // LLM / External (pink) - Layer 6
  {id:'llm_api', label:'LLM API', cat:'llm', phase:['all','act','resolve'],
   desc:'Model calls via the configured LLM provider. AdaptiveResilientLanguageModel handles retry backoff.', x:170, y:620},
  {id:'concat_act', label:'ConcatActComponent', cat:'entity', phase:['all','act'],
   desc:'Concatenates all context components into prompt → calls LLM → returns agent action.', x:760, y:620},
];

const WIRING_EDGES = [
  // Clock feeds into everything
  {from:'clock', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'clock', to:'tick_gated', phase:['all','act'], type:'solid'},
  {from:'clock', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  {from:'clock', to:'locations', phase:['all','observe'], type:'solid'},
  {from:'clock', to:'next_gm', phase:['all','night'], type:'solid'},
  {from:'clock_const', to:'make_obs', phase:['all','observe'], type:'dashed'},
  {from:'clock_const', to:'locations', phase:['all','observe'], type:'dashed'},
  // Instructions/Players feed observation and resolution
  {from:'instructions', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'instructions', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  {from:'player_chars', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'player_chars', to:'tick_gated', phase:['all','act'], type:'solid'},
  {from:'player_chars', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  {from:'locations_const', to:'make_obs', phase:['all','observe'], type:'dashed'},
  {from:'locations_const', to:'locations', phase:['all','observe'], type:'dashed'},
  // Location tracking
  {from:'locations', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'locations', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  {from:'locations', to:'conv_director', phase:['all','observe'], type:'solid'},
  {from:'locations', to:'social_sched', phase:['all','observe'], type:'solid'},
  {from:'locations', to:'temporal_nudge', phase:['all','observe'], type:'solid'},
  // Events
  {from:'display_events', to:'filtered_events', phase:['all','observe'], type:'solid'},
  {from:'filtered_events', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'filtered_events', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  // Social/Economic → MakeObs
  {from:'social_sched', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'conv_director', to:'make_obs', phase:['all','observe'], type:'dashed'},
  {from:'food_consumption', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'fiscal_sched', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'temporal_nudge', to:'make_obs', phase:['all','observe'], type:'dashed'},
  {from:'async_conv', to:'make_obs', phase:['all','observe'], type:'solid'},
  {from:'async_conv', to:'tick_gated', phase:['all','act'], type:'solid'},
  {from:'async_conv', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  // MakeObs → Entity Agent
  {from:'make_obs', to:'e_observation', phase:['all','observe','act'], type:'solid'},
  // TickGated → Entity components
  {from:'tick_gated', to:'concat_act', phase:['all','act'], type:'solid'},
  // Entity internal wiring
  {from:'e_instructions', to:'concat_act', phase:['all','act'], type:'solid'},
  {from:'e_location', to:'concat_act', phase:['all','act'], type:'solid'},
  {from:'e_schedule', to:'concat_act', phase:['all','act'], type:'solid'},
  {from:'e_observation', to:'concat_act', phase:['all','act'], type:'solid'},
  {from:'e_observation', to:'e_situation', phase:['all','act'], type:'solid'},
  {from:'e_situation', to:'e_self', phase:['all','act'], type:'solid'},
  {from:'e_situation', to:'e_movement', phase:['all','act'], type:'solid'},
  {from:'e_self', to:'e_movement', phase:['all','act'], type:'solid'},
  {from:'e_self', to:'e_person', phase:['all','act'], type:'solid'},
  {from:'e_situation', to:'e_person', phase:['all','act'], type:'solid'},
  {from:'e_movement', to:'e_person', phase:['all','act'], type:'solid'},
  {from:'e_movement', to:'concat_act', phase:['all','act'], type:'solid'},
  {from:'e_person', to:'concat_act', phase:['all','act'], type:'solid'},
  // LLM
  {from:'concat_act', to:'llm_api', phase:['all','act'], type:'solid'},
  {from:'llm_api', to:'concat_act', phase:['all','act'], type:'dashed'},
  // Action → Resolution
  {from:'concat_act', to:'event_resolution', phase:['all','resolve'], type:'solid'},
  {from:'event_resolution', to:'display_events', phase:['all','resolve'], type:'solid'},
  {from:'event_resolution', to:'locations', phase:['all','resolve'], type:'dashed'},
];

const CAT_COLORS = {
  infra: '#0284c7', observe: '#7c3aed', economic: '#d97706',
  resolve: '#16a34a', entity: '#e11d48', llm: '#db2777'
};

let wiringZoom = {x:0, y:0, scale:1};
let wiringSelected = null;

function renderWiringGraph() {
  const svg = document.getElementById('wiringGraph');
  const phase = document.getElementById('wiringPhase').value;
  const W = svg.getBoundingClientRect().width || 1050;
  const H = 620;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const visibleNodes = WIRING_NODES.filter(n => n.phase.includes(phase));
  const visibleIds = new Set(visibleNodes.map(n => n.id));
  const visibleEdges = WIRING_EDGES.filter(e =>
    e.phase.includes(phase) && visibleIds.has(e.from) && visibleIds.has(e.to)
  );

  const nodeMap = {};
  visibleNodes.forEach(n => { nodeMap[n.id] = n; });

  // Scale positions to fit
  const maxX = Math.max(...visibleNodes.map(n => n.x)) + 160;
  const maxY = Math.max(...visibleNodes.map(n => n.y)) + 40;
  const scaleX = (W - 40) / Math.max(maxX, 1);
  const scaleY = (H - 40) / Math.max(maxY, 1);
  const sc = Math.min(scaleX, scaleY, 1);

  let html = '<defs>';
  Object.entries(CAT_COLORS).forEach(([cat, color]) => {
    html += `<marker id="arrow-${cat}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${color}88"/></marker>`;
  });
  html += '</defs>';
  html += `<g id="wiringContent" transform="translate(${wiringZoom.x + 20},${wiringZoom.y + 20}) scale(${wiringZoom.scale * sc})">`;

  // Edges
  visibleEdges.forEach(e => {
    const from = nodeMap[e.from], to = nodeMap[e.to];
    if (!from || !to) return;
    const cat = from.cat;
    const color = CAT_COLORS[cat] || '#64748b';
    const opacity = wiringSelected ? (wiringSelected === e.from || wiringSelected === e.to ? 1 : 0.12) : 0.45;
    const dash = e.type === 'dashed' ? 'stroke-dasharray="5,4"' : '';
    const sw = wiringSelected === e.from || wiringSelected === e.to ? 2.5 : 1.5;
    html += `<line x1="${from.x+65}" y1="${from.y+18}" x2="${to.x+65}" y2="${to.y+18}" stroke="${color}" stroke-width="${sw}" opacity="${opacity}" ${dash} marker-end="url(#arrow-${cat})"/>`;
  });

  // Nodes
  visibleNodes.forEach(n => {
    const color = CAT_COLORS[n.cat] || '#64748b';
    const isSelected = wiringSelected === n.id;
    const dimmed = wiringSelected && !isSelected &&
      !visibleEdges.some(e => (e.from === wiringSelected && e.to === n.id) || (e.to === wiringSelected && e.from === n.id));
    const opacity = dimmed ? 0.2 : 1;
    const stroke = isSelected ? color : (color + '66');
    const sw = isSelected ? 3 : 1.5;
    const fill = isSelected ? (color + '18') : '#ffffff';
    html += `<g class="wiring-node" data-id="${n.id}" style="cursor:pointer;opacity:${opacity};" onclick="toggleWiringSelect('${n.id}')" onmouseenter="showWiringTooltip(event,'${n.id}')" onmouseleave="hideWiringTooltip()">`;
    html += `<rect x="${n.x}" y="${n.y}" width="130" height="36" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
    html += `<rect x="${n.x}" y="${n.y}" width="4" height="36" rx="2" fill="${color}"/>`;
    html += `<text x="${n.x+12}" y="${n.y+22}" font-size="10" font-weight="600" fill="${color}" font-family="system-ui">${n.label}</text>`;
    html += '</g>';
  });

  html += '</g>';
  svg.innerHTML = html;

  // Pan/zoom handlers
  let isDragging = false, dragStart = {x:0,y:0};
  svg.onmousedown = (e) => { isDragging = true; dragStart = {x: e.clientX - wiringZoom.x, y: e.clientY - wiringZoom.y}; svg.style.cursor = 'grabbing'; };
  svg.onmousemove = (e) => { if (!isDragging) return; wiringZoom.x = e.clientX - dragStart.x; wiringZoom.y = e.clientY - dragStart.y; const g = document.getElementById('wiringContent'); if (g) g.setAttribute('transform', `translate(${wiringZoom.x+20},${wiringZoom.y+20}) scale(${wiringZoom.scale * sc})`); };
  svg.onmouseup = () => { isDragging = false; svg.style.cursor = 'grab'; };
  svg.onmouseleave = () => { isDragging = false; svg.style.cursor = 'grab'; };
  svg.onwheel = (e) => { e.preventDefault(); const delta = e.deltaY > 0 ? 0.9 : 1.1; wiringZoom.scale = Math.max(0.3, Math.min(3, wiringZoom.scale * delta)); renderWiringGraph(); };
}

function toggleWiringSelect(id) {
  wiringSelected = (wiringSelected === id) ? null : id;
  renderWiringGraph();
}

function showWiringTooltip(event, id) {
  const node = WIRING_NODES.find(n => n.id === id);
  if (!node) return;
  const tip = document.getElementById('wiringTooltip');
  const catLabels = {infra:'GM Infrastructure', observe:'GM Observation', economic:'GM Economic', resolve:'GM Resolution', entity:'Entity Agent', llm:'LLM/External'};
  tip.innerHTML = `<div style="font-weight:700;margin-bottom:4px;">${node.label}</div><div style="font-size:0.65rem;color:${CAT_COLORS[node.cat]};margin-bottom:6px;">${catLabels[node.cat] || node.cat}</div><div style="opacity:0.85;">${node.desc}</div>`;
  tip.style.display = 'block';
  const rect = tip.parentElement.getBoundingClientRect();
  tip.style.left = Math.min(event.clientX - rect.left + 12, rect.width - 340) + 'px';
  tip.style.top = (event.clientY - rect.top - 60) + 'px';
}

function hideWiringTooltip() {
  document.getElementById('wiringTooltip').style.display = 'none';
}

function resetWiringZoom() {
  wiringZoom = {x:0, y:0, scale:1};
  wiringSelected = null;
  renderWiringGraph();
}

// ---- Agent Interview Mode ----
let interviewAgent = null;
let interviewHistory = [];
let lastConcatContext = "";
let interviewCutoffIdx = null;

function initInterview() {
  if (!cogData || !cogData.agents) return;
  const sel = document.getElementById('interviewAgentSelect');
  if (!sel) return;
  sel.innerHTML = '';
  const agentNames = Object.keys(cogData.agents).sort();
  if (agentNames.length === 0) return;

  const defaultAgent = '__DEFAULT_INTERVIEW_AGENT__' || (agentNames.includes('Julia Hicks') ? 'Julia Hicks' : agentNames[0]);

  agentNames.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === defaultAgent) opt.selected = true;
    sel.appendChild(opt);
  });

  selectInterviewAgent(defaultAgent || agentNames[0]);

  if ('__DEFAULT_INTERVIEW_AGENT__') {
    switchTab('interview', document.getElementById('tabBtnInterview'));
  }
}

function selectInterviewAgent(name) {
  if (!cogData || !cogData.agents || !cogData.agents[name]) return;
  interviewAgent = name;
  interviewHistory = [];
  const a = cogData.agents[name];
  const p = a.persona || {};
  const mems = a.memories || [];

  document.getElementById('interviewAgentCardName').textContent = name;
  document.getElementById('interviewChatAgentName').textContent = name;
  document.getElementById('interviewAgentStatusBadge').textContent = 'Status: ' + (p.layoff_status || 'Resident');
  document.getElementById('interviewAgentMemBadge').textContent = `🧠 Mems: ${mems.length}`;

  // Find current location
  let currentLoc = p.home_place || 'town_square';
  for (let i = mems.length - 1; i >= 0; i--) {
    const m = mems[i].raw || mems[i].text || '';
    const match = m.match(/\/\/\s*([a-z_0-9]+)[\s\[:]/);
    if (match) { currentLoc = match[1]; break; }
  }
  document.getElementById('interviewAgentLocBadge').textContent = `📍 Loc: ${currentLoc}`;
  document.getElementById('chatHeaderSub').textContent = `@ ${currentLoc} • Replicating Concordia Component Pipeline`;

  // Init Slider
  const slider = document.getElementById('interviewCutoffSlider');
  slider.min = 0;
  slider.max = Math.max(0, mems.length - 1);
  slider.value = slider.max;
  interviewCutoffIdx = mems.length > 0 ? mems.length - 1 : 0;
  updateInterviewCutoffUI();

  // Reset Chat Stream
  renderWelcomeChat(name, currentLoc);
}

function onInterviewAgentChange() {
  const sel = document.getElementById('interviewAgentSelect');
  selectInterviewAgent(sel.value);
}

function onInterviewCutoffInput() {
  const slider = document.getElementById('interviewCutoffSlider');
  interviewCutoffIdx = parseInt(slider.value, 10);
  updateInterviewCutoffUI();
}

function setInterviewCutoffPct(pct) {
  const a = cogData.agents[interviewAgent];
  if (!a || !a.memories) return;
  const mems = a.memories;
  const idx = Math.round((pct / 100) * (mems.length - 1));
  const slider = document.getElementById('interviewCutoffSlider');
  slider.value = idx;
  interviewCutoffIdx = idx;
  updateInterviewCutoffUI();
}

function updateInterviewCutoffUI() {
  const a = cogData.agents[interviewAgent];
  if (!a || !a.memories || a.memories.length === 0) return;
  const mems = a.memories;
  const idx = Math.min(Math.max(0, interviewCutoffIdx !== null ? interviewCutoffIdx : mems.length - 1), mems.length - 1);
  const mem = mems[idx];
  const isLatest = (idx === mems.length - 1);

  document.getElementById('interviewCutoffDisplay').textContent = isLatest ? `Latest (Memory #${idx + 1})` : `Memory #${idx + 1} of ${mems.length}`;
  document.getElementById('interviewCutoffMemPreview').textContent = mem ? (mem.raw || mem.text) : 'None';
}

function renderWelcomeChat(name, loc) {
  const stream = document.getElementById('interviewChatStream');
  stream.innerHTML = `
    <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; font-size: 0.8rem; line-height: 1.5;">
      <div style="font-weight: 700; color: var(--accent-purple); margin-bottom: 4px;">🎙️ Interview Mode Ready</div>
      <div>You are connected to <b>${esc(name)}</b> currently at <code>${esc(loc)}</code>.</div>
      <div style="color: var(--text-muted); margin-top: 4px;">
        Each question runs Concordia's <code>pre_act</code> perception pipeline (<b>SituationPerception → SelfPerception → PersonBySituation</b>) before producing <b>${esc(name)}</b>'s response.
      </div>
    </div>
  `;
}

function onInterviewInputKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendInterviewMessage();
  }
}

function sendQuickPrompt(btn) {
  const input = document.getElementById('interviewMsgInput');
  input.value = btn.textContent;
  sendInterviewMessage();
}

async function sendInterviewMessage() {
  const input = document.getElementById('interviewMsgInput');
  const msg = input.value.trim();
  if (!msg || !interviewAgent) return;

  appendChatTurn('user', msg);
  input.value = '';

  const sendBtn = document.getElementById('interviewSendBtn');
  sendBtn.disabled = true;
  sendBtn.textContent = '⏳ Thinking...';

  // Loading bubble
  const stream = document.getElementById('interviewChatStream');
  const loadingId = 'loading-' + Date.now();
  const loadingEl = document.createElement('div');
  loadingEl.id = loadingId;
  loadingEl.className = 'chat-msg agent';
  loadingEl.innerHTML = `
    <div class="chat-avatar agent">👤</div>
    <div class="chat-bubble" style="background:#f8fafc;color:var(--text-muted);font-style:italic;">
      ${esc(interviewAgent)} is resolving cognitive perceptions (Situation → Self → PersonBySituation)...
    </div>
  `;
  stream.appendChild(loadingEl);
  stream.scrollTop = stream.scrollHeight;

  try {
    const resp = await fetch('/api/interview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        agent: interviewAgent,
        message: msg,
        history: interviewHistory,
        cutoff_index: interviewCutoffIdx,
      }),
    });
    const data = await resp.json();
    loadingEl.remove();

    if (data.error) {
      appendChatTurn('system', `Error: ${data.error}`);
    } else {
      interviewHistory.push({
        role: 'user',
        text: msg,
        timestamp: new Date().toISOString()
      });
      interviewHistory.push({
        role: 'agent',
        text: data.reply,
        situation_perception: data.situation_perception,
        self_perception: data.self_perception,
        person_by_situation: data.person_by_situation,
        current_location: data.current_location,
        cutoff_index: data.cutoff_index,
        model: data.model,
        timestamp: new Date().toISOString()
      });
      lastConcatContext = data.concat_context || '';
      appendChatTurn('agent', data.reply, data);
    }
  } catch (err) {
    loadingEl.remove();
    appendChatTurn('system', `Network error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = '💬 Send';
  }
}

function appendChatTurn(role, text, meta) {
  const stream = document.getElementById('interviewChatStream');
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-msg ${role}`;

  if (role === 'user') {
    msgDiv.innerHTML = `
      <div class="chat-avatar user">💬</div>
      <div class="chat-bubble">
        <div style="font-size:0.68rem;opacity:0.8;margin-bottom:2px;">Researcher</div>
        <div>${esc(text)}</div>
      </div>
    `;
  } else if (role === 'agent') {
    const uniqueId = 'cot-' + Math.random().toString(36).substr(2, 9);
    const sitP = meta ? meta.situation_perception : '';
    const selfP = meta ? meta.self_perception : '';
    const pBySit = meta ? meta.person_by_situation : '';
    const isSim = meta && meta.is_simulated;
    const modelName = meta && meta.model ? meta.model : '__INTERVIEW_MODEL__';

    msgDiv.innerHTML = `
      <div class="chat-avatar agent">👤</div>
      <div class="chat-bubble" style="max-width: 90%;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-weight:700;color:var(--accent-purple);">${esc(interviewAgent)}</span>
          ${isSim ? `<span style="font-size:0.62rem;color:var(--accent-amber);background:#fef3c7;padding:1px 6px;border-radius:4px;" title="${esc(meta && meta.model_call_error || 'Offline')}">Simulated (Fallback)</span>` : `<span style="font-size:0.62rem;color:var(--accent-emerald);background:#dcfce7;padding:1px 6px;border-radius:4px;font-weight:600;">✨ ${esc(modelName)}</span>`}
          <span style="font-size:0.65rem;color:var(--text-muted);">${meta && meta.current_location ? '@ ' + esc(meta.current_location) : ''}</span>
        </div>
        <div style="font-size:0.84rem;line-height:1.5;">${esc(text)}</div>

        ${meta ? `
        <div class="cot-drawer">
          <div class="cot-toggle" onclick="toggleCot('${uniqueId}')">
            <span>🧠 Cognitive Chain-of-Thought (Components)</span>
            <span id="arrow-${uniqueId}">▾</span>
          </div>
          <div class="cot-content" id="${uniqueId}">
            <div class="cot-item">
              <div class="cot-label">📍 1. Situation Perception:</div>
              <div class="cot-val">${esc(sitP)}</div>
            </div>
            <div class="cot-item">
              <div class="cot-label">👤 2. Self Perception:</div>
              <div class="cot-val">${esc(selfP)}</div>
            </div>
            <div class="cot-item">
              <div class="cot-label">🧭 3. Person by Situation:</div>
              <div class="cot-val">${esc(pBySit)}</div>
            </div>
          </div>
        </div>
        ` : ''}
      </div>
    `;
  } else {
    msgDiv.innerHTML = `<div style="color:var(--accent-rose);font-size:0.75rem;padding:6px;width:100%;text-align:center;">${esc(text)}</div>`;
  }

  stream.appendChild(msgDiv);
  stream.scrollTop = stream.scrollHeight;
}

function toggleCot(id) {
  const el = document.getElementById(id);
  const arrow = document.getElementById('arrow-' + id);
  if (el.classList.contains('open')) {
    el.classList.remove('open');
    if (arrow) arrow.textContent = '▾';
  } else {
    el.classList.add('open');
    if (arrow) arrow.textContent = '▴';
  }
}

function resetInterviewChat() {
  interviewHistory = [];
  const locBadge = document.getElementById('interviewAgentLocBadge');
  const loc = locBadge ? locBadge.textContent.replace('📍 Loc: ', '') : 'lighthouse';
  renderWelcomeChat(interviewAgent, loc);
  const statusEl = document.getElementById('saveCloudtopStatus');
  if (statusEl) {
    statusEl.style.display = 'none';
    statusEl.textContent = '';
  }
}

function formatInterviewMarkdown() {
  let md = `# 🎙️ Interview Transcript: ${interviewAgent}\n\n`;
  md += `- **Date:** ${new Date().toISOString()}\n`;
  md += `- **Model:** __INTERVIEW_MODEL__\n`;
  md += `- **Run ID:** __RUN_ID__\n\n---\n\n`;
  interviewHistory.forEach(turn => {
    if (turn.role === 'user') {
      md += `### 🧑‍🔬 Researcher\n${turn.text}\n\n`;
    } else if (turn.role === 'agent') {
      md += `### 👤 ${interviewAgent}\n${turn.text}\n\n`;
      if (turn.situation_perception || turn.self_perception || turn.person_by_situation) {
        md += `<details><summary>🧠 Cognitive Perceptions (Components)</summary>\n\n`;
        if (turn.current_location) md += `- **Current Location:** \`${turn.current_location}\`\n`;
        if (turn.situation_perception) md += `- **Situation Perception:** ${turn.situation_perception}\n`;
        if (turn.self_perception) md += `- **Self Perception:** ${turn.self_perception}\n`;
        if (turn.person_by_situation) md += `- **Person by Situation:** ${turn.person_by_situation}\n`;
        if (turn.cutoff_index !== undefined) md += `- **Memory Cutoff Index:** ${turn.cutoff_index}\n`;
        if (turn.model) md += `- **Model:** ${turn.model}\n`;
        md += `\n</details>\n\n`;
      }
    }
  });
  return md;
}

function exportInterviewTranscript() {
  if (interviewHistory.length === 0) {
    alert('No interview messages to export yet.');
    return;
  }
  const md = formatInterviewMarkdown();
  const blob = new Blob([md], {type: 'text/markdown'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const agentSlug = interviewAgent.toLowerCase().replace(/\\s+/g, '_');
  a.href = url;
  a.download = `interview_${agentSlug}.md`;
  a.click();
}

async function saveInterviewToCloudtop() {
  if (interviewHistory.length === 0) {
    alert('No interview messages to save yet. Send an interview question first!');
    return;
  }
  const statusEl = document.getElementById('saveCloudtopStatus');
  const btn = document.getElementById('saveCloudtopBtn');
  if (btn) btn.disabled = true;
  if (statusEl) {
    statusEl.style.display = 'block';
    statusEl.style.color = 'var(--text-muted)';
    statusEl.textContent = '💾 Saving interview to cloudtop...';
  }
  try {
    const md = formatInterviewMarkdown();
    const resp = await fetch('/api/save_interview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        agent: interviewAgent,
        history: interviewHistory,
        markdown: md
      })
    });
    const res = await resp.json();
    if (res.error) {
      if (statusEl) {
        statusEl.style.color = 'var(--accent-rose)';
        statusEl.textContent = `❌ Error: ${res.error}`;
      }
      alert(`Failed to save interview: ${res.error}`);
    } else {
      if (statusEl) {
        statusEl.style.color = 'var(--accent-emerald)';
        statusEl.textContent = `✅ Saved ${res.turns_count} turns to cloudtop:\n📄 ${res.md_filename}\n📊 ${res.json_filename}`;
      }
      alert(`✅ Interview saved successfully to Cloudtop!\n\n📄 Markdown Transcript:\n${res.md_path}\n\n📊 JSON Structured Data (with cognitive components):\n${res.json_path}`);
    }
  } catch (err) {
    if (statusEl) {
      statusEl.style.color = 'var(--accent-rose)';
      statusEl.textContent = `❌ Network error: ${err.message}`;
    }
    alert(`Network error saving interview: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function openConcatModal() {
  document.getElementById('concatModalText').textContent = lastConcatContext || '(No context assembled yet. Send an interview message first to see the full ConcatActComponent context string.)';
  document.getElementById('concatModalOverlay').classList.add('open');
}

function closeConcatModal() {
  document.getElementById('concatModalOverlay').classList.remove('open');
}

function copyConcatText() {
  const txt = document.getElementById('concatModalText').textContent;
  navigator.clipboard.writeText(txt).then(() => {
    alert('ConcatAct context copied to clipboard!');
  });
}

// ---- Init ----
loadCognitive();
applyPreset();
</script>
</body>
</html>"""
  res = raw_html.replace('__DEFAULT_INTERVIEW_AGENT__', _INTERVIEW_AGENT.value)
  res = res.replace('__INTERVIEW_MODEL__', _INTERVIEW_MODEL.value)
  run_id_label = _active_run_id() or 'N/A'
  res = res.replace('__RUN_ID__', run_id_label)
  configs_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs'
  )
  event_names = sorted(
      f[: -len('_events.py')]
      for f in os.listdir(configs_dir)
      if f.endswith('_events.py')
  )
  res = res.replace(
      '__EVENT_OPTIONS__',
      '\n'.join(
          f'              <option value="{n}">{n}</option>'
          for n in event_names
      ),
  )

  # The Map tab is only available when a map_dashboard is reachable. It is
  # embedded as an iframe so that the Studio needs no extra static assets.
  map_url = _MAP_URL.value.strip()
  if map_url:
    tab_map = (
        "  <button class=\"tab-btn active\" onclick=\"switchTab('map', this)\">"
        '🗺️ Map</button>'
    )
    panel_map = (
        '<!-- Tab 0: Map (embedded map_dashboard) -->\n'
        '<div class="panel active" id="panel-map" style="flex-direction:'
        ' column;">\n'
        f'  <iframe id="mapFrame" src="{map_url}" title="Simulation map"'
        ' style="flex:1; width:100%; border:0;"></iframe>\n'
        '</div>'
    )
    cls_tab_cognitive = 'tab-btn'
    cls_panel_cognitive = 'panel'
  else:
    tab_map = ''
    panel_map = ''
    cls_tab_cognitive = 'tab-btn active'
    cls_panel_cognitive = 'panel active'
  res = res.replace('__TAB_MAP__', tab_map)
  res = res.replace('__PANEL_MAP__', panel_map)
  res = res.replace('__CLS_TAB_COGNITIVE__', cls_tab_cognitive)
  res = res.replace('__CLS_PANEL_COGNITIVE__', cls_panel_cognitive)

  header_title = '🔬 Concordia — Simulation Studio'
  doc_title = 'Concordia — Simulation Studio'
  res = res.replace('__HEADER_TITLE__', header_title)
  res = res.replace('__DOC_TITLE__', doc_title)
  return res


# ---------------------------------------------------------------------------
# Unified HTTP Server (Studio + Map Dashboard)
# ---------------------------------------------------------------------------


class _StudioHandler(http.server.BaseHTTPRequestHandler):
  """Serves the unified Simulation Studio, embedded Map Dashboard, and APIs."""

  protocol_version = "HTTP/1.1"

  def log_message(self, fmt, *args):
    del fmt, args

  def _send_bytes(self, status: int, body: bytes, ctype: str):
    self.send_response(status)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):  # pylint: disable=invalid-name
    """Serves Studio HTML, embedded Map Dashboard, and JSON GET endpoints."""
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path
    query = urllib.parse.parse_qs(parsed.query)

    if path in ("/", "/index.html"):
      body = _generate_html().encode("utf-8")
      self._send_bytes(200, body, "text/html; charset=utf-8")
      return

    if path in ("/map", "/map/index.html"):
      index = os.path.join(map_dashboard.STATIC_DIR, "index.html")
      if not os.path.isfile(index):
        self._send_bytes(
            500, b"static/index.html is missing.", "text/plain; charset=utf-8"
        )
        return
      with open(index, "rb") as f:
        self._send_bytes(200, f.read(), "text/html; charset=utf-8")
      return

    if path.startswith("/static/"):
      full = map_dashboard._resolve_static(path)  # pylint: disable=protected-access
      if not full:
        self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")
        return
      ctype, _ = mimetypes.guess_type(full)
      if full.endswith(".js"):
        ctype = "text/javascript; charset=utf-8"
      elif full.endswith(".css"):
        ctype = "text/css; charset=utf-8"
      with open(full, "rb") as f:
        self._send_bytes(200, f.read(), ctype or "application/octet-stream")
      return

    if path in ("/api/config", "/api/atlas", "/api/location_history"):
      status, payload = map_dashboard.handle_api(path, query)
      self._send_bytes(
          status,
          json.dumps(payload, default=str).encode("utf-8"),
          "application/json; charset=utf-8",
      )
      return

    if path == "/api/data":
      data = get_run_data()
      self._send_bytes(
          200,
          json.dumps(data, default=str).encode("utf-8"),
          "application/json; charset=utf-8",
      )
      return

    if path == "/api/cognitive":
      data = get_cognitive_data()
      self._send_bytes(
          200,
          json.dumps(data, default=str).encode("utf-8"),
          "application/json; charset=utf-8",
      )
      return

    self.send_error(404)

  def do_POST(self):  # pylint: disable=invalid-name
    """Handles live interview turn and interview save requests."""
    path = self.path.split("?")[0]
    length = int(self.headers.get("Content-Length", 0))
    raw = self.rfile.read(length) if length > 0 else b"{}"
    if path == "/api/interview":
      try:
        req = json.loads(raw)
        result = _execute_interview_turn(
            agent_name=req.get("agent", ""),
            user_message=req.get("message", ""),
            conversation_history=req.get("history", []),
            cutoff_index=req.get("cutoff_index"),
        )
      except Exception as e:
        result = {"error": str(e)}
      self._send_bytes(
          200, json.dumps(result).encode("utf-8"), "application/json"
      )
    elif path == "/api/save_interview":
      try:
        req = json.loads(raw)
        result = _save_interview_data(
            agent_name=req.get("agent", ""),
            history=req.get("history", []),
            markdown_content=req.get("markdown", ""),
        )
      except Exception as e:
        result = {"error": str(e)}
      self._send_bytes(
          200, json.dumps(result).encode("utf-8"), "application/json"
      )
    else:
      self.send_error(404)


class _ThreadingHTTPServer(
    socketserver.ThreadingMixIn, http.server.HTTPServer
):
  daemon_threads = True
  allow_reuse_address = True


def main(argv):
  del argv
  map_dashboard.init_source(
      run_dir=_RUN_DIR.value,
      run_id=_RUN_ID.value,
      geography=_GEOGRAPHY.value,
      personas_dir=_PERSONAS_DIR.value,
      expected_agents=_EXPECTED_AGENTS.value,
      expected_ticks=_EXPECTED_TICKS.value,
  )
  threading.Thread(target=get_cognitive_data, daemon=True).start()
  threading.Thread(target=get_run_data, daemon=True).start()

  addr = ("", _PORT.value)
  server = _ThreadingHTTPServer(addr, _StudioHandler)
  logging.info(
      "Unified Simulation Studio + Map Dashboard running at "
      "http://localhost:%d",
      _PORT.value,
  )
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    logging.info("Shutting down.")
    server.shutdown()


if __name__ == "__main__":
  app.run(main)
