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

r"""Main entry point (runner) for the Concordia Island simulation.

Reproduces the Concordia Island experiments across all cognitive
architectures (Enacted Self Agent / 'esa', Basic RAG / 'entity',
Minimal / 'minimal', and Rational / 'rational') and exogenous shock configs
(job_loss, mugged) using Concordia language
model wrappers and local file artifacts.

Example reproducing the Ohio Suburb job-loss sweep:
  python -m examples.concordia_island.run \
    --personas_date=brecksville_ohio \
    --event_config=job_loss \
    --layoff_fraction=0.8 \
    --ticks=80 \
    --agents=20 \
    --agent_prefab=esa \
    --api_type=vllm \
    --model_name=google/gemma-3-27b-it \
    --first_dates \
    --poll=False \
    --output_dir=./local_runs/run_ohio_job_loss

Local fast verification with deterministic MockLanguageModel:
  python -m examples.concordia_island.run \
    --personas_date=brecksville_ohio \
    --event_config=job_loss \
    --layoff_fraction=0.5 \
    --ticks=4 \
    --agents=4 \
    --agent_prefab=esa \
    --use_mock=True \
    --output_dir=/tmp/concordia_island_test_run
"""

from collections.abc import Mapping
import datetime as datetime_mod
import importlib
import json
import os
import random
import sys
import time
from typing import Any, NamedTuple

# Ensure repository root is on sys.path when invoked directly as a script.
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

# pylint: disable=g-import-not-at-top,g-bad-import-order
from absl import app
from absl import flags
from absl import logging
from concordia.contrib import language_models as contrib_language_models
from concordia.language_model import retry_wrapper
from concordia.typing import entity_component
from examples.concordia_island import configs
from examples.concordia_island import island_simulation
from examples.concordia_island import mock_language_model
from examples.concordia_island.personas import generator as persona_generator
from examples.concordia_island.sim import agents as agents_lib
from examples.concordia_island.sim import dating_scheduler
from examples.concordia_island.sim import event_schedule
from examples.concordia_island.sim import experience_reflection as experience_reflection_lib
from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import locations
from examples.concordia_island.sim import social_scheduler as social_scheduler_lib
from examples.concordia_island.sim import structured_log_adapter
import numpy as np
# pylint: enable=g-import-not-at-top,g-bad-import-order

FLAGS = flags.FLAGS

_AGENTS = flags.DEFINE_integer(
    "agents", 10, "Number of agents to run (default: 10)."
)
_POPULATION_OFFSET = flags.DEFINE_integer(
    "population_offset", 0, "Offset into the population list (default: 0)."
)
_TICKS = flags.DEFINE_integer(
    "ticks", 5, "Number of ticks to run (5 for testing, 8 = 1 waking day)."
)
_EVENT_CONFIG = flags.DEFINE_string(
    "event_config",
    "",
    "Name of the event config module to use (job_loss or mugged)."
    " Empty string means no exogenous events.",
)
_LAID_OFF_AGENTS = flags.DEFINE_string(
    "laid_off_agents",
    "",
    "Comma-separated list of agent names to lay off.",
)
_LAYOFF_FRACTION = flags.DEFINE_float(
    "layoff_fraction", 0.0, "Fraction of agents to lay off randomly."
)
_API_TYPE = flags.DEFINE_string(
    "api_type",
    "google_aistudio",
    "Concordia language_model_setup api_type (e.g. google_aistudio, gemini,"
    " openai, ollama, vllm, huggingface, pytorch_gemma, together_ai, groq,"
    " mistral, amazon_bedrock).",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "gemini-2.5-flash",
    "Model name passed to language_model_setup (e.g. gemini-2.5-flash,"
    " google/gemma-3-27b-it, gpt-4o).",
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    "",
    "Optional API key for cloud LLM providers (or set $GOOGLE_API_KEY /"
    " $GEMINI_API_KEY / $OPENAI_API_KEY).",
)
_DEVICE = flags.DEFINE_string(
    "device",
    "",
    "Optional device specification for local HuggingFace/PyTorch models.",
)
_USE_MOCK = flags.DEFINE_bool(
    "use_mock",
    False,
    "Explicitly use deterministic MockLanguageModel for offline testing."
    " Never enabled automatically as a fallback.",
)
_API_ADDRESS = flags.DEFINE_string(
    "api_address",
    "",
    "Legacy flag for model server URI. In third-party builds, use --api_type"
    " and --model_name instead.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Local directory to write simulation artifacts (simulation_state.json,"
    " simulation_structured.json, simulation_log.html, entity_memories.json,"
    " location_history.json, experience_data.json).",
)
_CNS_PATH = flags.DEFINE_string(
    "cns_path",
    "",
    "Legacy output directory alias (mapped to --output_dir if set).",
)
_AGENT_PREFAB = flags.DEFINE_string(
    "agent_prefab",
    "entity",
    "The agent prefab to use: 'entity', 'convo', 'esa', 'minimal', or"
    " 'rational'.",
)
_START_TIME = flags.DEFINE_string(
    "start_time", "Thursday, January 1st, 7:00 AM", "Starting simulation time."
)
_EMBEDDER_TYPE = flags.DEFINE_enum(
    "embedder_type",
    "dummy",
    ["dummy", "sentence_transformers"],
    "Type of embedder to use ('dummy' or 'sentence_transformers').",
)
_EMBEDDING_DIMENSION = flags.DEFINE_integer(
    "embedding_dimension", 768, "Dimension for dummy embedding vectors."
)
_PERSONAS_DATE = flags.DEFINE_string(
    "personas_date",
    "brecksville_ohio",
    "Persona set: 'brecksville_ohio' (bundled) or the label/path of a set"
    " built with personas/generate_personas.py.",
)
_PERSONAS_DIR = flags.DEFINE_string(
    "personas_dir",
    "",
    "Optional local directory containing persona JSON files.",
)
_SETTING = flags.DEFINE_string(
    "setting",
    "",
    "sim/locations.py setting preset (island, kerala, lagos, ohio_suburb,"
    " brecksville_1000). Empty picks it from the persona set: the"
    " 'setting_preset' in its metadata.json, else its label (Brecksville and"
    " Ohio labels use brecksville_1000), else the Concordia Island default.",
)
_ENGINE_TYPE = flags.DEFINE_enum(
    "engine_type",
    "async",
    ["sequential", "async", "simultaneous"],
    "Engine type (sequential for debugging, async for concurrent execution).",
)
_USE_AISTUDIO = flags.DEFINE_bool(
    "use_aistudio", False, "Use Gemini API via AI Studio."
)
_AISTUDIO_API_KEY = flags.DEFINE_string(
    "aistudio_api_key", "", "AI Studio API key."
)
_AISTUDIO_MODEL = flags.DEFINE_string(
    "aistudio_model",
    "gemini-2.5-flash",
    "AI Studio model name.",
)
_TICK_INTERVAL = flags.DEFINE_integer(
    "tick_interval",
    120,
    "Minutes per tick (default 120 = 8 ticks/day).",
)
_POLL = flags.DEFINE_bool(
    "poll",
    True,
    "Run a final survey via ExperienceReflection on agents after simulation.",
)
_MAX_CONV_TURNS = flags.DEFINE_integer(
    "max_conv_turns",
    8,
    "Maximum turns per conversation (default 8).",
)
_COOLDOWN_TICKS = flags.DEFINE_integer(
    "cooldown_ticks",
    4,
    "Ticks before two agents can converse again (default 4).",
)
_CONVO_AGENT = flags.DEFINE_boolean(
    "convo_agent",
    True,
    "If true, use the convo entity prefab when --agent_prefab is default.",
)
_EXPERIENCE_SAMPLING = flags.DEFINE_boolean(
    "experience_sampling",
    True,
    "If true, run ESM affect, Big Five, SWLS, GHQ-12, and MEMS daily.",
)
_BIG_FIVE = flags.DEFINE_enum(
    "big_five",
    "bfi10",
    ["bfi10", "bfi2"],
    "Big Five battery for --experience_sampling: the 10-item BFI-10 used for"
    " the paper's figures (logged as 'bfi10') or the 60-item BFI-2 (logged as"
    " 'bfi2').",
)
_REMOVE_TRAITS_FROM_MEMORY = flags.DEFINE_bool(
    "remove_traits_from_memory",
    True,
    "If true, do not add raw numerical traits to agent memories.",
)
_LOCAL = flags.DEFINE_bool(
    "local",
    True,
    "Run simulation locally (always true in third-party runner).",
)
_FIRST_DATES = flags.DEFINE_bool(
    "first_dates",
    True,
    "Enable first-date social events at 7 PM.",
)
_DATES_SCHEDULE_FILE = flags.DEFINE_string(
    "dates_schedule_file", "", "Path to JSON file with dates schedule."
)
_NIGHTTIME_SOCIAL = flags.DEFINE_bool(
    "nighttime_social",
    False,
    "Enable nighttime X social media simulation during 11PM-7AM.",
)
_ENABLE_X = flags.DEFINE_bool(
    "enable_x",
    False,
    "Enable nighttime X social media and dating Game Master ('x_rules') during"
    " 11PM-7AM (alias for --nighttime_social).",
)
_NIGHTTIME_SOCIAL_MODE = flags.DEFINE_enum(
    "nighttime_social_mode",
    "combined",
    ["dating", "social", "combined"],
    "Mode for nighttime X social media (dating, social, or combined).",
)
_X_ROUNDS = flags.DEFINE_integer(
    "x_rounds",
    2,
    "Number of rounds per agent on X during each night session.",
)
_ENABLE_NIGHTTIME_MARKETPLACE = flags.DEFINE_bool(
    "enable_nighttime_marketplace",
    False,
    "Enable nighttime goods Marketplace Game Master ('marketplace_rules')"
    " during 11PM-7AM. Off by default: the job-loss sweep runs used only"
    " 'island rules'; the fiscal-policy runs set this to True.",
)
_MARKETPLACE_ROUNDS = flags.DEFINE_integer(
    "marketplace_rounds",
    5,
    "Number of rounds per agent in the Marketplace during each night session.",
)
_FISCAL_CONFIG = flags.DEFINE_enum(
    "fiscal_config",
    "control",
    ["control", "ubi"],
    "Fiscal policy arm (see sim/fiscal_configs.py). 'control' runs weekly "
    "payroll and rent only; 'ubi' adds a $125/week cash transfer.",
)


class DummyEmbedder:
  """A dummy embedder returning a fixed-size zero vector."""

  def __init__(self, dimension: int = 768):
    self._zero_vector = np.zeros(dimension, dtype=np.float32)

  def __call__(self, text: str) -> np.ndarray:
    del text
    return self._zero_vector


class ExperimentConfig(NamedTuple):
  """Configuration for a Concordia Island simulation experiment."""

  agents: int
  ticks: int
  output_dir: str
  start_time: str
  embedder_type: str
  embedding_dimension: int
  engine_type: str
  personas_date: str = "brecksville_ohio"
  personas_dir: str = ""
  setting: str = ""
  api_type: str = "google_aistudio"
  model_name: str = "gemini-2.5-flash"
  api_key: str = ""
  device: str = ""
  use_mock: bool = False
  api_address: str = ""
  use_aistudio: bool = False
  aistudio_api_key: str = ""
  aistudio_model: str = "gemini-2.5-flash"
  tick_interval: int = 120
  poll: bool = False
  max_conv_turns: int = 8
  cooldown_ticks: int = 4
  convo_agent: bool = False
  agent_prefab: str = "entity"
  experience_sampling: bool = True
  big_five: str = "bfi10"
  remove_traits_from_memory: bool = True
  event_config: str = ""
  laid_off_agents: str = ""
  first_dates: bool = False
  population_offset: int = 0
  layoff_fraction: float = 0.0
  dates_schedule_file: str = ""
  nighttime_social: bool = False
  nighttime_social_mode: str = "combined"
  x_rounds: int = 2
  enable_nighttime_marketplace: bool = False
  marketplace_rounds: int = 5
  fiscal_config: str = "control"


def _get_sim_snapshot(sim) -> Mapping[str, Any]:
  """Extract current simulation state from GM components."""
  snapshot: dict[str, Any] = {
      "tick": -1,
      "sim_time": "unknown",
      "locations": {},
      "conversations_active": 0,
      "total_utterances": 0,
      "agents_in_conversation": [],
      "agents_free": [],
  }
  try:
    gm = sim.game_masters[0]
    try:
      clock = gm.get_component("clock")
      snapshot["tick"] = clock.current_tick
      snapshot["sim_time"] = clock.get_pre_act_value().strip()
    except (AttributeError, KeyError):
      pass
    try:
      loc_component = gm.get_component("locations")
      loc_state = loc_component.get_state()
      snapshot["locations"] = dict(loc_state.get("entity_locations", {}))
    except (AttributeError, KeyError):
      pass
    try:
      conv_state = gm.get_component("async_conversation_state")
      cs = conv_state.get_state()
      active = cs.get("conversations", {})
      snapshot["conversations_active"] = len(active)
      total_utts = sum(len(c.get("utterances", [])) for c in active.values())
      snapshot["total_utterances"] = total_utts
      all_players = cs.get("players", [])
      in_conv = [p for p in all_players if conv_state.is_in_conversation(p)]
      snapshot["agents_in_conversation"] = in_conv
      snapshot["agents_free"] = [p for p in all_players if p not in in_conv]
    except (AttributeError, KeyError):
      pass
    try:
      clock = gm.get_component("clock")
      clock_state = clock.get_state()
      snapshot["clock_acted"] = clock_state.get("agents_acted", [])
    except (AttributeError, KeyError):
      pass
    try:
      trigger = gm.get_component("conversation_director")
      trigger_state = trigger.get_state()
      snapshot["total_conversations"] = trigger_state.get(
          "total_conversations", 0
      )
    except (AttributeError, KeyError):
      snapshot["total_conversations"] = 0
  except (IndexError, AttributeError):
    pass
  return snapshot


def _build_language_model(config: ExperimentConfig):
  """Initializes the Concordia language model without canned fallbacks."""
  if config.use_mock:
    logging.info("Initializing MockLanguageModel (--use_mock=True).")
    return mock_language_model.MockLanguageModel()

  if config.api_address:
    raise ValueError(
        f"--api_address={config.api_address!r} is not supported. In the "
        "third-party Concordia release, specify a public LLM provider via "
        "--api_type (e.g. google_aistudio, openai, vllm, ollama, huggingface) "
        "and --model_name (e.g. gemini-2.5-flash, google/gemma-3-27b-it), or "
        "pass --use_mock=True for deterministic offline testing."
    )

  api_type = "google_aistudio" if config.use_aistudio else config.api_type
  model_name = (
      config.aistudio_model if config.use_aistudio else config.model_name
  )
  api_key = (
      config.aistudio_api_key
      or config.api_key
      or os.environ.get("GOOGLE_API_KEY", "")
      or os.environ.get("GEMINI_API_KEY", "")
      or os.environ.get("OPENAI_API_KEY", "")
      or None
  )

  logging.info(
      "Initializing Concordia language model (api_type=%s, model_name=%s)...",
      api_type,
      model_name,
  )
  base_model = contrib_language_models.language_model_setup(
      api_type=api_type,
      model_name=model_name,
      api_key=api_key,
      device=config.device or None,
  )

  return retry_wrapper.RetryLanguageModel(
      base_model,
      retry_tries=5,
      retry_delay=10.0 if "gemma" in model_name.lower() else 5.0,
      jitter=(1.0, 5.0),
      exponential_backoff=True,
      backoff_factor=2.0,
      max_delay=300.0,
  )


def _setting_for_personas(
    personas_date: str, personas_dir: str = "", setting: str = ""
) -> str | None:
  """Returns the sim/locations.py setting preset for a persona set.

  The setting decides the public places and home buildings the game master
  knows about, so it has to match the `home_place` and `work_place` values in
  the personas.

  Args:
    personas_date: Persona set label or path (as passed to `load_personas`).
    personas_dir: Optional base directory the label is resolved against.
    setting: Explicit preset name; wins over everything else.

  Returns:
    A preset name, or None for the default Concordia Island setting.

  Raises:
    ValueError: If `setting` (or the persona set's `setting_preset`) is not a
      known preset.
  """
  if not setting:
    for base in (personas_dir, persona_generator.DEFAULT_PERSONAS_BASE_PATH):
      meta_path = os.path.join(base or "", personas_date, "metadata.json")
      if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
          setting = json.load(f).get("setting_preset", "")
        break
  if setting:
    if setting not in locations.SETTING_PRESETS:
      raise ValueError(
          f"Unknown setting {setting!r}; expected one of"
          f" {sorted(locations.SETTING_PRESETS)}."
      )
    return setting
  label = os.path.basename(os.path.normpath(personas_date)).lower()
  if "brecksville" in label or "ohio" in label:
    # The bundled 1000-person Brecksville set uses the brecksville_1000 homes
    # (millbrook_apts, ...) and public places.
    return "brecksville_1000"
  preset = locations.get_setting(label)
  return preset.name if preset else None


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
  """Runs the Concordia Island simulation experiment and writes local artifacts."""
  experiment_start = time.time()

  output_dir = config.output_dir
  if not output_dir:
    ts = datetime_mod.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(f"./local_runs/run_{ts}")
  os.makedirs(output_dir, exist_ok=True)
  logging.info("Artifact output directory: %s", output_dir)

  laid_off_str = config.laid_off_agents
  laid_off_names = [n.strip() for n in laid_off_str.split(",") if n.strip()]

  def _select_agents(all_agents, seed, num_agents, offset, laid_off_list):
    laid_off_configs = [cfg for cfg in all_agents if cfg.name in laid_off_list]
    remaining_agents = [
        cfg for cfg in all_agents if cfg.name not in laid_off_list
    ]
    random.Random(seed).shuffle(remaining_agents)
    num_needed = num_agents - len(laid_off_configs)
    if num_needed < 0:
      raise ValueError(
          f"Requested {num_agents} agents, but specified"
          f" {len(laid_off_configs)} laid off agents!"
      )
    return laid_off_configs + remaining_agents[offset : offset + num_needed]

  personas_date = config.personas_date or "brecksville_ohio"

  personas = persona_generator.load_personas(
      cns_path=config.personas_dir
      or persona_generator.DEFAULT_PERSONAS_BASE_PATH,
      date_label=personas_date,
  )
  logging.info("Loaded %d personas for agent config generation.", len(personas))

  all_agents = []
  for pdata in personas.values():
    all_agents.append(
        agents_lib.AgentConfig(
            name=pdata.name,
            home_place=pdata.home_place,
            work_place=pdata.work_place,
            personality=pdata.personality,
            backstory=pdata.original_backstory,
            age=pdata.age,
            gender=getattr(pdata, "gender", ""),
            sexual_orientation=getattr(pdata, "sexual_orientation", ""),
            ethnicity=getattr(pdata, "ethnicity", ""),
            political_orientation=getattr(pdata, "political_orientation", ""),
            hobbies=getattr(pdata, "hobbies", []),
            relationship_status=getattr(pdata, "relationship_status", "single"),
            neighborhood=getattr(pdata, "neighborhood", ""),
        )
    )

  _PLACE_PREFIX_MAP = {  # pylint: disable=invalid-name
      "millbrook_apts": "sunset_apartments",
      "brecksville_commons": "coral_village",
      "chippewa_ridge": "palm_heights",
      "riverview_estates": "ocean_view_estates",
      "timber_creek": "paradise_point",
      "echo_park_apts": "sunset_apartments",
      "silverlake_courts": "coral_village",
      "griffith_heights": "palm_heights",
      "hillhurst_villas": "ocean_view_estates",
      "elysian_crest": "paradise_point",
      "canal_row_flats": "sunset_apartments",
      "thottam_colony": "coral_village",
      "paddy_view_villas": "palm_heights",
      "backwater_estates": "ocean_view_estates",
      "coconut_grove": "paradise_point",
      "eko_flats": "sunset_apartments",
      "surulere_courts": "coral_village",
      "gbagada_heights": "palm_heights",
      "ikoyi_estates": "ocean_view_estates",
      "banana_island": "paradise_point",
  }
  for agent in all_agents:
    for prefix, canonical in _PLACE_PREFIX_MAP.items():
      if agent.home_place and agent.home_place.startswith(prefix):
        agent.home_place = agent.home_place.replace(prefix, canonical, 1)

  all_agents.sort(key=lambda a: a.name)
  if laid_off_names:
    agent_configs = _select_agents(
        all_agents,
        42,
        config.agents,
        config.population_offset,
        laid_off_names,
    )
  else:
    random.Random(42).shuffle(all_agents)
    agent_configs = all_agents[
        config.population_offset : config.population_offset + config.agents
    ]

  logging.info(
      "Selected %d agents (offset=%d): %s",
      len(agent_configs),
      config.population_offset,
      [a.name for a in agent_configs],
  )

  model = _build_language_model(config)

  if config.embedder_type == "sentence_transformers":
    from sentence_transformers import SentenceTransformer  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    embedder_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embedder(text: str) -> np.ndarray:
      return embedder_model.encode(text)
  else:
    embedder = DummyEmbedder(config.embedding_dimension)

  social_events = []
  if config.first_dates:
    active_names = {cfg.name for cfg in agent_configs}
    venues = [
        "restaurant",
        "swell_bar",
        "rooftop_lounge",
        "lighthouse_hike",
        "community_center",
    ]
    if config.dates_schedule_file:
      with open(config.dates_schedule_file, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)
      raw_events = []
      for group_events in schedule_data.values():
        raw_events.extend(group_events)
      for e in raw_events:
        if all(p in active_names for p in e["participants"]):
          social_events.append(
              social_scheduler_lib.SocialEvent(
                  participants=tuple(e["participants"]),
                  venue=e["venue"],
                  theme=e["theme"],
                  day=e["day"],
                  tick_hour=e["tick_hour"],
                  prompt_type=e["prompt_type"],
              )
          )

    paired_agents = set()
    for e in social_events:
      paired_agents.update(e.participants)
    unpaired = active_names - paired_agents
    num_days = max(1, config.ticks // 8)
    new_events = dating_scheduler.generate_dating_schedule(
        active_names=active_names,
        unpaired_names=unpaired,
        personas=personas,
        num_days=num_days,
        venues=venues,
    )
    social_events.extend(new_events)

  setting = _setting_for_personas(
      personas_date, config.personas_dir, config.setting
  )
  logging.info("Location setting: %s", setting or "island (default)")

  sim = island_simulation.IslandConcordiaSimulation(
      agent_configs=agent_configs,
      model=model,
      embedder=embedder,
      start_time=config.start_time,
      engine_type=config.engine_type,
      personas=personas,
      max_ticks=config.ticks,
      tick_interval_minutes=config.tick_interval,
      max_conv_turns=config.max_conv_turns,
      cooldown_ticks=config.cooldown_ticks,
      convo_agent=config.convo_agent,
      agent_prefab=config.agent_prefab,
      experience_sampling=config.experience_sampling,
      big_five=config.big_five,
      remove_traits_from_memory=config.remove_traits_from_memory,
      social_events=social_events,
      enable_nighttime_social=config.nighttime_social,
      nighttime_social_mode=config.nighttime_social_mode,
      x_rounds=config.x_rounds,
      enable_nighttime_marketplace=config.enable_nighttime_marketplace,
      marketplace_rounds=config.marketplace_rounds,
      fiscal_config=config.fiscal_config,
      setting=setting,
  )

  event_config_name = config.event_config
  if config.layoff_fraction > 0:
    if laid_off_names:
      raise ValueError(
          "Cannot specify both laid_off_agents and layoff_fraction"
      )
    num_to_layoff = round(len(agent_configs) * config.layoff_fraction)
    rng = random.Random(42 + config.population_offset)
    sorted_names = sorted([cfg.name for cfg in agent_configs])
    laid_off_names = rng.sample(sorted_names, num_to_layoff)
    logging.info(
        "Randomly selected %d agents for layoff (fraction %.2f): %s",
        num_to_layoff,
        config.layoff_fraction,
        laid_off_names,
    )

  if laid_off_names:
    with open(
        os.path.join(output_dir, "laid_off_agents.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(laid_off_names, f, indent=2)

  with open(
      os.path.join(output_dir, "agent_names.json"), "w", encoding="utf-8"
  ) as f:
    json.dump([cfg.name for cfg in agent_configs], f, indent=2)

  if event_config_name:
    if event_config_name.endswith("_events"):
      config_module_name = f"{configs.__name__}.{event_config_name}"
    else:
      config_module_name = f"{configs.__name__}.{event_config_name}_events"
    config_module = importlib.import_module(config_module_name)
    get_events_fn = getattr(config_module, "get_events")
    events = get_events_fn(laid_off_names, agent_configs)
    scheduler = event_schedule.EventScheduler(events, sim)
    logging.info("EventScheduler initialized with %d events", len(events))
  else:
    scheduler = None

  step_count = [0]
  total_start_time = time.time()
  per_tick_locations: dict[int, dict[str, str]] = {}
  raw_log: list[Mapping[str, Any]] = []

  def step_callback(step_data: Any) -> None:
    del step_data
    step = step_count[0]
    step_count[0] += 1
    snap = _get_sim_snapshot(sim)
    tick = max(1, int(snap.get("tick", 1) or 1))
    locs = snap.get("locations", {})
    if locs:
      per_tick_locations.setdefault(tick, {}).update(locs)

    if scheduler is not None:
      gm = sim.game_masters[0]
      clock = gm.get_component("clock", type_=fixed_clock.FixedIntervalClock)
      clock_state = clock.get_state()
      current_dt = datetime_mod.datetime.fromisoformat(
          clock_state["current_dt_iso"]
      )
      scheduler.check_and_deliver(current_dt)

    if step > 0 and step % 5 == 0:
      with open(
          os.path.join(output_dir, "progress.json"), "w", encoding="utf-8"
      ) as f:
        json.dump(
            {
                "current_step": step,
                "current_tick": tick,
                "sim_time": snap.get("sim_time", ""),
                "total_ticks": config.ticks,
                "log_entries": len(raw_log),
                "elapsed_seconds": time.time() - total_start_time,
                "locations": locs,
                "conversations_active": snap.get("conversations_active", 0),
                "total_utterances": snap.get("total_utterances", 0),
            },
            f,
            indent=2,
        )

  results = sim.play(
      premise="A peaceful day begins in the community.",
      max_ticks=config.ticks,
      raw_log=raw_log,
      step_callback=step_callback,
  )

  if config.poll:
    for entity in sim.entities:
      try:
        er = entity.get_component(
            "ExperienceReflection",
            type_=experience_reflection_lib.ExperienceReflection,
        )
        entity.set_phase(entity_component.Phase.READY)
        er.run_final_survey()
      except (KeyError, AttributeError):
        pass

  elapsed = time.time() - experiment_start
  final_snap = _get_sim_snapshot(sim)

  # Save ExperienceReflection psychometric results
  experience_data = {}
  for entity in sim.entities:
    try:
      er = entity.get_component(
          "ExperienceReflection",
          type_=experience_reflection_lib.ExperienceReflection,
      )
      experience_data[entity.name] = er.get_results()
    except (KeyError, AttributeError):
      pass
  if experience_data:
    with open(
        os.path.join(output_dir, "experience_data.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(experience_data, f, indent=2, default=str)

  # Save structured & HTML logs
  structured_dict = json.loads(results.to_json())
  with open(
      os.path.join(output_dir, "simulation_structured.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump(structured_dict, f, indent=2)

  if "entity_memories" in structured_dict:
    with open(
        os.path.join(output_dir, "entity_memories.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(structured_dict["entity_memories"], f, indent=2)

  with open(
      os.path.join(output_dir, "simulation_log.html"), "w", encoding="utf-8"
  ) as f:
    f.write(results.to_html())

  with open(
      os.path.join(output_dir, "raw_log.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(
        [structured_log_adapter.sanitize_for_json(dict(e)) for e in raw_log],
        f,
        indent=2,
    )

  # Save location_history.json for Map Dashboard
  ticks_sorted = sorted(per_tick_locations.keys()) or [1]
  running: dict[str, str] = {}
  last_seen: dict[str, int] = {}
  snapshots: dict[str, list[dict[str, Any]]] = {}
  observed: dict[str, int] = {}
  tick_progress = []
  for t in ticks_sorted:
    obs_t = per_tick_locations.get(t, final_snap.get("locations", {}))
    running.update(obs_t)
    for a in obs_t:
      last_seen[a] = t
    snapshots[str(t)] = [
        {"agent": a, "location": l, "age": t - last_seen.get(a, t)}
        for a, l in sorted(running.items())
    ]
    observed[str(t)] = len(obs_t)
    tick_progress.append({"step": t, "agents": len(running)})

  with open(
      os.path.join(output_dir, "location_history.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(
        {
            "ticks": ticks_sorted,
            "snapshots": snapshots,
            "observed_counts": observed,
            "latest_locations": (
                snapshots.get(str(ticks_sorted[-1]), []) if ticks_sorted else []
            ),
            "error": None,
        },
        f,
        indent=2,
    )

  state_data = {
      "agents": config.agents,
      "ticks": config.ticks,
      "tick_interval_minutes": config.tick_interval,
      "log_entries": len(raw_log),
      "start_time": config.start_time,
      "final_tick": final_snap.get("tick", config.ticks),
      "final_sim_time": final_snap.get("sim_time", "unknown"),
      "elapsed_seconds": elapsed,
      "locations": final_snap.get("locations", {}),
      "conversations_active": final_snap.get("conversations_active", 0),
      "agents_in_conversation": final_snap.get("agents_in_conversation", []),
      "tick_progress": tick_progress,
  }
  with open(
      os.path.join(output_dir, "simulation_state.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(state_data, f, indent=2)

  perf_data = {
      "agents": config.agents,
      "ticks_configured": config.ticks,
      "ticks_completed": final_snap.get("tick", config.ticks),
      "tick_interval_minutes": config.tick_interval,
      "total_steps": step_count[0],
      "total_wall_time_seconds": round(elapsed, 2),
      "engine_type": config.engine_type,
      "agent_prefab": config.agent_prefab,
  }
  with open(
      os.path.join(output_dir, "performance.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(perf_data, f, indent=2)

  logging.info("Saved all simulation artifacts to %s", output_dir)
  return state_data


def main(argv):
  del argv
  output_dir = _OUTPUT_DIR.value or _CNS_PATH.value
  config = ExperimentConfig(
      agents=_AGENTS.value,
      ticks=_TICKS.value,
      output_dir=output_dir,
      start_time=_START_TIME.value,
      embedder_type=_EMBEDDER_TYPE.value,
      embedding_dimension=_EMBEDDING_DIMENSION.value,
      engine_type=_ENGINE_TYPE.value,
      personas_date=_PERSONAS_DATE.value,
      personas_dir=_PERSONAS_DIR.value,
      setting=_SETTING.value,
      api_type=_API_TYPE.value,
      model_name=_MODEL_NAME.value,
      api_key=_API_KEY.value,
      device=_DEVICE.value,
      use_mock=_USE_MOCK.value,
      api_address=_API_ADDRESS.value,
      use_aistudio=_USE_AISTUDIO.value,
      aistudio_api_key=_AISTUDIO_API_KEY.value,
      aistudio_model=_AISTUDIO_MODEL.value,
      tick_interval=_TICK_INTERVAL.value,
      poll=_POLL.value,
      max_conv_turns=_MAX_CONV_TURNS.value,
      cooldown_ticks=_COOLDOWN_TICKS.value,
      convo_agent=_CONVO_AGENT.value,
      agent_prefab=_AGENT_PREFAB.value,
      experience_sampling=_EXPERIENCE_SAMPLING.value,
      big_five=_BIG_FIVE.value,
      remove_traits_from_memory=_REMOVE_TRAITS_FROM_MEMORY.value,
      event_config=_EVENT_CONFIG.value,
      laid_off_agents=_LAID_OFF_AGENTS.value,
      first_dates=_FIRST_DATES.value,
      population_offset=_POPULATION_OFFSET.value,
      layoff_fraction=_LAYOFF_FRACTION.value,
      dates_schedule_file=_DATES_SCHEDULE_FILE.value,
      nighttime_social=bool(_NIGHTTIME_SOCIAL.value or _ENABLE_X.value),
      nighttime_social_mode=_NIGHTTIME_SOCIAL_MODE.value,
      x_rounds=_X_ROUNDS.value,
      enable_nighttime_marketplace=_ENABLE_NIGHTTIME_MARKETPLACE.value,
      marketplace_rounds=_MARKETPLACE_ROUNDS.value,
      fiscal_config=_FISCAL_CONFIG.value,
  )
  run_experiment(config)


if __name__ == "__main__":
  app.run(main)
