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

"""Concordia-based Island Simulation.

Full Concordia integration with:
- situated_in_time_and_place__GameMaster pattern
- Asynchronous engine for parallel per-agent loops
- Async conversation components for multi-turn dialogues
- Proper HTML logging via SimulationLog.to_html()
"""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.environment.engines import asynchronous
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib

from examples.concordia_island.prefabs import esa_island_entity
from examples.concordia_island.prefabs import island_convo_entity
from examples.concordia_island.prefabs import island_entity
from examples.concordia_island.prefabs import island_gm
from examples.concordia_island.prefabs import marketplace_night_gm
from examples.concordia_island.prefabs import minimal_island_entity
from examples.concordia_island.prefabs import rational_island_entity
from examples.concordia_island.prefabs import xlike_gm
from examples.concordia_island.sim import agents as agents_lib
from examples.concordia_island.sim import locations as locations_lib


def create_island_config(
    agent_configs: Sequence[agents_lib.AgentConfig],
    start_time: str = "Thursday, January 1st, 7:00 AM",
    personas: dict[str, Any] | None = None,
    use_relevant_memories: bool = False,
    convo_agent: bool = False,
    agent_prefab: str | None = None,
    remove_traits_from_memory: bool = True,
    social_events: Any = None,
    enable_nighttime_social: bool = False,
    nighttime_social_mode: str = "combined",
    x_rounds: int = 2,
    enable_nighttime_marketplace: bool = False,
    marketplace_rounds: int = 5,
    setting: str | None = None,
    fiscal_config: str = "control",
    food_min_daily: int = 3,
    starting_food_units: int = 12,
    annual_interest_rate: float = 0.035,
    economic_profiles: dict[str, Any] | None = None,
    laid_off_agents: Sequence[str] = (),
    clock: Any | None = None,
    max_ticks: int | None = None,
    tick_interval_minutes: int = 120,
) -> prefab_lib.Config:
  """Create a Concordia Config for island simulation.

  Args:
    agent_configs: List of AgentConfig for each island resident
    start_time: Starting time string
    personas: Optional dict mapping agent name -> PersonaData with memories
    use_relevant_memories: If True, use relevant memories component.
    convo_agent: If True, use convo entity with PinkNoiseStrategy.
    agent_prefab: The agent prefab to use.
    remove_traits_from_memory: If True, remove traits from memory.
    social_events: List of social events for the GM.
    enable_nighttime_social: If True, enable nighttime X social network GM.
    nighttime_social_mode: Mode for nighttime X GM: dating, social, or
      combined.
    x_rounds: Number of rounds per nighttime X session.
    enable_nighttime_marketplace: If True, enable nighttime marketplace.
    marketplace_rounds: Number of rounds per marketplace night session.
    setting: The setting name for customization.
    fiscal_config: Fiscal configuration name (control or ubi).
    food_min_daily: Minimum daily food units needed per person.
    starting_food_units: Initial food units in pantry per household.
    annual_interest_rate: Annual interest rate on savings.
    economic_profiles: Optional pre-built economic profiles dict.
    laid_off_agents: List of agent names who are laid off.
    clock: Optional shared FixedIntervalClock instance.
    max_ticks: Optional maximum number of clock ticks.
    tick_interval_minutes: Minutes per clock tick.

  Returns:
    Concordia Config ready for Simulation
  """
  from examples.concordia_island.sim import economic_profile as econ_prof_lib  # pylint: disable=g-import-not-at-top

  if economic_profiles is None and agent_configs:
    economic_profiles = econ_prof_lib.build_economic_profiles(
        agent_configs,
        food_min_daily=food_min_daily,
        laid_off_agents=laid_off_agents,
        starting_food_units=starting_food_units,
    )

  # "persistent" is the ESA prefab's former name; runs recorded before the
  # rename (e.g. data/paper_runs/*/metadata.json) still carry it.
  if agent_prefab in ("esa", "persistent"):
    entity_prefab_name = "island__ESAEntity"
  elif agent_prefab == "convo":
    entity_prefab_name = "island__ConvoEntity"
  elif agent_prefab == "entity":
    entity_prefab_name = "island__Entity"
  elif agent_prefab == "minimal":
    entity_prefab_name = "island__MinimalEntity"
  elif agent_prefab == "rational":
    entity_prefab_name = "island__RationalEntity"
  else:
    entity_prefab_name = (
        "island__ConvoEntity" if convo_agent else "island__Entity"
    )

  prefabs = {
      "island__Entity": island_entity.IslandEntity(),
      "island__ConvoEntity": island_convo_entity.IslandConvoEntity(),
      "island__ESAEntity": esa_island_entity.ESAIslandEntity(),
      "island__MinimalEntity": minimal_island_entity.MinimalIslandEntity(),
      "island__RationalEntity": rational_island_entity.RationalIslandEntity(),
      "island__GameMaster": island_gm.IslandGameMaster(),
      "island__XLikeGM": xlike_gm.XLikeGameMaster(),
      "island__MarketplaceGM": (
          marketplace_night_gm.MarketplaceNightGameMaster()
      ),
  }

  instances = []

  # Build initial_locations dict for GM
  initial_locations = {}
  available_locations = locations_lib.get_all_public_location_names(
      setting=setting
  )

  for agent_config in agent_configs:
    # Each agent starts at their home
    initial_locations[agent_config.name] = agent_config.home_place

    # Generate initial observation (waking up at home)
    initial_observation = locations_lib.get_initial_observation(
        agent_name=agent_config.name,
        home_place=agent_config.home_place,
        time=start_time,
    )

    memories = []
    formative_memories = []
    traits = {}
    age = None
    if personas and agent_config.name in personas:
      memories = personas[agent_config.name].memories
      formative_memories = personas[agent_config.name].formative_memories
      traits = personas[agent_config.name].traits
      age = personas[agent_config.name].age

    entity_params: dict[str, Any] = {
        "name": agent_config.name,
        "personality": agent_config.personality,
        "backstory": agent_config.backstory,
        "home_place": agent_config.home_place,
        "work_place": agent_config.work_place,
        "available_locations": available_locations,
        "initial_observation": initial_observation,
        "memories": memories,
        "formative_memories": formative_memories,
        "traits": traits,
        "age": age,
        "remove_traits_from_memory": remove_traits_from_memory,
    }
    instance = prefab_lib.InstanceConfig(
        prefab=entity_prefab_name,
        role=prefab_lib.Role.ENTITY,
        params=entity_params,
    )
    instances.append(instance)

  if clock is None:
    from examples.concordia_island.sim import conversation as async_conv  # pylint: disable=g-import-not-at-top
    from examples.concordia_island.sim import fixed_clock  # pylint: disable=g-import-not-at-top

    clock = fixed_clock.FixedIntervalClock(
        start_time=start_time,
        tick_interval_minutes=tick_interval_minutes,
        waking_hour_start=7,
        waking_hour_end=23,
        player_names=[agent.name for agent in agent_configs],
        conversation_state_key=async_conv.DEFAULT_ASYNC_CONVERSATION_KEY,
        max_ticks=max_ticks,
        pre_act_label="\nCurrent time",
        initial_locations=initial_locations
        if isinstance(initial_locations, dict)
        else None,
        locations_key="locations",
    )

  if social_events is None:
    social_events = []

  gm_params: dict[str, Any] = {
      "name": "island rules",
      "start_time": start_time,
      "initial_locations": initial_locations,
      "use_relevant_memories": use_relevant_memories,
      "agent_configs": agent_configs,
      "social_events": social_events,
      "enable_nighttime_social": enable_nighttime_social,
      "enable_nighttime_marketplace": enable_nighttime_marketplace,
      "setting": setting,
      "fiscal_config": fiscal_config,
      "food_min_daily": food_min_daily,
      "economic_profiles": economic_profiles,
      "laid_off_agents": laid_off_agents,
      "clock": clock,
  }
  gm_instance = prefab_lib.InstanceConfig(
      prefab="island__GameMaster",
      role=prefab_lib.Role.GAME_MASTER,
      params=gm_params,
  )
  instances.append(gm_instance)

  if enable_nighttime_social:
    x_next_gm = (
        "marketplace_rules"
        if enable_nighttime_marketplace
        else "island rules"
    )
    x_gm_params: dict[str, Any] = {
        "name": "x_rules",
        "forum_name": "X",
        "island_gm_name": "island rules",
        "next_gm_name": x_next_gm,
        "mark_night_complete_on_clock": (
            not enable_nighttime_marketplace
        ),
        "max_steps": x_rounds,
        "mode": nighttime_social_mode,
        "social_events": social_events,
        "clock": clock,
    }
    x_gm_instance = prefab_lib.InstanceConfig(
        prefab="island__XLikeGM",
        role=prefab_lib.Role.GAME_MASTER,
        params=x_gm_params,
    )
    instances.append(x_gm_instance)

  if enable_nighttime_marketplace:
    marketplace_gm_params: dict[str, Any] = {
        "name": "marketplace_rules",
        "player_names": [agent.name for agent in agent_configs],
        "island_gm_name": "island rules",
        "marketplace_gm_name": "marketplace_rules",
        "max_rounds": marketplace_rounds,
        "annual_interest_rate": annual_interest_rate,
        "economic_profiles": economic_profiles,
        "clock": clock,
    }
    marketplace_gm_instance = prefab_lib.InstanceConfig(
        prefab="island__MarketplaceGM",
        role=prefab_lib.Role.GAME_MASTER,
        params=marketplace_gm_params,
    )
    instances.append(marketplace_gm_instance)

  return prefab_lib.Config(prefabs=prefabs, instances=instances)


class IslandConcordiaSimulation:
  """Concordia-based island simulation.

  Usage:
    sim = IslandConcordiaSimulation(
        agent_configs=ISLAND_AGENTS[:10],
        model=model,
        embedder=embedder,
    )
    results = sim.play(max_steps=48)
    html = results.to_html()
  """

  def __init__(
      self,
      agent_configs: Sequence[agents_lib.AgentConfig],
      model: language_model.LanguageModel,
      embedder: Any,
      start_time: str = "Thursday, January 1st, 7:00 AM",
      engine_type: str = "async",
      personas: dict[str, Any] | None = None,
      use_relevant_memories: bool = False,
      max_ticks: int | None = None,
      tick_interval_minutes: int = 120,
      max_conv_turns: int = 8,
      cooldown_ticks: int = 4,
      convo_agent: bool = False,
      experience_sampling: bool = True,
      big_five: str = "bfi10",
      agent_prefab: str | None = None,
      remove_traits_from_memory: bool = True,
      social_events: Any = None,
      enable_nighttime_social: bool = False,
      nighttime_social_mode: str = "combined",
      x_rounds: int = 2,
      enable_nighttime_marketplace: bool = False,
      marketplace_rounds: int = 5,
      setting: str | None = None,
      fiscal_config: str = "control",
      food_min_daily: int = 3,
      starting_food_units: int = 12,
      annual_interest_rate: float = 0.035,
      economic_profiles: dict[str, Any] | None = None,
      laid_off_agents: Sequence[str] = (),
      clock: Any | None = None,
  ):
    from examples.concordia_island.sim import economic_profile as econ_prof_lib  # pylint: disable=g-import-not-at-top

    if economic_profiles is None and agent_configs:
      economic_profiles = econ_prof_lib.build_economic_profiles(
          agent_configs,
          food_min_daily=food_min_daily,
          laid_off_agents=laid_off_agents,
          starting_food_units=starting_food_units,
      )
    self._economic_profiles = economic_profiles

    self._config = create_island_config(
        agent_configs,
        start_time,
        personas,
        use_relevant_memories,
        convo_agent=convo_agent,
        agent_prefab=agent_prefab,
        remove_traits_from_memory=remove_traits_from_memory,
        social_events=social_events,
        enable_nighttime_social=enable_nighttime_social,
        nighttime_social_mode=nighttime_social_mode,
        x_rounds=x_rounds,
        enable_nighttime_marketplace=enable_nighttime_marketplace,
        marketplace_rounds=marketplace_rounds,
        setting=setting,
        fiscal_config=fiscal_config,
        food_min_daily=food_min_daily,
        starting_food_units=starting_food_units,
        annual_interest_rate=annual_interest_rate,
        economic_profiles=self._economic_profiles,
        laid_off_agents=laid_off_agents,
        clock=clock,
        max_ticks=max_ticks,
        tick_interval_minutes=tick_interval_minutes,
    )
    self._model = model
    self._embedder = embedder
    self._max_ticks = max_ticks

    for instance in self._config.instances:
      mutable_params: dict[str, Any] = dict(instance.params)
      if instance.role == prefab_lib.Role.GAME_MASTER:
        if max_ticks is not None:
          mutable_params["max_ticks"] = max_ticks
        mutable_params["tick_interval_minutes"] = tick_interval_minutes
        mutable_params["max_conv_turns"] = max_conv_turns
        mutable_params["cooldown_ticks"] = cooldown_ticks
        # The island GM switches entities into the nighttime GM(s) per entity
        # under the async engine and all at once otherwise.
        mutable_params["engine_type"] = engine_type
      elif instance.role == prefab_lib.Role.ENTITY:
        # Pass tick interval so entity components (e.g. ExperienceReflection)
        # can compute time-based periods.
        mutable_params["tick_interval_minutes"] = tick_interval_minutes
        mutable_params["experience_sampling"] = experience_sampling
        mutable_params["big_five"] = big_five
      instance.params = mutable_params

    if engine_type == "sequential":
      from concordia.environment.engines import sequential  # pylint: disable=g-import-not-at-top

      self._engine = sequential.Sequential()
    elif engine_type == "async":
      self._engine = asynchronous.Asynchronous(sleep_time=0.1)
    else:
      from concordia.environment.engines import simultaneous  # pylint: disable=g-import-not-at-top

      self._engine = simultaneous.Simultaneous()

    self._sim = simulation.Simulation(
        config=self._config,
        model=self._model,
        embedder=self._embedder,
        engine=self._engine,
    )

  @property
  def economic_profiles(self) -> dict[str, Any] | None:
    return self._economic_profiles

  @property
  def entities(self):
    return self._sim.entities

  @property
  def game_masters(self):
    return self._sim.game_masters

  def play(
      self,
      premise: str | None = None,
      max_steps: int | None = None,
      max_ticks: int | None = None,
      raw_log: list[Mapping[str, Any]] | None = None,
      checkpoint_path: str | None = None,
      step_callback: Any | None = None,
      get_state_callback: Any | None = None,
  ):
    """Run the simulation.

    Args:
      premise: Initial premise/setup text
      max_steps: Maximum engine steps (safety limit). If not set and max_ticks
        is provided, defaults to max_ticks * 100.
      max_ticks: Number of clock ticks to run. The simulation terminates when
        the FixedIntervalClock reaches this many ticks. This is the preferred
        way to control simulation duration.
      raw_log: Optional list to accumulate raw log entries
      checkpoint_path: Optional path for checkpointing
      step_callback: Optional callback(step, result) called after each step
      get_state_callback: Optional callback called with checkpoint data dict
        when saving a checkpoint. Used for state checkpointing.

    Returns:
      SimulationLog with results and to_html() method
    """
    num_agents = len(self.entities)
    if max_ticks is not None:
      if max_steps is None:
        max_steps = max_ticks * 100_000 * num_agents
    elif self._max_ticks is not None:
      if max_steps is None:
        max_steps = self._max_ticks * 100_000 * num_agents
    elif max_steps is None:
      max_steps = 48

    return self._sim.play(
        premise=premise,
        max_steps=max_steps,
        raw_log=raw_log,
        checkpoint_path=checkpoint_path,
        step_callback=step_callback,
        get_state_callback=get_state_callback,
    )

  def get_raw_log(self) -> list[Mapping[str, Any]]:
    """Get the raw log for structured_log conversion.

    Post-processes the engine's raw log to normalize Step numbers. The async
    engine uses a per-thread iteration counter (including idle polls) as Step,
    which can produce misleading gaps. This assigns sequential per-agent action
    counts instead, preserving the original as 'raw_step'.

    Returns:
      Normalized raw log entries.
    """
    raw = self._sim.get_raw_log()
    agent_counters: dict[str, int] = {}
    normalized = []
    for entry in raw:
      entry = dict(entry)
      thread = entry.get("thread", "")
      if thread:
        agent_counters[thread] = agent_counters.get(thread, 0) + 1
        entry["raw_step"] = entry.get("Step", 0)
        entry["Step"] = agent_counters[thread]
      normalized.append(entry)
    return normalized

  def get_state(self) -> dict[str, Any]:
    """Get simulation state for checkpointing."""
    return {
        "entities": [e.name for e in self.entities],
        "config": str(self._config),
    }

  def make_checkpoint_data(self) -> dict[str, Any]:
    """Create a checkpoint data dict from the current simulation state."""
    return self._sim.make_checkpoint_data()
