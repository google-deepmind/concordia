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

"""Economic data structures and configurations for Concordia Island."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any


@dataclasses.dataclass
class AgentEconomicProfile:
  """Economic state and configuration for an individual agent."""

  name: str
  economic_class: (
      str  # 'lower_middle', 'middle', 'upper_middle', 'upper', 'elite'
  )
  weekly_wage: float  # Tier base * occupation multiplier (0 if laid off)
  weekly_rent: float  # From housing tier
  family_size: int  # 1 (single) or 2 (married/partnered)
  food_min_daily: int = 3  # F_MIN per person per day
  liquid_balance: float = 0.0  # Checking account (for spending & rent)
  savings_balance: float = 0.0  # Savings account (earns interest)
  food_balance: int = 0  # Current food unit stockpile
  is_employed: bool = True  # False if laid off
  fjg_enrolled: bool = False  # True if in Federal Job Guarantee program
  inventory: dict[str, int] = dataclasses.field(default_factory=dict)

  @property
  def daily_food_needed(self) -> int:
    return self.food_min_daily * self.family_size

  @property
  def total_balance(self) -> float:
    return self.liquid_balance + self.savings_balance


@dataclasses.dataclass
class FiscalEvent:
  """A scheduled economic event that fires on a specific day/tick."""

  name: str  # e.g. 'payroll', 'rent', 'ubi_transfer'
  event_type: str  # 'credit' | 'debit'
  target_account: str  # 'liquid' | 'savings' | 'food'
  amount_source: str  # 'fixed' | 'wage' | 'rent' | 'profile_field'
  amount_value: float | None = None  # For 'fixed': the dollar amount
  profile_field: str | None = None  # For 'profile_field': which field to read
  day_of_week: int | None = None  # 0=Mon..6=Sun. None = every day
  tick_of_day: int = 0  # 0-7 (which tick within the day)
  observation_template: str = (
      ""  # Template with {name}, {amount}, {balance}, {rent}, {day_name}
  )
  arrears_template: str = ""  # Template when funds are insufficient for debit
  condition: str | None = (
      None  # 'is_employed', 'not_employed', 'fjg_enrolled', etc.
  )
  one_time_day: int | None = None  # Fire only on simulation day N
  start_day: int | None = None  # Only start firing on or after simulation day N


# Housing tier configuration
TIER_CONFIG = {
    "lower_middle": {
        "starting_cash": 500.0,
        "weekly_wage_base": 600.0,
        "weekly_rent": 200.0,
    },
    "middle": {
        "starting_cash": 1000.0,
        "weekly_wage_base": 900.0,
        "weekly_rent": 350.0,
    },
    "upper_middle": {
        "starting_cash": 2000.0,
        "weekly_wage_base": 1400.0,
        "weekly_rent": 550.0,
    },
    "upper": {
        "starting_cash": 5000.0,
        "weekly_wage_base": 2200.0,
        "weekly_rent": 900.0,
    },
    "elite": {
        "starting_cash": 15000.0,
        "weekly_wage_base": 4000.0,
        "weekly_rent": 1500.0,
    },
}

# Ohio suburb / generic island home_place prefix -> economic class
HOME_PREFIX_TO_CLASS = {
    # Canonical / Island
    "sunset_apartments": "lower_middle",
    "coral_village": "middle",
    "palm_heights": "upper_middle",
    "ocean_view_estates": "upper",
    "paradise_point": "elite",
    # Ohio suburb
    "millbrook_apts": "lower_middle",
    "brecksville_commons": "middle",
    "chippewa_ridge": "upper_middle",
    "riverview_estates": "upper",
    "timber_creek": "elite",
    # LA
    "echo_park_apts": "lower_middle",
    "silverlake_courts": "middle",
    "griffith_heights": "upper_middle",
    "hillhurst_villas": "upper",
    "elysian_crest": "elite",
    # Kerala
    "canal_row_flats": "lower_middle",
    "thottam_colony": "middle",
    "paddy_view_villas": "upper_middle",
    "backwater_estates": "upper",
    "coconut_grove": "elite",
    # Lagos
    "eko_flats": "lower_middle",
    "surulere_courts": "middle",
    "gbagada_heights": "upper_middle",
    "ikoyi_estates": "upper",
    "banana_island": "elite",
}

# Occupation multipliers
OCCUPATION_MULTIPLIER = {
    "cafe": 0.85,
    "restaurant": 0.85,
    "general_store": 0.85,
    "market": 0.95,
    "shopping_plaza": 0.95,
    "school": 1.00,
    "library": 1.00,
    "community_center": 1.05,
    "office_floor_creative": 1.10,
    "office_floor_finance": 1.15,
    "office_floor_tech": 1.20,
    "medical_clinic": 1.25,
    "office_building": 1.30,
}
DEFAULT_OCCUPATION_MULTIPLIER = 0.40  # Unemployed/retired welfare


def get_economic_class(home_place: str) -> str:
  """Determine economic class from home_place string prefix."""
  if not home_place:
    return "lower_middle"
  for prefix, econ_class in HOME_PREFIX_TO_CLASS.items():
    if home_place.startswith(prefix):
      return econ_class
  return "lower_middle"


def _occupation_multiplier(work_place: str | None) -> float:
  """Returns the wage multiplier for a workplace.

  Note the two fallbacks are deliberately different values. Having no
  workplace at all means the agent is unemployed or retired and receives
  welfare (0.40 of the tier base). Having a workplace we simply do not
  recognise means the agent is employed, so they earn the unscaled tier base
  (1.0). Collapsing these would cut the wage of anyone in an unlisted job by
  60%.

  Args:
    work_place: The agent's workplace, or None/empty if they have none.

  Returns:
    The multiplier to apply to the economic tier's base weekly wage.
  """
  if not work_place:
    return DEFAULT_OCCUPATION_MULTIPLIER
  if work_place in OCCUPATION_MULTIPLIER:
    return OCCUPATION_MULTIPLIER[work_place]
  # Workplaces are sometimes qualified with a role, e.g. "cafe_barista" for
  # the "cafe" entry.
  for known_place, multiplier in OCCUPATION_MULTIPLIER.items():
    if known_place in work_place:
      return multiplier
  return 1.0


def build_economic_profiles(
    agent_configs: Sequence[Any],
    food_min_daily: int = 3,
    laid_off_agents: Sequence[str] = (),
    starting_food_units: int = 12,
    starting_inventory: Mapping[str, int] | None = None,
) -> dict[str, AgentEconomicProfile]:
  """Construct a dictionary of AgentEconomicProfile objects from agent configs.

  Args:
    agent_configs: Sequence of AgentConfig-like objects containing attributes:
      name, home_place, work_place, relationship_status.
    food_min_daily: Daily food units needed per person.
    laid_off_agents: Sequence of agent names who are laid off.
    starting_food_units: Initial food units in pantry per household (default
      12).
    starting_inventory: Initial item inventory (default {"white t-shirt": 1}).

  Returns:
    Dict mapping agent name to AgentEconomicProfile.
  """
  laid_off_set = set(laid_off_agents)
  profiles = {}
  for config in agent_configs:
    name = getattr(config, "name", "")
    home_place = getattr(config, "home_place", "")
    work_place = getattr(config, "work_place", None)
    rel_status = str(getattr(config, "relationship_status", "single")).lower()

    econ_class = get_economic_class(home_place)
    tier_info = TIER_CONFIG.get(econ_class, TIER_CONFIG["lower_middle"])

    # Family size: only married/partnered have 2, single is 1
    if "married" in rel_status or "partner" in rel_status:
      family_size = 2
    else:
      family_size = 1

    is_laid_off = name in laid_off_set
    is_employed = bool(work_place) and not is_laid_off

    # Occupation multiplier. A laid-off agent earns nothing; everyone else
    # earns their tier's base wage scaled by their occupation.
    if is_laid_off:
      weekly_wage = 0.0
    else:
      weekly_wage = round(
          tier_info["weekly_wage_base"] * _occupation_multiplier(work_place), 2
      )

    weekly_rent = tier_info["weekly_rent"]
    starting_cash = tier_info["starting_cash"]

    # Seed food_balance with starting_food_units (default 12) so agents
    # don't starve early in the simulation.
    initial_food = (
        starting_food_units
        if starting_food_units is not None
        else (food_min_daily * family_size)
    )

    initial_inventory = (
        dict(starting_inventory)
        if starting_inventory is not None
        else {"white t-shirt": 1}
    )

    profiles[name] = AgentEconomicProfile(
        name=name,
        economic_class=econ_class,
        weekly_wage=weekly_wage,
        weekly_rent=weekly_rent,
        family_size=family_size,
        food_min_daily=food_min_daily,
        liquid_balance=starting_cash,
        savings_balance=0.0,
        food_balance=initial_food,
        is_employed=is_employed,
        fjg_enrolled=False,
        inventory=initial_inventory,
    )

  return profiles
