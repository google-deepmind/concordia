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
  weekly_wage: (
      float  # BLS income before taxes / 52 for the class (0 if laid off)
  )
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
  # Calendar bounds (ISO dates, inclusive), e.g. "2026-01-08". Prefer these to
  # `start_day`: they match the `deliver_date` of the scheduled news event that
  # announces a policy, so a transfer cannot silently start before (or long
  # after) its announcement if the run's start date or day counting changes.
  start_date: str | None = None
  end_date: str | None = None
  # Fire every N days counted from `start_date` (e.g. 7 = weekly on the
  # weekday of `start_date`). Requires `start_date`; ignores `day_of_week`.
  interval_days: int | None = None
  # Fire once a month on the day-of-month of `start_date` (last day of the
  # month if shorter). Requires `start_date`; ignores `day_of_week`.
  monthly: bool = False
  # For liquid debits: take any shortfall in checking from savings.
  savings_fallback: bool = False


# Housing tier configuration. Weekly rent and weekly wage are not configured
# here: both are BLS Consumer Expenditure Survey values for the class's income
# quintile, from BLS_ANNUAL_VALUES (see bls_weekly_rent and
# bls_weekly_income), so rent / wage equals the BLS Shelter share of income
# before taxes for every employed agent.
TIER_CONFIG = {
    "lower_middle": {
        "starting_cash": 500.0,
    },
    "middle": {
        "starting_cash": 1000.0,
    },
    "upper_middle": {
        "starting_cash": 2000.0,
    },
    "upper": {
        "starting_cash": 5000.0,
    },
    "elite": {
        "starting_cash": 15000.0,
    },
}

# BLS Consumer Expenditure Surveys (CE), 2023: mean annual values per consumer
# unit, in US dollars per year, by quintile of income before taxes, for the
# categories "Shelter" (owned dwellings, rented dwellings and other lodging)
# and "Income before taxes".
#   "us": Table 1101, "Quintiles of income before taxes: Annual expenditure
#     means, shares, standard errors, and relative standard errors, Consumer
#     Expenditure Surveys, 2023"
#     (https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/cu-income-quintiles-before-taxes-2023.xlsx).
#   "midwest": 2023 CE Interview public-use microdata
#     (https://www.bls.gov/cex/pumd_data.htm), consumer units with
#     REGION == 2, grouped by national income quintiles (INC_RNKM, the same
#     income groups as Table 1101). Calendar-year means follow BLS "Getting
#     Started with CE Public-Use Microdata". Applied to all U.S. consumer
#     units, the method reproduces the published Table 1101 means within
#     5% (max relative error: shelter 2.1%, income before taxes 1.3%).
# Values were copied programmatically, at full precision, from the internal
# generated BLS data on 2026-10-20.
BLS_ANNUAL_VALUES: dict[str, dict[str, dict[str, float]]] = {
    "us": {
        "shelter": {
            "Q1": 8754.0,
            "Q2": 11196.0,
            "Q3": 13843.0,
            "Q4": 16441.0,
            "Q5": 27226.0,
        },
        "income_before_taxes": {
            "Q1": 15596.0,
            "Q2": 40751.0,
            "Q3": 71057.0,
            "Q4": 116717.0,
            "Q5": 264518.0,
        },
    },
    "midwest": {
        "shelter": {
            "Q1": 7146.533250156708,
            "Q2": 9852.71678285154,
            "Q3": 11606.406876497567,
            "Q4": 13572.71826519834,
            "Q5": 23148.11708226571,
        },
        "income_before_taxes": {
            "Q1": 14904.41634778282,
            "Q2": 40960.34219237088,
            "Q3": 71988.76062279119,
            "Q4": 118612.77949723389,
            "Q5": 250855.97986353168,
        },
    },
}
# Economic class -> quintile of income before taxes. The five classes map onto
# the five quintiles in order; wages and rents are both that quintile's BLS
# means.
CLASS_TO_BLS_QUINTILE = {
    "lower_middle": "Q1",
    "middle": "Q2",
    "upper_middle": "Q3",
    "upper": "Q4",
    "elite": "Q5",
}
# The sim's single weekly "rent" debit equals the BLS "Shelter" category:
# owned dwellings (mortgage interest, property taxes, maintenance, insurance),
# rented dwellings and other lodging. "Housing" additionally includes
# utilities, household operations, housekeeping supplies and furnishings,
# which are not a payment for the dwelling itself.
BLS_RENT_CATEGORY = "shelter"
# Weekly wages are BLS mean income before taxes per consumer unit / 52 (each
# agent is treated as one consumer unit, as for rent).
BLS_INCOME_CATEGORY = "income_before_taxes"
# The region is chosen per setting preset by bls_region_for_setting: Midwest
# values for the Ohio (Brecksville) presets, national values for every other
# setting until region-specific data exists.
DEFAULT_BLS_REGION = "us"
MIDWEST_SETTING_PREFIXES = ("brecksville",)
WEEKS_PER_YEAR = 52


def bls_region_for_setting(setting: str | None) -> str:
  """Returns the BLS region for a setting preset key.

  Args:
    setting: A key of locations.SETTING_PRESETS (e.g. "brecksville_1000",
      "kerala") or None for the default island places.

  Returns:
    "midwest" for the Ohio (Brecksville) presets, otherwise "us".
  """
  if setting and setting.lower().startswith(MIDWEST_SETTING_PREFIXES):
    return "midwest"
  return DEFAULT_BLS_REGION


def bls_weekly_income(
    economic_class: str,
    region: str = DEFAULT_BLS_REGION,
) -> float:
  """Returns the weekly wage for a class: BLS mean income before taxes / 52.

  Args:
    economic_class: One of CLASS_TO_BLS_QUINTILE's keys.
    region: Region key of BLS_ANNUAL_VALUES ("us" or "midwest").

  Returns:
    The weekly amount in dollars, rounded to cents.
  """
  return bls_weekly_rent(
      economic_class, region=region, category=BLS_INCOME_CATEGORY
  )


def bls_weekly_rent(
    economic_class: str,
    region: str = DEFAULT_BLS_REGION,
    category: str = BLS_RENT_CATEGORY,
) -> float:
  """Returns the weekly value of a BLS category for a class: annual mean / 52.

  Args:
    economic_class: One of CLASS_TO_BLS_QUINTILE's keys.
    region: Region key of BLS_ANNUAL_VALUES ("us" or "midwest").
    category: "shelter" (default) or "income_before_taxes".

  Returns:
    The weekly amount in dollars, rounded to cents.

  Raises:
    KeyError: If the class, region or category is unknown.
  """
  if economic_class not in CLASS_TO_BLS_QUINTILE:
    raise KeyError(f"Unknown economic class {economic_class!r}.")
  if region not in BLS_ANNUAL_VALUES:
    raise KeyError(
        f"Unknown BLS region {region!r}; have {sorted(BLS_ANNUAL_VALUES)}."
    )
  region_values = BLS_ANNUAL_VALUES[region]
  if category not in region_values:
    raise KeyError(f"BLS region {region!r} has no category {category!r}.")
  annual = region_values[category][CLASS_TO_BLS_QUINTILE[economic_class]]
  return round(annual / WEEKS_PER_YEAR, 2)


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
    "adelabu_terraces": "upper_middle",
    "ogunlana_residences": "upper",
    "bode_thomas_villas": "elite",
}

# Agents with no workplace at all (unemployed/retired) receive welfare of this
# fraction of their class's BLS weekly income. Employed agents earn exactly the
# class's BLS weekly income: occupation multipliers were removed so the rent
# share of income matches BLS by quintile.
NO_WORKPLACE_INCOME_FRACTION = 0.40


def get_economic_class(home_place: str) -> str:
  """Determine economic class from home_place string prefix."""
  if not home_place:
    return "lower_middle"
  for prefix, econ_class in HOME_PREFIX_TO_CLASS.items():
    if home_place.startswith(prefix):
      return econ_class
  return "lower_middle"


def build_economic_profiles(
    agent_configs: Sequence[Any],
    food_min_daily: int = 3,
    laid_off_agents: Sequence[str] = (),
    starting_food_units: int = 12,
    starting_inventory: Mapping[str, int] | None = None,
    *,
    bls_region: str,
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
    bls_region: Region of the BLS housing data that sets weekly rents. Callers
      derive it from the setting preset with bls_region_for_setting; it has no
      default so a caller cannot silently use the wrong region.

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

    # A laid-off agent earns nothing; an employed agent earns the class's BLS
    # weekly income; an agent with no workplace gets welfare.
    if is_laid_off:
      weekly_wage = 0.0
    elif work_place:
      weekly_wage = bls_weekly_income(econ_class, region=bls_region)
    else:
      weekly_wage = round(
          bls_weekly_income(econ_class, region=bls_region)
          * NO_WORKPLACE_INCOME_FRACTION,
          2,
      )

    weekly_rent = bls_weekly_rent(econ_class, region=bls_region)
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
