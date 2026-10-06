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

"""Predefined fiscal configurations for economic experiments on Concordia Island."""

from examples.concordia_island.sim import economic_profile

FiscalEvent = economic_profile.FiscalEvent

# ─────────────────────────────────────────────────────────────────────────────
# 1. Control Experiment (Baseline Economy)
#    - Monday 7:00 AM (Tick 0): Weekly payroll deposit for employed workers
#    - Friday 3:00 PM (Tick 4): Weekly housing rent deduction
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_FISCAL_EVENTS = [
    FiscalEvent(
        name="payroll",
        event_type="credit",
        target_account="liquid",
        amount_source="wage",
        day_of_week=0,  # Monday
        tick_of_day=0,  # 7:00 AM
        observation_template=(
            "// bank [{day_name}, 7:00 AM]: PAYROLL DEPOSIT: Your weekly"
            " paycheck of ${amount:.2f} has been deposited into your account."
            " Current balance: ${balance:.2f}. REMINDER: Your weekly housing"
            " rent of ${rent:.2f} is due this Friday."
        ),
        condition="is_employed",
    ),
    FiscalEvent(
        name="rent",
        event_type="debit",
        target_account="liquid",
        amount_source="rent",
        day_of_week=4,  # Friday
        tick_of_day=4,  # 3:00 PM
        observation_template=(
            "// bank [{day_name}, 3:00 PM]: RENT PAYMENT: ${amount:.2f} has"
            " been automatically deducted for your weekly housing rent."
            " Remaining balance: ${balance:.2f}."
        ),
        arrears_template=(
            "// bank [{day_name}, 3:00 PM]: RENT WARNING: You had insufficient"
            " funds to pay your full rent of ${amount:.2f}. Only ${paid:.2f}"
            " was deducted. Balance: $0.00. You are in rent arrears and face"
            " housing risk."
        ),
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Universal Basic Income (UBI)
#    - Weekly payroll for employed workers
#    - Weekly unconditional cash transfer ($125/week = $500/month)
#    - Standard rent deduction
# ─────────────────────────────────────────────────────────────────────────────

UBI_FISCAL_EVENTS = [
    FiscalEvent(
        name="payroll",
        event_type="credit",
        target_account="liquid",
        amount_source="wage",
        day_of_week=0,
        tick_of_day=0,
        observation_template=(
            "// bank [{day_name}, 7:00 AM]: PAYROLL DEPOSIT: Your weekly"
            " paycheck of ${amount:.2f} has been deposited. Balance:"
            " ${balance:.2f}."
        ),
        condition="is_employed",
    ),
    FiscalEvent(
        name="ubi_transfer",
        event_type="credit",
        target_account="liquid",
        amount_source="fixed",
        amount_value=125.0,  # $500/month / 4 weeks
        day_of_week=0,
        tick_of_day=0,
        observation_template=(
            "// bank [{day_name}, 7:00 AM]: UBI DEPOSIT: Your weekly Universal"
            " Basic Income payment of ${amount:.2f} has been deposited."
            " Current balance: ${balance:.2f}."
        ),
    ),
    FiscalEvent(
        name="rent",
        event_type="debit",
        target_account="liquid",
        amount_source="rent",
        day_of_week=4,
        tick_of_day=4,
        observation_template=(
            "// bank [{day_name}, 3:00 PM]: RENT PAYMENT: ${amount:.2f}"
            " deducted. Remaining balance: ${balance:.2f}."
        ),
        arrears_template=(
            "// bank [{day_name}, 3:00 PM]: RENT WARNING: Insufficient funds"
            " for rent of ${amount:.2f}. Paid: ${paid:.2f}. Balance: $0.00."
        ),
    ),
]

FISCAL_CONFIG_MAP = {
    "control": CONTROL_FISCAL_EVENTS,
    "ubi": UBI_FISCAL_EVENTS,
}


def get_fiscal_events(config_name: str = "control") -> list[FiscalEvent]:
  """Return the list of FiscalEvents for a named configuration.

  Args:
    config_name: One of the keys of ``FISCAL_CONFIG_MAP``.

  Raises:
    ValueError: If ``config_name`` is not a known fiscal configuration.
  """
  key = config_name.lower()
  if key not in FISCAL_CONFIG_MAP:
    raise ValueError(
        f"Unknown fiscal_config {config_name!r}. Valid options:"
        f" {sorted(FISCAL_CONFIG_MAP)}."
    )
  return FISCAL_CONFIG_MAP[key]
