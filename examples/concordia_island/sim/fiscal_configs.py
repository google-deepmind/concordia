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

# First day of the policy arms: Thursday, two days after the 2026-01-06 layoff
# in configs/job_loss_events.py. Policy events start at 9:00 AM that day.
POLICY_START_DATE = "2026-01-08"
# Monthly Universal Basic Income, paid as one lump sum per month.
UBI_MONTHLY_AMOUNT = 500.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Control Experiment (Baseline Economy)
#    - Monday 9:00 AM (Tick 1): Weekly payroll deposit for employed workers
#    - Friday 3:00 PM (Tick 4): Weekly housing rent deduction
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_FISCAL_EVENTS = [
    FiscalEvent(
        name="payroll",
        event_type="credit",
        target_account="liquid",
        amount_source="wage",
        day_of_week=0,  # Monday
        tick_of_day=1,  # 9:00 AM
        observation_template=(
            "// bank [{day_name}, {date}, 9:00 AM]: PAYROLL DEPOSIT: Your"
            " weekly paycheck of ${amount:.2f} has been deposited into your"
            " account. Current balance: ${balance:.2f}. REMINDER: Your weekly"
            " housing rent of ${rent:.2f} is due this Friday."
        ),
        condition="is_employed",
    ),
    FiscalEvent(
        name="rent",
        event_type="debit",
        target_account="liquid",
        amount_source="rent",
        savings_fallback=True,
        day_of_week=4,  # Friday
        tick_of_day=4,  # 3:00 PM
        observation_template=(
            "// bank [{day_name}, {date}, 3:00 PM]: RENT PAYMENT: ${amount:.2f}"
            " has been automatically deducted for your weekly housing rent."
            " Remaining balance: ${balance:.2f}."
        ),
        arrears_template=(
            "// bank [{day_name}, {date}, 3:00 PM]: RENT WARNING: You had"
            " insufficient funds to pay your full rent of ${amount:.2f}. Only"
            " ${paid:.2f} was deducted. Balance: $0.00. You are in rent arrears"
            " and face housing risk."
        ),
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Universal Basic Income (UBI)
#    - Weekly payroll for employed workers
#    - Monthly unconditional cash transfer ($500/month, one lump sum) from
#      POLICY_START_DATE, on that day of each month
#    - Standard rent deduction
# ─────────────────────────────────────────────────────────────────────────────

UBI_FISCAL_EVENTS = [
    FiscalEvent(
        name="payroll",
        event_type="credit",
        target_account="liquid",
        amount_source="wage",
        day_of_week=0,
        tick_of_day=1,
        observation_template=(
            "// bank [{day_name}, {date}, 9:00 AM]: PAYROLL DEPOSIT: Your"
            " weekly paycheck of ${amount:.2f} has been deposited. Balance:"
            " ${balance:.2f}."
        ),
        condition="is_employed",
    ),
    FiscalEvent(
        name="ubi_transfer",
        event_type="credit",
        target_account="liquid",
        amount_source="fixed",
        amount_value=UBI_MONTHLY_AMOUNT,
        # The full month's payment is deposited as one lump sum at 9:00 AM on
        # POLICY_START_DATE (after the 2026-01-06 layoff), then on the same day
        # of each following month (the last day of a shorter month), so short
        # and long runs both receive "$N a month". Earlier versions paid $125
        # every Monday from day 1, i.e. before the layoff.
        start_date=POLICY_START_DATE,
        monthly=True,
        tick_of_day=1,
        observation_template=(
            "// bank [{day_name}, {date}, 9:00 AM]: UBI DEPOSIT: Your monthly"
            " Universal Basic Income payment of ${amount:.2f} has been"
            " deposited. Current balance: ${balance:.2f}."
        ),
    ),
    FiscalEvent(
        name="rent",
        event_type="debit",
        target_account="liquid",
        amount_source="rent",
        savings_fallback=True,
        day_of_week=4,
        tick_of_day=4,
        observation_template=(
            "// bank [{day_name}, {date}, 3:00 PM]: RENT PAYMENT: ${amount:.2f}"
            " deducted. Remaining balance: ${balance:.2f}."
        ),
        arrears_template=(
            "// bank [{day_name}, {date}, 3:00 PM]: RENT WARNING: Insufficient"
            " funds for rent of ${amount:.2f}. Paid: ${paid:.2f}. Balance:"
            " $0.00."
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
