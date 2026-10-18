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

"""The World GM's standing orders: the peoples' adopted reflexes.

A people adopts standing orders by an action (`actors.ADOPT_ORDERS`); from
then on they are contrib reflexes on the bus that place its market orders
(`world/automation.py`). This component installs that engine among the World
GM's components and checkpoints it with the GM.
"""

from typing import Any, cast, override

from examples.deep_time.world import automation as automation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_STANDING_ORDERS_KEY = '__standing_orders__'


class StandingOrders(entity_component.ContextComponent):
  """The peoples' standing orders (reflexes), a World GM component."""

  def __init__(self, automation: automation_lib.Automation):
    """Initializes the component.

    Args:
      automation: The engine of the peoples' standing orders.
    """
    super().__init__()
    self._automation = automation

  @property
  def automation(self) -> automation_lib.Automation:
    return self._automation

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    return self._automation.get_state()

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._automation.set_state(cast(dict[str, Any], state))
