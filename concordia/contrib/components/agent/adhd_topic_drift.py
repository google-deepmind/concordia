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

"""ADHD topic-drift component for modeling human user topic drift.

Note that this component doesn't necessarily create non-neurotypical
behavior. It simply periodically encourages the entity to change
conversation topics in an impulsive semantically associative fashion.
"""

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component


class AdhdTopicDrift(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Periodically injects a strong instruction to pivot topics.

  Activates every `period` steps. When active, returns a strongly worded
  instruction to pivot the conversation to a tangentially related topic.
  When inactive, returns the empty string so it contributes nothing to
  the calling component's context.
  """

  def __init__(
      self,
      period: int = 4,
      pre_act_label: str = "\n",
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._period = period
    self._step_count = 0

  def _make_pre_act_value(self) -> str:
    name = self.get_entity().name
    self._step_count += 1
    if self._step_count % self._period == 0:
      return (
          f"IMPORTANT INSTRUCTION: {name} MUST abruptly change the subject"
          " right now. They pivot the conversation to a tangentially related"
          " topic — something that is linked by free association, wordplay, or"
          f" a personal memory rather than by logic. {name} does NOT continue"
          " the current thread. Instead they leap sideways, impulsive and"
          " surprising, but not entirely implausible."
      )
    return ""

  def pre_act(self, action_spec):
    del action_spec
    value = self.get_pre_act_value()
    if not value:
      return ""
    return f"{self.get_pre_act_label()}:\n{value}\n"

  def get_state(self) -> entity_component.ComponentState:
    return {"step_count": self._step_count, "period": self._period}

  def set_state(self, state: entity_component.ComponentState) -> None:
    if "step_count" in state:
      self._step_count = int(state["step_count"])
    if "period" in state:
      self._period = int(state["period"])
