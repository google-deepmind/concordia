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

"""A role a rule plays: its acting component answers by the rule.

A harness that measures the game (the push audit, the corpus probe) has a
role played by a script: the role's entity is built as for anyone, and its
act component is this rule's (`GameContext.scripted`), as a model's is the
stock act component and a person's the stock `HumanActComponent`. The
entity's question reaches the rule as its action spec, with what the
entity's components say this turn (its contexts, as a model would be
prompted with them), and the answer is its action, which the World GM
resolves.
"""

from collections.abc import Callable, Mapping
from typing import override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# A rule: the role's question (its action spec) and its contexts (what its
# components say this turn) to its action.
Rule = Callable[[entity_lib.ActionSpec, Mapping[str, str]], str]


class RuleAct(entity_component.ActingComponent):
  """Acts by a rule: the answer to each question is the rule's."""

  def __init__(self, rule: Rule):
    """Initializes the component.

    Args:
      rule: The role's question (its action spec) and its contexts to its
        action.
    """
    super().__init__()
    self._rule = rule

  @override
  def get_action_attempt(
      self,
      contexts: Mapping[str, str],
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    return self._rule(action_spec, contexts)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state
