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

"""An acting component that answers by a rule: a bot, a baseline, a watcher.

Choosing who plays a role means choosing its acting component: a language
model (`ConcatActComponent`), a person (`HumanActComponent`), or a rule. This
component is the rule: a callable that reads the action spec and the
entity's contexts (what its context components say) and returns the answer.
The rest of the entity stays the same, so a rule-played role sees exactly
what a model-played one would.

A rule that keeps state (a plan's progress, a count) exposes `get_state` and
`set_state`; the component saves and restores it with its own, so a restored
simulation plays on exactly as the live one.

  def watch(action_spec, contexts):
    del action_spec, contexts
    return 'watch'

  act = rule_act.RuleActComponent(watch)
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import cast, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# A rule: the action spec and the contexts, to the answer.
Rule = Callable[[entity_lib.ActionSpec, Mapping[str, str]], str]


class RuleActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Answers every action spec by a rule."""

  def __init__(self, rule: Rule):
    """Initializes the component.

    Args:
      rule: Called with the action spec (its `call_to_action` with the
        entity's name in place of `{name}`) and the contexts; returns the
        answer. For a choice, the answer must be one of the options (else
        `get_action_attempt` raises ValueError: a rule never has its answer
        silently replaced). A rule with state exposes `get_state()`
        (JSON-safe) and `set_state(state)`.
    """
    super().__init__()
    self._rule = rule

  @override
  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    name = self.get_entity().name
    spec = dataclasses.replace(
        action_spec,
        call_to_action=action_spec.call_to_action.replace('{name}', name),
    )
    answer = self._rule(spec, context)
    if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      dataclasses.replace(
          action_spec, output_type=entity_lib.OutputType.CHOICE
      ).validate(answer)
    self._logging_channel({
        'Key': 'Rule',
        'Summary': f'{name}: {answer}',
        'Value': answer,
    })
    return answer

  @override
  def get_state(self) -> entity_component.ComponentState:
    get_state = getattr(self._rule, 'get_state', None)
    if not callable(get_state):
      return {}
    return {'rule': cast(entity_component.ComponentState, get_state())}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    set_state = getattr(self._rule, 'set_state', None)
    if callable(set_state) and 'rule' in state:
      set_state(state['rule'])
