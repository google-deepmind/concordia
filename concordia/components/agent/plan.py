# Copyright 2023 DeepMind Technologies Limited.
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

"""Agent components for planning."""

from collections.abc import Sequence

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component

DEFAULT_PRE_ACT_LABEL = 'Plan'


class Plan(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Sequence[str] = (),
      goal_component_key: str | None = None,
      force_time_horizon: str | bool = False,
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      components: keys of components to use to build the context for planning.
      goal_component_key: index into `components` to use to represent the goal
        of planning
      force_time_horizon: If not False, then use this time horizon to plan for
        instead of asking the LLM to determine the time horizon.
      pre_act_label: Prefix to add to the output of the component when called
        in `pre_act`.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._components = components
    self._goal_component_key = goal_component_key
    self._force_time_horizon = force_time_horizon

    self._current_plan = ''

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}')

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    if self._goal_component_key:
      goal_component = self.get_entity().get_component(
          self._goal_component_key, type_=action_spec_ignored.ActionSpecIgnored
      )
    else:
      goal_component = None

    component_states = '\n'.join([
        self._component_pre_act_display(key) for key in self._components])

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{component_states}\n')
    if goal_component is not None:
      prompt.statement(
          f'Current goal: {goal_component.get_pre_act_value()}.')

    prompt.statement(f'Current plan: {self._current_plan}')

    prompt.statement('Some kinds of goals can be achieved without planning. '
                     'These goals might be those that have a more "intuitive" '
                     'quality to them. They differ from the kinds of goals '
                     'that can only be accomplished with a lot of mental '
                     'effort, reflection, and prospective thinking about the '
                     'future -- i.e. qualities of goals that require planning.')

    time_horizon_elicitation_prompt = (
        f'Given the above, over what time horizon should {agent_name} '
        'plan their future actions and behaviors? Answer in the '
        'form of a time interval. If multiple planning horizons '
        'seem important then only answer regarding the one where '
        'planning is most urgent. If the most immediate task '
        'can be accomplished without planning (e.g. by intuition '
        'or habit), then skip that task and answer only concerning '
        'the next most urgent task where planning is needed.'
    )

    if self._force_time_horizon:
      time_horizon = self._force_time_horizon
      prompt.statement(
          f'Question: {time_horizon_elicitation_prompt}\n'
          f'Answer: The time horizon over which to plan is {time_horizon}')
    else:
      _ = prompt.open_question(
          question=time_horizon_elicitation_prompt,
          max_tokens=1000,
          terminators=(),
          answer_prefix='The time horizon over which to plan is ',
      )

    if not self._current_plan:
      should_replan = True
    else:
      should_replan = prompt.yes_no_question(
          f'Given the above, should {agent_name} change their current plan? '
      )

    if should_replan:
      # Replan on the first step and when the LLM suggests the agent should.
      self._current_plan = prompt.open_question(
          question=(
              f'Write {agent_name}\'s step-by-step plan for how they intend to'
              ' accomplish their goal over the time horizon mentioned above.'
          ),
          max_tokens=1200,
          terminators=(),
      )

    result = self._current_plan

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    with self._lock:
      return {'current_plan': self._current_plan}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    with self._lock:
      self._current_plan = state['current_plan']
