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

from collections.abc import Mapping
import types

from concordia.components.agent.unstable import action_spec_ignored
from concordia.components.agent.unstable import observation as observation_component_module
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import logging
from concordia.typing.unstable import entity_component

DEFAULT_PRE_ACT_KEY = 'Plan'


class Plan(action_spec_ignored.ActionSpecIgnored):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str = (
          observation_component_module.DEFAULT_OBSERVATION_COMPONENT_NAME),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      goal_component_name: str | None = None,
      clock_component_name: str | None = None,
      force_time_horizon: str | bool = False,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      observation_component_name: The name of the observation component from
        which to retrieve observations.
      components: components to build the context of planning. This is a mapping
        of the component name to a label to use in the prompt.
      goal_component_name: index into `components` to use to represent the goal
        of planning
      clock_component_name: index into `components` to use to represent the
        current time
      force_time_horizon: If not False, then use this time horizon to plan for
        instead of asking the LLM to determine the time horizon.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._observation_component_name = observation_component_name
    self._components = dict(components)
    self._goal_component_name = goal_component_name
    self._clock_component_name = clock_component_name
    self._force_time_horizon = force_time_horizon

    self._current_plan = ''

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    observation_component = self.get_entity().get_component(
        self._observation_component_name,
        type_=action_spec_ignored.ActionSpecIgnored)
    latest_observations = observation_component.get_pre_act_value()

    if self._goal_component_name:
      goal_component = self.get_entity().get_component(
          self._goal_component_name,
          type_=action_spec_ignored.ActionSpecIgnored)
    else:
      goal_component = None

    component_states = '\n'.join([
        f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{component_states}\n')
    if goal_component is not None:
      prompt.statement(
          f'Current goal: {goal_component.get_pre_act_value()}.')
    prompt.statement(f'Current plan: {self._current_plan}')
    prompt.statement(f'Current situation: {latest_observations}')

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
              f"Write {agent_name}'s step-by-step plan for how they intend to"
              ' accomplish their goal over the time horizon mentioned above.'
          ),
          max_tokens=1200,
          terminators=(),
      )

    result = self._current_plan

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {
          'current_plan': self._current_plan,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    with self._lock:
      self._current_plan = state['current_plan']
