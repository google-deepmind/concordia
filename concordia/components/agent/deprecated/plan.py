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

from collections.abc import Callable, Mapping
import datetime
import types

from concordia.components.agent.deprecated import action_spec_ignored
from concordia.components.agent.deprecated import memory_component
from concordia.components.agent.deprecated import observation
from concordia.deprecated.memory_bank import legacy_associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging


DEFAULT_PRE_ACT_KEY = 'Plan'
_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class Plan(action_spec_ignored.ActionSpecIgnored):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      goal_component_name: str | None = None,
      num_memories_to_retrieve: int = 10,
      horizon: str = 'the rest of the day',
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      observation_component_name: The name of the observation component from
        which to retrieve observations.
      memory_component_name: The name of the memory component from which to
        retrieve memories
      components: components to build the context of planning. This is a mapping
        of the component name to a label to use in the prompt.
      clock_now: time callback to use for the state.
      goal_component_name: index into `components` to use to represent the goal
        of planning
      num_memories_to_retrieve: how many memories to retrieve as conditioning
        for the planning chain of thought
      horizon: string describing how long the plan should last
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._observation_component_name = observation_component_name
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._goal_component_name = goal_component_name
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._horizon = horizon

    self._current_plan = ''

    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    observation_component = self.get_entity().get_component(
        self._observation_component_name,
        type_=observation.Observation)
    latest_observations = observation_component.get_pre_act_value()

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    memories = [mem.text for mem in memory.retrieve(
        query=latest_observations,
        scoring_fn=_ASSOCIATIVE_RETRIEVAL,
        limit=self._num_memories_to_retrieve)]

    if self._goal_component_name:
      goal_component = self.get_entity().get_component(
          self._goal_component_name,
          type_=action_spec_ignored.ActionSpecIgnored)
      memories = memories + [mem.text for mem in memory.retrieve(
          query=goal_component.get_pre_act_value(),
          scoring_fn=_ASSOCIATIVE_RETRIEVAL,
          limit=self._num_memories_to_retrieve)]
    else:
      goal_component = None

    memories = '\n'.join(memories)

    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])

    in_context_example = (
        ' Please format the plan like in this example: [21:00 - 22:00] watch TV'
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{component_states}\n')
    prompt.statement(f'Relevant memories:\n{memories}')
    if goal_component is not None:
      prompt.statement(
          f'Current goal: {goal_component.get_pre_act_value()}.')  # pylint: disable=undefined-variable
    prompt.statement(f'Current plan: {self._current_plan}')
    prompt.statement(f'Current situation: {latest_observations}')

    time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
    prompt.statement(f'The current time is: {time_now}\n')
    should_replan = prompt.yes_no_question(
        f'Given the above, should {agent_name} change their current '
        'plan? '
    )

    if should_replan or not self._current_plan:
      # Replan on the first turn and whenever the LLM suggests the agent should.
      goal_mention = '.'
      if self._goal_component_name:
        goal_mention = ', keep in mind the goal.'
      self._current_plan = prompt.open_question(
          f"Write {agent_name}'s plan for {self._horizon}."
          ' Provide a detailed schedule'
          + goal_mention
          + in_context_example,
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
