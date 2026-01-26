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
import datetime
from typing import Callable
from concordia.associative_memory.deprecated import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


class SimPlan(component.Component):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: list[component.Component],
      clock_now: Callable[[], datetime.datetime],
      name: str = 'plan',
      goal: component.Component | None = None,
      num_memories_to_retrieve: int = 5,
      horizon: str = 'the rest of the day',
      verbose: bool = False,
      log_color='green',
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      components: components to build the context of planning
      clock_now: time callback to use for the state.
      name: name of the component
      goal: a component to represent the goal of planning
      num_memories_to_retrieve: how many memories to retrieve as conditioning
        for the planning chain of thought
      horizon: string describing how long the plan should last
      verbose: whether or not to print intermediate reasoning steps
      log_color: color for debug logging
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._log_color = log_color
    self._components = components
    self._name = name
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._goal_component = goal
    self._horizon = horizon
    self._clock_now = clock_now
    self._last_update = datetime.datetime.min

    self._latest_memories = ''
    self._last_observation = []
    self._current_plan = ''
    self._history = []

    self._verbose = verbose

  def name(self) -> str:
    return self._name

  def state(self):
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def observe(self, observation: str):
    self._last_observation.append(observation)

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update(self):
    if self._last_update == self._clock_now():
      return
    self._last_update = self._clock_now()

    observation = '\n'.join(self._last_observation)
    self._last_observation = []
    memories = list(self._memory.retrieve_associative(
        observation,
        k=self._num_memories_to_retrieve,
        use_recency=True,
        add_time=True,
    ))
    if self._goal_component:
      memories = memories + list(self._memory.retrieve_associative(
          self._goal_component.state(),
          k=self._num_memories_to_retrieve,
          use_recency=True,
          add_time=True,
      ))
    memories = '\n'.join(memories)

    components = '\n'.join([
        f"{self._agent_name}'s {construct.name()}:\n{construct.state()}"
        for construct in self._components
    ])

    in_context_example = (
        ' Please format the plan like in this example: [21:00 - 22:00] watch TV'
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{components}\n')
    prompt.statement(f'Relevant memories:\n{memories}')
    if self._goal_component:
      prompt.statement(f'Current goal: {self._goal_component.state()}.')
    prompt.statement(f'Current plan: {self._current_plan}')
    prompt.statement(f'Current situation: {observation}')

    time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
    prompt.statement(f'The current time is: {time_now}\n')
    should_replan = prompt.yes_no_question(
        f'Given the above, should {self._agent_name} change their current '
        'plan? '
    )

    if should_replan or not self._state:
      goal_mention = '.'
      if self._goal_component:
        goal_mention = ', keep in mind the goal.'
      self._current_plan = prompt.open_question(
          f"Write {self._agent_name}'s plan for {self._horizon}. Please,"
          ' provide a detailed schedule'
          + goal_mention
          + in_context_example,
          max_tokens=1200,
          terminators=(),
      )

    self._state = self._current_plan

    if self._verbose:
      self._log('\n' + prompt.view().text() + '\n')

    update_log = {
        'Summary': (
            f'detailed plan of {self._agent_name} '
            + f'for {self._horizon}'
        ),
        'State': self._state,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
