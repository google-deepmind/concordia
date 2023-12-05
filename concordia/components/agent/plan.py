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

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class SimPlan(component.Component):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: list[component.Component],
      goal: component.Component | None = None,
      num_memories_to_retrieve: int = 5,
      timescale: str = 'the rest of the day',
      time_adverb: str = 'hourly',
      verbose: bool = False,
      log_colour='green',
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      components: components to build the context of planning
      goal: a component to represent the goal of planning
      num_memories_to_retrieve: how many memories to retrieve as conditioning
        for the planning chain of thought
      timescale: string describing how long the plan should last
      time_adverb: string describing the rate of steps in the plan
      verbose: whether or not to print intermediate reasoning steps
      log_colour: colour for logging
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._log_colour = log_colour
    self._components = components
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._goal_component = goal
    self._timescale = timescale
    self._time_adverb = time_adverb

    self._latest_memories = ''
    self._last_observation = []
    self._current_plan = ''

    self._verbose = verbose

  def name(self) -> str:
    return 'Plan'

  def state(self):
    return self._state

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def observe(self, observation: str):
    self._last_observation.append(observation)

  def update(self, push_to_mem=True):
    observation = '\n'.join(self._last_observation)
    self._last_observation = []
    memories = self._memory.retrieve_associative(
        observation,
        k=self._num_memories_to_retrieve,
        use_recency=True,
        add_time=True,
    )
    if self._goal_component:
      memories = memories + self._memory.retrieve_associative(
          self._goal_component.state(),
          k=self._num_memories_to_retrieve,
          use_recency=True,
          add_time=True,
      )
    memories = '\n'.join(memories)

    components = '\n'.join(
        [
            f"{self._agent_name}'s "
            + (construct.name() + ':\n' + construct.state())
            for construct in self._components
        ]
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{components}\n')
    prompt.statement(f'Relevant memories:\n{memories}')
    if self._goal_component:
      prompt.statement(f'Current goal: {self._goal_component.state()}.')
    prompt.statement(f'Current plan: {self._current_plan}')
    prompt.statement(f'Current situation: {observation}')
    should_replan = prompt.yes_no_question(
        (
            f'Given the above, should {self._agent_name} change their current '
            + 'plan?'
        )
    )

    if should_replan or not self._state:
      goal_mention = '.'
      if self._goal_component:
        goal_mention = ', keep in mind the goal.'
      self._current_plan = prompt.open_question(
          f"What is {self._agent_name}'s plan for {self._timescale}? Please,"
          f' provide a {self._time_adverb} schedule'
          + goal_mention,
          max_characters=1200,
          max_tokens=1200,
          terminators=(),
      )
      if self._goal_component:
        self._state = (
            f'The goal: {self._goal_component.state()}\n{self._current_plan}'
        )
      else:
        self._state = self._current_plan

    if self._verbose:
      self._log('\n' + prompt.view().text() + '\n')
