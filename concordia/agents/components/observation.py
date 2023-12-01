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


"""Agent components for representing observation stream."""

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class Observation(component.Component):
  """Component that stacks current observations together, clears on update."""

  def __init__(
      self,
      agent_name: str,
      memory: associative_memory.AssociativeMemory,
      component_name: str = 'Current observation',
      verbose: bool = False,
      log_colour='green',
  ):
    """Initialize the observation component.

    Args:
      agent_name: the name of the agent
      memory: memory for writing observations into
      component_name: the name of this component
      verbose: whether or not to print intermediate reasoning steps
      log_colour: colour for logging
    """
    self._agent_name = agent_name
    self._log_colour = log_colour
    self._name = component_name
    self._memory = memory

    self._last_observation = []

    self._verbose = verbose

  def name(self) -> str:
    return self._name

  def state(self):
    if self._verbose:
      self._log('\n'.join(self._last_observation) + '\n')
    return '\n'.join(self._last_observation) + '\n'

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def observe(self, observation: str):
    self._last_observation.append(observation)
    self._memory.add(
        f'[observation] {observation}',
        tags=['observation'],
    )

  def update(self):
    self._last_observation = []
    return ''


class ObservationSummary(component.Component):
  """Component that summarises current observations on update."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      agent_name: str,
      components: list[component.Component],
      verbose: bool = False,
      log_colour='green',
  ):
    """Initialize a construct containing the agent's plan for the day.

    Args:
      model: a language model
      agent_name: the name of the agent
      components: components to condition observation summarisation
      verbose: whether or not to print intermediate reasoning steps
      log_colour: colour for logging
    """
    self._model = model
    self._state = ''
    self._agent_name = agent_name
    self._log_colour = log_colour
    self._components = components

    self._last_observation = []

    self._verbose = verbose

  def name(self) -> str:
    return 'Summary of recent observations'

  def state(self):
    return self._state

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def observe(self, observation: str):
    self._last_observation.append(observation)

  def update(self):
    context = '\n'.join(
        [
            f"{self._agent_name}'s "
            + (construct.name() + ':\n' + construct.state())
            for construct in self._components
        ]
    )

    numbered_observations = [
        f'{i}. {observation}'
        for i, observation in enumerate(self._last_observation)
    ]
    current_observations = '\n'.join(numbered_observations)

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(context + '\n')
    prompt.statement(
        'Current observations, numbered in chronological order:\n'
        + f'{current_observations}\n'
    )
    self._state = prompt.open_question(
        'Summarize the observations into one sentence.'
    )

    self._last_observation = []

    if self._verbose:
      self._log('\nObservation summary:')
      self._log('\n' + prompt.view().text() + '\n')
