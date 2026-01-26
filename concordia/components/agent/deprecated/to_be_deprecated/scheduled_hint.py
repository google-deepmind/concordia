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

"""Agent component for scheduled hinting."""
import datetime
from typing import Callable, Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
import termcolor


class ScheduledHint(component.Component):
  """Deliver a specific hints to an agent at specific times."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      agent_name: str,
      components: Sequence[component.Component] | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      hints: Sequence[Callable[[str, datetime.datetime], str]] | None = None,
      verbose: bool = False,
  ):
    """Initializes the PersonBySituation component.

    Args:
      name: The name of the component.
      model: The language model to use.
      agent_name: The name of the agent.
      components: The components to condition the answer on.
      clock_now: time callback to use for the state.
      hints: Sequence of possible hints to apply on each step. Each checks a
        condition based on the incoming chain of thought and either outputs a
        string or not.
      verbose: Whether to print the state of the component.
    """

    self._verbose = verbose
    self._model = model
    self._state = ''
    self._components = components or []
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._hints = hints
    self._name = name
    self._history = []

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self) -> None:
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f"{self._agent_name}'s "
        + (construct.name() + ':\n' + construct.state())
        for construct in self._components
    ])
    chain_of_thought.statement(
        f'Current time: {self._clock_now()}\n' + component_states)
    hint_outputs = []
    for hint in self._hints:
      hint_output = hint(chain_of_thought.copy().view().text(),
                         self._clock_now())
      if hint_output:
        hint_outputs.append(hint_output)

    hint_outputs_str = ' '.join(hint_outputs)
    self._state = f'{hint_outputs_str}'

    self._last_chain = chain_of_thought
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
        'Chain of thought': self._last_chain.view().text().splitlines(),
    }
    self._history.append(update_log)
