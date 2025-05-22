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


"""Agent identity component."""
import concurrent
import datetime
from typing import Callable, Sequence
from concordia.associative_memory.deprecated import associative_memory
from concordia.components.agent.deprecated.to_be_deprecated import characteristic
from concordia.language_model import language_model
from concordia.typing.deprecated import component


class SimIdentity(component.Component):
  """Identity component containing a few characteristics.

  Identity is built out of 3 characteristics:
  1. 'core characteristics',
  2. 'current daily occupation',
  3. 'feeling about recent progress in life',
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      name: str = 'identity',
      clock_now: Callable[[], datetime.datetime] | None = None,
  ):
    """Initialize an identity component.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      name: the name of the component
      clock_now: time callback to use for the state.
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._name = name
    self._clock_now = clock_now
    self._last_update = datetime.datetime.min
    self._history = []

    self._identity_component_names = [
        'core characteristics',
        'current daily occupation',
        'feeling about recent progress in life',
    ]

    self._identity_components = []

    for component_name in self._identity_component_names:
      self._identity_components.append(
          characteristic.Characteristic(
              model=model,
              memory=self._memory,
              agent_name=self._agent_name,
              characteristic_name=component_name,
          )
      )

  def name(self) -> str:
    return self._name

  def state(self):
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_components(self) -> Sequence[component.Component]:
    # Since this component handles updating of its subcomponents itself, we
    # therefore do not need to return them here.
    return []

  def update(self):
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(c.update) for c in self._identity_components]
    for future in futures:
      future.result()

    self._state = '\n'.join(
        [f'{c.name()}: {c.state()}' for c in self._identity_components]
    )

    update_log = {
        'date': self._clock_now(),
        'Summary': self._name,
        'State': self._state,
    }
    self._history.append(update_log)
