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
from typing import Callable
from concordia.associative_memory import associative_memory
from concordia.components.agent import characteristic
from concordia.language_model import language_model
from concordia.typing import component


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
      clock_now: Callable[[], datetime.datetime] | None = None,
  ):
    """Initialize an identity component.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      clock_now: time callback to use for the state.
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._last_update = datetime.datetime.min

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
    return 'Identity'

  def state(self):
    return self._state

  def get_last_log(self):
    current_log = {
        'Summary': f'identity of {self._agent_name}',
        'state': self._state,
    }
    for comp in self._identity_components:
      last_log = comp.get_last_log()
      if last_log:
        if 'date' in last_log.keys():
          last_log.pop('date')
        current_log[comp.name()] = last_log
    return current_log

  def update(self):
    if self._clock_now() == self._last_update:
      return
    self._last_update = self._clock_now()

    with concurrent.futures.ThreadPoolExecutor() as executor:
      for c in self._identity_components:
        executor.submit(c.update)

    self._state = '\n'.join(
        [f'{c.name()}: {c.state()}' for c in self._identity_components]
    )
