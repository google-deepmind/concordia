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
  ):
    """Initialize an identity component.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name

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

  def update(self):
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for c in self._identity_components:
        executor.submit(c.update)

    self._state = f'Name: {self._agent_name}\n' + '\n'.join(
        [c.state() for c in self._identity_components]
    )
