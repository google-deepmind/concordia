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


"""Agent component for tracking the somatic state."""

import concurrent
from concordia.agents.components import characteristic
from concordia.associative_memory import associative_memory
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component


class SomaticState(component.Component):
  """Somatic state component containing a five characteristics.

  Somatic state is comprised of hunger, thirst, fatigue, pain and feeling
  socially connected to life.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock: game_clock.GameClock,
      summarize: bool = True,
  ):
    """Initialize somatic state component.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      clock: the game clock is needed to know when is the current time
      summarize: if True, the resulting state will be a one sentence summary,
        otherwise state it would be a concatentation of five separate
        characteristics
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock = clock
    self._summarize = summarize

    self._characteristic_names = [
        'level of hunger',
        'level of thirst',
        'level of fatigue',
        'level of pain',
        'level of feeling socially connected in life',
    ]

    self._characteristics = []

    extra_instructions = (
        'Be literal. Do not use any metaphorical language. '
        + 'When there is insufficient evidence to infer a '
        + 'specific answer then guess the most likely one. '
        + 'Never express uncertainty unless '
        + f'{self._agent_name} would be uncertain.'
    )

    for characteristic_name in self._characteristic_names:
      self._characteristics.append(
          characteristic.Characteristic(
              model=model,
              memory=self._memory,
              agent_name=self._agent_name,
              characteristic_name=characteristic_name,
              state_clock=self._clock,
              extra_instructions=extra_instructions,
          )
      )

  def name(self) -> str:
    return 'Somatic state'

  def state(self):
    return self._state

  def update(self):
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for c in self._characteristics:
        executor.submit(c.update)

    self._state = '\n'.join(
        [
            f"{self._agent_name}'s {c.name()}: " + c.state()
            for c in self._characteristics
        ]
    )
    if self._summarize:
      prompt = (
          f'Summarize the somatic state of {self._agent_name} in one'
          ' sentence given the readings below. Only mention readings that'
          f' deviate from the norm, for example if {self._agent_name} is not'
          ' hungry do not mention hunger at all.\nReadings:\n'
          + self._state
      )
      self._state = f'{self._agent_name} is ' + self._model.sample_text(
          f'{prompt}\n {self._agent_name} is ', max_tokens=500
      )
