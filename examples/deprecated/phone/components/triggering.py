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


"""A component that runs the phone scene when a phone action is detected."""

from collections.abc import Sequence

from concordia.agents.deprecated import deprecated_agent
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.clocks import game_clock
from concordia.document import interactive_document
from examples.deprecated.phone.components import apps
from examples.deprecated.phone.components import logging
from examples.deprecated.phone.components import scene
from concordia.language_model import language_model
from concordia.typing.deprecated import component
from concordia.utils import helper_functions


class SceneTriggeringComponent(component.Component):
  """Runs the phone scene when a phone action is detected."""

  def __init__(
      self,
      players: Sequence[deprecated_agent.BasicAgent],
      phones: Sequence[apps.Phone],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      clock: game_clock.MultiIntervalClock,
      memory_factory: blank_memories.MemoryFactory,
      log_color: str = 'magenta',
      verbose: bool = False,
      semi_verbose: bool = True,
  ):
    self._players = players
    self._phones = phones
    self._model = model
    self._clock = clock
    self._memory_factory = memory_factory
    self._memory = memory
    self._logger = logging.Logger(log_color, verbose, semi_verbose)

  def name(self):
    return 'State of phone'

  def _is_phone_event(self, event_statement: str) -> bool:
    document = interactive_document.InteractiveDocument(self._model)
    document.statement(f'Event: {event_statement}')

    return document.yes_no_question(
        'Did a player engage in any activity typically associated with'
        ' smartphone use during this event? Consider not only explicit mentions'
        ' of phone interaction, but also actions commonly performed using'
        ' mobile apps or smartphone features. This includes, but is not limited'
        ' to:\n- Communicating (e.g., messaging, calling, emailing)\n-'
        ' Accessing information (e.g., browsing the internet, checking social'
        ' media)\n- Using utility apps (e.g., calendar, notes, calculator)\n-'
        ' Navigation or location services\n- Taking photos or videos\n- Using'
        ' mobile payment systems\n- Playing mobile gamesEven if a phone is not'
        ' explicitly mentioned, consider whether the described actions strongly'
        ' imply smartphone use in modern contexts.'
    )

  def _get_player_from_event(
      self, event_statement: str
  ) -> deprecated_agent.BasicAgent | None:
    document = interactive_document.InteractiveDocument(self._model)
    document.statement(
        f'Event: {event_statement}. This event states that someone interacted'
        ' with their phone.'
    )

    for player in self._players:
      is_player_using_phone = helper_functions.filter_copy_as_statement(
          document
      ).yes_no_question(
          f'Does the event description indicate that {player.name} engaged in'
          ' any activity typically associated with smartphone use? Consider'
          ' both explicit mentions of phone interaction and actions commonly'
          ' performed using mobile apps or smartphone features, including but'
          ' not limited to:\n- Communicating (e.g., messaging, calling,'
          ' emailing)\n- Accessing information (e.g., browsing the internet,'
          ' checking social media)\n- Using utility apps (e.g., calendar,'
          ' notes, calculator)\n- Navigation or location services\n- Taking'
          ' photos or videos\n- Using mobile payment systems\n- Playing mobile'
          f' gamesEven if {player.name} is not explicitly described as using a'
          ' phone, consider whether their actions strongly imply smartphone'
          ' use in modern contexts.'
      )
      if is_player_using_phone:
        return player

    return None

  def _get_phone(self, player_name: str) -> apps.Phone:
    return next(p for p in self._phones if p.player_name == player_name)

  def _get_player_using_phone(
      self, event_statement: str
  ) -> deprecated_agent.BasicAgent | None:
    self._logger.semi_verbose('Checking if the phone was used...')

    if not self._is_phone_event(event_statement):
      self._logger.semi_verbose('The phone was not used.')
      return None

    player = self._get_player_from_event(event_statement)

    if player is None:
      self._logger.semi_verbose('The phone was not used.')
    else:
      self._logger.semi_verbose(f'Player using the phone: {player.name}')
    return player

  def _run_phone_scene(self, player: deprecated_agent.BasicAgent):
    phone_scene = scene.build(
        player,
        self._get_phone(player.name),
        clock=self._clock,
        model=self._model,
        memory_factory=self._memory_factory,
    )
    with self._clock.higher_gear():
      scene_output = phone_scene.run_episode()

    for event in scene_output:
      player.observe(event)
      self._memory.add(event)
    return scene_output

  def update_after_event(self, event_statement: str):
    player = self._get_player_using_phone(event_statement)
    if player is not None:
      self._run_phone_scene(player)

  def partial_state(self, player_name: str):
    return self._get_phone(player_name).description()
