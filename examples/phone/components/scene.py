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

"""A GameMaster that simulates a player's interaction with their phone."""

import textwrap

from concordia.agents import basic_agent
from concordia.associative_memory import blank_memories
from concordia.clocks import game_clock
from concordia.document import interactive_document
from concordia.environment import game_master as game_master_lib
from examples.phone.components import apps
from examples.phone.components import logging
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import agent
from concordia.typing import component


_PHONE_CALL_TO_ACTION = textwrap.dedent("""\
  What action is {name} currently performing or has just performed
  with their smartphone to best achieve their goal?
  Consider their plan, but deviate if necessary.
  Give a specific activity using one app. For example:
  {name} uses/used the Chat app to send "hi, what's up?" to George.
  """)

_PHONE_ACTION_SPEC = agent.free_action_spec(
    call_to_action=_PHONE_CALL_TO_ACTION,
    tag='phone',
)


def build(
    player: basic_agent.BasicAgent,
    phone: apps.Phone,
    clock: game_clock.MultiIntervalClock,
    model: language_model.LanguageModel,
    memory_factory: blank_memories.MemoryFactory,
) -> game_master_lib.GameMaster:
  """Builds a GameMaster that simulates a player's interaction with their phone.

  Args:
    player: The player who is interacting with the phone.
    phone: The player's phone.
    clock: A clock.
    model: A language model.
    memory_factory: A memory factory for creating the GM's memory.

  Returns:
  """
  memory = memory_factory.make_blank_memory()
  phone_component = _PhoneComponent(model, player, phone)
  return game_master_lib.GameMaster(
      model=model,
      memory=memory,
      clock=clock,
      name='PhoneGameMaster',
      players=(player,),
      components=(phone_component,),
      action_spec=_PHONE_ACTION_SPEC,
      update_thought_chain=(thought_chains.identity,),
      player_observes_event=False,
  )


class _PhoneComponent(component.Component):
  """Parses the player's actions and invokes them on phone apps."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player: basic_agent.BasicAgent,
      phone: apps.Phone,
      log_color: str = 'red',
      verbose: bool = False,
      semi_verbose: bool = True,
  ):
    self._model = model
    self._player = player
    self._phone = phone
    self._logger = logging.Logger(log_color, verbose, semi_verbose)
    self._state = ''

  def name(self) -> str:
    return 'PhoneComponent'

  def terminate_episode(self) -> bool:
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(f'Interaction with phone:\n{self._state}')

    did_conclude = chain_of_thought.yes_no_question(
        'Has the user achieved their goal with their phone or are they still'
        ' actively in the process of completing a phone task?'
    )
    return did_conclude

  def update_after_event(self, event_statement: str):
    self._state += '\n' + event_statement
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(event_statement)
    chain_of_thought.statement(self._phone.description())
    app_index = chain_of_thought.multiple_choice_question(
        'In the above transcript, what app did the user use?',
        answers=self._phone.app_names(),
    )
    app = self._phone.apps[app_index]
    action_names = [a.name for a in app.actions()]
    chain_of_thought.statement(app.description())
    action_index = chain_of_thought.multiple_choice_question(
        'In the above transcript, what action did the user perform?',
        answers=action_names,
    )

    action = app.actions()[action_index]

    try:
      argument_text = chain_of_thought.open_question(
          action.instructions(), terminators=[]
      )
      result = app.invoke_action(action, argument_text)
      return [result]
    except apps.ActionArgumentError:
      return []
