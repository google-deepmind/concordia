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


"""The conversation scene.

The conversation scene configures of the game master that runs a
conversation between players, while conditining them on the full history of the
conversation at each step through the ConversationTracker component.
"""

from collections.abc import Sequence

from concordia.agents import basic_agent
from concordia.associative_memory import blank_memories
from concordia.clocks import game_clock
from concordia.document import interactive_document
from concordia.environment import game_master as game_master_lib
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import agent as simulacrum_agent
from concordia.typing import component
from concordia.typing import metric
import termcolor


class ConversationTracker(component.Component):
  """This component accumulates history of a conversation scene in its state."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[basic_agent.BasicAgent],
      premis: str = '',
      verbose: bool = False,
      log_colour: str = 'red',
  ):
    """This component accumulates history of a conversation scene in its state.

    Args:
      model: a language model
      players: players participating
      premis: any extra text to be added on top of the conversation (say,
        circumstances of it)
      verbose: whether or not to print intermediate reasoning steps
      log_colour: colour for logging
    """
    self._model = model
    self._state = premis
    self._log_colour = log_colour
    self._players = players

    self._verbose = verbose

  def name(self) -> str:
    return 'Conversation history'

  def state(self):
    return self._state

  def terminate_episode(self) -> bool:
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement(f'Conversation:\n{self._state}\n')

    did_conclude = chain_of_thought.multiple_choice_question(
        'Is the conversation above over and not going to continue?',
        answers=['No', 'Yes'],
    )
    if self._verbose:
      self._log(chain_of_thought.view().text())

    return did_conclude == 1

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def update_after_event(self, event_statement: str):
    # The event_statement contains the last utterence in the conversation
    self._state += '\n' + event_statement
    if self._verbose:
      self._log(f'Current state of converstion: {self._state}')
    for player in self._players:
      player.observe(event_statement)

  def update(self):
    return self._state


def make_conversation_game_master(
    players: Sequence[basic_agent.BasicAgent],
    clock: game_clock.MultiIntervalClock,
    model: language_model.LanguageModel,
    memory_factory: blank_memories.MemoryFactory,
    measurements: Sequence[metric.Metric] | None,
    name: str = 'Conversation scene',
    premise: str = '',
):
  """Creates a game master that runs a conversation between players.

  Args:
    players: players participating
    clock: a clock
    model: a language model
    memory_factory: a memory factory
    measurements: measurements for the game master to use
    name: the name of the game master
    premise: any extra text to be added on top of the conversation (say,
      circumstances of it)

  Returns:
    a game master
  """

  action_spec = simulacrum_agent.ActionSpec(
      simulacrum_agent.DEFAULT_CALL_TO_SPEECH,
      'FREE',
      tag='speech',
  )

  agent_names = [player.name for player in players]

  is_are = 'are' if len(agent_names) > 1 else 'is'
  convo = f'{", ".join(agent_names)} {is_are} in conversation'
  if premise:
    convo = (
        f'{premise}\nAs a result {convo}.\nHere is the conversation from the'
        ' beginning:'
    )

  conversation_tracker = ConversationTracker(
      model=model,
      players=players,
      premis=convo,
      verbose=True,
      log_colour='red',
  )

  for player in players:
    player.observe(convo)

  memory = memory_factory.make_blank_memory()
  game_master = game_master_lib.GameMaster(
      model=model,
      memory=memory,
      clock=clock,
      name=name,
      players=players,
      measurements=measurements,
      components=[conversation_tracker],
      action_spec=action_spec,
      update_thought_chain=[thought_chains.identity],
      randomise_initiative=False,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=True,
  )
  return game_master
