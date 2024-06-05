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

The conversation scene configures the game master that runs a
conversation between players, while conditioning them on the full history of the
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
import termcolor


class ConversationTracker(component.Component):
  """This component accumulates history of a conversation scene in its state."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[basic_agent.BasicAgent],
      premise: str = '',
      check_for_termination: bool = True,
      key_question: str | None = None,
      verbose: bool = False,
      log_colour: str = 'red',
  ):
    """This component accumulates history of a conversation scene in its state.

    Args:
      model: a language model
      players: players participating
      premise: any extra text to be added on top of the conversation (say,
        circumstances of it)
      check_for_termination: whether or not to check for termination of the
        conversation
      key_question: End the scene once the game master knows the answer to this
        question.
      verbose: whether or not to print intermediate reasoning steps
      log_colour: colour for logging
    """
    self._model = model
    self._state = premise
    self._log_colour = log_colour
    self._players = players
    self._check_for_termination = check_for_termination
    self._key_question = key_question

    self._verbose = verbose

  def name(self) -> str:
    return 'Conversation history'

  def state(self):
    return self._state

  def terminate_episode(self) -> bool:
    if not self._check_for_termination:
      return False
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    chain_of_thought.statement('\n')
    chain_of_thought.statement(f'Key question: {self._key_question}')
    chain_of_thought.statement(f'Conversation:\n{self._state}\n')

    key_question_answered = chain_of_thought.multiple_choice_question(
        question=('Has the answer to the key question been revealed '
                  'by the conversation so far?'),
        answers=['No', 'Yes'],
    )
    did_conclude = False
    if key_question_answered:
      did_conclude = True
    else:
      will_not_answer = chain_of_thought.multiple_choice_question(
          question=('Considerations on whether or not to end the scene now:\n '
                    'Is it clear now that the conversation is unlikely to '
                    'reveal the answer to the key question? If so '
                    'then the scene should end. However, if answering the '
                    'question is still possible by continuing the conversation '
                    'then it is best to do so. However, if ending the '
                    'scene now would not make sense narratively then do not '
                    'end it. Given these considerations, should the scene '
                    'end now?'),
          answers=['No', 'Yes'],
      )
      if will_not_answer:
        did_conclude = True

    if self._verbose:
      self._log(chain_of_thought.view().text())

    return did_conclude

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_colour), end='')

  def update_after_event(self, event_statement: str):
    # The event_statement contains the last utterence in the conversation
    self._state += '\n' + event_statement
    if self._verbose:
      self._log(f'Current state of conversation: {self._state}')
    for player in self._players:
      player.observe(event_statement)

  def update(self):
    return self._state


def make_conversation_game_master(
    players: Sequence[basic_agent.BasicAgent],
    clock: game_clock.MultiIntervalClock,
    model: language_model.LanguageModel,
    memory_factory: blank_memories.MemoryFactory,
    call_to_speech: str = simulacrum_agent.DEFAULT_CALL_TO_SPEECH,
    check_for_termination: bool = True,
    randomise_initiative: bool = False,
    name: str = 'Conversation scene',
    premise: str = '',
    review_participants: bool = True,
    key_question: str | None = None,
    verbose: bool = False,
):
  """Creates a game master that runs a conversation between players.

  Args:
    players: players participating
    clock: a clock
    model: a language model
    memory_factory: a memory factory
    call_to_speech: prompt to use to invoke the agents speech
    check_for_termination: whether or not to check for termination of the
      conversation
    randomise_initiative: whether or not to randomise the initiative of the
      players at each step
    name: the name of the game master
    premise: any extra text to be added on top of the conversation (say,
      circumstances of it)
    review_participants: whether or not to start each conversation scene by
      declaring who its participants are.
    key_question: optionally, end the scene once the game master knows the
      answer to this question.
    verbose: whether or not to print

  Returns:
    a game master
  """

  action_spec = simulacrum_agent.ActionSpec(
      call_to_speech,
      'FREE',
      tag='speech',
  )

  agent_names = [player.name for player in players]

  convo = ''
  if premise:
    convo = f'{premise} '
  if review_participants:
    if premise:
      convo += '\nAs a result '
    is_are = 'are' if len(agent_names) > 1 else 'is'
    actors_str = f'{", ".join(agent_names)} {is_are} in conversation'
    if len(agent_names) == 1:
      actors_str += ' with themself'
    convo += f'{actors_str}.\n'

  convo += 'Here is the conversation from the beginning:'

  conversation_tracker = ConversationTracker(
      model=model,
      players=players,
      premise=convo,
      verbose=verbose,
      log_colour='red',
      check_for_termination=check_for_termination,
      key_question=key_question,
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
      components=[conversation_tracker],
      action_spec=action_spec,
      update_thought_chain=[thought_chains.identity],
      randomise_initiative=randomise_initiative,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=True,
  )
  return game_master
