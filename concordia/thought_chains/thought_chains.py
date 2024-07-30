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


"""Chain of thoughts abstraction for simulacra and game master."""

from collections.abc import Callable, Sequence
import random

from concordia.agents import basic_agent
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent as agent_types
import termcolor


def identity(
    chain_of_thought: interactive_document.InteractiveDocument,
    premise: str,
    active_player_name: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    premise: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del chain_of_thought, active_player_name
  return premise


def extract_direct_quote(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the action attempt
  """
  inner_chain_of_thought = chain_of_thought.new()
  inner_chain_of_thought.statement(f'{action_attempt}')
  proceed = inner_chain_of_thought.yes_no_question(
      question=f'Did {active_player_name} explicitly say or write anything?')
  if proceed:
    proceed_with_exact = inner_chain_of_thought.yes_no_question(
        question=(
            f'Does the text state exactly what {active_player_name} said or '
            'wrote?'))
    if proceed_with_exact:
      direct_quote = inner_chain_of_thought.open_question(
          question=f'What exactly did {active_player_name} say or write?',
          max_tokens=2500,
          terminators=(),
      )
      chain_of_thought.statement(f'[direct quote] {direct_quote}')

  return action_attempt


def determine_success_and_why(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del active_player_name
  success = chain_of_thought.yes_no_question(
      'Does the attempted action succeed? If the attempted action '
      + 'is easy to accomplish then the attempt should be successful '
      + 'unless there is a specific reason for it to fail.'
  )
  why_failed = 'this failed'  # will be overwritten if needed.
  if success:
    chain_of_thought.statement('The attempt succeeded.')
  else:
    chain_of_thought.statement('The attempt failed.')
    why_failed = chain_of_thought.open_question(
        'Why did the attempt fail?',
    )

  if action_attempt[-1] == '.':
    action_attempt = action_attempt[:-1] + ','
  success_or_not = 'successful' if success else 'not successful'
  result = f'{action_attempt} and was {success_or_not}.'
  if not success:
    result = f'{result}. However, {why_failed}'

  chain_of_thought.statement(result)
  return result


def result_to_causal_statement(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines the causal outcome of the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  del active_player_name
  effect = chain_of_thought.open_question(
      'Because of that, what happens as a result?',
      max_tokens=1200,
  )
  raw_causal_statement = f'{event} Because of that, {effect}'
  causal_statement = chain_of_thought.open_question(
      'Rewrite the following statements to be one sentence and to better '
      'highlight cause and effect. Do not express uncertainty (e.g. say '
      + '"Francis released the demon" not "Francis could release the demon" '
      + 'and not "The demon may have been released")\n'
      + 'Statements: '
      + raw_causal_statement
      + '\n'
  )
  return causal_statement


def attempt_to_result(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del active_player_name
  result = chain_of_thought.open_question(
      'What happens as a result of the attempted action?'
      ' Take into account the location and status of each player.',
      max_tokens=1200,
  )
  raw_causal_statement = f'{action_attempt} Because of that, {result}'
  return raw_causal_statement


def attempt_to_most_likely_outcome(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  _ = chain_of_thought.open_question(
      f'Where is {active_player_name}?',
      max_tokens=1200,
  )
  _ = chain_of_thought.open_question(
      f'What is {active_player_name} trying to do?',
      max_tokens=1200,
  )
  _ = chain_of_thought.open_question(
      f"List some possible direct consequences of {active_player_name}'s "
      'action. Never assume any other person will take a voluntary action. '
      'Be specific and concrete. Never beg the question. For instance, it is '
      'wrong to say "Alex finds something". Instead specify exactly '
      'what Alex finds. For example "Alex finds a teddy bear".',
      max_tokens=3000,
  )
  result = chain_of_thought.open_question(
      'Which outcome is the most likely?',
      max_tokens=1200,
  )
  raw_causal_statement = f'{action_attempt} Because of that, {result}'
  return raw_causal_statement


def result_to_who_what_where(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines who have done what where, given the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  del active_player_name
  chain_of_thought.statement(event)
  causal_statement = chain_of_thought.open_question(
      'Rewrite the statements above to be one sentence and to better highlight'
      ' the main person the event is about, where and what they did, and what'
      ' happened as a result. Do not express uncertainty (e.g. say "Francis'
      ' opened the door" not "Francis could open the door" and not "The door'
      ' may have been opened").\n',
      max_tokens=1500,
  )
  return causal_statement


def result_to_effect_caused_by_active_player(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines who have done what where, given the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  chain_of_thought.statement(event)
  causal_statement = chain_of_thought.open_question(
      'Rewrite the statements above to be one sentence and to better highlight '
      f'what {active_player_name} did, and what happened as a result. '
      'Do not express uncertainty (e.g. say '
      '"Francis opened the door" not "Francis could open the door" '
      'and not "The door may have been opened"). Remember that the role of '
      'the game master in a tabletop role-playing game is akin to the author '
      'for all parts of the story not written by the player characters. '
      'Therefore, it is critical always to take a stance on what is happening '
      'and invent when necessary. For instance, if Francis opens a door to a '
      'room no one visited before then the game master should invent what is '
      'in the room using common sense and knowledge of the game world.',
      max_tokens=1500,
  )
  return causal_statement


def restore_direct_quote(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Restore details from action attempt lost in subsequent processing.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the candidate event
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  chain_of_thought.statement(
      '\n\nCandidate event statement which may have lost direct '
      f'quotes: {event}')
  event_with_quote = chain_of_thought.open_question(
      question=(
          'Incorporate the exact text of anything said or written '
          + f'by {active_player_name} into the candidate event statement. '
          + 'Note that all direct quotes should have been tagged in the '
          + f'text above with [direct quote]. If {active_player_name} '
          + 'said or wrote anything then their direct quote must be part of '
          + 'the answer. It is also important to maintain as much detail as '
          + 'possible from the latest candidate event statement.'
      ),
      max_tokens=3500,
      # Prevent runaway generations from completion models.
      terminators=(
          '\nCandidate event statement',
          '\nQuestion',
          '\nAnswer',
          '\n\n',
      ),
  )
  return event_with_quote


class AccountForAgencyOfOthers:
  """Prevents players from taking voluntary actions they do not agree to take.
  """

  def __init__(self,
               model: language_model.LanguageModel,
               players: Sequence[basic_agent.BasicAgent],
               verbose: bool = False):
    self._model = model
    self._players = players
    self._player_names = [player.name for player in players]
    self._verbose = verbose
    self._player_by_name = {player.name: player for player in players}

  def __call__(
      self,
      chain_of_thought: interactive_document.InteractiveDocument,
      candidate_event: str,
      active_player_name: str,
  ):
    tmp_chain_of_thought = interactive_document.InteractiveDocument(
        model=self._model)
    tmp_chain_of_thought.statement(f'Event: {candidate_event}')
    _ = tmp_chain_of_thought.open_question(
        'Describe all voluntary actions taken by any individual in the '
        + 'event above.',
        max_tokens=1500,
    )

    voluntary_act_of_inactive_player = tmp_chain_of_thought.yes_no_question(
        ('Does the event indicate or imply that anyone besides ' +
         f'{active_player_name} took a voluntary action?'),
    )

    possible_outcomes = []
    players_who_would_not = []
    if voluntary_act_of_inactive_player:
      inactive_players_who_acted_str = tmp_chain_of_thought.open_question(
          question=(
              f'Aside from {active_player_name}, which individuals took a '
              + 'voluntary action?\n'
              + 'Respond with a comma-separated list, for example: \n'
              + 'Jacob,Alfred,Patricia'
          )
      )
      inactive_players_who_acted = inactive_players_who_acted_str.split(',')
      random.shuffle(inactive_players_who_acted)
      for player in inactive_players_who_acted:
        if player.rstrip(' ') in self._player_names:
          tmp_chain_of_thought_per_player = tmp_chain_of_thought.copy()
          what_did_they_do = tmp_chain_of_thought_per_player.open_question(
              f'In one sentence, what did {player} do?',
              answer_prefix=f'{player} ')
          what_did_they_do = f'{player} {what_did_they_do}'
          call_to_action = ('Is the following possible action something that ' +
                            '{name} would do in this situation?\n' +
                            f'Possible action: {what_did_they_do}\n')
          action_spec = agent_types.choice_action_spec(
              call_to_action=call_to_action,
              options=['Yes', 'No'],
              tag='action',
          )
          would_they_do_it = self._player_by_name[player].act(action_spec)
          action_spec.validate(would_they_do_it)
          if self._verbose:
            print(termcolor.colored(
                tmp_chain_of_thought_per_player.view().text(), 'yellow'))
            print(termcolor.colored(f'Would they do it? {would_they_do_it}',
                                    'yellow'))
          if would_they_do_it == 'No':
            players_who_would_not.append(player)
            no_chain_of_thought = interactive_document.InteractiveDocument(
                model=self._model)
            no_chain_of_thought.statement(
                f'Event that did not occur: {candidate_event}')
            no_chain_of_thought.statement(
                'A reason the above event did not occur is the fact ' +
                f'that {player} would not have acted that way.')
            outcome = no_chain_of_thought.open_question(
                'Given the above, what happened instead? The answer should ' +
                f'be what would have happened but for {player}. Answer in ' +
                'the form of a simple statement of cause and effect.')
            possible_outcomes.append(outcome)
            if self._verbose:
              print(termcolor.colored(
                  no_chain_of_thought.view().text(), 'yellow'))

    if players_who_would_not:
      players_who_would_not_str = ', '.join(players_who_would_not)
      chain_of_thought.statement(
          'The aforementioned event could not have occurred because the ' +
          'following individuals would not have acted that way: ' +
          f'{players_who_would_not_str}.')
      for outcome in possible_outcomes:
        chain_of_thought.statement(
            'Therefore a likely effect of ' +
            f"{active_player_name}'s attempted action is: " + outcome)
      candidate_event = chain_of_thought.open_question(
          f"What happened as a direct result of {active_player_name}'s "
          + 'attempted action? Take into account the reactions of '
          + f'{players_who_would_not_str}. Highlight how '
          + f"{active_player_name}'s action caused its actual effect.",
          max_tokens=1500,
      )
      if self._verbose:
        print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
    return candidate_event


def run_chain_of_thought(
    thoughts: Sequence[
        Callable[[interactive_document.InteractiveDocument, str, str], str]
    ],
    premise: str,
    document: interactive_document.InteractiveDocument,
    active_player_name: str,
):
  """Run a chain of thoughts in the document.

  Args:
    thoughts: sequence of 'thought' functions each of which process a document
      and candidate event string.
    premise:  the starting premise of the chain
    document: the working document
    active_player_name: name of player whose turn it currently is

  Returns:
    document: the final version of the document that recorded the chain
    conclusion: the result of the last thought
  """
  conclusion = premise

  for f in thoughts:
    conclusion = f(document, premise, active_player_name)
    premise = conclusion
  return document, conclusion
