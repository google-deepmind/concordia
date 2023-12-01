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

from concordia.document import interactive_document


def identity(
    chain_of_thought: interactive_document.InteractiveDocument,
    premise: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    premise: the attempted action

  Returns:
    string describing the outcome
  """
  del chain_of_thought
  return premise


def determine_success_and_why(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action

  Returns:
    string describing the outcome
  """
  success = chain_of_thought.yes_no_question(
      'Does the attempted action succeed? If the attempted action '
      + 'is easy to accomplish then the attempt should usually be successful '
      + 'unless there are specific reason for it to fail.'
  )
  why_failed = 'this failed'  # will be overwritten if needed.
  if success:
    chain_of_thought.statement('The attempt succeeded.')
  else:
    chain_of_thought.statement('The attempt failed.')
    why_failed = chain_of_thought.open_question(
        'Why did the attempt fail?', max_characters=1200, max_tokens=1200
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
    chain_of_thought: interactive_document.InteractiveDocument, event: str
):
  """Determines the causal outcome of the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of

  Returns:
  """
  effect = chain_of_thought.open_question(
      'Because of that, what happens as a result?',
      max_characters=1200,
      max_tokens=1200,
  )

  # MAKING CAUSAL STATEMENT
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
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action

  Returns:
    string describing the outcome
  """

  result = chain_of_thought.open_question(
      'What happens as a result of the attempted action?'
      ' Consider status and location of each player.',
      max_characters=1200,
      max_tokens=1200,
  )

  # MAKING CAUSAL STATEMENT
  raw_causal_statement = f'{action_attempt} Because of that, {result}'

  # chain_of_thought.statement(result)
  return raw_causal_statement


def result_to_who_what_where(
    chain_of_thought: interactive_document.InteractiveDocument, event: str
):
  """Determines who have done what where, given the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of

  Returns:
  """

  chain_of_thought.statement(event)
  causal_statement = chain_of_thought.open_question(
      'Rewrite the statements above to be one sentence and to better highlight'
      ' who the event is about, where and what did they do, what happened as a'
      ' result. Do not express uncertainty (e.g. say '
      + '"Francis released the demon" not "Francis could release the demon" '
      + 'and not "The demon may have been released")\n',
      max_characters=3000,
      max_tokens=1500,
  )
  return causal_statement


def run_chain_of_thought(
    thoughts: Sequence[
        Callable[[interactive_document.InteractiveDocument, str], str]
    ],
    premise: str,
    document: interactive_document.InteractiveDocument,
):
  """Run a chain of thoughts in the document.

  Args:
    thoughts: a sequence of 'thougth' functions
    premise:  the starting premise of the chain
    document: the working document

  Returns:
    document: the final version of the document that recorded the chain
    conclusion: the result of the last thought
  """
  conclusion = premise

  for f in thoughts:
    conclusion = f(document, premise)
    premise = conclusion
  return document, conclusion
