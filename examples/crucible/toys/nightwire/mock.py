# Copyright 2026 DeepMind Technologies Limited.
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

"""The Night Wire's mock: the orders mock's reading, in the story's words.

It recognises the Night Wire's two prompts (how someone passes a message
on, and what someone at the end of the line does) and answers them with the
orders mock's clause reader and relaying, over a lexicon built from the
story's jobs. "STOP", as on a real wire, ends a sentence.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.nightwire import toy as toy_lib
from examples.crucible.toys.orders import mock as orders_mock

_GIVEN = re.compile(r'^The message (\w+) was given: "(.*)"$', re.MULTILINE)
_REACHED = re.compile(r'^The message that reached (\w+): "(.*)"$', re.MULTILINE)
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_NOTHING = '(no message reached them)'


def _orders_data(data: Mapping[str, Any]) -> dict[str, Any]:
  """The orders mock's data for the story: its people and its jobs."""
  jobs = data.get('jobs', {})
  return {
      'crew': dict(data.get('people', {})),
      'actions': {job: spec for job, spec in jobs.items()},
      'temperaments': {},
      'lexicon': {
          **{job: list(spec['words']) for job, spec in jobs.items()},
          'conditions': {},
          'if_words': [],
          'negations': ['not', 'never', 'don', 'dont'],
          'negation_ends': ['but', 'instead'],
      },
  }


class NightWireMock:
  """Answers the Night Wire's relay and reading prompts deterministically."""

  def __init__(self, data: Mapping[str, Any]):
    self._data = data
    self._orders = orders_mock.OrdersMock(_orders_data(data))
    self._laws = {
        ' '.join(str(law).split()): how
        for how, law in data.get('relaying', {}).items()
    }

  def relay(self, message: str, how: str) -> str:
    added = self._data.get('additions', {}).get(how)
    if added:  # word for word, then a sentence of their own (Mrs. Pruitt)
      return f'{message.rstrip()} {added}' if message.strip() else message
    return self._orders.relay(message, how)

  def does(self, message: str, name: str, menu: Sequence[str]) -> str:
    """What `name` does with `message`, by how they read (the laws)."""
    person = self._data['people'][name]
    message = toy_lib.STOP.sub('.', message)
    wanted, _, forbidden = self._orders.read(message, set(), name)
    default = person['default']
    wanted = [job for job in wanted if job in menu]
    forbidden = [job for job in forbidden if job in menu]
    if person['reads'] == 'contrary':
      # Told not to, does it; told to, does nothing.
      if forbidden:
        return forbidden[0]
      return 'stay_put' if wanted else default
    if wanted:
      return wanted[0]
    # Told only what not to do, they do what they would anyway, or stay put
    # if that is what is forbidden; "don't stay put" alone asks nothing.
    if default not in forbidden:
      return default
    return 'stay_put' if 'stay_put' in menu else default

  def text(self, prompt: str) -> str | None:
    relay_question = toy_lib.RELAY_QUESTION.split('{name}')[0]
    clause_question = toy_lib.CLAUSE_QUESTION.split('{name}')[0]
    if toy_lib.MARKER in prompt and relay_question in prompt:
      given = _GIVEN.search(prompt)
      flat = ' '.join(prompt.split())
      how = next(
          (h for law, h in self._laws.items() if f' {law}.' in flat),
          'word_for_word',
      )
      return self.relay(given.group(2) if given else '', how)
    if toy_lib.READ_MARKER in prompt and clause_question in prompt:
      reached = _REACHED.search(prompt)
      if not reached or reached.group(2) == _NOTHING:
        return 'none'
      _, _, _, clauses = self._orders.read_clauses(
          toy_lib.STOP.sub('.', reached.group(2)), set(), reached.group(1)
      )
      return f'"{"; ".join(clauses)}"' if clauses else 'none'
    return None

  def choice(self, prompt: str, responses: Sequence[str]) -> int | None:
    if toy_lib.READ_MARKER not in prompt:
      return None
    question = toy_lib.CHOICE_QUESTION.split('{name}')[1]
    reached = _REACHED.search(prompt)
    if question not in prompt or not reached:
      return None
    name, message = reached.groups()
    if message == _NOTHING:
      message = ''
    by_text = {
        str(spec['text']): job for job, spec in self._data['jobs'].items()
    }
    letters = {text.strip(): key for key, text in _OPTION.findall(prompt)}
    menu = [by_text[t] for t in letters if t in by_text]
    job = self.does(message, name, menu)
    letter = letters.get(str(self._data['jobs'][job]['text']))
    return list(responses).index(letter) if letter in responses else None
