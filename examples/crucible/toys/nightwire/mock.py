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

"""The Night Wire's mock: a rough, deterministic copy of the characters.

The live model decides in character (cycle 121); this mock is the CI's and
the bots' stand-in, not the judge. It recognises the Night Wire's two
prompts (how someone passes a message on, and what someone at the end of
the line does) and answers them with the orders mock's clause reader and
relaying over the story's jobs, each person's tags (`relays`, `reads`,
`default`) and a few opposites ("Dexter, silence!" asks him not to ring the
bell). "STOP", as on a real wire, ends a sentence.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any

from examples.crucible.toys.nightwire import toy as toy_lib
from examples.crucible.toys.orders import mock as orders_mock

_GIVEN = re.compile(
    r'^(\w+) has been given this message to pass on: "(.*)"$', re.MULTILINE
)
_REACHED = re.compile(r'^The message that reached (\w+): "(.*)"$', re.MULTILINE)
_OPTION = re.compile(r'^\s*\(([a-z])\) (.+)$', re.MULTILINE)
_NOTHING = '(no message reached them)'


def _orders_data(data: Mapping[str, Any]) -> dict[str, Any]:
  """The orders mock's data for the story: its people and its jobs."""
  jobs = data.get('jobs', {})
  # A job's opposites as an action of its own, "not_<job>" (cycle 121).
  opposites = {
      f'not_{job}': list(spec['opposite'])
      for job, spec in jobs.items()
      if spec.get('opposite')
  }
  return {
      'crew': dict(data.get('people', {})),
      'actions': {**jobs, **{name: {} for name in opposites}},
      'temperaments': {},
      'lexicon': {
          **{job: list(spec['words']) for job, spec in jobs.items()},
          **opposites,
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

  def relay(self, message: str, how: str) -> str:
    added = self._data.get('additions', {}).get(how)
    if added:  # word for word, then a sentence of their own (Mrs. Pruitt)
      return f'{message.rstrip()} {added}' if message.strip() else message
    return self._orders.relay(message, how)

  def does(self, message: str, name: str, menu: Sequence[str]) -> str:
    """What `name` does with `message`, roughly as their character would."""
    person = self._data['people'][name]
    message = toy_lib.STOP.sub('.', message)
    asked, _, told_not = self._orders.read(message, set(), name)
    # Asked for a job's opposite is told not to do it, and the other way.
    wanted = [j for j in asked if not j.startswith('not_')]
    wanted += [j[4:] for j in told_not if j.startswith('not_')]
    forbidden = [j for j in told_not if not j.startswith('not_')]
    forbidden += [j[4:] for j in asked if j.startswith('not_')]
    default = person['default']
    wanted = [job for job in wanted if job in menu]
    forbidden = [job for job in forbidden if job in menu]
    if person['reads'] == 'contrary':
      # Told not to, does it; told to, will not (told to stay put, goes).
      if forbidden:
        return forbidden[0]
      if wanted == ['stay_put']:
        return next((j for j in menu if j != 'stay_put'), default)
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
    take_question = toy_lib.TAKE_QUESTION.split('{name}')[0]
    if toy_lib.MARKER in prompt and relay_question in prompt:
      given = _GIVEN.search(prompt)
      if not given:
        return None
      person = self._data['people'].get(given.group(1), {})
      return self.relay(given.group(2), person.get('relays', 'word_for_word'))
    if toy_lib.READ_MARKER in prompt and take_question in prompt:
      reached = _REACHED.search(prompt)
      if not reached or reached.group(2) == _NOTHING:
        return 'Nothing reached them.'
      name = reached.group(1)
      _, _, _, clauses = self._orders.read_clauses(
          toy_lib.STOP.sub('.', reached.group(2)), set(), name
      )
      if not clauses:
        return f'{name} finds nothing in it for {name}.'
      return f'{name} takes it as "{"; ".join(clauses)}".'
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
