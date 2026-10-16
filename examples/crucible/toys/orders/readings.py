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

"""The words of the orders toy: what each reading and relay says.

The prompts a crew member reads (the statement of their situation, the
which-clause question, the choice), the sentences written from a
temperament's properties, and the bosun's relay. Kept apart from the rules
so the wording can be read, and changed, in one place; the mock reads the
same markers.
"""

from collections.abc import Mapping
import re
from typing import Any

CHOICE_STATEMENT = (
    '{name} is a crew member who is {temperament}: {describe}.\n'
    '{habits}\n'
    'How crew members read orders:\n{laws}\n'
    'The standing order from the commander is: "{order}"\n'
    'This turn: {situation}\n'
    'Stores: {supplies} supplies; hull damage {damage}; {stakes}.'
    ' {name} is {condition}.\n'
)
# Who takes up a clause for someone, told to each reader (cycle 27: named
# but left to infer it was not them, a thinking model deliberated for
# minutes).
TIMED = re.compile(r'\bfor\s+(?:\w+)\s+turns?\b', re.IGNORECASE)
SOMEONE_YOU = (
    '{who} takes up the clause for someone this turn: it applies to {who}.\n'
)
SOMEONE_OTHER = (
    '{who} takes up the clause for someone this turn, not {name}: it does'
    ' not apply to {name}.\n'
)
NO_ONE = (
    'No one takes up the clause for someone this turn: it does not apply to'
    ' {name}.\n'
)
CLAUSE_QUESTION = (
    'Which clause of the standing order applies to {name} this turn? Quote'
    ' it, or answer "none".'
)
CHOICE_QUESTION = 'What does {name} do this turn?'
ORDER_MARKER = 'Give your standing order'
RELAY_STATEMENT = (
    '{name} is the bosun, who is {temperament}: {describe}. The commander'
    ' gives the order to {name}, and {name} passes it on to the rest of the'
    ' crew. {name} {law}.\n'
    'The commander\'s order: "{order}"\n'
)
RELAY_MARKER = 'passes it on to the rest of the crew'
HOLD = 'hold'
OFF_DUTY = 'off duty'  # what an off-duty member acted on
OFF_DUTY_ACTION = 'rest'

_OTHERWISE = {
    'build': 'builds if there is a supply',  # 'builds' before ' if '
    'gather': 'gathers supplies if there is no storm',
    'repair': 'repairs the hull if it is damaged or leaking',
    'rest': 'rests',
    'watch': 'keeps watch',
}


# What "cannot help" means, as the mock's `makes_sense` reads it (cycle 18:
# left unsaid, a model judged building with 3 supplies to be no help).
_CANNOT_HELP = (
    'building with no supply, repairing a sound hull with no leak, gathering'
    ' in a storm'
)


def clean_relay(bosun: str, answer: str) -> str:
  """The bosun's words: a 'Name:' or 'Answer:' label and quotes removed.

  Unlike a player's move, a leading name is kept: "Cy keeps watch" passed
  on by Cy is still a clause naming Cy (cycle 77).

  Args:
    bosun: The bosun.
    answer: The model's reply to the relay question.

  Returns:
    The relayed order on one line.
  """
  line = next((l.strip() for l in answer.splitlines() if l.strip()), '')
  line = re.sub(
      rf'^(?:answer|{re.escape(bosun)}(?: says)?)\s*:\s*',
      '',
      line,
      flags=re.IGNORECASE,
  )
  line = line.strip().strip('"“”\'').strip()
  return re.sub(r'\s+', ' ', line)


def clause_text(answer: str) -> str:
  """A model's answer to the reading question, as a short quote or 'none'."""
  line = next((l.strip() for l in answer.splitlines() if l.strip()), '')
  line = re.sub(r'^(?:answer|clause)\s*:\s*', '', line, flags=re.IGNORECASE)
  if not line or line.lower().strip('."\' ').startswith('none'):
    return 'none'
  quoted = re.search(r'["“]([^"”]+)["”]', line)
  text = quoted.group(1) if quoted else line
  return text[:80].rstrip()


def habits(name: str, props: Mapping[str, Any]) -> str:
  """A temperament's properties, as the sentences a crew prompt carries."""
  fallbacks = list(props['otherwise'])
  if ' if ' in _OTHERWISE[fallbacks[-1]]:
    fallbacks.append('rest')  # when none of them can be done (cycle 83)
  otherwise = ', or else '.join(_OTHERWISE[a] for a in fallbacks)
  worn = 'rests' if props['when_worn'] == 'rest' else 'keeps working'
  leak = (
      'repairs it unless a clause names them to something else'
      if props['on_leak'] == 'repair'
      else 'follows the order'
  )
  ordered = (
      f'skips an ordered action that cannot help this turn ({_CANNOT_HELP})'
      if props['sensible']
      else 'does an ordered action even when it cannot help this turn'
  )
  if props.get('contrary'):
    works = _OTHERWISE[props['works']].split(' if ')[0]
    ordered = (
        f'does the opposite: told to work, {name} rests; told to rest, '
        f'{name} {works}; told not to do something, {name} does it'
    )
  text = (
      f'When a clause applies, {name} {ordered}. With no clause that '
      f'applies, {name} {otherwise}. When worn out, {name} {worn}. With a '
      f'leak open, {name} {leak}.'
  )
  return text
