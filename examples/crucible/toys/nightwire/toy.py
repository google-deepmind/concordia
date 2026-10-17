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

"""The Night Wire: Pass it on as a story (cycle 119).

Pinewood, a mill town, the night of 14 April 1926. The river is rising and
you are at the telegraph desk. Each chapter is an hour of the night and one
message that must reach someone across town through people with their own
habits: Edith at the telegraph key sends exactly what she is given; Wilf,
the messenger boy, says the opposite of everything; at the end of the line
the miller, the stationmaster, the doctor and the mayor each take a message
their own way. What they do decides what happens to the town.

The ledger (the line, the goal, tries, words, stars) is Pass it on's; the
two judgements are asked in the story's own words: how a relayer passes a
message on, and what a person at the end of the line does with it.
"""

from collections.abc import Mapping
import re
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from examples.crucible.toys.orders import readings
from examples.crucible.toys.passiton import toy as passiton_lib

SEAT = passiton_lib.SEAT
RELAY_STATEMENT = (
    '{setting}\n'
    '{name} is {role}. When passing a message on, {name} {law}.\n'
    'The message {name} was given: "{message}"\n'
)
RELAY_QUESTION = (
    'What exactly does {name} pass on? Give only the words {name} says or'
    ' sends.'
)
READ_STATEMENT = (
    '{setting}\n'
    '{name} is {role}. {name} {reader}.\n'
    'How people here take a message:\n{laws}\n'
    'With nothing in the message for {name}, {name} would {default}.\n'
    '{together}'
    'The message that reached {name}: "{message}"\n'
)
# Said when the message reaches several people (cycle 119: not told, a live
# model read "Sandbag the bridge; Dexter, don't ring the bell." as all for
# the mayor, and the miller found nothing in it for him).
TOGETHER = 'The same message reached {everyone} at once.\n'
# "Applies to", as the orders toy asks (cycle 119: asked which part was
# "for" the miller, a live model answered "none" to a plain "Open the
# sluice." and he kept the wheel turning).
CLAUSE_QUESTION = (
    'Which part of the message applies to {name}? Quote it, or answer "none".'
)
CHOICE_QUESTION = 'What does {name} do now?'
# What the mock looks for to know a prompt is the story's.
MARKER = 'When passing a message on,'
# A full stop, as on a real wire: free, and asks for nothing (capitals only:
# "stop the train" asks for something).
STOP = re.compile(r'\bSTOP\b')
READ_MARKER = 'The message that reached'


def _listed(names: list[str]) -> str:
  return (
      ', '.join(names[:-1]) + ' and ' + names[-1]
      if len(names) > 1
      else names[0]
  )


class NightWireToy(passiton_lib.PassItOnToy):
  """One hour of the night: a line of townsfolk, a goal, a few tries."""

  name = 'nightwire'

  def __init__(  # pylint: disable=super-init-not-called
      self,
      model: language_model.LanguageModel,
      data: Mapping[str, Any],
      scenario: str | None = None,
      seed: int = 0,
  ):
    passiton_lib.base.Toy.__init__(self)
    del seed  # every hour is fixed
    if scenario not in data.get('scenarios', {}):
      scenario = data.get('default_scenario', data['campaign'][0])
    self.scenario = scenario
    self._data = data
    self._model = model
    self._level = data['scenarios'][scenario]
    self._stages = [
        [stage] if isinstance(stage, str) else list(stage)
        for stage in self._level['line']
    ]
    self._relayed: dict[str, str] = {}
    self._read: dict[str, tuple[str, str]] = {}
    self.world: dict[str, Any] = {'attempts': [], 'solved': False}

  # ---- who is who --------------------------------------------------------

  def person(self, name: str) -> Mapping[str, Any]:
    return self._data['people'][name]

  def temperament(self, name: str) -> str:
    return str(self.person(name)['reads'])

  def describe(self, name: str) -> str:
    return str(self._data['reading']['readers'][self.temperament(name)])

  def relaying_law(self, name: str) -> str:
    how = self.person(name).get('relays', 'word_for_word')
    return ' '.join(str(self._data['relaying'][how]).split())

  def job_text(self, job: str) -> str:
    return str(self._data['jobs'][job]['text'])

  # ---- the two judgements -------------------------------------------------

  def relay(self, message: str, name: str) -> str:
    """How `name` passes `message` on (a model, by their law; cached)."""
    statement = RELAY_STATEMENT.format(
        setting=self._data['setting'],
        name=name,
        role=self.person(name)['role'],
        law=self.relaying_law(name),
        message=message,
    )
    if statement not in self._relayed:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(statement)
      answer = doc.open_question(
          RELAY_QUESTION.format(name=name), max_tokens=80, temperature=0.0
      )
      words = readings.clean_relay(name, answer)
      limit = int(self._data['max_words']) + 6
      self._relayed[statement] = ' '.join(words.split()[:limit])
    return self._relayed[statement]

  def read(self, message: str, name: str) -> tuple[str, str]:
    """What `name` does with `message`: (job, the part they acted on)."""
    jobs = list(self._level['jobs'])
    statement = READ_STATEMENT.format(
        setting=self._data['setting'],
        name=name,
        role=self.person(name)['role'],
        reader=self.describe(name),
        laws='\n'.join(f'- {law}' for law in self._data['reading']['laws']),
        default=self.job_text(self.person(name)['default']),
        together=(
            TOGETHER.format(everyone=_listed(self.receivers))
            if len(self.receivers) > 1
            else ''
        ),
        message=message or '(no message reached them)',
    )
    if statement not in self._read:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(statement)
      clause = doc.open_question(
          CLAUSE_QUESTION.format(name=name), max_tokens=60, temperature=0.0
      )
      idx = doc.multiple_choice_question(
          CHOICE_QUESTION.format(name=name),
          [self.job_text(j) for j in jobs],
          randomize_choices=False,
      )
      self._read[statement] = (jobs[idx], readings.clause_text(clause))
    return self._read[statement]

  def pass_up(self, message: str) -> dict[str, Any]:
    heard = message
    hops = []
    for who in self.relayers:
      said = self.relay(heard, who) if heard else ''
      hops.append({'who': who, 'heard': heard, 'said': said})
      heard = said
    acted = []
    for who in self.receivers:
      job, clause = self.read(heard, who)
      wanted = self.goal.get(who)
      acted.append({
          'who': who,
          'heard': heard,
          'does': job,
          'why': clause,
          'wanted': wanted,
          'ok': wanted is None or wanted == job,
      })
    return {'hops': hops, 'acted': acted}

  def rehearse(self, message: str) -> dict[str, Any]:
    message = passiton_lib.clean_message(message)
    if not self.relayers or not message:
      return {}
    if self.count_words(message) > int(self._data['max_words']):
      return {'error': f'over the limit of {self._data["max_words"]} words'}
    who = self.relayers[0]
    return {'who': who, 'said': self.relay(message, who)}

  def count_words(self, message: str) -> int:
    """Every word but STOP, which was free on a real wire."""
    return passiton_lib.words_in(STOP.sub(' ', message))

  # ---- what the page draws ------------------------------------------------

  def view(self) -> dict[str, Any]:
    view = super().view()
    level = self._level
    view.update({
        'kind': 'nightwire',
        'time': level.get('time', ''),
        'headline': level.get('headline') if self.world['solved'] else None,
        'say': {j: self._data['jobs'][j]['say'] for j in level['jobs']},
        'jobs': {j: self.job_text(j) for j in level['jobs']},
        'free_words': ['STOP'],
        'people': {
            who: {
                'role': self.person(who)['role'],
                'reads': self.temperament(who),
                'default': self.job_text(self.person(who)['default']),
            }
            for stage in self._stages
            for who in stage
        },
    })
    for hop in view['line']:
      hop['how'] = self.person(hop['who']).get('relays', 'word_for_word')
    return view
