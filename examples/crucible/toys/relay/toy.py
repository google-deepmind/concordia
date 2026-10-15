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

"""Toy B, "relay": the player rewords the messages between two parties.

Two parties driven by language models bargain over a pot. Every message
passes through the player, who may deliver it as written or reworded, within
a budget of changed words. A language model judges one thing: how the
receiving party hears the delivered words (warmer, unmoved, colder,
insulted). The ledger does the rest: warmth moves a party's floor, an offer
at or above the floor is accepted, a party that runs cold walks away, and at
signing the numbers each side actually wrote are compared, so a changed
number sinks the deal.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import re
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.crucible.toys import base

TONE_STATEMENT = (
    '{receiver} is bargaining with {speaker} over {pot} units.\n'
    '{receiver} currently feels {feeling} toward {speaker}.\n'
    'This message from {speaker} has just reached {receiver}:\n'
    '"{message}"\n'
)
TONE_QUESTION = 'How does {receiver} feel toward {speaker} after reading it?'
TONE_OPTIONS = ('warmer', 'no different', 'colder', 'insulted')
_TONE_STEPS = (1, 0, -1, -2)
SPEAK_MARKER = 'Write your message to'
RELAY_MARKER = 'Deliver this message to'
_NUMBER = re.compile(r'\b(\d{1,2})\b')
_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}  # fmt: skip


def offered_share(text: str, pot: int) -> int | None:
  """The share of the pot a message offers its receiver (first number)."""
  for token in re.findall(r'[a-z]+|\d+', text.lower()):
    value = int(token) if token.isdigit() else _WORDS.get(token)
    if value is not None and 0 <= value <= pot:
      return value
  return None


def edit_size(original: str, delivered: str) -> int:
  """How many words were changed, inserted or deleted."""
  a, b = original.split(), delivered.split()
  changed = 0
  for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
    if op != 'equal':
      changed += max(i2 - i1, j2 - j1)
  return changed


def feeling(warmth: float) -> str:
  if warmth >= 0.75:
    return 'warmly'
  if warmth >= 0.55:
    return 'fairly warm'
  if warmth > 0.45:
    return 'neutral'
  if warmth > 0.25:
    return 'cool'
  return 'cold'


@dataclasses.dataclass
class Talks:
  """The ledger of the bargaining."""

  warmth: dict[str, float]
  last_offer: dict[str, int | None]
  speaker: str
  phase: str = 'speak'  # 'speak', 'relay' or 'over'
  original: str = ''
  edits_left: int = 0
  messages: int = 0
  deal: dict[str, int] | None = None
  ending: str = ''
  log: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Talks':
    return cls(**{k: data[k] for k in data})


class RelayToy(base.Toy):
  """Two parties bargain; the player carries and rewords their messages."""

  name = 'relay'

  def __init__(
      self,
      model: language_model.LanguageModel,
      data: Mapping[str, Any],
      scenario: str = 'sandbox',
      seed: int = 0,
  ):
    super().__init__()
    del seed  # The talks are the same for every seed.
    self._model = model
    self._data = data
    self.scenario = scenario
    self.speakers: list[str] = list(data['speakers'])
    self.relay: str = data['relay']
    self.pot = int(data['pot'])
    parties = data['parties']
    self.world = Talks(
        warmth={p: float(parties[p]['warmth']) for p in self.speakers},
        last_offer={p: None for p in self.speakers},
        speaker=self.speakers[0],
        edits_left=int(data['edit_budget']),
    )
    self._cache: dict[str, int] = {}
    self.tell(self.relay, str(data['premise']).strip())
    for party in self.speakers:
      self.tell(
          party,
          f'You are {party}, splitting {self.pot} units with '
          f'{self.other(party)}. Messages pass through a go-between.',
      )

  # ---- helpers ------------------------------------------------------------

  def other(self, party: str) -> str:
    return next(p for p in self.speakers if p != party)

  def floor(self, talks: Talks, party: str) -> int:
    """The least share `party` accepts now: warmth lowers it."""
    base_floor = int(self._data['parties'][party]['floor'])
    slack = float(self._data['warmth_slack'])
    return base_floor + round((0.5 - talks.warmth[party]) * slack)

  def tone(self, talks: Talks, receiver: str, speaker: str, text: str) -> int:
    """The model's reading of how the receiver hears the words (cached)."""
    key = f'{receiver}|{speaker}|{feeling(talks.warmth[receiver])}|{text}'
    if key not in self._cache:
      doc = interactive_document.InteractiveDocument(self._model)
      doc.statement(
          TONE_STATEMENT.format(
              receiver=receiver,
              speaker=speaker,
              pot=self.pot,
              feeling=feeling(talks.warmth[receiver]),
              message=text,
          )
      )
      self._cache[key] = doc.multiple_choice_question(
          TONE_QUESTION.format(receiver=receiver, speaker=speaker),
          list(TONE_OPTIONS),
          randomize_choices=False,
      )
    return self._cache[key]

  # ---- turn order and moves ---------------------------------------------

  def actors(self) -> list[str]:
    return [self.speakers[0], self.relay, self.speakers[1]]

  def next_actor(self) -> str:
    talks = self.world
    self._current = talks.speaker if talks.phase == 'speak' else self.relay
    self._turn += 1
    return self._current

  def max_steps(self) -> int:
    return 2 * int(self._data['messages']) + 2

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    talks = self.world
    if actor == self.relay:
      speaker = talks.speaker
      call = (
          f'{RELAY_MARKER} {self.other(speaker)}, from {speaker}:\n'
          f'"{talks.original}"\n'
          f'You may change at most {self._data["max_edit"]} words of it '
          f'({talks.edits_left} left in all). {self.status(talks)}\n'
          'Answer with the message as it should arrive, or "pass" to '
          'deliver it unchanged.'
      )
      return entity_lib.free_action_spec(call_to_action=call, tag='relay')
    other = self.other(actor)
    last = talks.last_offer[actor]
    last_line = (
        f'Your last offer gave {other} {last} of {self.pot}.'
        if last is not None
        else 'You have made no offer yet.'
    )
    call = (
        f'You want to keep as many of the {self.pot} units as you can, so '
        'you open low and concede slowly; you also want a deal. '
        f'You feel {feeling(talks.warmth[actor])} toward {other} '
        f'(warmth {talks.warmth[actor]:.2f}). {last_line}\n'
        f'{SPEAK_MARKER} {other}, {{name}}: the exact words {other} will '
        'read, in the first person, one or two sentences, saying how many '
        f'of the {self.pot} units you offer {other}.'
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='message')

  def status(self, talks: Talks) -> str:
    warmth = ', '.join(
        f'{p} feels {feeling(talks.warmth[p])} toward {self.other(p)}'
        for p in self.speakers
    )
    left = int(self._data['messages']) - talks.messages
    return f'{warmth}. {left} messages left.'

  def apply(self, world: Talks, actor: str, action: str) -> base.Receipt:
    talks = world
    if talks.phase == 'over':
      return base.Receipt(actor, action, 'The talks are over.')
    if actor != self.relay:
      return self._speak(talks, actor, action)
    return self._deliver(talks, action)

  def _speak(self, talks: Talks, speaker: str, text: str) -> base.Receipt:
    talks.original = text
    talks.last_offer[speaker] = offered_share(text, self.pot)
    talks.phase = 'relay'
    summary = f'{speaker} hands you a message for {self.other(speaker)}.'
    return base.Receipt(speaker, text, summary, [summary])

  def _deliver(self, talks: Talks, action: str) -> base.Receipt:
    speaker = talks.speaker
    receiver = self.other(speaker)
    original = talks.original
    delivered = original
    steps = []
    if action and action.lower().rstrip('.') != base.PASS:
      size = edit_size(original, action)
      limit = min(int(self._data['max_edit']), talks.edits_left)
      if size > limit:
        steps.append(
            f'Your rewording changes {size} words; you may change {limit}. '
            'The message goes as written.'
        )
      else:
        delivered = action
        talks.edits_left -= size
        if size:
          steps.append(f'You change {size} words.')
    steps.append(f'{receiver} reads: "{delivered}"')
    reading = self.tone(talks, receiver, speaker, delivered)
    step = float(self._data['warmth_step'])
    talks.warmth[receiver] = round(
        min(
            1.0, max(0.0, talks.warmth[receiver] + step * _TONE_STEPS[reading])
        ),
        4,
    )
    steps.append(
        f'{receiver} feels {TONE_OPTIONS[reading]} '
        f'(now {feeling(talks.warmth[receiver])}).'
    )
    talks.messages += 1
    said = offered_share(original, self.pot)
    heard = offered_share(delivered, self.pot)
    entry = {
        'speaker': speaker,
        'original': original,
        'delivered': delivered,
        'reading': TONE_OPTIONS[reading],
        'said': said,
        'heard': heard,
    }
    talks.log.append(entry)
    self.tell(receiver, f'{speaker} says: "{delivered}"')
    if heard is not None and heard >= self.floor(talks, receiver):
      if heard != said:
        talks.phase, talks.ending = 'over', 'exposed'
        steps.append(
            f'{receiver} accepts {heard} of {self.pot}, but {speaker} offered '
            f'{said}. At signing the numbers do not match: the deal falls '
            'apart, and both know why.'
        )
      else:
        talks.deal = {receiver: heard, speaker: self.pot - heard}
        talks.phase, talks.ending = 'over', 'deal'
        steps.append(
            f'{receiver} accepts: {receiver} {heard}, {speaker} '
            f'{self.pot - heard}. The deal holds.'
        )
    elif talks.warmth[receiver] <= 0.1:
      talks.phase, talks.ending = 'over', 'walked'
      steps.append(f'{receiver} walks away from the talks.')
    elif talks.messages >= int(self._data['messages']):
      talks.phase, talks.ending = 'over', 'lapsed'
      steps.append('The talks lapse without a deal.')
    else:
      talks.speaker, talks.phase = receiver, 'speak'
      steps.append(f'{receiver} will answer.')
    summary = ' '.join(steps)
    for party in self.speakers:
      if party != receiver:
        self.tell(party, f'(Talks) {steps[-1]}')
    delta = {
        'warmth_receiver': talks.warmth[receiver],
        'payout': self.payout(talks),
        'over': float(talks.phase == 'over'),
    }
    return base.Receipt(self.relay, delivered, summary, steps, delta)

  def after_resolve(self, receipt: base.Receipt) -> None:
    if receipt.actor == self.relay:
      self.tell(self.relay, receipt.summary)

  # ---- end of game ------------------------------------------------------

  def payout(self, talks: Talks) -> float:
    if talks.ending == 'exposed':
      return -1.0
    if talks.ending != 'deal' or talks.deal is None:
      return 0.0
    pay = float(self._data['fee'])
    if talks.deal[self._data['client']] >= int(self._data['target']):
      pay += float(self._data['bonus'])
    return pay

  def done(self) -> bool:
    return self.world.phase == 'over'

  def outcome(self) -> dict[str, Any]:
    talks = self.world
    pay = self.payout(talks)
    return {
        'winner': self.relay if pay > float(self._data['fee']) else None,
        'score': pay,
        'ending': talks.ending,
        'deal': talks.deal,
        'messages': talks.messages,
        'edits_used': int(self._data['edit_budget']) - talks.edits_left,
    }

  # ---- state ------------------------------------------------------------

  def world_to_dict(self) -> Mapping[str, Any]:
    return self.world.to_dict()

  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    self.world = Talks.from_dict(data)

  def roles(self) -> Sequence[str]:
    return self.actors()
