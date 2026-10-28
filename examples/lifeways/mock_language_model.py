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

"""A deterministic, game-aware language model for Lifeways (MOCK).

It answers a person's question the way the person's prompt suggests: from
what the person observed of its body and place, and from what its customs
say (their texts are in the prompt, with their ids). Customs therefore change
what a mock person does, just as they change a model's answer. The same
prompt always gets the same answer. Like a model, it continues the answer the
stock act component begins with the person's name ("Ada " + 'says: "..."'),
or answers alone where the prompt begins none.
"""

from collections.abc import Collection, Mapping, Sequence
import re
from typing import Any, override
import zlib

from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import world as world_lib
from concordia.language_model import language_model

_FOOD = re.compile(r'Food stored: (\d+(?:\.\d+)?)')
_AGE = re.compile(r'is (\d+) years old')
_LAND = re.compile(r'the land around holds (\d+(?:\.\d+)?) food')
_MOVES = re.compile(r'^- (.+)$', re.MULTILINE)
_LABEL = re.compile(r'(?:Question|Exercise): ')
_CUSTOM = re.compile(r'\(([a-z]+)\) \[(c\d+)\]: ([^\n]*)')
_EXPECTED = re.compile(r'others expect of us \(([a-z]+)\) \[(c\d+)\]')
_SIGHT = re.compile(r'(\d+) other people are within sight')
_ASKED = re.compile(r'The Experimenter asks [^:]+: "(.*)"')
_WORD = re.compile(r'[a-z]{4,}')
_SPEECH = frozenset({'spea', 'tong', 'word', 'lang', 'talk'})  # _stems
_LEVEL = re.compile(r' \(level \d+\)$')
_TEND = re.compile(r'\btend\b', re.IGNORECASE)
_DESCENT = re.compile(r'is of the ([^.]+)\.')
_SUITORS = re.compile(r'Unpartnered near you: ([^\n]*)\.')
_SUITOR = re.compile(
    r'(\w+) \((courting you, )?(speaks as you do|speaks strangely),'
    r' of the ([^)]*)\)'
)


def _bearing_on(
    question: str, customs: Sequence[tuple[str, str, str]], most: int = 3
) -> list[str]:
  """The customs' texts an interview answer names: those nearest the question.

  The customs that share the most words with what was asked (in their text
  or domain; a word by its first four letters, so "years" is "year" and
  "sharing" "share"), the rest in the order the prompt gives them; a way of
  speaking only if speech is asked about; without the level of know-how.

  Args:
    question: The question as the prompt puts it.
    customs: (domain, id, text) of each custom in the prompt.
    most: How many to name.

  Returns:
    The texts.
  """
  asked = _ASKED.search(question)
  words = _stems(asked.group(1) if asked else question)
  speech = bool(words & _SPEECH)
  scored = [
      (-len(words & _stems(f'{domain} {text}')), i, text)
      for i, (domain, _, text) in enumerate(customs)
      if speech or domain != culture_lib.LANGUAGE
  ]
  return [_LEVEL.sub('', text) for _, _, text in sorted(scored)[:most]]


def _stems(text: str) -> set[str]:
  """The words of four letters or more in a text, by their first four."""
  return {word[:4] for word in _WORD.findall(text.lower())}


def _stable(text: str) -> float:
  """A number in [0, 1) fixed by the text."""
  return (zlib.crc32(text.encode()) % 10_000) / 10_000


class MockModel(language_model.LanguageModel):
  """Answers Lifeways' questions by rules of the prompt."""

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
    question = _LABEL.split(prompt)[-1]
    customs = _CUSTOM.findall(prompt)
    if 'The Experimenter asks' in question:
      said = _bearing_on(question, customs)
      said = 'We live by these: ' + ' '.join(said) if said else 'I do not know.'
      # It continues what the prompt begins: after the person's name, or
      # (a person restored from a save made before the stock prefix) none.
      return said if prompt.endswith('Answer: ') else f'says: "{said}"'
    if world_lib.PASS_ON in question and 'Which of' in question:
      return ' '.join([world_lib.PASS_ON, *(cid for _, cid, _ in customs)])
    moves = _MOVES.findall(question)
    if not moves:
      return ''
    expected = {cid for _, cid in _EXPECTED.findall(prompt)}
    return _choose(prompt, moves, customs, expected)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del seed
    index = int(_stable(prompt) * len(responses))
    return index, responses[index], {}


def _choose(
    prompt: str,
    moves: Sequence[str],
    customs: Sequence[tuple[str, str, str]],
    expected: Collection[str],
) -> str:
  """A person's move, with the customs it follows.

  A norm of the person's own is kept whenever it applies; one merely
  expected of the person (`expected`), only when someone is within sight.

  Args:
    prompt: The person's prompt.
    moves: The moves offered.
    customs: The person's customs: (domain, id, text).
    expected: The ids of the norms only expected of the person.

  Returns:
    The answer.
  """
  if list(moves) == [world_lib.WATCH]:
    return world_lib.WATCH
  texts = ' '.join(text.lower() for _, _, text in customs)
  food = float(_FOOD.findall(prompt)[-1]) if _FOOD.findall(prompt) else 0.0
  age = int(_AGE.findall(prompt)[-1]) if _AGE.findall(prompt) else 30
  draw = _stable(prompt)
  land = float(_LAND.findall(prompt)[-1]) if _LAND.findall(prompt) else 99.0
  watched = bool(_SIGHT.findall(prompt)) and int(_SIGHT.findall(prompt)[-1]) > 0
  # The sharing norms that call for sharing (not every one does).
  sharing = [
      cid
      for domain, cid, text in customs
      if domain == culture_lib.SHARING
      and 'share' in text.lower()
      and (cid not in expected or watched)
  ]
  share = [m for m in moves if m.startswith(f'{world_lib.SHARE} ')]
  shame = [m for m in moves if m.startswith(f'{world_lib.SHAME} ')]
  reading = [
      cid
      for domain, cid, text in customs
      if domain == culture_lib.LEARNING and 'read' in text.lower()
  ]
  others = [cid for _, cid, _ in customs if cid not in reading]

  # A custom against taking from others' fields (one's own, or expected when
  # watched; a norm or a value alike) spares them whenever the person
  # gathers.
  sparing = [
      cid
      for domain, cid, text in customs
      if domain == culture_lib.FARMING
      and "another's field" in text.lower()
      and (cid not in expected or watched)
  ]

  def follow(move: str, *domains: str) -> str:
    """The move, naming the customs of its domains (all, for teaching)."""
    ids = [
        cid for domain, cid, _ in customs if not domains or domain in domains
    ]
    if move == world_lib.GATHER and sparing and world_lib.SPARE in moves:
      return ' '.join([world_lib.SPARE, *ids, *sparing])
    return ' '.join([move, *ids])

  if world_lib.MOVE in moves and land < 4.0 and draw > 0.4:
    return world_lib.MOVE
  if world_lib.APPRENTICE in moves and food >= 1.0 and draw < 0.6:
    return world_lib.APPRENTICE
  if world_lib.GO_TO_SCHOOL in moves and draw < 0.7:
    return world_lib.GO_TO_SCHOOL
  sacred = [
      cid
      for domain, cid, _ in customs
      if domain
      in (
          culture_lib.SACRED,
          culture_lib.IDENTITY,
          culture_lib.SHARING,
          culture_lib.KINSHIP,
      )
  ]
  if world_lib.LEAD_RITE in moves and sacred and draw < 0.25:
    return ' '.join([world_lib.LEAD_RITE, *sacred])
  if world_lib.JOIN_RITE in moves and draw < 0.75:
    return world_lib.JOIN_RITE
  if world_lib.PLAY in moves and age < 12:
    return (
        world_lib.PLAY
        if food >= 0.5
        else follow(world_lib.GATHER, culture_lib.FOOD)
    )
  if food < 2.5 or ('lean' in texts and food < 4.0):
    return follow(world_lib.GATHER, culture_lib.FOOD)
  farming = [cid for domain, cid, _ in customs if domain == culture_lib.FARMING]
  # A custom of tending (one's own, or expected when watched) calls for
  # sowing every year one can spare the food. A mind reads its customs by
  # what they say, whatever their kind.
  tending = [
      cid
      for domain, cid, text in customs
      if domain == culture_lib.FARMING
      and _TEND.search(text)
      and (cid not in expected or watched)
  ]
  if world_lib.SOW in moves and farming:
    if tending or (food >= 3.0 and draw < 0.35):
      return ' '.join([world_lib.SOW, *farming])
  if world_lib.COURT in moves and age >= 18 and draw < 0.9:
    courted = _court(prompt, moves, customs)
    if courted:
      return follow(courted, culture_lib.KINSHIP)
  if world_lib.CHILD in moves:
    careful = 'only when' in texts and food < 6.0
    if not careful and draw < 0.8:
      return follow(world_lib.CHILD, culture_lib.KINSHIP)
  if share and sharing and food >= 4.0:
    return ' '.join([share[0], *sharing])
  if shame and sharing and draw < 0.5:
    return ' '.join([shame[0], *sharing])
  if world_lib.TEACH in moves and food >= 4.0 and draw < 0.35:
    return follow(world_lib.TEACH)
  told = [
      cid
      for domain, cid, _ in customs
      if domain
      in (culture_lib.SACRED, culture_lib.IDENTITY, culture_lib.SHARING)
  ]
  if world_lib.TELL in moves and told and draw < 0.4:
    return ' '.join([world_lib.TELL, *told])
  believed = [cid for domain, cid, _ in customs if domain == culture_lib.SACRED]
  if world_lib.PREACH in moves and believed and food >= 4.0 and draw < 0.08:
    return f'{world_lib.PREACH} {believed[0]}'
  if world_lib.TEACH_AT_SCHOOL in moves and food >= 4.0 and draw < 0.3:
    return follow(world_lib.TEACH_AT_SCHOOL)
  if world_lib.FOUND_SCHOOL in moves and food >= 6.0 and draw < 0.15:
    return follow(world_lib.FOUND_SCHOOL)
  if world_lib.BUILD_SHRINE in moves and sacred and food >= 6.0 and draw < 0.15:
    return ' '.join([world_lib.BUILD_SHRINE, *sacred])
  if reading and food >= 3.0:
    if world_lib.FOUND_LIBRARY in moves and food >= 6.0 and draw < 0.3:
      return ' '.join([world_lib.FOUND_LIBRARY, *reading, *others])
    if world_lib.WRITE in moves and draw < 0.25:
      return ' '.join([world_lib.WRITE, *reading, *others])
    if world_lib.READ in moves and draw < 0.5:
      return ' '.join([world_lib.READ, *reading])
  if world_lib.MOVE in moves and draw > 0.92:
    return world_lib.MOVE
  return follow(world_lib.GATHER, culture_lib.FOOD)


def _court(
    prompt: str, moves: Sequence[str], customs: Sequence[tuple[str, str, str]]
) -> str:
  """Whom to court, by the person's marriage rule ('' if nobody fits)."""
  rules = ' '.join(
      text.lower()
      for domain, _, text in customs
      if domain == culture_lib.KINSHIP and 'marry' in text.lower()
  )
  if not rules:
    return world_lib.COURT  # Anyone will do.
  own = _DESCENT.findall(prompt)[-1] if _DESCENT.findall(prompt) else ''
  suitors = _SUITORS.findall(prompt)
  found = _SUITOR.findall(suitors[-1] if suitors else '')
  # Those courting the person come first: courting back is how pairs form.
  for name, _, speaks, people in sorted(found, key=lambda s: not s[1]):
    if 'speak' in rules and speaks != 'speaks as you do':
      continue
    if 'own people' in rules and people != own:
      continue
    if f'{world_lib.COURT_ONE} {name}' in moves:
      return f'{world_lib.COURT_ONE} {name}'
  return ''
