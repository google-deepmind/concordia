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

"""Rumours: news that spreads from person to person, and changes as it goes.

`Rumors` keeps the news going round a world and who has heard which version
of it. A rumour starts with those who saw what it tells. Each step, everyone
who heard it lately (within `tell_for` steps) tells it to each of their
contacts with chance `chance`; the game says who is in contact with whom
(for example the contrib `SocialTies.contacts`). A retelling keeps `keep` of
the teller's fidelity and, with chance `garble`, drops a word: what reaches
the far end of a chain is less sure, and shorter. A rumour nobody has told
for `forget_after` steps is forgotten, and at most `max_rumors` are kept
(the oldest go first), so a world that runs indefinitely stays bounded.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_RUMORS_COMPONENT_KEY`). It adds nothing to prompts: the game tells
each hearer what reached it (`spread` returns who heard what, from whom).
Draws are seeded by the rumour, the step and the names, so a run replays.

  rumors = rumors_lib.Rumors(seed=7)
  rumor_id = rumors.start('A flood at the ford.', ['Ada'], step=3)
  for heard in rumors.spread({'Ada': ['Bo'], 'Bo': ['Cy']}, step=4):
    tell(heard.hearer, f'Word reaches you from {heard.teller}: {heard.text}')
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import threading
from typing import Any, override
import zlib

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# The key a game master registers its rumours under, by convention.
DEFAULT_RUMORS_COMPONENT_KEY = '__rumors__'

# The chance a teller passes a rumour to a contact in a step; for how many
# steps after hearing one tells it; the fidelity a retelling keeps; the
# chance a retelling drops a word; after how many untold steps a rumour is
# forgotten; and the most rumours kept.
CHANCE = 0.5
TELL_FOR = 3
KEEP = 0.9
GARBLE = 0.2
FORGET_AFTER = 20
MAX_RUMORS = 50
# A rumour keeps at least this many words, however garbled.
_FEWEST_WORDS = 3


@dataclasses.dataclass(frozen=True)
class Heard:
  """A rumour reaching someone.

  Attributes:
    rumor_id: The rumour.
    hearer: Who heard it.
    teller: From whom.
    text: The version that reached the hearer.
    fidelity: How true to the first telling it is (1 at the source).
    about: Whom the rumour is about, if anyone ('' if no one).
    kind: The game's kind of news ('' if none).
  """

  rumor_id: str
  hearer: str
  teller: str
  text: str
  fidelity: float
  about: str = ''
  kind: str = ''


@dataclasses.dataclass
class _Knower:
  text: str
  fidelity: float
  heard: int
  teller: str


@dataclasses.dataclass
class _Rumor:
  id: str
  text: str
  about: str
  kind: str
  started: int
  last_told: int
  knowers: dict[str, _Knower]


def _stable(name: str) -> int:
  return zlib.crc32(name.encode('utf-8'))


class Rumors(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """The rumours going round, and who heard which version. Thread-safe."""

  def __init__(
      self,
      seed: int = 0,
      chance: float = CHANCE,
      tell_for: int = TELL_FOR,
      keep: float = KEEP,
      garble: float = GARBLE,
      forget_after: int = FORGET_AFTER,
      max_rumors: int = MAX_RUMORS,
  ):
    """Initializes the rumours, with none going round.

    Args:
      seed: Seeds every draw.
      chance: The chance a teller passes a rumour to a contact in a step.
      tell_for: For how many steps after hearing one tells a rumour.
      keep: The share of the teller's fidelity a retelling keeps.
      garble: The chance a retelling drops a word.
      forget_after: A rumour untold for this many steps is forgotten.
      max_rumors: The most rumours kept (the oldest are forgotten first).

    Raises:
      ValueError: If a chance or `keep` is outside [0, 1], or a count below
        1.
    """
    super().__init__()
    for label, value in (
        ('chance', chance),
        ('keep', keep),
        ('garble', garble),
    ):
      if not 0.0 <= value <= 1.0:
        raise ValueError(f'{label} must be in [0, 1], not {value}.')
    for label, value in (
        ('tell_for', tell_for),
        ('forget_after', forget_after),
        ('max_rumors', max_rumors),
    ):
      if value < 1:
        raise ValueError(f'{label} must be at least 1, not {value}.')
    self._seed = int(seed)
    self._chance = float(chance)
    self._tell_for = int(tell_for)
    self._keep = float(keep)
    self._garble = float(garble)
    self._forget_after = int(forget_after)
    self._max_rumors = int(max_rumors)
    self._lock = threading.RLock()
    self._rumors: dict[str, _Rumor] = {}
    self._next = 1

  def start(
      self,
      text: str,
      knowers: Iterable[str],
      step: int,
      about: str = '',
      kind: str = '',
  ) -> str:
    """Starts a rumour among those who saw what it tells.

    Args:
      text: What it says.
      knowers: Who knows it first (they tell it from the next step).
      step: The step it starts.
      about: Whom it is about, if anyone (a game may weigh such news, say
        for reputation).
      kind: The game's kind of news.

    Returns:
      The rumour's id.
    """
    with self._lock:
      rumor_id = f'r{self._next}'
      self._next += 1
      self._rumors[rumor_id] = _Rumor(
          id=rumor_id,
          text=str(text),
          about=str(about),
          kind=str(kind),
          started=int(step),
          last_told=int(step),
          knowers={
              name: _Knower(str(text), 1.0, int(step), '')
              for name in sorted(set(knowers))
          },
      )
      while len(self._rumors) > self._max_rumors:
        oldest = min(self._rumors.values(), key=lambda r: (r.started, r.id))
        del self._rumors[oldest.id]
      return rumor_id

  def spread(
      self, contacts: Mapping[str, Sequence[str]], step: int
  ) -> list[Heard]:
    """A step of telling: rumours pass from their recent hearers to contacts.

    Those who hear a rumour this step tell it from the next. Rumours untold
    for `forget_after` steps are then forgotten.

    Args:
      contacts: Whom each person may tell (others' lists are not read: a
        tie both ways needs both entries).
      step: The step.

    Returns:
      Who heard what this step, from whom, in a replayable order.
    """
    with self._lock:
      heard: list[Heard] = []
      for rumor in sorted(self._rumors.values(), key=lambda r: _order(r.id)):
        tellers = sorted(
            name
            for name, knower in rumor.knowers.items()
            if step - self._tell_for <= knower.heard < step
        )
        for teller in tellers:
          told = rumor.knowers[teller]
          for hearer in sorted(set(contacts.get(teller, ()))):
            if hearer in rumor.knowers or hearer == teller:
              continue
            rng = np.random.default_rng([
                self._seed,
                _order(rumor.id),
                step,
                _stable(teller),
                _stable(hearer),
            ])
            if rng.random() >= self._chance:
              continue
            text = told.text
            if rng.random() < self._garble:
              text = _drop_a_word(text, rng)
            knower = _Knower(text, told.fidelity * self._keep, step, teller)
            rumor.knowers[hearer] = knower
            rumor.last_told = step
            heard.append(
                Heard(
                    rumor_id=rumor.id,
                    hearer=hearer,
                    teller=teller,
                    text=text,
                    fidelity=knower.fidelity,
                    about=rumor.about,
                    kind=rumor.kind,
                )
            )
      for rumor_id in [
          r.id
          for r in self._rumors.values()
          if step - r.last_told >= self._forget_after
      ]:
        del self._rumors[rumor_id]
      self._logging_channel({
          'Key': 'Rumours',
          'Value': {'going round': len(self._rumors), 'heard': len(heard)},
      })
      return heard

  def forget(self, name: str) -> None:
    """Forgets someone as a knower of every rumour (one who has left)."""
    with self._lock:
      for rumor in self._rumors.values():
        rumor.knowers.pop(name, None)

  def known_by(self, name: str) -> list[dict[str, Any]]:
    """The rumours someone has heard, the latest first.

    Args:
      name: Who.

    Returns:
      Each as {id, text (its version), fidelity, heard, teller, about,
      kind}.
    """
    with self._lock:
      found = [
          {
              'id': rumor.id,
              'text': knower.text,
              'fidelity': knower.fidelity,
              'heard': knower.heard,
              'teller': knower.teller,
              'about': rumor.about,
              'kind': rumor.kind,
          }
          for rumor in self._rumors.values()
          for knower in [rumor.knowers.get(name)]
          if knower is not None
      ]
      return sorted(found, key=lambda r: (-r['heard'], _order(r['id'])))

  def rumors(self) -> list[dict[str, Any]]:
    """The rumours going round, the latest first.

    Returns:
      Each as {id, text (as first told), about, kind, started, last_told,
      reach (how many know it), least (the lowest fidelity it reached
      with), furthest (the version heard with it)}.
    """
    with self._lock:
      furthest = {
          rumor.id: min(
              rumor.knowers.values(),
              key=lambda k: (k.fidelity, -k.heard),
              default=_Knower(rumor.text, 1.0, rumor.started, ''),
          )
          for rumor in self._rumors.values()
      }
      return [
          {
              'id': rumor.id,
              'text': rumor.text,
              'about': rumor.about,
              'kind': rumor.kind,
              'started': rumor.started,
              'last_told': rumor.last_told,
              'reach': len(rumor.knowers),
              'least': furthest[rumor.id].fidelity,
              'furthest': furthest[rumor.id].text,
          }
          for rumor in sorted(
              self._rumors.values(),
              key=lambda r: (-r.started, -_order(r.id)),
          )
      ]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'next': self._next,
          'rumors': [
              {
                  'id': rumor.id,
                  'text': rumor.text,
                  'about': rumor.about,
                  'kind': rumor.kind,
                  'started': rumor.started,
                  'last_told': rumor.last_told,
                  'knowers': {
                      name: [k.text, k.fidelity, k.heard, k.teller]
                      for name, k in sorted(rumor.knowers.items())
                  },
              }
              for rumor in sorted(
                  self._rumors.values(), key=lambda r: _order(r.id)
              )
          ],
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('rumors')
    rumors: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    saved_next = state.get('next')
    with self._lock:
      self._next = int(saved_next) if isinstance(saved_next, int) else 1
      self._rumors = {}
      for record in rumors:
        knowers = record.get('knowers', {})
        rumor = _Rumor(
            id=str(record['id']),
            text=str(record.get('text', '')),
            about=str(record.get('about', '')),
            kind=str(record.get('kind', '')),
            started=int(record.get('started', 0)),
            last_told=int(record.get('last_told', 0)),
            knowers={
                str(name): _Knower(
                    str(text), float(fidelity), int(heard), str(teller)
                )
                for name, (text, fidelity, heard, teller) in knowers.items()
            },
        )
        self._rumors[rumor.id] = rumor


def _order(rumor_id: str) -> int:
  """A rumour's number, for ordering ('r12' -> 12)."""
  return int(rumor_id.lstrip('r') or 0)


def _drop_a_word(text: str, rng: np.random.Generator) -> str:
  """The text with one word dropped (the first is kept), if it is long."""
  words = text.split()
  if len(words) <= _FEWEST_WORDS:
    return text
  del words[int(rng.integers(1, len(words)))]
  return ' '.join(words)
