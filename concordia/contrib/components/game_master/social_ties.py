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

"""Social ties: who is close to whom, as a game master keeps it.

`SocialTies` is a weighted, undirected network between named participants.
Each tie has a strength between 0 and 1 and the kinds of bond that made it
(say `kin`, `work`, `friend`). Whatever brings two people together
strengthens their tie, saturating towards 1; every step, every tie fades,
and a tie that fades below `floor` is forgotten. Each participant keeps at
most `max_ties` ties (the strongest), so a network that runs indefinitely
stays bounded.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_SOCIAL_TIES_COMPONENT_KEY`): the game says who met whom. It adds
nothing to the GM's prompts. Other mechanics read it as a contact structure
(`contacts`): who hears a rumour, who catches a sickness, whose good name
travels.

  ties = social_ties.SocialTies(decay=0.1)
  ties.strengthen('Ada', 'Bo', 0.3, kind='work')
  ties.step()  # A step passes: every tie fades a little.
  ties.ties_of('Ada')  # [('Bo', 0.27)]
"""

from collections.abc import Iterable, Mapping, Sequence
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its social ties under, by convention.
DEFAULT_SOCIAL_TIES_COMPONENT_KEY = '__social_ties__'

# How much of a tie's strength fades each step, below which strength a tie
# is forgotten, and how many ties each participant keeps at most.
DECAY = 0.1
FLOOR = 0.05
MAX_TIES = 20


def _pair(a: str, b: str) -> tuple[str, str]:
  return (a, b) if a <= b else (b, a)


class SocialTies(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A weighted network of ties that strengthen with contact and fade.

  Every public method is thread-safe. Ties are undirected: `strength(a, b)`
  is `strength(b, a)`.
  """

  def __init__(
      self,
      decay: float = DECAY,
      floor: float = FLOOR,
      max_ties: int = MAX_TIES,
  ):
    """Initializes the network, with no ties.

    Args:
      decay: The share of each tie's strength that fades each step.
      floor: A tie weaker than this is forgotten.
      max_ties: The most ties a participant keeps (its strongest).

    Raises:
      ValueError: If `decay` is not in [0, 1), `floor` not in [0, 1), or
        `max_ties` is below 1.
    """
    super().__init__()
    if not 0.0 <= decay < 1.0:
      raise ValueError(f'decay must be in [0, 1), not {decay}.')
    if not 0.0 <= floor < 1.0:
      raise ValueError(f'floor must be in [0, 1), not {floor}.')
    if max_ties < 1:
      raise ValueError(f'max_ties must be at least 1, not {max_ties}.')
    self._decay = float(decay)
    self._floor = float(floor)
    self._max_ties = int(max_ties)
    self._lock = threading.RLock()
    # (a, b) with a <= b: [strength, kinds].
    self._ties: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
    self._of: dict[str, set[str]] = {}

  def strengthen(
      self, a: str, b: str, amount: float, kind: str = 'friend'
  ) -> float:
    """Brings two participants closer.

    The tie's strength `s` becomes `s + amount * (1 - s)`: the closer two
    already are, the less one more meeting adds.

    Args:
      a: One participant.
      b: The other (a tie to oneself is ignored).
      amount: How much of the distance to 1 this closes, in [0, 1].
      kind: The kind of bond (kept with the tie's other kinds).

    Returns:
      The tie's new strength (0 for a tie to oneself).
    """
    if a == b:
      return 0.0
    amount = min(1.0, max(0.0, float(amount)))
    with self._lock:
      key = _pair(a, b)
      strength, kinds = self._ties.get(key, (0.0, frozenset()))
      strength += amount * (1.0 - strength)
      self._ties[key] = (strength, kinds | {kind})
      self._of.setdefault(a, set()).add(b)
      self._of.setdefault(b, set()).add(a)
      for name in (a, b):
        self._keep_strongest(name)
      return self._ties.get(key, (0.0, frozenset()))[0]

  def step(self) -> list[tuple[str, str]]:
    """A step passes: every tie fades, and the faintest are forgotten.

    Returns:
      The ties forgotten, as sorted pairs.
    """
    with self._lock:
      forgotten = []
      for key, (strength, kinds) in sorted(self._ties.items()):
        strength *= 1.0 - self._decay
        if strength < self._floor:
          forgotten.append(key)
          self._drop(*key)
        else:
          self._ties[key] = (strength, kinds)
      self._logging_channel({
          'Key': 'Ties',
          'Value': {'held': len(self._ties), 'forgotten': len(forgotten)},
      })
      return forgotten

  def remove(self, name: str) -> None:
    """Forgets every tie of a participant (one who has left the game)."""
    with self._lock:
      for other in sorted(self._of.get(name, ())):
        self._drop(name, other)
      self._of.pop(name, None)

  def strength(self, a: str, b: str) -> float:
    """How close two participants are (0 if not tied)."""
    with self._lock:
      return self._ties.get(_pair(a, b), (0.0, frozenset()))[0]

  def kinds(self, a: str, b: str) -> list[str]:
    """The kinds of bond between two participants, sorted."""
    with self._lock:
      return sorted(self._ties.get(_pair(a, b), (0.0, frozenset()))[1])

  def ties_of(
      self, name: str, limit: int | None = None, kind: str | None = None
  ) -> list[tuple[str, float]]:
    """A participant's ties, strongest first (then by name).

    Args:
      name: Whose ties.
      limit: The most to return (all if None).
      kind: Only ties with a bond of this kind, if given.

    Returns:
      (other, strength) pairs.
    """
    with self._lock:
      found = sorted(
          (
              (other, self._ties[_pair(name, other)][0])
              for other in self._of.get(name, ())
              if kind is None or kind in self._ties[_pair(name, other)][1]
          ),
          key=lambda tie: (-round(tie[1], 9), tie[0]),
      )
      return found if limit is None else found[: max(0, limit)]

  def contacts(
      self, names: Iterable[str] | None = None, min_strength: float = 0.0
  ) -> dict[str, list[str]]:
    """Who each participant is tied to: a contact structure.

    Args:
      names: Whose contacts (every participant with a tie if None).
      min_strength: Only ties at least this strong.

    Returns:
      Each name's tied others, sorted by name.
    """
    with self._lock:
      chosen = sorted(self._of) if names is None else sorted(set(names))
      return {
          name: sorted(
              other
              for other in self._of.get(name, ())
              if self._ties[_pair(name, other)][0] >= min_strength
          )
          for name in chosen
      }

  def size(self) -> int:
    """How many ties the network holds."""
    with self._lock:
      return len(self._ties)

  def mean_strength(self) -> float:
    """The mean strength of the ties held (0 with none)."""
    with self._lock:
      if not self._ties:
        return 0.0
      return sum(s for s, _ in self._ties.values()) / len(self._ties)

  def _keep_strongest(self, name: str) -> None:
    """Drops a participant's weakest ties beyond `max_ties` (holds lock)."""
    others = self._of.get(name, set())
    if len(others) <= self._max_ties:
      return
    ranked = sorted(
        others, key=lambda o: (self._ties[_pair(name, o)][0], o), reverse=True
    )
    for other in ranked[self._max_ties :]:
      self._drop(name, other)

  def _drop(self, a: str, b: str) -> None:
    """Forgets one tie (holds lock)."""
    self._ties.pop(_pair(a, b), None)
    for x, y in ((a, b), (b, a)):
      if x in self._of:
        self._of[x].discard(y)
        if not self._of[x]:
          del self._of[x]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'ties': [
              [a, b, strength, sorted(kinds)]
              for (a, b), (strength, kinds) in sorted(self._ties.items())
          ]
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('ties')
    ties: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    restored: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
    of: dict[str, set[str]] = {}
    for record in ties:
      try:
        a, b, strength, kinds = record
        key = _pair(str(a), str(b))
        restored[key] = (float(strength), frozenset(map(str, kinds)))
      except (TypeError, ValueError):
        continue  # A malformed record is skipped.
      of.setdefault(key[0], set()).add(key[1])
      of.setdefault(key[1], set()).add(key[0])
    with self._lock:
      self._ties, self._of = restored, of


def strengthen_all(
    ties: SocialTies,
    groups: Mapping[str, Sequence[str]] | Iterable[Sequence[str]],
    amount: float,
    kind: str,
) -> None:
  """Strengthens the tie of every pair within each group.

  Args:
    ties: The network.
    groups: Groups of participants who met (a mapping's values, or an
      iterable of groups), e.g. those who worked one place or joined one rite.
    amount: How much each pair is brought closer.
    kind: The kind of bond.
  """
  members = groups.values() if isinstance(groups, Mapping) else groups
  for group in members:
    names = sorted(set(group))
    for i, a in enumerate(names):
      for b in names[i + 1 :]:
        ties.strengthen(a, b, amount, kind)
