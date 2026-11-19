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

"""Reputation: what each participant thinks of the others, quality by quality.

`Reputation` keeps, for each observer, a view of each actor it has seen or
heard of: a score per quality the game names (say `generous` or `honest`),
above zero for good report and below for bad. A deed seen first-hand moves
each witness's view by its weight (`witness`); word of it heard second-hand
moves the hearer's view by a smaller weight the game chooses (`hear`), for
example a rumour's fidelity times `hearsay`. Every step each view fades by
`decay`, and one that fades below `floor` is forgotten. Each observer keeps
views of at most `max_known` actors (the strongest), so a world that runs
indefinitely stays bounded.

An actor's public standing in a quality (`standing`) is the mean of the
views of those who hold one; an observer's own view is `view`.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_REPUTATION_COMPONENT_KEY`). It adds nothing to prompts: the game
decides what people are told of each other's good name, and what it sways.

  reputation = reputation_lib.Reputation(decay=0.1)
  reputation.witness('Ada', 'generous', 1.0, witnesses=['Bo', 'Cy'])
  reputation.hear('Di', 'Ada', 'generous', 0.5)  # Word of it, second-hand.
  reputation.standing('Ada', 'generous')  # (1.0 + 1.0 + 0.5) / 3
"""

from collections.abc import Iterable, Sequence
import threading
from typing import Any, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# The key a game master registers its reputation under, by convention.
DEFAULT_REPUTATION_COMPONENT_KEY = '__reputation__'

# The share of a view that fades each step; the strength below which a view
# is forgotten; the most actors an observer holds views of; the weight of
# word heard second-hand, against a deed seen.
DECAY = 0.1
FLOOR = 0.05
MAX_KNOWN = 30
HEARSAY = 0.5


class Reputation(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """What each observer thinks of each actor, by quality. Thread-safe."""

  def __init__(
      self,
      decay: float = DECAY,
      floor: float = FLOOR,
      max_known: int = MAX_KNOWN,
  ):
    """Initializes the reputation, with no views held.

    Args:
      decay: The share of each view that fades each step.
      floor: A view weaker than this (either way) is forgotten.
      max_known: The most actors an observer holds views of.

    Raises:
      ValueError: If `decay` is not in [0, 1), `floor` is negative, or
        `max_known` is below 1.
    """
    super().__init__()
    if not 0.0 <= decay < 1.0:
      raise ValueError(f'decay must be in [0, 1), not {decay}.')
    if floor < 0.0:
      raise ValueError(f'floor must not be negative, not {floor}.')
    if max_known < 1:
      raise ValueError(f'max_known must be at least 1, not {max_known}.')
    self._decay = float(decay)
    self._floor = float(floor)
    self._max_known = int(max_known)
    self._lock = threading.RLock()
    # observer -> actor -> quality -> score.
    self._views: dict[str, dict[str, dict[str, float]]] = {}

  def witness(
      self,
      actor: str,
      quality: str,
      weight: float,
      witnesses: Iterable[str],
  ) -> None:
    """A deed seen first-hand: each witness's view of the actor moves.

    Args:
      actor: Who did it.
      quality: What it shows (the game's word, e.g. 'generous').
      weight: How much, above zero for good report and below for bad.
      witnesses: Who saw it (the actor itself is skipped).
    """
    with self._lock:
      for observer in sorted(set(witnesses)):
        self._move(observer, actor, quality, weight)

  def hear(
      self, listener: str, actor: str, quality: str, weight: float
  ) -> None:
    """Word of a deed, heard second-hand: the listener's view moves.

    Args:
      listener: Who heard it.
      actor: Whom it was about.
      quality: What it shows.
      weight: How much (the game weighs hearsay, e.g. `HEARSAY` times how
        sure the word was).
    """
    with self._lock:
      self._move(listener, actor, quality, weight)

  def step(self) -> int:
    """A step passes: every view fades, and the faintest are forgotten.

    Returns:
      How many views (observer, actor, quality) are held after.
    """
    with self._lock:
      held = 0
      for observer in sorted(self._views):
        known = self._views[observer]
        for actor in sorted(known):
          qualities = {
              quality: score * (1.0 - self._decay)
              for quality, score in known[actor].items()
              if abs(score * (1.0 - self._decay)) >= self._floor
          }
          if qualities:
            known[actor] = qualities
            held += len(qualities)
          else:
            del known[actor]
        if not known:
          del self._views[observer]
      self._logging_channel({'Key': 'Reputation', 'Value': {'views': held}})
      return held

  def forget(self, name: str) -> None:
    """Forgets someone, as observer and as actor (one who has left)."""
    with self._lock:
      self._views.pop(name, None)
      for observer in list(self._views):
        self._views[observer].pop(name, None)
        if not self._views[observer]:
          del self._views[observer]

  def view(self, observer: str, actor: str, quality: str) -> float:
    """What one thinks of another in a quality (0 if no view)."""
    with self._lock:
      return self._views.get(observer, {}).get(actor, {}).get(quality, 0.0)

  def standing(self, actor: str, quality: str) -> float:
    """An actor's public standing in a quality: the mean view held of it.

    Args:
      actor: Whose standing.
      quality: In what.

    Returns:
      The mean of the views of those who hold one (0 if nobody does).
    """
    with self._lock:
      views = self._views_of(actor, quality)
      return sum(views) / len(views) if views else 0.0

  def known_by(self, actor: str, quality: str) -> int:
    """How many hold a view of an actor in a quality."""
    with self._lock:
      return len(self._views_of(actor, quality))

  def known_for(self, actor: str) -> list[tuple[str, float, int]]:
    """What an actor is known for, the strongest standing first.

    Args:
      actor: Whose name.

    Returns:
      (quality, standing, how many hold a view) for each quality anyone
      holds a view of.
    """
    with self._lock:
      qualities = sorted({
          quality
          for known in self._views.values()
          for quality in known.get(actor, {})
      })
      found = [
          (q, self.standing(actor, q), self.known_by(actor, q))
          for q in qualities
      ]
      return sorted(found, key=lambda f: (-abs(round(f[1], 9)), f[0]))

  def ranked(
      self, actors: Iterable[str], quality: str, observer: str | None = None
  ) -> list[str]:
    """Actors in order of report in a quality, the best first.

    Args:
      actors: Whom to rank.
      quality: In what.
      observer: Ranked by this observer's own views if given, else by
        public standing.

    Returns:
      The actors, the best report first (ties by name).
    """
    with self._lock:
      score = (
          (lambda a: self.view(observer, a, quality))
          if observer is not None
          else (lambda a: self.standing(a, quality))
      )
      return sorted(set(actors), key=lambda a: (-round(score(a), 9), a))

  def _views_of(self, actor: str, quality: str) -> list[float]:
    return [
        known[actor][quality]
        for known in self._views.values()
        if quality in known.get(actor, {})
    ]

  def _move(
      self, observer: str, actor: str, quality: str, weight: float
  ) -> None:
    """Moves one view (holds lock); keeps the observer's strongest views."""
    if observer == actor or not weight:
      return
    known = self._views.setdefault(observer, {})
    qualities = known.setdefault(actor, {})
    qualities[quality] = qualities.get(quality, 0.0) + float(weight)
    if len(known) > self._max_known:
      ranked = sorted(
          known,
          key=lambda a: (max(abs(s) for s in known[a].values()), a),
          reverse=True,
      )
      for dropped in ranked[self._max_known :]:
        del known[dropped]

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'views': [
              [observer, actor, quality, score]
              for observer, known in sorted(self._views.items())
              for actor, qualities in sorted(known.items())
              for quality, score in sorted(qualities.items())
          ]
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('views')
    views: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    restored: dict[str, dict[str, dict[str, float]]] = {}
    for record in views:
      try:
        observer, actor, quality, score = record
        held = float(score)
      except (TypeError, ValueError):
        continue  # A malformed record is skipped.
      restored.setdefault(str(observer), {}).setdefault(str(actor), {})[
          str(quality)
      ] = held
    with self._lock:
      self._views = restored


def hearsay_weight(
    weight: float, fidelity: float, hearsay: float = HEARSAY
) -> float:
  """The weight of word heard: a deed's weight, discounted.

  Args:
    weight: What the deed weighs when seen.
    fidelity: How sure the word is (e.g. a rumour's fidelity, 1 at source).
    hearsay: The discount for hearing rather than seeing.

  Returns:
    `weight * fidelity * hearsay`.
  """
  return float(weight) * float(fidelity) * float(hearsay)
