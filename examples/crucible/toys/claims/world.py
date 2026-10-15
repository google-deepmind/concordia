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

"""The claims toy's world: a board of claims and its argument physics.

Claims are sentences. Whether two claims agree, conflict or are unrelated is
a judgement about meaning, which a language model makes once per pair
(`toy.py`). Everything after that is deterministic: each standing claim's
weight is its base, plus a share of the claims that agree with it, minus a
share of the claims it conflicts with,

    w(i) = clamp(base(i) + gain * sum agree w - drag * sum conflict w, 0, cap),

iterated (damped) to a fixed point. A claim whose weight ends below
`fall_below` falls, permanently, and the weights settle again without it: one
beat of a cascade per round of falls. New claims start below the line, so
they stand only while something standing supports them, and cutting a claim
that others lean on can bring them down with it.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

AGREE = 'agree'
CONFLICT = 'conflict'
RESTATES = 'restates'
UNRELATED = 'unrelated'


@dataclasses.dataclass(frozen=True)
class Physics:
  """The numbers of the argument physics (data/claims.yaml)."""

  gain: float = 0.15
  drag: float = 0.15
  fall_below: float = 0.35
  cap: float = 2.0
  iterations: int = 200
  damping: float = 0.5

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Physics':
    return cls(**{k: data[k] for k in data})


@dataclasses.dataclass
class Claim:
  """One claim on the board."""

  key: str
  text: str
  author: str  # An actor's name, or '' for the board's own claims.
  base: float = 1.0
  weight: float = 1.0
  standing: bool = True
  thesis: bool = False


@dataclasses.dataclass
class Board:
  """All claims, the judged relations between them, and the ledger."""

  claims: list[Claim] = dataclasses.field(default_factory=list)
  # Symmetric relations between claim keys: 'a|b' (a < b) -> AGREE/CONFLICT.
  edges: dict[str, str] = dataclasses.field(default_factory=dict)
  # Every judgement the model has made, by text pair, so a preview and the
  # move it previews (and a resumed game) never ask twice.
  judged: dict[str, str] = dataclasses.field(default_factory=dict)
  ink: dict[str, int] = dataclasses.field(default_factory=dict)
  moves: dict[str, int] = dataclasses.field(default_factory=dict)
  history: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  def claim(self, key: str) -> Claim:
    return next(c for c in self.claims if c.key == key)

  def standing(self) -> list[Claim]:
    return [c for c in self.claims if c.standing]

  def thesis_of(self, author: str) -> Claim | None:
    return next(
        (c for c in self.claims if c.thesis and c.author == author), None
    )

  def next_key(self) -> str:
    return f'c{len(self.claims) + 1}'

  def relation(self, a: str, b: str) -> str:
    return self.edges.get(edge_key(a, b), UNRELATED)

  def side_weight(self, author: str) -> float:
    return sum(c.weight for c in self.standing() if c.author == author)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Board':
    return cls(
        claims=[Claim(**c) for c in data['claims']],
        edges=dict(data['edges']),
        judged=dict(data['judged']),
        ink={k: int(v) for k, v in data['ink'].items()},
        moves={k: int(v) for k, v in data['moves'].items()},
        history=[dict(h) for h in data['history']],
    )


def edge_key(a: str, b: str) -> str:
  return '|'.join(sorted((a, b)))


def judged_key(a: str, b: str) -> str:
  return f'{a}␟{b}'


def settle(board: Board, physics: Physics) -> None:
  """Iterates the standing claims' weights to their fixed point."""
  standing = board.standing()
  agree: dict[str, list[str]] = {c.key: [] for c in standing}
  conflict: dict[str, list[str]] = {c.key: [] for c in standing}
  for pair, kind in board.edges.items():
    a, b = pair.split('|')
    if a not in agree or b not in agree:
      continue
    target = agree if kind == AGREE else conflict
    target[a].append(b)
    target[b].append(a)
  weights = {c.key: c.weight for c in standing}
  for _ in range(physics.iterations):
    new = {}
    for c in standing:
      target = (
          c.base
          + physics.gain * sum(weights[k] for k in agree[c.key])
          - physics.drag * sum(weights[k] for k in conflict[c.key])
      )
      target = min(physics.cap, max(0.0, target))
      new[c.key] = (
          physics.damping * weights[c.key] + (1 - physics.damping) * target
      )
    delta = max((abs(new[k] - weights[k]) for k in new), default=0.0)
    weights = new
    if delta < 1e-9:
      break
  for c in standing:
    c.weight = round(weights[c.key], 6)


def cascade(board: Board, physics: Physics) -> list[list[str]]:
  """Settles the board and lets claims fall until none is below the line.

  Args:
    board: The board, mutated.
    physics: The physics.

  Returns:
    The beats of the cascade: the keys that fell together in each round.
  """
  beats = []
  while True:
    settle(board, physics)
    falling = [c for c in board.standing() if c.weight < physics.fall_below]
    if not falling:
      return beats
    for c in falling:
      c.standing = False
      c.weight = 0.0
    beats.append([c.key for c in falling])


def add_claim(
    board: Board,
    text: str,
    author: str,
    relations: Mapping[str, str],
    base: float = 1.0,
    thesis: bool = False,
) -> Claim:
  """Adds a claim and its judged relations to the standing claims."""
  claim = Claim(
      key=board.next_key(), text=text, author=author, base=base, thesis=thesis
  )
  board.claims.append(claim)
  for other, kind in relations.items():
    if kind in (AGREE, CONFLICT):
      board.edges[edge_key(claim.key, other)] = kind
  return claim


def word_count(text: str) -> int:
  return len(text.split())


def summarize_beats(board: Board, beats: Sequence[Sequence[str]]) -> list[str]:
  """One line per beat of a cascade."""
  lines = []
  for i, beat in enumerate(beats):
    names = ', '.join(f'{k} "{board.claim(k).text}"' for k in beat)
    lines.append(f'{"Falls" if i == 0 else "Then falls"}: {names}.')
  return lines
