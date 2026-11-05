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

"""Feuds: grievances between groups, raids, and peace.

`Feuds` keeps the grievances groups hold against one another (the groups
are the game's: peoples, clans, households, or single participants). An
injury adds to the injured group's grievance against the injurer
(`grieve`). Every step each grievance fades by `decay`, and one below
`floor` is forgotten. A group whose grievance against another is at least
`threshold` is hostile to it (`hostile`): a feud.

A raid (`raid`) is a contest the game stages between attackers and
defenders, each with a strength the game reckons (numbers, arms, walls):
the attackers win with chance `attack / (attack + defence)`, drawn from a
seed, the raid and the step. Whatever the outcome, the defenders' grievance
against the attackers grows by `injury` (more if they lost), so feuds feed
themselves. Peace (`make_peace`) clears the grievances both ways and holds
for `truce` steps (`at_peace`), during which grievances between the two do
not grow.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_FEUDS_COMPONENT_KEY`). It adds nothing to prompts: the game decides
what a raid takes and who it harms, and who may make peace.

  feuds = feuds_lib.Feuds(seed=0, threshold=1.0)
  feuds.grieve('Hill people', 'River people', 1.5, step=3, why='a field')
  feuds.hostile('Hill people', 'River people')  # True.
  outcome = feuds.raid('Hill people', 'River people', 3.0, 2.0, step=4)
"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any, override
import zlib

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# The key a game master registers its feuds under, by convention.
DEFAULT_FEUDS_COMPONENT_KEY = '__feuds__'

# The share of a grievance that fades each step; below which it is
# forgotten; at which it makes a feud; what a raid adds to the defenders'
# grievance (twice that if they lost); for how many steps peace holds.
DECAY = 0.1
FLOOR = 0.05
THRESHOLD = 1.0
INJURY = 0.5
TRUCE = 20


@dataclasses.dataclass(frozen=True)
class Raid:
  """A raid's outcome.

  Attributes:
    attackers: The raiding group.
    defenders: The raided group.
    won: Whether the attackers won.
    chance: The chance they had.
    step: When.
  """

  attackers: str
  defenders: str
  won: bool
  chance: float
  step: int


class Feuds(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Grievances between groups, raids and truces. Thread-safe."""

  def __init__(
      self,
      seed: int = 0,
      decay: float = DECAY,
      floor: float = FLOOR,
      threshold: float = THRESHOLD,
      injury: float = INJURY,
      truce: int = TRUCE,
  ):
    """Initializes the feuds, with no grievance held.

    Args:
      seed: Seeds the raids' draws.
      decay: The share of a grievance that fades each step.
      floor: A grievance below this is forgotten.
      threshold: A grievance at least this makes a feud.
      injury: What a raid adds to the defenders' grievance (twice if they
        lost).
      truce: For how many steps peace holds.

    Raises:
      ValueError: If `decay` is outside [0, 1), or another setting is
        negative (or `threshold` not above zero).
    """
    super().__init__()
    if not 0.0 <= decay < 1.0:
      raise ValueError(f'decay must be in [0, 1), not {decay}.')
    if threshold <= 0.0:
      raise ValueError(f'threshold must be above 0, not {threshold}.')
    for label, value in (
        ('floor', floor),
        ('injury', injury),
        ('truce', truce),
    ):
      if value < 0:
        raise ValueError(f'{label} must not be negative, not {value}.')
    self._seed = int(seed)
    self._decay = float(decay)
    self._floor = float(floor)
    self._threshold = float(threshold)
    self._injury = float(injury)
    self._truce = int(truce)
    self._lock = threading.RLock()
    # (by, against) -> grievance.
    self._grievances: dict[tuple[str, str], float] = {}
    # sorted (a, b) -> the step peace ends.
    self._truces: dict[tuple[str, str], int] = {}
    self._raids = 0

  def grieve(
      self, by: str, against: str, amount: float, step: int, why: str = ''
  ) -> float:
    """An injury: the injured group's grievance against the injurer grows.

    Args:
      by: The injured group.
      against: The injurer.
      amount: How much.
      step: When (no grievance grows between groups at peace).
      why: What was done (logged).

    Returns:
      The grievance after.
    """
    if by == against or amount <= 0.0:
      return self.grievance(by, against)
    with self._lock:
      if self._at_peace(by, against, step):
        return self.grievance(by, against)
      key = (by, against)
      self._grievances[key] = self._grievances.get(key, 0.0) + float(amount)
      self._logging_channel({
          'Key': 'Grievance',
          'Value': {'by': by, 'against': against, 'why': why},
      })
      return self._grievances[key]

  def raid(
      self,
      attackers: str,
      defenders: str,
      attack: float,
      defence: float,
      step: int,
  ) -> Raid:
    """A raid: the attackers win with chance attack / (attack + defence).

    The defenders' grievance against the attackers grows by `injury`, twice
    that if they lost. A raid ends any truce between the two.

    Args:
      attackers: The raiding group.
      defenders: The raided group.
      attack: The attackers' strength (at least 0).
      defence: The defenders' strength (at least 0).
      step: When.

    Returns:
      The outcome.
    """
    attack, defence = max(0.0, float(attack)), max(0.0, float(defence))
    chance = attack / (attack + defence) if attack + defence > 0 else 0.0
    with self._lock:
      self._raids += 1
      rng = np.random.default_rng([
          self._seed,
          step,
          zlib.crc32(attackers.encode('utf-8')),
          zlib.crc32(defenders.encode('utf-8')),
          self._raids,
      ])
      won = bool(rng.random() < chance)
      self._truces.pop(_pair(attackers, defenders), None)  # Peace is broken.
      key = (defenders, attackers)
      self._grievances[key] = self._grievances.get(key, 0.0) + self._injury * (
          2.0 if won else 1.0
      )
      return Raid(attackers, defenders, won, chance, step)

  def make_peace(self, a: str, b: str, step: int) -> None:
    """Peace: grievances both ways are cleared, and a truce holds."""
    with self._lock:
      self._grievances.pop((a, b), None)
      self._grievances.pop((b, a), None)
      self._truces[_pair(a, b)] = int(step) + self._truce

  def step(self, step: int) -> None:
    """A step passes: grievances fade, the faintest are forgotten, and
    truces that ran out end."""
    with self._lock:
      for key in sorted(self._grievances):
        held = self._grievances[key] * (1.0 - self._decay)
        if held < self._floor:
          del self._grievances[key]
        else:
          self._grievances[key] = held
      for key in [k for k, ends in self._truces.items() if ends <= step]:
        del self._truces[key]

  def grievance(self, by: str, against: str) -> float:
    """How much a group grieves against another (0 if not at all)."""
    with self._lock:
      return self._grievances.get((by, against), 0.0)

  def hostile(self, by: str, against: str) -> bool:
    """Whether a group feuds with another (its grievance at the threshold)."""
    return self.grievance(by, against) >= self._threshold

  def at_peace(self, a: str, b: str, step: int) -> bool:
    """Whether two groups are under a truce at a step."""
    with self._lock:
      return self._at_peace(a, b, step)

  def feuds(self) -> list[dict[str, Any]]:
    """The grievances held: {by, against, grievance, hostile}, the deepest
    first."""
    with self._lock:
      return [
          {
              'by': by,
              'against': against,
              'grievance': held,
              'hostile': held >= self._threshold,
          }
          for (by, against), held in sorted(
              self._grievances.items(), key=lambda kv: (-kv[1], kv[0])
          )
      ]

  def truces(self) -> list[dict[str, Any]]:
    """The truces holding: {between, until}."""
    with self._lock:
      return [
          {'between': list(k), 'until': ends}
          for k, ends in sorted(self._truces.items())
      ]

  def forget(self, group: str) -> None:
    """Forgets a group (one that is gone)."""
    with self._lock:
      self._grievances = {
          k: v for k, v in self._grievances.items() if group not in k
      }
      self._truces = {k: v for k, v in self._truces.items() if group not in k}

  def _at_peace(self, a: str, b: str, step: int) -> bool:
    return self._truces.get(_pair(a, b), step) > step

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'grievances': [
              [by, against, held]
              for (by, against), held in sorted(self._grievances.items())
          ],
          'truces': [
              [a, b, ends] for (a, b), ends in sorted(self._truces.items())
          ],
          'raids': self._raids,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('grievances')
    grievances: Sequence[Any] = (
        saved if isinstance(saved, (list, tuple)) else ()
    )
    saved_truces = state.get('truces')
    truces: Sequence[Any] = (
        saved_truces if isinstance(saved_truces, (list, tuple)) else ()
    )
    saved_raids = state.get('raids')
    with self._lock:
      self._grievances = {
          (str(by), str(against)): float(held)
          for by, against, held in grievances
      }
      self._truces = {_pair(str(a), str(b)): int(ends) for a, b, ends in truces}
      self._raids = int(saved_raids) if isinstance(saved_raids, int) else 0


def _pair(a: str, b: str) -> tuple[str, str]:
  return (a, b) if a <= b else (b, a)
