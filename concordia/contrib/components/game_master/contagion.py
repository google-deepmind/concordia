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

"""Contagion: a sickness that passes between participants who meet.

`Contagion` follows one disease through a population, by contact. Each
participant is susceptible (the default), exposed (`incubation` steps
before it can pass the sickness on), infectious (for `infectious_for`
steps), or immune (for `immunity_for` steps, then susceptible again; 0:
never immune). Each step, every infectious participant passes the sickness
to each susceptible contact the game gives with chance `transmission`,
unless the game says that contact is protected this step (it kept apart,
say). When an illness ends, the participant dies with chance `mortality`
times a vulnerability the game gives (by age, say), else recovers.

Several diseases are several components, under several keys. Draws are
seeded by the step and the names, so a run replays. Only the sick and the
immune are recorded, so a world that runs indefinitely stays bounded.

It is state and rules only, for the GM's other components to call by key
(`DEFAULT_CONTAGION_COMPONENT_KEY`). It adds nothing to prompts: the game
tells who fell sick, recovered or died, and decides what sickness costs.

  sickness = contagion.Contagion(seed=1, transmission=0.3)
  sickness.infect(['Ada'], step=0)
  report = sickness.step({'Ada': ['Bo', 'Cy']}, step=1)
  report.exposed  # e.g. {'Bo': 'Ada'}: Bo caught it from Ada.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
import dataclasses
import threading
from typing import Any, override
import zlib

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np

# The key a game master registers its contagion under, by convention.
DEFAULT_CONTAGION_COMPONENT_KEY = '__contagion__'

SUSCEPTIBLE = 'susceptible'
EXPOSED = 'exposed'
INFECTIOUS = 'infectious'
IMMUNE = 'immune'

# The chance an infectious participant passes the sickness to a contact in
# a step; the steps from catching it to passing it on; the steps one passes
# it on; the steps one stays immune after (0: never); the chance an illness
# ends in death (before vulnerability).
TRANSMISSION = 0.2
INCUBATION = 0
INFECTIOUS_FOR = 1
IMMUNITY_FOR = 10
MORTALITY = 0.1


@dataclasses.dataclass(frozen=True)
class Outbreak:
  """What a step of contagion did.

  Attributes:
    exposed: Who caught it this step, and from whom.
    sick: Who became infectious this step.
    recovered: Whose illness ended and who lived.
    died: Whose illness ended in death.
  """

  exposed: Mapping[str, str]
  sick: Sequence[str]
  recovered: Sequence[str]
  died: Sequence[str]


def _stable(name: str) -> int:
  return zlib.crc32(name.encode('utf-8'))


class Contagion(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """One disease, passing between participants by contact. Thread-safe."""

  def __init__(
      self,
      seed: int = 0,
      transmission: float = TRANSMISSION,
      incubation: int = INCUBATION,
      infectious_for: int = INFECTIOUS_FOR,
      immunity_for: int = IMMUNITY_FOR,
      mortality: float = MORTALITY,
  ):
    """Initializes the contagion, with nobody sick.

    Args:
      seed: Seeds every draw.
      transmission: The chance an infectious participant passes the
        sickness to a contact in a step.
      incubation: The steps from catching it to passing it on.
      infectious_for: The steps one passes it on.
      immunity_for: The steps one stays immune after recovering (0: never).
      mortality: The chance an illness ends in death, before vulnerability.

    Raises:
      ValueError: If a chance is outside [0, 1], `infectious_for` is below
        1, or a count is negative.
    """
    super().__init__()
    for label, value in (
        ('transmission', transmission),
        ('mortality', mortality),
    ):
      if not 0.0 <= value <= 1.0:
        raise ValueError(f'{label} must be in [0, 1], not {value}.')
    if infectious_for < 1:
      raise ValueError(f'infectious_for must be at least 1: {infectious_for}.')
    for label, value in (
        ('incubation', incubation),
        ('immunity_for', immunity_for),
    ):
      if value < 0:
        raise ValueError(f'{label} must not be negative, not {value}.')
    self._seed = int(seed)
    self._transmission = float(transmission)
    self._incubation = int(incubation)
    self._infectious_for = int(infectious_for)
    self._immunity_for = int(immunity_for)
    self._mortality = float(mortality)
    self._lock = threading.RLock()
    # name -> [state, steps left in it].
    self._cases: dict[str, list[Any]] = {}

  def infect(self, names: Iterable[str], step: int) -> list[str]:
    """Some catch the sickness from outside (only the susceptible).

    Args:
      names: Who.
      step: When (for the record; they pass it on from the next step).

    Returns:
      Those who caught it.
    """
    del step
    with self._lock:
      caught = [n for n in sorted(set(names)) if n not in self._cases]
      for name in caught:
        self._catch(name)
      return caught

  def step(
      self,
      contacts: Mapping[str, Sequence[str]],
      step: int,
      vulnerability: Mapping[str, float] | None = None,
      protected: Collection[str] = (),
  ) -> Outbreak:
    """A step of the disease: it passes on, ripens, and ends.

    First the infectious pass it to their contacts (the newly caught are
    exposed, or infectious from the next step with no incubation). Then each
    illness moves on a step: the exposed whose incubation is over become
    infectious, and the infectious whose illness is over die or recover;
    immunity wanes.

    Args:
      contacts: Whom each participant meets this step.
      step: The step.
      vulnerability: Each participant's factor on `mortality` (1 if not
        given).
      protected: Who cannot catch it this step.

    Returns:
      What the step did.
    """
    vulnerability = vulnerability or {}
    with self._lock:
      exposed: dict[str, str] = {}
      spreaders = sorted(
          n for n, (state, _) in self._cases.items() if state == INFECTIOUS
      )
      for source in spreaders:
        for other in sorted(set(contacts.get(source, ()))):
          if other in self._cases or other in exposed or other in protected:
            continue
          rng = np.random.default_rng(
              [self._seed, step, _stable(source), _stable(other)]
          )
          if rng.random() < self._transmission:
            exposed[other] = source
      sick: list[str] = []
      recovered: list[str] = []
      died: list[str] = []
      for name in sorted(self._cases):
        state, left = self._cases[name]
        left -= 1
        if left > 0:
          self._cases[name] = [state, left]
        elif state == EXPOSED:
          self._cases[name] = [INFECTIOUS, self._infectious_for]
          sick.append(name)
        elif state == INFECTIOUS:
          chance = self._mortality * float(vulnerability.get(name, 1.0))
          rng = np.random.default_rng([self._seed, step, _stable(name), 1])
          if rng.random() < chance:
            died.append(name)
            del self._cases[name]
          else:
            recovered.append(name)
            if self._immunity_for:
              self._cases[name] = [IMMUNE, self._immunity_for]
            else:
              del self._cases[name]
        else:  # Immunity wanes.
          del self._cases[name]
      for name in sorted(exposed):
        if self._catch(name):
          sick.append(name)
      self._logging_channel({
          'Key': 'Contagion',
          'Value': {
              'caught': len(exposed),
              'died': len(died),
              'recovered': len(recovered),
          },
      })
      return Outbreak(
          exposed=exposed, sick=sick, recovered=recovered, died=died
      )

  def status(self, name: str) -> tuple[str, int]:
    """A participant's state, and the steps left in it (0 if susceptible)."""
    with self._lock:
      state, left = self._cases.get(name, [SUSCEPTIBLE, 0])
      return str(state), int(left)

  def counts(self) -> dict[str, int]:
    """How many are exposed, infectious and immune."""
    with self._lock:
      found = {EXPOSED: 0, INFECTIOUS: 0, IMMUNE: 0}
      for state, _ in self._cases.values():
        found[state] += 1
      return found

  def infectious(self) -> list[str]:
    """Who can pass it on now, by name."""
    with self._lock:
      return sorted(
          n for n, (state, _) in self._cases.items() if state == INFECTIOUS
      )

  def forget(self, name: str) -> None:
    """Forgets a participant (one who has left)."""
    with self._lock:
      self._cases.pop(name, None)

  def _catch(self, name: str) -> bool:
    """Starts an illness (holds lock); True if it is infectious at once."""
    if self._incubation:
      self._cases[name] = [EXPOSED, self._incubation]
      return False
    self._cases[name] = [INFECTIOUS, self._infectious_for]
    return True

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'cases': [
              [name, state, left]
              for name, (state, left) in sorted(self._cases.items())
          ]
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = state.get('cases')
    cases: Sequence[Any] = saved if isinstance(saved, (list, tuple)) else ()
    with self._lock:
      self._cases = {
          str(name): [str(case_state), int(left)]
          for name, case_state, left in cases
          if case_state in (EXPOSED, INFECTIOUS, IMMUNE)
      }
