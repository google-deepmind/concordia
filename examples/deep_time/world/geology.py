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

"""The lithosphere: seeded volcanic eruptions.

Eruptions arrive as a Poisson process. Each one throws up bright ash (a
decaying albedo pulse that cools the planet: geology → climate) and mineral
dust that fertilises bare ground for a few ticks (geology → biosphere), and is
announced on `geo.events` (geology → information). The random stream is a
seeded NumPy generator whose state is checkpointed, so runs are reproducible
and resumable.
"""

# pylint: disable=g-complex-comprehension,missing-function-docstring

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class Eruption:
  volcano: str
  magnitude: int
  ash_albedo: float
  fertility: float

  def describe(self) -> str:
    size = {1: 'small', 2: 'large', 3: 'great'}.get(self.magnitude, 'vast')
    return (
        f'A {size} eruption at {self.volcano} (magnitude {self.magnitude})'
        ' veils the sky with ash and dusts the bare ground with minerals.'
    )


class Lithosphere:
  """Generates eruptions from a seeded random stream."""

  def __init__(self, config: Mapping[str, Any], seed: int = 0):
    """Initializes the lithosphere from `data/geology.yaml`."""
    eruptions = config['eruptions']
    self.rate = float(eruptions['rate_per_tick'])
    low, high = eruptions['magnitude']
    self.magnitude_range = (int(low), int(high))
    self.ash_per_magnitude = float(eruptions['ash_albedo_per_magnitude'])
    self.fertility_per_magnitude = float(eruptions['fertility_per_magnitude'])
    self.fertility_decay = float(config['fertility_decay'])
    self.volcanoes = list(config['volcanoes'])
    self.count = 0
    self._rng = np.random.default_rng(seed)

  def forecast(self, ticks: int) -> list[tuple[int, Eruption]]:
    """The eruptions of the next `ticks` ticks, without drawing them.

    The stream is seeded, so the geologists' forecast is exact: a copy of
    the random state is stepped ahead and the real one is left untouched.

    Args:
      ticks: How many ticks ahead to look.

    Returns:
      (ticks from now, eruption) pairs, soonest first.
    """
    # A copy of the generator: the live one is never touched (forecasts run
    # on slow-mind threads while the world steps).
    copy = np.random.default_rng()
    copy.bit_generator.state = dict(self._rng.bit_generator.state)
    return [
        (ahead, eruption)
        for ahead in range(1, ticks + 1)
        for eruption in self._draw(copy)
    ]

  def step(self) -> list[Eruption]:
    """Draws this tick's eruptions."""
    eruptions = self._draw(self._rng)
    self.count += len(eruptions)
    return eruptions

  def _draw(self, rng: np.random.Generator) -> list[Eruption]:
    eruptions = []
    for _ in range(int(rng.poisson(self.rate))):
      magnitude = int(
          rng.integers(self.magnitude_range[0], self.magnitude_range[1] + 1)
      )
      volcano = self.volcanoes[int(rng.integers(len(self.volcanoes)))]
      eruptions.append(
          Eruption(
              volcano=volcano,
              magnitude=magnitude,
              ash_albedo=magnitude * self.ash_per_magnitude,
              fertility=magnitude * self.fertility_per_magnitude,
          )
      )
    return eruptions

  def get_state(self) -> dict[str, Any]:
    return {'count': self.count, 'rng': self._rng.bit_generator.state}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.count = int(state['count'])
    self._rng.bit_generator.state = dict(state['rng'])
