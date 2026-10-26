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

"""Births and deaths for a simulation whose population changes as it plays.

A game master that decides who is born and who dies (its world's rules) does
not add or remove entities itself: it is given two callables, `spawn` and
`retire`, which this module's `Population` provides for a stock
`generic.Simulation`:

- `spawn(params)` adds an entity of a prefab (`Simulation.add_entity`);
- `retire(name)` keeps a record of the entity (its params and final state, in
  an `Archive`) and removes it (`Simulation.remove_entity`).

With the asynchronous engine, an entity added while the simulation plays gets
an observe-act loop of its own, and one removed ends its loop.

  population = population_lib.Population(simulation, prefab='person')
  # The game master's components are given population.spawn and
  # population.retire; on a birth: population.spawn({'name': 'Ada', ...}).

An indefinite run would keep every dead entity for ever; the `Archive` keeps
only the latest `keep` in memory and hands each record to `sink` (e.g. to
append it to a JSONL file).
"""

from collections.abc import Callable, Mapping
import dataclasses
import json
import threading
from typing import Any

from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Archive:
  """The retired: each entity's params and final state, kept beside the run.

  The latest `keep` records are held in memory; `sink`, if given, receives
  every record.
  """

  keep: int = 100
  records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  sink: Callable[[dict[str, Any]], None] | None = None

  def __post_init__(self):
    self._lock = threading.Lock()

  def add(self, record: dict[str, Any]) -> None:
    """Keeps a record (the oldest go past `keep`) and hands it to the sink.

    Args:
      record: The record.
    """
    with self._lock:
      self.records.append(record)
      del self.records[: max(0, len(self.records) - self.keep)]
    if self.sink is not None:
      self.sink(record)


class Population:
  """Adds and removes a simulation's entities as its game master decides."""

  def __init__(
      self,
      simulation: simulation_lib.Simulation,
      prefab: str,
      archive: Archive | None = None,
  ):
    """Initializes the population.

    Args:
      simulation: The simulation whose entities come and go.
      prefab: The prefab (a key of the simulation config's prefabs) new
        entities are built from.
      archive: Where the retired are recorded (a new one by default).
    """
    self._simulation = simulation
    self._prefab = prefab
    self.archive = archive if archive is not None else Archive()

  def spawn(self, params: Mapping[str, Any]) -> None:
    """Adds an entity of the prefab.

    Args:
      params: The entity's params (its `name` among them).
    """
    self._simulation.add_entity(
        prefab_lib.InstanceConfig(
            prefab=self._prefab,
            role=prefab_lib.Role.ENTITY,
            params=dict(params),
        )
    )

  def retire(self, name: str) -> None:
    """Records an entity and removes it; an unknown name is let be.

    The record holds its name, params and final state, made JSON-safe.

    Args:
      name: The entity's name.
    """
    entity = next(
        (e for e in self._simulation.entities if e.name == name), None
    )
    if entity is None:
      return
    config = self._simulation.get_entity_prefab_config(name)
    state = (
        entity.get_state()
        if isinstance(entity, entity_component.EntityWithComponents)
        else {}
    )
    self.archive.add({
        'name': name,
        'params': _json_safe(dict(config.params) if config else {}),
        'state': _json_safe(state),
    })
    self._simulation.remove_entity(name)


def _json_safe(data: Any) -> Any:
  return json.loads(json.dumps(data, default=str))
