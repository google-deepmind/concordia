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

"""Builds and runs a Lifeways simulation on the stock asynchronous engine.

The stock `generic.Simulation` runs the stock `Asynchronous` engine. The
World GM's `Lifecycle` reaches the simulation for births
(`Simulation.add_entity`) and deaths (an archive of the person, then
`Simulation.remove_entity`); this harness wires it and otherwise only runs the
simulation and reads its state.
"""

import collections
from collections.abc import Callable, Mapping
import contextlib
import dataclasses
import json
import os
from typing import Any
import zlib

from concordia.environment.engines import asynchronous
from examples.lifeways import mock_language_model
from examples.lifeways.components import culture
from examples.lifeways.components import game_master as gm_components
from examples.lifeways.prefabs import experimenter as experimenter_lib
from examples.lifeways.prefabs import person as person_lib
from examples.lifeways.prefabs import world_gm
from examples.lifeways.world import world as world_lib
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import prefab as prefab_lib
import numpy as np

EMBEDDING_SIZE = 16
# An engine iteration bound no run reaches: the World GM ends a run.
UNBOUNDED_STEPS = 10**12


def hash_embedder(text: str) -> np.ndarray:
  """A cheap deterministic sentence embedder (memory needs one)."""
  rng = np.random.default_rng(zlib.crc32(text.encode()))
  return rng.standard_normal(EMBEDDING_SIZE)


@dataclasses.dataclass
class Archive:
  """The dead: each person's final state and life, kept beside the run.

  The most recent `keep` are held in memory; `sink`, if given, receives every
  record (e.g. to append it to a JSONL file).
  """

  keep: int = 100
  records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  sink: Callable[[dict[str, Any]], None] | None = None

  def add(self, record: dict[str, Any]) -> None:
    self.records.append(record)
    del self.records[: max(0, len(self.records) - self.keep)]
    if self.sink is not None:
      self.sink(record)


def build_config(
    settings: Mapping[str, Any], lifecycle: gm_components.Lifecycle
) -> prefab_lib.Config:
  """The Concordia config of a world: its founders, the Experimenter, the GM."""
  founders = world_lib.World(settings).found()
  instances = [
      prefab_lib.InstanceConfig(
          prefab=person_lib.PERSON_PREFAB,
          role=prefab_lib.Role.ENTITY,
          params=params,
      )
      for params in founders
  ]
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=experimenter_lib.EXPERIMENTER_PREFAB,
          role=prefab_lib.Role.ENTITY,
          params={'name': world_lib.EXPERIMENTER},
      )
  )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=world_gm.WORLD_GM_PREFAB,
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': world_gm.WORLD_GM_NAME,
              'settings': dict(settings),
              'lifecycle': lifecycle,
          },
      )
  )
  return prefab_lib.Config(
      prefabs={
          person_lib.PERSON_PREFAB: person_lib.Person(),
          experimenter_lib.EXPERIMENTER_PREFAB: experimenter_lib.Experimenter(),
          world_gm.WORLD_GM_PREFAB: world_gm.WorldGameMaster(),
      },
      instances=instances,
      default_premise='',
      default_max_steps=UNBOUNDED_STEPS,
  )


@dataclasses.dataclass
class Run:
  """A built simulation and what the harness keeps beside it."""

  sim: simulation_lib.Simulation
  lifecycle: gm_components.Lifecycle
  archive: Archive

  def world(self) -> world_lib.World:
    """The World GM's world (for reading: measures, the page)."""
    return (
        self.sim.game_masters[0]
        .get_component(  # pyrefly: ignore[missing-attribute]
            gm_components.DEFAULT_WORLD_KEY,
            type_=gm_components.WorldComponent,
        )
        .world
    )

  def customs_census(self) -> dict[str, Any]:
    """What the living hold, read from their entities' Learners.

    Measurement only: the harness reads each living person's state, which
    the World GM never does.

    Returns:
      How many hold customs, the distinct customs held, the mean held per
      person, Simpson's diversity of what is held, the customs held by
      channel, the norms held by form, the customs held by kind and domain,
      and the five most widely held (id, share of people).
    """
    living = set(self.world().people)
    held: collections.Counter[str] = collections.Counter()
    channels: collections.Counter[str] = collections.Counter()
    forms: collections.Counter[str] = collections.Counter()
    loci: collections.Counter[str] = collections.Counter()
    people = 0
    for entity in list(self.sim.entities):
      if entity.name not in living:
        continue
      learner = entity.get_component(  # pyrefly: ignore[missing-attribute]
          person_lib.LEARNER, type_=culture.Learner
      )
      people += 1
      for custom in learner.get_state()['held']:
        held[custom['id']] += 1
        channels[custom['via']] += 1
        loci[f'{custom["kind"]} {custom["domain"]}'] += 1
        if custom['form']:
          forms[custom['form']] += 1
    total = sum(held.values())
    return {
        'people': people,
        'distinct': len(held),
        'per_person': round(total / people, 2) if people else 0.0,
        'diversity': (
            round(1.0 - sum((n / total) ** 2 for n in held.values()), 3)
            if total
            else 0.0
        ),
        'by_channel': dict(sorted(channels.items())),
        'norms_by_form': dict(sorted(forms.items())),
        'by_kind_and_domain': dict(sorted(loci.items())),
        'top': [
            [custom_id, round(n / people, 2)]
            for custom_id, n in sorted(
                held.items(), key=lambda x: (-x[1], x[0])
            )[:5]
        ],
    }

  def play(
      self, log_retention: int = 200, quiet: bool = True, **kwargs
  ) -> None:
    """Runs until the World GM ends the run (its `max_years`, or a stop).

    Args:
      log_retention: Raw log entries kept (`Simulation.play`).
      quiet: Send what the engine prints (every observation and action) to
        the null device; a run that never ends would print without end.
      **kwargs: More `Simulation.play` arguments.
    """
    with contextlib.ExitStack() as stack:
      if quiet:
        sink = stack.enter_context(open(os.devnull, 'w'))
        stack.enter_context(contextlib.redirect_stdout(sink))
      self.sim.play(
          premise='',
          max_steps=UNBOUNDED_STEPS,
          log_retention=log_retention,
          **kwargs,
      )


def build_simulation(
    settings: Mapping[str, Any],
    model: language_model.LanguageModel | None = None,
    sleep_time: float = 0.002,
    archive: Archive | None = None,
) -> Run:
  """A runnable Lifeways simulation.

  Args:
    settings: The world's settings (`scenarios.settings`).
    model: The model people act with; the deterministic mock by default.
    sleep_time: Seconds a thread waits when its person may not act yet.
    archive: Where the dead are kept (a new one by default).

  Returns:
    The simulation, its lifecycle and its archive.
  """
  lifecycle = gm_components.Lifecycle()
  sim = simulation_lib.Simulation(
      config=build_config(settings, lifecycle),
      model=model or mock_language_model.MockModel(),
      embedder=hash_embedder,
      engine=asynchronous.Asynchronous(sleep_time=sleep_time),
  )
  archive = archive or Archive()

  def spawn(params: Mapping[str, Any]) -> None:
    sim.add_entity(
        prefab_lib.InstanceConfig(
            prefab=person_lib.PERSON_PREFAB,
            role=prefab_lib.Role.ENTITY,
            params=dict(params),
        )
    )

  def retire(name: str) -> None:
    entity = next((e for e in sim.entities if e.name == name), None)
    if entity is None:
      return
    config = sim.get_entity_prefab_config(name)
    archive.add({
        'name': name,
        'params': (
            json.loads(json.dumps(dict(config.params), default=str))
            if config
            else {}
        ),
        'state': json.loads(json.dumps(entity.get_state(), default=str)),  # pyrefly: ignore[missing-attribute]
    })
    sim.remove_entity(name)

  lifecycle.spawn = spawn
  lifecycle.retire = retire
  return Run(sim=sim, lifecycle=lifecycle, archive=archive)


def load_checkpoint(run: Run, checkpoint: Mapping[str, Any]) -> None:
  """Restores a saved year into a new simulation of the same settings.

  People who died since the start leave; people born since join from the
  params their checkpoint keeps; then every entity's and the GM's state is
  restored the standard way.

  Args:
    run: A new run of the same settings (`build_simulation`).
    checkpoint: `Simulation.make_checkpoint_data()` of the saved year.
  """
  sim = run.sim
  saved = checkpoint.get('entities', {})
  for entity in list(sim.entities):
    if entity.name not in saved:
      sim.remove_entity(entity.name)
  present = {entity.name for entity in sim.entities}
  for name, data in saved.items():
    if name not in present:
      sim.add_entity(
          prefab_lib.InstanceConfig(
              prefab=data['prefab_type'],
              role=prefab_lib.Role.ENTITY,
              params=dict(data['entity_params']),
          )
      )
  sim.load_from_checkpoint(dict(checkpoint))
