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

The stock `generic.Simulation` runs the stock `Asynchronous` engine. The born
join the game and the dead leave it through the World GM's stock
`PlayerRoster`; the engine follows. This harness only runs the simulation and
reads its state: the GM's `Lifecycle` tells it of each year closed, each
person who died (for the archive) and the customs nobody holds any more.
"""

import collections
from collections.abc import Callable, Iterator, Mapping
import contextlib
import dataclasses
import json
import logging
import os
import sys
import threading
from typing import Any, TextIO

from absl import logging as absl_logging
from concordia.environment.engines import asynchronous
from examples.lifeways import mock_language_model
from examples.lifeways import scenarios
from examples.lifeways.components import culture
from examples.lifeways.components import game_master as gm_components
from examples.lifeways.prefabs import experimenter as experimenter_lib
from examples.lifeways.prefabs import person as person_lib
from examples.lifeways.prefabs import world_gm
from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import world as world_lib
from concordia.contrib import language_models
from concordia.language_model import language_model
from concordia.language_model import profiled_language_model
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import profiler

# An engine iteration bound no run reaches: the World GM ends a run.
UNBOUNDED_STEPS = 10**12


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
    settings: Mapping[str, Any],
    lifecycle: gm_components.Lifecycle,
    experimenter: Callable[..., Any] | None = None,
) -> prefab_lib.Config:
  """The Concordia config of a world: its founders, the Experimenter, the GM.

  Args:
    settings: The world's settings.
    lifecycle: How the World GM reaches the simulation (wired by the caller).
    experimenter: Optional. Builds the Experimenter's acting component (a
      person's `HumanActComponent`); by default a rule plays it.

  Returns:
    The config.
  """
  founders = scenarios.founders(settings)
  experimenter_params: dict[str, Any] = {'plan': dict(settings.get('plan', {}))}
  if experimenter is not None:
    experimenter_params['act_component_factory'] = experimenter
  instances = [
      prefab_lib.InstanceConfig(
          prefab=person_lib.PERSON_PREFAB,
          role=prefab_lib.Role.ENTITY,
          params=dict(founder['params']),
      )
      for founder in founders
  ]
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=experimenter_lib.EXPERIMENTER_PREFAB,
          role=prefab_lib.Role.ENTITY,
          params=experimenter_params,
      )
  )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab=world_gm.WORLD_GM_PREFAB,
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': world_gm.WORLD_GM_NAME,
              'settings': dict(settings),
              'founders': founders,
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


class _Quiet:
  """Stdout to the null device while any quiet run plays, process-wide.

  `sys.stdout` is the process's: two runs playing at once share one
  redirect, made by the first to start and undone by the last to end.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._runs = 0
    self._stdout: TextIO | None = None
    self._sink: TextIO | None = None

  @contextlib.contextmanager
  def __call__(self) -> Iterator[None]:
    with self._lock:
      if self._runs == 0:
        self._stdout = sys.stdout
        self._sink = open(os.devnull, 'w', encoding='utf-8')
        sys.stdout = self._sink
      self._runs += 1
    try:
      yield
    finally:
      with self._lock:
        self._runs -= 1
        if self._runs == 0 and self._sink is not None:
          sys.stdout = self._stdout
          self._sink.close()
          self._sink = None


_quiet = _Quiet()

# The model people think with by default: the deterministic mock.
MOCK = 'mock'


def make_model(
    spec: str = MOCK,
) -> tuple[
    language_model.LanguageModel | None, profiler.ProfilerContext | None
]:
  """The model people think with, and what counts its calls.

  Args:
    spec: `mock`, or `<api>:<model>` for live minds through the contrib
      model setup (e.g. `ollama:llama3.2:3b`). Together's key is read from
      the environment (`TOGETHER_API_KEY`), never from an argument.

  Returns:
    The model (None: the deterministic mock) and its call counter.

  Raises:
    ValueError: If the spec names Together and no key is in the
      environment.
  """
  if spec == MOCK:
    return None, None
  api, _, name = spec.partition(':')
  key = None
  if api == 'together_ai':
    key = os.environ.get('TOGETHER_API_KEY') or os.environ.get(
        'TOGETHER_AI_API_KEY'
    )
    if not key:
      raise ValueError('No Together key in the environment.')
  model = language_models.language_model_setup(
      api_type=api, model_name=name, api_key=key
  )
  counts = profiler.ProfilerContext()
  counts.enable()
  return (
      profiled_language_model.ProfiledLanguageModel(
          model, name, profiler_instance=counts
      ),
      counts,
  )


def quiet() -> contextlib.AbstractContextManager[None]:
  """Sends stdout to the null device for as long as the context lasts.

  A run is quiet while it plays. A process that must stay quiet until its
  engines' last turns have ended too (a page server, whose closed worlds'
  loops finish after `play` returns) holds this for its whole life.

  Returns:
    The context.
  """
  return _quiet()


class _BornUnconfigured(logging.Filter):
  """Drops the stock checkpoint's warning for a person the roster keeps."""

  def filter(self, record: logging.LogRecord) -> bool:
    return record.msg != 'Prefab config not found for entity %s'


_BORN_UNCONFIGURED = _BornUnconfigured()


@dataclasses.dataclass
class Run:
  """A built simulation and what the harness keeps beside it."""

  sim: simulation_lib.Simulation
  lifecycle: gm_components.Lifecycle
  archive: Archive
  # The stock engine the simulation runs on (its `pause` / `play`).
  engine: asynchronous.Asynchronous

  def checkpoint(self) -> dict[str, Any]:
    """The stock checkpoint (`Simulation.make_checkpoint_data`).

    The stock checkpoint keeps the founders who live; the people born since
    are kept in the World GM's `PlayerRoster` state. The warning the stock
    checkpoint logs for each of the born (it has no prefab config of theirs)
    is dropped.

    Returns:
      The checkpoint data.
    """
    logger = absl_logging.get_absl_logger()
    logger.addFilter(_BORN_UNCONFIGURED)
    try:
      return self.sim.make_checkpoint_data()
    finally:
      logger.removeFilter(_BORN_UNCONFIGURED)

  def world_state(self) -> Mapping[str, Any]:
    """A copy of the world's state, taken under the World GM's lock.

    Safe from any thread at any time (a page polling a running world).

    Returns:
      `World.get_state()`.
    """
    return self._world_component().get_state()

  def world(self) -> world_lib.World:
    """The World GM's live world, for reading.

    Read it only where the GM's lock is held (in a `Lifecycle` callback) or
    when the run is not playing; elsewhere use `world_state`.

    Returns:
      The world.
    """
    return self._world_component().world

  def world_lock(self) -> threading.RLock:
    """The World GM's lock: holding it, no year closes and no move lands.

    Returns:
      The lock (reentrant).
    """
    return self._world_component().lock

  def _world_component(self) -> gm_components.WorldComponent:
    return self.sim.game_masters[0].get_component(  # pyrefly: ignore[missing-attribute]
        gm_components.DEFAULT_WORLD_KEY, type_=gm_components.WorldComponent
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

  def census_row(self, year: int) -> dict[str, Any]:
    """The year's census row, with what the living hold and the peoples.

    Call it as a year closes (`Lifecycle.year_closed`), where the world may
    be read.

    Args:
      year: The year just begun (`year_closed`'s).

    Returns:
      The World's census of the year that closed (or just its year, if the
      world keeps none), `held` (`customs_census`) and `peoples`.
    """
    world = self.world()
    return {
        **(world.history[-1] if world.history else {'year': year - 1}),
        'held': self.customs_census(),
        'peoples': self.peoples(),
    }

  def held_counts(self) -> collections.Counter[str]:
    """How many of the living hold each custom, read from their Learners.

    Measurement only, as `customs_census`; call it where `world()` may be
    read.

    Returns:
      Each custom's id, and how many living people hold it.
    """
    living = set(self.world().people)
    held: collections.Counter[str] = collections.Counter()
    for entity in list(self.sim.entities):
      if entity.name in living:
        learner = entity.get_component(  # pyrefly: ignore[missing-attribute]
            person_lib.LEARNER, type_=culture.Learner
        )
        held.update(custom['id'] for custom in learner.get_state()['held'])
    return held

  def peoples(self) -> dict[str, Any]:
    """The peoples: clusters of the grown whose marriage rules allow each other.

    Measurement only: the harness reads each living person's marriage rule
    (its customs) and the world's public facts (speech, descent). Call it
    where `world()` may be read.

    Returns:
      How many peoples there are, and their sizes, largest first.
    """
    world = self.world()
    grown = [
        name
        for name in world.living()
        if world.people[name].age(world.year) >= world_lib.ADULT_AGE
    ]
    rules: dict[str, Mapping[str, Any] | None] = {name: None for name in grown}
    for entity in list(self.sim.entities):
      if entity.name in rules:
        rules[entity.name] = next(
            (
                c.spec.get('params', {})
                for c in entity.get_all_context_components().values()  # pyrefly: ignore[missing-attribute]
                if isinstance(c, culture.Custom)
                and c.kind == culture_lib.MARRIAGE
            ),
            None,
        )
    root = {name: name for name in grown}

    def find(name: str) -> str:
      while root[name] != name:
        name = root[name]
      return name

    for i, a in enumerate(grown):
      for b in grown[i + 1 :]:
        heard = world.understanding(a, b)
        same = world.people[a].descent == world.people[b].descent
        if culture_lib.accepts(rules[a], heard, same) and culture_lib.accepts(
            rules[b], heard, same
        ):
          root[find(a)] = find(b)
    sizes = collections.Counter(find(name) for name in grown)
    return {
        'peoples': len(sizes),
        'sizes': sorted(sizes.values(), reverse=True),
    }

  def play(
      self, log_retention: int = 200, quiet: bool = True, **kwargs
  ) -> None:
    """Runs until the World GM ends the run (its `max_years`, or a stop).

    Args:
      log_retention: Raw log entries kept (`Simulation.play`).
      quiet: Send what the engine prints (every observation and action) to
        the null device; a run that never ends would print without end. The
      redirect is the process's: it lasts while any quiet run plays.
      **kwargs: More `Simulation.play` arguments.
    """
    with contextlib.ExitStack() as stack:
      if quiet:
        stack.enter_context(_quiet())
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
    experimenter: Callable[..., Any] | None = None,
) -> Run:
  """A runnable Lifeways simulation.

  Args:
    settings: The world's settings (`scenarios.settings`).
    model: The model people act with; the deterministic mock by default.
    sleep_time: Seconds a thread waits when its person may not act yet.
    archive: Where the dead are kept (a new one by default).
    experimenter: Optional. Builds the Experimenter's acting component from
      its context order (a person's `HumanActComponent`).

  Returns:
    The simulation, its lifecycle and its archive.
  """
  lifecycle = gm_components.Lifecycle()
  engine = asynchronous.Asynchronous(sleep_time=sleep_time)
  sim = simulation_lib.Simulation(
      config=build_config(settings, lifecycle, experimenter),
      model=model or mock_language_model.MockModel(),
      embedder=person_lib.hash_embedder,
      engine=engine,
  )
  archive = archive or Archive()

  def departed(person: entity_lib.Entity) -> None:
    state = (
        person.get_state()
        if isinstance(person, entity_component.EntityWithComponents)
        else {}
    )
    archive.add({
        'name': person.name,
        'state': json.loads(json.dumps(state, default=str)),
    })

  lifecycle.departed = departed
  return Run(sim=sim, lifecycle=lifecycle, archive=archive, engine=engine)


def load_checkpoint(run: Run, checkpoint: Mapping[str, Any]) -> None:
  """Restores a saved year into a new simulation of the same settings.

  The stock load restores the founders who still live and the World GM; the
  GM's `PlayerRoster` takes away the founders who died and rebuilds the people
  born since.

  Args:
    run: A new run of the same settings (`build_simulation`).
    checkpoint: `Simulation.make_checkpoint_data()` of the saved year.
  """
  run.sim.load_from_checkpoint(dict(checkpoint))
