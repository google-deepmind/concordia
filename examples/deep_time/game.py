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

"""Shared game state passed to prefabs through their params.

Concordia prefabs receive `params`; like `examples/resource_dilemma` does with
its `sim_state`, Deep Time passes one `GameContext` to every prefab so that the
World GM and the entities share the planet, the topic bus and the clock.
"""

# pylint: disable=g-long-ternary,missing-function-docstring

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import threading
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import human_act_component
from concordia.components.agent import pubsub
from concordia.components.game_master import make_observation
from concordia.contrib.components.game_master import numeric_time
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import sim_time
from concordia.contrib.pubsub import topics
from examples.deep_time import actors as actors_lib
from examples.deep_time import attention as attention_lib
from examples.deep_time import wakeups as wakeups_lib
from examples.deep_time.world import automation as automation_lib
from examples.deep_time.world import data
from examples.deep_time.world import evolution as evolution_lib
from examples.deep_time.world import geology as geology_lib
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing
from concordia.language_model import language_model
from concordia.typing import human_input
from concordia.typing import prefab as prefab_lib
from concordia.utils import measurements as measurements_lib


@dataclasses.dataclass(frozen=True)
class Mind:
  """A slow mind (a `pubsub.Publisher`), listed for cognition meters."""

  owner: str
  label: str
  topic: str
  latency_ticks: int
  publisher: pubsub.Publisher


@dataclasses.dataclass
class GameContext:
  """Everything the World GM and entity prefabs share."""

  scenario_name: str
  scenario: Mapping[str, Any]
  seed: int
  planet: planet_lib.Planet
  clock: numeric_time.NumericClock
  bus: topics.TopicBus
  observation_queue: make_observation.ObservationQueue
  measurements: measurements_lib.Measurements
  evolution: evolution_lib.Evolution | None = None
  # Deterministic sim-time scheduler for slow minds; None in real-async mode.
  sim: sim_time.SimTimeScheduler | None = None
  actors: dict[str, actors_lib.Actor] = dataclasses.field(default_factory=dict)
  minds: list[Mind] = dataclasses.field(default_factory=list)
  # Topic-name prefixes each entity hears as observations (contrib
  # TopicObservations on the World GM routes them).
  access: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
  # Set by `simulation.build_simulation`: the model for new species' minds
  # and the function that adds an entity to the running simulation.
  slow_model: language_model.LanguageModel | None = None
  spawner: Callable[[prefab_lib.InstanceConfig], None] | None = None
  spawned: list[str] = dataclasses.field(default_factory=list)
  # Ticks for instrument transports: `sim` in sim-time mode, else its own.
  wire: sim_time.SimTimeScheduler = dataclasses.field(
      default_factory=sim_time.SimTimeScheduler
  )
  # Numeric JSON fields per topic that instrument noise perturbs.
  noisy_fields: Mapping[str, Sequence[str]] = dataclasses.field(
      default_factory=dict
  )
  # Latest reading each (owner, topic) instrument delivered, for the
  # epistemic gap.
  beliefs: dict[tuple[str, str], topics.Message] = dataclasses.field(
      default_factory=dict
  )
  # The message each instrument last delivered to its owner, by (owner,
  # label), after any transport's delay, losses and noise: what the page
  # draws a role's readings from. `tapped` lists the
  # instruments that report here, so one that has received nothing yet is
  # not mistaken for one that does not report.
  received: dict[tuple[str, str], topics.Message] = dataclasses.field(
      default_factory=dict
  )
  tapped: set[tuple[str, str]] = dataclasses.field(default_factory=set)
  # What each entity with limited attention is watching.
  attention: dict[str, attention_lib.Attention] = dataclasses.field(
      default_factory=dict
  )
  # Event-driven acting for species, in scenarios with `wakeups`.
  wakeups: wakeups_lib.WakeUps | None = None
  # Peoples' standing orders as reflexes, in scenarios with a market.
  automation: automation_lib.Automation | None = None
  # Entities a person plays, by name, and the reader that waits for them.
  humans: dict[str, human_input.HumanInput] = dataclasses.field(
      default_factory=dict
  )
  _inbox: list[tuple[str, str]] = dataclasses.field(default_factory=list)

  def with_bus(self, components: Mapping[str, Any]) -> dict[str, Any]:
    """An entity's components behind a contrib `TopicBusComponent`.

    A checkpoint is restored the standard Concordia way: entities first, then
    game masters, each component in order. Put first, the bus component puts
    the bus back before any of the entity's beliefs restores its own state,
    so retained messages replayed by the restore cannot disturb them (as the
    contrib async-deliberation entity does).

    Args:
      components: The entity's own components.

    Returns:
      The components, the bus component first.
    """
    return {
        topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY: topics.TopicBusComponent(
            self.bus
        ),
        **components,
    }

  def pushes(
      self, options: Sequence[str], actor: str | None = None
  ) -> dict[str, str]:
    """What each option does to the world, measured.

    The Steward's interventions, the species' and grazers' behaviours and
    the colony's recipes and designs carry a `push` in their data; other
    options carry none. Mission Control's moves land a light delay later.
    An actor that works out its own pushes (a people's labor and learning)
    adds them.

    Args:
      options: A role's options this turn.
      actor: The role's entity name, if known.

    Returns:
      Option -> push, for the options that have one.
    """
    known: dict[str, str] = {}
    for intervention in self.planet.interventions():
      if intervention.push:
        known[intervention.option.lower()] = intervention.push
    for name in ('species.yaml', 'fauna.yaml'):
      for behaviour in data.load(name).get('behaviours', []):
        if behaviour.get('push'):
          known[str(behaviour['label']).lower()] = str(behaviour['push'])
    colony = self.planet.colony
    if colony is not None:
      later = f'in {colony.light_delay} years, if they follow it: '
      for recipe in colony.recipes:
        push = colony.pushes.get(recipe.key)
        if push:
          known[f'put more labor into: {recipe.label}'] = push
          known[f'tell the colonists: put more labor into {recipe.label}'] = (
              later + push
          )
      for key, label in colony.design_labels.items():
        push = colony.pushes.get(f'design:{key}')
        if push:
          known[f'transmit the design for {label}'] = (
              f'in {colony.light_delay} years: {push}'
          )
    found = {o: known[o.lower()] for o in options if o.lower() in known}
    own = getattr(self.actors.get(actor or ''), 'pushes', None)
    if own is not None:
      found.update(own(options))
    return found

  def act_params(self, name: str) -> dict[str, Any]:
    """Prefab params that hand `name`'s actions to its person, if any.

    The upstream `HumanActComponent` replaces the model's act component; it
    is given the same component order, so the person reads exactly the
    context a model playing the role would be prompted with.

    Args:
      name: The entity's name.

    Returns:
      `{'act_component_factory': ...}` for a human-played entity, else `{}`.
    """
    reader = self.humans.get(name)
    if reader is None:
      return {}

    def factory(
        order: Sequence[str],
    ) -> human_act_component.HumanActComponent:
      return human_act_component.HumanActComponent(
          reader, component_order=order
      )

    return {'act_component_factory': factory}

  def attention_slots(self) -> int | None:
    """Instruments the Steward can watch at once (None: all of them)."""
    tier = self.planet.instrument()
    if not self.scenario.get('attention') or tier is None:
      return None
    return int(tier['attention_slots'])

  def transport_for(self, topic: str) -> operators.Transport:
    """The channel the Steward's instruments use now for a topic."""
    tier = self.planet.instrument()
    if tier is None:
      return operators.Transport()
    return operators.Transport(
        latency=float(tier['latency']),
        drop_rate=float(tier['drop_rate']),
        noise_sigma=float(tier['noise_sigma']),
        fields=tuple(self.noisy_fields.get(topic, ())),
        decimals=2,
    )

  def note_belief(self, owner: str, message: topics.Message) -> None:
    self.beliefs[(owner, message.topic)] = message

  def believed(self, owner: str, topic: str, path: str) -> float | None:
    """A numeric field of the reading an owner's instrument last delivered."""
    message = self.beliefs.get((owner, topic))
    if message is None:
      return None
    try:
      value = operators.lookup(json.loads(message.payload), path)
    except (json.JSONDecodeError, KeyError):
      return None
    return float(value) if isinstance(value, (int, float)) else None

  def record_beliefs(self) -> None:
    """Adds what the Steward believes the temperature is to the history.

    Called after publishing, so it is what the Steward will act on this tick:
    `believed_temp_c` and `epistemic_gap_c`, and for
    forecasts the tick the reading shows (`read_tick`) and the noise of the
    instruments it came through (`read_sigma`).
    """
    row = self.planet.history[-1]
    for owner, topic in sorted(self.beliefs):
      if topic != 'climate.global':
        continue
      believed = self.believed(owner, topic, 'temp_c')
      if believed is not None:
        row['believed_temp_c'] = round(believed, 2)
        row['epistemic_gap_c'] = round(abs(believed - row['temp_c']), 2)
        # The moment it shows and its instruments' noise, for forecasts.
        shows = self.beliefs[(owner, topic)].time
        if shows is not None:
          row['read_tick'] = int(shows)
        tier = self.planet.instrument()
        if tier is not None:
          row['read_sigma'] = float(tier['noise_sigma'])
      return

  def advance_wire(self) -> None:
    """Delivers instrument readings due this tick (if not done by `sim`)."""
    if self.wire is not self.sim:
      self.wire.advance()

  _inbox_lock: threading.Lock = dataclasses.field(
      default_factory=threading.Lock
  )

  def add_actor(
      self, actor: actors_lib.Actor, hears: tuple[str, ...] = ()
  ) -> None:
    """Registers an actor and the topics it hears as observations."""
    self.actors[actor.name] = actor
    self.access[actor.name] = tuple(hears)

  def make_mind(
      self,
      owner: str,
      label: str,
      topic: str,
      component: action_spec_ignored.ActionSpecIgnored,
      latency_ticks: int,
      **publisher_kwargs: Any,
  ) -> pubsub.Publisher:
    """Builds a slow mind publishing to a topic and registers it.

    Args:
      owner: Entity that owns the mind.
      label: Short name, e.g. 'Advisor'.
      topic: Topic its completed answers are published on.
      component: The component computed in the background.
      latency_ticks: Sim-time latency (ignored in async mode).
      **publisher_kwargs: Other `pubsub.Publisher` options, e.g.
        `trigger_on_observe=True` for an alarm-driven mind.

    Returns:
      A `pubsub.Publisher` with `skip_if_busy=True`.
    """
    publisher = self.bus.make_publisher(
        topic,
        component=component,
        source=f'{owner} {label.lower()}' if label else owner,
        skip_if_busy=True,
        scheduler=self.publisher_scheduler(
            latency_ticks, key=f'{owner}/{label}'
        ),
        **publisher_kwargs,
    )
    self.minds.append(Mind(owner, label, topic, latency_ticks, publisher))
    return publisher

  def cognition_meters(self) -> list[dict[str, Any]]:
    """Per-mind counters from `Publisher.compute_count` and friends."""
    return [
        {
            'owner': mind.owner,
            'mind': mind.label,
            'topic': mind.topic,
            'latency_ticks': mind.latency_ticks if self.sim else None,
            'computations': mind.publisher.compute_count,
            'last_seconds': round(mind.publisher.last_compute_duration, 4),
            'busy': mind.publisher.is_busy(),
        }
        for mind in self.minds
    ]

  def active_entities(self) -> list[str]:
    return [
        name
        for name, a in self.actors.items()
        if a.is_active()
        and (self.wakeups is None or self.wakeups.is_awake(name))
    ]

  def observe(self, entity_name: str, text: str) -> None:
    """Queues an observation for an entity's next turn."""
    self.observation_queue.add(entity_name, text, ())

  def tick_headline(self) -> str:
    return (
        f'Tick {self.planet.tick} of {self.planet.max_ticks}'
        f' ({self.clock.formatted_now()}, {self.planet.era} era).'
    )

  def publisher_scheduler(self, latency_ticks: int, key: str):
    """Scheduler for a slow mind: sim-time latency, or threads if async."""
    return sim_time.publisher_scheduler(self.sim, latency_ticks, key)

  def on_species_message(self, message: topics.Message) -> None:
    """Collects slow-mind outputs; may be called from any thread."""
    if self.bus.is_restoring:  # A checkpoint's replay, not a new answer.
      return
    parts = message.topic.split('.')
    if (
        len(parts) == 4
        and parts[3] in ('adaptation', 'pact')
        and message.payload
    ):
      with self._inbox_lock:
        self._inbox.append((parts[2], f'{parts[3]}:{message.payload}'))
    elif (
        len(parts) == 3
        and parts[0] == 'civ'
        and parts[2] in ('treaty', 'press')
        and message.payload
    ):
      with self._inbox_lock:
        self._inbox.append((parts[1], f'{parts[2]}:{message.payload}'))

  def deliver_adaptations(self) -> list[planet_lib.WorldEvent]:
    """Hands landed slow-mind answers to the planet, in arrival order.

    Adaptation answers become trait changes; pact and treaty answers settle
    proposals, and the proposer hears the outcome on its diplomacy topic;
    broadsheets go on the Gazette and the author's propaganda topic.

    Returns:
      Events to announce.
    """
    with self._inbox_lock:
      inbox, self._inbox = self._inbox, []
    events = []
    for species, message in inbox:
      kind, answer = message.split(':', 1)
      if kind == 'adaptation':
        events.append(
            planet_lib.WorldEvent(
                planet_lib.BIO_EVENTS,
                self.planet.queue_adaptation(species, answer),
            )
        )
        continue
      if kind == 'press':
        news = self.planet.publish_broadsheet(species, answer)
        if news is not None:
          events.append(planet_lib.WorldEvent(planet_lib.CIV_EVENTS, news))
          self.bus.publish(
              f'civ.{species}.propaganda', answer.strip(), source='world'
          )
        continue
      if kind == 'treaty':
        settled = self.planet.answer_treaty(species, answer)
        topic = planet_lib.CIV_EVENTS
      else:
        settled = self.planet.answer_pact(species, answer)
        topic = planet_lib.BIO_EVENTS
      if settled is None:
        continue
      proposer, text = settled
      events.append(planet_lib.WorldEvent(topic, text))
      self.bus.publish(f'diplomacy.{proposer}.{species}', text, source='world')
    return events

  def publish_state(self, events: Sequence[planet_lib.WorldEvent] = ()) -> None:
    publishing.publish_state(
        self.bus, self.planet, self.clock.formatted_now(), events
    )

  def get_state(self) -> dict[str, Any]:
    """The world's own state, checkpointed by the World GM's `WorldStep`.

    The topic bus, the sim-time scheduler and the clock are components'
    state, checkpointed the standard Concordia way (`TopicBusComponent` and
    `TopicObservations`, `SimTimePublisher` and the transports' operator
    state, `NumericClock`), not here.

    Returns:
      JSON-safe state.
    """
    return {
        'scenario_name': self.scenario_name,
        'seed': self.seed,
        'planet': self.planet.get_state(),
        'observation_queue': self.observation_queue.get_all(),
        'attention': {k: a.get_state() for k, a in self.attention.items()},
        'wakeups': self.wakeups.get_state() if self.wakeups else None,
        'automation': self.automation.get_state() if self.automation else None,
        # What each instrument last delivered (the epistemic gap and the
        # page read these between deliveries).
        'beliefs': [
            [owner, topic, message.to_dict()]
            for (owner, topic), message in sorted(self.beliefs.items())
        ],
        'received': [
            [owner, label, message.to_dict()]
            for (owner, label), message in sorted(self.received.items())
        ],
        # Slow minds' answers that landed but are not yet handed over.
        'inbox': [list(item) for item in self._inbox],
        # Species founded during play (their entities are founded again
        # before a checkpoint is loaded, see `simulation.load_checkpoint`).
        'spawned': list(self.spawned),
        # What each role's last moves did, as the world resolved them (the
        # page shows each move's outcome).
        'logs': {
            name: [list(entry) for entry in actor.log]
            for name, actor in sorted(self.actors.items())
        },
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.planet.set_state(state['planet'])
    self.observation_queue.set_all(state['observation_queue'])
    self.beliefs = {
        (owner, topic): topics.Message.from_dict(message)
        for owner, topic, message in state.get('beliefs', [])
    }
    self.received = {
        (owner, label): topics.Message.from_dict(message)
        for owner, label, message in state.get('received', [])
    }
    with self._inbox_lock:
      self._inbox = [(str(k), str(v)) for k, v in state.get('inbox', [])]
    self.spawned = [str(name) for name in state.get('spawned', [])]
    if self.wakeups is not None and state.get('wakeups'):
      self.wakeups.set_state(state['wakeups'])
    if self.automation is not None and state.get('automation'):
      self.automation.set_state(state['automation'])
    for owner, watched in state.get('attention', {}).items():
      if owner in self.attention:
        self.attention[owner].set_state(watched)
    for name, entries in state.get('logs', {}).items():
      if name in self.actors:
        self.actors[name].log = [
            (int(tick), str(action), str(outcome))
            for tick, action, outcome in entries
        ]


def _merged(base: Mapping[str, Any], overrides: Mapping[str, Any]):
  """`base` with `overrides` merged in, recursively (for per-scenario data)."""
  merged = dict(base)
  for key, value in overrides.items():
    if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
      merged[key] = _merged(merged[key], value)
    else:
      merged[key] = value
  return merged


def new_game(
    scenario_name: str, time_mode: str = 'sim', seed: int = 0
) -> GameContext:
  """Creates the shared state for a scenario at tick 0 and publishes it.

  Args:
    scenario_name: Key in `data/scenarios.yaml`.
    time_mode: 'sim' runs slow minds on a deterministic `SimTimeScheduler` with
      latencies in ticks; 'async' runs them on threads, so wall-clock model
      latency becomes in-game latency.
    seed: Seed of the planet's random processes (eruptions).

  Returns:
    The shared game state.
  """
  if time_mode not in ('sim', 'async'):
    raise ValueError(f'Unknown time mode {time_mode!r}')
  scenario = data.scenario(scenario_name)
  evolution = evolution_lib.Evolution(data.load('species.yaml'))
  physics = data.load('daisyworld.yaml')
  metabolisms = {}
  if 'species' in scenario:
    # A scenario may replace the daisies with its own life (e.g. microbial
    # mats with metabolisms for the Great Oxidation).
    physics['species'] = {
        key: {k: v for k, v in traits.items() if k != 'metabolism'}
        for key, traits in scenario['species'].items()
    }
    metabolisms = {
        key: traits['metabolism']
        for key, traits in scenario['species'].items()
        if 'metabolism' in traits
    }
    evolution.species = [
        evolution_lib.SpeciesInfo(**info) for info in scenario['species_info']
    ]
  instruments = data.load('instruments.yaml')
  interventions = planet_lib.interventions_from_config(
      data.load('interventions.yaml')
  )
  if 'interventions' in scenario:
    allowed = list(scenario['interventions'])
    interventions = sorted(
        (i for i in interventions if i.key in allowed),
        key=lambda i: allowed.index(i.key),
    )
  # A scenario may scale a move's effect to its world (`effects:`), and each
  # move's push is the one measured on this world (push_audit.py), unless
  # the scenario words it itself (under the ice, a haze tips the planet).
  effects = scenario.get('effects') or {}
  measured = (data.load('pushes.yaml') or {}).get(scenario_name, {})
  pushes = {
      **{k: v for k, v in measured.items() if isinstance(v, str)},
      **(scenario.get('pushes') or {}),
  }
  interventions = [
      dataclasses.replace(
          i,
          effect={**i.effect, **effects.get(i.key, {})},
          push=str(pushes.get(i.key, i.push)),
      )
      for i in interventions
  ]
  planet = planet_lib.Planet(
      scenario,
      physics,
      interventions,
      evolution,
      geology_lib.Lithosphere(data.load('geology.yaml'), seed=seed),
      carbon_config=_merged(
          data.load('carbon.yaml'), scenario.get('carbon_overrides', {})
      ),
      seed=seed,
      fauna_config=(
          data.load('fauna.yaml')
          if 'grazers' in scenario.get('fauna', ())
          else None
      ),
      symbiosis_config=data.load('symbiosis.yaml'),
      chemistry_config=(
          data.load('oxygen.yaml') if scenario.get('chemistry') else None
      ),
      metabolisms=metabolisms,
      instrument_tiers=instruments['tiers'],
      colony_config=(
          _merged(data.load('colony.yaml'), scenario['colony'])
          if scenario.get('colony')
          else None
      ),
      civ_config=(
          {
              **_merged(
                  data.load('civ.yaml'), scenario.get('civ_overrides', {})
              ),
              **data.load('techs.yaml'),
              'backlash': (
                  data.load('backlash.yaml')
                  if scenario['civilization'].get('backlash')
                  else None
              ),
              'trade': (
                  data.load('trade.yaml')
                  if scenario['civilization'].get('trade')
                  else None
              ),
              'media': (
                  data.load('media.yaml')
                  if scenario['civilization'].get('media')
                  else None
              ),
          }
          if scenario.get('civilization')
          else None
      ),
  )
  # What each climate move does turn by turn, for the Steward's forecast.
  planet.move_profiles = {
      str(k): [float(x) for x in v]
      for k, v in (measured.get('profiles') or {}).items()
  }
  if planet.colony is not None:  # Its recipes' pushes, measured likewise.
    planet.colony.pushes = {
        **(measured.get('colony') or {}),
        **planet.colony.pushes,
    }
  sim = sim_time.SimTimeScheduler() if time_mode == 'sim' else None
  clock = numeric_time.NumericClock(
      numeric_time.NumericTimeModel(
          start=numeric_time.parse_duration(scenario['start_time'])
      ),
      step=scenario['tick_length'],
  )
  measurements = measurements_lib.Measurements()
  bus = topics.TopicBus(clock=clock.ticks, measurements=measurements)
  publishing.declare_topics(bus, data.load('topics.yaml'))
  context = GameContext(
      scenario_name=scenario_name,
      scenario=scenario,
      seed=seed,
      planet=planet,
      clock=clock,
      bus=bus,
      observation_queue=make_observation.ObservationQueue(),
      measurements=measurements,
      evolution=evolution,
      sim=sim,
      wire=sim or sim_time.SimTimeScheduler(),
      noisy_fields=instruments['noisy_fields'],
      wakeups=(
          wakeups_lib.WakeUps(bus)
          if scenario.get('wakeups')
          or (scenario.get('civilization') or {}).get('wakeups')
          else None
      ),
  )
  if planet.civ is not None and planet.trade is not None:
    context.automation = automation_lib.Automation(
        bus, planet.civ, planet.trade, data.load('trade.yaml')['automation']
    )
  bus.matching('bio.species.').subscribe(context.on_species_message)
  bus.matching('civ.').subscribe(context.on_species_message)
  context.publish_state()
  return context
