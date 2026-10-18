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

"""Instruments: how entities perceive topics.

Every instrument is run by the World GM (its `Deliveries` component), which
hands the entity's stock `TopicSubscriber` only what arrives. Readings are
re-rendered every tick with their age by the contrib `operators.with_age` (a
`combine_latest` with the `world.tick` stream), so an entity can tell a
fresh reading from a stale one: staleness is visible.

The Steward's instruments also run through a contrib `operators.transport`
whose properties come from its current instrument tier (`data/
instruments.yaml`): readings arrive late, some are lost, and temperatures
carry noise; news crosses the light delay and the couriers' roads the same
way. What each instrument delivered is kept on the `GameContext`, so the
report can compare belief with truth (the epistemic gap). An entity's own
state (its storehouse, its genome, what its minds weigh) reaches it as
sheets the World GM publishes (`sheet`, `brief`).
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield

from collections.abc import Callable, Sequence
import dataclasses
import json
import zlib

from concordia.components.agent import pubsub
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import topics
from examples.deep_time import attention as attention_lib
from examples.deep_time import forecast as forecast_lib
from examples.deep_time import game as game_lib
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import publishing
import reactivex
from reactivex import operators as ops

AGE_TEMPLATE = '{text} [tick {time}, {age}]'


def tick_stream(bus: topics.TopicBus) -> reactivex.Observable:
  """The current tick, from the `world.tick` topic."""
  return bus.observable('world.tick').pipe(ops.map(lambda m: m.time))


def with_earlier_readings(
    planet: planet_lib.Planet,
    render: Callable[[topics.Message], str],
    count: int = 3,
) -> Callable[[topics.Message], str]:
  """A temperature renderer with the readings acted on before, and a forecast.

  The earlier readings are what the Steward believed on the last `count`
  ticks (the planet's history keeps them, so they survive a checkpoint):
  late and noisy as they were, together they show which way the planet is
  heading, which one reading cannot. The forecast reckons them forward with
  the moves the Steward made since.
  """

  def out(message: topics.Message) -> str:
    text = render(message)
    earlier = [
        r['believed_temp_c']
        for r in planet.history
        if 'believed_temp_c' in r and r['tick'] < planet.tick
    ][-count:]
    if earlier:
      text += (
          ' Earlier readings you acted on: '
          + ', '.join(f'{v:.1f}' for v in earlier)
          + ' C (oldest first).'
      )
    try:
      value = json.loads(message.payload).get('temp_c')
    except (TypeError, ValueError, AttributeError):
      value = None
    if isinstance(value, (int, float)) and message.time is not None:
      text += forecast_lib.words(
          forecast_lib.of_planet(planet, (int(message.time), float(value)))
      )
    return text

  return out


def publish(
    context: game_lib.GameContext,
    owner: str,
    kind: str,
    function: Callable[[], str],
) -> None:
  """The World GM publishes an entity's sheet each step (a role's rules).

  What an entity knows of its own state (its storehouse, its genome) and what
  its slow minds weigh (the offer awaiting it, the almanac): the World GM
  computes `function` each step and publishes it on the entity's own topic
  (`sheets.<owner>.<kind>`, which no entity hears as observations).

  Args:
    context: Shared game state.
    owner: The entity.
    kind: The sheet ('storehouse', 'genome', 'offer', ...).
    function: The sheet's text, computed by the GM ('' for nothing).
  """
  context.add_sheet(sheet_topic(owner, kind), function)


def read(
    context: game_lib.GameContext, owner: str, kind: str, *, label: str
) -> pubsub.Subscriber:
  """An entity's stock `Subscriber` to its own sheet (never the world)."""
  return pubsub.Subscriber(
      initial_value='',
      pre_act_label=label,
      subject=context.bus.values(sheet_topic(owner, kind)),
  )


def reader(
    context: game_lib.GameContext, owner: str, kind: str
) -> Callable[[], str]:
  """Reads the latest sheet the World GM published for a mind ('' before)."""
  topic = sheet_topic(owner, kind)

  def latest() -> str:
    message = context.bus.latest(topic)
    return '' if message is None else str(message.payload)

  return latest


def sheet_topic(owner: str, kind: str) -> str:
  """The topic of an entity's own sheet."""
  slug = ''.join(c if c.isalnum() else '_' for c in owner.lower()).strip('_')
  return f'sheets.{slug}.{kind}'


@dataclasses.dataclass(frozen=True)
class Instrument:
  """An entity's instrument: what it reads of some topics, and how.

  A spec: the World GM runs it in its `Deliveries` (`deliver`, from the
  role's rules: its transport, ageing and rendering, which may read the
  world), and the entity's stock `TopicSubscriber` holds what arrives
  (`subscribe`, among its components), and nothing else.

  Attributes:
    owner: The entity.
    label: The instrument's label (unique for the entity).
    topics: The topics it reads.
    initial: Belief before the first reading arrives.
    pre_act_label: The component's label.
    build: Builds the instrument's stream from the bus (for the GM).
    tapped: Whether what reaches the owner is kept for the page.
  """

  owner: str
  label: str
  topics: tuple[str, ...]
  initial: str
  pre_act_label: str
  build: Callable[[], reactivex.Observable]
  tapped: bool = True

  @property
  def key(self) -> str:
    return f'{self.owner}/{self.label}'


def deliver(context: game_lib.GameContext, instrument: Instrument) -> None:
  """The World GM runs an instrument (a role's rules install it)."""
  if instrument.tapped:
    context.tapped.add((instrument.owner, instrument.label))
  for name in instrument.topics:
    context.bus.register_subscriber(name, instrument.owner)
  context.deliveries.deliver(instrument.key, instrument.build)


def subscribe(
    context: game_lib.GameContext, instrument: Instrument
) -> topics.TopicSubscriber:
  """An entity's stock `TopicSubscriber` to what its instrument delivers."""
  return topics.TopicSubscriber(
      publisher_component_key=None,
      initial_value=instrument.initial,
      pre_act_label=instrument.pre_act_label,
      subject=context.deliveries.channel(instrument.key),
  )


def instrument(
    context: game_lib.GameContext,
    topic: str,
    *,
    owner: str,
    label: str,
    initial: str,
    render: Callable[[topics.Message], str] | None = None,
    transported: bool = False,
    attention: attention_lib.Attention | None = None,
    channel: Callable[[], operators.Transport] | None = None,
) -> Instrument:
  """An instrument on one topic whose readings show their age.

  Args:
    context: Shared game state.
    topic: Topic to read.
    owner: Entity that holds the belief.
    label: Component label.
    initial: Belief before the first reading arrives.
    render: Message to text (defaults to the topic's renderer).
    transported: Carry readings over the Steward's current instrument tier
      (late, lossy, noisy) instead of delivering them exactly.
    attention: The owner's attention. While this instrument (named `label`) is
      not watched, it receives nothing and shows one short line.
    channel: The transport to carry readings over (with `transported`); defaults
      to the Steward's current instrument tier. A light-delay link between
      worlds is a channel with latency.

  Returns:
    Its spec (`deliver` runs it; `subscribe` holds what arrives).
  """
  to_text = render or publishing.renderer_for(topic)

  def build() -> reactivex.Observable:
    """The instrument, run by the World GM (`components.deliveries`)."""

    def shown(message: topics.Message) -> str:
      if attention is None or attention.watching(label):
        return to_text(message)
      return 'Not watched; this reading is out of date'

    aged = operators.with_age(
        tick_stream(context.bus), render=shown, template=AGE_TEMPLATE
    )

    # What reached the owner, as data, for the page: only while it is
    # watched, as the prompt shows it (a reading still on its way when the
    # owner looked away is not read).
    def note(message: topics.Message) -> None:
      if attention is None or attention.watching(label):
        context.received[(owner, label)] = message

    aged = reactivex.compose(ops.do_action(note), aged)
    if transported:
      seed = zlib.crc32(f'{context.seed}:{owner}:{topic}'.encode())
      aged = reactivex.compose(
          operators.transport(
              channel or (lambda: context.transport_for(topic)),
              seed=seed,
              scheduler=context.wire,
          ),
          ops.do_action(lambda m: context.note_belief(owner, m)),
          aged,
      )
    if attention is not None:
      aged = reactivex.compose(
          ops.filter(lambda _: attention.watching(label)), aged
      )
    return context.bus.observable(topic).pipe(aged)

  return Instrument(
      owner=owner,
      label=label,
      topics=(topic,),
      initial=initial,
      pre_act_label=f'{label} ({topic})',
      build=build,
  )


def local_news(
    context: game_lib.GameContext,
    *,
    owner: str,
    polity: str,
    label: str = 'Local news',
    size: int = 4,
) -> Instrument:
  """News of places near a people's home, later the farther away.

  The people subscribes to the local topics of the regions within reach of
  its home; a contrib transport delays each message by its distance (by
  courier, or faster once the people has wire news) and keeps late ones, so
  the digest is in order of arrival, as news really comes.

  Args:
    context: Shared game state (its planet must have a civilization).
    owner: The people's entity name.
    polity: The people's key.
    label: The component's label.
    size: How many items the digest keeps.

  Returns:
    Its spec: the couriers, run by the World GM.
  """
  planet = context.planet
  assert planet.civ is not None
  config = planet.news_config
  home = planet.civ.polities[polity].home
  reach = int(config['reach_regions'])
  near = [
      r
      for r in range(planet.grid.size)
      if planet.region_steps(home, r) <= reach
  ]

  def away(message: topics.Message) -> int:
    try:
      return planet.region_steps(
          home, int(json.loads(message.payload)['region'])
      )
    except (ValueError, KeyError, TypeError):
      return 0

  def by_distance(message: topics.Message) -> operators.Transport:
    civ = planet.civ
    wired = civ is not None and civ.unlocked(polity, 'wire_news')
    per = float(
        config['wired_ticks_per_region' if wired else 'ticks_per_region']
    )
    return operators.Transport(latency=away(message) * per, keep_late=True)

  def render(message: topics.Message) -> str:
    try:
      text = json.loads(message.payload)['text']
    except (ValueError, KeyError, TypeError):
      text = message.payload
    steps = away(message)
    where = 'here' if steps == 0 else f'{steps} region{"s" * (steps > 1)} away'
    return f'tick {message.time:g}: {text} ({where})'

  names = [f'local.r{r}' for r in near]

  def build() -> reactivex.Observable:
    """The couriers, run by the World GM (`components.deliveries`)."""
    return reactivex.merge(*(context.bus.observable(n) for n in names)).pipe(
        operators.transport(
            by_distance,
            per_message=True,
            seed=zlib.crc32(f'{context.seed}:{owner}:local'.encode()),
            scheduler=context.wire,
        ),
        operators.rolling_digest(size, render=render),
    )

  return Instrument(
      owner=owner,
      label=label,
      topics=tuple(names),
      initial='No news of nearby lands yet.',
      pre_act_label=label,
      build=build,
      tapped=False,
  )


def news_digest(
    context: game_lib.GameContext,
    topic_names: Sequence[str],
    *,
    owner: str,
    label: str,
    initial: str,
    size: int = 4,
) -> pubsub.Subscriber:
  """The last few messages on several topics, newest last."""
  return context.bus.make_subscriber(
      list(topic_names),
      subscriber_name=owner,
      initial_value=initial,
      pre_act_label=label,
      transform=operators.rolling_digest(
          size, render=lambda m: f'tick {m.time:g}: {m.payload}'
      ),
  )
