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

"""A named-topic bus for ReactiveX publish/subscribe simulations.

`concordia.components.agent.pubsub` gives a single entity a background
`Publisher` and a non-blocking `Subscriber`. Multi-entity simulations also need
to *name* streams, share them between entities, keep the latest value for late
joiners, and see who listens to what. `TopicBus` provides that layer:

* Hierarchical topic names (``climate.region.3``) with per-topic retention:
  latest value, a replay buffer, or live-only.
* Every message is wrapped in a `Message` envelope stamped with a sequence
  number, an optional simulation time from an injected clock and a source.
* `make_subscriber` and `make_publisher` build ordinary `pubsub.Subscriber` and
  `pubsub.Publisher` components bound to topics, so entities perceive the world
  through standard Concordia components.
* Prefix subscriptions (`matching`) use the same prefix semantics as
  `interrupt_scheduling.InterruptMask`, and `as_interrupt_event` converts a
  message into an `interrupt_scheduling.Event`.
* Topology introspection (`topology`, `to_mermaid`, `to_dot`), per-topic
  counters, an optional tap into `measurements.Measurements`, and latest-value
  checkpointing via `get_state` / `set_state`.

Example:

```python
bus = topics.TopicBus(clock=lambda: world.tick)
bus.declare('climate.global', description='Global mean temperature.')
thermometer = bus.make_subscriber(
    'climate.global', subscriber_name='Alice', initial_value='unknown')
bus.publish('climate.global', '{"temp_c": 21.5}', source='climate model')
thermometer.get_current_value()  # -> '{"temp_c": 21.5}'
```
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
import threading
from typing import Any, cast, override
import weakref

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import pubsub
from concordia.components.game_master import interrupt_scheduling
from concordia.contrib.pubsub import sim_time
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
import reactivex
from reactivex import abc as reactivex_abc
from reactivex import operators as ops
from reactivex import subject as reactivex_subject

DEFAULT_MEASUREMENTS_PREFIX = 'topic/'
DEFAULT_TOPIC_BUS_COMPONENT_KEY = '__topic_bus__'


class TopicBusComponent(entity_component.ContextComponent):
  """A component backed by a `TopicBus` (analogous to `AssociativeMemory`)."""

  def __init__(self, bus: 'TopicBus'):
    super().__init__()
    self._bus = bus

  @property
  def bus(self) -> 'TopicBus':
    return self._bus

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'bus': self._bus.get_state()}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    bus_state = state.get('bus')
    if isinstance(bus_state, Mapping) and self._bus.get_state() != bus_state:
      self._bus.set_state(dict(bus_state))


class TopicSubscriber(pubsub.Subscriber):
  """A `pubsub.Subscriber` that checkpoints stateful `contrib` operators."""

  @override
  def get_state(self) -> entity_component.ComponentState:
    state = dict(super().get_state())
    hooks = getattr(self._subject, '_concordia_state_hooks', ())
    if hooks:
      state['operator_state'] = [get_fn() for get_fn, _ in hooks]
    return state

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    hooks = getattr(self._subject, '_concordia_state_hooks', ())
    op_states = state.get('operator_state')
    if (
        hooks
        and isinstance(op_states, Sequence)
        and not isinstance(op_states, str)
    ):
      for (_, set_fn), op_state in zip(hooks, op_states):
        if isinstance(op_state, Mapping):
          set_fn(op_state)
    super().set_state(state)

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'TopicSubscriber':
    del memo
    copied = TopicSubscriber(
        publisher_component_key=self._publisher_component_key,
        initial_value=self._initial_value,
        pre_act_label=self.get_pre_act_label(),
        subject=self._subject,
    )
    with copied._value_lock:  # pylint: disable=protected-access
      copied._current_value = self.get_current_value()  # pylint: disable=protected-access
    return copied


class Retention(enum.Enum):
  """How much history a topic keeps for subscribers that join late."""

  # Late subscribers immediately receive the most recent message, if any.
  LATEST = 'latest'
  # Late subscribers receive up to `replay_size` most recent messages.
  REPLAY = 'replay'
  # Late subscribers receive only messages published after they subscribe.
  NONE = 'none'


@dataclasses.dataclass(frozen=True)
class TopicSpec:
  """Declaration of a topic.

  Attributes:
    name: Hierarchical, dot-separated topic name, e.g. `climate.region.3`.
    retention: How much history late subscribers receive.
    replay_size: Number of messages kept when `retention` is `REPLAY`.
    description: Free-text description used in topology dumps.
  """

  name: str
  retention: Retention = Retention.LATEST
  replay_size: int = 1
  description: str = ''


@dataclasses.dataclass(frozen=True)
class Message:
  """A message published on a topic.

  Attributes:
    topic: Name of the topic the message was published on.
    payload: The message body. Use JSON for structured data.
    seq: Bus-wide monotonically increasing sequence number.
    time: Simulation time from the bus clock when published, or None.
    source: Name of the publishing entity or system, if known.
  """

  topic: str
  payload: str
  seq: int
  time: float | None = None
  source: str = ''

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Message':
    """Builds a message from the output of `to_dict`."""
    time = data.get('time')
    return cls(
        topic=str(data['topic']),
        payload=str(data['payload']),
        seq=int(data['seq']),
        time=float(time) if time is not None else None,
        source=str(data.get('source', '')),
    )


def payload(message: Message) -> str:
  """Returns the payload of a message (the default subscriber renderer)."""
  return message.payload


def as_interrupt_event(message: Message) -> interrupt_scheduling.Event:
  """Converts a message into an `interrupt_scheduling.Event`.

  The topic becomes the event tag, so `InterruptMask` prefixes match topic
  prefixes directly.

  Args:
    message: The message to convert.

  Returns:
    An event stamped with the message time (or its sequence number when the
    bus has no clock).
  """
  return interrupt_scheduling.Event(
      timestamp=message.time if message.time is not None else message.seq,
      tag=message.topic,
      source=message.source or '__external__',
      description=message.payload,
  )


@dataclasses.dataclass
class _Topic:
  spec: TopicSpec
  subject: reactivex_subject.Subject
  history: list[Message]
  count: int = 0
  publishers: list[str] = dataclasses.field(default_factory=list)
  subscribers: list[str] = dataclasses.field(default_factory=list)


def _make_subject(spec: TopicSpec) -> reactivex_subject.Subject:
  if spec.retention == Retention.LATEST:
    return reactivex_subject.ReplaySubject(buffer_size=1)
  if spec.retention == Retention.REPLAY:
    return reactivex_subject.ReplaySubject(buffer_size=max(1, spec.replay_size))
  return reactivex_subject.Subject()


def _history_size(spec: TopicSpec) -> int:
  if spec.retention == Retention.LATEST:
    return 1
  if spec.retention == Retention.REPLAY:
    return max(1, spec.replay_size)
  return 1  # Still remember the latest message for `latest()` and state.


def _quote(text: str) -> str:
  return text.replace('"', "'")


class TopicBus:
  """Named ReactiveX topics shared by the entities of a simulation.

  The bus is thread-safe. Publishing is serialized by a re-entrant lock so that
  sequence numbers and delivery order agree, and so that a subscriber may
  publish to another topic from inside its callback.
  """

  def __init__(
      self,
      *,
      clock: Callable[[], float] | None = None,
      topics: Iterable[TopicSpec] = (),
      default_retention: Retention = Retention.LATEST,
      measurements: measurements_lib.Measurements | None = None,
      measurements_prefix: str = DEFAULT_MEASUREMENTS_PREFIX,
  ):
    """Initializes the bus.

    Args:
      clock: Optional callable returning the current simulation time. Its value
        stamps every published message.
      topics: Topics to declare up front.
      default_retention: Retention for topics declared implicitly by publishing
        or subscribing to an undeclared name.
      measurements: Optional measurements sink. Every message is mirrored to the
        channel `measurements_prefix + topic` as a dictionary.
      measurements_prefix: Channel prefix used with `measurements`.
    """
    self._clock = clock
    self._default_retention = default_retention
    self._measurements = measurements
    self._measurements_prefix = measurements_prefix
    self._lock = threading.RLock()
    self._topics: dict[str, _Topic] = {}
    self._firehose = reactivex_subject.Subject()
    self._seq = 0
    self._closed = False
    self._restoring = False
    self._connections: list[reactivex_abc.DisposableBase] = []
    self._reflex_engines: list[weakref.ReferenceType[Any]] = []
    for spec in topics:
      self.declare(spec)

  @property
  def is_restoring(self) -> bool:
    """Returns True while `set_state` is replaying retained messages."""
    return self._restoring

  def register_reflex_engine(self, engine: Any) -> None:
    """Registers a `Reflexes` engine so bus checkpoints include its state."""
    with self._lock:
      if all(ref() is not engine for ref in self._reflex_engines):
        self._reflex_engines.append(weakref.ref(engine))

  def declare(
      self,
      topic: TopicSpec | str,
      *,
      retention: Retention | None = None,
      replay_size: int = 1,
      description: str = '',
  ) -> TopicSpec:
    """Declares a topic, or returns the existing declaration.

    Args:
      topic: A full `TopicSpec`, or a topic name.
      retention: Retention when `topic` is a name. Defaults to the bus default.
      replay_size: Replay buffer size when `topic` is a name.
      description: Description when `topic` is a name.

    Returns:
      The topic's spec. Re-declaring an existing topic keeps the original spec
      but fills in a missing description.

    Raises:
      ValueError: If the name is empty or contains whitespace.
    """
    if isinstance(topic, TopicSpec):
      spec = topic
    else:
      spec = TopicSpec(
          name=topic,
          retention=retention or self._default_retention,
          replay_size=replay_size,
          description=description,
      )
    if not spec.name or any(c.isspace() for c in spec.name):
      raise ValueError(f'Invalid topic name: {spec.name!r}')
    with self._lock:
      existing = self._topics.get(spec.name)
      if existing is not None:
        if spec.description and not existing.spec.description:
          existing.spec = dataclasses.replace(
              existing.spec, description=spec.description
          )
        return existing.spec
      self._topics[spec.name] = _Topic(
          spec=spec, subject=_make_subject(spec), history=[]
      )
      return spec

  def _topic(self, name: str) -> _Topic:
    with self._lock:
      if name not in self._topics:
        self.declare(name)
      return self._topics[name]

  def has_topic(self, name: str) -> bool:
    """Returns True if the topic has been declared."""
    with self._lock:
      return name in self._topics

  def topics(self, prefix: str = '') -> list[str]:
    """Returns the sorted names of declared topics starting with `prefix`."""
    with self._lock:
      return sorted(name for name in self._topics if name.startswith(prefix))

  def spec(self, name: str) -> TopicSpec:
    """Returns the spec of a declared topic."""
    with self._lock:
      return self._topics[name].spec

  def now(self) -> float | None:
    """Returns the current bus clock time, or None without a clock."""
    return self._clock() if self._clock is not None else None

  def publish(  # pylint: disable=redefined-outer-name
      self, topic: str, payload: Any, *, source: str = ''
  ) -> Message:
    """Publishes a payload on a topic.

    Args:
      topic: Topic name. Undeclared topics are declared with default retention.
      payload: Message body. Non-string payloads are converted with `str`.
      source: Name of the publisher, recorded in the message and topology.

    Returns:
      The published message.

    Raises:
      RuntimeError: If the bus has been closed.
    """
    with self._lock:
      if self._closed:
        raise RuntimeError('Cannot publish on a closed TopicBus.')
      state = self._topic(topic)
      self._seq += 1
      message = Message(
          topic=topic,
          payload=payload if isinstance(payload, str) else str(payload),
          seq=self._seq,
          time=self.now(),
          source=source,
      )
      state.count += 1
      state.history.append(message)
      del state.history[: -_history_size(state.spec)]
      if source and source not in state.publishers:
        state.publishers.append(source)
      state.subject.on_next(message)
      self._firehose.on_next(message)
      if self._measurements is not None:
        self._measurements.publish_datum(
            self._measurements_prefix + topic, message.to_dict()
        )
      return message

  def observable(self, topic: str) -> reactivex.Observable:
    """Returns the stream of `Message`s on a topic, honouring retention."""
    return self._topic(topic).subject

  def values(self, topic: str) -> reactivex.Observable:
    """Returns the stream of payload strings on a topic."""
    return self.observable(topic).pipe(ops.map(payload))

  def matching(self, prefix: str = '') -> reactivex.Observable:
    """Returns live messages on every topic whose name starts with `prefix`.

    The empty prefix matches every topic, as in
    `interrupt_scheduling.InterruptMask`. Only messages published after
    subscription are delivered.

    Args:
      prefix: Topic-name prefix.
    """
    return self._firehose.pipe(ops.filter(lambda m: m.topic.startswith(prefix)))

  def latest(self, topic: str) -> Message | None:
    """Returns the most recent message on a topic, or None."""
    with self._lock:
      state = self._topics.get(topic)
      if state is None or not state.history:
        return None
      return state.history[-1]

  def history(self, topic: str) -> list[Message]:
    """Returns the retained messages on a topic, oldest first."""
    with self._lock:
      state = self._topics.get(topic)
      return list(state.history) if state is not None else []

  def message_count(self, topic: str) -> int:
    """Returns the number of messages ever published on a topic."""
    with self._lock:
      state = self._topics.get(topic)
      return state.count if state is not None else 0

  def register_publisher(self, topic: str, name: str) -> None:
    """Records `name` as a publisher of `topic` for topology dumps."""
    with self._lock:
      state = self._topic(topic)
      if name and name not in state.publishers:
        state.publishers.append(name)

  def register_subscriber(self, topic: str, name: str) -> None:
    """Records `name` as a subscriber of `topic` for topology dumps."""
    with self._lock:
      state = self._topic(topic)
      if name and name not in state.subscribers:
        state.subscribers.append(name)

  def make_subscriber(
      self,
      topic: str | Sequence[str],
      *,
      subscriber_name: str = '',
      initial_value: str = '',
      pre_act_label: str | None = None,
      render: Callable[[Message], str] = payload,
      transform: (
          Callable[[reactivex.Observable], reactivex.Observable] | None
      ) = None,
      pipeline: Sequence[Callable[[reactivex.Observable], Any]] = (),
  ) -> pubsub.Subscriber:
    """Builds a `pubsub.Subscriber` component bound to one or more topics.

    Args:
      topic: Topic to subscribe to, or several topics whose messages are merged
        into one stream (retained messages are replayed topic by topic).
      subscriber_name: Name of the subscribing entity, for topology dumps.
      initial_value: The subscriber's value before any message arrives.
      pre_act_label: Label of the component. Defaults to the topic name(s).
      render: Converts each `Message` to the string the entity sees.
      transform: Optional operator from the topic's `Message` stream to strings
        (for example `operators.with_age(...)`); replaces `render`.
      pipeline: Extra ReactiveX operators applied to the rendered strings, for
        example `reactivex.operators.distinct_until_changed()`.

    Returns:
      A `Subscriber` whose current value tracks the topic.
    """
    names = [topic] if isinstance(topic, str) else list(topic)
    if not names:
      raise ValueError('At least one topic is required.')
    for name in names:
      if subscriber_name:
        self.register_subscriber(name, subscriber_name)
      else:
        self._topic(name)
    if len(names) == 1:
      source = self.observable(names[0])
    else:
      source = reactivex.merge(*(self.observable(name) for name in names))
    to_text = transform if transform is not None else ops.map(render)
    stack: list[list[Any]] = []
    # Collect any state hooks registered by operators in `to_text`/`pipeline`,
    # even when followed by standard ReactiveX operators like `ops.map`.
    from concordia.contrib.pubsub import operators as pubsub_operators  # pylint: disable=g-import-not-at-top

    del stack
    with pubsub_operators.collect_hooks() as collected_hooks:
      stream = source.pipe(to_text, *pipeline)
    hooks = list(getattr(stream, '_concordia_state_hooks', ()))
    for h in collected_hooks:
      if h not in hooks:
        hooks.append(h)
    if hooks:
      setattr(stream, '_concordia_state_hooks', hooks)
    return TopicSubscriber(
        publisher_component_key=None,
        initial_value=initial_value,
        pre_act_label=pre_act_label or ', '.join(names),
        subject=stream,
    )

  def connect_publisher(
      self,
      topic: str,
      publisher: pubsub.Publisher,
      *,
      source: str = '',
  ) -> reactivex_abc.DisposableBase:
    """Republishes every value a `pubsub.Publisher` completes on a topic.

    Args:
      topic: Topic to publish to.
      publisher: The publisher whose completed values are forwarded.
      source: Publisher name recorded on messages and in the topology.

    Returns:
      A disposable that disconnects the publisher from the bus.
    """
    self.register_publisher(topic, source)

    def forward(value: str) -> None:
      self.publish(topic, value, source=source)

    connection = publisher.observable.subscribe(on_next=forward)
    with self._lock:
      self._connections.append(connection)
    return connection

  def make_publisher(
      self,
      topic: str,
      function: (
          Callable[[], str] | action_spec_ignored.ActionSpecIgnored | None
      ) = None,
      *,
      component: action_spec_ignored.ActionSpecIgnored | None = None,
      source: str = '',
      **publisher_kwargs: Any,
  ) -> pubsub.Publisher:
    """Builds a `pubsub.Publisher` whose completed values go to a topic.

    Args:
      topic: Topic to publish completed values to.
      function: A callable or component, as for `pubsub.Publisher`.
      component: A component, as for `pubsub.Publisher`.
      source: Publisher name recorded on messages and in the topology.
      **publisher_kwargs: Other `pubsub.Publisher` arguments, such as
        `skip_if_busy`, `trigger_on_observe` or `scheduler`.

    Returns:
      The publisher. Add it to an entity like any other component.
    """
    publisher = sim_time.SimTimePublisher(
        function, component=component, **publisher_kwargs
    )
    self.connect_publisher(topic, publisher, source=source)
    return publisher

  def stats(self) -> dict[str, int]:
    """Returns the number of messages published per topic."""
    with self._lock:
      return {name: t.count for name, t in sorted(self._topics.items())}

  def topology(self) -> dict[str, dict[str, Any]]:
    """Returns publishers, subscribers and traffic for every topic."""
    with self._lock:
      return {
          name: {
              'retention': t.spec.retention.value,
              'description': t.spec.description,
              'publishers': list(t.publishers),
              'subscribers': list(t.subscribers),
              'messages': t.count,
          }
          for name, t in sorted(self._topics.items())
      }

  def to_mermaid(self) -> str:
    """Returns the publisher -> topic -> subscriber graph as Mermaid."""
    lines = ['flowchart LR']
    nodes: dict[str, str] = {}

    def node(kind: str, name: str) -> str:
      key = f'{kind}:{name}'
      if key not in nodes:
        nodes[key] = f'{kind[0]}{len(nodes)}'
        if kind == 'topic':
          lines.append(f'  {nodes[key]}(["{_quote(name)}"])')
        else:
          lines.append(f'  {nodes[key]}["{_quote(name)}"]')
      return nodes[key]

    for name, info in self.topology().items():
      topic_node = node('topic', name)
      for publisher in info['publishers']:
        lines.append(f'  {node("entity", publisher)} --> {topic_node}')
      for subscriber in info['subscribers']:
        lines.append(f'  {topic_node} --> {node("entity", subscriber)}')
    return '\n'.join(lines)

  def to_dot(self) -> str:
    """Returns the publisher -> topic -> subscriber graph as Graphviz DOT."""
    lines = ['digraph topics {', '  rankdir=LR;']
    for name, info in self.topology().items():
      lines.append(f'  "{_quote(name)}" [shape=ellipse];')
      for publisher in info['publishers']:
        lines.append(f'  "{_quote(publisher)}" -> "{_quote(name)}";')
      for subscriber in info['subscribers']:
        lines.append(f'  "{_quote(name)}" -> "{_quote(subscriber)}";')
    lines.append('}')
    return '\n'.join(lines)

  def get_state(self) -> dict[str, Any]:
    """Returns retained messages and counters for checkpointing."""
    with self._lock:
      state: dict[str, Any] = {
          'seq': self._seq,
          'topics': {
              name: {
                  'retention': t.spec.retention.value,
                  'replay_size': t.spec.replay_size,
                  'description': t.spec.description,
                  'count': t.count,
                  'history': [m.to_dict() for m in t.history],
              }
              for name, t in self._topics.items()
          },
      }
      live_reflexes = [
          r
          for ref in self._reflex_engines
          if (r := ref()) is not None and r.names()
      ]
      if live_reflexes:
        state['reflexes'] = [r.get_state() for r in live_reflexes]
      return state

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Restores retained messages and counters.

    Restored messages are replayed into each topic's subject (so current and
    future subscribers see the latest values) but are not sent to `matching`
    streams or to measurements, so automation does not fire again. Call this
    before the simulation resumes.

    Args:
      state: Output of `get_state`.
    """
    with self._lock:
      self._restoring = True
      try:
        self._seq = int(state.get('seq', 0))
        saved_topics = state.get('topics', {})
        if not isinstance(saved_topics, Mapping):
          saved_topics = {}
        for name, existing in self._topics.items():
          if name not in saved_topics:
            existing.count = 0
            existing.history.clear()
            queue = getattr(existing.subject, 'queue', None)
            if isinstance(queue, list):
              queue.clear()
        for name, data in saved_topics.items():
          if not isinstance(data, Mapping):
            continue
          spec = self.declare(
              TopicSpec(
                  name=name,
                  retention=Retention(data.get('retention', 'latest')),
                  replay_size=int(data.get('replay_size', 1)),
                  description=str(data.get('description', '')),
              )
          )
          topic = self._topics[spec.name]
          topic.count = int(data.get('count', 0))
          topic.history = [
              Message.from_dict(m) for m in data.get('history', [])
          ]
          queue = getattr(topic.subject, 'queue', None)
          if isinstance(queue, list):
            queue.clear()
          for message in topic.history:
            topic.subject.on_next(message)
        saved_reflexes = state.get('reflexes')
        if isinstance(saved_reflexes, Sequence) and not isinstance(
            saved_reflexes, str
        ):
          live_reflexes = [
              r for ref in self._reflex_engines if (r := ref()) is not None
          ]
          for engine, r_state in zip(live_reflexes, saved_reflexes):
            if isinstance(r_state, Mapping):
              engine.set_state(cast(Mapping[str, Any], r_state))
      finally:
        self._restoring = False

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> 'TopicBus':
    del memo
    return self

  def close(self) -> None:
    """Completes every topic stream and disconnects publishers."""
    with self._lock:
      if self._closed:
        return
      self._closed = True
      connections = list(self._connections)
      self._connections.clear()
      topics = list(self._topics.values())
    for connection in connections:
      connection.dispose()
    for topic in topics:
      topic.subject.on_completed()
    self._firehose.on_completed()
