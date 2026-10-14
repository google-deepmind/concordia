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

"""Topic messages as game-master observations, routed by access lists.

Game masters deliver observations through `make_observation.MakeObservation`,
usually by pushing strings into its `ObservationQueue` by hand. When the world
already publishes its state on a `contrib.pubsub.topics.TopicBus`, the
routing can be declarative instead: `TopicObservations` subscribes to the bus
and queues each message for every entity whose access list matches the topic.

Access lists are `interrupt_scheduling.InterruptMask`s over topic names, so
`('geo.', 'bio.events')` hears every geology topic and the biology event
stream, `('',)` hears everything and `()` hears nothing. Lists can change at
run time (`grant`, `revoke`), for example when an entity researches an
instrument or a spy network. Optionally, the component also publishes every
resolved event (`EVENT_TAG` observations of the game master) on a topic, so
game-master state changes reach topic subscribers too.

Example:

```python
queue = make_observation.ObservationQueue()
routes = topic_observations.TopicObservations(
    bus,
    queue,
    access={'Alice': ('weather.',), 'Bob': ('weather.', 'market.')},
    event_topic='events.resolved',
)
# Add `routes` to the game master's components, and build its
# MakeObservation with external_queue=queue.
```
"""

from collections.abc import Callable, Mapping, Sequence
import threading
from typing import Any, cast

from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import event_resolution
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import make_observation
from concordia.contrib.pubsub import topics
from concordia.typing import entity_component

DEFAULT_TOPIC_OBSERVATIONS_COMPONENT_KEY = '__topic_observations__'
DEFAULT_TOPIC_OBSERVATIONS_PRE_ACT_LABEL = 'Topic observations'

Renderer = Callable[[topics.Message], str]


class TopicObservations(action_spec_ignored.ActionSpecIgnored):
  """Queues topic messages as observations for entities that may hear them."""

  def __init__(
      self,
      bus: topics.TopicBus,
      queue: make_observation.ObservationQueue,
      *,
      access: Mapping[str, Sequence[str]] | None = None,
      render: Renderer | Mapping[str, Renderer] = topics.payload,
      event_topic: str | None = None,
      pre_act_label: str = DEFAULT_TOPIC_OBSERVATIONS_PRE_ACT_LABEL,
  ):
    """Initializes the component and subscribes to the bus.

    Args:
      bus: The topic bus to listen to.
      queue: The observation queue of the game master's `MakeObservation`.
      access: Topic-name prefixes each entity may hear.
      render: Converts a message into observation text: one function, or a
        mapping from topic prefix to function (longest prefix wins; topics with
        no matching prefix use the payload).
      event_topic: If set, resolved events observed by the game master are
        published on this topic.
      pre_act_label: Label shown in the game master's context.
    """
    super().__init__(pre_act_label)
    self._bus = bus
    self._queue = queue
    self._render = render
    self._event_topic = event_topic
    self._access_lock = threading.Lock()
    self._access: dict[str, tuple[str, ...]] = {
        name: tuple(prefixes) for name, prefixes in (access or {}).items()
    }
    self._delivered: dict[str, int] = {}
    self._subscription = bus.matching('').subscribe(self._on_message)

  def access(self, entity: str) -> tuple[str, ...]:
    """The topic prefixes an entity may hear."""
    with self._access_lock:
      return self._access.get(entity, ())

  def set_access(self, entity: str, prefixes: Sequence[str]) -> None:
    with self._access_lock:
      self._access[entity] = tuple(prefixes)

  def grant(self, entity: str, prefix: str) -> None:
    """Lets an entity hear topics starting with `prefix`."""
    with self._access_lock:
      current = self._access.get(entity, ())
      if prefix not in current:
        self._access[entity] = current + (prefix,)

  def revoke(self, entity: str, prefix: str) -> None:
    """Stops an entity hearing topics starting with exactly `prefix`."""
    with self._access_lock:
      current = self._access.get(entity, ())
      self._access[entity] = tuple(p for p in current if p != prefix)

  def delivered(self) -> dict[str, int]:
    """Number of observations queued per entity so far."""
    with self._access_lock:
      return dict(self._delivered)

  def listeners(self, topic: str) -> list[str]:
    """Entities whose access list matches a topic."""
    event = interrupt_scheduling.Event(
        timestamp=0, tag=topic, source='', description=''
    )
    with self._access_lock:
      return [
          name
          for name, prefixes in self._access.items()
          if interrupt_scheduling.mask_from_prefixes(prefixes).matches(event)
      ]

  def _renderer_for(self, topic: str) -> Renderer:
    if callable(self._render):
      return self._render
    matches = [p for p in self._render if topic.startswith(p)]
    if not matches:
      return topics.payload
    return self._render[max(matches, key=len)]

  def _on_message(self, message: topics.Message) -> None:
    listeners = self.listeners(message.topic)
    if not listeners:
      return
    text = self._renderer_for(message.topic)(message)
    if not text:
      return
    for name in listeners:
      self._queue.add(name, text, ())
    with self._access_lock:
      for name in listeners:
        self._delivered[name] = self._delivered.get(name, 0) + 1

  def pre_observe(self, observation: str) -> str:
    tag = event_resolution.EVENT_TAG
    if self._event_topic and observation.startswith(tag):
      event = observation[len(tag) :].strip()
      if event:
        source = ''
        try:
          source = self.get_entity().name
        except RuntimeError:
          pass
        self._bus.publish(self._event_topic, event, source=source)
    return ''

  def _make_pre_act_value(self) -> str:
    with self._access_lock:
      return '\n'.join(
          f'{name} hears: {", ".join(p or "(everything)" for p in prefixes)}'
          for name, prefixes in sorted(self._access.items())
          if prefixes
      )

  def close(self) -> None:
    """Stops listening to the bus."""
    self._subscription.dispose()

  def get_state(self) -> entity_component.ComponentState:
    with self._access_lock:
      return {
          'access': {k: list(v) for k, v in self._access.items()},
          'delivered': dict(self._delivered),
          'bus': self._bus.get_state(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    bus_state = state.get('bus')
    if isinstance(bus_state, Mapping):
      self._bus.set_state(cast(Mapping[str, Any], bus_state))
    access = state.get('access')
    delivered = state.get('delivered')
    with self._access_lock:
      if isinstance(access, Mapping):
        access_map = cast(Mapping[str, Sequence[str]], access)
        self._access = {
            str(k): tuple(str(p) for p in v) for k, v in access_map.items()
        }
      if isinstance(delivered, Mapping):
        delivered_map = cast(Mapping[str, int], delivered)
        self._delivered = {str(k): int(v) for k, v in delivered_map.items()}

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'TopicObservations':
    self.close()
    with self._access_lock:
      access = {k: tuple(v) for k, v in self._access.items()}
      delivered = dict(self._delivered)
    copied = TopicObservations(
        self._bus,
        self._queue,
        access=access,
        render=self._render,
        event_topic=self._event_topic,
        pre_act_label=self.get_pre_act_label(),
    )
    with copied._access_lock:  # pylint: disable=protected-access
      copied._delivered = delivered  # pylint: disable=protected-access
    del memo
    return copied
