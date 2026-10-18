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

"""The World GM's deliveries: what reaches each entity from the topics.

An entity reads a topic through an instrument: its readings arrive late,
some are lost, some carry noise (a contrib `operators.transport` whose
channel the world sets: the Steward's instrument tier, the light delay
between worlds, a people's couriers), and each shows its age (contrib
`operators.with_age`). The World GM runs every instrument here and hands the
entity only what arrives, rendered, through a stock `TopicSubscriber`; the
readings still on their way and the channels' random draws are this
component's state, checkpointed with the GM.
"""

from collections.abc import Callable
import threading
from typing import Any, cast, override

from concordia.contrib.pubsub import operators
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import reactivex
from reactivex import subject as reactivex_subject

DEFAULT_DELIVERIES_KEY = '__deliveries__'


class Deliveries(entity_component.ContextComponent):
  """Runs each entity's instruments for the World GM; keeps what is in flight."""

  def __init__(self):
    super().__init__()
    self._lock = threading.Lock()
    # Each instrument's checkpoint hooks, by '<owner>/<label>'.
    self._hooks: dict[str, list[operators.StateHook]] = {}
    # What reaches each instrument's entity, by '<owner>/<label>'.
    self._channels: dict[str, reactivex_subject.ReplaySubject] = {}

  def channel(self, key: str) -> reactivex_subject.ReplaySubject:
    """What reaches an entity through an instrument (keeps the latest).

    The entity's subscriber may join before the World GM runs the
    instrument (`deliver`), or after: it hears the latest arrival either way.

    Args:
      key: The instrument, as '<owner>/<label>'.

    Returns:
      The rendered readings as they arrive.
    """
    with self._lock:
      if key not in self._channels:
        self._channels[key] = reactivex_subject.ReplaySubject(buffer_size=1)
      return self._channels[key]

  def deliver(
      self, key: str, build: Callable[[], reactivex.Observable]
  ) -> None:
    """Runs an instrument for an entity, into its channel (`channel`).

    Args:
      key: The instrument, as '<owner>/<label>' (unique).
      build: Builds the instrument's stream from the bus (its transport,
        ageing and rendering).
    """
    with operators.collect_hooks() as collected:
      stream = build()
    hooks = list(getattr(stream, operators.HOOKS_ATTR, ()))
    hooks.extend(h for h in collected if h not in hooks)
    arrived = self.channel(key)
    with self._lock:
      if key in self._hooks:
        raise ValueError(f'Instrument {key!r} is delivered twice.')
      self._hooks[key] = hooks
    stream.subscribe(on_next=arrived.on_next)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          key: [get() for get, _ in hooks]
          for key, hooks in self._hooks.items()
          if hooks
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = cast(dict[str, Any], state)
    with self._lock:
      hooks = dict(self._hooks)
    for key, states in saved.items():
      for (_, restore), hook_state in zip(hooks.get(key, ()), states):
        restore(hook_state)
