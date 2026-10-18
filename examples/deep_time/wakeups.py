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

"""Wake-ups: entities that act only when something they care about happens.

In event-driven eras a species does not deliberate every
tick. It sleeps until a topic in its `InterruptMask` carries news (biological
or geological events, diplomacy addressed to it, or an alarm about its own
meadows) or until its `Timer` says "reconsider". Its beliefs keep updating
while it sleeps; only its decisions, and so its language model calls, are
event-driven.

Everything here is stock Concordia or contrib: the upstream
`interrupt_scheduling.EntityScheduler` keeps masks and timers, contrib
`topics.as_interrupt_event` turns bus messages into its `Event`s, and the
alarms are contrib `operators.deadband` reflexes over the census. Both the
scheduler and the wake-ups are World GM components, checkpointed with it;
the GM's `WorldStep` steps them.
"""

# pylint: disable=missing-function-docstring

from collections.abc import Mapping
import json
import threading
from typing import Any, cast

from concordia.components.game_master import interrupt_scheduling
from concordia.contrib.components.game_master import numeric_time
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import topics
from concordia.typing import entity_component
from reactivex import operators as ops

DEFAULT_WAKEUPS_COMPONENT_KEY = '__wakeups__'


class _Scheduler(interrupt_scheduling.EntityScheduler):
  """The upstream scheduler, plus entities that join mid-run."""

  def add_player(self, name: str) -> None:
    if name in self._masks:
      return
    self._player_names.append(name)
    self._masks[name] = interrupt_scheduling.MATCH_ALL
    self._timers[name] = interrupt_scheduling.Timer(
        expiry=self.get_current_time(), entity_name=name, description='joined'
    )
    self._pending[name] = []


class WakeUps(entity_component.ContextComponent):
  """A World GM component: which event-driven entities are awake each tick.

  Its masks and timers are kept by the upstream `EntityScheduler`
  (`scheduler`), another World GM component; this one keeps what wakes whom:
  the events since the last step, the alarms' deadbands and the tallies.
  """

  def __init__(self, bus: topics.TopicBus):
    """Initializes wake-ups.

    Args:
      bus: The topic bus whose messages are the events.
    """
    super().__init__()
    self._bus = bus
    self.scheduler = _Scheduler(
        player_names=[],
        time_model=numeric_time.NumericTimeModel(
            0.0, units={'tick': 1.0}, default_unit='tick'
        ),
    )
    self._masks: dict[str, interrupt_scheduling.InterruptMask] = {}
    self._timer_ticks: dict[str, int] = {}
    self._inbox: list[topics.Message] = []
    self._lock = threading.Lock()
    self.awake: set[str] = set()
    self.woken_by: dict[str, dict[str, int]] = {}
    # Each alarm's deadband (checkpointed: the value it last fired at).
    self._alarm_hooks: list[operators.StateHook] = []
    bus.matching('').subscribe(self._collect)

  def _collect(self, message: topics.Message) -> None:
    if self._bus.is_restoring:  # A checkpoint's replay wakes no one.
      return
    if not str(message.payload).strip():  # An empty message is no news.
      return
    with self._lock:
      self._inbox.append(message)

  def add(self, name: str, key: str, rules: Mapping[str, Any]) -> None:
    """Puts an entity under wake-ups and installs its alarm reflexes.

    Args:
      name: Entity name.
      key: The entity's key, substituted for `{key}` in the rules.
      rules: A `wakeups` section of the data: `prefixes` (topics that wake the
        entity), `timer_ticks`, `alert_topic`, and `alarms`, each a deadband
        reflex `{topic, path, threshold, text}` whose `text` may use `{name}`
        and `{value}`.
    """
    self._masks[name] = interrupt_scheduling.mask_from_prefixes(
        [str(p).format(key=key) for p in rules['prefixes']]
    )
    self._timer_ticks[name] = int(rules['timer_ticks'])
    self.scheduler.add_player(name)
    self.woken_by[name] = {'timer': 0, 'event': 0}
    alert_topic = str(rules['alert_topic']).format(key=key)

    def alarm(template: str, value: Any) -> None:
      text = template.format(name=name, value=float(value))
      self._bus.publish(alert_topic, text, source='world')

    for rule in rules.get('alarms', ()):
      stream = self._bus.observable(str(rule['topic'])).pipe(
          ops.filter(lambda _: not self._bus.is_restoring),
          operators.json_field(str(rule['path']).format(key=key)),
          operators.deadband(float(rule['threshold'])),
      )
      self._alarm_hooks.extend(getattr(stream, operators.HOOKS_ATTR, ()))
      stream.subscribe(
          lambda value, template=str(rule['text']): alarm(template, value)
      )

  def managed(self, name: str) -> bool:
    return name in self._masks

  def step(self, tick: int) -> set[str]:
    """Wakes entities for this tick's events and timers; returns them."""
    with self._lock:
      inbox, self._inbox = self._inbox, []
    scheduler = self.scheduler
    scheduler.advance_time(max(scheduler.get_current_time(), float(tick)))
    reasons: dict[str, str] = {}
    for name in scheduler.player_names:
      timer = scheduler.get_timer(name)
      if timer is not None and timer.expiry <= tick:
        reasons[name] = 'timer'
    for message in inbox:
      event = topics.as_interrupt_event(message)
      for name in scheduler.get_entities_to_poll(event):
        reasons.setdefault(name, 'event')
    for name, reason in reasons.items():
      scheduler.poll_entity(name)
      scheduler.set_mask(name, self._masks[name])
      scheduler.set_timer(
          name,
          interrupt_scheduling.Timer(
              expiry=tick + self._timer_ticks[name],
              entity_name=name,
              description='reconsider',
          ),
      )
      self.woken_by[name][reason] += 1
    self.awake = set(reasons)
    return self.awake

  def is_awake(self, name: str) -> bool:
    return not self.managed(name) or name in self.awake

  def get_state(self) -> entity_component.ComponentState:
    # Masks and timers are the scheduler's state (its own component).
    with self._lock:
      inbox = [m.to_dict() for m in self._inbox]
    return {
        'awake': sorted(self.awake),
        'woken_by': json.loads(json.dumps(self.woken_by)),
        # Events since the last step (they wake entities at the next), and
        # where each alarm's deadband stands.
        'inbox': inbox,
        'alarms': [get() for get, _ in self._alarm_hooks],
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    saved = cast(Mapping[str, Any], state)
    self.awake = set(saved.get('awake', []))
    with self._lock:
      self._inbox = [
          topics.Message.from_dict(m) for m in saved.get('inbox', [])
      ]
    for (_, restore), alarm in zip(self._alarm_hooks, saved.get('alarms', [])):
      restore(alarm)
    self.woken_by = {
        name: dict(counts) for name, counts in saved.get('woken_by', {}).items()
    }
