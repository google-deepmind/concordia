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

"""ReactiveX publisher and subscriber agent components.

These components allow a slower computation (`Publisher`) on an entity to run
asynchronously in the background and publish its output via ReactiveX once it
finishes, while a paired `Subscriber` component on the same entity provides the
most recently completed value immediately during `pre_act` so the entity is
not blocked from acting.
"""

from collections.abc import Callable, Mapping, Sequence
import functools
import threading
import time
from typing import Any, cast, override

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging as concordia_logging
import pandas as pd
import reactivex
from reactivex import abc as reactivex_abc
from reactivex import operators as reactivex_operators
from reactivex import scheduler as reactivex_scheduler
from reactivex import subject as reactivex_subject

DEFAULT_PUBLISHER_COMPONENT_KEY = '__publisher__'
DEFAULT_PUBLISHER_PRE_ACT_LABEL = 'Publisher'
DEFAULT_SUBSCRIBER_PRE_ACT_LABEL = 'Subscribed Value'


class _SnapshottedComponentProxy(action_spec_ignored.ActionSpecIgnored):
  """Proxy returning a snapshotted pre-act value while delegating attributes."""

  def __init__(
      self,
      component: action_spec_ignored.ActionSpecIgnored,
      pre_act_value: str | None,
  ):
    super().__init__(pre_act_label=component.get_pre_act_label())
    self._wrapped_component = component
    self._snapshotted_value = pre_act_value

  @override
  def _make_pre_act_value(self) -> str:
    if self._snapshotted_value is None:
      self._snapshotted_value = (
          self._wrapped_component._make_pre_act_value()  # pylint: disable=protected-access
      )
    return self._snapshotted_value

  @override
  def get_pre_act_value(self) -> str:
    with self._lock:
      return self._make_pre_act_value()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped_component, name)


class _PhaseSafeMemoryProxy(memory_component.Memory):
  """Memory proxy allowing background threads to read memory across phases."""

  def __init__(self, memory: memory_component.Memory):
    self._memory = memory

  def _with_lock_or_call(self, method_name: str, *args: Any, **kwargs: Any):
    lock = getattr(self._memory, '_lock', None)
    bank = getattr(self._memory, '_memory_bank', None)
    if lock is not None and bank is not None:
      with lock:
        if method_name == 'retrieve_recent' and isinstance(bank, list):
          limit = args[0] if args else kwargs.get('limit', 1)
          return bank[-limit:]
        if method_name == 'scan' and isinstance(bank, list):
          selector_fn = args[0] if args else kwargs['selector_fn']
          return [mem for mem in bank if selector_fn(mem)]
        if method_name == 'get_all_memories_as_text' and isinstance(bank, list):
          return tuple(bank)
        if hasattr(bank, method_name):
          return getattr(bank, method_name)(*args, **kwargs)
    return getattr(self._memory, method_name)(*args, **kwargs)

  @override
  def get_state(self) -> Mapping[str, Any]:
    return self._memory.get_state()

  @override
  def set_state(self, state: Mapping[str, Any]) -> None:
    self._memory.set_state(state)

  def retrieve_associative(
      self,
      query: str,
      limit: int = 1,
  ) -> Sequence[str]:
    return self._with_lock_or_call('retrieve_associative', query, limit)

  @override
  def retrieve_recent(
      self,
      limit: int = 1,
  ) -> Sequence[str]:
    return self._with_lock_or_call('retrieve_recent', limit)

  @override
  def scan(
      self,
      selector_fn: Callable[[str], bool],
  ) -> Sequence[str]:
    return self._with_lock_or_call('scan', selector_fn)

  @override
  def add(self, text: str) -> None:
    lock = getattr(self._memory, '_lock', None)
    buffer = getattr(self._memory, '_buffer', None)
    if lock is not None and isinstance(buffer, list):
      with lock:
        buffer.append(text)
      return
    self._memory.add(text)

  @override
  def extend(self, texts: Sequence[str]) -> None:
    for text in texts:
      self.add(text)

  @override
  def update(self) -> None:
    self._memory.update()

  @override
  def get_raw_memory(self) -> pd.DataFrame:
    return self._with_lock_or_call('get_raw_memory')

  @override
  def get_all_memories_as_text(self) -> Sequence[str]:
    return self._with_lock_or_call('get_all_memories_as_text')

  def __getattr__(self, name: str) -> Any:
    return getattr(self._memory, name)


class _EntitySnapshotProxy(entity_component.EntityWithComponents):
  """Entity proxy allowing a wrapped component to run across entity phases."""

  def __init__(self, entity: entity_component.EntityWithComponents):
    self._entity = entity
    self._thread_local = threading.local()

  def set_thread_snapshot(self, snapshot: Mapping[str, str] | None) -> None:
    self._thread_local.snapshot = (
        dict(snapshot) if snapshot is not None else None
    )

  @property
  @override
  def name(self) -> str:
    return self._entity.name

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    return self._entity.act(action_spec)

  @override
  def observe(self, observation: str) -> None:
    self._entity.observe(observation)

  @override
  def get_phase(self) -> entity_component.Phase:
    return entity_component.Phase.PRE_ACT

  @override
  def get_component(
      self,
      name: str,
      *,
      type_: type[entity_component.ComponentT] = entity_component.BaseComponent,
  ) -> entity_component.ComponentT:
    component = self._entity.get_component(name, type_=type_)
    if isinstance(component, memory_component.Memory):
      return cast(
          entity_component.ComponentT,
          _PhaseSafeMemoryProxy(component),
      )
    snapshot: dict[str, str] | None = getattr(
        self._thread_local, 'snapshot', None
    )
    if isinstance(component, action_spec_ignored.ActionSpecIgnored):
      snapshotted_val = snapshot.get(name) if snapshot is not None else None
      if snapshotted_val is None and isinstance(component, Subscriber):
        snapshotted_val = component.get_current_value()
      proxy = _SnapshottedComponentProxy(component, snapshotted_val)
      if snapshot is not None and snapshotted_val is None:
        val = proxy.get_pre_act_value()
        snapshot[name] = val
      return cast(entity_component.ComponentT, proxy)
    return component

  @override
  def get_state(self) -> entity_component.EntityState:
    return self._entity.get_state()

  @override
  def set_state(self, state: entity_component.EntityState) -> None:
    self._entity.set_state(state)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._entity, name)


class Publisher(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Component that triggers a slow computation asynchronously via ReactiveX.

  When `pre_act` (or `get_pre_act_value`) is called on the `Publisher`—and
  optionally in `post_observe` when `trigger_on_observe=True`—it dispatches its
  computation to a ReactiveX scheduler (by default a `ThreadPoolScheduler`) and
  returns immediately without blocking the entity. When the computation
  finishes, the result is published on a ReactiveX `Subject` so any subscribed
  `Subscriber` component will use the updated value on subsequent actions.
  """

  def __init__(
      self,
      function: (
          Callable[[], str] | action_spec_ignored.ActionSpecIgnored | None
      ) = None,
      *,
      component: action_spec_ignored.ActionSpecIgnored | None = None,
      pre_act_label: str = DEFAULT_PUBLISHER_PRE_ACT_LABEL,
      skip_if_busy: bool = True,
      trigger_on_observe: bool = False,
      subject: reactivex_subject.Subject | None = None,
      scheduler: reactivex_abc.SchedulerBase | None = None,
  ):
    """Initializes the Publisher component.

    Args:
      function: A callable returning a string to compute asynchronously, or an
        `ActionSpecIgnored` component if passed positionally. Exactly one of
        `function` or `component` must be provided.
      component: An `ActionSpecIgnored` component whose `_make_pre_act_value`
        computation should be executed asynchronously. Exactly one of `function`
        or `component` must be provided.
      pre_act_label: Label used for component logging and state metadata.
      skip_if_busy: If True, triggers occurring while a previous computation is
        still in flight are skipped. If False, a new computation is started on
        every trigger and only the value from the most recent trigger (by
        monotonic sequence ID) is published when completed.
      trigger_on_observe: If True, also triggers the asynchronous computation in
        `post_observe` whenever `entity.observe()` is called (in addition to
        `pre_act`). Defaults to False.
      subject: Optional ReactiveX `Subject` to publish completed values to. If
        None, a new `reactivex.subject.Subject` is created.
      scheduler: Optional ReactiveX `SchedulerBase` on which to execute the
        computation. Defaults to `reactivex.scheduler.ThreadPoolScheduler()`.

    Raises:
      ValueError: If both or neither of `function` and `component` are provided.
    """
    super().__init__(pre_act_label)
    if isinstance(function, action_spec_ignored.ActionSpecIgnored):
      if component is not None:
        raise ValueError(
            'Exactly one of `function` or `component` must be provided.'
        )
      component = function
      function = None

    if (function is None) == (component is None):
      raise ValueError(
          'Exactly one of `function` or `component` must be provided.'
      )

    self._function: Callable[[], str] | None = function
    self._component: action_spec_ignored.ActionSpecIgnored | None = component
    self._entity_proxy: _EntitySnapshotProxy | None = None
    self._skip_if_busy: bool = skip_if_busy
    self._trigger_on_observe: bool = trigger_on_observe
    self._subject: reactivex_subject.Subject = (
        subject if subject is not None else reactivex_subject.Subject()
    )
    self._scheduler: reactivex_abc.SchedulerBase = (
        scheduler
        if scheduler is not None
        else reactivex_scheduler.ThreadPoolScheduler()
    )

    self._condition_lock = threading.Condition()
    self._publish_lock = threading.Lock()
    self._in_flight: int = 0
    self._next_sequence_id: int = 0
    self._latest_completed_sequence_id: int = 0
    self._latest_value: str | None = None
    self._compute_count: int = 0
    self._last_compute_duration: float = 0.0
    self._subscribers: list[Subscriber] = []

    self._trigger_subject: reactivex_subject.Subject = (
        reactivex_subject.Subject()
    )
    self._trigger_subscription: (
        reactivex_abc.DisposableBase
    ) = self._trigger_subject.pipe(
        reactivex_operators.flat_map(
            lambda trigger: self._make_task_observable(trigger[0], trigger[1])
        ),
    ).subscribe(
        on_error=lambda err: logging.error(
            'Unexpected error in Publisher stream: %s', err
        ),
    )

  @property
  def observable(self) -> reactivex.Observable:
    """Returns the ReactiveX Observable stream of published string values."""
    return self._subject

  @property
  def subject(self) -> reactivex_subject.Subject:
    """Returns the underlying ReactiveX Subject."""
    return self._subject

  @property
  def trigger_on_observe(self) -> bool:
    """Returns whether background computation is also triggered on observe()."""
    return self._trigger_on_observe

  @property
  def compute_count(self) -> int:
    """Returns the number of completed background computations."""
    with self._condition_lock:
      return self._compute_count

  @property
  def last_compute_duration(self) -> float:
    """Returns the duration in seconds of the last completed computation."""
    with self._condition_lock:
      return self._last_compute_duration

  def register_subscriber(self, subscriber: 'Subscriber') -> None:
    """Registers a Subscriber so its pre-act value is latched before triggers."""
    with self._condition_lock:
      if subscriber not in self._subscribers:
        self._subscribers.append(subscriber)

  def is_busy(self) -> bool:
    """Returns True if at least one asynchronous computation is in flight."""
    with self._condition_lock:
      return self._in_flight > 0

  def wait(self, timeout: float | None = None) -> bool:
    """Waits until all currently in-flight computations have completed.

    Args:
      timeout: Maximum time in seconds to wait, or None to wait indefinitely.

    Returns:
      True if all in-flight computations completed before the timeout expired,
      or False if the timeout expired while still busy.
    """
    with self._condition_lock:
      return self._condition_lock.wait_for(
          lambda: self._in_flight == 0, timeout=timeout
      )

  def close(self) -> None:
    """Disposes of the internal ReactiveX trigger subscription."""
    self._trigger_subscription.dispose()

  @override
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    if self._component is not None:
      self._entity_proxy = _EntitySnapshotProxy(entity)
      self._component.set_entity(self._entity_proxy)

  @override
  def set_logging_channel(
      self, logging_channel: concordia_logging.LoggingChannel
  ) -> None:
    super().set_logging_channel(logging_channel)
    if isinstance(self._component, entity_component.ComponentWithLogging):
      self._component.set_logging_channel(logging_channel)

  def _make_task_observable(
      self,
      sequence_id: int,
      snapshot: Mapping[str, str],
  ) -> reactivex.Observable:
    return reactivex.from_callable(
        functools.partial(self._compute, sequence_id, snapshot),
        scheduler=self._scheduler,
    ).pipe(
        reactivex_operators.filter(lambda item: item is not None),
        reactivex_operators.do_action(
            on_next=self._publish_if_latest,
            on_completed=self._on_task_completed,
        ),
    )

  def _compute(
      self,
      sequence_id: int,
      snapshot: Mapping[str, str],
  ) -> tuple[int, str] | None:
    start_time = time.perf_counter()
    try:
      if self._function is not None:
        value = self._function()
      else:
        assert self._component is not None
        if self._entity_proxy is not None:
          self._entity_proxy.set_thread_snapshot(snapshot)
        try:
          value = self._component._make_pre_act_value()  # pylint: disable=protected-access
          with self._component._lock:  # pylint: disable=protected-access
            self._component._pre_act_value = value  # pylint: disable=protected-access
        finally:
          if self._entity_proxy is not None:
            self._entity_proxy.set_thread_snapshot(None)
      duration = time.perf_counter() - start_time
      with self._condition_lock:
        self._compute_count += 1
        self._last_compute_duration = duration
      return (sequence_id, value)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Error during asynchronous Publisher computation.')
      return None

  def _publish_if_latest(self, result: tuple[int, str] | None) -> None:
    if result is None:
      return
    sequence_id, value = result
    with self._publish_lock:
      with self._condition_lock:
        if sequence_id <= self._latest_completed_sequence_id:
          return
        self._latest_completed_sequence_id = sequence_id
        self._latest_value = value
      self._subject.on_next(value)

  def _on_task_completed(self) -> None:
    with self._condition_lock:
      self._in_flight = max(0, self._in_flight - 1)
      self._condition_lock.notify_all()

  def _latch_subscribers_for_current_act(self) -> None:
    """Ensures subscribers on the same entity latch their previous value."""
    subscribers: list[Subscriber] = []
    with self._condition_lock:
      subscribers.extend(self._subscribers)
    if self._entity is not None and hasattr(
        self._entity, 'get_all_context_components'
    ):
      all_components = getattr(self._entity, 'get_all_context_components')()
      for comp in all_components.values():
        if isinstance(comp, Subscriber) and comp not in subscribers:
          subscribers.append(comp)

    for subscriber in subscribers:
      try:
        sub_entity = subscriber.get_entity()
      except ValueError:
        continue
      if sub_entity.get_phase() in (
          entity_component.Phase.PRE_ACT,
          entity_component.Phase.POST_ACT,
      ):
        subscriber.get_pre_act_value()

  def _referenced_component_keys(self) -> set[str] | None:
    """Returns keys referenced by the wrapped component, or None if unknown."""
    if self._component is None:
      return set()
    keys: set[str] = set()
    has_explicit_deps = False
    comp_keys = getattr(self._component, '_components', None)
    if isinstance(comp_keys, (list, tuple, set, frozenset)):
      has_explicit_deps = True
      keys.update(map(str, comp_keys))
    dep_key = getattr(self._component, '_dependency_key', None)
    if isinstance(dep_key, str):
      has_explicit_deps = True
      keys.add(dep_key)
    return keys if has_explicit_deps else None

  def _snapshot_sibling_components(
      self, *, from_observe: bool = False
  ) -> dict[str, str]:
    """Snapshots sibling ActionSpecIgnored components."""
    snapshot: dict[str, str] = {}
    if (
        self._component is None
        or self._entity is None
        or not hasattr(self._entity, 'get_all_context_components')
    ):
      return snapshot

    referenced_keys = self._referenced_component_keys()
    all_components = getattr(self._entity, 'get_all_context_components')()
    in_act_phase = self._entity.get_phase() in (
        entity_component.Phase.PRE_ACT,
        entity_component.Phase.POST_ACT,
    )

    for key, comp in all_components.items():
      if (
          comp is self
          or comp is self._component
          or isinstance(comp, Publisher)
          or not isinstance(comp, action_spec_ignored.ActionSpecIgnored)
      ):
        continue
      if isinstance(comp, Subscriber):
        snapshot[key] = (
            comp.get_pre_act_value()
            if in_act_phase
            else comp.get_current_value()
        )
        continue
      if referenced_keys is not None and key not in referenced_keys:
        continue
      if in_act_phase and not from_observe:
        snapshot[key] = comp.get_pre_act_value()
      elif referenced_keys is not None and key in referenced_keys:
        snapshot[key] = comp._make_pre_act_value()  # pylint: disable=protected-access
    return snapshot

  def _flush_entity_memories(self) -> None:
    """Commits buffered observations on any Memory components of the entity."""
    if self._entity is not None and hasattr(
        self._entity, 'get_all_context_components'
    ):
      all_components = getattr(self._entity, 'get_all_context_components')()
      for comp in all_components.values():
        if isinstance(comp, memory_component.Memory):
          comp.update()

  def _trigger_computation(self, *, from_observe: bool = False) -> str:
    """Dispatches the background computation to the ReactiveX scheduler."""
    if not from_observe:
      self._latch_subscribers_for_current_act()

    with self._condition_lock:
      latest = self._latest_value or ''
      if self._skip_if_busy and self._in_flight > 0:
        self._logging_channel({
            'Key': self.get_pre_act_label(),
            'Value': latest,
            'Status': 'skipped_busy',
            'Trigger': 'observe' if from_observe else 'pre_act',
        })
        return latest
      self._next_sequence_id += 1
      sequence_id = self._next_sequence_id
      self._in_flight += 1

    try:
      snapshot = self._snapshot_sibling_components(from_observe=from_observe)
      self._trigger_subject.on_next((sequence_id, snapshot))
    except Exception:
      self._on_task_completed()
      raise

    with self._condition_lock:
      latest = self._latest_value or ''
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': latest,
        'Status': 'triggered',
        'Trigger': 'observe' if from_observe else 'pre_act',
    })
    return latest

  @override
  def _make_pre_act_value(self) -> str:
    return self._trigger_computation(from_observe=False)

  @override
  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Triggers the async computation and returns empty string for prompt."""
    del action_spec
    self.get_pre_act_value()
    return ''

  @override
  def pre_observe(self, observation: str) -> str:
    if self._component is not None:
      self._component.pre_observe(observation)
    return ''

  @override
  def post_observe(self) -> str:
    if self._component is not None:
      self._component.post_observe()
    if self._trigger_on_observe:
      self._flush_entity_memories()
      self._trigger_computation(from_observe=True)
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._condition_lock:
      state: dict[str, Any] = {
          'pre_act_label': self.get_pre_act_label(),
          'skip_if_busy': self._skip_if_busy,
          'trigger_on_observe': self._trigger_on_observe,
          'latest_value': self._latest_value,
      }
      if self._component is not None:
        state['component'] = self._component.get_state()
      return state

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._condition_lock:
      if 'skip_if_busy' in state:
        self._skip_if_busy = bool(state['skip_if_busy'])
      if 'trigger_on_observe' in state:
        self._trigger_on_observe = bool(state['trigger_on_observe'])
      if 'latest_value' in state:
        self._latest_value = (
            str(state['latest_value'])
            if state['latest_value'] is not None
            else None
        )
      if self._component is not None and isinstance(
          state.get('component'), Mapping
      ):
        self._component.set_state(
            cast(entity_component.ComponentState, state['component'])
        )


class Subscriber(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Component that subscribes to a Publisher and exposes its latest value.

  During `pre_act`, `Subscriber` immediately returns the most recently published
  value (or `initial_value` if no computation has completed yet), allowing the
  entity to act without waiting for an in-flight slow computation to finish.
  """

  def __init__(
      self,
      publisher_component_key: str | None = DEFAULT_PUBLISHER_COMPONENT_KEY,
      *,
      initial_value: str = '',
      pre_act_label: str = DEFAULT_SUBSCRIBER_PRE_ACT_LABEL,
      publisher: Publisher | None = None,
      subject: reactivex.Observable | None = None,
  ):
    """Initializes the Subscriber component.

    Args:
      publisher_component_key: Key of the `Publisher` component on the same
        entity to subscribe to when `set_entity` is called. Ignored if
        `publisher` or `subject` is explicitly provided.
      initial_value: Value returned by `Subscriber` before the `Publisher`
        completes its first computation.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      publisher: Optional `Publisher` instance to subscribe to directly.
      subject: Optional ReactiveX `Observable` or `Subject` to subscribe to
        directly.
    """
    super().__init__(pre_act_label)
    self._publisher_component_key: str | None = publisher_component_key
    self._initial_value: str = initial_value
    self._current_value: str = initial_value
    self._value_lock = threading.Lock()
    self._subject: reactivex.Observable | None = subject
    self._subscription: reactivex_abc.DisposableBase | None = None

    if publisher is not None:
      self.subscribe_to(publisher)
    elif subject is not None:
      self._subscribe_to_observable(subject)

  def _on_next(self, value: str) -> None:
    with self._value_lock:
      self._current_value = str(value)

  def _subscribe_to_observable(self, observable: reactivex.Observable) -> None:
    if self._subscription is not None:
      self._subscription.dispose()
    self._subject = observable
    self._subscription = observable.subscribe(on_next=self._on_next)

  def subscribe_to(self, publisher: Publisher) -> None:
    """Subscribes this component to the given `Publisher`."""
    publisher.register_subscriber(self)
    self._subscribe_to_observable(publisher.observable)

  def close(self) -> None:
    """Disposes of the active ReactiveX subscription, if any."""
    if self._subscription is not None:
      self._subscription.dispose()
      self._subscription = None

  def get_current_value(self) -> str:
    """Returns the latest received value (or initial value if none received)."""
    with self._value_lock:
      return self._current_value

  @override
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    if self._subscription is None and self._publisher_component_key is not None:
      try:
        publisher = entity.get_component(
            self._publisher_component_key, type_=Publisher
        )
      except KeyError:
        publisher = None
      if publisher is not None:
        self.subscribe_to(publisher)

  def _ensure_connected(self) -> None:
    if self._subscription is not None or self._publisher_component_key is None:
      return
    publisher = self.get_entity().get_component(
        self._publisher_component_key, type_=Publisher
    )
    self.subscribe_to(publisher)

  @override
  def _make_pre_act_value(self) -> str:
    self._ensure_connected()
    value = self.get_current_value()
    self._logging_channel({'Key': self.get_pre_act_label(), 'Value': value})
    return value

  def get_dynamic_state(self) -> dict[str, Any]:
    return {'value': self.get_current_value()}

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._value_lock:
      return {
          'pre_act_label': self.get_pre_act_label(),
          'publisher_component_key': self._publisher_component_key,
          'initial_value': self._initial_value,
          'current_value': self._current_value,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._value_lock:
      if 'publisher_component_key' in state:
        key = state['publisher_component_key']
        self._publisher_component_key = str(key) if key is not None else None
      if 'initial_value' in state and isinstance(state['initial_value'], str):
        self._initial_value = state['initial_value']
      if 'current_value' in state and isinstance(state['current_value'], str):
        self._current_value = state['current_value']
