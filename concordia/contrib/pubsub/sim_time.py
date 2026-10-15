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

"""Deterministic simulation-time scheduling for pub/sub components.

`pubsub.Publisher` runs its computation on a ReactiveX scheduler; by default a
`ThreadPoolScheduler`, so a slow mind finishes whenever the language model
answers. That is the right behaviour for live play, but it makes runs
irreproducible and tests timing-dependent.

`SimTimeScheduler` is a virtual-time scheduler whose clock counts simulation
steps ("ticks"). Work runs only when the game advances the scheduler, on the
advancing thread, in a deterministic order: by due tick, then by a caller
supplied key, then by submission order. `LatencyScheduler` is a view that
delays work by a fixed number of ticks, so latency becomes a game parameter:

```python
sim = sim_time.SimTimeScheduler()
strategy = pubsub.Publisher(
    component=plan_component,
    scheduler=sim.latency(3, key='Aurelia strategy'),
)
...
# Once per simulation step, e.g. in the game master's NEXT_ACTING handler:
sim.advance()
```

A deliberation triggered at tick `t` then completes exactly when the scheduler
reaches tick `t + 3`, whatever the wall-clock speed of the model. Ticks map to
seconds in ReactiveX's time representation (tick `n` is `n` seconds after the
epoch), so standard time-based operators also run in simulation time.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import datetime
import heapq
import math
import threading
from typing import Any, TypeVar, override

from concordia.components.agent import pubsub
from concordia.typing import entity_component
from reactivex import abc as reactivex_abc
from reactivex import typing as reactivex_typing
from reactivex.scheduler import periodicscheduler
from reactivex.scheduler import scheduleditem
from reactivex.scheduler import scheduler as reactivex_scheduler

_TState = TypeVar('_TState')


@dataclasses.dataclass(order=True)
class _Entry:
  due: int
  key: str
  seq: int
  item: scheduleditem.ScheduledItem = dataclasses.field(compare=False)


def ticks_of(duetime: reactivex_typing.RelativeTime) -> int:
  """Converts a ReactiveX relative time (seconds or timedelta) to whole ticks.

  Args:
    duetime: A relative time. One second is one tick; fractions round up.

  Returns:
    A non-negative number of ticks.
  """
  if isinstance(duetime, datetime.timedelta):
    seconds = duetime.total_seconds()
  else:
    seconds = float(duetime)
  return max(0, math.ceil(seconds - 1e-9))


class SimTimeScheduler(periodicscheduler.PeriodicScheduler):
  """A ReactiveX scheduler driven by simulation ticks.

  Thread-safe for scheduling: entities may trigger publishers from engine
  worker threads. Work runs only inside `advance`, `advance_to` or `run_due`,
  on the calling thread.
  """

  def __init__(self, start_tick: int = 0):
    """Initializes the scheduler.

    Args:
      start_tick: Initial value of the tick counter.
    """
    super().__init__()
    self._tick = int(start_tick)
    self._queue: list[_Entry] = []
    self._seq = 0
    self._executed = 0
    self._running = False
    self._queue_lock = threading.Lock()

  @property
  def tick(self) -> int:
    """The current simulation tick."""
    with self._queue_lock:
      return self._tick

  @property
  def now(self) -> datetime.datetime:
    """The current tick in ReactiveX time (tick n is n seconds past epoch)."""
    return self.to_datetime(float(self.tick))

  @property
  def executed(self) -> int:
    """Total number of actions run so far."""
    with self._queue_lock:
      return self._executed

  def pending(self, key: str | None = None) -> int:
    """Returns the number of queued, uncancelled actions (optionally by key)."""
    with self._queue_lock:
      return sum(
          1
          for e in self._queue
          if not e.item.is_cancelled() and (key is None or e.key == key)
      )

  def due(self, key: str | None = None) -> list[int]:
    """Returns the due ticks of queued, uncancelled actions, earliest first.

    Args:
      key: Only actions scheduled under this key (e.g. one slow mind's
        `LatencyScheduler`); None for all.

    Returns:
      When each such action will run: what is "in flight", and when it lands.
    """
    with self._queue_lock:
      return sorted(
          e.due
          for e in self._queue
          if not e.item.is_cancelled() and (key is None or e.key == key)
      )

  def next_due(self) -> int | None:
    """Returns the earliest due tick of queued work, or None if idle."""
    with self._queue_lock:
      live = [e.due for e in self._queue if not e.item.is_cancelled()]
      return min(live) if live else None

  def schedule_at(
      self,
      due_tick: int,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
      *,
      key: str = '',
      scheduler: reactivex_scheduler.Scheduler | None = None,
  ) -> reactivex_abc.DisposableBase:
    """Queues an action for a tick.

    Args:
      due_tick: Tick at which the action runs. Past ticks run at the next
        `advance` or `run_due`.
      action: The ReactiveX action.
      state: Optional state passed to the action.
      key: Tie-breaker among actions due on the same tick. Use a stable name
        (e.g. the publishing entity) so that runs are reproducible when actions
        are submitted from parallel threads.
      scheduler: Scheduler passed to the action (defaults to this one).

    Returns:
      A disposable that cancels the action if it has not yet run.
    """
    item = scheduleditem.ScheduledItem(
        scheduler or self, state, action, self.to_datetime(float(due_tick))
    )
    with self._queue_lock:
      heapq.heappush(self._queue, _Entry(int(due_tick), key, self._seq, item))
      self._seq += 1
    return item.disposable

  def schedule(
      self,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    """Queues an action for the current tick (it runs at the next advance)."""
    return self.schedule_at(self.tick, action, state)

  def schedule_relative(
      self,
      duetime: reactivex_typing.RelativeTime,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    """Queues an action `duetime` ticks (seconds) from now."""
    return self.schedule_at(self.tick + ticks_of(duetime), action, state)

  def schedule_absolute(
      self,
      duetime: reactivex_typing.AbsoluteTime,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    """Queues an action for an absolute tick (seconds past epoch)."""
    due = math.ceil(self.to_seconds(duetime) - 1e-9)
    return self.schedule_at(due, action, state)

  def _run_until(self, target: int) -> int:
    with self._queue_lock:
      if self._running:
        raise RuntimeError('SimTimeScheduler cannot be advanced reentrantly.')
      if target < self._tick:
        raise ValueError(f'Cannot move from tick {self._tick} back to {target}')
      self._running = True
    count = 0
    try:
      while True:
        with self._queue_lock:
          if not self._queue or self._queue[0].due > target:
            break
          entry = heapq.heappop(self._queue)
          self._tick = max(self._tick, entry.due)
        if not entry.item.is_cancelled():
          entry.item.invoke()
          count += 1
    finally:
      with self._queue_lock:
        self._tick = target
        self._executed += count
        self._running = False
    return count

  def advance(self, ticks: int = 1) -> int:
    """Moves the clock forward and runs everything due up to the new tick.

    Work due on the current tick (for example, scheduled with `schedule`
    since the last advance) also runs. Actions scheduled while advancing run
    in the same call if they are due by the target tick.

    Args:
      ticks: Number of ticks to advance (zero runs due work only).

    Returns:
      The number of actions run.
    """
    if ticks < 0:
      raise ValueError('ticks must be non-negative')
    return self._run_until(self.tick + int(ticks))

  def advance_to(self, tick: int) -> int:
    """Advances to an absolute tick; see `advance`."""
    return self._run_until(int(tick))

  def run_due(self) -> int:
    """Runs work due on or before the current tick without advancing."""
    return self._run_until(self.tick)

  def latency(self, ticks: int, key: str = '') -> 'LatencyScheduler':
    """Returns a view that delays work by `ticks`, ordered by `key`."""
    return LatencyScheduler(self, ticks, key=key)

  def get_state(self) -> dict[str, Any]:
    """Returns the clock for checkpointing.

    Queued actions are callables and are not serialized. Before saving,
    either advance until `pending()` is zero or accept that in-flight work is
    dropped (publishers then simply run again on their next trigger).
    """
    with self._queue_lock:
      return {'tick': self._tick, 'executed': self._executed}

  def set_state(self, state: dict[str, Any]) -> None:
    """Restores the clock from `get_state`."""
    with self._queue_lock:
      target_tick = int(state.get('tick', self._tick))
      target_executed = int(state.get('executed', self._executed))
      if (self._tick, self._executed) != (target_tick, target_executed):
        for entry in self._queue:
          entry.item.disposable.dispose()
        self._queue.clear()
      self._tick = target_tick
      self._executed = target_executed

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'SimTimeScheduler':
    del memo
    return self


class LatencyScheduler(reactivex_scheduler.Scheduler):
  """A view on a `SimTimeScheduler` that delays work by a number of ticks.

  Pass it as `scheduler=` to `pubsub.Publisher`. The latency can change at run
  time (`set_latency`), for example when a faster communication technology is
  researched; work already queued keeps its original due tick.
  """

  def __init__(self, base: SimTimeScheduler, ticks: int, key: str = ''):
    """Initializes the view.

    Args:
      base: The scheduler that owns the clock and the queue.
      ticks: Delay applied to every scheduled action.
      key: Tie-breaker for actions due on the same tick (see
        `SimTimeScheduler.schedule_at`).
    """
    super().__init__()
    if ticks < 0:
      raise ValueError('Latency must be non-negative.')
    self._base = base
    self._ticks = int(ticks)
    self._key = key
    self._latency_lock = threading.Lock()

  @property
  def base(self) -> SimTimeScheduler:
    return self._base

  @property
  def key(self) -> str:
    return self._key

  @property
  def latency_ticks(self) -> int:
    with self._latency_lock:
      return self._ticks

  def set_latency(self, ticks: int) -> None:
    """Changes the delay applied to future work."""
    if ticks < 0:
      raise ValueError('Latency must be non-negative.')
    with self._latency_lock:
      self._ticks = int(ticks)

  @property
  def now(self) -> datetime.datetime:
    return self._base.now

  def _schedule_in(
      self,
      extra: int,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None,
  ) -> reactivex_abc.DisposableBase:
    due = self._base.tick + self.latency_ticks + extra
    return self._base.schedule_at(
        due, action, state, key=self._key, scheduler=self
    )

  def schedule(
      self,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    return self._schedule_in(0, action, state)

  def schedule_relative(
      self,
      duetime: reactivex_typing.RelativeTime,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    return self._schedule_in(ticks_of(duetime), action, state)

  def schedule_absolute(
      self,
      duetime: reactivex_typing.AbsoluteTime,
      action: reactivex_abc.ScheduledAction[_TState],
      state: _TState | None = None,
  ) -> reactivex_abc.DisposableBase:
    due = math.ceil(self.to_seconds(duetime) - 1e-9)
    return self._base.schedule_at(
        due, action, state, key=self._key, scheduler=self
    )

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'LatencyScheduler':
    del memo
    return self


def publisher_scheduler(
    base: SimTimeScheduler | None, ticks: int, key: str
) -> reactivex_abc.SchedulerBase | None:
  """Returns a latency view, or None (Publisher default) without a base.

  Lets one code path build publishers for both time modes: sim-time when a
  `SimTimeScheduler` is given, real asynchronous threads otherwise.

  Args:
    base: The sim-time scheduler, or None for real asynchronous time.
    ticks: Latency in ticks when `base` is given.
    key: Tie-breaker key when `base` is given.
  """
  if base is None:
    return None
  return base.latency(ticks, key=key)


class SimTimePublisher(pubsub.Publisher):
  """A `pubsub.Publisher` that checkpoints in-flight `SimTimeScheduler` tasks."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._pending_tasks: dict[int, dict[str, Any]] = {}

  def _is_virtual_scheduler(self) -> bool:
    return isinstance(self._scheduler, (SimTimeScheduler, LatencyScheduler))

  def _schedule_virtual_task(
      self,
      sequence_id: int,
      snapshot: Mapping[str, str],
      due_tick: int | None = None,
  ) -> None:
    if due_tick is None:
      if isinstance(self._scheduler, SimTimeScheduler):
        due_tick = self._scheduler.tick
      elif isinstance(self._scheduler, LatencyScheduler):
        due_tick = self._scheduler.base.tick + self._scheduler.latency_ticks
      else:
        due_tick = 0
    snap_copy = dict(snapshot)
    with self._condition_lock:
      self._pending_tasks[sequence_id] = {
          'sequence_id': sequence_id,
          'due_tick': due_tick,
          'snapshot': snap_copy,
          'disposable': None,
      }

    def action(scheduler: Any, state: Any) -> None:
      del scheduler, state
      with self._condition_lock:
        if sequence_id not in self._pending_tasks:
          return
        self._pending_tasks.pop(sequence_id, None)
      try:
        result = self._compute(sequence_id, snap_copy)
        if result is not None:
          self._publish_if_latest(result)
      finally:
        self._on_task_completed()

    if isinstance(self._scheduler, SimTimeScheduler):
      disposable = self._scheduler.schedule_at(due_tick, action)
    elif isinstance(self._scheduler, LatencyScheduler):
      disposable = self._scheduler.base.schedule_at(
          due_tick,
          action,
          key=self._scheduler.key,
          scheduler=self._scheduler,
      )
    else:
      disposable = None
    with self._condition_lock:
      if sequence_id in self._pending_tasks:
        self._pending_tasks[sequence_id]['disposable'] = disposable

  @override
  def close(self) -> None:
    with self._condition_lock:
      old_tasks = list(self._pending_tasks.values())
      self._pending_tasks.clear()
      self._in_flight = 0
      self._condition_lock.notify_all()
    for task in old_tasks:
      disp = task.get('disposable')
      if disp is not None:
        disp.dispose()
    super().close()

  @override
  def _trigger_computation(self, *, from_observe: bool = False) -> str:
    if not self._is_virtual_scheduler():
      return super()._trigger_computation(from_observe=from_observe)

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
      self._schedule_virtual_task(sequence_id, snapshot)
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
  def get_state(self) -> entity_component.ComponentState:
    state = dict(super().get_state())
    with self._condition_lock:
      state['compute_count'] = self._compute_count
      state['next_sequence_id'] = self._next_sequence_id
      state['latest_completed_sequence_id'] = self._latest_completed_sequence_id
      if self._pending_tasks:
        state['in_flight_tasks'] = [
            {
                'sequence_id': t['sequence_id'],
                'due_tick': t['due_tick'],
                'snapshot': dict(t['snapshot']),
            }
            for _, t in sorted(self._pending_tasks.items())
        ]
    if isinstance(self._scheduler, SimTimeScheduler):
      state['scheduler_state'] = self._scheduler.get_state()
    elif isinstance(self._scheduler, LatencyScheduler):
      state['scheduler_state'] = self._scheduler.base.get_state()
      state['latency_ticks'] = self._scheduler.latency_ticks
    return state

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    sched_state = state.get('scheduler_state')
    if isinstance(sched_state, Mapping):
      if isinstance(self._scheduler, SimTimeScheduler):
        self._scheduler.set_state(dict(sched_state))
      elif isinstance(self._scheduler, LatencyScheduler):
        self._scheduler.base.set_state(dict(sched_state))
    latency_ticks = state.get('latency_ticks')
    if isinstance(latency_ticks, (int, float)) and isinstance(
        self._scheduler, LatencyScheduler
    ):
      self._scheduler.set_latency(int(latency_ticks))
    super().set_state(state)
    with self._condition_lock:
      old_tasks = list(self._pending_tasks.values())
      self._pending_tasks.clear()
      self._in_flight = 0
      compute_count = state.get('compute_count')
      if isinstance(compute_count, (int, float)):
        self._compute_count = int(compute_count)
      next_seq = state.get('next_sequence_id')
      if isinstance(next_seq, (int, float)):
        self._next_sequence_id = int(next_seq)
      latest_seq = state.get('latest_completed_sequence_id')
      if isinstance(latest_seq, (int, float)):
        self._latest_completed_sequence_id = int(latest_seq)
    for task in old_tasks:
      disp = task.get('disposable')
      if disp is not None:
        disp.dispose()
    in_flight = state.get('in_flight_tasks')
    if (
        self._is_virtual_scheduler()
        and isinstance(in_flight, Sequence)
        and not isinstance(in_flight, str)
    ):
      for task_data in in_flight:
        if not isinstance(task_data, Mapping):
          continue
        seq_raw = task_data.get('sequence_id')
        due_raw = task_data.get('due_tick')
        if not isinstance(seq_raw, (int, float)) or not isinstance(
            due_raw, (int, float)
        ):
          continue
        seq_id = int(seq_raw)
        due_tick = int(due_raw)
        snap_raw = task_data.get('snapshot', {})
        snap = (
            {str(k): str(v) for k, v in snap_raw.items()}
            if isinstance(snap_raw, Mapping)
            else {}
        )
        with self._condition_lock:
          self._in_flight += 1
          self._next_sequence_id = max(self._next_sequence_id, seq_id)
        self._schedule_virtual_task(seq_id, snap, due_tick=due_tick)

  def __deepcopy__(
      self, memo: dict[int, Any] | None = None
  ) -> 'SimTimePublisher':
    comp_copy = (
        copy.deepcopy(self._component, memo)
        if self._component is not None
        else None
    )
    return SimTimePublisher(
        function=self._function if comp_copy is None else None,
        component=comp_copy,
        pre_act_label=self.get_pre_act_label(),
        skip_if_busy=self._skip_if_busy,
        trigger_on_observe=self._trigger_on_observe,
        subject=self._subject,
        scheduler=self._scheduler,
    )
