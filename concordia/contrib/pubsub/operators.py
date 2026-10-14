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

"""ReactiveX operators that turn topic streams into prompt-ready beliefs.

A `pubsub.Subscriber` shows its latest string to the entity's prompt. These
operators build that string from `topics.Message` streams, composing standard
ReactiveX operators:

* `json_field`: pull one field out of a JSON payload.
* `with_age`: re-render a belief every tick with its age ("3 ticks old"), so
  entities can tell fresh readings from stale ones.
* `rolling_digest`: the last *n* messages as one block of text.
* `deadband`: emit only when a numeric value moves by at least a threshold,
  like a real instrument's alarm hysteresis.
* `combine_labelled`: several labelled beliefs as one line.
* `transport`: carry messages over an imperfect channel (latency, dropped
  messages, noise and bias on numeric fields), so beliefs can be late, missing
  or wrong.

Each function returns an operator usable in `Observable.pipe`, or as the
`transform` / `pipeline` arguments of `TopicBus.make_subscriber`:

```python
ticks = bus.observable('world.tick').pipe(ops.map(lambda m: m.time))
thermometer = bus.make_subscriber(
    'climate.global',
    transform=operators.with_age(ticks, render=render_climate),
)
```
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import json
import math
import random
import threading
from typing import Any, cast

from concordia.contrib.pubsub import topics
import reactivex
from reactivex import abc as reactivex_abc
from reactivex import operators as ops

Operator = Callable[[reactivex.Observable], reactivex.Observable]
StateHook = tuple[
    Callable[[], dict[str, Any]], Callable[[Mapping[str, Any]], None]
]
HOOKS_ATTR = '_concordia_state_hooks'

_SKIP = object()
_hook_tls = threading.local()


@contextlib.contextmanager
def collect_hooks() -> Iterator[list[StateHook]]:
  """Collects state hooks created while piping an observable stream."""
  stack: list[list[StateHook]] = getattr(_hook_tls, 'stack', None) or []
  _hook_tls.stack = stack
  bucket: list[StateHook] = []
  stack.append(bucket)
  try:
    yield bucket
  finally:
    stack.pop()


def attach_hooks(
    source: reactivex.Observable | None,
    out: reactivex.Observable,
    hook: StateHook | None = None,
) -> reactivex.Observable:
  """Attaches upstream and new state checkpoint hooks to `out`."""
  hooks: list[StateHook] = []
  if source is not None:
    hooks.extend(getattr(source, HOOKS_ATTR, ()))
  if hook is not None:
    hooks.append(hook)
  stack: list[list[StateHook]] | None = getattr(_hook_tls, 'stack', None)
  if stack:
    for h in hooks:
      if h not in stack[-1]:
        stack[-1].append(h)
  if hooks:
    setattr(out, HOOKS_ATTR, hooks)
  return out


def _payload_of(item: Any) -> str:
  return item.payload if isinstance(item, topics.Message) else str(item)


def lookup(data: Any, path: str) -> Any:
  """Returns `data[path]` for a dotted path; integers index lists.

  Args:
    data: Parsed JSON.
    path: Dotted path such as `'cover.white'` or `'items.0.name'`.

  Raises:
    KeyError: If the path does not exist.
  """
  current = data
  for part in path.split('.') if path else []:
    if isinstance(current, list):
      try:
        current = current[int(part)]
      except (ValueError, IndexError) as e:
        raise KeyError(path) from e
    elif isinstance(current, Mapping) and part in current:
      current = current[part]
    else:
      raise KeyError(path)
  return current


def json_field(path: str, default: Any = _SKIP) -> Operator:
  """Maps messages (or payload strings) to a field of their JSON payload.

  Args:
    path: Dotted path into the parsed payload (see `lookup`).
    default: Value emitted when the payload is not JSON or lacks the field. If
      omitted, such items are dropped.

  Returns:
    An operator.
  """

  def extract(item: Any) -> Any:
    try:
      return lookup(json.loads(_payload_of(item)), path)
    except (json.JSONDecodeError, KeyError, TypeError):
      return default

  composed = reactivex.compose(
      ops.map(extract), ops.filter(lambda value: value is not _SKIP)
  )
  return lambda source: attach_hooks(source, composed(source))


def age_text(age: float | None, unit: str = 'tick') -> str:
  """Formats an age such as `'fresh'`, `'1 tick old'` or `'4 ticks old'`."""
  if age is None:
    return 'age unknown'
  if age <= 0:
    return 'fresh'
  value = f'{age:g}'
  return f'{value} {unit if age == 1 else unit + "s"} old'


def with_age(
    now: reactivex.Observable,
    render: Callable[[topics.Message], str] = topics.payload,
    *,
    template: str = '{text} [{age}]',
    unit: str = 'tick',
) -> Operator:
  """Renders each message with its age, re-rendering whenever `now` changes.

  Uses `combine_latest`, so a belief that stops being updated visibly ages:
  `"21.5 C [fresh]"` becomes `"21.5 C [3 ticks old]"` three ticks later.

  Args:
    now: Stream of the current time in the same units as `Message.time` (for a
      `TopicBus` whose clock counts ticks, the times of `world.tick` messages).
      Until it emits, ages are unknown.
    render: Converts a message to text.
    template: Format with `{text}`, `{age}` (from `age_text`), `{age_value}` and
      `{time}` (the message time).
    unit: Unit name for `age_text`.

  Returns:
    An operator from `Message`s to strings.
  """

  def format_pair(pair: tuple[topics.Message, Any]) -> str:
    message, current = pair
    age = None
    if current is not None and message.time is not None:
      age = max(0.0, float(current) - float(message.time))
    return template.format(
        text=render(message),
        age=age_text(age, unit),
        age_value='' if age is None else f'{age:g}',
        time='' if message.time is None else f'{message.time:g}',
    )

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    state: dict[str, Any] = {
        'message': None,
        'has_now': False,
        'now': None,
        'last_emitted': None,
    }
    observer_ref: list[Any] = []

    def emit_if_changed() -> None:
      msg = state['message']
      if msg is None or not observer_ref:
        return
      current = state['now'] if state['has_now'] else None
      formatted = format_pair((msg, current))
      if formatted != state['last_emitted']:
        state['last_emitted'] = formatted
        observer_ref[0].on_next(formatted)

    def on_subscribe(observer, scheduler=None):
      observer_ref[:] = [observer]

      def on_message(msg: topics.Message) -> None:
        state['message'] = msg
        emit_if_changed()

      def on_now(current: Any) -> None:
        state['has_now'] = True
        state['now'] = current
        emit_if_changed()

      sub_now = now.subscribe(on_now, observer.on_error, scheduler=scheduler)
      sub_src = source.subscribe(
          on_message,
          observer.on_error,
          observer.on_completed,
          scheduler=scheduler,
      )

      def dispose() -> None:
        sub_src.dispose()
        sub_now.dispose()

      return dispose

    def get_state() -> dict[str, Any]:
      msg = state['message']
      return {
          'op': 'with_age',
          'message': msg.to_dict() if isinstance(msg, topics.Message) else None,
          'has_now': state['has_now'],
          'now': state['now'],
          'last_emitted': state['last_emitted'],
      }

    def set_state(saved: Mapping[str, Any]) -> None:
      msg_data = saved.get('message')
      state['message'] = (
          topics.Message.from_dict(msg_data)
          if isinstance(msg_data, Mapping)
          else None
      )
      state['has_now'] = bool(saved.get('has_now', False))
      state['now'] = saved.get('now')
      state['last_emitted'] = saved.get('last_emitted')

    return attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


def rolling_digest(
    n: int,
    render: Callable[[topics.Message], str] = topics.payload,
    *,
    separator: str = '\n',
    bullet: str = '- ',
) -> Operator:
  """Accumulates the last `n` rendered messages into one block of text."""
  if n < 1:
    raise ValueError('n must be at least 1')

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    window: list[str] = []

    def on_subscribe(observer, scheduler=None):
      def on_next(item: Any) -> None:
        text = render(item) if isinstance(item, topics.Message) else str(item)
        window.append(text)
        del window[:-n]
        observer.on_next(separator.join(bullet + t for t in window))

      return source.subscribe(
          on_next, observer.on_error, observer.on_completed, scheduler=scheduler
      )

    def get_state() -> dict[str, Any]:
      return {'op': 'rolling_digest', 'window': list(window)}

    def set_state(saved: Mapping[str, Any]) -> None:
      items = saved.get('window')
      if isinstance(items, Sequence) and not isinstance(items, str):
        window[:] = [str(x) for x in items][-n:]

    return attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


def deadband(threshold: float, key: Callable[[Any], float] = float) -> Operator:
  """Emits the first item, then only items that moved by at least `threshold`.

  The reference is the last *emitted* item, so slow drifts still trigger once
  their total change crosses the threshold.

  Args:
    threshold: Minimum absolute change of `key(item)` to emit.
    key: Numeric value of an item.

  Returns:
    An operator.
  """

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    last: list[float] = []

    def on_subscribe(observer, scheduler=None):
      def on_next(item: Any) -> None:
        value = key(item)
        if not last or abs(value - last[0]) >= threshold:
          last[:] = [value]
          observer.on_next(item)

      return source.subscribe(
          on_next, observer.on_error, observer.on_completed, scheduler=scheduler
      )

    def get_state() -> dict[str, Any]:
      return {'op': 'deadband', 'last': list(last)}

    def set_state(saved: Mapping[str, Any]) -> None:
      items = saved.get('last')
      if isinstance(items, Sequence) and not isinstance(items, str):
        last[:] = [float(x) for x in items]

    return attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


def combine_labelled(
    streams: Mapping[str, reactivex.Observable],
    *,
    separator: str = '; ',
    missing: str = '?',
) -> reactivex.Observable:
  """Combines labelled string streams into `"label: value; label: value"`.

  Args:
    streams: Label to stream of strings, in display order.
    separator: Text between entries.
    missing: Shown for a stream that has not emitted yet.

  Returns:
    A stream that emits whenever any input emits.
  """
  labels = list(streams)
  values: dict[str, str] = {label: missing for label in labels}

  def format_current() -> str:
    return separator.join(f'{label}: {values[label]}' for label in labels)

  def on_subscribe(observer, scheduler=None):
    if not labels:
      observer.on_completed()
      return lambda: None
    ready = [False]
    subs = []
    last_idx = len(labels) - 1
    for idx, label in enumerate(labels):

      def make_on_next(lbl: str, is_last: bool):
        def on_next(val: Any) -> None:
          values[lbl] = (
              val.payload if isinstance(val, topics.Message) else str(val)
          )
          if is_last:
            ready[0] = True
          if ready[0]:
            observer.on_next(format_current())

        return on_next

      subs.append(
          streams[label]
          .pipe(ops.start_with(missing))
          .subscribe(
              make_on_next(label, idx == last_idx),
              observer.on_error,
              scheduler=scheduler,
          )
      )

    def dispose() -> None:
      for s in subs:
        s.dispose()

    return dispose

  def get_state() -> dict[str, Any]:
    return {'op': 'combine_labelled', 'values': dict(values)}

  def set_state(saved: Mapping[str, Any]) -> None:
    saved_vals = saved.get('values')
    if isinstance(saved_vals, Mapping):
      for k in labels:
        if k in saved_vals:
          values[k] = str(saved_vals[k])

  out = reactivex.create(on_subscribe)
  hooks: list[StateHook] = []
  for s in streams.values():
    hooks.extend(getattr(s, HOOKS_ATTR, ()))
  hooks.append((get_state, set_state))
  setattr(out, HOOKS_ATTR, hooks)
  return out


@dataclasses.dataclass(frozen=True)
class Transport:
  """How messages travel from a topic to one subscriber.

  Attributes:
    latency: Delay in scheduler time units (ticks on a `SimTimeScheduler`).
    drop_rate: Probability that a message is lost.
    noise_sigma: Standard deviation of Gaussian noise added to each numeric
      field in `fields`.
    bias: Constant offset added to each numeric field in `fields`.
    fields: Dotted JSON paths (see `lookup`) that noise and bias apply to. Paths
      that are missing or not numeric are left alone.
    decimals: Rounding of perturbed values, or None to keep full precision.
    keep_late: Deliver a message even when a newer one arrived first (a news
      feed that wants every item); by default it is discarded, so a belief never
      goes backwards.
  """

  latency: float = 0.0
  drop_rate: float = 0.0
  noise_sigma: float = 0.0
  bias: float = 0.0
  fields: Sequence[str] = ()
  decimals: int | None = 3
  keep_late: bool = False

  def __post_init__(self):
    if self.latency < 0:
      raise ValueError('latency must be non-negative')
    if not 0.0 <= self.drop_rate <= 1.0:
      raise ValueError('drop_rate must be in [0, 1]')
    if self.noise_sigma < 0:
      raise ValueError('noise_sigma must be non-negative')


def _perturb_path(
    data: Any, path: str, change: Callable[[float], float]
) -> bool:
  """Applies `change` to the number at `path` in place; True if changed."""
  parts = path.split('.')
  try:
    parent = lookup(data, '.'.join(parts[:-1]))
  except KeyError:
    return False
  last = parts[-1]
  if isinstance(parent, list):
    try:
      index = int(last)
      value = parent[index]
    except (ValueError, IndexError):
      return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      parent[index] = change(float(value))
      return True
    return False
  if isinstance(parent, dict) and last in parent:
    value = parent[last]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      parent[last] = change(float(value))
      return True
  return False


def transport(
    spec: Transport | Callable[[], Transport] | Callable[[Any], Transport],
    *,
    seed: int = 0,
    scheduler: reactivex_abc.SchedulerBase | None = None,
    per_message: bool = False,
) -> Operator:
  """Carries `topics.Message`s over an imperfect channel.

  In order: a message may be dropped; its numeric `fields` get bias and noise
  (the payload is re-serialised and `time` is kept, so `with_age` shows how
  old the reading really is); then it is delivered `latency` later on
  `scheduler`. A message that arrives after a newer one (possible when the
  latency shrinks) is discarded, so a belief never goes backwards. Each
  subscription draws from its own `random.Random(seed)`, so a run with a
  deterministic publish order is reproducible.

  `spec` may be a callable returning the current `Transport`; it is read for
  every message, so a channel can be upgraded mid-run (a better instrument).
  With `per_message=True` the callable receives the message itself, so each
  message can travel differently, e.g. news that takes longer to arrive the
  farther away it happened:

  ```python
  def by_distance(message):
    away = distance(home, json.loads(message.payload)['where'])
    return operators.Transport(latency=away / speed, keep_late=True)

  local_news = bus.make_subscriber(
      ['local.r3', 'local.r4'],
      transform=operators.transport(
          by_distance, per_message=True, scheduler=sim_scheduler),
  )
  ```

  ```python
  thermometer = bus.make_subscriber(
      'climate.global',
      transform=reactivex.compose(
          operators.transport(
              operators.Transport(latency=2, noise_sigma=1.5,
                                  fields=('temp_c',)),
              scheduler=sim_scheduler,
          ),
          operators.with_age(ticks, render=render_climate),
      ),
  )
  ```

  Args:
    spec: The channel's properties, or a callable returning them.
    seed: Seed of the channel's random draws.
    scheduler: Scheduler for delayed deliveries; required whenever a latency is
      positive.
    per_message: Whether a callable `spec` takes each message as its argument
      (otherwise it takes none).

  Returns:
    An operator from messages to (possibly late, noisy) messages.
  """
  if per_message and not callable(spec):
    raise ValueError('per_message needs a callable spec')
  if not callable(spec) and spec.latency > 0 and scheduler is None:
    raise ValueError('a scheduler is required when latency > 0')

  def current(item: Any) -> Transport:
    if isinstance(spec, Transport):
      return spec
    if per_message:
      return cast(Callable[[Any], Transport], spec)(item)
    return cast(Callable[[], Transport], spec)()

  def corrupt(message: Any, rng: random.Random, channel: Transport) -> Any:
    perturbs = bool(channel.fields) and (
        channel.noise_sigma > 0 or channel.bias != 0
    )
    if not perturbs or not isinstance(message, topics.Message):
      return message
    try:
      data = json.loads(message.payload)
    except json.JSONDecodeError:
      return message

    def change(value: float) -> float:
      noisy = value + channel.bias
      if channel.noise_sigma > 0:
        noisy += rng.gauss(0.0, channel.noise_sigma)
      return (
          noisy if channel.decimals is None else round(noisy, channel.decimals)
      )

    changed = [_perturb_path(data, path, change) for path in channel.fields]
    if not any(changed):
      return message
    return dataclasses.replace(message, payload=json.dumps(data))

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    rng = random.Random(seed)
    lock = threading.Lock()
    state: dict[str, Any] = {
        'newest': None,
        'pending': 0,
        'completed': False,
        'next_id': 0,
    }
    pending_deliveries: dict[int, dict[str, Any]] = {}
    observer_ref: list[Any] = []

    def deliver(
        item: Any, keep_late: bool = False, delivery_id: int | None = None
    ) -> None:
      with lock:
        if delivery_id is not None:
          if delivery_id not in pending_deliveries:
            return
          pending_deliveries.pop(delivery_id, None)
        state['pending'] = max(0, state['pending'] - 1)
        order = item.seq if isinstance(item, topics.Message) else None
        stale = (
            not keep_late
            and order is not None
            and state['newest'] is not None
            and (order < state['newest'])
        )
        if order is not None and (
            state['newest'] is None or order > state['newest']
        ):
          state['newest'] = order
        finish = state['completed'] and state['pending'] == 0
      if not stale and observer_ref:
        observer_ref[0].on_next(item)
      if finish and observer_ref:
        observer_ref[0].on_completed()

    def deliver_later(
        item: Any, keep: bool, delivery_id: int
    ) -> Callable[..., None]:
      return lambda *_: deliver(item, keep, delivery_id)

    def schedule_delivery(
        item: Any,
        keep: bool,
        due_tick: int | None,
        latency: float | None = None,
    ) -> None:
      if scheduler is None:
        raise ValueError('a scheduler is required when latency > 0')
      with lock:
        state['pending'] += 1
        state['next_id'] += 1
        delivery_id = state['next_id']
        serialized_item = (
            item.to_dict() if isinstance(item, topics.Message) else item
        )
        entry: dict[str, Any] = {
            'id': delivery_id,
            'due_tick': due_tick,
            'keep_late': keep,
            'is_message': isinstance(item, topics.Message),
            'item': serialized_item,
            'disposable': None,
        }
        pending_deliveries[delivery_id] = entry

      action = deliver_later(item, keep, delivery_id)
      if due_tick is not None and hasattr(scheduler, 'schedule_at'):
        disposable = getattr(scheduler, 'schedule_at')(due_tick, action)
      elif due_tick is not None and hasattr(scheduler, 'base'):
        base = getattr(scheduler, 'base')
        key = getattr(scheduler, 'key', '')
        disposable = base.schedule_at(
            due_tick, action, key=key, scheduler=scheduler
        )
      else:
        delay = latency if latency is not None else 0.0
        disposable = scheduler.schedule_relative(delay, action)
      with lock:
        if delivery_id in pending_deliveries:
          pending_deliveries[delivery_id]['disposable'] = disposable

    def on_subscribe(observer, subscribe_scheduler=None):
      observer_ref[:] = [observer]

      def on_next(item: Any) -> None:
        channel = current(item)
        if channel.drop_rate > 0 and rng.random() < channel.drop_rate:
          return
        item = corrupt(item, rng, channel)
        keep = channel.keep_late
        if channel.latency <= 0:
          with lock:
            state['pending'] += 1
          deliver(item, keep)
          return
        if scheduler is None:
          raise ValueError('a scheduler is required when latency > 0')
        delay_ticks = max(0, math.ceil(float(channel.latency) - 1e-9))
        due_tick: int | None = None
        if hasattr(scheduler, 'tick'):
          due_tick = int(getattr(scheduler, 'tick')) + delay_ticks
        elif hasattr(scheduler, 'base') and hasattr(
            getattr(scheduler, 'base'), 'tick'
        ):
          base = getattr(scheduler, 'base')
          lat = int(getattr(scheduler, 'latency_ticks', 0))
          due_tick = int(base.tick) + lat + delay_ticks
        schedule_delivery(item, keep, due_tick, latency=channel.latency)

      def on_completed() -> None:
        with lock:
          state['completed'] = True
          finish = state['pending'] == 0
        if finish:
          observer.on_completed()

      return source.subscribe(
          on_next,
          observer.on_error,
          on_completed,
          scheduler=subscribe_scheduler,
      )

    def get_state() -> dict[str, Any]:
      with lock:
        ver, internal, gauss = rng.getstate()
        pending_list = [
            {
                'due_tick': e['due_tick'],
                'keep_late': e['keep_late'],
                'is_message': e['is_message'],
                'item': e['item'],
            }
            for _, e in sorted(pending_deliveries.items())
        ]
        result: dict[str, Any] = {
            'op': 'transport',
            'rng': [ver, list(internal), gauss],
            'newest': state['newest'],
            'pending_deliveries': pending_list,
        }
      if scheduler is not None:
        if hasattr(scheduler, 'get_state'):
          result['scheduler_state'] = getattr(scheduler, 'get_state')()
        elif hasattr(scheduler, 'base') and hasattr(
            getattr(scheduler, 'base'), 'get_state'
        ):
          result['scheduler_state'] = getattr(scheduler, 'base').get_state()
      return result

    def set_state(saved: Mapping[str, Any]) -> None:
      sched_state = saved.get('scheduler_state')
      if isinstance(sched_state, Mapping) and scheduler is not None:
        if hasattr(scheduler, 'set_state'):
          getattr(scheduler, 'set_state')(dict(sched_state))
        elif hasattr(scheduler, 'base') and hasattr(
            getattr(scheduler, 'base'), 'set_state'
        ):
          getattr(scheduler, 'base').set_state(dict(sched_state))
      with lock:
        old_entries = list(pending_deliveries.values())
        pending_deliveries.clear()
        state['pending'] = 0
        state['newest'] = saved.get('newest')
        rng_data = saved.get('rng')
        if (
            isinstance(rng_data, Sequence)
            and not isinstance(rng_data, str)
            and len(rng_data) == 3
        ):
          ver, internal, gauss = rng_data
          rng.setstate((
              int(ver),
              tuple(int(x) for x in internal),
              float(gauss) if gauss is not None else None,
          ))
      for entry in old_entries:
        disp = entry.get('disposable')
        if disp is not None:
          disp.dispose()
      pending_data = saved.get('pending_deliveries')
      if isinstance(pending_data, Sequence) and not isinstance(
          pending_data, str
      ):
        for d in pending_data:
          if not isinstance(d, Mapping):
            continue
          raw_item = d.get('item')
          if d.get('is_message') and isinstance(raw_item, Mapping):
            item = topics.Message.from_dict(raw_item)
          else:
            item = raw_item
          due = d.get('due_tick')
          due_tick = int(due) if due is not None else None
          schedule_delivery(item, bool(d.get('keep_late', False)), due_tick)

    return attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe
