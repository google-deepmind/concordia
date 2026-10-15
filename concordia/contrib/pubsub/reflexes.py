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

"""Reflexes: standing orders as declarative, allowlisted ReactiveX pipelines.

A slow mind (or a person) writes a reflex as plain data; this module checks it
against an allowlist of operators and installs it on a `TopicBus`, where it
runs at every message with no language model call:

```python
spec = reflexes.ReflexSpec.from_dict({
    'name': 'sell_ore',
    'input': 'civ.census',
    'pipeline': [
        {'op': 'json_field', 'path': 'polities.basalt.stocks.ore'},
        {'op': 'threshold', 'above': 6},
    ],
    'emit': {'topic': 'civ.basalt.orders', 'template': 'sell ore ({value:g})'},
    'cooldown': 3,
})
engine = reflexes.Reflexes(bus)
engine.install(spec, owner='Basalt clans')
```

No code is evaluated: every stage is a named operator from `OPERATORS` whose
parameters are validated. A caller may restrict which operators an owner may
use (for example, operators unlocked by research). The engine counts every
reflex's fires and remembers its last value, and it checkpoints the installed
specs so a restored run re-installs them.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
import copy
import dataclasses
import json
import threading
from typing import Any, override

from concordia.components.agent import action_spec_ignored
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import topics
import reactivex
from reactivex import operators as ops

DEFAULT_REFLEXES_COMPONENT_KEY = '__reflexes__'
DEFAULT_REFLEXES_PRE_ACT_LABEL = 'Reflexes'

Operator = Callable[[reactivex.Observable], reactivex.Observable]


def _number(value: Any) -> float:
  return float(value)


def _threshold(params: Mapping[str, Any]) -> Operator:
  """Passes numeric values strictly above `above` and/or below `below`."""
  above, below = params.get('above'), params.get('below')
  if above is None and below is None:
    raise ValueError('threshold needs `above` or `below`')

  def passes(value: Any) -> bool:
    number = _number(value)
    if above is not None and not number > float(above):
      return False
    if below is not None and not number < float(below):
      return False
    return True

  op = ops.filter(passes)
  return lambda source: operators.attach_hooks(source, op(source))


def _moving_average(params: Mapping[str, Any]) -> Operator:
  """Emits the mean of the last `n` numeric values."""
  n = int(params['n'])
  if n < 1:
    raise ValueError('moving_average needs n >= 1')

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    window: list[float] = []

    def on_subscribe(observer, scheduler=None):
      def on_next(value: Any) -> None:
        window.append(_number(value))
        del window[:-n]
        observer.on_next(sum(window) / len(window))

      return source.subscribe(
          on_next, observer.on_error, observer.on_completed, scheduler=scheduler
      )

    def get_state() -> dict[str, Any]:
      return {'op': 'moving_average', 'window': list(window)}

    def set_state(saved: Mapping[str, Any]) -> None:
      items = saved.get('window')
      if isinstance(items, Sequence) and not isinstance(items, str):
        window[:] = [float(x) for x in items][-n:]

    return operators.attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


def _change(params: Mapping[str, Any]) -> Operator:
  """Emits the difference between each numeric value and the previous one."""
  del params

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    prev: list[float] = []

    def on_subscribe(observer, scheduler=None):
      def on_next(value: Any) -> None:
        num = _number(value)
        if prev:
          delta = num - prev[0]
          prev[:] = [num]
          observer.on_next(delta)
        else:
          prev[:] = [num]

      return source.subscribe(
          on_next, observer.on_error, observer.on_completed, scheduler=scheduler
      )

    def get_state() -> dict[str, Any]:
      return {'op': 'change', 'prev': list(prev)}

    def set_state(saved: Mapping[str, Any]) -> None:
      items = saved.get('prev')
      if isinstance(items, Sequence) and not isinstance(items, str):
        prev[:] = [float(x) for x in items]

    return operators.attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


def _scale(params: Mapping[str, Any]) -> Operator:
  """Multiplies numeric values by `factor`."""
  factor = float(params['factor'])
  op = ops.map(lambda value: _number(value) * factor)
  return lambda source: operators.attach_hooks(source, op(source))


def _every(params: Mapping[str, Any]) -> Operator:
  """Emits every `n`-th item."""
  n = int(params['n'])
  if n < 1:
    raise ValueError('every needs n >= 1')

  def subscribe(source: reactivex.Observable) -> reactivex.Observable:
    count = [0]

    def on_subscribe(observer, scheduler=None):
      def on_next(value: Any) -> None:
        count[0] += 1
        if count[0] % n == 0:
          observer.on_next(value)

      return source.subscribe(
          on_next, observer.on_error, observer.on_completed, scheduler=scheduler
      )

    def get_state() -> dict[str, Any]:
      return {'op': 'every', 'count': count[0]}

    def set_state(saved: Mapping[str, Any]) -> None:
      if 'count' in saved and saved['count'] is not None:
        count[0] = int(saved['count'])

    return operators.attach_hooks(
        source, reactivex.create(on_subscribe), (get_state, set_state)
    )

  return subscribe


# Operator name -> (required params, optional params, builder).
OPERATORS: Mapping[
    str,
    tuple[
        frozenset[str],
        frozenset[str],
        Callable[[Mapping[str, Any]], Operator],
    ],
] = {
    'json_field': (
        frozenset({'path'}),
        frozenset(),
        lambda p: operators.json_field(str(p['path'])),
    ),
    'threshold': (frozenset(), frozenset({'above', 'below'}), _threshold),
    'deadband': (
        frozenset({'threshold'}),
        frozenset(),
        lambda p: operators.deadband(float(p['threshold']), key=_number),
    ),
    'moving_average': (frozenset({'n'}), frozenset(), _moving_average),
    'change': (frozenset(), frozenset(), _change),
    'scale': (frozenset({'factor'}), frozenset(), _scale),
    'every': (frozenset({'n'}), frozenset(), _every),
}


@dataclasses.dataclass(frozen=True)
class ReflexSpec:
  """A validated reflex.

  Attributes:
    name: Unique name (per engine).
    input: Topic whose messages feed the pipeline.
    pipeline: Stages, each `{'op': <name in OPERATORS>, ...params}`. A message
      enters as its payload unless the first stage is `json_field`.
    emit_topic: Topic the reflex publishes to when a value reaches the end.
    template: Text published, formatted with `{value}` and `{name}`.
    cooldown: Minimum bus time between two fires (0: no limit).
  """

  name: str
  input: str
  pipeline: tuple[Mapping[str, Any], ...]
  emit_topic: str
  template: str = '{value}'
  cooldown: float = 0.0

  @classmethod
  def from_dict(
      cls,
      data: Mapping[str, Any],
      allowed_ops: Collection[str] | None = None,
  ) -> 'ReflexSpec':
    """Validates a reflex written as data (e.g. parsed from JSON).

    Args:
      data: `{name, input, pipeline, emit: {topic, template}, cooldown}`.
      allowed_ops: Operators this author may use (None: all of `OPERATORS`).

    Returns:
      The spec.

    Raises:
      ValueError: If the reflex is malformed or uses a forbidden operator.
    """
    if not isinstance(data, Mapping):
      raise ValueError('a reflex must be an object')
    unknown = set(data) - {'name', 'input', 'pipeline', 'emit', 'cooldown'}
    if unknown:
      raise ValueError(f'unknown reflex fields: {sorted(unknown)}')
    name, source = data.get('name'), data.get('input')
    if not isinstance(name, str) or not name:
      raise ValueError('a reflex needs a name')
    if not isinstance(source, str) or not source:
      raise ValueError('a reflex needs an input topic')
    emit = data.get('emit')
    if not isinstance(emit, Mapping) or not isinstance(emit.get('topic'), str):
      raise ValueError('a reflex needs emit.topic')
    stages = data.get('pipeline', ())
    if not isinstance(stages, Sequence) or isinstance(stages, str):
      raise ValueError('pipeline must be a list')
    checked = []
    for stage in stages:
      if not isinstance(stage, Mapping) or stage.get('op') not in OPERATORS:
        raise ValueError(f'unknown operator in {stage!r}')
      op = stage['op']
      if allowed_ops is not None and op not in allowed_ops:
        raise ValueError(f'operator {op!r} is not allowed here')
      required, optional, builder = OPERATORS[op]
      params = {k: v for k, v in stage.items() if k != 'op'}
      missing = required - set(params)
      extra = set(params) - required - optional
      if missing or extra:
        raise ValueError(
            f'{op}: missing {sorted(missing)}, unexpected {sorted(extra)}'
        )
      builder(params)  # Raises on bad values.
      checked.append(dict(stage))
    cooldown = float(data.get('cooldown', 0.0))
    if cooldown < 0:
      raise ValueError('cooldown must be non-negative')
    return cls(
        name=name,
        input=source,
        pipeline=tuple(checked),
        emit_topic=str(emit['topic']),
        template=str(emit.get('template', '{value}')),
        cooldown=cooldown,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        'name': self.name,
        'input': self.input,
        'pipeline': [dict(s) for s in self.pipeline],
        'emit': {'topic': self.emit_topic, 'template': self.template},
        'cooldown': self.cooldown,
    }


def parse_specs(
    text: str, allowed_ops: Collection[str] | None = None
) -> tuple[list[ReflexSpec], list[str]]:
  """Parses reflexes from free text containing a JSON object or list.

  Language models often wrap JSON in prose; the first `[`...`]` or `{`...`}`
  span is used.

  Args:
    text: The text.
    allowed_ops: As for `ReflexSpec.from_dict`.

  Returns:
    (valid specs, error messages for the rejected ones).
  """
  starts = [i for i in (text.find('['), text.find('{')) if i >= 0]
  if not starts:
    return [], ['no JSON found']
  start = min(starts)
  end = max(text.rfind(']'), text.rfind('}'))
  try:
    data = json.loads(text[start : end + 1])
  except json.JSONDecodeError as e:
    return [], [f'invalid JSON: {e.msg}']
  items = data if isinstance(data, list) else [data]
  specs, errors = [], []
  for item in items:
    try:
      specs.append(ReflexSpec.from_dict(item, allowed_ops))
    except ValueError as e:
      errors.append(str(e))
  return specs, errors


@dataclasses.dataclass
class _Installed:
  spec: ReflexSpec
  owner: str
  subscription: Any
  hooks: list[operators.StateHook] = dataclasses.field(default_factory=list)
  fires: int = 0
  last_value: Any = None
  last_fired: float | None = None


class Reflexes(action_spec_ignored.ActionSpecIgnored):
  """Installs reflexes on a TopicBus and keeps their statistics."""

  def __init__(
      self,
      bus: topics.TopicBus,
      *,
      pre_act_label: str = DEFAULT_REFLEXES_PRE_ACT_LABEL,
  ):
    super().__init__(pre_act_label)
    self._bus = bus
    self._installed: dict[str, _Installed] = {}
    self._reflexes_lock = threading.RLock()
    self._bus.register_reflex_engine(self)

  def install(self, spec: ReflexSpec, owner: str = '') -> None:
    """Installs a reflex, replacing any reflex with the same name."""
    with self._reflexes_lock:
      self.remove(spec.name)
      stages = [OPERATORS[s['op']][2](s) for s in spec.pipeline]
      first_is_field = bool(spec.pipeline) and (
          spec.pipeline[0]['op'] == 'json_field'
      )
      stream = self._bus.observable(spec.input).pipe(
          ops.filter(lambda _: not self._bus.is_restoring)
      )
      if not first_is_field:
        stream = stream.pipe(ops.map(topics.payload))
      hooks: list[operators.StateHook] = []
      for stage in stages:
        stream = stream.pipe(stage)
        for h in getattr(stream, operators.HOOKS_ATTR, ()):
          if h not in hooks:
            hooks.append(h)
      entry = _Installed(spec=spec, owner=owner, subscription=None, hooks=hooks)
      entry.subscription = stream.subscribe(
          lambda value, entry=entry: self._fire(entry, value)
      )
      self._installed[spec.name] = entry

  def _fire(self, entry: _Installed, value: Any) -> None:
    if self._bus.is_restoring:
      return
    now = self._bus.now()
    with self._reflexes_lock:
      if self._installed.get(entry.spec.name) is not entry:
        return
      entry.last_value = value
      if (
          entry.spec.cooldown > 0
          and entry.last_fired is not None
          and now is not None
          and now - entry.last_fired < entry.spec.cooldown
      ):
        return
      entry.fires += 1
      entry.last_fired = now
    text = entry.spec.template.format(
        value=_display(value), name=entry.spec.name
    )
    self._bus.publish(
        entry.spec.emit_topic,
        text,
        source=entry.owner or f'reflex:{entry.spec.name}',
    )

  def remove(self, name: str) -> bool:
    """Uninstalls a reflex; returns whether it existed."""
    with self._reflexes_lock:
      entry = self._installed.pop(name, None)
    if entry is None:
      return False
    if entry.subscription is not None:
      entry.subscription.dispose()
    return True

  def names(self, owner: str | None = None) -> list[str]:
    with self._reflexes_lock:
      return sorted(
          n
          for n, e in self._installed.items()
          if owner is None or e.owner == owner
      )

  def stats(self, name: str) -> dict[str, Any]:
    """Returns fire count, last value, last fired time, and owner of `name`."""
    with self._reflexes_lock:
      entry = self._installed[name]
      return {
          'fires': entry.fires,
          'last_value': entry.last_value,
          'last_fired': entry.last_fired,
          'owner': entry.owner,
      }

  def describe(self, owner: str | None = None) -> str:
    """One line per installed reflex, for prompts and reports."""
    lines = []
    with self._reflexes_lock:
      entries = [self._installed[name] for name in self.names(owner)]
    for entry in entries:
      name = entry.spec.name
      lines.append(
          f'{name}: on {entry.spec.input} -> {entry.spec.emit_topic}'
          f' (fired {entry.fires} times)'
      )
    return '\n'.join(lines) if lines else 'No reflexes.'

  @override
  def _make_pre_act_value(self) -> str:
    return self.describe()

  @override
  def get_state(self) -> dict[str, Any]:
    with self._reflexes_lock:
      return {
          name: {
              'spec': e.spec.to_dict(),
              'owner': e.owner,
              'fires': e.fires,
              'last_value': e.last_value,
              'last_fired': e.last_fired,
              'pipeline_state': [get_fn() for get_fn, _ in e.hooks],
          }
          for name, e in self._installed.items()
      }

  @override
  def set_state(self, state: Mapping[str, Any]) -> None:
    """Replaces every installed reflex with the checkpointed ones."""
    for name in self.names():
      self.remove(name)
    for name, saved in state.items():
      if not isinstance(saved, Mapping) or 'spec' not in saved:
        continue
      self.install(ReflexSpec.from_dict(saved['spec']), saved.get('owner', ''))
      with self._reflexes_lock:
        entry = self._installed[name]
        entry.fires = int(saved.get('fires', 0))
        entry.last_value = saved.get('last_value')
        entry.last_fired = saved.get('last_fired')
        pipe_state = saved.get('pipeline_state')
        if isinstance(pipe_state, Sequence) and not isinstance(pipe_state, str):
          for (_, set_fn), st in zip(entry.hooks, pipe_state):
            if isinstance(st, Mapping):
              set_fn(st)

  def close(self) -> None:
    for name in self.names():
      self.remove(name)

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> 'Reflexes':
    self.close()
    copied = Reflexes(self._bus, pre_act_label=self.get_pre_act_label())
    copied.set_state(copy.deepcopy(self.get_state(), memo))
    return copied


def _display(value: Any) -> Any:
  if isinstance(value, float) and value.is_integer():
    return int(value)
  return value
