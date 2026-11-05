# Copyright 2023 DeepMind Technologies Limited.
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

"""A component that ignores the action spec in the `pre_act` method."""

import abc
import copy
import threading
from typing import Any, Final, override

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ActionSpecIgnored(
    entity_component.ContextComponent, metaclass=abc.ABCMeta
):
  """A component that ignores the action spec in the `pre_act` method.

  As a consequence, its `pre_act` state can be accessed safely by other
  components. This is useful for components that need to condition their
  `pre_act` state on the state of other components. Derived classes should
  implement `_make_pre_act_value` and `pre_act_label` instead of
  `pre_act`. The pre_act context will be constructed as `f'{key}: {value}'`
  (or as the bare value, if the component is made with
  `prefix_pre_act=False`); derived classes may change this by overriding
  `_format_pre_act`. This will be cached and cleaned up in `update`.
  """

  def __init__(self, pre_act_label: str, *, prefix_pre_act: bool = True):
    """Initializes the component.

    Args:
      pre_act_label: The label of the component's pre-act value, used as a
        prefix of the context returned by `pre_act` and to name the value
        elsewhere (e.g. in logs and in other components' prompts).
      prefix_pre_act: Whether `pre_act` returns the value prefixed by its
        label (the label, a colon and a newline, then the value and a final
        newline). If False, `pre_act` returns the value exactly as made, e.g.
        for a component whose context is read as a result (such as a game
        master's next game master or termination decision) rather than as part
        of a prompt.
    """
    super().__init__()
    self._pre_act_value: str | None = None
    self._pre_act_label: Final[str] = pre_act_label
    self._prefix_pre_act: Final[bool] = prefix_pre_act
    self._lock: threading.Lock = threading.Lock()

  @abc.abstractmethod
  def _make_pre_act_value(self) -> str:
    """Creates the pre-act value."""
    raise NotImplementedError()

  def get_pre_act_value(self) -> str:
    """Gets the pre-act value.

    Returns:
      The pre-act value, as created by `_make_pre_act_value`. Note that
      `pre_act` returns a string of the form f'{key}: {value}'.

    Raises:
      ValueError: If the entity is not in the `PRE_ACT` or `POST_ACT` phase.
    """
    if (
        self.get_entity().get_phase() != entity_component.Phase.PRE_ACT
        and self.get_entity().get_phase() != entity_component.Phase.POST_ACT
    ):
      raise ValueError(
          "You can only access the pre-act value in the `PRE_ACT` or "
          "`POST_ACT` phase. The entity is currently in the "
          f"{self.get_entity().get_phase()} phase."
      )

    with self._lock:
      if self._pre_act_value is None:
        self._pre_act_value = self._make_pre_act_value()
      return self._pre_act_value

  def get_pre_act_label(self) -> str:
    """Returns the key used as a prefix in the string returned by `pre_act`."""
    return self._pre_act_label

  def _format_pre_act(self, label: str, value: str) -> str:
    """Formats the context that `pre_act` returns.

    By default this is the label, a colon and a newline, then the value and a
    final newline; or the value alone if the component was made with
    `prefix_pre_act=False`. Derived classes may override it to present their
    context differently.

    Args:
      label: The component's pre-act label.
      value: The component's pre-act value.

    Returns:
      The context returned by `pre_act`.
    """
    if not self._prefix_pre_act:
      return value
    return f"{label}:\n{value}\n"

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return self._format_pre_act(
        self.get_pre_act_label(), self.get_pre_act_value()
    )

  def update(self) -> None:
    with self._lock:
      self._pre_act_value = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(component_name, type_=ActionSpecIgnored)
        .get_pre_act_value()
    )

  @override
  def set_state(self, state: entity_component.ComponentState) -> Any:
    return None

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Any:
    cls = self.__class__
    result = cls.__new__(cls)
    if memo is not None:
      memo[id(self)] = result
    lock_type = type(threading.Lock())
    rlock_type = type(threading.RLock())
    for k, v in self.__dict__.items():
      if k == "_entity":
        setattr(result, k, None)
      elif isinstance(v, lock_type):
        setattr(result, k, threading.Lock())
      elif isinstance(v, rlock_type):
        setattr(result, k, threading.RLock())
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result
