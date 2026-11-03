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

"""A modular entity agent using the new component system."""

from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import functools
import threading
import traceback
import types
from typing import Any, cast, override

from absl import logging
from concordia.typing import entity
from concordia.typing import entity_component
from concordia.utils import concurrency

# Builds a context component from the name and the JSON-serializable spec it
# was added with (see `EntityAgent.add_component`).
ComponentFactory = Callable[
    [str, entity_component.ComponentState], entity_component.ContextComponent
]


class EntityAgent(entity_component.EntityWithComponents):
  """An agent that has its functionality defined by components.

  The agent has a set of components that define its functionality. The agent
  must have at least an ActComponent and an ObserveComponent. The agent will
  call the ActComponent's `act` method when it needs to act, and the
  ObservationComponent's `observe` method when they need to process an
  observation.

  Context components may be added and removed while the agent lives
  (`add_component`, `remove_component`), including by its own components
  during `act` or `observe`: a change requested during a call takes effect
  when that call ends, so the components never change in the middle of a
  phase.
  """

  def __init__(
      self,
      agent_name: str,
      act_component: entity_component.ActingComponent,
      context_components: Mapping[str, entity_component.ContextComponent] = (
          types.MappingProxyType({})
      ),
      component_factory: ComponentFactory | None = None,
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_components: The ContextComponents that will be used by the agent.
      component_factory: Optional. Rebuilds components added with a spec
        (`add_component`) when `set_state` restores a state that has them.
    """
    super().__init__()
    self._agent_name = agent_name
    self._control_lock = threading.Lock()
    self._phase_lock = threading.Lock()
    self._phase = entity_component.Phase.READY
    self._capture_key_by_thread: dict[int, str] = {}
    self._active_capture_key: str = agent_name

    self._act_component = act_component
    self._act_component.set_entity(self)

    self._context_components = dict(context_components)
    for component in self._context_components.values():
      component.set_entity(self)

    # The components given at construction (restorable if removed), the specs
    # of components added since (None: not restorable), and changes waiting
    # for the current act or observe call to end.
    self._component_factory = component_factory
    self._built_in_components = dict(self._context_components)
    self._added_specs: dict[str, entity_component.ComponentState | None] = {}
    self._changes_lock = threading.Lock()
    self._pending_changes: list[
        tuple[
            str,
            entity_component.ContextComponent | None,
            entity_component.ComponentState | None,
        ]
    ] = []

  @override
  @functools.cached_property
  def name(self) -> str:
    return self._agent_name

  @override
  def get_phase(self) -> entity_component.Phase:
    with self._phase_lock:
      return self._phase

  def _set_phase(self, phase: entity_component.Phase) -> None:
    with self._phase_lock:
      self._phase.check_successor(phase)
      self._phase = phase

  @override
  def get_component(
      self,
      name: str,
      *,
      type_: type[entity_component.ComponentT] = entity_component.BaseComponent,
  ) -> entity_component.ComponentT:
    component = self._context_components[name]
    return cast(entity_component.ComponentT, component)

  def get_act_component(self) -> entity_component.ActingComponent:
    return self._act_component

  def get_all_context_components(
      self,
  ) -> Mapping[str, entity_component.ContextComponent]:
    return types.MappingProxyType(self._context_components)

  def add_component(
      self,
      name: str,
      component: entity_component.ContextComponent,
      spec: entity_component.ComponentState | None = None,
  ) -> None:
    """Adds a context component to the agent.

    The component is owned by this agent from then on. Requested while the
    agent is neither acting nor observing, it is added at once; requested
    during `act` or `observe` (e.g. by one of the agent's components), it is
    added when that call ends.

    Args:
      name: The component's name (its key in the agent's contexts).
      component: The component, not owned by any other entity.
      spec: Optional JSON-serializable description of the component. With a
        `component_factory`, `set_state` rebuilds the component from it;
        without a spec, a restored state does not include the component.

    Raises:
      ValueError: If the agent has, or is about to have, a component of that
        name, or the component is owned by another entity.
    """
    try:
      owner = component.get_entity()
    except RuntimeError:
      owner = None
    if owner is not None and owner is not self:
      raise ValueError(f'Component {name} belongs to another entity.')
    with self._changes_lock:
      if self._will_have_component(name):
        raise ValueError(f'{self.name} already has a component named {name}.')
      self._pending_changes.append((name, component, spec))
    self._apply_changes_if_idle()

  def remove_component(self, name: str) -> None:
    """Removes a context component from the agent.

    Requested while the agent is neither acting nor observing, the component
    is removed at once; requested during `act` or `observe`, it is removed
    when that call ends. A removed component is not reused by other agents.

    Args:
      name: The component's name.

    Raises:
      KeyError: If the agent has, or is about to have, no such component.
      ValueError: If the acting component's explicit context order names it.
    """
    get_order = getattr(self._act_component, 'get_context_concat_order', None)
    order = cast(Sequence[str] | None, get_order() if get_order else None)
    if order is not None and name in order:
      raise ValueError(
          f'The acting component of {self.name} orders component {name}; it'
          ' cannot be removed.'
      )
    with self._changes_lock:
      if not self._will_have_component(name):
        raise KeyError(f'{self.name} has no component named {name}.')
      self._pending_changes.append((name, None, None))
    self._apply_changes_if_idle()

  def _will_have_component(self, name: str) -> bool:
    """Whether the agent has the component once pending changes apply."""
    present = name in self._context_components
    for changed, component, _ in self._pending_changes:
      if changed == name:
        present = component is not None
    return present

  def _apply_changes_if_idle(self) -> None:
    """Applies pending changes now unless a call is under way."""
    if self._control_lock.acquire(blocking=False):
      try:
        self._apply_pending_changes()
      finally:
        self._control_lock.release()

  def _apply_pending_changes(self) -> None:
    """Applies added and removed components. Requires the control lock."""
    with self._changes_lock:
      changes, self._pending_changes = self._pending_changes, []
    if not changes:
      return
    # Replaced, not changed in place: readers may hold the old mapping.
    components = dict(self._context_components)
    for name, component, spec in changes:
      if component is None:
        del components[name]
        self._added_specs.pop(name, None)
      else:
        component.set_entity(self)
        self._component_added(name, component)
        components[name] = component
        self._added_specs[name] = spec
    self._context_components = components

  def _component_added(
      self, name: str, component: entity_component.ContextComponent
  ) -> None:
    """Called when a component joins the agent after construction."""
    del name, component

  def _parallel_call_(
      self,
      method_name: str,
      *args,
      executor: futures.ThreadPoolExecutor | None = None,
  ) -> entity_component.ComponentContextMapping:
    """Calls the named method in parallel on all components.

    If a component instance is registered under multiple names, its method
    will only be called once. The result of that call will be mapped to all
    names under which it was registered.

    All calls will be issued with the same payloads.

    Args:
      method_name: The name of the method to call.
      *args: The arguments to pass to the method.
      executor: An optional existing ThreadPoolExecutor to use.

    Returns:
      A ComponentsContext, that is, a mapping of component name to the result of
      the method call.
    """
    # 1. Identify unique component instances.
    unique_components = list(set(self._context_components.values()))

    # 2. Create and execute tasks for each unique component instance once.
    tasks_for_unique = {
        str(id(component)): functools.partial(
            getattr(component, method_name), *args
        )
        for component in unique_components
    }
    results_by_component_id = concurrency.run_tasks(
        tasks_for_unique, executor=executor
    )

    # 3. Construct the final results dictionary.
    final_results: dict[str, str] = {}
    for name, component in self._context_components.items():
      final_results[name] = results_by_component_id[str(id(component))]

    return types.MappingProxyType(final_results)

  @override
  def act(
      self, action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC
  ) -> str:
    with self._control_lock:
      # Activate per-thread capture key so log data from this act() call
      # is routed to the correct entity thread's capture context.
      key_override = self._capture_key_by_thread.get(
          threading.current_thread().ident  # pyrefly: ignore[bad-argument-type]
      )
      if key_override is not None:
        self._active_capture_key = key_override
      try:
        self._set_phase(entity_component.Phase.PRE_ACT)
        contexts = self._parallel_call_('pre_act', action_spec)
        action_attempt = self._act_component.get_action_attempt(
            contexts, action_spec
        )

        self._set_phase(entity_component.Phase.POST_ACT)
        self._parallel_call_('post_act', action_attempt)

        self._set_phase(entity_component.Phase.UPDATE)
        self._parallel_call_('update')

        self._set_phase(entity_component.Phase.READY)
      except Exception:
        # Ensure correct error handling in the case of multiple threads
        # using the same entity by setting the phase to ready before raising.
        self.set_phase(entity_component.Phase.READY)
        raise
      finally:
        self._active_capture_key = self._agent_name
        self._apply_pending_changes()
    # Changes requested as the call ended, after the line above.
    self._apply_changes_if_idle()
    return action_attempt

  @override
  def observe(self, observation: str) -> None:
    with self._control_lock:
      # Activate per-thread capture key (same as act()).
      key_override = self._capture_key_by_thread.get(
          threading.current_thread().ident  # pyrefly: ignore[bad-argument-type]
      )
      if key_override is not None:
        self._active_capture_key = key_override
      try:
        self._set_phase(entity_component.Phase.PRE_OBSERVE)
        self._parallel_call_('pre_observe', observation)

        self._set_phase(entity_component.Phase.POST_OBSERVE)
        self._parallel_call_('post_observe')

        self._set_phase(entity_component.Phase.UPDATE)
        self._parallel_call_('update')

        self._set_phase(entity_component.Phase.READY)
      except Exception:
        # Ensure correct error handling in the case of multiple threads
        # using the same entity by setting the phase to ready before raising.
        self.set_phase(entity_component.Phase.READY)
        raise
      finally:
        self._active_capture_key = self._agent_name
        self._apply_pending_changes()
    # Changes requested as the call ended, after the line above.
    self._apply_changes_if_idle()

  def set_state(
      self, entity_components_state: entity_component.EntityState
  ) -> None:
    """Sets the state of the agent.

    Components added and removed after construction are added and removed
    again as the state records them (rebuilding added components through the
    `component_factory`), before every component's state is restored.

    Args:
      entity_components_state: A state from `get_state`.

    Raises:
      ValueError: If the state has added components but the agent has no
        `component_factory`.
    """
    # A state without changes records the components given at construction.
    dynamic = cast(
        Mapping[str, Any],
        entity_components_state.get('dynamic_components') or {},
    )
    changed = self._added_specs or set(self._context_components) != set(
        self._built_in_components
    )
    if dynamic or changed:
      with self._control_lock:
        self._restore_component_set(dynamic)

    # Restore context components
    context_components_state = entity_components_state.get(
        'context_components', {}
    )
    for component_name, component in self._context_components.items():
      if component_name in context_components_state:
        try:
          component.set_state(context_components_state[component_name])  # pyrefly: ignore[bad-argument-type]
        except Exception:  # pylint: disable=broad-exception-caught
          logging.error(
              'Error setting state for component %s: %s',
              component_name, traceback.format_exc()
          )

    # Restore act component
    act_state = entity_components_state.get('act_component')
    if act_state:
      try:
        self._act_component.set_state(act_state)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.error(
            'Error setting state for act component: %s', traceback.format_exc()
        )

  def _restore_component_set(self, dynamic: Mapping[str, Any]) -> None:
    """Makes the components those a state records. Requires the lock."""
    added: dict[str, entity_component.ComponentState] = dict(
        dynamic.get('added', {})
    )
    removed = set(dynamic.get('removed', ()))
    factory = self._component_factory
    if added and factory is None:
      raise ValueError(
          f'{self.name} needs a component_factory to restore added components.'
      )
    with self._changes_lock:
      self._pending_changes = []
    components = {
        name: component
        for name, component in self._built_in_components.items()
        if name not in removed
    }
    specs = {}
    for name, spec in added.items():
      current = self._context_components.get(name)
      if current is not None and self._added_specs.get(name) == spec:
        components[name] = current  # Already this component.
      elif factory is not None:
        component = factory(name, spec)
        component.set_entity(self)
        self._component_added(name, component)
        components[name] = component
      specs[name] = spec
    self._added_specs = specs
    self._context_components = components

  def get_state(self) -> entity_component.EntityState:
    """Returns the state of the agent as a dictionary."""
    components = self._context_components
    state: dict[str, entity_component.ComponentState] = {
        'act_component': self._act_component.get_state(),
        'context_components': {
            component_name: component.get_state()
            for component_name, component in components.items()
        },
    }
    removed = sorted(set(self._built_in_components) - set(components))
    added = {
        name: spec
        for name, spec in self._added_specs.items()
        if spec is not None and name in components
    }
    if added or removed:
      state['dynamic_components'] = {'added': added, 'removed': removed}
    return state

  def set_phase(self, phase: entity_component.Phase) -> None:
    with self._phase_lock:
      self._phase = phase

