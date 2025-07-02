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

from collections.abc import Mapping
import functools
import threading
import traceback
import types
from typing import cast

from concordia.components.agent.deprecated import no_op_context_processor
from concordia.typing import entity
from concordia.typing import entity_component
from concordia.utils import concurrency
from typing_extensions import override

# TODO: b/313715068 - remove disable once pytype bug is fixed.
# pytype: disable=override-error


class EntityAgent(entity_component.EntityWithComponents):
  """An agent that has its functionality defined by components.

  The agent has a set of components that define its functionality. The agent
  must have at least an ActComponent and an ObserveComponent. The agent will
  call the ActComponent's `act` method when it needs to act, and the
  ObservationComponent's `observe` method when they need to process an
  observation.
  """

  def __init__(
      self,
      agent_name: str,
      act_component: entity_component.ActingComponent,
      context_processor: (
          entity_component.ContextProcessorComponent | None
      ) = None,
      context_components: Mapping[str, entity_component.ContextComponent] = (
          types.MappingProxyType({})
      ),
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_processor: The component that will be used to process contexts. If
        None, a NoOpContextProcessor will be used.
      context_components: The ContextComponents that will be used by the agent.
    """
    super().__init__()
    self._agent_name = agent_name
    self._control_lock = threading.Lock()
    self._phase_lock = threading.Lock()
    self._phase = entity_component.Phase.READY

    self._act_component = act_component
    self._act_component.set_entity(self)

    if context_processor is None:
      self._context_processor = no_op_context_processor.NoOpContextProcessor()
    else:
      self._context_processor = context_processor
    self._context_processor.set_entity(self)

    self._context_components = dict(context_components)
    for component in self._context_components.values():
      component.set_entity(self)

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

  def _parallel_call_(
      self,
      method_name: str,
      *args,
  ) -> entity_component.ComponentContextMapping:
    """Calls the named method in parallel on all components.

    If a component instance is registered under multiple names, its method
    will only be called once. The result of that call will be mapped to all
    names under which it was registered.

    All calls will be issued with the same payloads.

    Args:
      method_name: The name of the method to call.
      *args: The arguments to pass to the method.

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
    results_by_component_id = concurrency.run_tasks(tasks_for_unique)

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
      self._set_phase(entity_component.Phase.PRE_ACT)
      contexts = self._parallel_call_('pre_act', action_spec)
      self._context_processor.pre_act(types.MappingProxyType(contexts))
      action_attempt = self._act_component.get_action_attempt(
          contexts, action_spec
      )

      self._set_phase(entity_component.Phase.POST_ACT)
      contexts = self._parallel_call_('post_act', action_attempt)
      self._context_processor.post_act(contexts)

      self._set_phase(entity_component.Phase.UPDATE)
      self._parallel_call_('update')

      self._set_phase(entity_component.Phase.READY)

      return action_attempt

  @override
  def observe(self, observation: str) -> None:
    with self._control_lock:
      self._set_phase(entity_component.Phase.PRE_OBSERVE)
      contexts = self._parallel_call_('pre_observe', observation)
      self._context_processor.pre_observe(contexts)

      self._set_phase(entity_component.Phase.POST_OBSERVE)
      contexts = self._parallel_call_('post_observe')
      self._context_processor.post_observe(contexts)

      self._set_phase(entity_component.Phase.UPDATE)
      self._parallel_call_('update')

      self._set_phase(entity_component.Phase.READY)

  def set_state(
      self, entity_components_state: entity_component.EntityState
  ) -> None:
    """Sets the state of the agent."""

    # Restore context components
    context_components_state = entity_components_state.get(
        'context_components', {}
    )
    for component_name, component in self._context_components.items():
      if component_name in context_components_state:
        try:
          component.set_state(context_components_state[component_name])
        except Exception:  # pylint: disable=broad-exception-caught
          print(
              f'Error setting state for component {component_name}:'
              f' {traceback.format_exc()}'
          )

    # Restore act component
    act_state = entity_components_state.get('act_component')
    if act_state:
      try:
        self._act_component.set_state(act_state)
      except Exception:  # pylint: disable=broad-exception-caught
        print(
            f'Error setting state for act component: {traceback.format_exc()}'
        )

    # Restore context processor
    proc_state = entity_components_state.get('context_processor')
    if proc_state:
      try:
        self._context_processor.set_state(proc_state)
      except Exception:  # pylint: disable=broad-exception-caught
        print(
            'Error setting state for context processor:'
            f' {traceback.format_exc()}'
        )

  def get_state(self) -> entity_component.EntityState:
    """Returns the state of the agent as a dictionary."""
    return {
        'act_component': self._act_component.get_state(),
        'context_processor': self._context_processor.get_state(),
        'context_components': {
            component_name: component.get_state()
            for component_name, component in self._context_components.items()
        },
    }
