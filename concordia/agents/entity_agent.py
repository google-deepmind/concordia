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
import types
from typing import cast

from concordia.components.agent.v2 import no_op_context_processor
from concordia.typing import component_v2
from concordia.typing import entity
from concordia.utils import concurrency
from typing_extensions import override

# TODO: b/313715068 - remove disable once pytype bug is fixed.
# pytype: disable=override-error


class EntityAgent(component_v2.EntityWithComponents):
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
      act_component: component_v2.ActingComponent,
      context_processor: component_v2.ContextProcessorComponent | None = None,
      context_components: Mapping[str, component_v2.ContextComponent] = (
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
    self._phase = component_v2.Phase.INIT

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
  def get_phase(self) -> component_v2.Phase:
    return self._phase

  @override
  def get_component(
      self,
      name: str,
      *,
      type_: type[component_v2.ComponentT] = component_v2.BaseComponent,
  ) -> component_v2.ComponentT:
    component = self._context_components[name]
    return cast(component_v2.ComponentT, component)

  def _parallel_call_(
      self,
      method_name: str,
      *args,
  ) -> component_v2.ComponentContextMapping:
    """Calls the named method in parallel on all components.

    All calls will be issued with the same payloads.

    Args:
      method_name: The name of the method to call.
      *args: The arguments to pass to the method.

    Returns:
      A ComponentsContext, that is, a mapping of component name to the result of
      the method call.
    """
    context_futures = {}
    with concurrency.executor() as pool:
      for name, component in self._context_components.items():
        context_futures[name] = pool.submit(
            getattr(component, method_name), *args
        )

    return {
        name: future.result() for name, future in context_futures.items()
    }

  @override
  def act(
      self, action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC
  ) -> str:
    self._phase = component_v2.Phase.PRE_ACT
    contexts = self._parallel_call_('pre_act', action_spec)
    self._context_processor.pre_act(types.MappingProxyType(contexts))
    action_attempt = self._act_component.get_action_attempt(
        contexts, action_spec
    )

    self._phase = component_v2.Phase.POST_ACT
    contexts = self._parallel_call_('post_act', action_attempt)
    self._context_processor.post_act(contexts)

    self._phase = component_v2.Phase.UPDATE
    self._parallel_call_('update')

    return action_attempt

  @override
  def observe(self, observation: str) -> None:
    self._phase = component_v2.Phase.PRE_OBSERVE
    contexts = self._parallel_call_('pre_observe', observation)
    self._context_processor.pre_observe(contexts)

    self._phase = component_v2.Phase.POST_OBSERVE
    contexts = self._parallel_call_('post_observe')
    self._context_processor.post_observe(contexts)

    self._phase = component_v2.Phase.UPDATE
    self._parallel_call_('update')

  def get_last_log(self):
    logs = self._parallel_call_('get_last_log')
    return {
        '__act__': self._act_component.get_last_log(),
        **logs,
    }
