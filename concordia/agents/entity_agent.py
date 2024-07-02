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
import enum
import functools
import types

from concordia.components.agent.v2 import no_op_context_processor
from concordia.typing import component_v2
from concordia.typing import entity
from concordia.utils import concurrency

from typing_extensions import override


_EMPTY_MAPPING = types.MappingProxyType({})


class Phase(enum.Enum):
  """Phases of the agent lifecycle.

  Attributes:
    INIT: The agent has just been created. No action has been requested nor
      observation has been received. This can be followed by a call to `pre_act`
      or `pre_observe`.
    PRE_ACT: The agent has received a request to act. Components are being
      requested for their action context. This will be followed by `POST_ACT`.
    POST_ACT: The agent has just submitted an action attempt. Components are
      being informed of the action attempt. This will be followed by
      `UPDATE`.
    PRE_OBSERVE: The agent has received an observation. Components are being
      informed of the observation. This will be followed by `POST_OBSERVE`.
    POST_OBSERVE: The agent has just observed. Components are given a chance to
      provide context after processing the observation. This will be followed by
      `UPDATE`.
    UPDATE: The agent is about to update its internal state. This will be
      followed by `PRE_ACT` or `PRE_OBSERVE`.
  """
  INIT = enum.auto()
  PRE_ACT = enum.auto()
  POST_ACT = enum.auto()
  PRE_OBSERVE = enum.auto()
  POST_OBSERVE = enum.auto()
  UPDATE = enum.auto()


class EntityAgent(entity.Entity):
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
      components: Mapping[str, component_v2.BaseComponent] = _EMPTY_MAPPING,
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_processor: The component that will be used to process contexts. If
        None, a NoOpContextProcessor will be used.
      components: The components that will be used by the agent.
    """
    self._agent_name = agent_name

    self._act_component = act_component
    self._act_component.set_entity(self)

    if context_processor is None:
      self._context_processor = no_op_context_processor.NoOpContextProcessor()
    else:
      self._context_processor = context_processor
    self._context_processor.set_entity(self)

    self._components = dict(components)
    for component in self._components.values():
      component.set_entity(self)

    self._phase = Phase.INIT

  @functools.cached_property
  @override
  def name(self) -> str:
    return self._agent_name

  def get_phase(self) -> Phase:
    """Returns the current phase of the agent."""
    return self._phase

  def get_component(self, component_name: str) -> component_v2.BaseComponent:
    """Returns the component with the given name."""
    return self._components[component_name]

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
      for name, component in self._components.items():
        context_futures[name] = pool.submit(
            getattr(component, method_name), *args
        )

    return {
        name: future.result() for name, future in context_futures.items()
    }

  @override
  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
    self._phase = Phase.PRE_ACT
    contexts = self._parallel_call_('pre_act', action_spec)

    action_attempt = self._act_component.get_action_attempt(
        contexts, action_spec)

    self._phase = Phase.POST_ACT
    contexts = self._parallel_call_('post_act', action_spec)
    self._context_processor.process(contexts)

    self._phase = Phase.UPDATE
    self._parallel_call_('update', ())

    return action_attempt

  @override
  def observe(
      self,
      observation: str,
  ) -> None:
    self._phase = Phase.PRE_OBSERVE
    contexts = self._parallel_call_('pre_observe', observation)
    self._context_processor.process(contexts)

    self._phase = Phase.POST_OBSERVE
    contexts = self._parallel_call_('post_observe', ())
    self._context_processor.process(contexts)

    self._phase = Phase.UPDATE
    self._parallel_call_('update', ())
