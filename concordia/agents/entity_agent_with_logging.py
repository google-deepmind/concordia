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

"""A modular entity agent using the new component system with side logging."""

from collections.abc import Mapping
import types
from typing import Any

from concordia.agents import entity_agent
from concordia.typing import agent
from concordia.typing import component_v2
from concordia.utils import measurements as measurements_lib

import reactivex as rx


class EntityAgentWithLogging(entity_agent.EntityAgent, agent.GenerativeAgent):
  """An agent that exposes the latest information of each component."""

  def __init__(
      self,
      agent_name: str,
      act_component: component_v2.ActingComponent,
      context_processor: component_v2.ContextProcessorComponent | None = None,
      context_components: Mapping[str, component_v2.ContextComponent] = (
          types.MappingProxyType({})
      ),
      component_logging: measurements_lib.Measurements | None = None,
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Whenever `get_last_log` is called, the latest values published in all the
    channels in the given measurements object will be returned as a mapping of
    channel name to value.

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_processor: The component that will be used to process contexts. If
        None, a NoOpContextProcessor will be used.
      context_components: The ContextComponents that will be used by the agent.
      component_logging: The channels where components publish events.
    """
    super().__init__(agent_name=agent_name,
                     act_component=act_component,
                     context_processor=context_processor,
                     context_components=context_components)
    self._log: Mapping[str, Any] = {}
    self._tick = rx.subject.Subject()
    self._component_logging = component_logging
    if self._component_logging is not None:
      self._channel_names = list(self._component_logging.available_channels())
      channels = [
          self._component_logging.get_channel(channel)  # pylint: disable=attribute-error  pytype mistakenly forgets that `_component_logging` is not None.
          for channel in self._channel_names
      ]
      rx.with_latest_from(self._tick, *channels).subscribe(self._set_log)
    else:
      self._channel_names = []

  def _set_log(self, log: tuple[Any, ...]) -> None:
    """Set the logging object to return from get_last_log.

    Args:
      log: A tuple with the tick first, and the latest log from each component.
    """
    tick_value, *channel_values = log
    assert tick_value is None
    self._log = dict(zip(self._channel_names, channel_values, strict=True))

  def get_last_log(self):
    self._tick.on_next(None)  # Trigger the logging.
    return self._log
