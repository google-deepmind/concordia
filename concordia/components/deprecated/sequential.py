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

"""Component that chain components in a sequential way, removing concurrency."""

from typing import Sequence

from absl import logging
from concordia.typing.deprecated import component
from concordia.utils import helper_functions


class Sequential(component.Component):
  """Chains components, removing concurrency."""

  def __init__(self, name: str, components: Sequence[component.Component]):
    self._components = components
    self._name = name
    logging.warning(
        'The Sequential component is deprecated. Please use Entity Components '
        'and specifically `action_spec_ignored` to achieve the same effect '
        'as the old Sequential component.')

  def update(self) -> None:
    for comp in self._components:
      helper_functions.apply_recursively(comp, function_name='update')

  def state(self) -> str:
    return '\n' + '\n'.join([
        comp.name() + ': ' + comp.state()
        for comp in self._components
        if comp.state()
    ])

  def partial_state(self, player_name: str) -> str | None:
    return '\n'.join([
        comp.partial_state(player_name)
        for comp in self._components
        if comp.partial_state(player_name)
    ])

  def observe(self, observation: str):
    for comp in self._components:
      comp.observe(observation)

  def update_before_event(self, cause_statement: str) -> None:
    for comp in self._components:
      helper_functions.apply_recursively(
          comp,
          function_name='update_before_event',
          function_arg=cause_statement)

  def update_after_event(self, event_statement: str) -> None:
    for comp in self._components:
      helper_functions.apply_recursively(
          comp,
          function_name='update_after_event',
          function_arg=event_statement)

  def terminate_episode(self) -> bool:
    for comp in self._components:
      if comp.terminate_episode():
        return True
    return False

  def name(self) -> str:
    return self._name

  def get_last_log(
      self,
  ):
    """Returns a dictionary with latest log of activity."""
    output = {}
    for comp in self._components:
      output[comp.name()] = comp.get_last_log()

    return output
