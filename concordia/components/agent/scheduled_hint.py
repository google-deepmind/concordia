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

"""Agent component for scheduled hinting."""
from collections.abc import Callable, Mapping, Sequence
import datetime
import types

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import logging

DEFAULT_PRE_ACT_KEY = '\nHint'


class ScheduledHint(action_spec_ignored.ActionSpecIgnored):
  """Deliver a specific hints to an agent at specific times."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      hints: Sequence[Callable[[str, datetime.datetime], str]] | None = None,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the PersonBySituation component.

    Args:
      model: The language model to use.
      components: The components to condition the answer on.
      clock_now: time callback to use for the state.
      hints: Sequence of possible hints to apply on each step. Each checks a
        condition based on the incoming chain of thought and either outputs a
        string or not.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._components = dict(components)
    self._clock_now = clock_now
    self._hints = hints
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    chain_of_thought = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f"{agent_name}'s"
        f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    chain_of_thought.statement(
        f'Current time: {self._clock_now()}\n' + component_states)
    hint_outputs = []
    for hint in self._hints:
      hint_output = hint(chain_of_thought.copy().view().text(),
                         self._clock_now())
      if hint_output:
        hint_outputs.append(hint_output)

    hint_outputs_str = ' '.join(hint_outputs)
    result = f'{hint_outputs_str}'

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': chain_of_thought.view().text().splitlines(),
    })

    return result
