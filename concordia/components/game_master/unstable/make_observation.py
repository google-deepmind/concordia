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

"""Component that helps a game master decide whose turn is next."""

from collections.abc import Mapping
import types

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging


DEFAULT_MAKE_OBSERVATION_COMPONENT_NAME = '__make_observation__'
DEFAULT_MAKE_OBSERVATION_PRE_ACT_KEY = '\nPrompt'

GET_ACTIVE_ENTITY_QUERY = 'Who is about to act?'


class MakeObservation(entity_component.ContextComponent):
  """A component that generates observations to send to players.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      pre_act_key: str = DEFAULT_MAKE_OBSERVATION_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._components = dict(components)
    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_value()
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      entity_name = self.get_entity().name
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join([
          f"{entity_name}'s"
          f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
          for key, prefix in self._components.items()
      ])
      prompt.statement(f'{component_states}\n')
      active_entity_name = prompt.open_question(
          GET_ACTIVE_ENTITY_QUERY)
      result = prompt.open_question(
          question=(f'What does {active_entity_name} observe now? Never '
                    'repeat information that was already provided to '
                    f'{active_entity_name} unless absolutely necessary. Keep '
                    'the story moving forward.'),
          max_tokens=1200)
      prompt_to_log = prompt.view().text()

    self._logging_channel(
        {'Key': self._pre_act_key,
         'Value': result,
         'Prompt': prompt_to_log})
    return result
