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

"""Selects a specific context component to contextualize the action."""

from typing import override

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class SelectActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """ActComponent treating specific ContextComponent as the action context."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      key: str,
      prefix_entity_name: bool = True,
      randomize_choices: bool = True,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for generating the action attempt.
      key: Key of the component to use to contextualize the action.
      prefix_entity_name: Whether to prefix the entity name to the output of
        `get_action_attempt` when the `action_spec` output type is `FREE`.
      randomize_choices: Whether to randomize the choices in the
        `get_action_attempt` when the `action_spec` output type is `CHOICE`.
    """
    super().__init__()
    self._model = model
    self._key = key
    self._prefix_entity_name = prefix_entity_name
    self._randomize_choices = randomize_choices

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    return contexts[self._key]

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + '\n')

    call_to_action = action_spec.call_to_action.replace(
        '{name}', self.get_entity().name
    )
    if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      output = ''
      if self._prefix_entity_name:
        output = self.get_entity().name + ' '
      output += prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=output,
          terminators=(),
          question_label='Exercise',
      )
      self._log(output, prompt)
      return output
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      idx = prompt.multiple_choice_question(
          question=call_to_action,
          answers=action_spec.options,
          randomize_choices=self._randomize_choices,
      )
      output = action_spec.options[idx]
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      if self._prefix_entity_name:
        prefix = self.get_entity().name + ' '
      else:
        prefix = ''
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt)
      try:
        return str(float(sampled_text))
      except ValueError:
        return 'nan'
    else:
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}. '
          'Supported output types are: FREE, CHOICE, and FLOAT.'
      )

  def _log(self, result: str, prompt: interactive_document.InteractiveDocument):
    self._logging_channel({
        'Summary': f'Action: {result}',
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
    })

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to a dictionary."""
    return {
        'key': self._key,
        'prefix_entity_name': self._prefix_entity_name,
        'randomize_choices': self._randomize_choices,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'key' in state:
      self._key = state['key']
    if 'prefix_entity_name' in state:
      self._prefix_entity_name = state['prefix_entity_name']
    if 'randomize_choices' in state:
      self._randomize_choices = state['randomize_choices']
