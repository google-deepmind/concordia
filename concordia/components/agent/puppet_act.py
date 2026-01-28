# Copyright 2024 DeepMind Technologies Limited.
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

"""An acting component that uses fixed responses or falls back to LLM."""

from collections.abc import Mapping

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class PuppetActComponent(
    entity_component.ContextComponent, entity_component.ActingComponent
):
  """A component for providing fixed responses or falling back to the LLM.

  This component allows an agent to respond with pre-configured fixed responses
  for specific calls to action. If no fixed response matches, it falls back to
  generating a response using the language model.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      fixed_responses: Mapping[str, str],
  ):
    """Initializes the PuppetActComponent.

    Args:
      model: The language model to use when no fixed response is available.
      fixed_responses: A mapping from call-to-action strings to fixed response
        strings. The call-to-action keys may contain {name} placeholders which
        will be formatted with the entity's name before matching.
    """
    super().__init__()
    self._model = model
    self._fixed_responses = fixed_responses

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    return "\n".join(context for context in contexts.values() if context)

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Returns a fixed response if available, otherwise generates one."""
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + "\n")

    call_to_action = action_spec.call_to_action
    try:
      cta_formatted = call_to_action.format(name=self.get_entity().name)
    except (KeyError, IndexError):
      cta_formatted = call_to_action

    formatted_fixed_responses = {
        key.format(name=self.get_entity().name): value
        for key, value in self._fixed_responses.items()
    }

    if cta_formatted in formatted_fixed_responses:
      return formatted_fixed_responses[cta_formatted]

    if call_to_action in formatted_fixed_responses:
      return formatted_fixed_responses[call_to_action]

    if action_spec.output_type == entity_lib.OutputType.FREE:
      output = self.get_entity().name + " "
      output += prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=output,
          terminators=('" ', "\n"),
          question_label="Exercise",
      )
      return output
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      return action_spec.options[idx]
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      prefix = self.get_entity().name + " "
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      try:
        return str(float(sampled_text))
      except ValueError:
        return "0.0"
    else:
      return ""

  def get_state(self) -> entity_component.ComponentState:
    """Returns the current state of the component."""
    return {"responses": self._fixed_responses}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
