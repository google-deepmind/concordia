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

"""A simple acting component that aggregates contexts from components."""

from collections.abc import Sequence, Mapping

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from typing_extensions import override


class ScriptedActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """A an acting component that uses a script to generate actions.
  
  This component is used to generate actions from a script. The script is a list
  of dictionaries, where each entry is a dictionary containing the name of the
  entity and the action to be performed.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      script: list[Mapping[str, str]],
      component_order: Sequence[str] | None = None,
      prefix_entity_name: bool = True,
  ):
    """Initializes the agent.

    Args:
      model: The language model to use for generating the action attempt.
      script: The script to execute. This is a list of dictionaries, where each
        entry is a dictionary containing the name of the entity that the line is
        associated with and the action to be performed.
      component_order: The order in which the component contexts will be
        assembled when calling the act component. If None, the contexts will be
        assembled in the iteration order of the `ComponentContextMapping` passed
        to `get_action_attempt`. If the component order is specified, but does
        not contain all the components passed to `get_action_attempt`, the
        missing components will be appended at the end in the iteration order of
        the `ComponentContextMapping` passed to `get_action_attempt`. The same
        component cannot appear twice in the component order. All components in
        the component order must be in the `ComponentContextMapping` passed to
        `get_action_attempt`.
      prefix_entity_name: Whether to prefix the entity name to the output of
        `get_action_attempt` when the `action_spec` output type is `FREE`. 

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._prefix_entity_name = prefix_entity_name
    if component_order is None:
      self._component_order = None
    else:
      self._component_order = tuple(component_order)
    if self._component_order is not None:
      if len(set(self._component_order)) != len(self._component_order):
        raise ValueError(
            'The component order contains duplicate components: '
            + ', '.join(self._component_order)
        )

    self._script = script
    self._lines = []
    # extract lines of the actor from the script
    # create an iterator over the lines
    self._line_index = 0

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    if self._component_order is None:
      return '\n'.join(
          context for context in contexts.values() if context
      )
    else:
      order = self._component_order + tuple(sorted(
          set(contexts.keys()) - set(self._component_order)))
      return '\n'.join(
          contexts[name] for name in order if contexts[name]
      )

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + '\n')

    if not self._lines:
      for line in self._script:
        if line['name'] == self.get_entity().name:
          self._lines.append(line['line'])

    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name
    )
    output = ''
    if self._prefix_entity_name:
      output = self.get_entity().name + ' '
    if self._line_index < len(self._lines):
      training_context = prompt.view().text() + output
      training_target = self._lines[self._line_index]
      output += prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=output,
          terminators=(),
          question_label='Exercise',
          forced_response=self._lines[self._line_index],
      )
      self._line_index += 1
      self._log(output, prompt, training_context, training_target)
      return output
    else:
      return ''

  def _log(self,
           result: str,
           prompt: interactive_document.InteractiveDocument,
           training_context: str,
           training_target: str):
    self._logging_channel({
        'Summary': f'Action: {result}',
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
        'SFT datum': {'context': training_context, 'target': training_target},
    })

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to a dictionary."""
    return {
        'script': self._script,
        'line_index': self._line_index,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._script = state['script']
    self._line_index = state['line_index']
    pass
