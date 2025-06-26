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

"""A game master acting component with specific calls per action type."""

from collections.abc import Sequence

from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import next_game_master as next_game_master_components
from concordia.components.game_master import terminate as terminate_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from typing_extensions import override


DEFAULT_ACT_COMPONENT_KEY = '__act__'
DEFAULT_PRE_ACT_LABEL = 'Act'

DEFAULT_TERMINATE_COMPONENT_KEY = (
    terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY)
DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY = (
    next_game_master_components.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY)
DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY = (
    make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY)
DEFAULT_NEXT_ACTING_COMPONENT_KEY = (
    next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY)
DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY = (
    next_acting_components.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY)
DEFAULT_RESOLUTION_COMPONENT_KEY = (
    event_resolution_components.DEFAULT_RESOLUTION_COMPONENT_KEY)


class SwitchAct(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """A component which calls the appropriate method for each action type.

  This component will receive the contexts from `pre_act` from all the
  components, and assemble them in the order specified to `__init__`. If the
  component order is not specified, then components will be assembled in the
  iteration order of the `ComponentContextMapping` passed to
  `get_action_attempt`. Components that return empty strings from `pre_act` are
  ignored.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      entity_names: Sequence[str],
      component_order: Sequence[str] | None = None,
  ):
    """Initializes the agent.

    Args:
      model: The language model to use for generating the action attempt.
      entity_names: sequence of entity names to choose from.
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

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._entity_names = entity_names
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

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    if self._component_order is None:
      result = '\n'.join(
          context for context in contexts.values() if context
      )
    else:
      order = self._component_order + tuple(sorted(
          set(contexts.keys()) - set(self._component_order)))
      result = '\n'.join(
          contexts[name] for name in order if contexts[name]
      )
    return result.replace('\n\n\n', '\n\n')

  def _terminate(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_TERMINATE_COMPONENT_KEY in contexts:
      result = str(contexts[DEFAULT_TERMINATE_COMPONENT_KEY])
      self._log(result, context, action_spec)
    else:
      # YOLO case
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      termination_bool = chain_of_thought.yes_no_question(
          question=action_spec.call_to_action)
      if termination_bool:
        result = 'Yes'
      else:
        result = 'No'
      self._log(result, chain_of_thought, action_spec)

    return result

  def _make_observation(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY in contexts:
      result = str(contexts[DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY])
      self._log(result, context, action_spec)
    else:
      # YOLO case
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      result = chain_of_thought.open_question(
          question=action_spec.call_to_action,
          max_tokens=1000)
      self._log(result, chain_of_thought, action_spec)

    return result

  def _next_acting(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_NEXT_ACTING_COMPONENT_KEY in contexts:
      result = str(contexts[DEFAULT_NEXT_ACTING_COMPONENT_KEY])
      self._log(result, context, action_spec)
    else:
      # YOLO case
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      next_entity_index = chain_of_thought.multiple_choice_question(
          question=action_spec.call_to_action,
          answers=self._entity_names)
      result = self._entity_names[next_entity_index]
      self._log(result, chain_of_thought, action_spec)

    return result

  def _next_entity_action_spec(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY in contexts:
      result = str(contexts[DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY])
      if not result:
        result = f'prompt: {entity_lib.DEFAULT_CALL_TO_ACTION};;type: free'
      self._log(result, context, action_spec)
    else:
      # YOLO case
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      _ = chain_of_thought.open_question(question=action_spec.call_to_action)
      # Then ask the GM to reformat their answer in whatever string format can
      # be used by the engine and its parser.
      chain_of_thought.statement(
          'Example formatted action specs:\n1). "prompt: p;;type: free"\n'
          '2). "prompt: p;;type: choice;;options: x, y, z".\nNote that p is a '
          'string of any length, typically a question, and x, y, z, etc are '
          'multiple choice answer responses. For instance, a valid format '
          'could be indicated as '
          'prompt: Where will Edgar go?;;type: choice;;'
          'options: home, London, Narnia, the third moon of Jupiter')
      next_action_spec_string = chain_of_thought.open_question(
          question='In what action spec format should the next player respond?')
      if 'type:' not in next_action_spec_string:
        next_action_spec_string = (
            f'prompt: {entity_lib.DEFAULT_CALL_TO_ACTION};;type: free')

      result = next_action_spec_string
      self._log(result, chain_of_thought, action_spec)

    return result

  def _resolve(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_RESOLUTION_COMPONENT_KEY in contexts:
      result = contexts[DEFAULT_RESOLUTION_COMPONENT_KEY]
      self._log(result, context, action_spec)
    else:
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      result = chain_of_thought.open_question(
          question=action_spec.call_to_action)
      self._log(result, chain_of_thought, action_spec)

    return result

  def _next_game_master(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec) -> str:
    context = self._context_for_action(contexts)
    if DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY in contexts:
      game_master = str(contexts[DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY])
      self._log(game_master, context, action_spec)
    else:
      # YOLO case
      chain_of_thought = interactive_document.InteractiveDocument(self._model)
      chain_of_thought.statement(context)
      game_master_idx = chain_of_thought.multiple_choice_question(
          question=action_spec.call_to_action,
          answers=action_spec.options)
      game_master = action_spec.options[game_master_idx]
      self._log(game_master, chain_of_thought, action_spec)

    return game_master

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + '\n')

    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name,
    )
    if action_spec.output_type == entity_lib.OutputType.FREE:
      output = self.get_entity().name + ' '
      output += prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=output,
          # This terminator protects against the model providing extra context
          # after the end of a directly spoken response, since it normally
          # puts a space after a quotation mark only in these cases.
          terminators=('" ', '\n'),
          question_label='Exercise',
      )
      self._log(output, prompt, action_spec)
      return output
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
      self._log(output, prompt, action_spec)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      prefix = self.get_entity().name + ' '
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt, action_spec)
      try:
        return str(float(sampled_text))
      except ValueError:
        return '0.0'
    elif action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return self._terminate(contexts, action_spec)
    elif action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      return self._next_game_master(contexts, action_spec)
    elif action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      return self._make_observation(contexts, action_spec)
    elif action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      return self._next_acting(contexts, action_spec)
    elif action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return self._next_entity_action_spec(contexts, action_spec)
    elif action_spec.output_type == entity_lib.OutputType.RESOLVE:
      return self._resolve(contexts, action_spec)
    else:
      raise NotImplementedError(
          (f'Unsupported output type: {action_spec.output_type}. '
           'Supported output types are: FREE, CHOICE, FLOAT, TERMINATE, '
           'MAKE_OBSERVATION, NEXT_ACTING, NEXT_ACTION_SPEC, NEXT_GAME_MASTER, '
           'and RESOLVE.')
      )

  def _log(self,
           result: str,
           prompt: str | interactive_document.InteractiveDocument,
           action_spec: entity_lib.ActionSpec):
    if isinstance(prompt, interactive_document.InteractiveDocument):
      prompt = prompt.view().text().splitlines()
    self._logging_channel({
        'Summary': result,
        'Action Spec': action_spec.call_to_action,
        'Value': result,
        'Prompt': prompt,
    })

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    pass
