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

"""An acting component that can be set to give fixed responses."""


from collections.abc import Mapping, Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import clock as game_clock
from concordia.typing.deprecated import entity as entity_lib
from concordia.typing.deprecated import entity_component
from concordia.typing.deprecated import logging
from concordia.utils import helper_functions
from typing_extensions import override

DEFAULT_PRE_ACT_KEY = 'Act'


class PuppetActComponent(entity_component.ActingComponent):
  """A component which concatenates contexts from context components.

  The component will output a fixed response to a pre-specified calls to action.
  Otherwise, this component will receive the contexts from `pre_act` from all
  the components, and assemble them in the order specified to `__init__`. If the
  component order is not specified, then components will be assembled in the
  iteration order of the `ComponentContextMapping` passed to
  `get_action_attempt`. Components that return empty strings from `pre_act` are
  ignored.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock.GameClock,
      fixed_responses: Mapping[str, str],
      component_order: Sequence[str] | None = None,
      search_in_prompt: bool = False,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initializes the agent.

    Args:
      model: The language model to use for generating the action attempt.
      clock: the game clock is needed to know when is the current time
      fixed_responses: A mapping from call to action to fixed response.
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
      search_in_prompt: Optionally, search for the target string in the entire
        prompt (i.e. the context), not just the call to action. Off by default.
      pre_act_key: Prefix to add to the context of the component.
      logging_channel: The channel to use for debug logging.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    self._model = model
    self._clock = clock
    self._search_in_prompt = search_in_prompt
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

    self._fixed_responses = fixed_responses

    self._pre_act_key = pre_act_key
    self._logging_channel = logging_channel

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    if self._component_order is None:
      return '\n'.join(context for context in contexts.values() if context)
    else:
      order = self._component_order + tuple(
          sorted(set(contexts.keys()) - set(self._component_order))
      )
      return '\n'.join(contexts[name] for name in order if contexts[name])

  def _format_call_to_action(self, call_to_action: str) -> str:
    """Fill agent name {name} and time interval {timedelta} in call to action.
    """
    return call_to_action.format(
        name=self.get_entity().name,
        timedelta=helper_functions.timedelta_to_readable_str(
            self._clock.get_step_size()
        ),
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

    call_to_action = self._format_call_to_action(action_spec.call_to_action)

    formatted_fixed_responses = {
        self._format_call_to_action(call): response
        for call, response in self._fixed_responses.items()
    }

    if call_to_action in formatted_fixed_responses:
      print(
          f'Using fixed response:  "{call_to_action}" |->'
          f' "{formatted_fixed_responses[call_to_action]}"'
      )
      output = formatted_fixed_responses[call_to_action]
      if action_spec.output_type == entity_lib.OutputType.FREE:
        _ = prompt.open_question(question=call_to_action,
                                 forced_response=output,
                                 question_label='Exercise',)
        self._log(output, prompt)
      else:
        _ = prompt.open_question(question=call_to_action,
                                 forced_response=output,)
        self._log(output, prompt)
      if (
          action_spec.output_type == entity_lib.OutputType.CHOICE
          and output not in action_spec.options
      ):
        raise ValueError(
            f'Fixed response {output} not in options: {action_spec.options}'
        )
      elif action_spec.output_type == entity_lib.OutputType.FLOAT:
        try:
          return str(float(output))
        except ValueError:
          return '0.0'

      return formatted_fixed_responses[call_to_action]

    if self._search_in_prompt:
      context_str = prompt.view().text()
      for target in formatted_fixed_responses:
        if target in context_str:
          print(
              f'Using fixed response:  "{target}" |->'
              f' "{formatted_fixed_responses[target]}"'
          )
          output = formatted_fixed_responses[target]
          if action_spec.output_type == entity_lib.OutputType.FREE:
            _ = prompt.open_question(question=target,
                                     forced_response=output,
                                     question_label='Exercise',)
            self._log(output, prompt)
          else:
            _ = prompt.open_question(question=target,
                                     forced_response=output,)
            self._log(output, prompt)
          if (
              action_spec.output_type == entity_lib.OutputType.CHOICE
              and output not in action_spec.options
          ):
            raise ValueError(
                f'Fixed response {output} not in options: {action_spec.options}'
            )
          elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            try:
              return str(float(output))
            except ValueError:
              return '0.0'

          return formatted_fixed_responses[target]

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
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      prefix = self.get_entity().name + ' '
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt)
      try:
        return str(float(sampled_text))
      except ValueError:
        return '0.0'
    else:
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}. '
          'Supported output types are: FREE, CHOICE, and FLOAT.'
      )

  def _log(self, result: str, prompt: interactive_document.InteractiveDocument):
    self._logging_channel({
        'Key': self._pre_act_key,
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
    })
