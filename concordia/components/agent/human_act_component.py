# Copyright 2026 DeepMind Technologies Limited.
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

"""An acting component that asks a human instead of a language model."""

from collections.abc import Sequence
import dataclasses
import json
import math
import types
from typing import cast
from typing import override
import uuid

from concordia.components.agent import concat
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import human_input


def validate_response(
    response: str, action_spec: entity_lib.ActionSpec
) -> None:
  """Validate human input, including the sequential engine's GM contracts.

  This is also available to adapters for immediate feedback before submission.
  HumanActComponent always validates again at the component boundary.

  Unlike language model sampling, human input must not silently fall back to
  an invented action, a different choice, or NaN. Choices are exact option
  values (not indices); free text and numbers are returned without rewriting.

  Raises:
    ValueError: If the human can correct an invalid response.
    NotImplementedError: If the output type is unsupported.
  """
  if not isinstance(response, str):
    raise ValueError('The response must be text.')
  output_type = action_spec.output_type
  if output_type == entity_lib.OutputType.SKIP_THIS_STEP:
    if response:
      raise ValueError('A skipped step must have an empty response.')
    return
  if output_type in entity_lib.CHOICE_ACTION_TYPES:
    # Reuse ActionSpec validation for GM choice types as well as player choices.
    dataclasses.replace(
        action_spec, output_type=entity_lib.OutputType.CHOICE
    ).validate(response)
    return
  if output_type == entity_lib.OutputType.FLOAT:
    action_spec.validate(response)
    if not math.isfinite(float(response)):
      raise ValueError('Enter a finite number (not NaN or infinity).')
    return
  if output_type not in entity_lib.FREE_ACTION_TYPES:
    raise NotImplementedError(f'Unsupported output type: {output_type}')
  if not response.strip():
    raise ValueError('Enter a non-empty response.')
  if output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
    dataclasses.replace(
        action_spec, output_type=entity_lib.OutputType.FREE
    ).validate(response)
    return

  # ActionSpec's constructor checks option cardinality but not JSON field
  # types. Validate these before passing the object to the standard converter.
  try:
    value = json.loads(response)
  except json.JSONDecodeError as exc:
    raise ValueError('Enter a valid JSON action specification.') from exc
  if not isinstance(value, dict):
    raise ValueError('The action specification must be a JSON object.')
  if not isinstance(value.get('call_to_action'), str):
    raise ValueError('The action specification needs a text call_to_action.')
  if not isinstance(value.get('output_type'), str):
    raise ValueError('The action specification needs a text output_type.')
  options = value.get('options', [])
  if not isinstance(options, list) or not all(
      isinstance(option, str) for option in options
  ):
    raise ValueError('options must be a JSON array of strings.')
  if value.get('tag') is not None and not isinstance(value['tag'], str):
    raise ValueError('tag must be text or null.')
  try:
    next_spec = entity_lib.action_spec_from_dict(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f'Invalid action specification: {exc}') from exc
  if next_spec.output_type not in (
      *entity_lib.PLAYER_ACTION_TYPES,
      entity_lib.OutputType.SKIP_THIS_STEP,
  ):
    raise ValueError('The next action specification must be for a player.')
  if (
      next_spec.output_type != entity_lib.OutputType.SKIP_THIS_STEP
      and not next_spec.call_to_action.strip()
  ):
    raise ValueError('Enter a non-empty call_to_action for the player.')


class HumanActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Ask the controller of this entity for every non-skipped action.

  Install this in a normal EntityAgent or EntityAgentWithLogging, for either a
  player or a GM. Other entities and all context-component lifecycle hooks are
  unchanged. LLM-backed context components still run; use observation, memory,
  constant, and other non-LLM components for a fully human-controlled player.

  A human GM answers engine requests explicitly, including NEXT_ACTING,
  NEXT_ACTION_SPEC, TERMINATE, and NEXT_GAME_MASTER. Context components may
  supply advice but never silently substitute their answer for the human's.

  The reader is a runtime dependency and is not serialized. Restore the
  component with the same reader wiring. No pending request survives a process
  restart; transport adapters must not replay old responses into a new run.
  """

  def __init__(
      self,
      input_reader: human_input.HumanInput,
      component_order: Sequence[str] | None = None,
  ):
    """Initialize a human policy with the same context ordering as ConcatAct.

    Args:
      input_reader: Blocking transport receiving the full ordered context.
      component_order: None uses context mapping order. An explicit sequence
        appends unspecified keys sorted; absent keys raise KeyError at act time.
        Duplicate keys raise ValueError here. Empty context values are omitted.
    """
    super().__init__()
    self._input_reader = input_reader
    self._component_order = concat.validate_component_order(component_order)

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
      return ''
    if action_spec.output_type not in (
        *entity_lib.FREE_ACTION_TYPES,
        *entity_lib.CHOICE_ACTION_TYPES,
        entity_lib.OutputType.FLOAT,
    ):
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}'
      )
    name = self.get_entity().name
    request = human_input.HumanInputRequest(
        request_id=uuid.uuid4().hex,
        entity_name=name,
        action_spec=dataclasses.replace(
            action_spec,
            call_to_action=action_spec.call_to_action.replace('{name}', name),
        ),
        contexts=types.MappingProxyType(dict(contexts)),
        context=concat.concat_contexts(contexts, self._component_order),
    )
    while True:
      response = self._input_reader(request)
      try:
        validate_response(response, action_spec)
      except ValueError as exc:
        request = dataclasses.replace(
            request,
            error=str(exc),
            previous_response=response if isinstance(response, str) else None,
        )
        continue
      self._logging_channel({
          'Summary': f'Action: {response}',
          'Value': response,
          'Prompt': (
              (
                  request.context + '\n' + request.action_spec.call_to_action
              ).splitlines()
          ),
          'Context': request.context,
          'Source': 'human',
      })
      return response

  def get_context_concat_order(self) -> Sequence[str] | None:
    """Return the configured order, or None for mapping iteration order."""
    return self._component_order

  @override
  def get_state(self) -> entity_component.ComponentState:
    # Match ConcatAct's state format; the input reader remains runtime-only.
    return {
        'component_order': (
            list(self._component_order) if self._component_order else None
        ),
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'component_order' in state:
      order = cast(Sequence[str] | None, state['component_order'])
      self._component_order = tuple(order) if order else None
