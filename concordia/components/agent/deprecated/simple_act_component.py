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


from collections.abc import Sequence

from concordia.language_model import language_model
from concordia.typing.deprecated import entity as entity_lib
from concordia.typing.deprecated import entity_component
from typing_extensions import override


class SimpleActComponent(entity_component.ActingComponent):
  """A simple acting component that aggregates contexts from components.

  This component will receive the contexts from `pre_act` from all the
  components, and assemble them in the order specified to `__init__`. If the
  component order is not specified, then components will be assembled in the
  iteration order of the `ComponentContextMapping` passed to
  `get_action_attempt`.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      component_order: Sequence[str] | None = None,
  ):
    """Initializes the agent.

    Args:
      model: The language model to use for generating the action attempt.
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
    self._model = model
    if component_order is None:
      self._component_order = None
    else:
      self._component_order = tuple(component_order)
    if self._component_order is not None:
      if len(set(self._component_order)) != len(self._component_order):
        raise ValueError(
            "The component order contains duplicate components: "
            + ", ".join(self._component_order)
        )

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    if self._component_order is None:
      return "\n".join(
          f"{name}: {context}" for name, context in contexts.items()
      )
    else:
      order = self._component_order + tuple(
          set(contexts.keys()) - set(self._component_order)
      )
      return "\n".join(
          f"{name}: {contexts[name]}" for name in order
      )

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    context = self._context_for_action(contexts)
    if action_spec.output_type == entity_lib.OutputType.CHOICE:
      _, response, _ = self._model.sample_choice(
          f"{context}\n\n{action_spec.call_to_action}\n",
          action_spec.options)
      return response
    sampled_text = self._model.sample_text(
        f"{context}\n\n{action_spec.call_to_action}\n",
    )
    if action_spec.output_type == entity_lib.OutputType.FREE:
      return sampled_text
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      try:
        return str(float(sampled_text))
      except ValueError:
        return "0.0"
    raise NotImplementedError(
        f"Unsupported output type: {action_spec.output_type}. "
        "Supported output types are: FREE, CHOICE, and FLOAT."
    )
