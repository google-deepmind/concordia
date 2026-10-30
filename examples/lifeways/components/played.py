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

"""How a person someone plays acts: by the player's answer, or by habit.

The player answers through the stock `HumanActComponent` and its input. An
input that gets no answer in time (or whose world is stopping) raises
`Unanswered`; it never makes up a move. Then the person acts by habit, a rule
of this component's own, logged as such: it gathers food, passes on all it
holds, or keeps its counsel.
"""

from collections.abc import Mapping
from typing import override

from concordia.components.agent import human_act_component
from examples.lifeways.components import culture
from examples.lifeways.world import world as world_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging as logging_lib


class Unanswered(Exception):
  """Nobody answered for the person in time, or its world is stopping."""


# What a played person does by habit, by the kind of question (`tag`), in
# words for the page.
HABITS: Mapping[str, str] = {
    world_lib.MOVES: 'gathers food',
    world_lib.PASSING_ON: 'passes on all they hold',
    world_lib.INTERVIEW: 'keeps their counsel',
}
# What a played person says by habit when asked by the Experimenter.
KEPT_COUNSEL = 'I would rather not say.'


def habit_words(kind: str) -> str:
  """What a played person does by habit for a kind of question, in words.

  Args:
    kind: The question's tag (`world.MOVES`, `PASSING_ON` or `INTERVIEW`).

  Returns:
    E.g. 'gathers food'; a question of another kind is a move.
  """
  return HABITS.get(kind, HABITS[world_lib.MOVES])


class PlayedAct(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Acts by the player's answer, or by the person's habit.

  It asks the player through a stock `HumanActComponent`. If its input
  raises `Unanswered`, the person acts by habit (`HABITS`): it gathers food,
  passes on the customs its `Learner` holds, or keeps its counsel. The habit
  is logged with `'Source': 'habit'`; the player's answers keep the human
  component's `'Source': 'human'`.
  """

  def __init__(
      self,
      human: human_act_component.HumanActComponent,
      learner_key: str,
  ):
    """Initializes the component.

    Args:
      human: Asks the player (its input raises `Unanswered` when unanswered).
      learner_key: The key of the person's `Learner`, which knows what it
        holds.
    """
    super().__init__()
    self._human = human
    self._learner_key = learner_key

  @override
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    self._human.set_entity(entity)

  @override
  def set_logging_channel(
      self, logging_channel: logging_lib.LoggingChannel
  ) -> None:
    super().set_logging_channel(logging_channel)
    self._human.set_logging_channel(logging_channel)

  def get_context_concat_order(self) -> list[str] | None:
    """The human component's context order (the entity reads it)."""
    order = self._human.get_context_concat_order()
    return list(order) if order is not None else None

  @override
  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    try:
      return self._human.get_action_attempt(context, action_spec)
    except Unanswered:
      answer = self._habit(action_spec)
      self._logging_channel({
          'Summary': f'Action by habit: {answer}',
          'Value': answer,
          'Source': 'habit',
      })
      return answer

  def _habit(self, action_spec: entity_lib.ActionSpec) -> str:
    """What the person does unasked, by the kind of question."""
    if action_spec.tag == world_lib.PASSING_ON:
      learner = self.get_entity().get_component(
          self._learner_key, type_=culture.Learner
      )
      return ' '.join([world_lib.PASS_ON, *learner.held()])
    if action_spec.tag == world_lib.INTERVIEW:
      return KEPT_COUNSEL
    return world_lib.GATHER

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'human': self._human.get_state()}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    human = state.get('human')
    if isinstance(human, Mapping):
      self._human.set_state(dict(human))
