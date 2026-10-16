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

"""Action resolution: the World GM's `__resolution__` component.

The Simultaneous engine joins every entity's `"Name: action"` into one
putative event, shows it to the game master with `PUTATIVE_EVENT_TAG` and then
asks for `RESOLVE`. Deep Time resolves structured actions deterministically,
in the fixed order the actors were registered (not the order threads
finished): each line goes to the acting entity's `Actor`, which changes the
world and reports what happened. No language model is called.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.game_master import event_resolution
from examples.deep_time import game as game_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG


def split_actions(text: str, names: Sequence[str]) -> dict[str, str]:
  """Splits a joined putative event into one action per entity name.

  Args:
    text: Lines of the form `"Name: action"`; an action may span lines.
    names: Known entity names.

  Returns:
    The action text keyed by entity name, in order of appearance.
  """
  actions: dict[str, list[str]] = {}
  current = None
  for line in text.splitlines():
    owner = next((n for n in names if line.startswith(f'{n}:')), None)
    if owner is not None:
      current = owner
      actions[current] = [line[len(owner) + 1 :].strip()]
    elif current is not None:
      actions[current].append(line)
  return {name: '\n'.join(parts).strip() for name, parts in actions.items()}


class ActorResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Resolves each entity's action through its `Actor`."""

  def __init__(self, context: game_lib.GameContext):
    super().__init__()
    self._context = context
    self._putative_event = ''

  @override
  def pre_observe(self, observation: str) -> str:
    if observation.startswith(PUTATIVE_EVENT_TAG):
      self._putative_event = observation[len(PUTATIVE_EVENT_TAG) :].strip()
    return ''

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    actions = split_actions(self._putative_event, list(self._context.actors))
    outcomes = []
    # The engine joins acts in thread-completion order; resolve them in the
    # fixed order actors were registered, so runs stay reproducible (two
    # offers to one partner in a tick must not race).
    for name in [n for n in self._context.actors if n in actions]:
      action = actions[name]
      actor = self._context.actors[name]
      actor.acts += 1
      outcome = actor.resolve(action)
      if not outcome:
        continue
      # Kept for the player's page: what their move did, and on which tick.
      actor.log.append((self._context.planet.tick, action, outcome))
      del actor.log[:-40]
      self._context.observe(name, outcome)
      if actor.outcome_topic:
        self._context.bus.publish(actor.outcome_topic, outcome, source=name)
      for topic, text in actor.drain_outbox():
        self._context.bus.publish(topic, text, source=name)
      outcomes.append(outcome)
    self._putative_event = ''
    result = ' '.join(outcomes)
    self._logging_channel({
        'Key': 'Resolution',
        'Summary': result,
        'Value': {'actions': actions, 'outcomes': outcomes},
    })
    return result

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'putative_event': self._putative_event}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._putative_event = str(state.get('putative_event', ''))
