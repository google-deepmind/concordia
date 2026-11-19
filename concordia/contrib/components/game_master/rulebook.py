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

"""Game master components that delegate to a rulebook: a plain Python object.

Many games are easiest to write as ordinary code: whose turn it is, what they
may do, what a move does, when the game ends. A `Rulebook` is that code, with
no Concordia plumbing in it. These components plug a rulebook into the
well-known game master keys that `switch_act.SwitchAct` dispatches to:

| Key | Component | Calls |
|---|---|---|
| `__next_acting__` | `RulebookNextActing` | `rulebook.next_actor()` |
| `__next_action_spec__` | `RulebookActionSpec` | `rulebook.action_spec(actor)` |
| `__resolution__` | `RulebookResolution` | `rulebook.resolve(actor, action)` |

`RulebookResolution` also delivers the rulebook's observations through a
`make_observation.ObservationQueue`, tells a `terminate.Terminate` component
to stop once `rulebook.done()`, and checkpoints the rulebook's state. The
prefab `contrib/prefabs/game_master/rulebook.py` assembles all of them.

A rulebook may still use a language model (for example, to judge what a
free-text move means); the game master itself never needs one.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, override

from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.components.game_master import terminate
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

ALL = 'all'


class Rulebook(Protocol):
  """The rules of a turn-based game, as a plain Python object."""

  def actors(self) -> Sequence[str]:
    """The names of the entities that play."""
    ...

  def next_actor(self) -> str:
    """The entity who acts next (called once per engine step)."""
    ...

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    """What `actor` may do now."""
    ...

  def resolve(self, actor: str, action: str) -> Any:
    """Applies `actor`'s move. `str()` of the result is the event."""
    ...

  def drain_observations(self) -> Sequence[tuple[str, str]]:
    """Observations queued since the last call: (recipient or 'all', text)."""
    ...

  def done(self) -> bool:
    """Whether the game is over."""
    ...

  def get_state(self) -> Mapping[str, Any]:
    """The rulebook's state, JSON-safe."""
    ...

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Restores the state `get_state` returned."""
    ...


def split_event(event: str, actors: Sequence[str]) -> tuple[str, str] | None:
  """Splits an engine event `<actor>: <action>` for a known actor.

  `ConcatActComponent` may also start the action with the actor's name, so a
  leading repetition of the name is removed too.

  Args:
    event: The putative event.
    actors: The known actors.

  Returns:
    (actor, action), or None if the event is not by a known actor.
  """
  for actor in sorted(actors, key=len, reverse=True):
    if event.startswith(f'{actor}:'):
      action = event[len(actor) + 1 :].strip()
      if action.startswith(f'{actor} '):
        action = action[len(actor) + 1 :].strip()
      return actor, action
  return None


class RulebookNextActing(entity_component.ContextComponent):
  """`__next_acting__`: whoever the rulebook says acts next."""

  def __init__(self, rulebook: Rulebook):
    super().__init__()
    self._rulebook = rulebook
    self.current_actor: str | None = None

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    self.current_actor = self._rulebook.next_actor()
    return self.current_actor

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'current_actor': self.current_actor}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    current = state.get('current_actor')
    self.current_actor = str(current) if current is not None else None


class RulebookActionSpec(entity_component.ContextComponent):
  """`__next_action_spec__`: the rulebook's spec for the current actor."""

  def __init__(self, rulebook: Rulebook, next_acting: RulebookNextActing):
    super().__init__()
    self._rulebook = rulebook
    self._next_acting = next_acting

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    actor = self._next_acting.current_actor or self._rulebook.actors()[0]
    return engine_lib.action_spec_to_string(self._rulebook.action_spec(actor))

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class RulebookResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """`__resolution__`: the rulebook resolves each move.

  It also delivers the rulebook's observations, stops the game when the
  rulebook is done, and holds the rulebook's state for checkpoints.
  """

  def __init__(
      self,
      rulebook: Rulebook,
      queue: make_observation.ObservationQueue,
      terminator: terminate.Terminate,
  ):
    """Initializes the component.

    Args:
      rulebook: The rules.
      queue: The queue `MakeObservation` delivers observations from.
      terminator: Told to terminate once the rulebook is done.
    """
    super().__init__()
    self._rulebook = rulebook
    self._queue = queue
    self._terminator = terminator
    self._putative_event = ''
    self.results: list[Any] = []

  def deliver(self) -> None:
    """Moves the rulebook's queued observations onto the shared queue."""
    actors = list(self._rulebook.actors())
    for recipient, text in self._rulebook.drain_observations():
      self._queue.add(recipient, text, actors)

  @override
  def pre_observe(self, observation: str) -> str:
    if observation.startswith(event_resolution.PUTATIVE_EVENT_TAG):
      self._putative_event = observation[
          len(event_resolution.PUTATIVE_EVENT_TAG) :
      ].strip()
    return ''

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    parsed = split_event(self._putative_event, list(self._rulebook.actors()))
    self._putative_event = ''
    if parsed is None:
      return ''
    result = self._rulebook.resolve(*parsed)
    self.results.append(result)
    self.deliver()
    if self._rulebook.done():
      self._terminator.terminate()
    event = str(result)
    to_dict = getattr(result, 'to_dict', None)
    self._logging_channel({
        'Key': 'Resolution',
        'Summary': event,
        'Value': to_dict() if callable(to_dict) else event,
    })
    return event

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        'rulebook': dict(self._rulebook.get_state()),
        'putative_event': self._putative_event,
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._rulebook.set_state(state['rulebook'])  # pyrefly: ignore[bad-argument-type]
    self._putative_event = str(state.get('putative_event', ''))
