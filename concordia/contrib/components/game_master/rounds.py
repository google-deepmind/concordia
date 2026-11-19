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

"""Rounds for the asynchronous engine: everyone answers, then all resolve.

The asynchronous engine runs each entity's observe-act loop in a thread of
its own, so who answers first depends on timing. `Rounds` makes such a
simulation reproducible. Each participant may act once a round; its answer
is recorded, not resolved. When the round's last answer is in, the round
resolves all at once: a resolver (another context component of the game
master, found by key) receives every answer in the order of the
participants' names. No answer takes effect before the round's last, so the
outcome never depends on which thread came first.

Install one `Rounds` at both `__next_acting__` and `__resolution__`, as the
stock `AsynchronousQuestionnaire` serves both:

  rounds = rounds_lib.Rounds(participants=['Ada', 'Bo'], resolver_key='market')
  components = {
      next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: rounds,
      switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: rounds,
      'market': Market(),  # Has resolve_round(round_number, answers).
      ...
  }

With a `deadline`, a round also closes after so many seconds, and whoever
has not answered does nothing that round: slow minds lag, and outcomes then
depend on timing (for live play).
"""

from collections.abc import Mapping, Sequence
import threading
import time
from typing import Any, Protocol, cast, override

from concordia.components.game_master import event_resolution
from concordia.components.game_master import terminate
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_NOTE = 'The answer is noted; the round resolves when everyone answers.'


class RoundResolver(Protocol):
  """A game master component that resolves a round's answers together."""

  def resolve_round(
      self, round_number: int, answers: Mapping[str, str]
  ) -> None:
    """Resolves a round.

    Args:
      round_number: The round, from 0.
      answers: Each participant's answer, in the order of their names.
    """


class Rounds(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Lets each participant act once a round; resolves the round together."""

  def __init__(
      self,
      participants: Sequence[str],
      resolver_key: str,
      max_rounds: int | None = None,
      deadline: float | None = None,
      terminate_key: str = terminate.DEFAULT_TERMINATE_COMPONENT_KEY,
      note: str = DEFAULT_NOTE,
  ):
    """Initializes the rounds.

    Args:
      participants: The names of those who answer each round.
      resolver_key: The key of the game master component that resolves a
        round (a `RoundResolver`).
      max_rounds: After this many rounds the game ends (the stock
        `Terminate` at `terminate_key` is told); None for never.
      deadline: If set, a round also closes after this many seconds.
      terminate_key: The key of the game master's stock `Terminate`.
      note: What the resolution step says of a recorded answer.
    """
    super().__init__()
    self._participants = list(participants)
    self._resolver_key = resolver_key
    self._max_rounds = max_rounds
    self._deadline = deadline
    self._terminate_key = terminate_key
    self._note = note
    self._lock = threading.RLock()
    self._round = 0
    self._answers: dict[str, str] = {}
    # Once the game ends nobody acts again: a thread that asked just before
    # the end must not start another round.
    self._ended = False
    self._began = time.monotonic()

  @property
  def round_number(self) -> int:
    with self._lock:
      return self._round

  def set_participants(self, names: Sequence[str]) -> None:
    """Changes who answers, from the current round on (e.g. births)."""
    with self._lock:
      self._participants = list(names)
      self._answers = {
          name: answer
          for name, answer in self._answers.items()
          if name in self._participants
      }

  @override
  def pre_observe(self, observation: str) -> str:
    tag = event_resolution.PUTATIVE_EVENT_TAG
    if not observation.startswith(tag):
      return ''
    name, _, answer = observation[len(tag) :].strip().partition(':')
    name = name.strip()
    with self._lock:
      if name in self._participants and name not in self._answers:
        self._answers[name] = answer.strip()
    return ''

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      return self._note
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    with self._lock:
      complete = all(name in self._answers for name in self._participants)
      late = (
          self._deadline is not None
          and time.monotonic() - self._began >= self._deadline
      )
      if (complete or late) and not self._ended:
        self._close_round()
      if self._ended:
        return ''
      return ','.join(
          name
          for name in action_spec.options
          if name in self._participants and name not in self._answers
      )

  def _close_round(self) -> None:
    answers = {name: self._answers[name] for name in sorted(self._answers)}
    resolver = cast(
        RoundResolver, self.get_entity().get_component(self._resolver_key)
    )
    resolver.resolve_round(self._round, answers)
    self._logging_channel({
        'Key': 'Round',
        'Summary': f'Round {self._round} resolved',
        'Value': answers,
    })
    self._round += 1
    self._answers = {}
    self._began = time.monotonic()
    if self._max_rounds is not None and self._round >= self._max_rounds:
      self._ended = True
      self.get_entity().get_component(
          self._terminate_key, type_=terminate.Terminate
      ).terminate()

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'round': self._round,
          'answers': dict(self._answers),
          'participants': list(self._participants),
          'ended': self._ended,
      }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    data: Mapping[str, Any] = state
    with self._lock:
      self._round = int(data.get('round', 0))
      self._answers = {
          str(k): str(v) for k, v in data.get('answers', {}).items()
      }
      self._participants = [str(n) for n in data.get('participants', ())]
      self._ended = bool(data.get('ended', False))
      self._began = time.monotonic()
