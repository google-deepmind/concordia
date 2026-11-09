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

"""The interface every crucible toy implements.

A toy is one candidate generative rule (GUIDE.md, Phase 1): deterministic
state that a language model's reading of a move, and a ledger, carry forward.
A toy is a `Rulebook` (`concordia/contrib/components/game_master/rulebook.py`):
the contrib rulebook game master drives it through Concordia's well-known
component keys, so a toy holds no Concordia plumbing of its own.

Language decides, a ledger accounts: a toy asks its `interpreter` (a language
model) only what words mean, and applies every consequence in code.
"""

import abc
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import re
from typing import Any

from concordia.typing import entity as entity_lib

PASS = 'pass'


@dataclasses.dataclass
class Receipt:
  """What one resolved move did.

  Attributes:
    actor: Who moved.
    action: The move as resolved (cleaned of the actor's name).
    summary: One line for the actor and the log.
    steps: The causal chain, in order: each line is one beat a client can
      replay with a pause between beats (GUIDE.md, resolution cadence).
    delta: Measured change in the quantities that decide the game.
  """

  actor: str
  action: str
  summary: str
  steps: list[str] = dataclasses.field(default_factory=list)
  delta: dict[str, float] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def __str__(self) -> str:
    return self.summary


def clean_action(actor: str, text: str) -> str:
  """The move an entity sent, without its name, quotes or extra lines.

  The engine joins `<name>: <action>`, and Concordia's `ConcatActComponent`
  may also start the action with the entity's name.

  Args:
    actor: The acting entity's name.
    text: The raw action text.

  Returns:
    The first non-empty line of the move, stripped.
  """
  text = text.strip()
  for prefix in (f'{actor}:', f'{actor} --', f'{actor} '):
    if text.startswith(prefix):
      text = text[len(prefix) :].strip()
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  text = lines[0] if lines else ''
  # Models sometimes wrap a move: 'Answer: Commander orders: "Build."'.
  text = re.sub(r'^(?:answer|response|reply)\s*:\s*', '', text, flags=re.I)
  wrapped = re.match(r'^[\w\' ]{1,40}:\s*["“](.+?)["”]\.?$', text)
  if wrapped:
    text = wrapped.group(1)
  else:
    # Narration around a move: '... he says plainly: "Ash, watch ahead.'
    quoted = re.search(r':\s*["“]([^"”]+)["”]?\.?$', text)
    if quoted and len(text) > 2 * len(quoted.group(1)):
      text = quoted.group(1)
  text = text.strip().strip('"“”\'').strip()
  return re.sub(r'\s+', ' ', text)


class Toy(abc.ABC):
  """A candidate generative rule with a deterministic, checkpointable state.

  Subclasses keep all mutable game state in `self.world`, which must be
  deep-copyable and JSON-safe through `get_state` / `set_state`. A preview
  runs the same resolution on a copy of the world, so every move can be
  measured before it is sent.
  """

  name: str = 'toy'

  def __init__(self) -> None:
    self._outbox: list[tuple[str, str]] = []
    self._turn = 0
    self._current: str | None = None

  # ---- turn order -------------------------------------------------------

  @abc.abstractmethod
  def actors(self) -> list[str]:
    """Entity names in turn order."""

  def next_actor(self) -> str:
    """The entity who acts this step (cycles through `actors`)."""
    order = self.actors()
    self._current = order[self._turn % len(order)]
    self._turn += 1
    return self._current

  # ---- moves --------------------------------------------------------------

  @abc.abstractmethod
  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    """What `actor` may do now, with the state rendered for them."""

  @abc.abstractmethod
  def apply(self, world: Any, actor: str, action: str) -> Receipt:
    """Resolves `action` on `world` (mutating it) and says what happened."""

  def resolve(self, actor: str, action: str) -> Receipt:
    """Resolves a move on the live world and queues its observations."""
    receipt = self.apply(self.world, actor, clean_action(actor, action))
    self.after_resolve(receipt)
    return receipt

  def preview(self, actor: str, action: str) -> Receipt:
    """What `action` would do now, measured on a copy of the world."""
    return self.simulate(actor, action)[0]

  def simulate(self, actor: str, action: str) -> tuple[Receipt, Any]:
    """Resolves `action` on a copy of the world; returns both."""
    world = copy.deepcopy(self.world)
    outbox = list(self._outbox)  # A preview tells no one anything.
    try:
      return self.apply(world, actor, clean_action(actor, action)), world
    finally:
      self._outbox = outbox

  def after_resolve(self, receipt: Receipt) -> None:
    """Hook: queue observations after a move (default: tell everyone)."""
    for name in self.actors():
      self.tell(name, receipt.summary)

  # ---- observations -------------------------------------------------------

  def tell(self, recipient: str, text: str) -> None:
    self._outbox.append((recipient, text))

  def drain_observations(self) -> list[tuple[str, str]]:
    out, self._outbox = self._outbox, []
    return out

  # ---- end of game ----------------------------------------------------------

  @abc.abstractmethod
  def done(self) -> bool:
    """Whether the game is over."""

  @abc.abstractmethod
  def outcome(self) -> dict[str, Any]:
    """The result: at least `winner` (a name or None) and `score`."""

  def max_steps(self) -> int:
    return 100

  # ---- state ----------------------------------------------------------------

  world: Any = None

  @abc.abstractmethod
  def world_to_dict(self) -> Mapping[str, Any]:
    """The world as JSON-safe data."""

  @abc.abstractmethod
  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    """Restores the world from `world_to_dict` output."""

  def get_state(self) -> dict[str, Any]:
    return {
        'world': dict(self.world_to_dict()),
        'turn': self._turn,
        'current': self._current,
        'outbox': [list(item) for item in self._outbox],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.world_from_dict(state['world'])
    self._turn = int(state['turn'])
    self._current = state['current']
    self._outbox = [(str(a), str(b)) for a, b in state['outbox']]
