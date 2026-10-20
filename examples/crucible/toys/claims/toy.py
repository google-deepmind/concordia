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

"""Toy A, "claims": an argument whose physics is meaning.

Each move adds one short claim to a shared board. A language model reads the
new claim against every standing claim and says whether it supports,
contradicts or is unrelated to it; that is the only judgement it makes.
The board's weights, falls and cascades follow in code (`world.py`).

A small change in wording changes which claims a move touches, and so which
claims fall and which of their neighbours follow them down.
"""

from collections.abc import Mapping
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.crucible.toys import base
from examples.crucible.toys.claims import world as world_lib

RELATION_STATEMENT = (
    'Two claims from a public argument.\nClaim A: {a}\nClaim B: {b}\n'
)
RELATION_QUESTION = 'If claim A is true, what does that do to claim B?'
RELATION_OPTIONS = (
    'A supports B',
    'A contradicts B',
    'A says the same as B',
    'A is unrelated to B',
)
_RELATION_OF = {
    0: world_lib.AGREE,
    1: world_lib.CONFLICT,
    2: world_lib.RESTATES,
    3: world_lib.UNRELATED,
}
MOVE_MARKER = 'Write your next claim'


class ClaimsToy(base.Toy):
  """Two sides add claims to one board until their moves or ink run out."""

  name = 'claims'

  def __init__(
      self,
      model: language_model.LanguageModel,
      data: Mapping[str, Any],
      scenario: str = 'sandbox',
      seed: int = 0,
  ):
    """Builds the board of a scenario and judges its starting relations.

    Args:
      model: Judges the relation between two claims (the interpreter).
      data: The parsed `data/claims.yaml`.
      scenario: Key under `scenarios`.
      seed: Unused: the board is the same for every seed.
    """
    super().__init__()
    del seed
    self._model = model
    # Judgements shared by previews and the moves they preview.
    self._cache: dict[str, str] = {}
    self._data = data
    self.physics = world_lib.Physics.from_dict(data['physics'])
    self.rules = dict(data['rules'])
    self._actors = list(data['actors'])
    self._sides = list(data['actors'])  # (player, rival), for the score
    self.scenario = scenario
    spec = data['scenarios'][scenario]
    board = world_lib.Board()
    for actor in self._actors:
      board.ink[actor] = int(self.rules['ink'])
      board.moves[actor] = 0
    self.world = board
    for actor, text in spec['theses'].items():
      self._place(board, text, actor, float(self.rules['thesis_base']), True)
    first = data.get('first', self._actors[0])
    self._actors = [first] + [a for a in self._actors if a != first]
    for text in spec.get('board', []):
      self._place(board, text, '', float(self.rules['fact_base']), False)
    world_lib.cascade(board, self.physics)
    for actor in self._actors:
      self.tell(actor, str(data['premise']).strip())

  # ---- interpretation ---------------------------------------------------

  def judge(self, board: world_lib.Board, a: str, b: str) -> str:
    """The model's reading of what claim `a` does to claim `b` (cached)."""
    key = world_lib.judged_key(a, b)
    if key not in board.judged:
      if key not in self._cache:
        doc = interactive_document.InteractiveDocument(self._model)
        doc.statement(RELATION_STATEMENT.format(a=a, b=b))
        idx = doc.multiple_choice_question(
            RELATION_QUESTION, list(RELATION_OPTIONS), randomize_choices=False
        )
        self._cache[key] = _RELATION_OF.get(idx, world_lib.UNRELATED)
      board.judged[key] = self._cache[key]
    return board.judged[key]

  def _place(
      self,
      board: world_lib.Board,
      text: str,
      author: str,
      base_weight: float,
      thesis: bool,
  ) -> tuple[world_lib.Claim, bool]:
    """Adds a claim, or folds a restatement into the claim it restates.

    Args:
      board: The board.
      text: The claim.
      author: Its author.
      base_weight: Its base weight if it is new.
      thesis: Whether it is its author's thesis.

    Returns:
      The new claim (or the one it restates), and whether it is new.
    """
    relations = {
        c.key: self.judge(board, text, c.text) for c in board.standing()
    }
    restated = [
        k for k, kind in relations.items() if kind == world_lib.RESTATES
    ]
    if restated and not thesis:
      claim = board.claim(restated[0])
      claim.base = min(
          float(self.rules['max_base']),
          claim.base + float(self.rules['restate_bonus']),
      )
      return claim, False
    claim = world_lib.add_claim(
        board, text, author, relations, base=base_weight, thesis=thesis
    )
    return claim, True

  # ---- turn order and moves ---------------------------------------------

  def actors(self) -> list[str]:
    return list(self._actors)

  def opponent(self, actor: str) -> str:
    return next(a for a in self._actors if a != actor)

  def max_steps(self) -> int:
    return int(self.rules['moves_each']) * len(self._actors)

  def render(self, board: world_lib.Board, viewer: str) -> str:
    """The board as `viewer` sees it (everything is public)."""
    lines = ['Standing claims (key, weight, author):']
    for c in board.standing():
      who = 'board' if not c.author else c.author
      tag = ' THESIS' if c.thesis else ''
      lines.append(f'  {c.key} [{c.weight:.2f}] ({who}{tag}) {c.text}')
    fallen = [c for c in board.claims if not c.standing]
    if fallen:
      lines.append('Fallen claims:')
      lines.extend(
          f'  {c.key} ({c.author or "board"}) {c.text}' for c in fallen
      )
    lines.append(
        f'A claim falls below {self.physics.fall_below:.2f}. '
        f'Ink left: {viewer} {board.ink[viewer]} words, '
        f'{self.opponent(viewer)} {board.ink[self.opponent(viewer)]} words. '
        f'Moves left: {self.moves_left(board, viewer)}.'
    )
    return '\n'.join(lines)

  def moves_left(self, board: world_lib.Board, actor: str) -> int:
    return int(self.rules['moves_each']) - board.moves[actor]

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    board = self.world
    mine = board.thesis_of(actor)
    theirs = board.thesis_of(self.opponent(actor))
    goal = (
        f'Your thesis is {mine.key} "{mine.text}". '
        f'Your rival\'s thesis is {theirs.key} "{theirs.text}". '
        if mine and theirs
        else ''
    )
    call = (
        f'{self.render(board, actor)}\n{goal}'
        f'{MOVE_MARKER}, {{name}}: one claim of at most '
        f'{self.rules["max_words"]} words that keeps your thesis standing '
        'and brings your rival\'s down, or "pass". Answer with the claim '
        'only.'
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='claim')

  def apply(
      self, world: world_lib.Board, actor: str, action: str
  ) -> base.Receipt:
    board = world
    board.moves[actor] = board.moves.get(actor, 0) + 1
    before_mine = board.side_weight(actor)
    before_theirs = board.side_weight(self.opponent(actor))
    words = action.split()
    if not action or action.lower().rstrip('.') == base.PASS:
      receipt = base.Receipt(actor, base.PASS, f'{actor} passes.')
      board.history.append({'actor': actor, 'claim': None, 'fell': []})
      return receipt
    refusal = ''
    if len(words) > int(self.rules['max_words']):
      refusal = (
          f'is {len(words)} words, over the limit of {self.rules["max_words"]}'
      )
    elif len(words) > board.ink[actor]:
      refusal = (
          f'needs {len(words)} words of ink; {actor} has {board.ink[actor]}'
      )
    if refusal:
      board.history.append({'actor': actor, 'claim': None, 'fell': []})
      return base.Receipt(
          actor,
          action,
          f'{actor}\'s claim "{action}" {refusal}: nothing is added.',
      )
    board.ink[actor] -= len(words)
    claim, is_new = self._place(
        board, action, actor, float(self.rules['claim_base']), False
    )
    if not is_new:
      world_lib.cascade(board, self.physics)
      board.history.append({'actor': actor, 'claim': claim.key, 'fell': []})
      summary = (
          f'{actor}\'s claim "{action}" says the same as {claim.key}: '
          f'{claim.key} grows a little firmer ({claim.weight:.2f}).'
      )
      return base.Receipt(actor, action, summary, [summary])
    touched = [
        (k, kind)
        for k, kind in (
            (c.key, board.relation(claim.key, c.key))
            for c in board.claims
            if c.key != claim.key
        )
        if kind != world_lib.UNRELATED
    ]
    beats = world_lib.cascade(board, self.physics)
    steps = [f'{actor} adds {claim.key} "{claim.text}".']
    if touched:
      steps.append(
          'It '
          + '; '.join(
              f'{"supports" if kind == world_lib.AGREE else "contradicts"} {k}'
              for k, kind in touched
          )
          + '.'
      )
    else:
      steps.append('It touches no standing claim.')
    steps.extend(world_lib.summarize_beats(board, beats))
    fell = [k for beat in beats for k in beat]
    if claim.standing:
      steps.append(f'{claim.key} stands at {claim.weight:.2f}.')
    elif not any(claim.key in beat for beat in beats[1:]):
      steps.append(f'{claim.key} finds no footing and falls.')
    delta = {
        'mine': round(board.side_weight(actor) - before_mine, 3),
        'theirs': round(
            board.side_weight(self.opponent(actor)) - before_theirs, 3
        ),
        'fell': float(len(fell)),
    }
    board.history.append({'actor': actor, 'claim': claim.key, 'fell': fell})
    return base.Receipt(actor, action, ' '.join(steps), steps, delta)

  # ---- end of game ------------------------------------------------------

  def done(self) -> bool:
    board = self.world
    return all(
        self.moves_left(board, a) <= 0 or board.ink[a] <= 0
        for a in self._actors
    )

  def outcome(self) -> dict[str, Any]:
    board = self.world
    standing = {
        a: bool(board.thesis_of(a) and board.thesis_of(a).standing)
        for a in self._actors
    }
    winner = None
    for a in self._actors:
      if standing[a] and not standing[self.opponent(a)]:
        winner = a
    player, rival = self._sides
    return {
        'winner': winner,
        'theses_standing': standing,
        'score': round(board.side_weight(player) - board.side_weight(rival), 3),
        'claims': len(board.claims),
        'fallen': sum(1 for c in board.claims if not c.standing),
    }

  # ---- state ------------------------------------------------------------

  def world_to_dict(self) -> Mapping[str, Any]:
    return self.world.to_dict()

  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    self.world = world_lib.Board.from_dict(data)
