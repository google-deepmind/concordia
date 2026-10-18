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

"""The take mind: a people's one-sentence take on what it heard.

Wrapped in a `pubsub.Publisher` in each people, it runs in the background
like the press mind. When the Gazette has printed a retelling of an omen
this people has not yet weighed, it asks, in the people's own voice (its
character is a sibling component), what they make of it and what they will
do. The answer is one sentence, published on `civ.<key>.takes`: the game
keeps it with the Gazette for the page, and the people hears its own take.
A take decides nothing by itself; the people still acts through its moves.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from examples.deep_time import game as game_lib
from concordia.language_model import language_model
from concordia.typing import entity_component

QUESTION = (
    'The Gazette, printed by the {owner}, says of the sign: "{retelling}".'
    ' In one sentence, in your own voice: what do {name} make of it, and what'
    ' will you do about it? Answer as the {name} themselves, saying "we", in'
    ' at most {words} words.'
)
# A take is one short sentence: what is said beyond this is cut.
TAKE_WORDS = 30


def short_take(answer: str) -> str:
  """The first line of an answer, cut to `TAKE_WORDS` words."""
  words = answer.strip().split('\n')[0].split()
  if len(words) <= TAKE_WORDS:
    return ' '.join(words)
  return ' '.join(words[:TAKE_WORDS]).rstrip(',;:') + '…'


class TakeMind(action_spec_ignored.ActionSpecIgnored):
  """One sentence on the newest retelling, in the people's own voice."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      context: game_lib.GameContext,
      owner: str,
      components: Sequence[str] = (),
      pre_act_label: str = 'Take mind',
  ):
    """Initializes the mind.

    Args:
      model: The slow model.
      context: Shared game state (the Gazette is the planet's).
      owner: Name of the people that prints the Gazette.
      components: Sibling components shown to the model (who the people
        are, how the world works, its storehouse).
      pre_act_label: Component label.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._context = context
    self._owner = owner
    self._components = tuple(components)
    # How many retellings this people has weighed (or passed over).
    self._taken = 0

  @override
  def _make_pre_act_value(self) -> str:
    gazette = self._context.planet.gazette
    told = self._context.bus.latest('media.omens')
    if gazette is None or told is None:
      return ''
    printed = sum(1 for sheet in gazette.printed if sheet.kind == 'omen')
    if printed <= self._taken:
      return ''
    self._taken = printed
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    answer = prompt.open_question(
        QUESTION.format(
            owner=self._owner,
            retelling=told.payload.strip(),
            name=self.get_entity().name,
            words=TAKE_WORDS,
        ),
        max_tokens=120,
        terminators=(),  # Some served models refuse stop words.
    )
    return short_take(answer)

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'taken': self._taken}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._taken = int(state.get('taken', 0))  # pyrefly: ignore[bad-argument-type]
