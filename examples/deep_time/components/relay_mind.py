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

"""The relay mind: the presses' owners retell the Steward's omens.

Wrapped in a `pubsub.Publisher` in the people that owns the presses, it runs
in the background like the press mind. When the Steward has spoken an omen
it has not yet retold, its printers write the Gazette's front page about it,
in character: a stance, a headline and two sentences, published on
`civ.<key>.retelling`. The game posts it on the Gazette, where its stance
sways every people's consent for coal as a broadsheet's does, and on
`media.omens`, which every people hears. The retelling is the printers'
reading of the sign, not the sign: the peoples never hear the Steward's
words but through them.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from examples.deep_time import game as game_lib
from concordia.language_model import language_model
from concordia.typing import entity_component


class RelayMind(action_spec_ignored.ActionSpecIgnored):
  """Retells the Steward's newest omen as the printers see it."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      context: game_lib.GameContext,
      components: Sequence[str] = (),
      pre_act_label: str = 'Relay mind',
  ):
    """Initializes the relay.

    Args:
      model: The slow model.
      context: Shared game state (the omens and the Gazette are the planet's).
      components: Sibling components shown to the model.
      pre_act_label: Component label.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._context = context
    self._components = tuple(components)
    # How many of the planet's omens have been retold (or passed over).
    self._retold = 0

  @override
  def _make_pre_act_value(self) -> str:
    planet = self._context.planet
    gazette = planet.gazette
    if gazette is None or len(planet.omens) <= self._retold:
      return ''
    # The newest omen; any spoken while the printers were busy go untold.
    _, sign, words = planet.omens[-1]
    self._retold = len(planet.omens)
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    return prompt.open_question(
        str(gazette.relay['question']).format(
            agent_name=self.get_entity().name,
            printers=' '.join(
                str(
                    self._context.scenario['civilization'].get('printers')
                    or gazette.relay['printers']
                ).split()
            ),
            sign=sign,
            words=words or '(nothing anyone could make out)',
        ),
        max_tokens=400,
        terminators=(),
    ).strip()

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'retold': self._retold}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._retold = int(state.get('retold', 0))  # pyrefly: ignore[bad-argument-type]
