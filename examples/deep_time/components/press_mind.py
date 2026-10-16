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

"""The press mind: a people's printers write a broadsheet for the Gazette.

Wrapped in a `pubsub.Publisher`, it runs in the background. It is gated like
the pact mind: unless the people has just commissioned a broadsheet (paid
paper for it), it returns at once and costs nothing. Its answer (a stance,
a headline and the body) is published on `civ.<key>.press`, where the game
posts it on the Gazette (a contrib `forum.ForumState`) for every people to
read and judge.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from examples.deep_time import game as game_lib
from concordia.language_model import language_model


class PressMind(action_spec_ignored.ActionSpecIgnored):
  """Writes a commissioned broadsheet for one people."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      context: game_lib.GameContext,
      polity: str,
      components: Sequence[str] = (),
      pre_act_label: str = 'Press mind',
  ):
    """Initializes the press.

    Args:
      model: The slow model.
      context: Shared game state (the Gazette lives on the planet).
      polity: The people's key.
      components: Sibling components shown to the model.
      pre_act_label: Component label.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._context = context
    self._polity = polity
    self._components = tuple(components)

  @override
  def _make_pre_act_value(self) -> str:
    gazette = self._context.planet.gazette
    if gazette is None or self._polity not in gazette.pending:
      return ''
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    return prompt.open_question(
        gazette.question_for(self._polity).format(
            agent_name=self.get_entity().name
        ),
        max_tokens=400,
        terminators=(),
    ).strip()
