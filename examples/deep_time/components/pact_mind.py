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

"""Pact minds: a species (or a people) deliberates over a proposal.

It is wrapped in `pubsub.Publisher(..., trigger_on_observe=True)`, so a
proposal arriving as an observation starts the deliberation at once
rather than at the species' next act. Most triggers
find no pending proposal and return immediately without calling the model;
only a real proposal costs a language model call. The answer is published on
`bio.species.<key>.pact` (a people's on `civ.<key>.treaty`); the entity
answers by its own action. The offer it weighs is a sheet the World GM
publishes for its entity (`perception.brief`), never the world itself.
"""

from collections.abc import Callable, Sequence
import json
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model


def offer_sheet(proposal: str, proposer: str, question: str) -> str:
  """The sheet a pact mind weighs (the World GM publishes it)."""
  return json.dumps(
      {'proposal': proposal, 'proposer': proposer, 'question': question}
  )


class PactMind(action_spec_ignored.ActionSpecIgnored):
  """Answers the pact (or treaty) offer awaiting one species or people."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      offer: Callable[[], str],
      components: Sequence[str] = (),
      pre_act_label: str = 'Pact mind',
  ):
    """Initializes the mind.

    Args:
      model: Model that deliberates.
      offer: The latest offer sheet the World GM published for this entity
        (`offer_sheet`), or '' when no offer awaits its answer.
      components: Sibling components shown to the model (e.g. genome, census).
      pre_act_label: Component label.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._offer = offer
    self._components = tuple(components)

  @override
  def _make_pre_act_value(self) -> str:
    sheet = self._offer()
    if not sheet:
      return ''
    offer = json.loads(sheet)
    name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    answer = prompt.open_question(
        str(offer['question']).format(
            proposal=offer['proposal'], agent_name=name
        ),
        max_tokens=400,
        terminators=(),
    )
    return f'Reply to {offer["proposer"]}: {answer.strip()}'
