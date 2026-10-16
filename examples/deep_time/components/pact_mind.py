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
`bio.species.<key>.pact` and settled by the World GM.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from examples.deep_time import game as game_lib
from concordia.language_model import language_model


class PactMind(action_spec_ignored.ActionSpecIgnored):
  """Answers pending pact proposals addressed to one species."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      context: game_lib.GameContext,
      species: str,
      components: Sequence[str] = (),
      pre_act_label: str = 'Pact mind',
  ):
    """Initializes the mind.

    Args:
      model: Model that deliberates.
      context: Shared game state (pending proposals live on the planet).
      species: The species this mind speaks for.
      components: Sibling components shown to the model (e.g. genome, census).
      pre_act_label: Component label.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._context = context
    self._species = species
    self._components = tuple(components)

  def _pending(self) -> tuple[str, str, str] | None:
    """(proposal text, proposer's name, question template), or None."""
    symbiosis = self._context.planet.symbiosis
    proposal = symbiosis.pending_for(self._species) if symbiosis else None
    if proposal is None or symbiosis is None:
      return None
    proposer = self._context.planet.species_name(proposal.proposer)
    return proposal.text, proposer, symbiosis.question

  @override
  def _make_pre_act_value(self) -> str:
    pending = self._pending()
    if pending is None:
      return ''
    text, proposer, question = pending
    name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    answer = prompt.open_question(
        question.format(proposal=text, agent_name=name),
        max_tokens=400,
        terminators=(),
    )
    return f'Reply to {proposer}: {answer.strip()}'


class TreatyMind(PactMind):
  """Answers treaty proposals (of any kind) addressed to one people."""

  def _pending(self) -> tuple[str, str, str] | None:
    planet = self._context.planet
    treaties, civ = planet.treaties, planet.civ
    proposal = treaties.pending_for(self._species) if treaties else None
    if proposal is None or treaties is None or civ is None:
      return None
    # The reply names the offer's kind, so it settles only that offer.
    what = treaties.kinds[proposal.kind].name
    return (
        proposal.text,
        f'{civ.polities[proposal.proposer].name} ({what})',
        treaties.question,
    )
