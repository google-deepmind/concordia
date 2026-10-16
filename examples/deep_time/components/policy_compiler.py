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

"""The policy compiler: a people's slow mind writes its standing orders.

Wrapped in a `pubsub.Publisher`, it runs in the background a few ticks at a
time. Once the people can write reflexes (its techs unlock the
`threshold` operator) it asks the slow model for standing orders as JSON;
before that it returns at once and costs nothing. Its answer is published on
`civ.<key>.policy`, where the game parses it with the contrib
`reflexes.parse_specs`, checks that each reflex only reads public topics and
only places this people's own orders, and installs it.
"""

from collections.abc import Sequence
from typing import override

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from examples.deep_time import game as game_lib
from concordia.language_model import language_model

QUESTION = """\
You are the scribes of {name}. Write the standing orders your people should
keep: each one watches the census and places a market order whenever a stock
crosses a level, so nobody has to remember. Use this JSON form, a list of up
to three orders:
[{{"name": "sell_ore", "input": "civ.census",
  "pipeline": [{{"op": "json_field", "path": "polities.{key}.stocks.ore"}},
               {{"op": "threshold", "above": 6}}],
  "emit": {{"topic": "civ.{key}.orders", "template": "ask ore"}},
  "cooldown": 2}}]
"ask <good>" sells and "bid <good>" buys. Goods: {goods}. Operators you may
use: {ops}. Answer with the JSON list only."""


class PolicyCompiler(action_spec_ignored.ActionSpecIgnored):
  """Asks the slow model for standing orders once reflexes are unlocked."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      context: game_lib.GameContext,
      polity: str,
      components: Sequence[str] = (),
      pre_act_label: str = 'Policy compiler',
  ):
    """Initializes the compiler.

    Args:
      model: The slow model.
      context: Shared game state (automation decides what is unlocked).
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
    automation = self._context.automation
    if automation is None:
      return ''
    ops = automation.allowed_ops(self._polity)
    if 'threshold' not in ops:
      return ''
    prompt = interactive_document.InteractiveDocument(self._model)
    for key in self._components:
      prompt.statement(f'{key}: {self.get_named_component_pre_act_value(key)}')
    question = QUESTION.format(
        name=self.get_entity().name,
        key=self._polity,
        goods=', '.join(sorted(automation.levels)),
        ops=', '.join(sorted(ops)),
    )
    return prompt.open_question(question, max_tokens=600, terminators=())
