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

"""The World GM of Lifeways: `SwitchAct` over the world's components.

| Key | Component |
|---|---|
| `__next_acting__` | `TurnGate`: who may act now; closes each year |
| `__next_action_spec__` | `Questions`: what each person is asked |
| `__resolution__` | `Answers`: records each answer for the year's close |
| `__make_observation__` | stock `MakeObservation` on the world's queue |
| `__terminate__` | stock `Terminate`, told by `TurnGate` at `max_years` |
| `lifeways_world` | `WorldComponent`: the world, its state the GM's |
| `__social_ties__` | contrib `SocialTies`: who is close to whom (the `ties` setting) |
| `__rumors__` | contrib `Rumors`: the word going round (the `rumours` setting) |
| `__reputation__` | contrib `Reputation`: what people think of one another (the `reputation` setting) |
| `__reciprocity__` | contrib `Reciprocity`: who owes whom food they were given (the `debts` setting) |
| `__contagion__` | contrib `Contagion`: a sickness passing between those who meet (the `sickness` setting) |

`__next_game_master__` is omitted: with a single game master the engine never
asks for it. The world is built from the GM's own params (`settings`), so no
object is shared with any entity.

The mechanics from `concordia/contrib` are GM components under their own keys:
this prefab builds each from the settings, registers it, and hands the same
object to the `World`, which calls it as a year closes. The GM saves each
with the rest of its state.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import player_roster
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.contrib.components.game_master import contagion
from concordia.contrib.components.game_master import reciprocity
from concordia.contrib.components.game_master import reputation as reputation_lib
from concordia.contrib.components.game_master import rumors as rumors_lib
from concordia.contrib.components.game_master import social_ties
from concordia.environment.engines import asynchronous
from examples.lifeways.components import game_master as gm_components
from examples.lifeways.prefabs import measures
from examples.lifeways.prefabs import person as person_lib
from examples.lifeways.world import world as world_lib
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

WORLD_GM_PREFAB = 'lifeways_world__GameMaster'
WORLD_GM_NAME = 'Lifeways'


def make_mechanics(settings: Mapping[str, Any]) -> world_lib.Mechanics:
  """The contrib mechanics a world's settings ask for, each built anew.

  Args:
    settings: The world's settings (`scenarios.settings`).

  Returns:
    The mechanics (one left out where its setting is missing or None).
  """
  ties = None
  if settings.get('ties'):
    tie_settings = dict(settings['ties'])
    ties = social_ties.SocialTies(
        decay=float(tie_settings.get('decay', social_ties.DECAY)),
        floor=float(tie_settings.get('floor', social_ties.FLOOR)),
        max_ties=int(tie_settings.get('max', social_ties.MAX_TIES)),
    )
  rumors = None
  if settings.get('rumours'):
    word = dict(settings['rumours'])
    rumors = rumors_lib.Rumors(
        seed=int(settings.get('seed', 0)),
        chance=float(word.get('chance', rumors_lib.CHANCE)),
        tell_for=int(word.get('tell_for', rumors_lib.TELL_FOR)),
        keep=float(word.get('keep', rumors_lib.KEEP)),
        garble=float(word.get('garble', rumors_lib.GARBLE)),
        forget_after=int(word.get('forget_after', rumors_lib.FORGET_AFTER)),
        max_rumors=int(word.get('max', rumors_lib.MAX_RUMORS)),
    )
  reputation = None
  if settings.get('reputation'):
    report = dict(settings['reputation'])
    reputation = reputation_lib.Reputation(
        decay=float(report.get('decay', reputation_lib.DECAY)),
        floor=float(report.get('floor', reputation_lib.FLOOR)),
        max_known=int(report.get('max_known', reputation_lib.MAX_KNOWN)),
    )
  debts = None
  if settings.get('debts'):
    owed = dict(settings['debts'])
    debts = reciprocity.Reciprocity(
        term=int(owed.get('term', reciprocity.TERM)),
        return_share=float(owed.get('return_share', reciprocity.RETURN_SHARE)),
        max_open=int(owed.get('max_open', reciprocity.MAX_OPEN)),
    )
  sickness = None
  if settings.get('sickness'):
    sick = dict(settings['sickness'])
    sickness = contagion.Contagion(
        seed=int(settings.get('seed', 0)),
        transmission=float(sick.get('transmission', contagion.TRANSMISSION)),
        incubation=int(sick.get('incubation', contagion.INCUBATION)),
        infectious_for=int(
            sick.get('infectious_for', contagion.INFECTIOUS_FOR)
        ),
        immunity_for=int(sick.get('immunity_for', contagion.IMMUNITY_FOR)),
        mortality=float(sick.get('mortality', contagion.MORTALITY)),
    )
  return world_lib.Mechanics(
      ties=ties,
      rumors=rumors,
      reputation=reputation,
      debts=debts,
      sickness=sickness,
  )


@dataclasses.dataclass
class WorldGameMaster(prefab_lib.Prefab):
  """A deterministic World GM: the land, the people's lives, the year.

  Params:
    name: The GM's name.
    settings: The world's settings (`scenarios.settings`).
    founders: The founders the scenario made (`scenarios.founders`), the
      same records whose `params` build the founders' entities.
    lifecycle: Optional. What the harness is told as the world plays
      (`Lifecycle`). A live object: it is not serializable, so checkpoints
      drop it and the restoring harness wires its own.
    person_prefab: The Config's person prefab, which the born are built from
      (a default `Person` if not given). Checkpoints drop it too; the Config
      gives it again on restore.
    played: Optional. The founder someone plays. Once it dies, the player
      lives on as someone born of its line (`World.heir`), built from
      `played_prefab`.
    played_prefab: Optional. The prefab a played person's heir is built from
      (a default `Person` if not given: in a twin, minds play the heir).

  The born join the game and the dead leave it through the GM's stock
  `PlayerRoster`, which holds the simulation's entities (`self.entities`).
  """

  description: str = (  # pyrefly: ignore[bad-override]
      'Lifeways World GM: steps the land and the people a year at a time and'
      ' resolves their answers by rule. Makes no language model calls.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': WORLD_GM_NAME, 'settings': {}}
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the World GM.

    Args:
      model: Passed to `SwitchAct` and `MakeObservation`, whose language
        model paths are never taken.
      memory_bank: Unused; the World GM keeps no associative memory.

    Returns:
      The game master entity.
    """
    del memory_bank
    self.check_extra_components()
    settings = dict(self.params.get('settings', {}))
    mechanics = make_mechanics(settings)
    world = world_lib.World(settings, mechanics)
    world.found(self.params.get('founders', ()))
    lifecycle = self.params.get('lifecycle') or gm_components.Lifecycle()
    players = self.entities if self.entities is not None else []
    roster = player_roster.PlayerRoster(
        players=players,
        player_prefabs={
            person_lib.PERSON_PREFAB: (
                self.params.get('person_prefab') or person_lib.Person()
            ),
            person_lib.PLAYED_PREFAB: (
                self.params.get('played_prefab') or person_lib.Person()
            ),
        },
        model=model,
        embedder=person_lib.hash_embedder,
    )
    queue = make_observation.ObservationQueue()
    gate = gm_components.TurnGate(
        lifecycle=lifecycle,
        observation_queue=queue,
        person_prefab=person_lib.PERSON_PREFAB,
        played=self.params.get('played'),
        played_prefab=person_lib.PLAYED_PREFAB,
        max_years=settings.get('max_years'),
        free=settings.get('free'),
    )
    # The first players' names, for the stock components that take fixed
    # names. They need no others: next acting is the `TurnGate`'s, which asks
    # the World, and the World addresses observations by name, never to all.
    names = [entity.name for entity in self.entities or ()]
    components: dict[str, entity_component.ContextComponent] = {
        gm_components.DEFAULT_WORLD_KEY: gm_components.WorldComponent(world),
        player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY: roster,
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: gate,
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            gm_components.Questions()
        ),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: gm_components.Answers(),
        make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model=model,
                player_names=names,
                call_to_make_observation=(
                    asynchronous.DEFAULT_CALL_TO_MAKE_OBSERVATION
                ),
                external_queue=queue,
                allow_llm_fallback=False,
            )
        ),
        terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminate.Terminate(),
    }
    if mechanics.ties is not None:
      components[social_ties.DEFAULT_SOCIAL_TIES_COMPONENT_KEY] = mechanics.ties
    if mechanics.rumors is not None:
      components[rumors_lib.DEFAULT_RUMORS_COMPONENT_KEY] = mechanics.rumors
    if mechanics.reputation is not None:
      components[reputation_lib.DEFAULT_REPUTATION_COMPONENT_KEY] = (
          mechanics.reputation
      )
    if mechanics.debts is not None:
      components[reciprocity.DEFAULT_RECIPROCITY_COMPONENT_KEY] = (
          mechanics.debts
      )
    if mechanics.sickness is not None:
      components[contagion.DEFAULT_CONTAGION_COMPONENT_KEY] = mechanics.sickness
    gate.deliver(world)  # The founders' first observations.
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=str(self.params.get('name', WORLD_GM_NAME)),
        act_component=switch_act.SwitchAct(model=model, entity_names=names),
        context_components=components,
        measurements=measures.measurements(),
    )
