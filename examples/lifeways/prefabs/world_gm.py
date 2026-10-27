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

`__next_game_master__` is omitted: with a single game master the engine never
asks for it. The world is built from the GM's own params (`settings`), so no
object is shared with any entity.
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
from concordia.environment.engines import asynchronous
from examples.lifeways.components import game_master as gm_components
from examples.lifeways.prefabs import measures
from examples.lifeways.prefabs import person as person_lib
from examples.lifeways.world import world as world_lib
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

WORLD_GM_PREFAB = 'lifeways_world__GameMaster'
WORLD_GM_NAME = 'Lifeways'


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
    world = world_lib.World(settings)
    world.found(self.params.get('founders', ()))
    lifecycle = self.params.get('lifecycle') or gm_components.Lifecycle()
    players = self.entities if self.entities is not None else []
    roster = player_roster.PlayerRoster(
        players=players,
        player_prefabs={
            person_lib.PERSON_PREFAB: (
                self.params.get('person_prefab') or person_lib.Person()
            )
        },
        model=model,
        embedder=person_lib.hash_embedder,
    )
    queue = make_observation.ObservationQueue()
    gate = gm_components.TurnGate(
        lifecycle=lifecycle,
        observation_queue=queue,
        person_prefab=person_lib.PERSON_PREFAB,
        max_years=settings.get('max_years'),
        free=settings.get('free'),
    )
    # The first players' names, for the stock components that take fixed
    # names. They need no others: next acting is the `TurnGate`'s, which asks
    # the World, and the World addresses observations by name, never to all.
    names = [entity.name for entity in self.entities or ()]
    components = {
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
    gate.deliver(world)  # The founders' first observations.
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=str(self.params.get('name', WORLD_GM_NAME)),
        act_component=switch_act.SwitchAct(model=model, entity_names=names),
        context_components=components,
        measurements=measures.measurements(),
    )
