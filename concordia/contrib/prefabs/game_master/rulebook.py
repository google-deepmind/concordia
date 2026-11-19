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

"""A game master prefab that runs a rulebook (a plain Python rules object).

It assembles standard pieces around the contrib rulebook components:
`switch_act.SwitchAct` dispatches each engine call to the component under
its well-known key, so the game master makes no language model calls:

| Key | Component |
|---|---|
| `__terminate__` | `terminate.Terminate`, stopped when the rulebook is done |
| `__make_observation__` | `make_observation.MakeObservation` on a shared queue, no LLM fallback |
| `__next_acting__` | `rulebook.RulebookNextActing` |
| `__next_action_spec__` | `rulebook.RulebookActionSpec` |
| `__resolution__` | `rulebook.RulebookResolution` |

Use it with the `Sequential` engine (one actor per step):

```python
from concordia.contrib.prefabs.game_master import rulebook as rulebook_gm
from concordia.typing import prefab as prefab_lib

prefab_lib.InstanceConfig(
    prefab='rulebook__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={'name': 'Referee', 'rulebook': my_rules},
)
```
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.contrib.components.game_master import rulebook as rulebook_lib
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A deterministic game master around a rulebook.

  Params:
    name: The game master's name.
    rulebook: An object implementing `rulebook_lib.Rulebook`.
  """

  description: str = (  # pyrefly: ignore[bad-override]
      'A game master whose turn order, action specs and resolution come from'
      ' a rulebook: a plain Python object implementing the rules. Makes no'
      ' language model calls of its own.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'Referee', 'rulebook': None}
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the game master.

    Args:
      model: Passed to `SwitchAct` and `MakeObservation`, whose language
        model paths are never taken.
      memory_bank: Unused; the game master keeps no associative memory.

    Returns:
      The game master entity.
    """
    del memory_bank
    self.check_extra_components()
    rules: rulebook_lib.Rulebook | None = self.params.get('rulebook')
    if rules is None:
      raise ValueError('The rulebook game master needs a `rulebook` param.')
    names = [entity.name for entity in self.entities or ()] or list(
        rules.actors()
    )
    queue = make_observation.ObservationQueue()
    terminator = terminate.Terminate()
    turn = rulebook_lib.RulebookNextActing(rules)
    resolution = rulebook_lib.RulebookResolution(rules, queue, terminator)
    resolution.deliver()  # Anything the rulebook queued during setup.
    components = {
        terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminator,
        make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model=model,
                player_names=names,
                external_queue=queue,
                allow_llm_fallback=False,
            )
        ),
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: turn,
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            rulebook_lib.RulebookActionSpec(rules, turn)
        ),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: resolution,
    }
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=self.params.get('name', 'Referee'),
        act_component=switch_act.SwitchAct(model=model, entity_names=names),
        context_components=components,
    )
