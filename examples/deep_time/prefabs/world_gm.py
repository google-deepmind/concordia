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

"""The World GM ("Gaia Engine"): a deterministic game master prefab.

It is assembled from standard pieces. `switch_act.SwitchAct` dispatches each
engine call to the component registered under the matching well-known key, so
the game master never needs a language model:

| Key | Component |
|---|---|
| `__terminate__` | `terminate.Terminate`, told to stop by the world step |
| `__make_observation__` | `make_observation.MakeObservation` on the shared
queue, LLM fallback off |
| `__next_acting__` | `WorldStep` (subclass of `NextActingAllEntities`): the
world pre-step |
| `__next_action_spec__` | `ActorActionSpecs`: per-entity specs from the actor
registry |
| `__resolution__` | `ActorResolution`: structured actions become world changes
|
| `__clock__` | contrib `NumericClock` shared with the topic bus |
| `__topic_observations__` | contrib `TopicObservations`: topic messages become
observations per access list |
| `__actors__` | `ActorRegistry`: each acting entity's rules (its action spec,
how its actions resolve), checkpointed with the GM |
| `__deliveries__` | `Deliveries`: runs every entity's instruments (their
transports, ageing and rendering) and publishes what reaches each on its
own topic |
| `entity_scheduler` | upstream `interrupt_scheduling.EntityScheduler`: the
event-driven entities' interrupt masks and timers (where there are any) |
| `__wakeups__` | `wakeups.WakeUps`: the events and alarms that wake them |
| `__standing_orders__` | `StandingOrders`: the peoples' adopted reflexes |

`__next_game_master__` is omitted: with a single game master the engine never
asks for it.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.contrib.components.game_master import numeric_time
from concordia.contrib.components.game_master import topic_observations
from concordia.environment.engines import simultaneous
from examples.deep_time import game as game_lib
from examples.deep_time import views
from examples.deep_time import wakeups as wakeups_lib
from examples.deep_time.components import action_specs
from examples.deep_time.components import actor_registry
from examples.deep_time.components import deliveries
from examples.deep_time.components import resolution
from examples.deep_time.components import standing_orders
from examples.deep_time.components import world_step
from examples.deep_time.world import publishing
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class WorldGameMaster(prefab_lib.Prefab):
  """A deterministic World GM that steps the planet once per engine step."""

  description: str = (  # pyrefly: ignore[bad-override]
      'Deep Time World GM: steps deterministic planet physics, publishes'
      ' topics, resolves structured actions. Makes no language model calls.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {'name': 'Deep Time', 'roles': []}
  )
  # The game (a shared service: copying the prefab keeps it).
  context: game_lib.GameContext | None = None
  # Each role's prefab, by its prefab key: the World GM installs each role's
  # rules from the role's plain params (`params['roles']`) as it builds.
  roles: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the World GM.

    Args:
      model: Unused by the GM's components; passed to `SwitchAct` and
        `MakeObservation`, whose language-model paths are never taken.
      memory_bank: Unused; the World GM keeps no associative memory.

    Returns:
      The game master entity.
    """
    del memory_bank
    name = self.params.get('name', 'Deep Time')
    context = self.context
    if context is None:
      raise ValueError('The World GM needs the game context.')
    # Every role's rules (its actor, wake-ups and attention, the sheets the
    # GM publishes for it and the instruments it runs), from its params, in
    # the order the roles were listed.
    premise = str(context.scenario.get('premise', ''))
    for role in self.params.get('roles', ()):
      self.roles[role['prefab']].rules(context, role['params'])
      if premise:  # Every role hears the premise first.
        context.observe(str(role['params']['name']), premise)
    # For a game a person's page shows: each role's view (crises, needs).
    if self.params.get('views'):
      views.register_world(context)
      for actor in sorted(context.actors):
        views.register(context, actor, str(self.params['views']))
    player_names = [entity.name for entity in self.entities or ()]

    components = {
        numeric_time.DEFAULT_NUMERIC_CLOCK_COMPONENT_KEY: context.clock,
        topic_observations.DEFAULT_TOPIC_OBSERVATIONS_COMPONENT_KEY: (
            topic_observations.TopicObservations(
                context.bus,
                context.observation_queue,
                access=context.access,
                render=publishing.OBSERVATION_RENDERERS,
                event_topic=publishing.RESOLUTIONS_TOPIC,
            )
        ),
        terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminate.Terminate(),
        make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
            make_observation.MakeObservation(
                model=model,
                player_names=player_names,
                call_to_make_observation=(
                    simultaneous.DEFAULT_CALL_TO_MAKE_OBSERVATION
                ),
                external_queue=context.observation_queue,
                allow_llm_fallback=False,
            )
        ),
        next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: world_step.WorldStep(
            context=context, player_names=player_names
        ),
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
            action_specs.ActorActionSpecs(context=context)
        ),
        switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: resolution.ActorResolution(
            context=context
        ),
        # The rules of every acting entity, checkpointed with the GM.
        actor_registry.DEFAULT_ACTOR_REGISTRY_KEY: context.registry,
        # Every entity's instruments: what reaches whom, when, how.
        deliveries.DEFAULT_DELIVERIES_KEY: context.deliveries,
        # Who is awake in event-driven eras: the upstream scheduler of masks
        # and timers, and what wakes whom (stepped by `WorldStep`).
        **(
            {
                interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY: (
                    context.wakeups.scheduler
                ),
                wakeups_lib.DEFAULT_WAKEUPS_COMPONENT_KEY: context.wakeups,
            }
            if context.wakeups is not None
            else {}
        ),
        # The peoples' adopted standing orders (reflexes on the bus).
        **(
            {
                standing_orders.DEFAULT_STANDING_ORDERS_KEY: (
                    standing_orders.StandingOrders(context.automation)
                )
            }
            if context.automation is not None
            else {}
        ),
    }
    act_component = switch_act.SwitchAct(model=model, entity_names=player_names)
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
    )
