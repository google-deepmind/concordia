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

"""World pre-step: the World GM's `__next_acting__` component.

The Simultaneous engine asks the game master `NEXT_ACTING` exactly once per
step, before any entity observes or acts, even when the action phase will be
skipped. That makes it the right moment for the deterministic world
pre-step: advance the clock and the sim-time scheduler (slow minds whose
latency has elapsed land), hand landed adaptations to the planet, run physics,
publish topics (which queues observations through the contrib
`TopicObservations` access lists) and judge the scenario. It then answers like
its parent,
`NextActingAllEntities`, but only with the entities active in this era.
"""

from collections.abc import Mapping
from typing import Any, cast, override

from concordia.components.game_master import next_acting
from concordia.components.game_master import terminate
from examples.deep_time import game as game_lib
from examples.deep_time.prefabs import species as species_prefab
from examples.deep_time.world import planet as planet_lib
from examples.deep_time.world import speciation as speciation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_TERMINATE_KEY = terminate.DEFAULT_TERMINATE_COMPONENT_KEY


class WorldStep(
    next_acting.NextActingAllEntities, entity_component.ComponentWithLogging
):
  """Advances the world one tick, then names the entities that act."""

  def __init__(
      self,
      context: game_lib.GameContext,
      player_names: list[str],
      terminate_component_key: str = DEFAULT_TERMINATE_KEY,
  ):
    """Initializes the component.

    Args:
      context: Shared game state.
      player_names: Names of all entities in the simulation.
      terminate_component_key: Key of the `terminate.Terminate` component that
        is told to end the game when the scenario is decided.
    """
    super().__init__(player_names=player_names)
    self._context = context
    self._terminate_component_key = terminate_component_key

  def _step_world(self) -> None:
    ctx = self._context
    ctx.clock.advance()
    # Slow minds whose latency has elapsed complete now, in a fixed order, and
    # their messages are stamped with the new tick (the scheduler and the
    # clock count ticks in lockstep).
    if ctx.sim is not None:
      # The slow minds due now weigh the world as the last resolution left
      # it, through the sheets the GM publishes (never the world itself).
      ctx.publish_sheets()
      ctx.sim.advance()
    ctx.advance_wire()
    # Minds' decisions reach the world only through their entities' actions
    # (resolved by `ActorResolution`), never here.
    report = ctx.planet.advance()
    events = list(report.events)
    for split in ctx.planet.take_new_species():
      events.append(self._found_species(split))
    # Observations follow from publishing: the World GM's TopicObservations
    # queues each message for the entities whose access lists match it.
    ctx.publish_state(events)
    # What each entity knows of its own state: computed here, by the GM, and
    # read by the entity's `Subscriber` (never from the world directly).
    ctx.publish_sheets()
    ctx.record_beliefs()
    if ctx.wakeups is not None:
      ctx.wakeups.step(ctx.planet.tick)
    headline = ctx.tick_headline()
    # A scenario flag (`play_on`) keeps the world going after its outcome,
    # for measurements (the push audit's): nobody plays such a game.
    if ctx.planet.outcome is not None and not ctx.scenario.get('play_on'):
      verdict = (
          f'The scenario ends in {ctx.planet.outcome}:'
          f' {ctx.planet.outcome_reason}.'
      )
      ctx.bus.publish('chronicle', verdict, source='world')
      self.get_entity().get_component(
          self._terminate_component_key, type_=terminate.Terminate
      ).terminate()
    self._logging_channel({
        'Key': 'World step',
        'Summary': headline,
        'Value': ctx.planet.history[-1],
    })

  def _found_species(
      self, split: speciation_lib.Split
  ) -> planet_lib.WorldEvent:
    """Names a new species with the slow model and adds it as an entity."""
    ctx = self._context
    evolution = ctx.planet.evolution
    assert evolution is not None
    parent = ctx.planet.species_name(split.parent)
    named = None
    if ctx.slow_model is not None and ctx.planet.speciation is not None:
      prompt = ctx.planet.speciation.naming_prompt.format(
          parent=parent,
          where=f'{"colder" if split.colder else "warmer"} {split.where}',
          temp=split.optimum_c,
          noun=ctx.planet.species_noun,
      )
      try:
        named = speciation_lib.parse_name(
            ctx.slow_model.sample_text(prompt, max_tokens=400, terminators=()),
            ctx.planet.species_noun,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        named = None
    name, lore = named or (
        evolution.info(split.child).name,
        f'They broke away from the {parent} in {split.where}.',
    )
    taken = {info.name for info in evolution.species if info.key != split.child}
    if name in taken:
      name = f'{name} of {split.where}'
    evolution.rename(split.child, name, lore)
    if ctx.spawner is not None and ctx.slow_model is not None:
      ctx.spawner(species_prefab.species_config(ctx, split.child))
      ctx.spawned.append(name)
    return planet_lib.WorldEvent(
        planet_lib.BIO_EVENTS, f'{name} have evolved from the {parent}: {lore}'
    )

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    # The engine maps names to entities before asking who acts, so species
    # founded during this step join from the next step on.
    known = set(self._player_names) | set(self._context.spawned)
    self._step_world()
    active = [name for name in self._context.active_entities() if name in known]
    # The engine needs at least one valid name even in eras without actors;
    # those entities then receive a SKIP_THIS_STEP action spec.
    return ','.join(active or self._player_names[:1])

  @override
  def get_state(self) -> entity_component.ComponentState:
    # The world rides in the World GM's checkpoint, the standard way: the
    # planet and the rest of the game's own state; the bus, scheduler and
    # clock are other components' (see `GameContext.get_state`).
    return {
        'player_names': list(self._player_names),
        'world': self._context.get_state(),
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    world = state.get('world')
    if isinstance(world, Mapping):
      self._context.set_state(cast(Mapping[str, Any], world))
