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

"""Astral Canticle: standard prefab entities, sequential engine and logs.

This demonstrates human input in the Auric Vesper setting. The lean
composition uses SwitchAct directly with standard memory/context components so
local models spend their time telling the story, not selecting a turn order.
"""

import pathlib
from typing import Any, cast

from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.agent import memory
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.environment.engines import sequential
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import structured_logging
import numpy as np

PLAYER = 'Ilyra Venn'
GM = 'Auric Vesper'
PREMISE = """The heliostat-city Auric Vesper orbits a sleeping moon. Its ancient
star-loom has begun tearing the veil into the mythic Luminous Deep. You are Ilyra
Venn, an astromancer-engineer. Sable-9, a reliquary automaton, protects the Umbral
pilgrims; Thorn of Io, a mycelial knight, tends the city's living machinery.

You stand in the Star-Loom Chamber. A brass tuning fork hangs beside a cracked
resonance cradle. Three pale threads run into the loom; the silver thread has
slipped loose. Your satchel holds a prism-lantern and an insulated star-spanner.
A warning bell tolls once. Sable-9 waits by the pilgrim lift. Thorn watches the
roots pressing through the deck. There is time to investigate before committing
to a repair. Your first aim: discover what is pulling the loom out of tune."""
LOCATIONS = """The Star-Loom Chamber connects west to the Reliquary Nave, east
to the Mycelial Gardens, and down to the Pilgrim Deck. The Nave and Gardens each
open onto the outer Heliostat Walk. Return paths remain open unless an observed
event changes them. The Canticle Clock begins at Cycle 1, Verse 1, Pulse 0,
Tide Ember. Each Verse has 13 Pulses, each Cycle 8 Verses; Verses 1–2 are Ember,
3–4 Glass, 5–6 Bloom, 7–8 Umbral. Time only moves forward."""
RULES = """Run a coherent, intimate science-fantasy text adventure. Keep the
established map, objects, inventory and consequences consistent. Accept natural
language and classic commands: LOOK, EXAMINE object, TAKE object, INVENTORY,
WEST/EAST/UP/DOWN, TALK TO name, USE object ON target, WAIT. Interpret commands
as attempts, not guaranteed success. LOOK and INVENTORY reveal known facts;
never punish inspection. Give specific feedback when an action is impossible.
Reveal clues before danger. There are several ways to stabilize the star-loom;
never require an exact magic phrase. NPCs act with their own goals. Do not choose
Ilyra's actions, thoughts or dialogue. Do not resolve the whole adventure in one
turn. State only observable consequences, no analysis, option menus or meta text.
Use brief vivid paragraphs, concrete objects and clear exits. Keep each response
under 120 words. Preserve character names. Track carried items and clock in the
fiction; do not invent a successful repair until the actions justify one."""


def configuration(reader, *, role='player', player_prefab='minimal'):
  """Configure standard prefabs; bind the human reader only at runtime.

  The reader lives in the wrapper classes' closure rather than prefab params or
  fields. Simulation can copy prefab definitions without copying a transport's
  locks, pending requests or session state. Each build gets fresh components.
  """
  if role not in ('player', 'gm'):
    raise ValueError(f'Unknown human role: {role}')
  if player_prefab not in ('minimal', 'basic'):
    raise ValueError(f'Unknown player prefab: {player_prefab}')
  player_factory = basic.Entity if player_prefab == 'basic' else minimal.Entity

  def human_policy(order):
    return human_act_component.HumanActComponent(reader, component_order=order)

  class HumanPlayer(prefab_lib.Prefab):
    """Keep the selected standard prefab's context, replacing only its act."""

    description = 'Human-controlled Ilyra with standard prefab context.'

    def build(self, model, memory_bank):
      return player_factory(params=self.params).build(
          model, memory_bank, act_component_factory=human_policy
      )

  class AstralGameMaster(prefab_lib.Prefab):
    """Compose Auric Vesper's context and standard human/SwitchAct policy."""

    description = 'Astral Canticle GM with fixed-order or human routing.'

    def build(self, model, memory_bank):
      names = [entity.name for entity in self.entities or ()]
      extra = {}
      if role == 'player':
        extra = {
            switch_act.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
                next_acting.NextActingInFixedOrder(sequence=names)
            ),
            switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
                next_acting.FixedActionSpec(
                    entity_lib.free_action_spec(
                        call_to_action='What do you do next, {name}?'
                    )
                )
            ),
            switch_act.DEFAULT_TERMINATE_COMPONENT_KEY: constant.Constant(
                'No', pre_act_label=''
            ),
        }
      params: dict[str, Any] = {**self.params, 'extra_components': extra}
      return minimal.Entity(params=params).build(
          model,
          memory_bank,
          act_component=(
              switch_act.SwitchAct(model=model, entity_names=names)
              if role == 'player'
              else None
          ),
          act_component_factory=human_policy if role == 'gm' else None,
      )

  player_params: dict[str, Any] = {
      'name': PLAYER,
      'goal': 'Repair the star-loom without sacrificing the sleeping moon.',
  }
  if player_prefab == 'minimal':
    player_params.update({
        'custom_instructions': 'You are Ilyra Venn. ' + RULES,
        'randomize_choices': False,
    })

  return prefab_lib.Config(
      prefabs={
          'minimal': minimal.Entity(),
          'basic': basic.Entity(),
          'human_player': HumanPlayer(),
          'astral_gm': AstralGameMaster(),
      },
      instances=[
          prefab_lib.InstanceConfig(
              prefab='human_player' if role == 'player' else player_prefab,
              role=prefab_lib.Role.ENTITY,
              params=player_params,
          ),
          prefab_lib.InstanceConfig(
              prefab='minimal',
              role=prefab_lib.Role.ENTITY,
              params={
                  'name': 'Sable-9',
                  'custom_instructions': (
                      'You are Sable-9, a precise but compassionate automaton.'
                      ' Act briefly in the first person. Do not control anyone'
                      ' else.'
                  ),
                  'goal': (
                      'Protect the pilgrims and help Ilyra investigate the'
                      ' resonance.'
                  ),
                  'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
              },
          ),
          prefab_lib.InstanceConfig(
              prefab='minimal',
              role=prefab_lib.Role.ENTITY,
              params={
                  'name': 'Thorn of Io',
                  'custom_instructions': (
                      'You are Thorn of Io, a patient mycelial knight. Act'
                      ' briefly in the first person. Do not control anyone'
                      ' else.'
                  ),
                  'goal': (
                      'Keep Auric Vesper alive and the Umbral pilgrims free.'
                  ),
                  'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
              },
          ),
          prefab_lib.InstanceConfig(
              prefab='astral_gm',
              role=prefab_lib.Role.GAME_MASTER,
              params={
                  'name': GM,
                  'custom_instructions': RULES + '\n' + LOCATIONS,
                  'goal': '',
                  'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
              },
          ),
      ],
      default_premise=PREMISE,
      default_max_steps=30,
  )


def _make_engine():
  return sequential.Sequential(
      call_to_make_observation=(
          'Describe only what {name} now perceives and the consequences they '
          'can see. Include their current location, known exits and carried '
          'items when useful. No hidden thoughts or plans of others. At most '
          '120 words; clear short paragraphs, no analysis.'
      ),
      call_to_resolve=(
          'Resolve the latest attempted action only. What visibly changes? '
          'Respect the map, inventory and previous events. Be concrete and '
          'brief: at most 120 words. Do not act for another character.'
      ),
  )


def build_simulation(
    model, reader, *, role='player', player_prefab='minimal', engine=None
) -> generic.Simulation:
  """Build, but do not run, the standard Simulation with fresh memory banks."""
  simulation = generic.Simulation(
      config=configuration(reader, role=role, player_prefab=player_prefab),
      model=model,
      embedder=lambda _: np.ones(8),
      engine=engine if engine is not None else _make_engine(),
  )
  for player in simulation.get_entities():
    player.observe(PREMISE + '\n' + LOCATIONS)
  return simulation


def build_cast(model, reader, *, role='player', player_prefab='minimal'):
  """Compatibility helper; the standard Simulation now constructs the cast."""
  simulation = build_simulation(
      model, reader, role=role, player_prefab=player_prefab
  )
  return simulation.get_entities(), simulation.get_game_masters()[0]


def play(
    model,
    session,
    output: pathlib.Path,
    *,
    role='player',
    max_steps=30,
    player_prefab='minimal',
):
  """Run via standard Simulation; save partial logs even on cancellation."""
  engine = _make_engine()
  simulation = build_simulation(
      model, session, role=role, player_prefab=player_prefab, engine=engine
  )
  output.mkdir(parents=True, exist_ok=True)

  def save():
    # Simulation owns execution, banks and raw records. Its play() returns the
    # final log; a public raw-log snapshot also lets this adapter persist partial
    # progress while play() is still running or when it raises on cancellation.
    log = structured_logging.SimulationLog.from_raw_log(
        simulation.get_raw_log()
    )
    log.attach_memories(
        entity_memories={
            player.name: list(
                cast(entity_component.EntityWithComponents, player)
                .get_component('__memory__', type_=memory.AssociativeMemory)
                .get_all_memories_as_text()
            )
            for player in simulation.get_entities()
        },
        game_master_memories=(
            simulation.game_master_memory_bank.get_all_memories_as_text()
        ),
    )
    for name, value in (
        ('simulation.json', log.to_json()),
        ('log.html', log.to_html(title='The Astral Canticle')),
    ):
      temporary = output / (name + '.tmp')
      temporary.write_text(value, encoding='utf-8')
      temporary.replace(output / name)

  def step_done(step):
    save()
    session.progress(step.step, step.acting_entity)

  try:
    simulation.play(max_steps=max_steps, step_callback=step_done)
    if role == 'player':
      # The final resolution still deserves a player-visible conclusion.
      player = simulation.get_entities()[0]
      observation = engine.make_observation(
          simulation.get_game_masters()[0], player
      )
      player.observe(observation)
      session.add_observation(observation)
  finally:
    save()
  session.finish('This chapter is complete. Your journal is ready to download.')
