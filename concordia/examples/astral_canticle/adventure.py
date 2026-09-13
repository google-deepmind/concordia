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
from typing import Any

from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.environment.engines import sequential
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.typing import entity as entity_lib
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


def build_cast(model, reader, *, role='player', player_prefab='minimal'):
  """Build the cast, selecting Ilyra's standard prefab and only replacing act.

  With ``basic``, Ilyra retains the library prefab's exact context configuration
  and LLM perception behavior. NPC and GM composition does not change.
  """
  if player_prefab not in ('minimal', 'basic'):
    raise ValueError(f'Unknown player prefab: {player_prefab}')

  def build(
      name,
      instructions,
      goal='',
      act_component=None,
      extra=None,
      prefab='minimal',
      human_controlled=False,
  ):
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=lambda _: np.ones(8)
    )
    factory = (
        (
            lambda order: human_act_component.HumanActComponent(
                reader, component_order=order
            )
        )
        if human_controlled
        else None
    )
    if prefab == 'basic':
      return basic.Entity(params={'name': name, 'goal': goal}).build(
          model=model,
          memory_bank=memory_bank,
          act_component=act_component,
          act_component_factory=factory,
      )
    params: dict[str, Any] = {
        'name': name,
        'custom_instructions': instructions,
        'goal': goal,
        'randomize_choices': False,
        'extra_components': extra or {},
    }
    return minimal.Entity(params=params).build(
        model=model,
        memory_bank=memory_bank,
        act_component=act_component,
        act_component_factory=factory,
    )

  players = [
      build(
          PLAYER,
          'You are Ilyra Venn. ' + RULES,
          'Repair the star-loom without sacrificing the sleeping moon.',
          prefab=player_prefab,
          human_controlled=role == 'player',
      ),
      build(
          'Sable-9',
          'You are Sable-9, a precise but compassionate automaton. '
          'Act briefly in the first person. Do not control anyone else.',
          'Protect the pilgrims and help Ilyra investigate the resonance.',
      ),
      build(
          'Thorn of Io',
          'You are Thorn of Io, a patient mycelial knight. '
          'Act briefly in the first person. Do not control anyone else.',
          'Keep Auric Vesper alive and the Umbral pilgrims free.',
      ),
  ]
  names = [player.name for player in players]
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
  gm = build(
      GM,
      RULES + '\n' + LOCATIONS,
      act_component=(
          None
          if role == 'gm'
          else switch_act.SwitchAct(model=model, entity_names=names)
      ),
      extra=extra,
      human_controlled=role == 'gm',
  )
  for player in players:
    player.observe(PREMISE + '\n' + LOCATIONS)
  return players, gm


def play(
    model,
    session,
    output: pathlib.Path,
    *,
    role='player',
    max_steps=30,
    player_prefab='minimal',
):
  """Run one adventure; persist standard structured logs after every step."""
  output.mkdir(parents=True, exist_ok=True)
  players, gm = build_cast(
      model, session, role=role, player_prefab=player_prefab
  )
  engine = sequential.Sequential(
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
  raw_log = []

  def save():
    log = structured_logging.SimulationLog.from_raw_log(raw_log)
    log.attach_memories(
        entity_memories={
            p.name: p.get_component('__memory__').get_all_memories_as_text()
            for p in players
        },
        game_master_memories=gm.get_component(
            '__memory__'
        ).get_all_memories_as_text(),
    )
    for name, value in (
        ('simulation.json', log.to_json()),
        (
            'log.html',
            log.to_html(title='The Astral Canticle'),
        ),
    ):
      temporary = output / (name + '.tmp')
      temporary.write_text(value, encoding='utf-8')
      temporary.replace(output / name)

  def step_done(step):
    save()
    session.progress(step.step, step.acting_entity)

  try:
    engine.run_loop(
        game_masters=[gm],
        entities=players,
        premise=PREMISE,
        max_steps=max_steps,
        log=raw_log,
        step_callback=step_done,
    )
    if role == 'player':
      # The last resolved action deserves a player-visible conclusion even
      # when there will be no next action prompt. Ask the standard GM API.
      observation = engine.make_observation(gm, players[0])
      players[0].observe(observation)
      session.add_observation(observation)
  finally:
    save()
  session.finish('This chapter is complete. Your journal is ready to download.')
