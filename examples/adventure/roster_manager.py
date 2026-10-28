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
"""Three-tier NPC lifecycle manager (Hot, Warm, Cold) for Adventure simulations.

Coordinates dynamic NPC spawning, promotion to Hot when co-located with the
player, demotion to Warm for off-screen plot progression, hibernation to Cold
via `entity.get_state()`, and rehydration via `entity.set_state()` with
catch-up observations.
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import re
import threading
from typing import Any, cast, override

from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.contrib.components.game_master import topic_observations
from concordia.contrib.pubsub import topics
from examples.adventure import world_ledger
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import async_measurements

DEFAULT_ROSTER_COMPONENT_KEY = '__roster_manager__'

EntityBuilder = Callable[
    [world_ledger.NpcRecord], entity_agent_with_logging.EntityAgentWithLogging
]


class ManagedNpcEntity(entity_lib.EntityWithLogging):
  """Stable entity wrapper for Concordia's `Asynchronous` engine.

  Keeps a persistent `ReactiveMeasurements` channel and thread capture key while
  the underlying NPC transitions between Hot, Warm, and Cold tiers or binds to a
  newly discovered procedural NPC.
  """

  def __init__(
      self,
      roster: 'RosterManager',
      initial_name: str,
      *,
      is_dynamic_slot: bool = False,
  ):
    """Initializes the instance."""
    super().__init__()
    self._roster = roster
    self._name = initial_name
    self._is_dynamic_slot = is_dynamic_slot
    self._bound = not is_dynamic_slot
    self._measurements = async_measurements.ReactiveMeasurements()
    self._lock = threading.RLock()

  @property
  @override
  def name(self) -> str:
    with self._lock:
      return self._name

  @property
  def is_bound(self) -> bool:
    """Returns True when this slot is bound to a named NPC in the ledger."""
    with self._lock:
      return self._bound

  def bind(self, npc_name: str) -> None:
    """Binds a dynamic slot to a newly spawned procedural NPC."""
    with self._lock:
      self._name = npc_name
      self._bound = True

  @property
  def measurements(self) -> async_measurements.ReactiveMeasurements:
    """Returns the persistent ReactiveMeasurements channel for this slot."""
    return self._measurements

  def _sync_thread_capture_key(self) -> None:
    gm = self._roster.game_master
    thread_id = threading.current_thread().ident
    if (
        gm is not None
        and thread_id is not None
        and hasattr(gm, 'set_capture_key_for_thread')
    ):
      gm.set_capture_key_for_thread(thread_id, self.name)

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    self._sync_thread_capture_key()
    if not self.is_bound:
      return ''
    live = self._roster.get_live_entity(self.name)
    if live is None:
      return ''
    raw_act = live.act(action_spec)
    return clean_npc_act_output(self.name, raw_act)

  def _clean_npc_act_output(self, raw_act: str) -> str:
    """Cleans duplicate name prefixes, 'Answer:' tags, and empty Warm outputs."""
    return clean_npc_act_output(self.name, raw_act)

  @override
  def observe(self, observation: str) -> None:
    self._sync_thread_capture_key()
    if not self.is_bound or not observation:
      return
    live = self._roster.get_live_entity(self.name)
    if live is not None:
      live.observe(observation)

  @override
  def get_last_log(self) -> dict[str, Any]:
    if not self.is_bound:
      return {}
    live = self._roster.get_live_entity(self.name)
    if live is not None:
      return dict(live.get_last_log())
    return {}


def clean_npc_act_output(actor_name: str, raw_act: str) -> str:
  """Cleans duplicate name prefixes, 'Answer:' tags, and empty Warm outputs."""
  actor_clean = actor_name.strip()
  text = (raw_act or '').strip()
  if not actor_clean:
    return text
  text = re.sub(
      r'^(?:'
      + re.escape(actor_clean)
      + r'\s+)?(?:answer|response|action)\s*:\s*',
      '',
      text,
      flags=re.IGNORECASE,
  ).strip()
  tokens = [t for t in actor_clean.split() if len(t) >= 3]
  if text.lower().startswith(actor_clean.lower()):
    rest = text[len(actor_clean) :].lstrip(' :')
    rest_lower = rest.lower()
    if rest_lower.startswith(actor_clean.lower()):
      text = rest
    else:
      for tok in tokens:
        tok_l = tok.lower()
        if rest_lower.startswith(tok_l + ' '):
          after_tok = rest[len(tok) :].lstrip()
          text = f'{actor_clean} {after_tok}'
          break
        if rest_lower.startswith((tok_l + "'", tok_l + '’', tok_l + ':')):
          after_tok = rest[len(tok) :]
          text = f'{actor_clean}{after_tok}'
          break
  if text.lower().startswith(actor_clean.lower()):
    after_name = text[len(actor_clean) :].lstrip()
    words = after_name.split()
    if len(words) >= 2:
      w0 = re.sub(r'[^a-z]', '', words[0].lower())
      w1 = re.sub(r'[^a-z]', '', words[1].lower())
      if len(w0) >= 4 and len(w1) >= 4 and (w0 == w1 or w1.startswith(w0)):
        words.pop(0)
        text = f'{actor_clean} ' + ' '.join(words)
  body = text
  if body.lower().startswith(actor_clean.lower()):
    body = body[len(actor_clean) :].strip(' .,:;-—')
  if len(body.split()) < 2:
    return (
        f'{actor_clean} checks the nearest hatch seal and braces their'
        ' position.'
    )
  if not text.lower().startswith(actor_clean.lower()):
    for tok in tokens:
      if text.lower().startswith(tok.lower() + ' '):
        text = f'{actor_clean} ' + text[len(tok) :].lstrip()
        break
  return text


class CleanNpcActComponent(
    agent_components.concat_act_component.ConcatActComponent
):
  """ConcatActComponent that normalizes NPC action output before logging."""

  @override
  def _log(self, result: str, prompt: Any) -> None:
    cleaned = clean_npc_act_output(self.get_entity().name, result)
    super()._log(cleaned, prompt)

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    raw = super().get_action_attempt(contexts, action_spec)
    return clean_npc_act_output(self.get_entity().name, raw)


def extract_entity_memories(
    entity: entity_agent_with_logging.EntityAgentWithLogging,
) -> list[str]:
  """Extracts all durable memory strings from an entity's memory component."""
  for key in (
      agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
      'memory',
  ):
    try:
      comp = entity.get_component(key)
    except (KeyError, AttributeError):
      continue
    if comp is None:
      continue
    bank = getattr(comp, '_memory_bank', None)
    if bank is not None and hasattr(bank, 'get_all_memories_as_text'):
      return [str(m) for m in bank.get_all_memories_as_text()]
    try:
      if hasattr(comp, 'get_all_memories_as_text'):
        return [str(m) for m in comp.get_all_memories_as_text()]
    except ValueError:
      pass
    if hasattr(comp, 'scan'):
      return [str(m) for m in comp.scan(lambda _: True)]
  return []


def add_memory_direct(
    entity: entity_agent_with_logging.EntityAgentWithLogging,
    text: str,
) -> None:
  """Adds a memory directly to an entity's memory bank and commits it."""
  for key in (
      agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
      'memory',
  ):
    try:
      comp = entity.get_component(key)
    except (KeyError, AttributeError):
      continue
    if comp is not None and hasattr(comp, 'add') and hasattr(comp, 'update'):
      comp.add(text)
      comp.update()
      return


class RosterManager(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Manages NPC entities across Hot, Warm, and Cold execution tiers."""

  def __init__(
      self,
      ledger: world_ledger.WorldLedger,
      entity_builder: EntityBuilder,
      *,
      topic_obs: topic_observations.TopicObservations | None = None,
      max_hot: int = 3,
      max_warm: int = 5,
  ):
    """Initializes the instance.

    Args:
      ledger: Shared WorldLedger tracking locations and NPC records.
      entity_builder: Factory creating an EntityAgentWithLogging from NpcRecord.
      topic_obs: Optional TopicObservations routing scene/rumor topics.
      max_hot: Maximum simultaneous Hot NPCs in the player's location.
      max_warm: Maximum simultaneous Warm NPCs active off-screen.
    """
    super().__init__()
    self._lock = threading.RLock()
    self._ledger = ledger
    self._entity_builder = entity_builder
    self._topic_obs = topic_obs
    self._max_hot = max_hot
    self._max_warm = max_warm
    self._live_entities: dict[
        str, entity_agent_with_logging.EntityAgentWithLogging
    ] = {}
    self._archived_memories: dict[str, list[str]] = {}
    self._managed_by_name: dict[str, ManagedNpcEntity] = {}
    self._dynamic_slots: list[ManagedNpcEntity] = []
    self._game_master: entity_lib.Entity | None = None
    self._sync_player_access()

  @property
  def game_master(self) -> entity_lib.Entity | None:
    """Returns the linked Game Master entity, if set."""
    if self._game_master is not None:
      return self._game_master
    return self._entity

  @override
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    self._game_master = entity

  def set_game_master(self, game_master: entity_lib.Entity) -> None:
    """Links the Game Master so dynamic slots can update thread capture keys."""
    self._game_master = game_master

  def _ensure_managed_slot(self, name: str) -> ManagedNpcEntity:
    """Returns or allocates a ManagedNpcEntity for `name`."""
    with self._lock:
      existing = self._managed_by_name.get(name)
      if existing is not None:
        return existing
      for slot in self._dynamic_slots:
        if not slot.is_bound:
          slot.bind(name)
          self._managed_by_name[name] = slot
          return slot
      managed = ManagedNpcEntity(self, name, is_dynamic_slot=False)
      self._managed_by_name[name] = managed
      return managed

  def _attach_managed_measurements(
      self,
      name: str,
      entity: entity_agent_with_logging.EntityAgentWithLogging,
  ) -> None:
    """Binds the persistent ManagedNpcEntity measurements onto `entity`."""
    managed = self._ensure_managed_slot(name)
    entity._component_logging = managed.measurements  # pylint: disable=protected-access

  def engine_entities(
      self, num_dynamic_slots: int = 16
  ) -> list[ManagedNpcEntity]:
    """Returns stable entity wrappers for Concordia's `Asynchronous` engine."""
    with self._lock:
      for rec_name in self._ledger.all_npcs():
        self._ensure_managed_slot(rec_name)
      while len(self._dynamic_slots) < num_dynamic_slots:
        idx = len(self._dynamic_slots)
        slot = ManagedNpcEntity(
            self, f'__npc_slot_{idx}__', is_dynamic_slot=True
        )
        self._dynamic_slots.append(slot)
      ordered = list(self._managed_by_name.values())
      for slot in self._dynamic_slots:
        if slot not in ordered:
          ordered.append(slot)
      return ordered

  @property
  def max_hot(self) -> int:
    """Returns the cap on concurrent Hot NPCs."""
    return self._max_hot

  @property
  def max_warm(self) -> int:
    """Returns the cap on concurrent Warm NPCs."""
    return self._max_warm

  def _sync_player_access(self) -> None:
    """Updates the player's TopicObservations access to their current room."""
    if self._topic_obs is not None:
      loc_id = self._ledger.player_location_id
      self._topic_obs.set_access(
          self._ledger.player_name,
          (f'scene.{loc_id}', 'world.rumors'),
      )

  def _sync_npc_access(self, name: str, location_id: str, active: bool) -> None:
    """Updates an NPC's TopicObservations subscriptions."""
    if self._topic_obs is None:
      return
    if not active:
      self._topic_obs.set_access(name, ())
    else:
      self._topic_obs.set_access(
          name,
          (f'scene.{location_id}', 'world.rumors'),
      )

  def get_live_entity(
      self, name: str
  ) -> entity_agent_with_logging.EntityAgentWithLogging | None:
    """Returns the live entity instance for a Hot or Warm NPC, or None."""
    with self._lock:
      return self._live_entities.get(name)

  def hot_entities(
      self,
  ) -> list[entity_agent_with_logging.EntityAgentWithLogging]:
    """Returns live entity objects for all NPCs currently in the Hot tier."""
    with self._lock:
      records = self._ledger.npcs_in_tier(world_ledger.NpcTier.HOT)
      return [
          self._live_entities[rec.name]
          for rec in records
          if rec.name in self._live_entities
      ]

  def warm_entities(
      self,
  ) -> list[entity_agent_with_logging.EntityAgentWithLogging]:
    """Returns live entity objects for all NPCs currently in the Warm tier."""
    with self._lock:
      records = self._ledger.npcs_in_tier(world_ledger.NpcTier.WARM)
      return [
          self._live_entities[rec.name]
          for rec in records
          if rec.name in self._live_entities
      ]

  def all_live_entities(
      self,
  ) -> list[entity_agent_with_logging.EntityAgentWithLogging]:
    """Returns all currently instantiated Hot and Warm NPC entities."""
    with self._lock:
      return list(self._live_entities.values())

  def spawn_npc(
      self,
      record: world_ledger.NpcRecord,
      *,
      initial_tier: world_ledger.NpcTier | None = None,
      rebalance: bool = True,
  ) -> entity_agent_with_logging.EntityAgentWithLogging | None:
    """Spawns a new NPC, seeds its formative memories, and assigns its tier.

    Args:
      record: NpcRecord describing the character.
      initial_tier: Explicit initial tier. If None, inferred from location.
      rebalance: Whether to run `rebalance_tiers()` after spawning.

    Returns:
      The live EntityAgentWithLogging if Hot or Warm, or None if hibernated to
      Cold.
    """
    with self._lock:
      existing = self._ledger.get_npc(record.name)
      if existing is not None and record.name in self._live_entities:
        return self._live_entities[record.name]

      loc_id = world_ledger.normalize_location_id(record.location_id)
      record.location_id = loc_id
      if initial_tier is None:
        if loc_id == self._ledger.player_location_id:
          initial_tier = world_ledger.NpcTier.HOT
        else:
          initial_tier = world_ledger.NpcTier.WARM
      record.tier = initial_tier
      record.last_active_tick = self._ledger.tick
      self._ledger.register_npc(record)

      entity = self._entity_builder(record)
      self._attach_managed_measurements(record.name, entity)
      persona_mem = (
          f'{record.name} is {record.role_summary}. '
          f'Speech & traits: {record.traits}.'
      )
      add_memory_direct(entity, persona_mem)
      for mem in record.formative_memories:
        if mem.strip():
          add_memory_direct(entity, mem.strip())

      self._live_entities[record.name] = entity
      self._archived_memories[record.name] = extract_entity_memories(entity)
      self._sync_npc_access(
          record.name,
          record.location_id,
          active=(initial_tier != world_ledger.NpcTier.COLD),
      )

      if initial_tier == world_ledger.NpcTier.COLD:
        self.hibernate_to_cold(record.name)
        return None

      if rebalance:
        self.rebalance_tiers()
      return self._live_entities.get(record.name)

  def hibernate_to_cold(self, name: str) -> dict[str, Any] | None:
    """Checkpoints a live NPC via `get_state()` and releases its instance.

    Args:
      name: Name of the NPC to hibernate.

    Returns:
      The serialized entity state dict, or None if the NPC was not found.
    """
    with self._lock:
      rec = self._ledger.get_npc(name)
      if rec is None:
        return None
      entity = self._live_entities.pop(name, None)
      if entity is None:
        return rec.cold_state

      self._archived_memories[name] = extract_entity_memories(entity)
      try:
        raw_state = copy.deepcopy(entity.get_state())
      except ValueError:
        if rec.cold_state:
          raw_state = copy.deepcopy(rec.cold_state)
        else:
          raw_state = {
              'context_components': {},
              'act_component': {},
          }
      context_states = raw_state.get('context_components')
      if isinstance(context_states, dict):
        # Do not checkpoint the shared TopicBus inside an individual Cold NPC,
        # so rehydrating this NPC later does not roll back the shared bus.
        context_states.pop(topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY, None)

      self._sync_npc_access(name, rec.location_id, active=False)
      self._ledger.set_npc_tier(
          name,
          world_ledger.NpcTier.COLD,
          cold_state=dict(raw_state),
      )
      return dict(raw_state)

  def rehydrate_from_cold(
      self,
      name: str,
      *,
      target_tier: world_ledger.NpcTier = world_ledger.NpcTier.HOT,
  ) -> entity_agent_with_logging.EntityAgentWithLogging | None:
    """Rebuilds a Cold NPC via `set_state()` and injects a catch-up summary.

    Args:
      name: Name of the Cold NPC to rehydrate.
      target_tier: Tier to place the rehydrated NPC into (HOT or WARM).

    Returns:
      The rehydrated EntityAgentWithLogging, or None if `name` is unknown.
    """
    with self._lock:
      rec = self._ledger.get_npc(name)
      if rec is None:
        return None
      if name in self._live_entities:
        self._ledger.set_npc_tier(name, target_tier)
        self._sync_npc_access(name, rec.location_id, active=True)
        return self._live_entities[name]

      entity = self._entity_builder(rec)
      self._attach_managed_measurements(name, entity)
      if rec.cold_state is not None:
        state_to_restore = copy.deepcopy(rec.cold_state)
        context_states = state_to_restore.get('context_components')
        if isinstance(context_states, dict):
          context_states.pop(topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY, None)
        entity.set_state(state_to_restore)
      existing_mems = set(extract_entity_memories(entity))
      for mem in self._archived_memories.get(name, []):
        if mem not in existing_mems:
          add_memory_direct(entity, mem)

      elapsed_ticks = max(0, self._ledger.tick - rec.last_active_tick)
      loc_node = self._ledger.get_location(rec.location_id)
      loc_name = loc_node.name if loc_node else rec.location_id
      recent_local = (
          loc_node.recent_events[-2:]
          if loc_node and loc_node.recent_events
          else []
      )
      catchup_parts = [
          f'Time is now {self._ledger.format_clock()} '
          f'({elapsed_ticks} steps elapsed since {name} last acted). '
          f'{name} is at {loc_name}.'
      ]
      if recent_local:
        catchup_parts.append('Recent local activity: ' + ' '.join(recent_local))
      director = self._ledger.director
      if director is not None and (
          target_tier == world_ledger.NpcTier.HOT or self._ledger.tick == 0
      ):
        rehydrate_note = director.rehydration_observation(name)
        if rehydrate_note:
          catchup_parts.append(rehydrate_note)
      catchup_obs = ' '.join(catchup_parts)
      entity.observe(catchup_obs)

      self._live_entities[name] = entity
      self._archived_memories[name] = extract_entity_memories(entity)
      self._ledger.set_npc_tier(name, target_tier)
      self._sync_npc_access(name, rec.location_id, active=True)
      return entity

  def promote_to_hot(
      self, name: str
  ) -> entity_agent_with_logging.EntityAgentWithLogging | None:
    """Promotes or rehydrates an NPC into the Hot tier."""
    with self._lock:
      rec = self._ledger.get_npc(name)
      if rec is None:
        return None
      if (
          rec.tier == world_ledger.NpcTier.COLD
          or name not in self._live_entities
      ):
        return self.rehydrate_from_cold(
            name, target_tier=world_ledger.NpcTier.HOT
        )
      was_warm = rec.tier == world_ledger.NpcTier.WARM
      self._ledger.set_npc_tier(name, world_ledger.NpcTier.HOT)
      self._sync_npc_access(name, rec.location_id, active=True)
      entity = self._live_entities[name]
      if was_warm and self._ledger.director is not None:
        had_spoken = any(
            e.get('kind') == 'npc_resolution' and e.get('actor') == name
            for e in self._ledger.get_narrative_log()
        )
        if had_spoken:
          rehydrate_note = self._ledger.director.rehydration_observation(name)
          if rehydrate_note:
            entity.observe(rehydrate_note)
      return entity

  def demote_to_warm(
      self, name: str
  ) -> entity_agent_with_logging.EntityAgentWithLogging | None:
    """Demotes a Hot NPC or rehydrates a Cold NPC into the Warm tier."""
    with self._lock:
      rec = self._ledger.get_npc(name)
      if rec is None:
        return None
      if (
          rec.tier == world_ledger.NpcTier.COLD
          or name not in self._live_entities
      ):
        return self.rehydrate_from_cold(
            name, target_tier=world_ledger.NpcTier.WARM
        )
      self._ledger.set_npc_tier(name, world_ledger.NpcTier.WARM)
      self._sync_npc_access(name, rec.location_id, active=True)
      return self._live_entities[name]

  def rebalance_tiers(self) -> dict[str, list[str]]:
    """Recomputes Hot, Warm, and Cold assignments based on player location.

    Returns:
      Dictionary mapping `'hot'`, `'warm'`, and `'cold'` to lists of NPC names.
    """
    with self._lock:
      self._sync_player_access()
      player_loc = self._ledger.player_location_id
      all_records = list(self._ledger.all_npcs().values())

      co_located = [r for r in all_records if r.location_id == player_loc]
      off_screen = [r for r in all_records if r.location_id != player_loc]

      # Prioritize directly addressed and most recently active co-located NPCs.
      co_located.sort(
          key=lambda r: (
              1 if r.directly_addressed else 0,
              r.last_active_tick,
              r.name,
          ),
          reverse=True,
      )
      desired_hot = {r.name for r in co_located[: self._max_hot]}

      # Candidate Warm NPCs are strictly off-screen NPCs (never co-located in
      # the player's current room). Strictly prioritize NPCs residing in core
      # hub locations over 2-hop outer stub rooms, and rotate by least recent
      # rumors.
      all_locs = list(self._ledger.all_locations().values())
      start_node = (
          max(all_locs, key=lambda loc: len(loc.exits)) if all_locs else None
      )
      hub_room_ids = (
          {start_node.location_id} | set(start_node.exits.values())
          if start_node is not None
          else {player_loc}
      )
      recent_rumors = self._ledger.get_rumors(limit=12)
      is_live_play = self._ledger.tick > 0
      late_act = self._ledger.tick >= 42
      start_id = (
          start_node.location_id if start_node is not None else player_loc
      )
      late_act_rooms = {start_id, 'the_deptford_dog_kennels'}
      warm_candidates = list(off_screen)
      warm_candidates.sort(
          key=lambda r: (
              1 if r.location_id in hub_room_ids else 0,
              1 if (is_live_play and bool(r.relationships)) else 0,
              0 if (is_live_play and r.tier == world_ledger.NpcTier.HOT) else 1,
              -sum(1 for rm in recent_rumors if r.name in rm),
              int(not late_act or r.location_id in late_act_rooms),
              1 if r.tier != world_ledger.NpcTier.COLD else 0,
              r.last_active_tick,
              r.name,
          ),
          reverse=True,
      )
      desired_warm = {r.name for r in warm_candidates[: self._max_warm]}

      for rec in all_records:
        if rec.name in desired_hot:
          self.promote_to_hot(rec.name)
        elif rec.name in desired_warm:
          self.demote_to_warm(rec.name)
        else:
          if (
              rec.tier != world_ledger.NpcTier.COLD
              or rec.name in self._live_entities
          ):
            self.hibernate_to_cold(rec.name)

      return {
          'hot': [
              r.name
              for r in self._ledger.npcs_in_tier(world_ledger.NpcTier.HOT)
          ],
          'warm': [
              r.name
              for r in self._ledger.npcs_in_tier(world_ledger.NpcTier.WARM)
          ],
          'cold': [
              r.name
              for r in self._ledger.npcs_in_tier(world_ledger.NpcTier.COLD)
          ],
      }

  def collect_all_entity_memories(
      self,
      player_entity: (
          entity_agent_with_logging.EntityAgentWithLogging | None
      ) = None,
  ) -> dict[str, list[str]]:
    """Returns a complete map of entity name -> memory list across all tiers."""
    with self._lock:
      result: dict[str, list[str]] = {}
      if player_entity is not None:
        result[player_entity.name] = extract_entity_memories(player_entity)
      for name, archived in self._archived_memories.items():
        result[name] = list(archived)
      for name, entity in self._live_entities.items():
        mems = extract_entity_memories(entity)
        self._archived_memories[name] = mems
        result[name] = mems
      return result

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      hot = len(self._ledger.npcs_in_tier(world_ledger.NpcTier.HOT))
      warm = len(self._ledger.npcs_in_tier(world_ledger.NpcTier.WARM))
      cold = len(self._ledger.npcs_in_tier(world_ledger.NpcTier.COLD))
      summary = f'hot={hot} warm={warm} cold={cold}'
      self._logging_channel({
          'Key': DEFAULT_ROSTER_COMPONENT_KEY,
          'Summary': summary,
          'Value': summary,
      })
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      archived = {k: list(v) for k, v in self._archived_memories.items()}
      live_states = {
          name: copy.deepcopy(entity.get_state())
          for name, entity in self._live_entities.items()
      }
      return cast(
          entity_component.ComponentState,
          {
              'max_hot': self._max_hot,
              'max_warm': self._max_warm,
              'archived_memories': archived,
              'live_entity_states': live_states,
          },
      )

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      if 'max_hot' in state and isinstance(state['max_hot'], (int, float)):
        self._max_hot = int(state['max_hot'])
      if 'max_warm' in state and isinstance(state['max_warm'], (int, float)):
        self._max_warm = int(state['max_warm'])
      raw_archived = state.get('archived_memories')
      if isinstance(raw_archived, Mapping):
        self._archived_memories = {
            str(k): [str(m) for m in v]
            for k, v in raw_archived.items()
            if isinstance(v, Sequence) and not isinstance(v, str)
        }
      raw_live = state.get('live_entity_states')
      if isinstance(raw_live, Mapping):
        for existing_name in list(self._live_entities):
          if existing_name not in raw_live:
            self._live_entities.pop(existing_name, None)
        for raw_name, ent_state in raw_live.items():
          if not isinstance(ent_state, Mapping):
            continue
          npc_name = str(raw_name)
          entity = self._live_entities.get(npc_name)
          if entity is None:
            rec = self._ledger.get_npc(npc_name)
            if rec is not None:
              entity = self._entity_builder(rec)
              self._attach_managed_measurements(npc_name, entity)
              self._live_entities[npc_name] = entity
          if entity is not None:
            entity.set_state(cast(entity_component.EntityState, ent_state))

  @override
  def get_dynamic_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'max_hot': self._max_hot,
          'max_warm': self._max_warm,
      }
