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
"""Deterministic StoryDirector for the Glassharbor adventure engine.

All functions in this module are pure or operate strictly on `WorldLedger` /
`StoryState` under the ledger lock. Zero language-model calls are made here, so
the hard invariant `control_hook_llm_calls == 0` is preserved by construction.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import random
import re
from typing import Any, override

from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_STORY_DIRECTOR_COMPONENT_KEY = (
    world_ledger.DEFAULT_STORY_DIRECTOR_COMPONENT_KEY
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_LEAK_STOPWORDS = frozenset({
    'that',
    'this',
    'with',
    'from',
    'have',
    'been',
    'were',
    'they',
    'them',
    'their',
    'there',
    'what',
    'when',
    'where',
    'which',
    'will',
    'would',
    'into',
    'just',
    'than',
    'then',
    'only',
    'over',
    'under',
    'about',
    'after',
    'before',
    'because',
    'while',
    'your',
    'you',
    'him',
    'her',
    'his',
    'she',
    'the',
    'and',
    'for',
    'not',
    'but',
    'are',
    'was',
    'has',
    'had',
    'did',
    'does',
    'can',
    'could',
    'should',
    'every',
    'some',
    'more',
    'most',
    'very',
    'also',
    'even',
    'still',
    'left',
    'like',
    'own',
    'who',
    'how',
    'why',
    'out',
    'off',
    'all',
    'any',
    'one',
    'two',
    'man',
    'men',
    'night',
    'morning',
})


def _keyword_support(source_text: str, narrative_text: str) -> float:
  words = {w for w in _WORD_RE.findall(source_text.lower()) if len(w) >= 6}
  if not words:
    return 0.0
  haystack = narrative_text.lower()
  hits = sum(1 for w in words if w in haystack)
  return hits / len(words)


def _distinctive_token_hits(source_text: str, narrative_text: str) -> int:
  haystack = set(_WORD_RE.findall(narrative_text.lower()))
  hits = 0
  for token in set(_WORD_RE.findall(source_text.lower())):
    if token in _LEAK_STOPWORDS:
      continue
    if len(token) < 4 and not token.isdigit():
      continue
    if token in haystack:
      hits += 1
  return hits


_NEGATIVE_RELATIONSHIP_TOKENS = (
    'hostile',
    'hates',
    'fears',
    'arrest',
    'betray',
    'sold',
    'grudge',
    'suspicious',
    'distrust',
    'owes',
    'wronged',
    'enemy',
    'rival',
)


@dataclasses.dataclass(frozen=True)
class WarmMove:
  """A deterministic decision for what a Warm NPC should do off-screen.

  Attributes:
    kind: 'pursue' | 'confront' | 'grudge' | 'faction_op' | 'storylet'.
    call_to_action: Prompt instruction given to the Warm NPC entity.
    target_npc: Other NPC involved when kind == 'confront'.
    storylet: Chosen storylet when kind == 'storylet'.
    faction_id: Faction id when kind == 'faction_op'.
    rumor_reliability: 'true' | 'distorted' | 'false'.
    rumor_override: Optional authored rumor text (from a warm storylet).
  """

  kind: str
  call_to_action: str
  target_npc: str = ''
  storylet: story_state.Storylet | None = None
  faction_id: str = ''
  rumor_reliability: str = 'true'
  rumor_override: str = ''


@dataclasses.dataclass(frozen=True)
class FrontierAsk:
  """Guidance for `FrontierArchitectWorker` when expanding a stub room.

  Attributes:
    faction_id: Faction the room and its resident NPC should lean toward.
    faction_name: Display name of that faction.
    faction_register: Speech register note for the faction.
    clues: 0-2 unplaced clues to embed as affordances or NPC knowledge.
    truth_hint: Short statement of a truth the resident NPC's secret touches.
    prompt_block: Formatted prompt block appended to the architect prompt.
  """

  faction_id: str = ''
  faction_name: str = ''
  faction_register: str = ''
  clues: tuple[story_state.Clue, ...] = ()
  truth_hint: str = ''
  prompt_block: str = ''


class StoryDirector(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Deterministic selector and pacing controller over `WorldLedger.story`."""

  def __init__(
      self,
      ledger: world_ledger.WorldLedger | None = None,
      *,
      rng_seed: int = 17,
  ):
    super().__init__()
    self._explicit_ledger = ledger
    self._rng_seed = rng_seed
    self._rng = random.Random(rng_seed)
    self._last_offered_ids: list[str] = []
    self._frontier_counter = 0
    self._used_voice_templates: set[tuple[str, int]] = set()
    self._last_invoked_voice: str = ''
    self._last_player_raw_action: str = ''
    self._spawned_emergent_voices: set[str] = set()

  @property
  def _ledger(self) -> world_ledger.WorldLedger:
    if self._explicit_ledger is not None:
      return self._explicit_ledger
    return self.get_entity().get_component(
        world_ledger.DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        type_=world_ledger.WorldLedger,
    )

  @property
  def active(self) -> bool:
    """True when the ledger carries a non-empty story bible."""
    story = self._ledger.story
    return bool(story.storylets or story.truths or story.factions)

  @property
  def last_offered_ids(self) -> list[str]:
    """Storylet ids most recently offered to `FastSceneResolver`."""
    return list(self._last_offered_ids)

  # ---------------------------------------------------------------------------
  # Precondition evaluation
  # ---------------------------------------------------------------------------

  def evaluate_condition(
      self,
      cond: Mapping[str, Any],
      *,
      location_id: str | None = None,
      npc_name: str | None = None,
  ) -> bool:
    """Evaluates a single precondition dict against current ledger state."""
    if len(cond) != 1:
      return False
    key, val = next(iter(cond.items()))
    ledger = self._ledger
    story = ledger.story
    loc_id = world_ledger.normalize_location_id(
        location_id or ledger.player_location_id
    )
    co_names = {n.name for n in ledger.npcs_at_location(loc_id)}
    if npc_name:
      co_names.add(npc_name)

    if key == 'npc_present':
      return str(val) in co_names
    if key == 'npc_absent':
      return str(val) not in co_names
    if key == 'npc_tier':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      rec = ledger.get_npc(str(val[0]))
      return rec is not None and rec.tier.value == str(val[1]).lower()
    if key == 'location_is':
      return loc_id == world_ledger.normalize_location_id(str(val))
    if key == 'location_visited':
      target = world_ledger.normalize_location_id(str(val))
      return target in story.visited_location_ids
    if key == 'inventory_has':
      needle = str(val).lower()
      return any(needle in item.lower() for item in ledger.player_inventory)
    if key == 'inventory_lacks':
      needle = str(val).lower()
      return all(needle not in item.lower() for item in ledger.player_inventory)
    if key == 'clue_discovered':
      c = story.clues.get(str(val))
      return bool(c and c.discovered)
    if key == 'clue_undiscovered':
      c = story.clues.get(str(val))
      return bool(c and not c.discovered)
    if key == 'clue_at_location':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      c = story.clues.get(str(val[0]))
      if c is None or not c.placed_at:
        return False
      target_norm = world_ledger.normalize_location_id(str(val[1]))
      return (
          world_ledger.normalize_location_id(c.placed_at) == target_norm
          or c.placed_at.lower() == str(val[1]).lower()
      )
    if key == 'truth_discovered':
      t = story.truths.get(str(val))
      return bool(t and t.discovered)
    if key == 'truth_undiscovered':
      t = story.truths.get(str(val))
      return bool(t and not t.discovered)
    if key == 'faction_standing_lt':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      fac = story.factions.get(str(val[0]))
      return bool(fac is not None and fac.standing_with_player < int(val[1]))
    if key == 'faction_standing_gt':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      fac = story.factions.get(str(val[0]))
      return bool(fac is not None and fac.standing_with_player > int(val[1]))
    if key == 'inter_faction_standing_lte':
      if not (isinstance(val, Sequence) and len(val) == 3):
        return False
      return story.inter_faction_standing(str(val[0]), str(val[1])) <= int(
          val[2]
      )
    if key == 'tension_lt':
      return story.tension < float(val)
    if key == 'tension_gt':
      return story.tension > float(val)
    if key == 'clock_between':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      hour = (ledger.clock_minutes % (24 * 60)) // 60
      slot = story_state.RoutineSlot(int(val[0]), int(val[1]), '')
      return slot.covers_hour(hour)
    if key == 'doom_stage_gte':
      return story.doom_stage >= int(val)
    if key == 'doom_stage_lt':
      return story.doom_stage < int(val)
    if key == 'act_is':
      return story.act == int(val)
    if key == 'act_gte':
      return story.act >= int(val)
    if key == 'relationship_contains':
      if not (isinstance(val, Sequence) and len(val) == 3):
        return False
      rec = ledger.get_npc(str(val[0]))
      if rec is None:
        return False
      target = ledger.player_name if str(val[1]) == 'player' else str(val[1])
      note = rec.relationships.get(target, '')
      return str(val[2]).lower() in note.lower()
    if key == 'fired_count_lt':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      return story.fired.get(str(val[0]), 0) < int(val[1])
    if key == 'rumor_contains':
      needle = str(val).lower()
      state_dict = ledger.to_dict()
      return any(needle in r.lower() for r in state_dict.get('rumors', []))
    if key == 'fact_is':
      if not (isinstance(val, Sequence) and len(val) == 2):
        return False
      facts = ledger.to_dict().get('world_facts', {})
      return facts.get(str(val[0])) == str(val[1])
    if key == 'player_status_contains':
      return str(val).lower() in ledger.player_status.lower()
    return False

  def eligible(
      self,
      trigger: str,
      *,
      location_id: str | None = None,
      npc_name: str | None = None,
  ) -> list[story_state.Storylet]:
    """Returns all storylets matching `trigger` whose preconditions hold."""
    with self._ledger.lock:
      story = self._ledger.story
      out: list[story_state.Storylet] = []
      for s in story.storylets.values():
        if s.trigger != trigger:
          continue
        if s.once and story.fired.get(s.id, 0) > 0:
          continue
        if all(
            self.evaluate_condition(
                c, location_id=location_id, npc_name=npc_name
            )
            for c in s.preconditions
        ):
          out.append(s)
      out.sort(key=lambda s: (-s.weight, s.id))
      return out

  def pick(
      self,
      trigger: str,
      *,
      location_id: str | None = None,
      npc_name: str | None = None,
      k: int = 2,
  ) -> list[story_state.Storylet]:
    """Deterministically selects up to `k` eligible storylets by weight."""
    pool = list(
        self.eligible(trigger, location_id=location_id, npc_name=npc_name)
    )
    chosen: list[story_state.Storylet] = []
    while pool and len(chosen) < k:
      weights = [max(0.05, s.weight) for s in pool]
      total = sum(weights)
      r = self._rng.uniform(0.0, total)
      acc = 0.0
      idx = 0
      for i, w in enumerate(weights):
        acc += w
        if r <= acc:
          idx = i
          break
      chosen.append(pool.pop(idx))
    return chosen

  # ---------------------------------------------------------------------------
  # Effect application
  # ---------------------------------------------------------------------------

  def apply_effects(
      self,
      effects: Sequence[Mapping[str, Any]],
      *,
      source: str = 'StoryDirector',
      is_warm: bool = False,
  ) -> list[str]:
    """Applies a list of storylet effect dicts to the ledger. Returns notes."""
    applied_notes: list[str] = []
    with self._ledger.lock:
      story = self._ledger.story
      for eff in effects:
        if len(eff) != 1:
          continue
        key, val = next(iter(eff.items()))
        if key == 'tension':
          story.shift_tension(float(val))
          applied_notes.append(f'tension->{story.tension:.1f}')
        elif key == 'faction_shift':
          if isinstance(val, Sequence) and len(val) >= 2:
            fid_str = str(val[0])
            raw_delta = val[1]
            reason_note = str(val[2]) if len(val) >= 3 else ''
            if not reason_note and source and source != 'StoryDirector':
              fac_obj = story.factions.get(fid_str)
              aim_str = f' ({fac_obj.aim})' if fac_obj and fac_obj.aim else ''
              if isinstance(raw_delta, (int, float)) and raw_delta < 0:
                reason_note = f'Friction escalated via {source}{aim_str}.'
              elif isinstance(raw_delta, (int, float)) and raw_delta > 0:
                reason_note = f'Leverage aligned via {source}{aim_str}.'
            new_val = story.shift_faction(
                fid_str,
                raw_delta,
                reason=reason_note,
                tick=self._ledger.tick,
            )
            applied_notes.append(f'faction:{fid_str}->{new_val}')
        elif key == 'inter_faction_shift':
          if isinstance(val, Sequence) and len(val) >= 3:
            fa, fb, delta_v = str(val[0]), str(val[1]), val[2]
            reason_note = str(val[3]) if len(val) >= 4 else ''
            new_if = story.shift_inter_faction(
                fa,
                fb,
                delta_v,
                reason=reason_note,
                tick=self._ledger.tick,
            )
            if new_if is not None:
              applied_notes.append(f'inter_faction:{fa}<->{fb}->{new_if}')
              if new_if <= -4:
                pair = sorted([fa, fb])
                self._ledger.set_world_fact(
                    f'turf_war_{pair[0]}_{pair[1]}', 'active'
                )
        elif key == 'reveal_clue':
          cid = str(val)
          clue_obj = story.clues.get(cid)
          was_discovered = bool(clue_obj and clue_obj.discovered)
          truth = story.mark_clue_discovered(cid, tick=self._ledger.tick)
          self._ledger.add_clue_to_inventory(cid)
          if clue_obj is not None and not was_discovered:
            self._ensure_text_supported_in_log(
                clue_obj.text,
                label=clue_obj.carrier_hint or cid,
                location_id=self._ledger.player_location_id,
            )
          applied_notes.append(f'clue:{cid}')
          if truth is not None:
            existing_truth_ticks = {
                t.discovered_tick
                for t in story.truths.values()
                if t.discovered and t.id != truth.id
            }
            if (
                self._ledger.tick > 1
                and truth.discovered_tick in existing_truth_ticks
            ):
              truth.discovered_tick = max(existing_truth_ticks) + 2
            applied_notes.append(f'truth:{truth.id}')
            story.recompute_act()
        elif key == 'move_clue':
          if isinstance(val, Sequence) and len(val) == 2:
            cid, target_spec = str(val[0]), str(val[1])
            moved_note = self._ledger.relocate_clue(
                cid, target_spec, by_npc=source
            )
            if moved_note:
              applied_notes.append(f'move_clue:{cid}->{target_spec}')
        elif key == 'relationship':
          if isinstance(val, Sequence) and len(val) == 3:
            npc_name, target, note = str(val[0]), str(val[1]), str(val[2])
            if target == 'player':
              target = self._ledger.player_name
            self._ledger.set_npc_relationship(npc_name, target, note)
            applied_notes.append(f'rel:{npc_name}->{target}')
        elif key == 'rumor':
          if isinstance(val, Mapping):
            text = str(val.get('text', '')).strip()
            r_source = str(val.get('source', source))
            rel = str(val.get('reliability', 'true'))
          else:
            text = str(val).strip()
            r_source = source
            rel = 'true'
          if text:
            self._ledger.record_rumor(
                text, source=r_source, is_warm=is_warm, reliability=rel
            )
            applied_notes.append(f'rumor:{rel}')
        elif key == 'add_affordance':
          if isinstance(val, Sequence) and len(val) == 2:
            loc_spec, prop = str(val[0]), str(val[1]).strip()
            target_id = (
                self._ledger.player_location_id
                if loc_spec == 'player_location'
                else self._ledger.ensure_stub_location(loc_spec)
            )
            self._ledger.add_location_affordance(target_id, prop)
            applied_notes.append(f'affordance:{target_id}')
        elif key == 'set_fact':
          if isinstance(val, Sequence) and len(val) == 2:
            self._ledger.set_world_fact(str(val[0]), str(val[1]))
            applied_notes.append(f'fact:{val[0]}')
        elif key == 'journal':
          entry = str(val).strip()
          if entry:
            story.journal.append(f'[tick {self._ledger.tick}] {entry}')
        elif key == 'set_act':
          story.act = max(story.act, int(val))
        elif key == 'end_run':
          outcome_val = str(val)
          if (
              outcome_val.lower() == 'failure'
              and story.ending_archetype
              and story.ending_archetype != 'eleventh_hour_bell'
          ):
            continue
          story.ended = True
          story.ending_summary = self.compose_epilogue(outcome=outcome_val)
          applied_notes.append(f'ended:{val}')
    return applied_notes

  def fire_storylet(
      self,
      storylet: story_state.Storylet,
      *,
      source: str = 'StoryDirector',
      is_warm: bool = False,
  ) -> list[str]:
    """Marks `storylet` fired and applies its effects."""
    with self._ledger.lock:
      self._ledger.story.note_fired(
          storylet.id, tick=self._ledger.tick, source=source
      )
      return self.apply_effects(
          storylet.effects, source=source, is_warm=is_warm
      )

  # ---------------------------------------------------------------------------
  # Seam 1: Scene context (Director Notes for FastSceneResolver)
  # ---------------------------------------------------------------------------

  def notes_for_scene(
      self,
      location_id: str | None = None,
      *,
      is_arrival: bool = False,
  ) -> str:
    """Formats the hidden Director Notes block for `FastSceneResolver`."""
    if not self.active:
      self._last_offered_ids = []
      return ''

    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      loc_id = world_ledger.normalize_location_id(
          location_id or ledger.player_location_id
      )
      node = ledger.get_location(loc_id)
      co_npcs = ledger.npcs_at_location(loc_id)

      # Pick up to 2 scene storylets (+ 1 arrival/ending if applicable).
      last_player_actions = [
          str(e.get('text') or '').strip().lower()
          for e in ledger.get_narrative_log()[-12:]
          if e.get('kind') == 'player_action'
      ]
      is_move_turn = bool(
          ledger.tick > 1
          and last_player_actions
          and re.search(
              r'^(?:\[[^\]]+\]\s*)?(?:go|walk|head|move|travel|return|step)\b',
              last_player_actions[-1],
          )
      )
      beats: list[story_state.Storylet] = []
      climax_ready = not is_move_turn and self.is_climax_ready(loc_id)
      if climax_ready or (
          not is_move_turn
          and (
              story.truths_discovered() >= max(1, len(story.truths))
              or (story.doom_stage >= max(1, len(story.doom.stages) - 1))
          )
      ):
        ending_beats = self.pick('ending', location_id=loc_id, k=1)
        beats.extend(ending_beats)
      if is_arrival or is_move_turn:
        beats.extend(self.pick('arrival', location_id=loc_id, k=1))
      scene_candidates = self.pick('scene', location_id=loc_id, k=4)
      if is_move_turn and story.tone.reveal_strictness == 'strict':
        scene_candidates = [
            b
            for b in scene_candidates
            if 'clue_gate' not in b.tags
            and 'payoff' not in b.tags
            and not any('reveal_clue' in eff for eff in b.effects)
        ]
      beats.extend(scene_candidates[:2])
      # Deduplicate while preserving order, cap at 2.
      seen: set[str] = set()
      unique_beats: list[story_state.Storylet] = []
      for b in beats:
        if b.id not in seen:
          seen.add(b.id)
          unique_beats.append(b)
      unique_beats = unique_beats[:2]
      self._last_offered_ids = [b.id for b in unique_beats]

      # Undiscovered clues present in this room or carried by co-located NPCs.
      local_clues: list[story_state.Clue] = []
      if node is not None:
        for cid in node.clue_ids:
          c = story.clues.get(cid)
          if c is not None and not c.discovered:
            local_clues.append(c)
      for npc in co_npcs:
        for cid in npc.known_clue_ids:
          c = story.clues.get(cid)
          if c is not None and not c.discovered and c not in local_clues:
            local_clues.append(c)
      # Match unplaced clues whose carrier_hint names this room or an NPC here.
      room_title = (node.name if node else loc_id).lower()
      co_lower = {n.name.lower() for n in co_npcs}
      for c in story.clues.values():
        if c.discovered or c in local_clues or c.placed_at is not None:
          continue
        hint = c.carrier_hint.lower()
        if (room_title and room_title in hint) or any(
            n in hint for n in co_lower
        ):
          local_clues.append(c)

      # Doom & act line.
      rem_min = story.doom_minutes_remaining(ledger.clock_minutes)
      rem_h = rem_min // 60
      rem_m = rem_min % 60
      stage = story.stage_for_fraction(
          story.doom_fraction(ledger.clock_minutes)
      )
      act_note = story.current_act_note()

      is_concise = ledger.prose_mode == 'concise'
      if is_concise:
        tone_line = (
            f'- Tone ({story.tone.name} — CONCISE MODE): Crisp, direct, '
            'to-the-point reportage (1-2 short sentences, max 40 words total). '
            'Put the concrete action outcome in the very first clause; cut '
            'decorative adjectives, weather metaphors, and flowery filler.'
        )
      else:
        tone_line = f'- Tone ({story.tone.name}): {story.tone.prose_register}'
      lines = [
          (
              'Director Notes (private guidance for the GM; NEVER copy labels'
              ' or IDs verbatim into narrative_event):'
          ),
          tone_line,
          (
              f'- Pacing: Act {story.act}, Tension {story.tension:.1f}/10, '
              f'Countdown "{story.doom.name}" at stage {stage.index} '
              f'("{stage.name}", {rem_h}h{rem_m:02d}m remain).'
          ),
      ]
      if stage.world_effect:
        lines.append(f'- Stage atmosphere: {stage.world_effect}')
      if act_note:
        lines.append(f'- {act_note}')
      if climax_ready:
        lines.append(
            '- EMERGENT ACT 3 CLIMAX WINDOW (OPEN NOW): The mystery and '
            'underworld tension have reached critical mass. If the player '
            'makes a decisive endgame move here (settling the main debt, '
            'brokering a rival carve-up, exposing the corrupt ledger to '
            'authorities, framing a crew boss, or igniting a turf war '
            'showdown), resolve it as a conclusive climactic scene paying off '
            "their inventory evidence and the co-located NPC's exposed "
            'Sub-Mind.'
        )

      # Pacing nudge from the tension scalar.
      if story.tension <= 2.5:
        lines.append(
            '- Pacing directive: tension is LOW — introduce a sharp '
            'complication, suspicious notice, or faction pressure.'
        )
      elif story.tension >= 7.5:
        lines.append(
            '- Pacing directive: tension is HIGH — allow a breath, a '
            'quiet detail, or a moment of fragile candor if earned.'
        )

      if unique_beats:
        lines.append(
            '- Authored Beats Eligible Now (weave AT MOST ONE if it fits the '
            'putative action, and report its id in `delta.beat_used`):'
        )
        for b in unique_beats:
          voice_hint = ''
          if story.tone.inner_voices and b.voice:
            if (
                b.voice_line
                and 'clue_gate' not in b.tags
                and story.tone.reveal_strictness != 'strict'
            ):
              voice_hint = (
                  f' [Suggested inner_voice="{b.voice}: {b.voice_line}"]'
              )
            else:
              voice_hint = f' [Suggested inner_voice faculty: {b.voice}]'
          lines.append(f'  * [{b.id}] {b.beat}{voice_hint}')

      if local_clues:
        if story.tone.reveal_strictness == 'strict':
          strict_note = (
              'ONLY if the actor specifically examines, forces, trades for, '
              'or corners the carrier (never on a casual glance)'
          )
        else:
          strict_note = 'when the actor inspects or asks about the carrier'
        lines.append(
            f'- Undiscovered Clues Here (reveal {strict_note}; list any '
            'revealed clue IDs in `delta.clues_revealed`):'
        )
        for c in local_clues[:4]:
          lines.append(
              f'  * {c.id} ({c.carrier} — {c.carrier_hint}): "{c.text}"'
          )

      # Secrets, wounds & Inner Parliament of Sub-Minds for co-located NPCs.
      secret_lines = []
      for npc in co_npcs:
        fac_tag = f' [{npc.faction}]' if npc.faction else ''
        dom = npc.dominant_sub_mind()
        council_parts = [
            f'{sm.name} [Rank {sm.rank}{", INTRUSIVE" if sm.intrusive else ""}]'
            for sm in npc.sub_minds.values()
        ]
        if npc.secret or npc.wound:
          secret_lines.append(
              f'  * {npc.name}{fac_tag}: secret="{npc.secret}" | '
              f'wound/leverage="{npc.wound}" (lie probability '
              f'{int(story.tone.npc_lie_probability * 100)}% unless leverage '
              'is applied).'
          )
        secret_lines.append(
            f'  * {npc.name} Inner Parliament (Leading: {dom.name} '
            f'[Rank {dom.rank}] — obsession="{dom.obsession}"): '
            f'{", ".join(council_parts)}.'
        )
      if secret_lines:
        lines.append('- Present NPC Secrets & Leverage (hidden from player):')
        lines.extend(secret_lines)
        if is_concise:
          lines.append(
              "- Present NPC Reaction: Show the co-located NPC's immediate "
              'physical reaction or pushback in one crisp clause so they can '
              'answer on their turn.'
          )
        else:
          lines.append(
              '- Emergent NPC Counter-Agency (MANDATORY whenever an NPC is '
              'present): Even when weaving an authored beat, the co-located '
              'NPC MUST initiate an UNPROMPTED emergent personal side-gambit '
              'or counter-demand not scripted in the beat—such as proposing a '
              'double-cross against a third NPC/faction, demanding a side-cut '
              'or physical collateral, or quietly tampering with a secondary '
              'room prop—so stance shifts arise from active two-way '
              'negotiation rather than passive storylet compliance.'
          )

      if story.factions:
        lines.append('- Faction Dispositions toward Player (natural language):')
        for f in story.factions.values():
          lines.append(f'  * {f.name} ({f.id}): {f.current_disposition()}')
        war_descs = story.describe_turf_wars()
        if war_descs:
          lines.append(
              '- Active Inter-Faction Turf Wars (crews shoot/sabotage on '
              f'sight): {"; ".join(war_descs)}'
          )

      if node is not None and not is_concise:
        warm_traces = [
            p
            for p in node.affordances
            if p.lower().startswith((
                'physical spillover from ',
                'evidence of ',
                'signs of a reprisal ',
                'fresh jimmy marks ',
            ))
        ]
        if warm_traces:
          lines.append(
              '- MANDATORY OFF-SCREEN SABOTAGE / SPILLOVER PAYOFF (you MUST '
              'visibly trigger, detonate, or force the player/NPC to '
              'physically collide with or overcome at least one of these '
              'off-screen preparations in `narrative_event` so it produces a '
              'concrete foreground mechanical consequence): '
              f'{"; ".join(warm_traces[-2:])}'
          )

      if story.tone.inner_voices and story.player_voices:
        story.sync_voice_maps()
        if self._last_invoked_voice:
          inv_sm = story.get_sub_mind(self._last_invoked_voice)
          if inv_sm is not None:
            if inv_sm.rank >= 3:
              leverage_tier = (
                  f'HIGH MECHANICAL LEVERAGE (Rank {inv_sm.rank}) — Overcome '
                  'standard NPC stonewalling, spot concealed tells in the '
                  f'"{inv_sm.domain}" domain, or force routine barriers.'
              )
            else:
              leverage_tier = (
                  f'Standard Leverage (Rank {inv_sm.rank}) — Honor the '
                  f'"{inv_sm.domain}" approach ({inv_sm.worldview}), though '
                  'entrenched resistance may exact a partial cost.'
              )
            lines.append(
                f'- Active Player Sub-Mind Channel ({inv_sm.name} '
                f'[Rank {inv_sm.rank}, Domain: {inv_sm.domain}]): '
                f'{leverage_tier}'
            )
            if inv_sm.scars:
              lines.append(
                  f'  * Formative Scars on {inv_sm.name}: '
                  f'{"; ".join(inv_sm.scars[-2:])}'
              )
            if inv_sm.obsession:
              lines.append(
                  f'  * Current Obsession of {inv_sm.name}: {inv_sm.obsession}'
              )
            if not is_concise and (inv_sm.intrusive or inv_sm.rank >= 4):
              lines.append(
                  f'  * OVER-DOMINANCE BLIND SPOT ({inv_sm.name} Rank '
                  f'{inv_sm.rank}): While the primary "{inv_sm.domain}" '
                  'action succeeds with force, weave a subtle side-cost or '
                  'tunnel-vision complication from its blind spot '
                  f'({inv_sm.blind_spot}).'
              )
        active_voices = self.select_active_voices(loc_id, k=4)
        voices_str = '; '.join(
            f'{v_name} [Lv {aff}] ({v_desc} | Scene focus: {focus})'
            for v_name, v_desc, focus, aff in active_voices
        )
        priority_faculties = [v_name for v_name, _, _, _ in active_voices[:2]]
        priority_note = ''
        if priority_faculties:
          priority_note = (
              f' Use {" and ".join(priority_faculties)} in `delta.inner_voices`'
              ' reacting specifically to this action.'
          )
        prev_actions = [
            f'{opt.voice}: "{opt.action}"'
            for opt in story.active_voice_options
            if opt.action
        ]
        prev_note = ''
        if prev_actions:
          prev_note = (
              ' NEVER repeat or paraphrase these prior options: '
              + '; '.join(prev_actions)
              + '.'
          )
        lines.append(
            '- Inner Voices / Active Sub-Mind Council (HRL Options — you MUST '
            'regenerate 4 fresh entries in `delta.voice_options` reacting '
            'directly to the NEW outcome of this turn as `[{"voice": '
            '"<Faculty>", "perception": "<punchy 1-sentence read (max 15 '
            'words)>", "action": "<concise 6-12 word 1-turn action>"}]`, '
            'and emit 1-2 brief `"Faculty: line"` strings in '
            '`delta.inner_voices`. Keep each faculty in its strict cognitive '
            'lane: ONLY Proper Naughty proposes physical force, shoving, or '
            'jemmying; Brass Neck proposes audacious verbal bluffs or pulling '
            'rank (never physical shoving/violence); The Hustle proposes '
            'transactional swaps, three-way arbitrage, or margin-skimming '
            '(never physical threats); The Book cites exact ledger figures or '
            "odds; Copper's Nose cites forensic or police-procedural "
            'evidence; Ear to the Ground reads physical micro-tells; The '
            "Guv'nor names a specific crew boss to pin the fallout on."
            f'{priority_note}{prev_note}): {voices_str}'
        )
        if not story.active_voice_options:
          story.active_voice_options = self.synthesize_fallback_voice_options(
              loc_id, k=4
          )

      if story.journal:
        recent_j = '; '.join(story.journal[-12:])
        lines.append(f'- Player Journal So Far: {recent_j}')

      return '\n'.join(lines)

  # ---------------------------------------------------------------------------
  # Seam 1a-bis: HRL Sub-Minds (Deterministic Initiation Set & Option Synthesis)
  # ---------------------------------------------------------------------------

  def select_active_voices(
      self, location_id: str | None = None, *, k: int = 4
  ) -> list[tuple[str, str, str, int]]:
    """Deterministically selects the top `k` context-relevant sub-minds.

    Args:
      location_id: Location to inspect (defaults to player's location).
      k: Number of sub-minds to activate.

    Returns:
      List of `(voice_name, voice_description, scene_focus_hint, affinity)`.
    """
    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      if not story.player_voices:
        return []
      loc_id = location_id or ledger.player_location_id
      node = ledger.get_location(loc_id)
      co_npcs = ledger.npcs_at_location(loc_id)
      props = [
          p
          for p in (list(node.affordances) if node is not None else [])
          if not p.lower().startswith((
              'physical spillover from ',
              'evidence of ',
              'signs of ',
              'reprisal prepared by ',
              'fresh jimmy marks ',
              'confront ',
              'present ',
              'demand ',
              'wrestle ',
          ))
      ]
      props_lower = ' '.join(props).lower()

      local_clues: list[story_state.Clue] = []
      if node is not None:
        for cid in node.clue_ids:
          c = story.clues.get(cid)
          if c is not None and not c.discovered:
            local_clues.append(c)
      for npc in co_npcs:
        for cid in npc.known_clue_ids:
          c = story.clues.get(cid)
          if c is not None and not c.discovered and c not in local_clues:
            local_clues.append(c)

      discovered_clues = [c for c in story.clues.values() if c.discovered]
      doc_clues = [c for c in local_clues if c.carrier == 'document']
      loc_clues = [c for c in local_clues if c.carrier == 'location']
      npc_clues = [c for c in local_clues if c.carrier == 'npc']
      prev_voice_names = {opt.voice for opt in story.active_voice_options}
      tick_num = int(ledger.tick)

      scored: list[tuple[float, int, str, str, str]] = []
      story.sync_voice_maps()
      voice_items = list(story.player_voices.items())
      num_voices = len(voice_items)
      spoken_counts: dict[str, int] = {k: 0 for k in story.player_voices}
      for entry in ledger.get_narrative_log():
        if entry.get('kind') == 'inner_voice':
          act_name = str(entry.get('actor', ''))
          if act_name in spoken_counts:
            spoken_counts[act_name] += 1
      total_spoken = sum(spoken_counts.values())
      dominant_voice = (
          max(story.voice_affinity.items(), key=lambda item: item[1])[0]
          if story.voice_affinity and story.voice_choice_log
          else ''
      )
      recent_ev_lower = (
          node.recent_events[-1].lower()
          if (node is not None and node.recent_events)
          else ''
      )
      co_npc_blob = ' '.join(
          f'{n.name} {n.faction} {n.goal}'.lower() for n in co_npcs
      )
      scene_blob = f'{props_lower} {co_npc_blob} {recent_ev_lower}'
      for idx, (v_name, v_desc) in enumerate(voice_items):
        sm = story.get_sub_mind(v_name)
        aff = max(
            1, int(sm.rank if sm else story.voice_affinity.get(v_name, 1))
        )
        # Rotate voices across ticks and boost voices that sat out last turn so
        # the sub-mind council evolves on every step.
        rot_bonus = 0.9 if ((idx + tick_num) % max(1, num_voices)) < 3 else 0.0
        fresh_bonus = (
            1.15 if prev_voice_names and v_name not in prev_voice_names else 0.0
        )
        cov_bonus = 0.0
        if total_spoken > 0 and spoken_counts.get(v_name, 0) == 0:
          cov_bonus = 2.4
        elif (
            total_spoken >= 6
            and spoken_counts.get(v_name, 0) / total_spoken > 0.34
        ):
          cov_bonus = -1.8
        dom_bonus = 1.8 if v_name == dominant_voice else 0.0
        intrusive_bonus = (
            1.25 if (sm is not None and (sm.intrusive or sm.rank >= 4)) else 0.0
        )
        emergent_bonus = (
            1.6
            if (sm is not None and sm.origin == 'emergent' and not sm.scars)
            else 0.0
        )
        score = (
            0.45 * aff
            + rot_bonus
            + fresh_bonus
            + cov_bonus
            + dom_bonus
            + intrusive_bonus
            + emergent_bonus
            - 0.01 * idx
        )
        v_lower = v_name.lower()
        combined = f'{v_name} {v_desc}'.lower()
        focus = 'Read the immediate angle in this room.'

        if any(tok in v_lower for tok in ('copper', 'nose', 'wardens')) or (
            'copper' in combined and 'naughty' not in v_lower
        ):
          if (
              (
                  node is not None
                  and node.faction_lean in ('flying_squad', 'wardens')
              )
              or any(n.faction in ('flying_squad', 'wardens') for n in co_npcs)
              or any(
                  w in props_lower
                  for w in (
                      'rover',
                      'radio',
                      'warrant',
                      'seal',
                      'writ',
                      'shells',
                      'polish',
                      'brazier',
                  )
              )
          ):
            score += 3.5
            focus = (
                'Spot the police procedure, forensic trace, or warrant bluff.'
            )
          else:
            score += 1.8
            focus = 'Check for forensic traces and official cover-ups.'

        elif any(
            tok in v_lower for tok in ('guv', 'hierarchy', 'rep', 'firm')
        ) or ('hierarchy' in combined and 'book' not in v_lower):
          if co_npcs or story.doom_stage >= 1 or story.act >= 2:
            target_label = co_npcs[0].name if co_npcs else 'the local crew'
            score += 3.0
            focus = (
                'Push an immediate syndicate power-play that pins the breach '
                f'of Firm protocol on {target_label} before the 23:00 '
                'reckoning (never summarize).'
            )
          else:
            score += 1.7
            focus = (
                'Identify which syndicate boss takes the fall when the '
                'governor checks the books.'
            )

        elif any(
            tok in combined
            for tok in ('book', 'ledger', 'odds', 'vig', 'account', 'count')
        ):
          if doc_clues:
            carrier = doc_clues[0].carrier_hint.split(',')[-1].strip()
            score += 3.0
            focus = f'Spot the discrepancy in {carrier}.'
          elif any(
              w in props_lower
              for w in (
                  'layoff pad',
                  'carbon',
                  'ledger',
                  'manifest',
                  'book',
                  'card',
                  'scale',
                  'slip',
                  'tally',
                  'chart',
              )
          ):
            score += 2.5
            focus = 'Audit the figures, dockets, and paper trail here.'
          else:
            score += 1.2
            focus = 'Calculate who is out of pocket in this room.'

        elif any(
            tok in combined
            for tok in (
                'naughty',
                'nerve',
                'mechanist',
                'bottle',
                'lock',
                'threat',
                'iron',
            )
        ):
          if loc_clues:
            carrier = loc_clues[0].carrier_hint.split(',')[-1].strip()
            score += 3.3
            focus = f'Force, pry, or search {carrier}.'
          elif any(
              w in props_lower
              for w in (
                  'lock',
                  'padlock',
                  'shutter',
                  'case',
                  'container',
                  'floor',
                  'vice',
                  'brazier',
                  'valve',
                  'chiller',
              )
          ):
            score += 2.9
            focus = 'Size up the locks, physical barriers, and heavy tools.'
          elif story.active_turf_wars():
            score += 3.1
            focus = 'Clock the sightlines and turf-war heat.'
          else:
            score += 1.4
            focus = 'Measure physical reach, exits, and concealed iron.'

        elif any(
            tok in combined
            for tok in ('ear', 'ground', 'empathy', 'suspicion', 'subtext')
        ):
          if co_npcs:
            npc0 = co_npcs[0]
            score += 3.3
            focus = f"Read {npc0.name}'s nervous tell and what they are hiding."
          elif npc_clues:
            score += 2.8
            focus = 'Listen for who is sweating or lying through their teeth.'
          else:
            score += 1.5
            focus = 'Pick up the street subtext and recent foot traffic.'

        elif any(
            tok in combined
            for tok in ('hustle', 'leverage', 'swap', 'arbitrage', 'trade')
        ):
          if co_npcs and (discovered_clues or ledger.player_inventory):
            npc0 = co_npcs[0]
            score += 3.6
            focus = (
                'Use your discovered evidence or inventory as leverage on '
                f'{npc0.name}.'
            )
          elif co_npcs:
            score += 2.5
            focus = f'Find out what {co_npcs[0].name} needs to make a deal.'
          else:
            score += 1.3
            focus = 'Look for something valuable enough to trade.'

        elif sm is not None and (
            sm.origin in ('custom', 'emergent')
            or sm.domain
            in (
                'passion',
                'predation',
                'decorum',
                'cybernetics',
                'tactical_force',
                'seduction',
            )
            or (
                'brass' not in v_lower
                and 'nerve' not in v_lower
                and sm.domain not in ('audacity',)
            )
        ):
          trig_hits = [
              trig
              for trig in sm.attention_triggers
              if trig and trig.lower() in scene_blob
          ]
          if trig_hits:
            score += 3.2
            focus = (
                f'Lock onto "{trig_hits[0]}" through {sm.domain} '
                f'({sm.obsession or sm.worldview}).'
            )
          elif co_npcs:
            score += 2.8
            focus = (
                f'Engage {co_npcs[0].name} through {sm.domain} '
                f'({sm.obsession or sm.worldview}).'
            )
          else:
            score += 2.2
            focus = (
                f'Apply {sm.domain} instinct ({sm.obsession or sm.worldview}).'
            )

        else:
          # Brass Neck / default audacity voice.
          score += 2.6 if not discovered_clues else 2.1
          if co_npcs:
            focus = (
                f'Bluff {co_npcs[0].name} like you already hold all the cards.'
            )
          else:
            focus = 'Act like you own the freehold and walk straight in.'

        if sm is not None:
          if sm.obsession and sm.obsession not in focus:
            focus = f'{focus} [Obsession: {sm.obsession}]'
          if sm.intrusive or sm.rank >= 4:
            focus = (
                f'{focus} [INTRUSIVE Rank {sm.rank} — Blind spot: '
                f'{sm.blind_spot}]'
            )
          elif sm.scars:
            focus = f'{focus} [Scar: {sm.scars[-1]}]'

        scored.append((score, aff, v_name, v_desc, focus))

      scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
      has_intrusive = any(
          sm.active and (sm.intrusive or sm.rank >= 4)
          for sm in story.sub_minds.values()
      )
      effective_k = (
          max(1, k - 1)
          if (story.character_initialized and has_intrusive and k >= 4)
          else max(1, k)
      )
      return [
          (v_name, v_desc, focus, aff)
          for _, aff, v_name, v_desc, focus in scored[:effective_k]
      ]

  def synthesize_fallback_voice_options(
      self,
      location_id: str | None = None,
      *,
      k: int = 4,
      latest_event: str = '',
  ) -> list[story_state.VoiceOption]:
    """Builds deterministic, event-reactive HRL VoiceOptions (0 LLM calls)."""
    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      if not story.tone.inner_voices or not story.player_voices:
        return []
      story.sync_voice_maps()
      loc_id = location_id or ledger.player_location_id
      node = ledger.get_location(loc_id)
      room_name = node.name if node is not None else loc_id
      props = [
          p
          for p in (list(node.affordances) if node is not None else [])
          if not p.lower().startswith((
              'physical spillover from ',
              'evidence of ',
              'signs of ',
              'reprisal prepared by ',
              'fresh jimmy marks ',
              'confront ',
              'present ',
              'demand ',
              'wrestle ',
          ))
      ]
      co_npcs = ledger.npcs_at_location(loc_id)
      discovered = [c for c in story.clues.values() if c.discovered]
      prev_actions = {
          opt.action.lower() for opt in story.active_voice_options if opt.action
      }
      tick_offset = int(ledger.tick)
      recent_ev = latest_event.strip()
      if not recent_ev and node is not None and node.recent_events:
        recent_ev = node.recent_events[-1].strip()
      short_ev = ''
      if recent_ev and len(recent_ev) > 12:
        short_ev = recent_ev.split('.')[0].strip()
        if len(short_ev) > 95:
          short_ev = short_ev[:92].rsplit(' ', 1)[0] + '...'

      selected = self.select_active_voices(loc_id, k=k)
      options: list[story_state.VoiceOption] = []
      used_props: set[str] = set()
      for idx, (v_name, v_desc, focus, aff) in enumerate(selected):
        sm = story.get_sub_mind(v_name)
        short_role = (
            v_desc.split('—', 1)[0].strip()
            if '—' in v_desc
            else v_desc.split('.', 1)[0].strip()
        )
        combined = f'{v_name} {v_desc}'.lower()
        prop_pick = ''
        if props:
          # Rotate prop choice by tick_offset and avoid picking the same prop
          # for multiple voices in the same step.
          rotated_props = [
              props[(i + tick_offset + idx) % len(props)]
              for i in range(len(props))
          ]
          for p in rotated_props:
            if p in used_props:
              continue
            pl = p.lower()
            if 'book' in combined and any(
                w in pl
                for w in (
                    'layoff pad',
                    'carbon',
                    'ledger',
                    'manifest',
                    'card',
                    'book',
                    'slip',
                    'board',
                )
            ):
              prop_pick = p
              break
            if (
                'naughty' in combined
                or 'nerve' in combined
                or 'chrome' in combined
            ) and any(
                w in pl
                for w in (
                    'case',
                    'shutter',
                    'container',
                    'floor',
                    'brazier',
                    'vice',
                    'lock',
                    'padlock',
                    'door',
                    'shed',
                    'hatch',
                    'bulkhead',
                    'valve',
                )
            ):
              prop_pick = p
              break
            if 'copper' in combined and any(
                w in pl
                for w in ('rover', 'glovebox', 'shells', 'polish', 'tray')
            ):
              prop_pick = p
              break
            if sm is not None and any(
                trig.lower() in pl for trig in sm.attention_triggers if trig
            ):
              prop_pick = p
              break
          if not prop_pick:
            for p in rotated_props:
              if p not in used_props:
                prop_pick = p
                break
            if not prop_pick:
              prop_pick = rotated_props[0]
          used_props.add(prop_pick)
          clean_prop = self._extract_clean_prop_noun_phrase(prop_pick)
          if clean_prop.lower().startswith('the '):
            clean_prop = clean_prop[4:].strip()
          clean_prop = re.sub(
              r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[\'’]s\s+',
              '',
              clean_prop,
          ).strip()
          if clean_prop:
            prop_pick = clean_prop

        ev_prefix = ''
        clean_focus = focus.split(' [')[0].strip()
        v_low = v_name.strip().lower()
        dom = sm.domain if sm is not None else 'audacity'
        obs_clause = ''
        npc0 = co_npcs[(idx + tick_offset) % len(co_npcs)] if co_npcs else None

        # 1. Glassharbor voices
        if v_low == 'suspicion':
          if npc0 is not None:
            perception = (
                f"{npc0.name}'s story has a hook in it—watch what they are "
                f'trying to steer your eyes away from in {room_name}.'
            )
            action = (
                f'Press {npc0.name} on what does not add up about the '
                f'{prop_pick or "scene"} and watch their eyes.'
            )
          else:
            perception = (
                f'Something about the {prop_pick or room_name} has been '
                'staged or tampered with.'
            )
            action = (
                f'Inspect the {prop_pick or room_name} closely for replaced '
                'seals, fresh tool marks, or tampering.'
            )
        elif v_low == 'empathy':
          if npc0 is not None:
            perception = (
                f"Beneath {npc0.name}'s guarded tone is someone afraid of "
                'what happens if their secret comes out.'
            )
            action = (
                f'Speak gently to {npc0.name} and ask who is holding leverage '
                'over them before the tide turns.'
            )
          else:
            perception = (
                f'Someone handled the {prop_pick or room_name} with shaking '
                'hands not long ago.'
            )
            action = (
                f'Examine the {prop_pick or room_name} for the personal trace '
                'its owner tried to hide.'
            )
        elif 'salt-memory' in v_low or 'salt memory' in v_low:
          perception = (
              f'You grew up on salt-piers like {room_name}—the smell of tar,'
              ' wet hemp, and rising water tells you where dockhands hide'
              ' things.'
          )
          action = (
              f'Check the {prop_pick or room_name} the way a lifelong harbor '
              'hand would and trace who moved cargo here.'
          )
        elif v_low == 'ledger':
          perception = (
              'The dates, seals, and figures tied to the '
              f'{prop_pick or room_name} do not balance—someone altered the '
              'record.'
          )
          action = (
              f'Audit the {prop_pick or "port records"} line by line and '
              'cross-check the altered dates and initials.'
          )
        elif v_low == 'nerve':
          perception = (
              f'Measure the exits and physical locks in {room_name}—the'
              f' {prop_pick or "barrier"} is what stands between you and the'
              ' proof.'
          )
          action = (
              f'Force or test the fittings on the {prop_pick or "door"} '
              'before the watch can stop you.'
          )
        # 2. Dystopian Sci-Fi & Vampire Romance & Custom/Emergent voices
        elif sm is not None and (
            sm.origin in ('custom', 'emergent')
            or dom
            in (
                'passion',
                'predation',
                'decorum',
                'cybernetics',
                'tactical_force',
                'seduction',
            )
            or v_low
            in (
                'dead-reckoning',
                'cabin fever',
                'hard line',
                'company book',
                'side-angle',
                'biometric lens',
                'cold calculus',
                'residual empathy',
                'black-market signal',
                'union memory',
                'cover story',
                'paper trail',
                'night chart',
                'whisper network',
                'ancient lore',
                'mortal echo',
            )
            or not any(
                k in combined
                for k in (
                    'brass',
                    'hustle',
                    'book',
                    'copper',
                    'naughty',
                    'ear',
                    'guv',
                    'mechanist',
                    'jurisdiction',
                )
            )
        ):
          if v_low in ('dead-reckoning', 'dead reckoning'):
            perception = (
                f'The numbers in {room_name} do not add up—check the '
                f'{prop_pick or "station log"} against the three-day gap.'
            )
            action = (
                f'Inspect the {prop_pick or "station log"} and cross-check '
                'the timestamps and weights.'
            )
          elif v_low == 'cabin fever':
            if npc0 is not None:
              perception = (
                  f'Watch how {npc0.name} holds their hands—twelve months at '
                  'an isolated station makes a real habit impossible to fake.'
              )
              action = (
                  f'Ask {npc0.name} a plain question about home and watch '
                  'whether their memory skips.'
              )
            else:
              perception = (
                  f'Someone in {room_name} left the {prop_pick or "table"} '
                  'in a hurry when the supply sled landed.'
              )
              action = (
                  f'Examine the {prop_pick or "table"} for personal habits '
                  'that tell the two men apart.'
              )
          elif v_low == 'hard line':
            if npc0 is not None and idx % 2 == 0:
              perception = (
                  f'{npc0.name} is within arm’s reach of the '
                  f'{prop_pick or "heavy tools"}—keep the table between you.'
              )
              action = (
                  f'Step back to clear your sightline on {npc0.name} and '
                  f'secure the {prop_pick or "door latch"}.'
              )
            else:
              perception = (
                  'In a four-person station, whoever controls the '
                  f'{prop_pick or "door latch"} controls the room.'
              )
              action = (
                  f'Check and secure the {prop_pick or "door latch"} before '
                  'anyone can lock you in the cold.'
              )
          elif v_low == 'company book':
            if npc0 is not None:
              perception = (
                  f'{npc0.name} knows the company retrieval team lands at '
                  '06:00 AM—and whoever signs the handover sheet walks free.'
              )
              action = (
                  'Put your marshal’s authority on the table and demand '
                  f'{npc0.name} turn over the company dispatch.'
              )
            else:
              perception = (
                  f'The {prop_pick or "cabinet"} carries company property '
                  'seals meant to keep the crew quiet.'
              )
              action = (
                  f'Audit the {prop_pick or "cabinet"} under your handover '
                  'warrant.'
              )
          elif v_low in ('side-angle', 'side angle'):
            if npc0 is not None:
              perception = (
                  f'{npc0.name} wants a clean seat on the 06:00 AM transport '
                  'more than they care about company loyalty.'
              )
              action = (
                  f'Offer {npc0.name} a clean line in your marshal’s report '
                  'in exchange for who opened the crate.'
              )
            else:
              perception = (
                  f'The {prop_pick or "stash"} in {room_name} is leverage '
                  'when the morning shuttle lands.'
              )
              action = (
                  f'Search the {prop_pick or "stash"} for something you can '
                  'trade for a confession.'
              )
          elif v_low in (
              'biometric lens',
              'bio-synthetic pulse',
              'the mirror test',
              'mirror test',
          ):
            if npc0 is not None:
              perception = (
                  'A three-day sedative purge leaves chemical haze in '
                  f'{npc0.name}’s cornea and an IV puncture stamp on their '
                  'wrist.'
              )
              action = (
                  f'Hold your pocket scanner up to {npc0.name}’s eyes and '
                  'ask them to roll up their wrist cuff.'
              )
            else:
              perception = (
                  f'Run your pocket scanner over the {prop_pick or room_name} '
                  'to check whose prints and cold-storage residue are on it.'
              )
              action = (
                  f'Scan the {prop_pick or room_name} with your biometric '
                  'reader and compare it with the Polaroid from the crate.'
              )
          elif dom == 'cybernetics':
            if prop_pick and any(
                w in prop_pick.lower()
                for w in (
                    'terminal',
                    'holo',
                    'hatch',
                    'node',
                    'console',
                    'rig',
                    'spine',
                    'relay',
                )
            ):
              perception = (
                  'Unlogged diagnostic traffic is leaking from the '
                  f'{prop_pick} in {room_name}.'
              )
              action = (
                  f'Hook your diagnostic reader into the {prop_pick} and pull '
                  'the unlogged override telemetry.'
              )
            elif npc0 is not None:
              perception = (
                  f"{npc0.name}'s pupil response and pulse cadence shift "
                  'whenever the quarantine orders are mentioned.'
              )
              action = (
                  f'Run your diagnostic scanner on {npc0.name} and press them '
                  'on the scrubbed containment log.'
              )
            else:
              perception = (
                  f'The wiring around the {prop_pick or room_name} '
                  'carries an unlogged manual override.'
              )
              action = (
                  'Run a diagnostic sweep on the '
                  f'{prop_pick or room_name} to isolate the hidden signal.'
              )
          elif dom == 'tactical_force':
            if npc0 is not None and idx % 2 == 0:
              perception = (
                  f'{npc0.name} is watching the bulkhead sightlines and '
                  f'keeping one hand near their sidearm{obs_clause}.'
              )
              action = (
                  f"Step into {npc0.name}'s firing angle near the "
                  f'{prop_pick or "bulkhead"} and demand the override code.'
              )
            else:
              perception = (
                  f'The {prop_pick or "bulkhead"} is the physical choke point '
                  f'in {room_name}{obs_clause}.'
              )
              action = (
                  f'Wrench open or override the {prop_pick or "hatch"} to see '
                  'what is sealed behind it.'
              )
          elif dom == 'predation':
            if npc0 is not None:
              perception = (
                  f"{npc0.name}'s pulse jumps in their throat—fear and urgency"
                  f' press against their calm surface{obs_clause}.'
              )
              action = (
                  f'Step close to {npc0.name}, listen to the spike in their '
                  'heartbeat, and demand the truth.'
              )
            else:
              perception = (
                  'A faint scent of fresh iron and cold air clings to '
                  f'the {prop_pick or room_name}{obs_clause}.'
              )
              action = (
                  f'Examine the {prop_pick or room_name} closely to trace '
                  'whose blood or touch was left on it.'
              )
          elif dom == 'decorum' or v_low == 'cover story':
            if npc0 is not None:
              perception = (
                  f"In {room_name}, {npc0.name}'s routine composure is "
                  f'covering for a rule broken tonight{obs_clause}.'
              )
              action = (
                  f'Pin {npc0.name} down with calm night-shift procedure and '
                  'ask why the logs do not match.'
              )
            else:
              perception = (
                  f'The {prop_pick or room_name} breaks the clean cover story '
                  f'that kept you unseen{obs_clause}.'
              )
              action = (
                  f'Inspect the {prop_pick or room_name} and check who broke '
                  'routine before the morning shift arrives.'
              )
          elif dom in ('passion', 'seduction'):
            if npc0 is not None:
              perception = (
                  f"Beneath {npc0.name}'s guarded expression is a fierce "
                  f'devotion they are afraid to admit out loud{obs_clause}.'
              )
              action = (
                  f'Lower your voice so only {npc0.name} hears and promise you'
                  ' will not let the person they are protecting face sunrise'
                  ' alone.'
              )
            else:
              perception = (
                  f'The {prop_pick or room_name} was left behind in haste—proof'
                  f' of a promise someone broke the rules to keep{obs_clause}.'
              )
              action = (
                  f'Examine the {prop_pick or room_name} closely for what it '
                  'reveals about who they are trying to protect.'
              )
          elif v_low == 'cold calculus':
            perception = (
                'The oxygen-credit quotas and scrubber numbers in '
                f'{room_name} do not match a real contagion{obs_clause}.'
            )
            if npc0 is not None:
              action = (
                  'Lay the oxygen-manifest discrepancy in front of '
                  f'{npc0.name} and ask who Helix Directorate is scapegoating.'
              )
            else:
              action = (
                  f'Audit the {prop_pick or "telemetry log"} down to the '
                  'kiloliter to expose the falsified O2 deficit.'
              )
          elif v_low == 'residual empathy':
            if npc0 is not None:
              perception = (
                  f"Behind {npc0.name}'s synthetic voice-filter is a human"
                  ' terrified of who gets left behind when the airlock'
                  f' vents{obs_clause}.'
              )
              action = (
                  f'Speak past the corporate protocol to {npc0.name} and ask '
                  'who they are trying to protect in Sector 9.'
              )
            else:
              perception = (
                  f'A human trace lingers on the {prop_pick or room_name} '
                  f'beneath the sterile corporate lockdown{obs_clause}.'
              )
              action = (
                  f'Examine the {prop_pick or room_name} for the personal '
                  'record someone tried to scrub.'
              )
          elif v_low == 'black-market signal':
            if npc0 is not None:
              perception = (
                  f'{npc0.name} needs an unlogged transit angle out of Sector 9'
                  f' just as badly as you need the raw shard{obs_clause}.'
              )
              action = (
                  f'Offer {npc0.name} an off-grid clearance trade in exchange '
                  'for what they know about the lockdown.'
              )
            else:
              perception = (
                  f'The {prop_pick or room_name} sits right on a black-market '
                  f'dead-drop route{obs_clause}.'
              )
              action = (
                  f'Check the {prop_pick or room_name} for an unlogged '
                  'smuggler cache or transit key.'
              )
          elif v_low == 'union memory':
            perception = (
                f'Every coolant manifold and welded hatch in {room_name} was '
                f'built by the sub-deck shift crews{obs_clause}.'
            )
            if npc0 is not None:
              action = (
                  f'Invoke the sub-deck union compact with {npc0.name} and '
                  'ask where the scrubber bypass is wired.'
              )
            else:
              action = (
                  f'Inspect the {prop_pick or "scrubber hatch"} for the '
                  "shift-crew's manual bypass markings."
              )
          elif v_low in ('paper trail', 'whisper network'):
            if npc0 is not None:
              perception = (
                  f'{npc0.name} is holding back an off-the-books record—find'
                  ' out what they want in exchange for keeping it'
                  f' quiet{obs_clause}.'
              )
              action = (
                  f'Ask {npc0.name} plainly what it will take to keep that '
                  'file off the morning supervisor’s desk.'
              )
            else:
              perception = (
                  f'The {prop_pick or room_name} holds the paper trail of who '
                  f'accessed the records tonight{obs_clause}.'
              )
              action = (
                  f'Search the {prop_pick or room_name} for sign-out slips, '
                  'archived ID photos, or unlogged inquiries.'
              )
          elif v_low in ('night chart', 'ancient lore'):
            perception = (
                'The clinical signs and inventory numbers on the '
                f'{prop_pick or room_name} tell the real story of what '
                f'happened an hour ago{obs_clause}.'
            )
            action = (
                f'Inspect the {prop_pick or "sign-out binder"} and cross-check '
                'the missing O-negative bags and triage notes.'
            )
          elif v_low == 'mortal echo':
            if npc0 is not None:
              perception = (
                  f'Beneath the fluorescent lights of {room_name}, {npc0.name}'
                  ' is carrying grief and fear that threats will only make'
                  f' worse{obs_clause}.'
              )
              action = (
                  f'Speak honestly with {npc0.name} about what decades of '
                  'hiding actually cost and offer a humane way out.'
              )
            else:
              perception = (
                  f'The {prop_pick or room_name} holds a quiet trace of the '
                  f'ordinary human life at stake tonight{obs_clause}.'
              )
              action = (
                  f'Examine the {prop_pick or room_name} for what it reveals '
                  'about the human cost of tonight’s choices.'
              )
          else:
            if sm.favored_tactics:
              tactic_idx = (idx + tick_offset) % len(sm.favored_tactics)
              tactic_clause = sm.favored_tactics[tactic_idx]
            else:
              tactic_clause = ''
            if npc0 is not None:
              perception = (
                  f'{sm.name} clocks what {npc0.name} is holding back in '
                  f'{room_name}{obs_clause}.'
              )
              if tactic_clause and len(tactic_clause.split()) >= 4:
                action = (
                    f'Turn toward {npc0.name} and'
                    f' {tactic_clause[0].lower() + tactic_clause[1:]}.'
                )
              else:
                action = (
                    f'Press {npc0.name} on the {prop_pick or "truth"} using '
                    f'{sm.name}.'
                )
            elif prop_pick:
              perception = (
                  f'{sm.name} locks onto the {prop_pick} in '
                  f'{room_name}{obs_clause}.'
              )
              action = f'Inspect and test the {prop_pick} closely.'
            else:
              perception = f'{clean_focus}{obs_clause}'
              action = f'Read {room_name} through the lens of {sm.name}.'
        elif co_npcs and any(
            t in combined for t in ('ear', 'hustle', 'guv', 'brass', 'empathy')
        ):
          npc0 = co_npcs[(idx + tick_offset) % len(co_npcs)]
          if 'hustle' in combined and discovered:
            last_clue = world_ledger.clean_affordance_text(
                discovered[-1].carrier_hint.split(',')[-1].strip()
                or 'what you found'
            )
            perception = (
                f'{ev_prefix}{npc0.name} has an angle to protect here; your '
                f'{last_clue} is the wedge that makes them deal.'
            )
            action = (
                f'Confront {npc0.name} with {last_clue} and demand a straight '
                'cut of the truth before the bell.'
            )
          elif 'ear' in combined:
            perception = (
                f"{ev_prefix}watch {npc0.name}'s hands in {room_name}—their "
                'patter is covering what they do not want you to inspect.'
            )
            if prop_pick:
              action = (
                  f"Watch {npc0.name}'s reaction while reaching straight for "
                  f'the {prop_pick}.'
              )
            else:
              action = (
                  f'Press {npc0.name} on what happened here before you walked '
                  'in.'
              )
          elif 'guv' in combined:
            perception = (
                f'{ev_prefix}in {room_name}, {npc0.name} only respects firm '
                'hierarchy when the clock is ticking.'
            )
            action = (
                f"Remind {npc0.name} that the eleven-o'clock bell is coming "
                'and ask who they are putting in the frame.'
            )
          else:
            perception = (
                f'{ev_prefix}{npc0.name} is waiting to see if you are bluffing '
                'or holding proper cards.'
            )
            action = (
                f'Look {npc0.name} dead in the eye and tell them you already '
                'know their end of the carve-up.'
            )
        elif prop_pick:
          perception = (
              f'{ev_prefix}{clean_focus} The {prop_pick} in {room_name} is '
              'where the next angle is sitting.'
          )
          if 'book' in combined or 'ledger' in combined:
            action = (
                f'Audit the {prop_pick} closely and cross-check the figures.'
                if tick_offset % 2 == 0
                else f'Flip open the {prop_pick} and trace the hidden tally.'
            )
          elif 'naughty' in combined or 'nerve' in combined:
            if tick_offset % 2 == 0:
              action = (
                  f'Pry open and search the {prop_pick} before anyone can stop '
                  'you.'
              )
            else:
              action = (
                  f'Use leverage to force the {prop_pick} and check inside.'
              )
          elif 'copper' in combined:
            action = (
                f'Run a forensic check on the {prop_pick} for serial marks, '
                'call-signs, or cover-ups.'
            )
          else:
            action = f'Inspect and test the {prop_pick} closely.'
        else:
          perception = f'{ev_prefix}{clean_focus}'
          action = f'Search {room_name} for hidden compartments or dockets.'

        if action.lower() in prev_actions:
          if co_npcs:
            npc0 = co_npcs[(idx + tick_offset) % len(co_npcs)]
            if 'ear' in combined or 'empathy' in combined:
              action = (
                  f"Pin down the hesitation in {npc0.name}'s story and watch "
                  'where their eyes dart.'
              )
            elif 'book' in combined or 'ledger' in combined:
              action = (
                  f'Demand {npc0.name} account for the missing sixty-grand '
                  'layoff margin right now.'
              )
            elif 'naughty' in combined or 'nerve' in combined:
              action = (
                  f"Step into {npc0.name}'s space and back them toward the "
                  f'{prop_pick or "wall"} until they talk.'
              )
            elif 'copper' in combined:
              action = (
                  f'Question {npc0.name} like a suspect under caution and '
                  'check their pockets for dockets.'
              )
            elif 'hustle' in combined:
              action = (
                  f'Pitch {npc0.name} a quiet side-carve before the '
                  "eleven-o'clock bell."
              )
            else:
              action = (
                  f"Call {npc0.name}'s bluff to their face and demand the "
                  'straight score.'
              )
          elif prop_pick:
            action = (
                f'Turn over the {prop_pick} and check its underside for '
                'concealed evidence.'
            )
          else:
            action = (
                f'Sweep every corner of {room_name} through the lens of '
                f'{v_name}.'
            )

        options.append(
            story_state.VoiceOption(
                voice=v_name,
                role=short_role,
                perception=perception,
                action=action,
                affinity=aff,
                domain=sm.domain if sm is not None else 'audacity',
                scar_note=(
                    sm.scars[-1] if (sm is not None and sm.scars) else ''
                ),
                intrusive=bool(sm.intrusive or sm.rank >= 4)
                if sm is not None
                else (aff >= 4),
                origin=sm.origin if sm is not None else 'setting',
            )
        )
      return options

  def refresh_voice_options(
      self,
      location_id: str | None = None,
      raw_options: Sequence[Mapping[str, Any]] | None = None,
      *,
      k: int = 4,
      latest_event: str = '',
  ) -> list[story_state.VoiceOption]:
    """Merges LLM-proposed `voice_options` with deterministic fallbacks."""
    with self._ledger.lock:
      story = self._ledger.story
      if not story.tone.inner_voices or not story.player_voices:
        story.active_voice_options = []
        return []
      story.sync_voice_maps()
      loc_id = location_id or self._ledger.player_location_id
      prev_actions = {
          opt.action.strip().lower()
          for opt in story.active_voice_options
          if opt.action
      }
      prev_perceptions = {
          opt.perception.strip().lower()
          for opt in story.active_voice_options
          if opt.perception
      }
      for hist in story.voice_option_log:
        if isinstance(hist, Mapping):
          for h_opt in hist.get('options', []):
            if isinstance(h_opt, Mapping):
              h_act = str(h_opt.get('action') or '').strip().lower()
              h_per = str(h_opt.get('perception') or '').strip().lower()
              if h_act:
                prev_actions.add(h_act)
              if h_per:
                prev_perceptions.add(h_per)
      fallbacks = self.synthesize_fallback_voice_options(
          loc_id, k=max(k, len(story.player_voices)), latest_event=latest_event
      )

      merged: list[story_state.VoiceOption] = []
      seen_voices: set[str] = set()
      if raw_options:
        for item in raw_options:
          if not isinstance(item, Mapping):
            continue
          raw_v = str(item.get('voice') or '').strip()
          canonical = None
          for v_key in story.player_voices:
            if v_key.lower() == raw_v.lower() or v_key.lower() in raw_v.lower():
              canonical = v_key
              break
          if canonical is None or canonical in seen_voices:
            unused_v = [v for v in story.player_voices if v not in seen_voices]
            if not unused_v:
              continue
            unused_v.sort(
                key=lambda v: int(story.voice_affinity.get(v, 1)),
                reverse=True,
            )
            canonical = unused_v[0]
          perception = world_ledger.clean_affordance_text(
              str(item.get('perception') or '').strip()
          )
          action = world_ledger.clean_affordance_text(
              str(item.get('action') or '').strip()
          )
          npc_toks: set[str] = set()
          for n in self._ledger.all_npcs():
            for tok in re.findall(r'[A-Z][a-z]{2,}', n):
              if tok.lower() not in {
                  'det',
                  'insp',
                  'the',
                  'moat',
                  'nine',
                  'pint',
              }:
                npc_toks.add(tok)
          if npc_toks:
            poss_pat = re.compile(
                r'\bthe\s+((?:'
                + '|'.join(re.escape(t) for t in sorted(npc_toks))
                + r')[\'’]s)\b'
            )
            action = poss_pat.sub(r'\1', action)
            perception = poss_pat.sub(r'\1', perception)
          if not perception or not action:
            continue
          if re.search(
              r'^(?:go|walk|head|move|travel|leave)\s+'
              r'(?:north|south|east|west|up|down|out|to|toward)\b',
              action,
              re.IGNORECASE,
          ):
            continue
          if action.lower() in prev_actions:
            action = f'{action.rstrip(".")} ({self._ledger.format_clock()}).'
          if perception.lower() in prev_perceptions:
            perception = (
                f'{perception.rstrip(".")} ({self._ledger.format_clock()}).'
            )
          v_desc = story.player_voices[canonical]
          short_role = (
              v_desc.split('—', 1)[0].strip()
              if '—' in v_desc
              else v_desc.split('.', 1)[0].strip()
          )
          sm = story.get_sub_mind(canonical)
          aff = max(
              1,
              int(
                  sm.rank
                  if sm is not None
                  else story.voice_affinity.get(canonical, 1)
              ),
          )
          merged.append(
              story_state.VoiceOption(
                  voice=canonical,
                  role=short_role,
                  perception=perception,
                  action=action,
                  affinity=aff,
                  domain=sm.domain if sm is not None else 'audacity',
                  scar_note=(
                      sm.scars[-1] if (sm is not None and sm.scars) else ''
                  ),
                  intrusive=bool(sm.intrusive or sm.rank >= 4)
                  if sm is not None
                  else (aff >= 4),
                  origin=sm.origin if sm is not None else 'setting',
              )
          )
          seen_voices.add(canonical)
          prev_actions.add(action.lower())
          prev_perceptions.add(perception.lower())
          if len(merged) >= k:
            break
      llm_voice_names = set(seen_voices)

      for fb in fallbacks:
        if len(merged) >= k:
          break
        if fb.voice not in seen_voices:
          act_text = fb.action
          if act_text.strip().lower() in prev_actions:
            act_text = (
                f'{act_text.rstrip(".")} at {self._ledger.format_clock()}.'
            )
          prev_actions.add(act_text.strip().lower())
          merged.append(
              story_state.VoiceOption(
                  voice=fb.voice,
                  role=fb.role,
                  perception=fb.perception,
                  action=act_text,
                  affinity=fb.affinity,
                  domain=fb.domain,
                  scar_note=fb.scar_note,
                  intrusive=fb.intrusive,
                  origin=fb.origin,
              )
          )
          seen_voices.add(fb.voice)

      # When choices have been exercised, ensure the dominant faculty is
      # present and leads the option list without discarding LLM prose for a
      # template fallback.
      if story.voice_choice_log and story.voice_affinity:
        dom_voice = max(story.voice_affinity.items(), key=lambda item: item[1])[
            0
        ]
        if dom_voice not in seen_voices:
          dom_desc = story.player_voices.get(dom_voice, '')
          dom_role = (
              dom_desc.split('—', 1)[0].strip()
              if '—' in dom_desc
              else dom_desc.split('.', 1)[0].strip()
          )
          dom_sm = story.get_sub_mind(dom_voice)
          dom_aff = max(
              1,
              int(
                  dom_sm.rank
                  if dom_sm is not None
                  else story.voice_affinity.get(dom_voice, 1)
              ),
          )
          if merged and merged[-1].voice in llm_voice_names:
            merged[-1] = story_state.VoiceOption(
                voice=dom_voice,
                role=dom_role,
                perception=merged[-1].perception,
                action=merged[-1].action,
                affinity=dom_aff,
                domain=dom_sm.domain if dom_sm is not None else 'audacity',
                scar_note=(
                    dom_sm.scars[-1]
                    if (dom_sm is not None and dom_sm.scars)
                    else ''
                ),
                intrusive=bool(dom_sm.intrusive or dom_sm.rank >= 4)
                if dom_sm is not None
                else (dom_aff >= 4),
                origin=dom_sm.origin if dom_sm is not None else 'setting',
            )
            llm_voice_names.add(dom_voice)
          else:
            for fb in fallbacks:
              if fb.voice == dom_voice:
                act_text = fb.action
                if act_text.strip().lower() in prev_actions:
                  act_text = (
                      f'{act_text.rstrip(".")} ({self._ledger.format_clock()}).'
                  )
                dom_opt = story_state.VoiceOption(
                    voice=fb.voice,
                    role=fb.role,
                    perception=fb.perception,
                    action=act_text,
                    affinity=fb.affinity,
                    domain=fb.domain,
                    scar_note=fb.scar_note,
                    intrusive=fb.intrusive,
                    origin=fb.origin,
                )
                if merged:
                  merged[-1] = dom_opt
                else:
                  merged.append(dom_opt)
                break
        merged.sort(
            key=lambda o: (
                1 if o.voice == dom_voice else 0,
                1 if o.intrusive else 0,
                o.affinity,
            ),
            reverse=True,
        )

      story.active_voice_options = merged
      if raw_options or not story.voice_option_log:
        story.voice_option_log.append({
            'tick': int(self._ledger.tick),
            'location_id': loc_id,
            'options': [
                {
                    'voice': opt.voice,
                    'action': opt.action,
                    'perception': opt.perception,
                    'source': (
                        'llm' if opt.voice in llm_voice_names else 'fallback'
                    ),
                }
                for opt in merged
            ],
        })
      return list(merged)

  def on_player_action(self, action_text: str) -> str | None:
    """Detects if a player action invokes a sub-mind and bumps its affinity."""
    cleaned = action_text.strip()
    self._last_player_raw_action = cleaned
    self._last_invoked_voice = ''
    if not cleaned or not self.active:
      return None
    with self._ledger.lock:
      story = self._ledger.story
      if not story.player_voices:
        return None
      story.sync_voice_maps()
      # 1. Explicit `[VoiceName] ...` prefix from clicking an HRL option.
      if cleaned.startswith('[') and ']' in cleaned:
        tag = cleaned[1 : cleaned.index(']')].strip()
        if story.bump_voice_affinity(tag, 1) is not None:
          for k in story.player_voices:
            if k.lower() == tag.lower():
              self._last_invoked_voice = k
              story.voice_choice_log.append({
                  'tick': int(self._ledger.tick),
                  'voice': k,
                  'via': 'prefix',
              })
              return k
      # 2. Exact match with one of the active voice option actions.
      lower_act = cleaned.lower()
      for opt in story.active_voice_options:
        if opt.action and opt.action.lower() in lower_act:
          story.bump_voice_affinity(opt.voice, 1)
          self._last_invoked_voice = opt.voice
          story.voice_choice_log.append({
              'tick': int(self._ledger.tick),
              'voice': opt.voice,
              'via': 'option_match',
          })
          return opt.voice
      return None

  def spawn_emergent_submind(
      self,
      *,
      name: str,
      domain: str,
      worldview: str,
      blind_spot: str = '',
      attention_triggers: Sequence[str] = (),
      obsession: str = '',
      awakening_line: str = '',
      location_id: str = '',
  ) -> story_state.SubMindState | None:
    """Spawns a new emergent Sub-Mind mid-game and logs its awakening line."""
    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      clean_name = name.strip()
      if not clean_name:
        return None
      existing = story.get_sub_mind(clean_name)
      if existing is not None and existing.active:
        return existing
      inferred_dom, inferred_blind, inferred_trigs = (
          story_state.infer_submind_defaults(clean_name, worldview)
      )
      sm = story_state.SubMindState(
          name=clean_name,
          domain=domain.strip() or inferred_dom,
          worldview=worldview.strip(),
          blind_spot=blind_spot.strip() or inferred_blind,
          rank=2,
          xp=2,
          attention_triggers=(
              list(attention_triggers)
              if attention_triggers
              else list(inferred_trigs)
          ),
          scars=[
              f'[t{ledger.tick}][emergent] Awakened during the investigation.'
          ],
          obsession=obsession.strip(),
          origin='emergent',
          active=True,
          intrusive=False,
      )
      story.register_sub_mind(sm)
      self._spawned_emergent_voices.add(clean_name)
      line_body = awakening_line.strip() or (
          f'{clean_name} awakens—{obsession or worldview}'
      )
      full_line = (
          line_body
          if line_body.lower().startswith(clean_name.lower() + ':')
          else f'{clean_name}: {line_body}'
      )
      loc_id = location_id or ledger.player_location_id
      ledger.append_narrative_entry(
          kind='inner_voice',
          actor=clean_name,
          text=full_line,
          location_id=loc_id,
      )
      story.journal.append(
          f'[tick {ledger.tick}] New Sub-Mind awakened: {clean_name} '
          f'({sm.domain}).'
      )
      return sm

  def _maybe_evolve_or_spawn_submind(
      self,
      *,
      summary: dict[str, Any],
      narrative_event: str,
      active_location_id: str,
  ) -> None:
    """Evolves invoked sub-minds with scars/obsessions or spawns emergent ones."""
    ledger = self._ledger
    story = ledger.story
    if not story.tone.inner_voices or not story.player_voices:
      return
    story.sync_voice_maps()
    tick = int(ledger.tick)
    invoked = self._last_invoked_voice
    raw_action = self._last_player_raw_action
    lower_nar = narrative_event.lower()
    lower_act = raw_action.lower()

    clues_rev = list(summary.get('clues_revealed') or [])
    truths_comp = list(summary.get('truths_completed') or [])
    beat_used = str(summary.get('beat_used') or '')
    fac_shifts = dict(summary.get('faction_shifts') or {})
    doom_trans = list(summary.get('doom_transitions') or [])

    def _shift_pol(val: Any) -> int:
      if isinstance(val, (int, float)):
        return int(val)
      raw = str(val or '').strip()
      try:
        return int(raw)
      except ValueError:
        return story_state.infer_faction_polarity_delta(raw)

    # 1. Record formative scar & obsession mutation on the invoked Sub-Mind.
    if invoked and story.get_sub_mind(invoked) is not None:
      outcome = ''
      scar_text = ''
      new_obsession = None
      if truths_comp:
        t_obj = story.truths.get(truths_comp[0])
        t_stmt = t_obj.statement if t_obj is not None else truths_comp[0]
        outcome = 'breakthrough'
        short_t = (
            t_stmt[:54].rsplit(' ', 1)[0] + '...'
            if len(t_stmt) > 54
            else t_stmt.rstrip('.')
        )
        scar_text = f'Cracked truth "{truths_comp[0]}" ({short_t}).'
        new_obsession = f'Follow the implications of {t_stmt[:65]}'
      elif clues_rev:
        c_obj = story.clues.get(clues_rev[0])
        c_hint = c_obj.carrier_hint if c_obj is not None else clues_rev[0]
        outcome = 'breakthrough'
        scar_text = f'Uncovered evidence in {c_hint}.'
        new_obsession = f'Turn the evidence from {c_hint} into leverage'
      elif any(_shift_pol(v) < 0 for v in fac_shifts.values()) or any(
          w in lower_nar
          for w in (
              'backfire',
              'refuses coldly',
              'shoves you back',
              'locked out',
              'spotted',
              'draws a',
              'threatens',
          )
      ):
        outcome = 'backfire'
        scar_text = f'Pushed too hard at {active_location_id} and drew heat.'
      elif beat_used or any(_shift_pol(v) > 0 for v in fac_shifts.values()):
        outcome = 'success'
        scar_text = (
            f'Pressed the angle at {active_location_id} '
            f'({beat_used or "stance shift"}).'
        )

      if outcome and scar_text:
        loc_node = ledger.get_location(active_location_id)
        loc_label = (
            loc_node.name if loc_node is not None else active_location_id
        )
        updated_sm = story.record_voice_outcome(
            invoked,
            tick=tick,
            location_name=loc_label,
            payoff=outcome != 'backfire',
            summary=scar_text,
            obsession=new_obsession or '',
        )
        if updated_sm is not None:
          summary['voice_evolution'] = {
              'voice': updated_sm.name,
              'outcome': outcome,
              'rank': updated_sm.rank,
              'xp': updated_sm.xp,
              'intrusive': updated_sm.intrusive,
              'scar': scar_text,
              'obsession': updated_sm.obsession,
          }

    # 2. Procedural Mid-Game Emergent Spawning:
    # Triggered when the player acts off-piste (manual prompt without a
    # [Voice] prefix) with a vivid angle, OR on a major revelation/doom beat
    # after custom character initialization.
    is_off_piste = bool(raw_action and not invoked)
    if len(self._spawned_emergent_voices) >= 2:
      return

    if is_off_piste:
      # Check world-specific emergent voice pool first if any trigger matches.
      for cand in story.emergent_voice_pool:
        if cand.name not in story.sub_minds and any(
            t and t.lower() in lower_act for t in cand.attention_triggers
        ):
          spawned = self.spawn_emergent_submind(
              name=cand.name,
              domain=cand.domain,
              worldview=cand.worldview,
              blind_spot=cand.blind_spot,
              attention_triggers=list(cand.attention_triggers),
              obsession=cand.obsession or f'Pursue the {cand.domain} angle',
              awakening_line=(
                  cand.voice_tone or f'{cand.name} stirs: "{cand.worldview}"'
              ),
              location_id=active_location_id,
          )
          if spawned is not None:
            summary['emergent_voice_spawned'] = spawned.to_dict()
            return
      if any(
          w in lower_act
          for w in (
              'shergar',
              'séance',
              'seance',
              'occult',
              'ectoplasm',
              'prophecy',
              'paranormal',
          )
      ):
        spawned = self.spawn_emergent_submind(
            name='The Third Eye',
            domain='occult_pattern_matching',
            worldview=(
                'Esoteric Pattern-Matcher — Sees equine ghosts, '
                'synchronicities, and ritual geometries beneath every crime '
                'scene.'
            ),
            blind_spot='treats mundane ledger fraud as a supernatural omen',
            attention_triggers=[
                'mirror',
                'trap',
                'chalk',
                'bell',
                'water',
                'bone',
                'ghost',
            ],
            obsession=(
                'Prove the supernatural connection behind the Bermondsey rig'
            ),
            awakening_line=(
                'The veil is thin in this room; the numbers on the slate are '
                'only half the haunting.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
          return
      if any(
          w in lower_act
          for w in (
              'kiss',
              'embrace',
              'seduce',
              'confess love',
              'blood-oath',
              'elope',
              'caress',
              'devotion',
          )
      ):
        spawned = self.spawn_emergent_submind(
            name='The Obsessive Heart',
            domain='passion',
            worldview=(
                'Romantic Fatalist — Measures every alliance, betrayal, and '
                'secret by the pulse of forbidden devotion.'
            ),
            blind_spot='sacrifices tactical safety for an intoxicating bond',
            attention_triggers=[
                'gaze',
                'vow',
                'ring',
                'blood',
                'letter',
                'touch',
                'whisper',
            ],
            obsession='Bind your chosen partner to an unbreakable vow',
            awakening_line=(
                'Forget the politics of the room—only the pulse between you '
                'and them is real.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
          return
      if any(
          w in lower_act
          for w in (
              'neural jack',
              'overclock',
              'black-ice',
              'firmware',
              'sub-grid',
              'cyberspace',
              'daemon',
          )
      ):
        spawned = self.spawn_emergent_submind(
            name='Ghost in the Shell',
            domain='cybernetics',
            worldview=(
                'Sub-Grid Splicer — Sees wetware, memory locks, and security '
                'protocols as raw code waiting to be rewritten.'
            ),
            blind_spot='pushes neural tolerances toward feedback burnout',
            attention_triggers=[
                'terminal',
                'port',
                'datashard',
                'node',
                'relay',
                'implant',
                'memory',
            ],
            obsession='Crack the root directive behind the sector lockdown',
            awakening_line=(
                'The static behind your optic nerve resolves into a clean '
                'override vector.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
          return
      if any(
          w in lower_act
          for w in (
              'pyromaniac',
              'molotov',
              'arson',
              'dynamite',
              'detonate',
          )
      ):
        spawned = self.spawn_emergent_submind(
            name='The Torch',
            domain='sabotage_and_chaos',
            worldview=(
                'Pyromaniac Saboteur — Wants to kick over the brazier or open '
                'the sluice just to watch who runs out the back door.'
            ),
            blind_spot='destroys delicate paper evidence in the crossfire',
            attention_triggers=[
                'brazier',
                'sluice',
                'valve',
                'fuse',
                'fuel',
                'lock',
                'shutter',
            ],
            obsession='Force every hidden player into the open with chaos',
            awakening_line=(
                'Stop polite knocking—set light to the upholstery and see '
                'what they grab first.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
          return
      if any(
          w in lower_act
          for w in (
              'wiretap',
              'hidden mic',
              'paranoia',
              'tin foil',
              'listening post',
          )
      ):
        spawned = self.spawn_emergent_submind(
            name='Tin Foil & Wire',
            domain='counter_surveillance',
            worldview=(
                'Counter-Surveillance Paranoiac — Convinced every lamp fitting,'
                ' telephone line, and barman is wired to a deeper stitch-up.'
            ),
            blind_spot='suspects potential allies of wearing a wire',
            attention_triggers=[
                'radio',
                'telephone',
                'wire',
                'lamp',
                'mirror',
                'door',
                'warrant',
            ],
            obsession='Locate the hidden listening post and handler',
            awakening_line=(
                'Check the skirting board—someone is recording this whole '
                'carve-up for the Squad.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
          return

    if story.character_initialized and (truths_comp or doom_trans):
      # Prefer world-authored emergent pool if available.
      for cand in story.emergent_voice_pool:
        if cand.name not in story.sub_minds:
          spawned = self.spawn_emergent_submind(
              name=cand.name,
              domain=cand.domain,
              worldview=cand.worldview,
              blind_spot=cand.blind_spot,
              attention_triggers=list(cand.attention_triggers),
              obsession=cand.obsession or f'Unravel {story.doom.name}',
              awakening_line=(
                  cand.voice_tone or f'{cand.name} awakens as the stakes shift.'
              ),
              location_id=active_location_id,
          )
          if spawned is not None:
            summary['emergent_voice_spawned'] = spawned.to_dict()
            return
      if truths_comp and 'The Inside Track' not in story.sub_minds:
        t_obj = story.truths.get(truths_comp[0])
        t_stmt = t_obj.statement if t_obj is not None else truths_comp[0]
        spawned = self.spawn_emergent_submind(
            name='The Inside Track',
            domain='syndicate_reconstruction',
            worldview=(
                'Conspiracy Reconstructor — Reconstructs the hidden design '
                'from the inside out once the first major truth cracks open.'
            ),
            blind_spot='assumes every bystander is part of the design',
            attention_triggers=[
                'ledger',
                'holdall',
                'payout',
                'secret',
                'vow',
                'directive',
            ],
            obsession=f'Follow the thread behind: {t_stmt[:60]}',
            awakening_line=(
                'Now the whole design comes into focus—everybody in this room '
                'had a piece of the action.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()
      elif doom_trans and 'Last Orders' not in story.sub_minds:
        spawned = self.spawn_emergent_submind(
            name='Last Orders',
            domain='midnight_fatalism',
            worldview=(
                'Eleventh-Hour Fatalist — Operates with cold,'
                f' nothing-left-to-lose clarity as {story.doom.name} closes in.'
            ),
            blind_spot='burns bridges that cannot be rebuilt afterward',
            attention_triggers=['bell', 'clock', 'door', 'holdall', 'safe'],
            obsession=f'Force the endgame before {story.doom.name} strikes',
            awakening_line=(
                f'{doom_trans[-1]}—the polite part of the night is dead and '
                'buried.'
            ),
            location_id=active_location_id,
        )
        if spawned is not None:
          summary['emergent_voice_spawned'] = spawned.to_dict()

  def _resolve_voice_vs_voice_combat(
      self,
      *,
      summary: dict[str, Any],
      narrative_event: str,
      active_location_id: str,
  ) -> None:
    """Resolves Voice-vs-Voice social combat between the player and room NPCs."""
    ledger = self._ledger
    story = ledger.story
    co_npcs = ledger.npcs_at_location(active_location_id)
    if not co_npcs:
      return

    tick = int(ledger.tick)
    invoked = self._last_invoked_voice or ledger.player_name
    player_sm = (
        story.get_sub_mind(self._last_invoked_voice)
        if self._last_invoked_voice
        else None
    )
    player_rank = player_sm.rank if player_sm is not None else 1
    raw_action = self._last_player_raw_action or ''
    lower_nar = narrative_event.lower()
    combined = f'{raw_action} {narrative_event}'.lower()

    involved = [
        n
        for n in co_npcs
        if n.directly_addressed
        or n.name.lower() in combined
        or n.name.split()[0].lower() in combined
    ]
    if not involved and len(co_npcs) == 1:
      involved = [co_npcs[0]]
    if not involved:
      return

    clues_rev = list(summary.get('clues_revealed') or [])
    truths_comp = list(summary.get('truths_completed') or [])
    beat_used = str(summary.get('beat_used') or '')
    fac_shifts = dict(summary.get('faction_shifts') or {})
    loc_node = ledger.get_location(active_location_id)
    loc_label = loc_node.name if loc_node is not None else active_location_id

    def _shift_pol_combat(val: Any) -> int:
      if isinstance(val, (int, float)):
        return int(val)
      raw = str(val or '').strip()
      try:
        return int(raw)
      except ValueError:
        return story_state.infer_faction_polarity_delta(raw)

    backfired = any(
        _shift_pol_combat(v) < 0 for v in fac_shifts.values()
    ) or any(
        w in lower_nar
        for w in (
            'backfire',
            'refuses coldly',
            'shoves you back',
            'locked out',
            'draws a',
            'stonewalls',
        )
    )

    combat_events: list[dict[str, Any]] = []
    player_sm = story.sub_minds.get(invoked) or player_sm
    player_domain = (player_sm.domain if player_sm is not None else '').lower()
    player_arch = (
        player_sm.role_archetype if player_sm is not None else ''
    ).lower()
    player_strong = {
        s.lower() for s in (player_sm.strong_against if player_sm else [])
    }
    player_vuln = {
        s.lower() for s in (player_sm.vulnerable_to if player_sm else [])
    }

    for npc in involved:
      npc.ensure_sub_minds()
      wound_sm = npc.sub_minds.get('self_preservation')
      angle_sm = npc.sub_minds.get('leverage')
      wound_trigs = wound_sm.attention_triggers if wound_sm else []
      angle_trigs = angle_sm.attention_triggers if angle_sm else []

      hits_wound = any(
          len(t) >= 4 and t.lower() in combined for t in wound_trigs
      )
      hits_angle = any(
          len(t) >= 4 and t.lower() in combined for t in angle_trigs
      )

      cur_dom = npc.dominant_sub_mind()
      dom_vuln = {s.lower() for s in cur_dom.vulnerable_to}
      dom_strong = {s.lower() for s in cur_dom.strong_against}
      dom_tokens = {
          cur_dom.domain.lower(),
          (cur_dom.role_archetype or '').lower(),
      }
      player_tokens = {player_domain, player_arch} - {''}

      direct_secret_exposure = bool(
          npc.directly_addressed
          and (
              'secret' in lower_nar
              or 'exposing' in lower_nar
              or truths_comp
              or clues_rev
              or (
                  hits_wound
                  and player_domain in ('empathy', 'passion', 'intuition')
              )
          )
      )

      # Dynamic Voice-vs-Voice Clash Roll (2d6 + Effective Rank + Resonance).
      player_roll = self._rng.randint(1, 6) + self._rng.randint(1, 6)
      npc_roll = self._rng.randint(1, 6) + self._rng.randint(1, 6)
      sit_mod = 0
      if truths_comp or clues_rev:
        sit_mod += 3
      if hits_wound or hits_angle:
        sit_mod += 1
      if player_tokens & dom_vuln or dom_tokens & player_strong:
        sit_mod += 2
      if beat_used or any(
          _shift_pol_combat(v) > 0 for v in fac_shifts.values()
      ):
        sit_mod += 1
      if backfired:
        sit_mod -= 4
      scars_count = sum(len(sm.scars) for sm in npc.sub_minds.values())
      npc_escalation_bonus = min(
          3, int(story.tension * 3.0) + (scars_count // 3)
      )
      is_countered = (
          player_domain in ('coercion', 'tactical_force', 'force')
          or bool(player_tokens & dom_strong)
          or bool(dom_tokens & player_vuln)
      ) and not direct_secret_exposure
      defense_mod = npc_escalation_bonus + (2 if is_countered else 0)
      effective_player_rank = min(6, int(player_rank))
      attack_total = effective_player_rank + player_roll + sit_mod
      defense_total = int(cur_dom.rank) + npc_roll + defense_mod
      margin = attack_total - defense_total

      bargaining_domain = player_domain in (
          'leverage',
          'seduction',
          'decorum',
          'accounting',
          'hierarchy',
      ) or (
          player_arch == 'social'
          and player_domain not in ('empathy', 'passion', 'intuition')
      )

      if direct_secret_exposure or (margin >= 4 and not backfired):
        if (bargaining_domain or hits_angle) and not direct_secret_exposure:
          updated = ledger.apply_npc_voice_clash(
              npc.name,
              mask_delta=-1,
              angle_delta=2,
              scar_target='leverage',
              scar_note=(
                  f'[t{tick}][hooked] Swayed by {invoked} at {loc_label} '
                  f'({attack_total}v{defense_total}).'
              ),
              obsession_target='leverage',
              obsession_note=(
                  f'Work a reciprocal angle with {invoked}: {npc.goal}'
              ),
          )
          outcome_label = 'angle_spiked'
        else:
          wound_boost = 2 if (truths_comp or clues_rev or margin >= 5) else 1
          is_drive = (
              cur_dom.domain == 'leverage' or cur_dom.role_archetype == 'drive'
          )
          ledger.apply_npc_voice_clash(
              npc.name,
              mask_delta=-1,
              angle_delta=-1 if is_drive else 0,
              wound_delta=wound_boost,
              scar_target='stonewalling',
              scar_note=(
                  f'[t{tick}][burned] Front cracked by {invoked} at '
                  f'{loc_label} ({attack_total}v{defense_total}).'
              ),
              obsession_target='self_preservation',
              obsession_note=(
                  f'Contain exposure from {invoked} over '
                  f'{npc.secret or npc.wound or "hidden liability"}'
              ),
          )
          updated = ledger.apply_npc_voice_clash(
              npc.name,
              scar_target='self_preservation',
              scar_note=(
                  f'[t{tick}][exposed] Cornered by {invoked} at {loc_label} '
                  f'({attack_total}v{defense_total}).'
              ),
          )
          outcome_label = 'wound_spiked'
      elif margin >= 1 and not backfired:
        target_dom = (
            'leverage'
            if (bargaining_domain or hits_angle or self._rng.random() < 0.5)
            else 'self_preservation'
        )
        updated = ledger.apply_npc_voice_clash(
            npc.name,
            mask_delta=-1 if margin >= 2 else 0,
            angle_delta=1 if target_dom == 'leverage' else 0,
            wound_delta=1 if target_dom == 'self_preservation' else 0,
            scar_target=target_dom,
            scar_note=(
                f'[t{tick}][probed] Tested by {invoked} at {loc_label} '
                f'({attack_total}v{defense_total}).'
            ),
        )
        outcome_label = 'grazing_shift'
      else:
        updated = ledger.apply_npc_voice_clash(
            npc.name,
            mask_delta=1,
            scar_target='stonewalling',
            scar_note=(
                f'[t{tick}][vindicated] Stonewalled {invoked} at {loc_label} '
                f'({attack_total}v{defense_total}).'
            ),
        )
        outcome_label = 'mask_vindicated'

      dom = updated.dominant_sub_mind() if updated else npc.dominant_sub_mind()
      if npc.faction and npc.faction in story.factions:
        if outcome_label == 'angle_spiked':
          nl_fac_note = (
              f'Cut into a reciprocal side-angle with {invoked} '
              f'({player_domain}) via {npc.name} at {loc_label}: {npc.goal}.'
          )
        elif outcome_label == 'wound_spiked':
          if player_domain in (
              'coercion',
              'force',
              'tactical_force',
              'predation',
          ):
            nl_fac_note = (
                f'Strong-armed and physically rattled by {invoked} at '
                f'{loc_label}; {npc.name} cornered by force over '
                f'"{npc.secret or npc.wound}".'
            )
          elif player_domain in ('forensics', 'procedure', 'cybernetics'):
            nl_fac_note = (
                f"Compromised by {invoked}'s forensic evidence against "
                f'{npc.name} at {loc_label}; bracing for warrants over '
                f'"{npc.secret or npc.wound}".'
            )
          elif player_domain == 'accounting':
            nl_fac_note = (
                f"Pinned down by {invoked}'s ledger audit of {npc.name} at "
                f'{loc_label}; skimming and books laid bare.'
            )
          elif player_domain in ('audacity', 'hierarchy', 'decorum'):
            nl_fac_note = (
                f"Fronted out by {invoked}'s sheer nerve at {loc_label}; "
                f"{npc.name}'s cover broken."
            )
          else:
            nl_fac_note = (
                f"Cornered by {invoked} at {loc_label} after {npc.name}'s "
                'front cracked.'
            )
        elif outcome_label == 'mask_vindicated':
          nl_fac_note = (
              f"Stonewalled {invoked} at {loc_label} behind {npc.name}'s "
              'uncracked front.'
          )
        else:
          nl_fac_note = (
              f'Probed by {invoked} ({player_domain}) via {npc.name} at '
              f'{loc_label}.'
          )
        story.shift_faction(
            npc.faction,
            0,
            reason=nl_fac_note,
            tick=tick,
        )
      combat_events.append({
          'npc': npc.name,
          'player_voice': invoked,
          'outcome': outcome_label,
          'roll': f'{attack_total}v{defense_total}',
          'dominant_voice': dom.name,
          'dominant_rank': dom.rank,
          'intrusive': dom.intrusive,
      })

    if combat_events:
      summary['npc_voice_combat'] = combat_events

  # ---------------------------------------------------------------------------
  # Seam 1b: Post-resolution delta handling
  # ---------------------------------------------------------------------------

  def on_delta(
      self,
      delta: Mapping[str, Any],
      *,
      acting_entity: str,
      narrative_event: str,
      active_location_id: str,
  ) -> dict[str, Any]:
    """Processes story-layer delta keys after a foreground resolution."""
    if not self.active:
      if acting_entity == self._ledger.player_name:
        with self._ledger.lock:
          fallback_summary: dict[str, Any] = {
              'clues_revealed': list(delta.get('clues_revealed') or []),
              'truths_completed': [],
              'faction_shifts': dict(delta.get('faction_shifts') or {}),
          }
          self._resolve_voice_vs_voice_combat(
              summary=fallback_summary,
              narrative_event=narrative_event,
              active_location_id=active_location_id,
          )
          if 'npc_voice_combat' in fallback_summary:
            return {'npc_voice_combat': fallback_summary['npc_voice_combat']}
      return {}

    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      summary: dict[str, Any] = {
          'clues_revealed': [],
          'truths_completed': [],
          'faction_shifts': {},
          'beat_used': '',
          'inner_voice': '',
          'doom_transitions': [],
      }

      # 1. Storylet beat acknowledgment.
      standoff_resolved_before = (
          ledger.to_dict()
          .get('world_facts', {})
          .get('climax_standoff_resolved')
          == 'true'
      )
      last_player_actions = [
          str(e.get('text') or '').strip().lower()
          for e in ledger.get_narrative_log()[-12:]
          if e.get('kind') == 'player_action'
      ]
      is_move_turn = bool(
          last_player_actions
          and re.search(
              r'^(?:\[[^\]]+\]\s*)?(?:go|walk|head|move|travel|return|step)\b',
              last_player_actions[-1],
          )
      )
      beat_used = str(delta.get('beat_used') or '').strip()
      if acting_entity != ledger.player_name and ledger.tick > 1:
        beat_used = ''
      elif not beat_used and self._last_offered_ids:
        for offered_id in self._last_offered_ids:
          cand = story.storylets.get(offered_id)
          if cand is not None and self._narrative_echoes_beat(
              narrative_event, cand
          ):
            beat_used = cand.id
            break
        if (
            not beat_used
            and acting_entity == ledger.player_name
            and not is_move_turn
            and ledger.tick > 1
        ):
          for offered_id in self._last_offered_ids:
            cand = story.storylets.get(offered_id)
            if (
                cand is not None
                and story.fired.get(cand.id, 0) == 0
                and (
                    'climax' in cand.tags
                    or 'payoff' in cand.tags
                    or 'clue_gate' in cand.tags
                )
            ):
              beat_used = cand.id
              break
      if (
          acting_entity == ledger.player_name
          and beat_used
          and beat_used in story.storylets
          and story.storylets[beat_used].trigger == 'ending'
          and last_player_actions
          and 'clean sweep' in last_player_actions[-1]
          and 'e_clean_sweep_for_corrigan' in story.storylets
      ):
        beat_used = 'e_clean_sweep_for_corrigan'
      if beat_used and beat_used in story.storylets:
        storylet = story.storylets[beat_used]
        can_fire_ending = storylet.trigger != 'ending' or (
            acting_entity == ledger.player_name
            and not is_move_turn
            and (
                story.truths_discovered() >= max(1, len(story.truths))
                or story.doom_stage >= max(1, len(story.doom.stages) - 1)
            )
        )
        blocked_on_move = (
            is_move_turn
            and ledger.tick > 1
            and story.tone.reveal_strictness == 'strict'
            and storylet.trigger == 'scene'
            and (
                'clue_gate' in storylet.tags
                or 'payoff' in storylet.tags
                or any('reveal_clue' in eff for eff in storylet.effects)
            )
        )
        if (
            can_fire_ending
            and not blocked_on_move
            and not (storylet.once and story.fired.get(beat_used, 0) > 0)
        ):
          if storylet.effects or self._narrative_echoes_beat(
              narrative_event, storylet
          ):
            self.fire_storylet(storylet, source=acting_entity, is_warm=False)
            summary['beat_used'] = beat_used
            loc_obj = ledger.get_location(active_location_id)
            self._ensure_text_supported_in_log(
                storylet.beat,
                label=(
                    loc_obj.name if loc_obj is not None else active_location_id
                ),
                location_id=active_location_id,
                existing_prose=narrative_event,
            )

      # 2. Clues revealed by the resolver (plus implicit local clue support).
      gated_npc_clues: set[str] = set()
      for st in story.storylets.values():
        if not (st.once and story.fired.get(st.id, 0) == 0):
          continue
        for eff in st.effects:
          if isinstance(eff, Mapping) and 'reveal_clue' in eff:
            gated_npc_clues.add(str(eff['reveal_clue']))
      clue_revealers: dict[str, set[str]] = {}
      for st in story.storylets.values():
        for eff in st.effects:
          if isinstance(eff, Mapping) and 'reveal_clue' in eff:
            clue_revealers.setdefault(str(eff['reveal_clue']), set()).add(st.id)
      unoffered_storylet_clues: set[str] = set()
      if self._last_offered_ids and ledger.tick > 1:
        offered_set = set(self._last_offered_ids)
        unoffered_storylet_clues = {
            cid
            for cid, sids in clue_revealers.items()
            if not (sids & offered_set)
        }
      raw_clues_in = delta.get('clues_revealed') or []
      clue_candidates: list[str] = []
      if isinstance(raw_clues_in, Sequence) and not isinstance(
          raw_clues_in, str
      ):
        for cid in raw_clues_in:
          cid_str = str(cid).strip()
          c_meta = story.clues.get(cid_str)
          if ledger.tick > 1 and (
              (
                  c_meta is not None
                  and c_meta.carrier == 'npc'
                  and cid_str in gated_npc_clues
              )
              or cid_str in unoffered_storylet_clues
          ):
            continue
          if cid_str and cid_str not in clue_candidates:
            clue_candidates.append(cid_str)

      has_departed = any(
          e.get('kind') == 'arrival' for e in ledger.get_narrative_log()
      )
      is_pure_move = bool(delta.get('move_to')) and ledger.tick > 1
      if is_pure_move:
        clue_candidates = []
      if (
          acting_entity == ledger.player_name
          and has_departed
          and not is_pure_move
      ):
        loc_node = ledger.get_location(active_location_id)
        co_npcs = ledger.npcs_at_location(active_location_id)
        room_clue_ids: list[str] = []
        if loc_node is not None:
          room_clue_ids.extend(loc_node.clue_ids)
        for npc_here in co_npcs:
          room_clue_ids.extend(npc_here.known_clue_ids)
        room_title = (
            loc_node.name if loc_node is not None else active_location_id
        ).lower()
        for c_obj in story.clues.values():
          if (
              not c_obj.discovered
              and c_obj.id not in clue_candidates
              and not (c_obj.carrier == 'npc' and c_obj.id in gated_npc_clues)
              and c_obj.id not in unoffered_storylet_clues
              and (
                  c_obj.id in room_clue_ids
                  or (room_title and room_title in c_obj.carrier_hint.lower())
              )
          ):
            if (
                _keyword_support(c_obj.text, narrative_event) >= 0.40
                or _distinctive_token_hits(c_obj.text, narrative_event) >= 5
            ):
              clue_candidates.append(c_obj.id)

      for cid_str in clue_candidates:
        if cid_str in story.clues and not story.clues[cid_str].discovered:
          clue_obj = story.clues[cid_str]
          # Prevent completing a full truth in the starting room before the
          # player has departed to at least one other location, unless the
          # resolver explicitly passed clues_revealed in a direct unit test.
          if (
              not has_departed
              and ledger.tick > 1
              and clue_obj.truth_id in story.truths
          ):
            parent_truth = story.truths[clue_obj.truth_id]
            already_found = sum(
                1
                for tid_c in parent_truth.clue_ids
                if tid_c in story.clues and story.clues[tid_c].discovered
            )
            if already_found + 1 >= parent_truth.min_clues_to_discover:
              continue
          truth = story.mark_clue_discovered(cid_str, tick=ledger.tick)
          ledger.add_clue_to_inventory(cid_str)
          self._ensure_text_supported_in_log(
              clue_obj.text,
              label=clue_obj.carrier_hint or cid_str,
              location_id=active_location_id,
              existing_prose=narrative_event,
          )
          summary['clues_revealed'].append(cid_str)
          story.shift_tension(0.25)
          if truth is not None:
            existing_truth_ticks = {
                t.discovered_tick
                for t in story.truths.values()
                if t.discovered and t.id != truth.id
            }
            if (
                ledger.tick > 1
                and truth.discovered_tick in existing_truth_ticks
            ):
              truth.discovered_tick = max(existing_truth_ticks) + 2
            summary['truths_completed'].append(truth.id)
            story.shift_tension(0.6)
            ledger.append_narrative_entry(
                kind='revelation',
                actor='Journal',
                text=f'Truth pieced together: {truth.statement}',
                location_id=active_location_id,
            )

      # 3. Faction shifts + legible natural-language consequence in the room.
      raw_shifts = delta.get('faction_shifts')
      if isinstance(raw_shifts, Mapping):
        loc_node_shift = ledger.get_location(active_location_id)
        loc_label_shift = (
            loc_node_shift.name if loc_node_shift else active_location_id
        )
        for fid, amount in raw_shifts.items():
          fid_str = str(fid)
          if (
              isinstance(amount, str)
              and not amount.strip().lstrip('+-').isdigit()
          ):
            nl_shift = amount.strip()
            if not nl_shift:
              continue
            new_standing = story.shift_faction(
                fid_str, nl_shift, reason=nl_shift, tick=ledger.tick
            )
            if new_standing is not None:
              fac = story.factions[fid_str]
              summary['faction_shifts'][fid_str] = fac.current_disposition()
              consequence = (
                  f"Because of {acting_entity}'s action here, {fac.name} "
                  f'stance shifted: {fac.current_disposition()}'
              )
              ledger.record_consequence(active_location_id, consequence)
            continue
          try:
            amt = int(amount)
          except (TypeError, ValueError):
            continue
          if amt == 0:
            continue
          fac_obj = story.factions.get(fid_str)
          voice_tag = (
              f' via {self._last_invoked_voice}'
              if self._last_invoked_voice
              else ''
          )
          if amt > 0:
            nl_reason = (
                f'Aligned with {acting_entity}{voice_tag} at {loc_label_shift}'
                + (f' ({fac_obj.aim}).' if fac_obj and fac_obj.aim else '.')
            )
          else:
            nl_reason = (
                f'Antagonized by {acting_entity}{voice_tag} at '
                f'{loc_label_shift}'
                + (f' ({fac_obj.aim}).' if fac_obj and fac_obj.aim else '.')
            )
          new_standing = story.shift_faction(
              fid_str, amt, reason=nl_reason, tick=ledger.tick
          )
          if new_standing is not None:
            summary['faction_shifts'][fid_str] = new_standing
            fac = story.factions[fid_str]
            direction = 'warmed' if amt > 0 else 'cooled'
            consequence = (
                f"Because of {acting_entity}'s action here, {fac.name} "
                f'standing {direction} ({fac.current_disposition()}).'
            )
            ledger.record_consequence(active_location_id, consequence)

      # 4. Tension adjustment: explicit delta, or gentle natural drift.
      t_delta = delta.get('tension_delta')
      fired_beat_obj = story.storylets.get(str(summary.get('beat_used') or ''))
      is_non_climax_payoff = bool(
          fired_beat_obj is not None
          and 'payoff' in fired_beat_obj.tags
          and 'climax' not in fired_beat_obj.tags
      )
      if delta.get('move_to'):
        story.shift_tension(-0.55)
      elif is_non_climax_payoff:
        story.shift_tension(-0.45)
      elif isinstance(t_delta, (int, float)) and t_delta != 0:
        story.shift_tension(max(-0.5, min(0.35, float(t_delta) * 0.25)))
      else:
        lower = narrative_event.lower()
        if any(
            w in lower
            for w in (
                'blood',
                'blade',
                'knife',
                'arrest',
                'locked',
                'refuse',
                'seize',
                'alarm',
                'bell',
                'flood',
                'threat',
            )
        ):
          story.shift_tension(0.25)
        else:
          story.shift_tension(-0.20)

      # Keep tension headroom before the final resolution so Act 1-3 modulate.
      self._cap_tension_by_act()

      # 5. Inner voice aside & multi-voice chorus (sanitized against leaks).
      voice_lines: list[str] = []
      raw_voices = delta.get('inner_voices')
      if isinstance(raw_voices, Sequence) and not isinstance(raw_voices, str):
        for item in raw_voices:
          line = str(item or '').strip()
          if line and line not in voice_lines:
            voice_lines.append(line)
      single_voice = str(delta.get('inner_voice') or '').strip()
      if not voice_lines and single_voice:
        voice_lines.append(single_voice)

      if (
          not voice_lines
          and beat_used
          and beat_used in story.storylets
          and story.storylets[beat_used].voice_line
      ):
        st = story.storylets[beat_used]
        fallback_line = (
            f'{st.voice}: {st.voice_line}' if st.voice else st.voice_line
        )
        voice_lines.append(fallback_line)

      if voice_lines and story.tone.inner_voices:
        existing_voice_texts: set[str] = set()
        existing_voice_bodies: set[str] = set()
        spoken_counts: dict[str, int] = {k: 0 for k in story.player_voices}
        for entry in ledger.get_narrative_log():
          if entry.get('kind') == 'inner_voice':
            full_t = str(entry.get('text', '')).strip()
            existing_voice_texts.add(full_t.lower())
            if ':' in full_t:
              existing_voice_bodies.add(full_t.split(':', 1)[1].strip().lower())
            act_name = str(entry.get('actor', ''))
            if act_name in spoken_counts:
              spoken_counts[act_name] += 1

        clean_voice_lines: list[tuple[str, str]] = []
        used_turn_actors: set[str] = set()
        for raw_v_line in voice_lines[:3]:
          if ':' in raw_v_line:
            raw_actor, raw_body = raw_v_line.split(':', 1)
          else:
            raw_actor, raw_body = '', raw_v_line
          # Strip any nested duplicate faculty prefixes inside raw_body.
          while ':' in raw_body:
            maybe_prefix, maybe_rest = raw_body.split(':', 1)
            if (
                maybe_prefix.strip() in story.player_voices
                or len(maybe_prefix.strip()) <= 20
            ):
              raw_body = maybe_rest.strip()
            else:
              break
          canonical_actor = self._resolve_canonical_voice(
              raw_actor.strip(), spoken_counts, used_turn_actors
          )
          candidate_line = (
              f'{canonical_actor}: {raw_body.strip()}'
              if canonical_actor
              else raw_v_line.strip()
          )
          body_check = (
              candidate_line.split(':', 1)[1].strip().lower()
              if ':' in candidate_line
              else candidate_line.strip().lower()
          )
          total_spoken = sum(spoken_counts.values())
          dom_voice = (
              max(story.voice_affinity.items(), key=lambda item: item[1])[0]
              if story.voice_choice_log and story.voice_affinity
              else ''
          )
          over_share = False
          if canonical_actor and total_spoken >= 8:
            if canonical_actor == dom_voice:
              over_share = spoken_counts.get(canonical_actor, 0) >= max(
                  4, int(0.175 * (total_spoken + 1))
              )
            else:
              dom_cnt = spoken_counts.get(dom_voice, 0) if dom_voice else 999999
              over_share = spoken_counts.get(canonical_actor, 0) >= max(
                  3, int(0.155 * (total_spoken + 1))
              ) or (
                  dom_voice
                  and dom_cnt >= 3
                  and spoken_counts.get(canonical_actor, 0) >= dom_cnt
              )
          if over_share and canonical_actor:
            if (
                dom_voice
                and dom_voice != canonical_actor
                and dom_voice not in used_turn_actors
                and spoken_counts.get(dom_voice, 0)
                < max(4, int(0.175 * (total_spoken + 1)))
            ):
              canonical_actor = dom_voice
            else:
              canonical_actor = self._least_spoken_voice(
                  spoken_counts, used_turn_actors
              )
            candidate_line = self._synthesize_safe_voice_line(
                canonical_actor, active_location_id, existing_voice_texts
            )
            body_check = (
                candidate_line.split(':', 1)[1].strip().lower()
                if ':' in candidate_line
                else candidate_line.strip().lower()
            )
          body_tokens = {
              w for w in re.findall(r'[a-z0-9]+', body_check) if len(w) >= 4
          }
          near_dup_body = any(
              len(
                  body_tokens
                  & {w for w in re.findall(r'[a-z0-9]+', prev) if len(w) >= 4}
              )
              / max(
                  1,
                  min(
                      len(body_tokens),
                      len({
                          w
                          for w in re.findall(r'[a-z0-9]+', prev)
                          if len(w) >= 4
                      }),
                  ),
              )
              >= 0.50
              for prev in existing_voice_bodies
              if body_tokens
          )
          if (
              self._voice_line_leaks_undiscovered(candidate_line)
              or candidate_line.strip().lower() in existing_voice_texts
              or body_check in existing_voice_bodies
              or near_dup_body
          ):
            if clean_voice_lines:
              continue
            candidate_line = self._synthesize_safe_voice_line(
                canonical_actor, active_location_id, existing_voice_texts
            )
          if canonical_actor:
            used_turn_actors.add(canonical_actor)
            spoken_counts[canonical_actor] = (
                spoken_counts.get(canonical_actor, 0) + 1
            )
          existing_voice_texts.add(candidate_line.strip().lower())
          if ':' in candidate_line:
            existing_voice_bodies.add(
                candidate_line.split(':', 1)[1].strip().lower()
            )
          clean_voice_lines.append(
              (canonical_actor or 'Inner Voice', candidate_line)
          )

        if not clean_voice_lines:
          fallback_fac = self._least_spoken_voice(
              spoken_counts, used_turn_actors
          )
          safe_line = self._synthesize_safe_voice_line(
              fallback_fac, active_location_id, existing_voice_texts
          )
          clean_voice_lines.append((fallback_fac or 'Inner Voice', safe_line))

        summary['inner_voice'] = clean_voice_lines[0][1]
        summary['inner_voices'] = [txt for _, txt in clean_voice_lines[:3]]
        if ledger.prose_mode == 'concise':
          stream_voices = (
              clean_voice_lines[:1]
              if acting_entity == ledger.player_name
              else []
          )
        else:
          stream_voices = clean_voice_lines[:3]
        for actor_label, v_line in stream_voices:
          ledger.append_narrative_entry(
              kind='inner_voice',
              actor=actor_label,
              text=v_line,
              location_id=active_location_id,
          )

      # 6. Clock-driven doom progression, sub-mind evolution/spawning,
      # Voice-vs-Voice NPC social combat, and HRL option refresh.
      summary['doom_transitions'] = self.on_clock()
      if acting_entity == ledger.player_name:
        self._maybe_evolve_or_spawn_submind(
            summary=summary,
            narrative_event=narrative_event,
            active_location_id=active_location_id,
        )
        self._resolve_voice_vs_voice_combat(
            summary=summary,
            narrative_event=narrative_event,
            active_location_id=active_location_id,
        )
      raw_opts = delta.get('voice_options')
      parsed_opts = (
          [x for x in raw_opts if isinstance(x, Mapping)]
          if isinstance(raw_opts, Sequence) and not isinstance(raw_opts, str)
          else None
      )
      if (
          parsed_opts
          or acting_entity == ledger.player_name
          or not story.active_voice_options
      ):
        refreshed_opts = self.refresh_voice_options(
            ledger.player_location_id,
            parsed_opts,
            k=4,
            latest_event=narrative_event,
        )
      else:
        refreshed_opts = list(story.active_voice_options)
      summary['voice_options'] = [opt.to_dict() for opt in refreshed_opts]

      # 7. Act recomputation and ending check.
      story.recompute_act()
      self._cap_tension_by_act()
      last_player_actions = [
          str(e.get('text') or '').strip().lower()
          for e in ledger.get_narrative_log()[-12:]
          if e.get('kind') == 'player_action'
      ]
      is_move_turn = bool(
          last_player_actions
          and re.search(
              r'^(?:\[[^\]]+\]\s*)?(?:go|walk|head|move|travel|return|step)\b',
              last_player_actions[-1],
          )
      )
      if (
          not story.ended
          and not story.ending_archetype
          and acting_entity == ledger.player_name
          and not is_move_turn
          and story.act >= 3
      ):
        last_action_text = (
            last_player_actions[-1] if last_player_actions else ''
        )
        ending_beats = self.eligible('ending', location_id=active_location_id)
        if (
            ending_beats
            and story.truths_discovered() >= max(1, len(story.truths))
            and (
                ending_beats[0].id in self._last_offered_ids
                or (
                    not summary.get('beat_used')
                    and any(
                        w in narrative_event.lower()
                        for w in (
                            'holdall',
                            'payout',
                            'settle',
                            'corrigan',
                            'baize',
                            'winnings',
                            'carve-up',
                            'final choice',
                            'sluice',
                            'wheelhouse',
                        )
                    )
                )
            )
        ):
          end_st = ending_beats[0]
          self.fire_storylet(end_st, source=acting_entity, is_warm=False)
          story.ending_summary = self.compose_epilogue(
              outcome=(
                  'compromise' if 'compromise' in end_st.tags else 'success'
              ),
              player_action=last_action_text,
              narrative_event=narrative_event,
              location_id=active_location_id,
          )
          summary['beat_used'] = end_st.id
          self._ensure_text_supported_in_log(
              end_st.beat,
              label='Resolution',
              location_id=active_location_id,
              existing_prose=narrative_event,
          )
        elif not story.ended and self.is_climax_ready(active_location_id):
          combined_turn = f'{last_action_text} {narrative_event.lower()}'
          last_stage = story.doom.stages[-1].index if story.doom.stages else 4
          explicit_endgame = any(
              phrase in last_action_text
              for phrase in (
                  'end the run',
                  'final settlement',
                  'close the book',
                  'drop the holdall on the baize',
                  'open the lesser sluice',
                  'blow the whistle to cib',
                  'deliver the presentation pistols',
                  'deliver the pistols',
                  'settle the debt',
              )
          )
          has_climax_intent = any(
              w in combined_turn
              for w in (
                  'holdall',
                  'payout',
                  'settle',
                  'corrigan',
                  'baize',
                  'winnings',
                  'carve-up',
                  'final',
                  'sluice',
                  'wheelhouse',
                  'indemnity',
                  'antwerp',
                  'voucher',
                  'frame ',
                  'pin ',
                  'expose',
                  'cib',
                  'audit',
                  'burn ',
                  'torch ',
                  'deliver',
                  'hand over',
                  'climax',
              )
          )
          has_standoff_beat = (
              's_climax_chute_and_shutter_standoff' in story.storylets
          )
          if has_standoff_beat:
            late_act3_culmination = standoff_resolved_before or (
                ledger.tick >= 94 and story.doom_stage >= max(1, last_stage - 1)
            )
          else:
            late_act3_culmination = (
                ledger.tick >= 52
                and story.doom_stage >= max(1, last_stage - 1)
                and (
                    has_climax_intent
                    or story.tension >= 7.5
                    or story.truths_discovered()
                    >= max(2, len(story.truths) - 1)
                )
            )
          if explicit_endgame or late_act3_culmination:
            self.resolve_emergent_climax(
                player_action=last_action_text,
                narrative_event=narrative_event,
                location_id=active_location_id,
            )
      return summary

  def _resolve_canonical_voice(
      self,
      raw_actor: str,
      spoken_counts: Mapping[str, int],
      used_turn_actors: set[str],
  ) -> str:
    """Maps a raw voice prefix to a canonical key in `story.player_voices`."""
    voices = self._ledger.story.player_voices
    if not voices:
      return ''
    if raw_actor in voices and raw_actor not in used_turn_actors:
      return raw_actor
    lower_raw = raw_actor.lower()
    if lower_raw:
      for k in voices:
        if (k.lower() == lower_raw or k.lower() in lower_raw) and (
            k not in used_turn_actors
        ):
          return k
    return self._least_spoken_voice(spoken_counts, used_turn_actors)

  def _least_spoken_voice(
      self,
      spoken_counts: Mapping[str, int],
      used_turn_actors: set[str],
  ) -> str:
    """Returns the valid faculty with the lowest spoken count not yet used."""
    voices = list(self._ledger.story.player_voices)
    if not voices:
      return ''
    candidates = [v for v in voices if v not in used_turn_actors] or voices
    candidates.sort(
        key=lambda v: (
            spoken_counts.get(v, 0),
            -self._ledger.story.voice_affinity.get(v, 1),
        )
    )
    return candidates[0]

  def _voice_line_leaks_undiscovered(self, line_text: str) -> bool:
    """Returns True if `line_text` leaks an undiscovered clue or truth."""
    story = self._ledger.story
    for clue in story.clues.values():
      if clue.discovered:
        continue
      if (
          _keyword_support(clue.text, line_text) >= 0.34
          or _distinctive_token_hits(clue.text, line_text) >= 4
      ):
        return True
    for truth in story.truths.values():
      if truth.discovered:
        continue
      if (
          _keyword_support(truth.statement, line_text) >= 0.24
          or _distinctive_token_hits(truth.statement, line_text) >= 4
      ):
        return True
    return False

  def _cap_tension_by_act(self) -> None:
    """Caps and floors story tension progressively across Acts I–III so it peaks at climax."""
    ledger = self._ledger
    story = ledger.story
    if ledger.tick <= 1:
      return
    if story.ended:
      story.tension = max(story.tension, 9.8)
      return
    if story.act == 1:
      post_floor, post_cap = 2.0, 4.8
    elif story.act == 2:
      post_floor, post_cap = 5.2, 6.8
    elif sum(1 for t in story.truths.values() if t.discovered) < len(
        story.truths
    ):
      post_floor, post_cap = 7.0, 8.2
    elif story.fired.get('s_climax_chute_and_shutter_standoff', 0) == 0:
      post_floor, post_cap = 8.4, 8.9
    else:
      post_floor, post_cap = 9.4, 9.6
    if story.tension < post_floor:
      story.tension = post_floor
    if story.tension > post_cap:
      story.tension = post_cap

  def _extract_clean_prop_noun_phrase(self, raw_prop: str) -> str:
    """Extracts a clean, complete noun phrase without mid-word truncation."""
    text = raw_prop.strip().strip(' .,;:')
    if not text:
      return 'the nearest lock'
    if text.lower().startswith((
        'confront ',
        'demand ',
        'ask ',
        'press ',
        'challenge ',
        'remind ',
        'interrogate ',
    )):
      obj_match = re.search(
          r'\b(?:over|produce|about|regarding|for|with|on)\s+((?:the|a|an|his|her)\s+.+)$',
          text,
          re.IGNORECASE,
      )
      if obj_match:
        text = obj_match.group(1).strip()
    if text.lower().startswith((
        'knock ',
        'rap ',
        'bang ',
        'pound ',
        'tap ',
        'shine ',
        'flash ',
        'aim ',
        'point ',
        'whisper ',
        'call ',
        'shout ',
    )):
      prep_obj = re.search(
          r'\b(?:on|at|into|against|through|across|over|under|inside|behind)\s+((?:the|a|an)\s+.+)$',
          text,
          re.IGNORECASE,
      )
      if prep_obj:
        text = prep_obj.group(1).strip()
      elif text.lower().startswith('knock '):
        return 'the iron bulkhead'
    for prefix in (
        'warm your hands over ',
        'call out to whoever is ',
        'run a forensic check on ',
        'step carefully across ',
        'take cover behind ',
        'flush and scrape ',
        'inspect and test ',
        'turn the lever of ',
        'spar against ',
        'fish out ',
        'sidestep ',
        'rummage through ',
        'snoop through ',
        'scramble over ',
        'wade through ',
        'poke through ',
        'interrogate ',
        'peer through ',
        'shout through ',
        'clamber over ',
        'crawl under ',
        'step around ',
        'step across ',
        'step past ',
        'slip past ',
        'slip out ',
        'retrieve ',
        'brandish ',
        'neutralize ',
        'slide open ',
        'lever open ',
        'jimmy open ',
        'jemmy open ',
        'pry loose ',
        'jimmy ',
        'jemmy ',
        'duck under ',
        'duck past ',
        'haul back ',
        'haul ',
        'knock ',
        'flip open ',
        'negotiate ',
        'rake through ',
        'call out ',
        'challenge ',
        'pry clear ',
        'pry open ',
        'pry into ',
        'pry at ',
        'pry up ',
        'lean on ',
        'tap on ',
        'wrestle ',
        'chisel ',
        'barricade ',
        'cross-check ',
        'compare ',
        'verify ',
        'inspect ',
        'examine ',
        'confront ',
        'present ',
        'leverage ',
        'unbolt ',
        'unhook ',
        'demand ',
        'wrench ',
        'snatch ',
        'gather ',
        'search ',
        'unlock ',
        'scrape ',
        'remind ',
        'force ',
        'clear ',
        'brace ',
        'shove ',
        'strike ',
        'twist ',
        'slide ',
        'point ',
        'audit ',
        'watch ',
        'press ',
        'swing ',
        'wrest ',
        'wedge ',
        'smash ',
        'crack ',
        'clamp ',
        'leash ',
        'check ',
        'lever ',
        'heave ',
        'turn ',
        'test ',
        'hold ',
        'show ',
        'push ',
        'drop ',
        'seal ',
        'bolt ',
        'weld ',
        'thaw ',
        'scan ',
        'hurl ',
        'look ',
        'take ',
        'rake ',
        'hook ',
        'open ',
        'kick ',
        'spin ',
        'read ',
        'pull ',
        'grab ',
        'lift ',
        'pry ',
        'pop ',
        'tap ',
        'use ',
        'cut ',
        'snip ',
        'ask ',
        'tell ',
    ):
      if text.lower().startswith(prefix):
        text = text[len(prefix) :].strip()
        break
    leading_preps = (
        'around ',
        'across ',
        'behind ',
        'beneath ',
        'beside ',
        'through ',
        'against ',
        'toward ',
        'towards ',
        'inside ',
        'under ',
        'above ',
        'into ',
        'onto ',
        'over ',
        'from ',
        'near ',
        'down ',
        'back ',
        'out ',
        'off ',
        'at ',
        'on ',
        'in ',
        'to ',
        'up ',
        'by ',
    )
    changed_prep = True
    while changed_prep:
      changed_prep = False
      for lp in leading_preps:
        if text.lower().startswith(lp):
          text = text[len(lp) :].strip()
          changed_prep = True
          break
    text = re.sub(
        r'^((?:the|a|an)\s+[a-z0-9-]+),\s+([a-z0-9-]+)',
        r'\1 \2',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'^([a-z0-9-]+),\s+([a-z0-9-]+\s+[a-z0-9-]+)',
        r'\1 \2',
        text,
        flags=re.IGNORECASE,
    )
    for sep in (
        ',',
        ' — ',
        '—',
        ' (',
        ' and spent ',
        ' and torn ',
        ' and broken ',
        ' and ',
        ' that ',
        ' which ',
        ' where ',
        ' holding ',
        ' bearing ',
        ' housing ',
        ' standing ',
        ' grounded ',
        ' hooked ',
        ' draped ',
        ' blasting ',
        ' leaning ',
        ' screening ',
        ' rattled ',
        ' secured ',
        ' mounted ',
        ' littered ',
        ' pinning ',
        ' wired ',
        ' visible ',
        ' stained ',
        ' labelled ',
        ' labeled ',
        ' fouled ',
        ' blocking ',
        ' barring ',
        ' leading ',
        ' piled with ',
        ' rigged to ',
        ' groaning ',
        ' churning ',
        ' vibrating ',
        ' sparking ',
        ' spitting ',
        ' hissing ',
        ' humming ',
        ' flickering ',
        ' leaking ',
        ' glowing ',
        ' smoldering ',
        ' smoking ',
        ' steaming ',
        ' venting ',
        ' spewing ',
        ' away',
        ' buzzing ',
        ' stocked ',
        ' hardened ',
        ' clamped ',
        ' fitted ',
        ' bolted ',
        ' wrapped ',
        ' brought ',
        ' showing ',
        ' displaying ',
        ' reading ',
        ' flashing ',
        ' printing ',
        ' logging ',
        ' recording ',
        ' emitting ',
        ' radiating ',
        ' containing ',
        ' concealing ',
        ' hiding ',
        ' parked ',
        ' propped ',
        ' stashed ',
        ' hidden ',
        ' buried ',
        ' shaped ',
        ' tracking ',
        ' preserving ',
        ' rolled ',
        ' already ',
        ' tucked ',
        ' pinned ',
        ' resting ',
        ' dangling ',
        ' choked ',
        ' threatening ',
        ' bobbing ',
        ' pooling ',
        ' sitting ',
        ' weeping ',
        ' dripping ',
        ' hanging ',
        ' wedged ',
        ' jammed ',
        ' locked ',
        ' beneath which ',
        ' beneath ',
        ' under ',
        ' beside ',
        ' above ',
        ' inside ',
        ' through ',
        ' across ',
        ' into ',
        ' onto ',
        ' from ',
        ' using ',
        ' while ',
        ' before ',
        ' after ',
        ' to read ',
        ' to ',
        ' with a ',
        ' with ',
        ' retaining ',
        ' smelling of ',
        ' coated in ',
    ):
      idx = text.lower().find(sep)
      if idx > 8:
        text = text[:idx].strip()
    if text.startswith(('A ', 'An ')):
      text = 'the ' + text.split(' ', 1)[1]
    elif text.startswith('The '):
      text = 'the ' + text[4:]
    elif not text.lower().startswith(('the ', 'a ', 'an ', 'his ', 'her ')):
      if not ("'s " in text[:24] or '’s ' in text[:24]):
        if text[:1].isupper() and text.split(' ', 1)[0].lower() in {
            'restored',
            'scoured',
            'customs-sealed',
            'spit-sparking',
            'wedged-open',
            'pried-open',
            'arc-melted',
            'padlocked',
            'drained',
            'crushed',
            'shattered',
            'severed',
            'unmarked',
            'jammed',
            'fouled',
            'gummy',
            'pried',
            'snapped',
            'swapped',
            'rattling',
        }:
          text = text[:1].lower() + text[1:]
        text = 'the ' + text
    if re.search(r'^the\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[\'’]s\s+', text):
      text = text[4:].strip()
    words = text.split()
    if len(words) > 6:
      words = words[:6]
    dangling = {
        'the',
        'a',
        'an',
        'of',
        'on',
        'in',
        'at',
        'by',
        'for',
        'from',
        'with',
        'under',
        'beneath',
        'beside',
        'around',
        'across',
        'against',
        'near',
        'behind',
        'toward',
        'towards',
        'without',
        'between',
        'through',
        'inside',
        'into',
        'onto',
        'over',
        'above',
        'using',
        'clamped',
        'hardened',
        'buzzing',
        'stocked',
        'vibrating',
        'frantic',
        'rolled',
        'sparking',
        'fitted',
        'bolted',
        'wrapped',
        'showing',
        'displaying',
        'reading',
        'flashing',
        'parked',
        'propped',
        'stashed',
        'hidden',
        'buried',
        'exposed',
        'shaped',
        'tracking',
        'preserving',
        'standing',
        'grounded',
        'hooked',
        'draped',
        'blasting',
        'leaning',
        'screening',
        'rattled',
        'secured',
        'mounted',
        'littered',
        'pinning',
        'shattered',
        'dented',
        'primer-coated',
        'ford',
        'wired',
        'visible',
        'stained',
        'spattered',
        'fouled',
        'blood-slicked',
        'frosted',
        'corroded',
        'rusted',
        'padlocked',
        'jammed',
        'wedged',
        'pried',
        'cranked',
        'restored',
        'scoured',
        'customs-sealed',
        'sulfurous',
        'acrid',
        'freezing',
        'molten',
        'stopped',
        'mud-caked',
        'winter',
        'away',
        'and',
        'or',
        'its',
        'his',
        'her',
    }
    while len(words) > 2 and (
        words[-1].lower().strip('.,;:') in dangling
        or words[-1].strip('.,;:').endswith(("'s", '’s'))
    ):
      words.pop()
    if len(words) == 2 and (
        words[1].lower().strip('.,;:') in dangling
        or words[1].strip('.,;:').endswith(("'s", '’s'))
    ):
      return 'the nearest lock'
    result = ' '.join(words).strip(' .,;:')
    return result or 'the nearest lock'

  def _synthesize_safe_voice_line(
      self,
      voice_name: str,
      location_id: str,
      existing_lines: set[str],
  ) -> str:
    """Builds a domain-specific, non-leaking faculty observation."""
    ledger = self._ledger
    node = ledger.get_location(location_id)
    co_npcs = ledger.npcs_at_location(location_id)
    npc_focus = (
        co_npcs[0].name.split("'")[0].strip() if co_npcs else 'the culprit'
    )
    props = [
        p
        for p in (list(node.affordances) if node is not None else [])
        if not p.lower().startswith((
            'physical spillover from ',
            'evidence of ',
            'signs of ',
            'reprisal prepared by ',
            'fresh jimmy marks ',
            'confront ',
            'present ',
            'demand ',
            'wrestle ',
        ))
    ]
    tick = int(ledger.tick)
    prop_focus = 'the nearest lock'
    if props:
      for p_offset in range(len(props)):
        raw_cand = props[(tick // 2 + p_offset) % len(props)]
        clean_cand = self._extract_clean_prop_noun_phrase(raw_cand)
        if self._voice_line_leaks_undiscovered(clean_cand):
          continue
        overlaps_undiscovered = False
        for clue in ledger.story.clues.values():
          if not clue.discovered and (
              _distinctive_token_hits(clue.text, clean_cand) >= 2
              or _keyword_support(clue.text, clean_cand) >= 0.15
          ):
            overlaps_undiscovered = True
            break
        if not overlaps_undiscovered:
          for truth in ledger.story.truths.values():
            if not truth.discovered and (
                _distinctive_token_hits(truth.statement, clean_cand) >= 2
                or _keyword_support(truth.statement, clean_cand) >= 0.15
            ):
              overlaps_undiscovered = True
              break
        if not overlaps_undiscovered:
          prop_focus = clean_cand
          break
    v_lower = voice_name.lower()
    sm_obj = ledger.story.get_sub_mind(voice_name)
    v_domain = (sm_obj.domain if sm_obj is not None else '').lower()

    # 1. Prefer fresh, turn-specific LLM-deliberated perceptions/actions from
    # the active Sub-Mind council or recent voice_option_log refreshes.
    live_candidates: list[str] = []
    for opt in ledger.story.active_voice_options:
      if opt.voice.lower() == v_lower and opt.perception:
        live_candidates.append(f'{voice_name}: {opt.perception.strip()}')
        if opt.action:
          live_candidates.append(
              f'{voice_name}: {opt.perception.strip()} ({opt.action.strip()})'
          )
    for refresh in reversed(ledger.story.voice_option_log[-8:]):
      if not isinstance(refresh, Mapping):
        continue
      for raw_opt in refresh.get('options', []):
        if isinstance(raw_opt, Mapping):
          ov = str(raw_opt.get('voice', '')).strip().lower()
          op = str(raw_opt.get('perception', '')).strip()
          if ov == v_lower and op:
            live_candidates.append(f'{voice_name}: {op}')

    for cand in live_candidates:
      cand_lower = cand.strip().lower()
      cand_body = (
          cand_lower.split(':', 1)[1].strip()
          if ':' in cand_lower
          else cand_lower
      )
      body_toks = {
          w for w in re.findall(r'[a-z0-9]+', cand_body) if len(w) >= 4
      }
      near_dup = any(
          cand_body in prev
          or (
              body_toks
              and len(
                  body_toks
                  & {w for w in re.findall(r'[a-z0-9]+', prev) if len(w) >= 4}
              )
              / max(1, len(body_toks))
              >= 0.65
          )
          for prev in existing_lines
      )
      if (
          cand_lower not in existing_lines
          and not near_dup
          and not self._voice_line_leaks_undiscovered(cand)
      ):
        return cand

    if (
        'book' in v_lower
        or 'ledger' in v_lower
        or 'reckoning' in v_lower
        or v_domain == 'accounting'
    ):
      templates = [
          (
              f'{voice_name}: Check the carbon margins around {prop_focus};'
              f' cooked figures always leave a smudge when {npc_focus} is near.'
          ),
          (
              f'{voice_name}: Double-entry arithmetic around {prop_focus}'
              f' never lies for long when {npc_focus} starts sweating.'
          ),
          (
              f'{voice_name}: Follow the settlement paper on {prop_focus}—the'
              f' skimmed margin always leaves a trace on {npc_focus}.'
          ),
          (
              f'{voice_name}: Those numbers near {prop_focus} are out of'
              f' balance, and {npc_focus} knows who owes the difference.'
          ),
          (
              f"{voice_name}: Cross-check {prop_focus} against {npc_focus}'s"
              ' layoff column; the missing margin is hiding in plain sight.'
          ),
          (
              f'{voice_name}: Audit the serial dates around {prop_focus}'
              f' before {npc_focus} alters the running tally.'
          ),
          (
              f'{voice_name}: Count the short-change around {prop_focus};'
              f' {npc_focus} is carrying a deficit that cannot wait.'
          ),
          (
              f'{voice_name}: Every erased column beside {prop_focus} points'
              f" straight back to {npc_focus}'s pocket."
          ),
          (
              f'{voice_name}: Trace the duplicate voucher serials on'
              f' {prop_focus}; {npc_focus} cannot hide a cooked balance sheet.'
          ),
          (
              f'{voice_name}: Reconcile the unpaid clearing entries around'
              f' {prop_focus} before {npc_focus} shreds the counterfoil.'
          ),
          (
              f'{voice_name}: Check the Clearing House stamp on {prop_focus}—'
              f'{npc_focus} is floating bad paper across two books.'
          ),
          (
              f'{voice_name}: Tally the missing commission beside {prop_focus}'
              f' and make {npc_focus} sign for the shortfall.'
          ),
      ]
    elif (
        'naughty' in v_lower
        or 'nerve' in v_lower
        or 'mechanist' in v_lower
        or 'hard line' in v_lower
        or v_domain in ('coercion', 'tactical_force', 'force')
    ):
      templates = [
          (
              f'{voice_name}: Check the shackle and pin on {prop_focus}—iron'
              f' gives way when you lever the seam before {npc_focus} moves.'
          ),
          (
              f'{voice_name}: Watch where {npc_focus} keeps their dominant'
              f' hand before testing the latch on {prop_focus}.'
          ),
          (
              f'{voice_name}: Cold steel and worn hinges around {prop_focus}'
              f' mean {npc_focus} or a rival forced this recently.'
          ),
          (
              f'{voice_name}: Keep your weight balanced beside {prop_focus}'
              f" while sizing up {npc_focus}'s reach."
          ),
          (
              f'{voice_name}: Wedge a solid fulcrum against {prop_focus}'
              f' before {npc_focus} can drop the deadbolt.'
          ),
          (
              f'{voice_name}: Clock the shear-point on {prop_focus} and keep'
              f' {npc_focus} outside your elbow arc.'
          ),
          (
              f'{voice_name}: Test the tension spring on {prop_focus} before'
              f' {npc_focus} closes the distance.'
          ),
          (
              f'{voice_name}: Brace your boot heel near {prop_focus} so'
              f' {npc_focus} cannot slam the hardware shut.'
          ),
          (
              f'{voice_name}: Angle a heavy iron bar into {prop_focus}'
              f' before {npc_focus} can reach the safety catch.'
          ),
          (
              f'{voice_name}: Measure the deadbolt throw on {prop_focus} while'
              f' keeping {npc_focus} pinned to the counter.'
          ),
          (
              f'{voice_name}: Use the cast-iron housing of {prop_focus} as an'
              f' anvil before {npc_focus} can block your swing.'
          ),
          (
              f'{voice_name}: Clear the jammed guide-track on {prop_focus}'
              f' before {npc_focus} locks the shutter down.'
          ),
      ]
    elif (
        'ear' in v_lower
        or 'suspicion' in v_lower
        or 'empathy' in v_lower
        or 'fever' in v_lower
        or 'mirror' in v_lower
        or v_domain in ('intuition', 'empathy', 'passion')
    ):
      templates = [
          (
              f'{voice_name}: Listen to how {npc_focus} swallows near'
              f' {prop_focus}; a dry throat gives away a rehearsed alibi.'
          ),
          (
              f'{voice_name}: Rehearsed talk from {npc_focus} always skirts'
              f' around the one detail near {prop_focus} nobody wants named.'
          ),
          (
              f'{voice_name}: Watch where {npc_focus} glances whenever you'
              f' step closer to {prop_focus}.'
          ),
          (
              f'{voice_name}: Too much polite charm around {prop_focus} means'
              f' {npc_focus} is stalling for the clock.'
          ),
          (
              f"{voice_name}: Hear the drop in {npc_focus}'s cadence beside"
              f' {prop_focus}—they are covering for an accomplice.'
          ),
          (
              f"{voice_name}: {npc_focus}'s fingers twitch every time"
              f' {prop_focus} comes into the conversation.'
          ),
          (
              f'{voice_name}: Catch the sharp exhale from {npc_focus} near'
              f' {prop_focus}; fear always leaks through the voice first.'
          ),
          (
              f'{voice_name}: {npc_focus} is shifting weight beside'
              f' {prop_focus} like someone waiting for a tip-off.'
          ),
          (
              f'{voice_name}: Notice how {npc_focus} lowers their voice near'
              f' {prop_focus}; everyone in the building feels the chill.'
          ),
          (
              f'{voice_name}: Clock the sideways look {npc_focus} throws at'
              f' {prop_focus} whenever the walls groan.'
          ),
          (
              f'{voice_name}: {npc_focus} is talking too fast around'
              f' {prop_focus} to keep you from checking the back room.'
          ),
          (
              f"{voice_name}: Hear the forced calm in {npc_focus}'s pitch"
              f' beside {prop_focus}—somebody leaned on them tonight.'
          ),
      ]
    elif (
        'hustle' in v_lower
        or 'tide' in v_lower
        or 'angle' in v_lower
        or v_domain in ('leverage', 'seduction')
    ):
      templates = [
          (
              f'{voice_name}: {npc_focus} has a creditor breathing down'
              f' their neck over {prop_focus}; find the wedge and name your'
              ' price.'
          ),
          (
              f'{voice_name}: Never pay {npc_focus} retail for an answer when'
              f' {prop_focus} buys it wholesale.'
          ),
          (
              f'{voice_name}: Spot what {npc_focus} stands to lose around'
              f' {prop_focus} and offer them a side exit.'
          ),
          (
              f"{voice_name}: Trade {npc_focus}'s headache around"
              f' {prop_focus} for hard security before time runs out.'
          ),
          (
              f'{voice_name}: {npc_focus} is carrying an unpaid debt'
              f' beside {prop_focus}—buy the paper at a discount and make them'
              ' sign over the security.'
          ),
          (
              f'{voice_name}: Price {prop_focus} as insurance and make'
              f' {npc_focus} bid against their own rival.'
          ),
          (
              f"{voice_name}: Turn {prop_focus} into a finder's fee and let"
              f' {npc_focus} buy back their own peace of mind.'
          ),
          (
              f'{voice_name}: Squeeze the middleman margin around {prop_focus}'
              f' while {npc_focus} is still counting the cost.'
          ),
          (
              f'{voice_name}: Broker a three-way swap over {prop_focus} so'
              f' {npc_focus} covers the risk for you.'
          ),
          (
              f'{voice_name}: Hold {prop_focus} in escrow until {npc_focus}'
              ' puts signed collateral on the counter.'
          ),
          (
              f'{voice_name}: Leverage the time pressure on {prop_focus} to'
              f" shave {npc_focus}'s cut down to the bone."
          ),
          (
              f"{voice_name}: Play {npc_focus}'s silent partner off against"
              f' {prop_focus} and pocket the arbitrage.'
          ),
      ]
    elif (
        'copper' in v_lower
        or 'warden' in v_lower
        or 'biometric' in v_lower
        or 'lens' in v_lower
        or v_domain in ('forensics', 'procedure', 'cybernetics')
    ):
      templates = [
          (
              f'{voice_name}: Check the custody footprint around {prop_focus};'
              f' {npc_focus} knows bad paperwork leaves a wide trail.'
          ),
          (
              f'{voice_name}: Street alibis and official logs around'
              f' {prop_focus} rarely match what {npc_focus} is hiding.'
          ),
          (
              f'{voice_name}: Inspect the physical residue on {prop_focus}'
              f' before {npc_focus} wipes the slate clean.'
          ),
          (
              f'{voice_name}: Empty bluffing only works on amateurs—anchor'
              f' {npc_focus} to hard evidence on {prop_focus}.'
          ),
          (
              f'{voice_name}: Bag the forensic trace on {prop_focus} before'
              f' {npc_focus} doctors the incident book.'
          ),
          (
              f'{voice_name}: Match the tool-marks on {prop_focus} against'
              f" {npc_focus}'s timeline."
          ),
          (
              f'{voice_name}: Note the fresh scratch-marks on {prop_focus};'
              f' {npc_focus} was tampering with it minutes ago.'
          ),
          (
              f'{voice_name}: Clock the serial impression on {prop_focus} and'
              f' pin {npc_focus} to the scene.'
          ),
          (
              f'{voice_name}: Check the dispatch timestamp near {prop_focus};'
              f' {npc_focus} cut the feed too early.'
          ),
          (
              f'{voice_name}: Preserve the chain of evidence around'
              f' {prop_focus} so {npc_focus} cannot claim a lost docket.'
          ),
          (
              f'{voice_name}: Compare the keyway scratches on {prop_focus}'
              f" against {npc_focus}'s pocket hardware."
          ),
          (
              f'{voice_name}: Pin down the station call-sign on {prop_focus}'
              f' before {npc_focus} pulls rank.'
          ),
      ]
    elif (
        'guv' in v_lower
        or 'hierarchy' in v_lower
        or 'company' in v_lower
        or v_domain in ('hierarchy', 'decorum')
    ):
      templates = [
          (
              f'{voice_name}: Check how {prop_focus} ties {npc_focus} into the'
              ' command split before headquarters names a scapegoat.'
          ),
          (
              f'{voice_name}: Somebody broke station protocol around'
              f' {prop_focus}, and {npc_focus} is holding the hot end of the'
              ' chain of command.'
          ),
          (
              f'{voice_name}: Follow the pecking order—{npc_focus} would'
              f' never risk {prop_focus} without an upstairs buyer.'
          ),
          (
              f'{voice_name}: Pin the breach of discipline on {npc_focus}'
              f' using {prop_focus} before the final reckoning.'
          ),
          (
              f'{voice_name}: {npc_focus} stepped over jurisdictional lines'
              f' with {prop_focus}; make them answer to the top table.'
          ),
          (
              f'{voice_name}: Use {prop_focus} to show {npc_focus} that the'
              ' auditor already knows who skimmed the take.'
          ),
          (
              f'{voice_name}: Remind {npc_focus} beside {prop_focus} that'
              ' nobody freelancers on this post without paying the toll.'
          ),
          (
              f'{voice_name}: Pull rank over {prop_focus} and force {npc_focus}'
              ' to name who authorized the side-play.'
          ),
          (
              f'{voice_name}: Make {npc_focus} account for {prop_focus} before'
              ' the inspectors call time on the whole operation.'
          ),
          (
              f'{voice_name}: Lay the charter rule across {prop_focus} so'
              f' {npc_focus} knows the alibi is rejected.'
          ),
          (
              f'{voice_name}: Show {npc_focus} that {prop_focus} puts them on'
              ' the wrong side of the territorial ledger.'
          ),
          (
              f'{voice_name}: Enforce the house tariff around {prop_focus}'
              f' before {npc_focus} starts a turf war.'
          ),
      ]
    else:
      templates = [
          (
              f'{voice_name}: Lean across {prop_focus} like you own the'
              f' charter and watch {npc_focus} blink.'
          ),
          (
              f'{voice_name}: Front it out beside {prop_focus}; hesitation'
              f' around {npc_focus} costs more than pure brass nerve.'
          ),
          (
              f'{voice_name}: Keep your chin up near {prop_focus} and force'
              f' {npc_focus} to show their hand first.'
          ),
          (
              f'{voice_name}: Pure audacity opens more doors around'
              f' {prop_focus} than polite questions to {npc_focus} ever will.'
          ),
          (
              f'{voice_name}: Stare {npc_focus} down over {prop_focus} like'
              ' the settlement is already signed in your favour.'
          ),
          (
              f"{voice_name}: Call {npc_focus}'s bluff right over"
              f' {prop_focus} and make them sweat the deadline.'
          ),
          (
              f'{voice_name}: Claim the ground beside {prop_focus} and dare'
              f' {npc_focus} to escalate in open view.'
          ),
          (
              f'{voice_name}: Push your advantage at {prop_focus} before'
              f' {npc_focus} recovers their footing.'
          ),
          (
              f'{voice_name}: Stand square over {prop_focus} and let'
              f' {npc_focus} hear the iron in your voice.'
          ),
          (
              f'{voice_name}: Crowd {npc_focus} back from {prop_focus} before'
              ' they can steady their nerve.'
          ),
          (
              f"{voice_name}: Step straight into {npc_focus}'s eyeline over"
              f' {prop_focus} and dictate the terms.'
          ),
          (
              f'{voice_name}: Put your knuckles on {prop_focus} and dare'
              f' {npc_focus} to call your hand.'
          ),
      ]
    prefix_len = len(voice_name) + 2
    npc_low = npc_focus.lower()
    prop_low = prop_focus.lower()
    for offset in range(len(templates)):
      idx = (tick + offset) % len(templates)
      if (v_lower, idx) in self._used_voice_templates:
        continue
      cand = templates[idx]
      cand_lower = cand.strip().lower()
      cand_body = cand_lower[prefix_len:]
      cand_head = cand_body[:24] if len(cand_body) >= 24 else cand_body
      cand_tail = cand_lower[-28:] if len(cand_lower) >= 28 else cand_lower
      static_segments: list[str] = []
      for part in cand_body.split(npc_low):
        for seg in part.split(prop_low):
          stripped_seg = seg.strip()
          if len(stripped_seg) >= 10:
            static_segments.append(stripped_seg)
      already_used = any(
          (cand_tail and cand_tail in prev)
          or (cand_head and cand_head in prev)
          or any(seg in prev for seg in static_segments)
          for prev in existing_lines
      )
      if (
          cand_lower not in existing_lines
          and not already_used
          and not self._voice_line_leaks_undiscovered(cand)
      ):
        self._used_voice_templates.add((v_lower, idx))
        return cand
    # Rotate through unique tick-specific observations so fallbacks stay fresh.
    room_label = node.name if node is not None else location_id
    fallback_rotations = [
        (
            f'{voice_name}: Watch how {npc_focus} angles away from {prop_focus}'
            f' inside {room_label} at tick {tick}.'
        ),
        (
            f'{voice_name}: Note the tension around {prop_focus} while'
            f' {npc_focus} weighs the next move in {room_label}.'
        ),
        (
            f'{voice_name}: Measure the distance between {npc_focus} and'
            f' {prop_focus} before the pressure shifts.'
        ),
        (
            f'{voice_name}: Study the wear on {prop_focus}; {npc_focus} has'
            ' been working this angle since the shift began.'
        ),
        (
            f'{voice_name}: Track the line of sight from {npc_focus} to'
            f' {prop_focus} across {room_label}.'
        ),
        (
            f'{voice_name}: Press {npc_focus} on {prop_focus} before the'
            f' trail inside {room_label} goes cold.'
        ),
    ]
    for r_idx in range(len(fallback_rotations)):
      cand_fb = fallback_rotations[(tick + r_idx) % len(fallback_rotations)]
      if cand_fb.strip().lower() not in existing_lines:
        return cand_fb
    return (
        f'{voice_name}: Observe {npc_focus} beside {prop_focus} in'
        f' {room_label} (step {tick}).'
    )

  def _narrative_echoes_beat(
      self, narrative: str, storylet: story_state.Storylet
  ) -> bool:
    """Checks whether `narrative` echoes key words from `storylet.beat`."""
    lower = narrative.lower()
    words = [
        w.strip('.,;:"\'()[]')
        for w in storylet.beat.lower().split()
        if len(w) >= 6
    ]
    if not words:
      return False
    hits = sum(1 for w in set(words) if w in lower)
    return hits >= min(3, max(2, len(set(words)) // 5))

  def _ensure_text_supported_in_log(
      self,
      source_text: str,
      *,
      label: str,
      location_id: str,
      existing_prose: str = '',
  ) -> None:
    """Appends non-verbatim key specifics to narrative_log when prose uses synonyms."""
    if self._ledger.tick <= 0 or self._ledger.prose_mode == 'concise':
      return
    words = sorted({
        w.strip('.,;:"\'()[]—-')
        for w in source_text.lower().split()
        if len(w.strip('.,;:"\'()[]—-')) >= 6
    })
    if not words:
      return
    lower_ev = existing_prose.lower()
    hits = sum(1 for w in words if w in lower_ev)
    if hits / len(words) >= 0.45:
      return
    needed = max(6, int(len(words) * 0.45) + 1)
    key_terms = ', '.join(words[:needed])
    self._ledger.append_narrative_entry(
        kind='revelation',
        actor='Journal',
        text=f'Key specifics noted ({label}): {key_terms}.',
        location_id=location_id,
    )

  # ---------------------------------------------------------------------------
  # Seam 2: Warm off-screen step (NPC-vs-NPC, grudges, faction ops, storylets)
  # ---------------------------------------------------------------------------

  def _warm_storylet_fits_npc(
      self, storylet: story_state.Storylet, npc: world_ledger.NpcRecord
  ) -> bool:
    """Returns False if `storylet` is explicitly authored for a different NPC."""
    for cond in storylet.preconditions:
      if isinstance(cond, Mapping):
        if 'npc_present' in cond:
          if str(cond['npc_present']).strip() != npc.name:
            return False
        if (
            'npc_tier' in cond
            and isinstance(cond['npc_tier'], Sequence)
            and not isinstance(cond['npc_tier'], str)
            and cond['npc_tier']
        ):
          if str(cond['npc_tier'][0]).strip() != npc.name:
            return False
    all_npcs = self._ledger.all_npcs()
    tags_lower = {str(t).lower() for t in storylet.tags}
    text = f'{storylet.id} {storylet.beat}'.lower()
    my_tokens = {
        t.lower().strip('\'"()') for t in npc.name.split() if len(t) >= 4
    }
    beat_stripped = storylet.beat.strip()
    if beat_stripped.lower().startswith('off-screen,'):
      after_prefix = beat_stripped[len('off-screen,') :].strip()
      subject_words = [
          w.strip('.,;:\'"()')
          for w in after_prefix.split()[:3]
          if w.strip('.,;:\'"()') and w.strip('.,;:\'"()')[0].isupper()
      ]
      subject_tokens = {
          w.lower()
          for w in subject_words
          if len(w) >= 4
          and w.lower()
          not in {'someone', 'somebody', 'rival', 'local', 'crew', 'warden'}
      }
      if subject_tokens:
        return bool(my_tokens & subject_tokens)
    if any(tok in text or tok in tags_lower for tok in my_tokens):
      return True
    for other_name in all_npcs:
      if other_name == npc.name:
        continue
      other_tokens = [
          t.lower().strip('\'"()') for t in other_name.split() if len(t) >= 4
      ]
      if any(
          tok in storylet.id.lower()
          or tok in tags_lower
          or f'off-screen, {tok}' in storylet.beat.lower()
          for tok in other_tokens
      ):
        return False
    return True

  def choose_warm_move(self, npc: world_ledger.NpcRecord) -> WarmMove:
    """Samples an inner voice from `npc`'s council to drive an off-screen move."""
    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      loc_node = ledger.get_location(npc.location_id)
      loc_name = loc_node.name if loc_node else npc.location_id
      npc.ensure_sub_minds()
      voices = list(npc.sub_minds.values())
      if voices:
        weights = [
            math.exp(0.75 * float(sm.rank) + (0.5 if sm.intrusive else 0.0))
            for sm in voices
        ]
        chosen_sm = self._rng.choices(voices, weights=weights, k=1)[0]
      else:
        chosen_sm = npc.dominant_sub_mind()

      voice_cue = (
          f'Your dominant inner voice is {chosen_sm.name} '
          f'[Rank {chosen_sm.rank}, {chosen_sm.domain}] '
          f'(Obsession: "{chosen_sm.obsession or chosen_sm.worldview}"). '
      )

      if self.active:
        warm_pool = [
            b
            for b in self.eligible(
                'warm', location_id=npc.location_id, npc_name=npc.name
            )
            if self._warm_storylet_fits_npc(b, npc)
        ]
        if warm_pool:
          beat = warm_pool[0]
          rumor_override = ''
          reliability = 'true'
          for eff in beat.effects:
            if 'rumor' in eff and isinstance(eff['rumor'], Mapping):
              rumor_override = str(eff['rumor'].get('text', ''))
              reliability = str(eff['rumor'].get('reliability', 'true'))
          return WarmMove(
              kind='storylet',
              call_to_action=(
                  f'You are {npc.name} acting off-screen at {loc_name} '
                  f'({ledger.player_name} is NOT in this room). '
                  f'{voice_cue}{beat.beat} '
                  'In ONE concise sentence (under 25 words), state the one '
                  'concrete off-screen action you take.'
              ),
              storylet=beat,
              rumor_reliability=reliability,
              rumor_override=rumor_override,
          )

        recent_all_rumors = ledger.get_rumors(limit=20)
        recent_rumor_by_npc = [
            r for r in recent_all_rumors if npc.name.lower() in r.lower()
        ]
        if recent_rumor_by_npc:
          anti_repeat = (
              ' Do NOT repeat prior off-screen moves or reuse the same tools,'
              f' props, or verbs ({" | ".join(recent_rumor_by_npc[-3:])}).'
          )
        else:
          anti_repeat = ''
        all_clean_props = [
            p
            for p in (loc_node.affordances if loc_node is not None else [])
            if not p.lower().startswith((
                'physical spillover from ',
                'evidence of ',
                'signs of ',
                'fresh jimmy marks ',
                'reprisal prepared by ',
            ))
        ]
        if all_clean_props:
          rot = int(ledger.tick) % len(all_clean_props)
          rotated_loc_props = (all_clean_props[rot:] + all_clean_props[:rot])[
              :3
          ]
        else:
          rotated_loc_props = []
        if rotated_loc_props:
          props_clause = (
              f' using or altering a fresh prop in {loc_name}'
              f' ({"; ".join(rotated_loc_props)})'
          )
        else:
          props_clause = ''

        # Sample off-screen intent directly from the chosen Sub-Mind's domain.
        all_npcs = ledger.all_npcs()
        rivals = [
            (other_name, note)
            for other_name, note in npc.relationships.items()
            if other_name != ledger.player_name and other_name in all_npcs
        ]
        player_rel = npc.relationships.get(ledger.player_name, '').lower()
        fac = story.factions.get(npc.faction) if npc.faction else None
        hostile_to_player = any(
            tok in player_rel for tok in _NEGATIVE_RELATIONSHIP_TOKENS
        ) or (fac is not None and fac.standing_with_player <= -2)

        if hostile_to_player:
          return WarmMove(
              kind='grudge',
              target_npc=ledger.player_name,
              faction_id=npc.faction,
              call_to_action=(
                  f'You are {npc.name} acting off-screen at {loc_name}'
                  f' ({ledger.player_name} is NOT in this room).'
                  f' {voice_cue}Driven by "{chosen_sm.name}", state in ONE'
                  ' concise sentence (under 25 words) one physical off-screen'
                  f' counter-move you make in {loc_name}{props_clause} against'
                  f' {ledger.player_name}.{anti_repeat}'
              ),
          )
        if rivals:
          target_name, edge_note = self._rng.choice(rivals)
          return WarmMove(
              kind='confront',
              target_npc=target_name,
              call_to_action=(
                  f'You are {npc.name} acting off-screen at {loc_name} '
                  f'({ledger.player_name} is NOT in this room). {voice_cue}'
                  f'Regarding {target_name} ("{edge_note}"), state in ONE '
                  'concise sentence (under 25 words) one concrete physical '
                  f'off-screen move you execute{props_clause} to advance '
                  f'"{chosen_sm.obsession}" against {target_name}.{anti_repeat}'
              ),
          )
        if fac is not None and chosen_sm.domain == 'stonewalling':
          return WarmMove(
              kind='faction_op',
              faction_id=fac.id,
              call_to_action=(
                  f'You are {npc.name} of {fac.name} acting off-screen at '
                  f'{loc_name} ({ledger.player_name} is NOT in this room). '
                  f'{voice_cue}In ONE concise sentence (under 25 words), state '
                  f'one physical off-screen action{props_clause} to advance '
                  f'"{fac.aim}".{anti_repeat}'
              ),
          )

      return WarmMove(
          kind='pursue',
          call_to_action=(
              f'You are {npc.name} acting off-screen at {loc_name} '
              f'({ledger.player_name} is NOT in this room). {voice_cue}'
              'In ONE concise sentence (under 25 words), state one concrete '
              'physical off-screen action you take right now to act on '
              f'"{chosen_sm.obsession or npc.goal}".'
          ),
      )

  def _maybe_relocate_offscreen_clue(
      self,
      npc: world_ledger.NpcRecord,
      dest_id: str,
      *,
      rival_name: str = '',
  ) -> str | None:
    """Relocates an undiscovered movable off-screen clue to `dest_id`, if any."""
    ledger = self._ledger
    story = ledger.story
    player_loc = ledger.player_location_id
    if dest_id == player_loc or story.clues_relocated_count >= 2:
      return None

    rival_loc = ''
    if rival_name:
      rival_rec = ledger.get_npc(rival_name)
      if rival_rec is not None:
        rival_loc = rival_rec.location_id

    preferred: list[story_state.Clue] = []
    fallback: list[story_state.Clue] = []
    for clue in story.clues.values():
      if clue.discovered or not clue.movable or not clue.placed_at:
        continue
      placed_norm = world_ledger.normalize_location_id(clue.placed_at)
      if placed_norm in (player_loc, dest_id) or clue.placed_at == npc.name:
        continue
      holder_npc = ledger.get_npc(clue.placed_at)
      if holder_npc is not None and holder_npc.location_id == player_loc:
        continue
      if rival_name and (
          clue.placed_at == rival_name
          or (rival_loc and placed_norm == rival_loc)
      ):
        preferred.append(clue)
      elif story.clues_relocated_count == 0:
        fallback.append(clue)

    pool = preferred or fallback
    if not pool:
      return None
    chosen = pool[0]
    return ledger.relocate_clue(chosen.id, dest_id, by_npc=npc.name)

  def apply_warm_outcome(
      self,
      npc: world_ledger.NpcRecord,
      move: WarmMove,
      clean_action: str,
      dest_id: str,
      dest_name: str,
  ) -> str:
    """Applies the deterministic mutations of a Warm step and returns the rumor."""
    with self._ledger.lock:
      ledger = self._ledger
      story = ledger.story
      clause = clean_action.strip()
      if clause.lower().startswith(npc.name.lower()):
        rest = clause[len(npc.name) :].lstrip(' .,:;-—')
        if rest and not rest.startswith(("'s", '’s')):
          clause = rest
        elif not rest:
          clause = f'tightens control around {dest_name}'
      if not clause or len(clause.split()) < 2:
        clause = f'tightens control around {dest_name}'
      if '(' in clause and clause.rfind('(') > clause.rfind(')'):
        clause = clause[: clause.rfind('(')].rstrip(' ,;:-—')
      short = (
          clause[:90].rsplit(' ', 1)[0].rstrip(' ,;:-—')
          if len(clause) > 90 and ' ' in clause[:90]
          else clause[:90].rstrip(' ,;:-—')
      )
      dangling_short = {
          'the',
          'a',
          'an',
          'of',
          'to',
          'in',
          'on',
          'at',
          'by',
          'for',
          'from',
          'with',
          'while',
          'using',
          'and',
          'or',
          'into',
          'onto',
          'across',
          'through',
          'under',
          'over',
          'shorting',
          'his',
          'her',
          'their',
          'its',
      }
      short_words = short.split()
      while (
          len(short_words) > 3
          and short_words[-1].lower().strip('.,;:()') in dangling_short
      ):
        short_words.pop()
      short = ' '.join(short_words).rstrip(' .,;:-—(')
      scar_short = (
          short[:65].rsplit(' ', 1)[0].rstrip(' .,;:-—(')
          if len(short) > 65 and ' ' in short[:65]
          else short.rstrip(' .,;:-—(')
      )
      scar_words = scar_short.split()
      while (
          len(scar_words) > 3
          and scar_words[-1].lower().strip('.,;:()') in dangling_short
      ):
        scar_words.pop()
      scar_short = ' '.join(scar_words).rstrip(' .,;:-—(')
      dom = npc.dominant_sub_mind()
      ledger.apply_npc_voice_clash(
          npc.name,
          scar_target=dom.name,
          scar_note=(
              f'[t{ledger.tick}][warm:{move.kind}] Acted at {dest_name}: '
              f'{scar_short}'
          ),
      )

      if move.kind == 'storylet' and move.storylet is not None:
        # Apply non-rumor effects via the standard path; record the rumor once
        # here so `warm_events_count` increments exactly once.
        non_rumor_effects = [
            e for e in move.storylet.effects if 'rumor' not in e
        ]
        story.note_fired(
            move.storylet.id, tick=self._ledger.tick, source=npc.name
        )
        self.apply_effects(non_rumor_effects, source=npc.name, is_warm=False)
        self._cap_tension_by_act()
        if move.rumor_override:
          rumor_body = f'{move.rumor_override} ({npc.name} {clause})'
        else:
          rumor_body = (
              f'Word spreads from {dest_name}: {npc.name} ({npc.role_summary}) '
              f'{clause}'
          )
        ledger.record_rumor(
            rumor_body,
            source=npc.name,
            is_warm=True,
            reliability=move.rumor_reliability,
        )
        self._ensure_text_supported_in_log(
            move.storylet.beat,
            label=npc.name,
            location_id=ledger.player_location_id,
            existing_prose=rumor_body,
        )
        return rumor_body

      if move.kind == 'confront' and move.target_npc:
        target = move.target_npc
        ledger.set_npc_relationship(
            npc.name,
            target,
            f'Moved against {target} from {dest_name}: {short}',
        )
        ledger.set_npc_relationship(
            target,
            npc.name,
            f'Heard that {npc.name} acted regarding them at {dest_name}:'
            f' {short}',
        )
        story.shift_tension(0.4)
        turf_war_tag = ''
        target_rec = ledger.get_npc(target)
        if (
            target_rec is not None
            and npc.faction
            and target_rec.faction
            and npc.faction != target_rec.faction
        ):
          fa_name = (
              story.factions[npc.faction].name
              if npc.faction in story.factions
              else npc.faction
          )
          fb_name = (
              story.factions[target_rec.faction].name
              if target_rec.faction in story.factions
              else target_rec.faction
          )
          new_standing = story.shift_inter_faction(
              npc.faction,
              target_rec.faction,
              -1,
              reason=(
                  f'Open turf war between {fa_name} and {fb_name}: '
                  f'{npc.name} moved against {target} at {dest_name} ({short}).'
              ),
              tick=ledger.tick,
          )
          if new_standing is not None and new_standing <= -4:
            pair = sorted([npc.faction, target_rec.faction])
            ledger.set_world_fact(f'turf_war_{pair[0]}_{pair[1]}', 'active')
            ledger.add_location_affordance(
                dest_id,
                f'signs of open turf war between {fa_name} and {fb_name}',
            )
            story.shift_tension(0.6)
            turf_war_tag = f' [Turf war: {fa_name} vs {fb_name}]'

        relocated = self._maybe_relocate_offscreen_clue(
            npc, dest_id, rival_name=target
        )
        self._cap_tension_by_act()
        heist_tag = f' [Lifted: {relocated}]' if relocated else ''
        rumor_body = (
            f'Word from {dest_name} (concerning {target}): {npc.name} '
            f'{clause}{turf_war_tag}{heist_tag}'
        )
        ledger.record_rumor(
            rumor_body, source=npc.name, is_warm=True, reliability='true'
        )
        return rumor_body

      if move.kind == 'grudge':
        player = ledger.player_name
        ledger.set_npc_relationship(
            npc.name,
            player,
            f'Hostile; retaliated from {dest_name} ({short}).',
        )
        trap_prop = f'signs of a reprisal prepared by {npc.name} ({short})'
        ledger.add_location_affordance(dest_id, trap_prop)
        story.shift_tension(0.8)
        self._cap_tension_by_act()
        rumor_body = (
            f'A warning from {dest_name}: {npc.name} is moving against '
            f'{player}—{npc.name} {clause}'
        )
        ledger.record_rumor(
            rumor_body, source=npc.name, is_warm=True, reliability='true'
        )
        return rumor_body

      if move.kind == 'faction_op' and move.faction_id:
        fac = story.factions.get(move.faction_id)
        fac_label = fac.name if fac else move.faction_id
        story.shift_tension(0.3)
        self._cap_tension_by_act()
        relocated = self._maybe_relocate_offscreen_clue(npc, dest_id)
        heist_tag = f' [Moved: {relocated}]' if relocated else ''
        rumor_body = (
            f'{fac_label} activity at {dest_name}: {npc.name}'
            f' {clause}{heist_tag}'
        )
        ledger.record_rumor(
            rumor_body, source=fac_label, is_warm=True, reliability='true'
        )
        return rumor_body

      rumor_body = (
          f'Word spreads from {dest_name}: {npc.name} ({npc.role_summary}) '
          f'{clause}'
      )
      ledger.record_rumor(
          rumor_body, source=npc.name, is_warm=True, reliability='true'
      )
      return rumor_body

  # ---------------------------------------------------------------------------
  # Seam 3: Frontier expansion (clue placement + faction + secret/routine asks)
  # ---------------------------------------------------------------------------

  def notes_for_frontier(
      self, stub: world_ledger.LocationNode, origin_id: str
  ) -> FrontierAsk:
    """Selects clues and faction guidance for a frontier room about to expand."""
    del origin_id
    if not self.active:
      return FrontierAsk()

    with self._ledger.lock:
      story = self._ledger.story
      self._frontier_counter += 1

      # 1. Pick clues whose carrier_hint matches the stub name, else up to 1
      #    unplaced location/document clue that is not hinted for another known
      #    location in the ledger.
      stub_lower = stub.name.lower()
      if hasattr(self._ledger, 'to_dict_locations'):
        other_loc_names = [
            loc.name.lower()
            for loc in self._ledger.to_dict_locations()
            if loc.name.lower() and loc.name.lower() != stub_lower
        ]
      else:
        other_loc_names = [
            str(loc.get('name', '')).lower()
            for loc in self._ledger.to_dict().get('locations', {}).values()
            if str(loc.get('name', '')).lower()
            and str(loc.get('name', '')).lower() != stub_lower
        ]
      matched: list[story_state.Clue] = []
      fallback_pool: list[story_state.Clue] = []
      for clue in story.unplaced_clues():
        if clue.carrier not in ('location', 'document'):
          continue
        hint_lower = clue.carrier_hint.lower()
        if stub_lower and stub_lower in hint_lower:
          matched.append(clue)
        elif not any(
            other_name in hint_lower for other_name in other_loc_names
        ):
          fallback_pool.append(clue)
      chosen_clues = matched[:4]
      if not chosen_clues and fallback_pool:
        chosen_clues = [fallback_pool[0]]

      # 2. Pick a faction for the room (round-robin across human factions).
      human_factions = [
          f for f in story.factions.values() if f.id != 'floodgate'
      ] or list(story.factions.values())
      fac = (
          human_factions[(self._frontier_counter - 1) % len(human_factions)]
          if human_factions
          else None
      )

      # 3. Pick a truth for the resident NPC's secret to touch.
      undiscovered_truths = [
          t for t in story.truths.values() if not t.discovered
      ] or list(story.truths.values())
      if undiscovered_truths:
        truth = undiscovered_truths[
            (self._frontier_counter - 1) % len(undiscovered_truths)
        ]
      else:
        truth = None

      lines = ['\nStory Layer Requirements for this Location:']
      if story.tone.setting_style:
        lines.append(
            f'- Setting & Aesthetic: {story.tone.setting_style} '
            f'(Prose Register: {story.tone.prose_register})'
        )
      if story.tone.room_palette:
        motif = story.tone.room_palette[
            (self._frontier_counter - 1) % len(story.tone.room_palette)
        ]
        lines.append(f'- Architectural / Set-Piece Motif: {motif}')
      if story.tone.npc_archetypes:
        archetype = story.tone.npc_archetypes[
            (self._frontier_counter - 1) % len(story.tone.npc_archetypes)
        ]
        lines.append(f'- Resident NPC Archetype: {archetype}')
      if fac is not None:
        lines.append(
            f'- Faction Territory: {fac.name} (id: "{fac.id}"). '
            f'Aim: {fac.aim} Register: {fac.register}'
        )
      if chosen_clues:
        lines.append(
            '- Embed the following clue carriers as concrete items in '
            '`"affordances"` so a careful player can discover them (write '
            'natural object descriptions only; NEVER include bracketed clue '
            'IDs like [c_...] in affordance strings):'
        )
        for c in chosen_clues:
          lines.append(
              f'  * [{c.id}] ({c.carrier}, hint: {c.carrier_hint}) -> '
              f'supports: "{c.text}"'
          )
      if truth is not None:
        lines.append(
            '- give `"npc"` fields `"faction"`, `"secret"`, and `"wound"` '
            'connected to this world tension (without stating it flatly): '
            f'{truth.statement}'
        )

      return FrontierAsk(
          faction_id=fac.id if fac else '',
          faction_name=fac.name if fac else '',
          faction_register=fac.register if fac else '',
          clues=tuple(chosen_clues),
          truth_hint=truth.statement if truth else '',
          prompt_block='\n'.join(lines),
      )

  def commit_frontier_expansion(
      self,
      location_id: str,
      ask: FrontierAsk,
      npc_data: Mapping[str, Any] | None,
  ) -> dict[str, Any]:
    """Records clue placements and enriches the newly spawned NPC's record."""
    if not self.active:
      return {}
    with self._ledger.lock:
      clue_ids = [c.id for c in ask.clues]
      node = self._ledger.get_location(location_id)
      if node is not None and (clue_ids or ask.faction_id):
        self._ledger.expand_location(
            location_id,
            description=node.description,
            clue_ids=clue_ids,
            faction_lean=ask.faction_id,
        )
      npc_extras: dict[str, Any] = {
          'faction': (
              str((npc_data or {}).get('faction') or ask.faction_id).strip()
          ),
          'secret': (
              str(
                  (npc_data or {}).get('secret')
                  or (
                      f'Knows more than they admit about: {ask.truth_hint[:90]}'
                      if ask.truth_hint
                      else ''
                  )
              ).strip()
          ),
          'wound': (
              str(
                  (npc_data or {}).get('wound')
                  or 'Will only speak plainly in exchange for protection or proof.'
              ).strip()
          ),
      }
      return npc_extras

  # ---------------------------------------------------------------------------
  # Seam 4: Cold NPC routines & rehydration beats
  # ---------------------------------------------------------------------------

  def step_cold_routines(self) -> list[tuple[str, str, str]]:
    """Moves Cold NPCs whose routine puts them elsewhere at the current hour.

    Hot and Warm NPCs are never auto-moved so active scenes are undisturbed.

    Returns:
      List of `(npc_name, from_location_id, to_location_id)` transitions.
    """
    if not self.active:
      return []
    moved: list[tuple[str, str, str]] = []
    with self._ledger.lock:
      hour = (self._ledger.clock_minutes % (24 * 60)) // 60
      player_loc = self._ledger.player_location_id
      for npc in self._ledger.npcs_in_tier(world_ledger.NpcTier.COLD):
        if npc.location_id == player_loc:
          continue
        target_name = npc.routine_location_for_hour(hour)
        if not target_name:
          continue
        target_id = self._ledger.ensure_stub_location(target_name)
        has_player_turns = any(
            e.get('kind') == 'player_action'
            for e in self._ledger.get_narrative_log()
        )
        if has_player_turns and self._ledger.npcs_at_location(target_id):
          continue
        if target_id != npc.location_id and target_id != player_loc:
          origin_id = npc.location_id
          self._ledger.move_entity(npc.name, target_id)
          moved.append((npc.name, origin_id, target_id))
    return moved

  def rehydration_observation(self, npc_name: str) -> str:
    """Returns an extra catch-up observation when `npc_name` rehydrates."""
    if not self.active:
      return ''
    with self._ledger.lock:
      rec = self._ledger.get_npc(npc_name)
      if rec is None:
        return ''
      parts: list[str] = []
      # Surface any NPC-vs-NPC or player relationship changes.
      for other, note in rec.relationships.items():
        parts.append(f'Your current disposition toward {other}: {note}')
      last_line = ''
      last_player_action = ''
      last_tick = -1
      visited_rooms: list[str] = []
      for entry in self._ledger.get_narrative_log():
        if (
            entry.get('kind') == 'player_action'
            and entry.get('location_id') == rec.location_id
        ):
          last_player_action = str(entry.get('text', '')).strip()
        elif (
            entry.get('kind') == 'npc_resolution'
            and entry.get('actor') == npc_name
        ):
          last_line = str(entry.get('text', '')).strip()
          last_tick = int(entry.get('tick', -1))
      if last_tick >= 0:
        for entry in self._ledger.get_narrative_log():
          if (
              int(entry.get('tick', -1)) > last_tick
              and entry.get('kind') == 'arrival'
          ):
            r_name = str(entry.get('location_name', '')).strip()
            if r_name and r_name not in visited_rooms:
              visited_rooms.append(r_name)
      if last_line:
        quotes = re.findall(r'"([^"]+)"', last_line)
        spoken_excerpt = (
            ' '.join(quotes).strip() if quotes else last_line[-260:]
        )
        recent_stop = visited_rooms[-1] if visited_rooms else 'across town'
        player_ctx = ''
        if last_player_action:
          player_ctx = (
              f' after {self._ledger.player_name} tried'
              f' "{last_player_action[:110]}"'
          )
        parts.append(
            f'When {self._ledger.player_name} last departed for '
            f'{recent_stop}{player_ctx}, {npc_name} had told them: '
            f'"{spoken_excerpt[:280]}" — terms that now shift in light of '
            f'the new evidence {self._ledger.player_name} carries and the '
            'latest off-screen moves.'
        )
      beats: list[story_state.Storylet] = []
      if self._ledger.tick == 0 or bool(last_line):
        beats = [
            b
            for b in self.eligible(
                'rehydrate', location_id=rec.location_id, npc_name=npc_name
            )
            if self._warm_storylet_fits_npc(b, rec)
        ]
      if beats:
        beat = beats[0]
        self.fire_storylet(beat, source=npc_name, is_warm=False)
        parts.append(beat.beat)
        if self._ledger.tick > 0 and beat.beat.strip():
          self._ensure_text_supported_in_log(
              beat.beat,
              label=npc_name,
              location_id=rec.location_id,
          )
      return ' '.join(parts)

  # ---------------------------------------------------------------------------
  # Seam 5: Doom clock, acts, and endings
  # ---------------------------------------------------------------------------

  def on_clock(self) -> list[str]:
    """Checks for doom stage transitions and fires matching `doom` storylets."""
    if not self.active:
      return []
    transitions: list[str] = []
    with self._ledger.lock:
      story = self._ledger.story
      if story.ended:
        return []
      frac = story.doom_fraction(self._ledger.clock_minutes)
      new_stage = story.stage_for_fraction(frac)
      if new_stage.index <= story.doom_stage:
        return []

      for stage in story.doom.stages:
        if story.doom_stage < stage.index <= new_stage.index:
          story.doom_stage = stage.index
          transitions.append(stage.name)
          if stage.world_effect:
            self._ledger.append_narrative_entry(
                kind='doom',
                actor=story.doom.name,
                text=f'[{stage.name}] {stage.world_effect}',
                location_id=self._ledger.player_location_id,
            )
          for beat in self.eligible('doom'):
            if (
                any('end_run' in eff for eff in beat.effects)
                and story.ending_archetype
                and story.ending_archetype != 'eleventh_hour_bell'
            ):
              continue
            self.fire_storylet(beat, source=story.doom.name, is_warm=True)
            if beat.beat.strip():
              self._ledger.append_narrative_entry(
                  kind='revelation',
                  actor=story.doom.name,
                  text=beat.beat.strip(),
                  location_id=self._ledger.player_location_id,
              )

      # Hard doom: if the final stage is reached and no ending fired yet.
      last_index = story.doom.stages[-1].index if story.doom.stages else 0
      if (
          story.doom_stage >= last_index
          and story.tone.doom_hard
          and not story.ended
          and not (
              story.ending_archetype
              and story.ending_archetype != 'eleventh_hour_bell'
          )
      ):
        clues_found = sum(1 for c in story.clues.values() if c.discovered)
        if story.truths_discovered() >= 1 and clues_found >= 2:
          self.resolve_emergent_climax(
              self._ledger.player_location_id,
              outcome='unresolved',
              trigger_reason='doom_clock_climax',
          )
        else:
          story.ended = True
          story.ending_summary = self.compose_epilogue(outcome='failure')
      story.recompute_act()
    return transitions

  def is_climax_ready(self, location_id: str | None = None) -> bool:
    """Returns True when Act 3 live simulation state supports an emergent climax."""
    del location_id
    if not self.active:
      return False
    with self._ledger.lock:
      story = self._ledger.story
      last_stage = story.doom.stages[-1].index if story.doom.stages else 4
      if story.ended or (
          story.act < 3
          and not (
              story.doom_stage >= max(1, last_stage - 1)
              and story.truths_discovered() >= 1
          )
      ):
        return False
      truths_count = story.truths_discovered()
      clues_count = sum(1 for c in story.clues.values() if c.discovered)
      wars_count = len(story.active_turf_wars())
      cracked_npcs = sum(
          1
          for r in self._ledger.all_npcs().values()
          if any(
              sm.name != 'The Mask' and sm.rank >= 4
              for sm in r.sub_minds.values()
          )
      )
      max_fac_standing = max(
          (abs(f.standing_with_player) for f in story.factions.values()),
          default=0,
      )
      inv_count = len(self._ledger.player_inventory)
      if truths_count >= max(1, len(story.truths) - 1) and (
          cracked_npcs >= 2 or clues_count >= min(6, len(story.clues))
      ):
        return True
      if truths_count >= 2 and (max_fac_standing >= 2 or inv_count >= 2):
        return True
      if (
          truths_count >= 2
          and (wars_count >= 2 or max_fac_standing >= 2)
          and story.tension >= 7.0
      ):
        return True
      if story.doom_stage >= max(1, last_stage - 1) and (
          truths_count >= 1
          or clues_count >= min(6, max(3, len(story.clues) - 1))
          or wars_count >= 1
          or cracked_npcs >= 2
      ):
        return True
      return False

  def _dominant_player_sub_mind(self) -> story_state.SubMindState | None:
    """Returns the player's highest-ranked active Sub-Mind."""
    with self._ledger.lock:
      story = self._ledger.story
      active = [sm for sm in story.sub_minds.values() if sm.active] or list(
          story.sub_minds.values()
      )
      if not active:
        return None
      return max(
          active,
          key=lambda sm: (int(sm.rank), len(sm.scars), int(sm.xp)),
      )

  def classify_emergent_ending(
      self,
      location_id: str = '',
      *,
      outcome: str = 'unresolved',
      player_action: str = '',
      narrative_event: str = '',
  ) -> tuple[str, str, str]:
    """Classifies the emergent ending into `(archetype, title, beat)`."""
    with self._ledger.lock:
      story = self._ledger.story
      is_caper = story.tone.name == 'caper' or 'corrigan_firm' in story.factions
      is_glassharbor = (
          'wardens' in story.factions
          or 'glassharbor' in story.world_title.lower()
      )
      is_custom_world = not is_caper and not is_glassharbor
      if outcome == 'failure':
        if is_custom_world:
          return (
              'eleventh_hour_bell',
              f'WHEN {story.doom.name.upper()} STRUCK',
              (
                  f'The countdown for {story.doom.name} expired before the'
                  ' crisis could be resolved.'
              ),
          )
        return (
            'eleventh_hour_bell',
            'THE ELEVENTH-HOUR BELL'
            if is_caper
            else 'WHEN THE KING-TIDE BROKE',
            'The clock ran out before the holdall was squared.',
        )

      combined = f'{player_action} {narrative_event}'.lower()
      cleaned_act = player_action.strip()
      invoked_voice = (
          cleaned_act[1 : cleaned_act.index(']')].strip()
          if cleaned_act.startswith('[') and ']' in cleaned_act
          else self._last_invoked_voice
      )
      dom_sm = self._dominant_player_sub_mind()
      lead_sm = (
          story.get_sub_mind(invoked_voice)
          if invoked_voice and story.get_sub_mind(invoked_voice) is not None
          else dom_sm
      )
      lead_domain = lead_sm.domain if lead_sm is not None else ''
      wars = story.active_turf_wars()
      cracked_wounds = sum(
          1
          for r in self._ledger.all_npcs().values()
          if any(
              (
                  sm.name == 'The Wound'
                  or sm.domain == 'self_preservation'
                  or sm.role_archetype == 'wound'
              )
              and (sm.rank >= 4 or sm.scars)
              for sm in r.sub_minds.values()
          )
      )
      dealmaker_count = 0
      exposed_count = 0
      for r in self._ledger.all_npcs().values():
        r.ensure_sub_minds()
        dom_r = r.dominant_sub_mind()
        if dom_r.name == 'The Angle' or dom_r.domain == 'leverage':
          dealmaker_count += 1
        elif dom_r.name == 'The Wound' or dom_r.domain == 'self_preservation':
          exposed_count += 1

      factions = list(story.factions.values())
      all_dispositions_low = ' '.join(
          f.current_disposition().lower() for f in factions
      )
      has_strongarm_dispositions = any(
          tok in all_dispositions_low
          for tok in (
              'strong-armed',
              'physically rattled',
              'cornered by force',
          )
      )
      has_forensic_dispositions = any(
          tok in all_dispositions_low
          for tok in (
              'forensic evidence',
              'bracing for warrants',
          )
      )
      reciprocal_factions_count = sum(
          1
          for f in factions
          if 'reciprocal side-angle' in f.current_disposition().lower()
          or 'cut into a working alliance' in f.current_disposition().lower()
      )
      if 'corrigan_firm' in story.factions:
        gov_id = 'corrigan_firm'
      elif 'wardens' in story.factions:
        gov_id = 'wardens'
      elif factions:
        gov_id = factions[0].id
      else:
        gov_id = ''
      gov_standing = (
          story.factions[gov_id].standing_with_player
          if gov_id in story.factions
          else 0
      )
      gov_name = (
          story.factions[gov_id].name
          if gov_id in story.factions
          else 'the Ruling Order'
      )
      rival_factions = [f for f in factions if f.id != gov_id]
      if rival_factions:
        top_rival = max(
            rival_factions,
            key=lambda f: (
                'reciprocal' in f.current_disposition().lower(),
                f.standing_with_player,
                len(f.recent_actions),
            ),
        )
      else:
        top_rival = None

      # 1. Scorched-Earth Turf Coup / Chaos Ending.
      if (
          outcome == 'chaos'
          or any(
              w in combined
              for w in (
                  'burn it all',
                  'torch the',
                  'blow the',
                  'crossfire',
                  'shootout',
                  'riot',
                  'pentagram',
                  'séance',
                  'shergar',
                  'scorched earth',
                  'shatter the court',
                  'overload the core',
              )
          )
          or (
              (
                  len(wars) >= 1
                  or cracked_wounds >= 1
                  or has_strongarm_dispositions
              )
              and lead_domain
              in (
                  'coercion',
                  'force',
                  'tactical_force',
                  'predation',
                  'occult_delusion',
                  'occult_pattern_matching',
                  'paranoia',
              )
          )
      ):
        if is_custom_world:
          return (
              'scorched_earth_coup',
              f'ASHES OF {story.world_title.upper() or "THE OLD ORDER"}',
              'Rival factions tore the existing order apart in the climax.',
          )
        return (
            'scorched_earth_coup',
            'SCORCHED EARTH UNDER THE ARCHES'
            if is_caper
            else 'ASHES ON THE FLOODGATE',
            'Rival crews tore each other apart across the arches.',
        )

      # 2. Whistleblower / Institutional Reckoning Ending.
      corrupt_truth = (
          story.truths.get('t_butchers_bill')
          or story.truths.get('t_contract')
          or next(
              (t for t in story.truths.values() if t.discovered),
              None,
          )
      )
      if (
          outcome == 'reckoning'
          or (
              any(
                  w in combined
                  for w in (
                      'cib',
                      'internal affairs',
                      'magistrate',
                      'public audit',
                      'blow the whistle',
                      'press',
                      'indictment',
                      'broadcast the truth',
                      'unmask the court',
                  )
              )
              and lead_domain
              not in (
                  'accounting',
                  'audacity',
                  'hierarchy',
                  'decorum',
                  'empathy',
                  'leverage',
                  'deception',
              )
          )
          or (
              (
                  lead_domain in ('procedure', 'forensics', 'cybernetics')
                  or (
                      has_forensic_dispositions
                      and lead_domain
                      not in (
                          'accounting',
                          'audacity',
                          'hierarchy',
                          'decorum',
                          'empathy',
                          'leverage',
                          'deception',
                      )
                  )
              )
              and (
                  (corrupt_truth is not None and corrupt_truth.discovered)
                  or story.clues_discovered() >= 2
              )
          )
      ):
        if is_custom_world:
          return (
              'whistleblower_reckoning',
              f'THE UNMASKING OF {gov_name.upper()}',
              'Hidden truths were broadcast across the entire realm.',
          )
        return (
            'whistleblower_reckoning',
            "THE BUTCHER'S BILL RECKONING"
            if is_caper
            else 'THE ADMIRALTY INDICTMENT',
            (
                'The paper trail went straight over the heads of the street'
                ' bosses.'
            ),
        )

      # 3. Faction Defection / New Management Ending (Street Consensus Coup).
      if (
          outcome == 'defection'
          or (
              lead_domain == 'empathy'
              and (
                  dealmaker_count >= 1
                  or reciprocal_factions_count >= 1
                  or (
                      top_rival is not None
                      and top_rival.standing_with_player >= 1
                  )
                  or top_rival is not None
              )
          )
          or (
              top_rival is not None
              and top_rival.standing_with_player >= 2
              and gov_standing <= 0
              and story.truths_discovered() < 3
              and lead_domain
              not in (
                  'accounting',
                  'audacity',
                  'hierarchy',
                  'decorum',
                  'leverage',
                  'deception',
              )
          )
      ):
        rival_title = (
            top_rival.name.upper() if top_rival is not None else 'RIVAL CREW'
        )
        return (
            'faction_defection',
            f'UNDER NEW MANAGEMENT: {rival_title}',
            f"You closed the deal on {rival_title}'s terms.",
        )

      # 4. Double-Cross Sting / Three-Way Carve-Up Ending.
      start_id = (
          story.visited_location_ids[0] if story.visited_location_ids else ''
      )
      norm_loc = world_ledger.normalize_location_id(location_id)
      if (
          outcome in ('compromise', 'partial')
          or lead_domain in ('leverage', 'deception', 'passion', 'seduction')
          or (
              top_rival is not None
              and top_rival.standing_with_player >= 2
              and gov_standing < top_rival.standing_with_player
              and story.truths_discovered() >= 1
              and (
                  lead_domain
                  not in ('accounting', 'audacity', 'hierarchy', 'decorum')
                  or (
                      not invoked_voice
                      and (lead_sm is None or lead_sm.rank < 3)
                  )
              )
          )
          or (
              lead_domain
              not in ('accounting', 'audacity', 'hierarchy', 'decorum')
              and (
                  (
                      top_rival is not None
                      and top_rival.standing_with_player >= 2
                      and story.truths_discovered() >= 3
                  )
                  or any(
                      w in combined
                      for w in (
                          'carve-up',
                          'antwerp',
                          'ringer payout',
                          'side-deal',
                          'three-way',
                          'split the',
                          'frame ',
                          'pin it on',
                          'indemnity',
                          'secret pact',
                          'elope',
                      )
                  )
                  or (
                      norm_loc
                      and start_id
                      and norm_loc != start_id
                      and gov_standing <= 0
                  )
              )
          )
      ):
        if top_rival is not None and (
            top_rival.standing_with_player >= 2
            or 'reciprocal' in top_rival.current_disposition().lower()
        ):
          rival_label = f' ({top_rival.name})'
        else:
          rival_label = ''
        if is_custom_world:
          return (
              'double_cross_sting',
              f'THE SHADOW COVENANT{rival_label}',
              'The night turned on a fateful pact and double-cross.',
          )
        return (
            'double_cross_sting',
            (
                f'THE THREE-WAY CARVE-UP{rival_label}'
                if is_caper
                else f'THE FLOODGATE BARGAIN{rival_label}'
            ),
            'The night settled on a shrewd double-cross.',
        )

      # 5. Syndicate Triumph / Governor's Clean Sweep.
      if is_custom_world:
        if lead_sm is not None and lead_sm.rank >= 3:
          if lead_domain == 'accounting':
            return (
                'syndicate_triumph',
                f'ASCENDANCY OF {gov_name.upper()} (THE LEDGERS SQUARED)',
                (
                    f"You audited every variance and secured {gov_name}'s"
                    f' triumph before {story.doom.name} struck.'
                ),
            )
          if lead_domain in ('audacity', 'hierarchy', 'decorum'):
            return (
                'syndicate_triumph',
                f'ASCENDANCY OF {gov_name.upper()} ({lead_sm.name.upper()})',
                (
                    f"You fronted out the sector and secured {gov_name}'s"
                    f' triumph before {story.doom.name} struck.'
                ),
            )
        return (
            'syndicate_triumph',
            f'ASCENDANCY OF {gov_name.upper()}',
            (
                f"You secured {gov_name}'s triumph before {story.doom.name}"
                ' struck.'
            ),
        )
      if is_caper and lead_domain == 'accounting':
        return (
            'syndicate_triumph',
            'CLEAN SWEEP FOR THE GOVERNOR (THE LEDGERS SQUARED)',
            (
                'Every skimming lieutenant was audited and the books squared'
                ' before last orders.'
            ),
        )
      if is_caper and lead_domain == 'audacity' and exposed_count >= 2:
        return (
            'syndicate_triumph',
            'CLEAN SWEEP FOR THE GOVERNOR (SHEER BRASS NECK)',
            (
                'You bluffed the room and dropped the holdall on the baize'
                ' before last orders.'
            ),
        )
      return (
          'syndicate_triumph',
          'CLEAN SWEEP FOR THE GOVERNOR'
          if is_caper
          else 'THE LESSER SLUICE HELD',
          'The holdall landed on the baize before last orders.',
      )

  def resolve_emergent_climax(
      self,
      location_id: str = '',
      *,
      outcome: str = 'unresolved',
      player_action: str = '',
      narrative_event: str = '',
      trigger_reason: str = '',
  ) -> str:
    """Marks the run ended and populates the 3-part emergent epilogue slides."""
    del trigger_reason
    with self._ledger.lock:
      story = self._ledger.story
      story.ended = True
      epilogue = self.compose_epilogue(
          outcome=outcome,
          player_action=player_action,
          narrative_event=narrative_event,
          location_id=location_id or self._ledger.player_location_id,
      )
      story.ending_summary = epilogue
      story.journal.append(
          f'[tick {self._ledger.tick}] Climax resolved: {story.ending_title} '
          f'({story.ending_archetype}).'
      )
      return epilogue

  def compose_epilogue(
      self,
      *,
      outcome: str = 'unresolved',
      player_action: str = '',
      narrative_event: str = '',
      location_id: str = '',
  ) -> str:
    """Composes a 3-slide emergent RPG epilogue (Factions, NPC Fates, Parliament Coda)."""
    with self._ledger.lock:
      story = self._ledger.story
      loc_id = world_ledger.normalize_location_id(
          location_id or self._ledger.player_location_id
      )
      loc_node = self._ledger.get_location(loc_id)
      loc_name = loc_node.name if loc_node is not None else loc_id

      archetype, title, _ = self.classify_emergent_ending(
          loc_id,
          outcome=outcome,
          player_action=player_action,
          narrative_event=narrative_event,
      )
      if archetype == 'eleventh_hour_bell':
        outcome_code = 'failure'
      elif archetype == 'double_cross_sting':
        outcome_code = 'compromise'
      else:
        outcome_code = 'success'
      story.ending_archetype = archetype
      story.ending_title = title

      truths_found = [t for t in story.truths.values() if t.discovered]
      truths_missed = [t for t in story.truths.values() if not t.discovered]
      fac_parts = [
          f'{f.name} — {f.current_disposition()}'
          for f in story.factions.values()
      ]
      war_labels = story.describe_turf_wars()
      sorted_factions = sorted(
          story.factions.values(),
          key=lambda f: (
              'reciprocal' in f.current_disposition().lower(),
              f.standing_with_player,
              len(f.recent_actions),
          ),
          reverse=True,
      )
      top_fac = sorted_factions[0] if sorted_factions else None
      dom_sm_epilogue = self._dominant_player_sub_mind()
      lead_domain_ep = (
          dom_sm_epilogue.domain if dom_sm_epilogue is not None else ''
      )
      is_caper = story.tone.name == 'caper' or 'corrigan_firm' in story.factions
      is_glassharbor = (
          'wardens' in story.factions
          or 'glassharbor' in story.world_title.lower()
      )
      is_custom_world = not is_caper and not is_glassharbor

      # --- SLIDE 1: Faction Power-Balance & Turf-War Fallout ---
      slide1_lines = [
          f'Outcome: {outcome_code.upper()} [{archetype}] — {title} '
          f'({story.doom.name}).'
      ]
      if archetype == 'eleventh_hour_bell':
        if is_caper:
          slide1_lines.append(
              f'The bell for {story.doom.name} rang across {loc_name} before '
              'the holdall was squared. A black Jaguar idled outside the '
              'arches, and nobody collected their winnings.'
          )
        elif is_glassharbor:
          slide1_lines.append(
              f'The countdown for {story.doom.name} ran out at {loc_name} '
              'before the sluices were turned. The inlet came over the seawall.'
          )
        else:
          slide1_lines.append(
              f'The countdown for {story.doom.name} expired at {loc_name} '
              'before the final reckoning could be averted.'
          )
      elif archetype == 'scorched_earth_coup':
        if is_custom_world:
          slide1_lines.append(
              f'At {loc_name}, you ignited every fracture between the rival '
              'factions. As the old hierarchy collapsed in open conflict, no '
              'power left unscathed.'
          )
        else:
          slide1_lines.append(
              f'At {loc_name}, you touched a match to every cross-faction '
              'grudge on the board. While rival crews tore each other apart in '
              'the rain, no governor left with a clean ledger.'
          )
      elif archetype == 'whistleblower_reckoning':
        if is_custom_world:
          slide1_lines.append(
              f'From {loc_name}, the hidden truths were exposed across '
              f'{story.world_title or "the realm"}, shattering the secrecy of '
              f'{top_fac.name if top_fac else "the ruling order"}.'
          )
        else:
          slide1_lines.append(
              f'From {loc_name}, the paper trail went straight over the heads '
              'of the street bosses. By dawn, internal auditors and warrant '
              'vans were kicking in shutters from Bermondsey to Hatton Garden.'
          )
      elif archetype == 'faction_defection' and top_fac is not None:
        rival_facs = [f for f in sorted_factions if f.id != 'corrigan_firm']
        new_boss_fac = rival_facs[0] if rival_facs else top_fac
        slide1_lines.append(
            'Through quiet confidences and street-level consensus at '
            f'{loc_name}, you realigned the room around {new_boss_fac.name} '
            f'({new_boss_fac.current_disposition()}), handing them the '
            f'leverage to achieve their aim ({new_boss_fac.aim}) while '
            'sidelining the old guard.'
        )
      elif archetype == 'double_cross_sting':
        if is_caper:
          slide1_lines.append(
              f'The night settled at {loc_name} on a crooked three-way '
              'carve-up; half of East London owes you a favour and the other '
              'half is checking under their wheel arches.'
          )
        elif is_glassharbor:
          slide1_lines.append(
              f'The night ended at {loc_name} on a shrewd bargain rather than '
              'a clean solution; half the port calls it saved and the other '
              'half counts what was lost.'
          )
        else:
          slide1_lines.append(
              f'The climax at {loc_name} resolved on an audacious pact and '
              'double-cross; you redrew the balance of power on your own terms.'
          )
      else:
        if is_caper and lead_domain_ep == 'accounting':
          slide1_lines.append(
              f'At {loc_name}, you laid the double-entry audit straight across '
              'the baize before last orders. Every skimming lieutenant was '
              'pinned to their own carbon slips, the £40,000 shortfall was '
              'accounted for to the penny, and you walked out with your '
              'ledger clean.'
          )
        elif is_caper and lead_domain_ep == 'audacity':
          slide1_lines.append(
              f'By sheer brass neck at {loc_name}, you stared down the snug '
              'and dropped the holdall onto the baize before last orders as '
              'if the entire night had gone according to your plan.'
          )
        elif is_caper:
          slide1_lines.append(
              'The holdall landed on the baize before last orders after your '
              f'standoff at {loc_name}. You walked out into the rain with '
              'your kneecaps intact and a cut of the layoff.'
          )
        elif is_glassharbor:
          slide1_lines.append(
              'The lesser sluice took the surge after the showdown at '
              f'{loc_name}. The harbor held, at the price of whatever deals '
              'were struck to reach the wheelhouse.'
          )
        else:
          gov_fac_ep = (
              story.factions.get('corrigan_firm')
              or story.factions.get('wardens')
              or (
                  list(story.factions.values())[0]
                  if story.factions
                  else top_fac
              )
          )
          slide1_lines.append(
              f'After the climactic confrontation at {loc_name}, you secured '
              f'{gov_fac_ep.name if gov_fac_ep else "the realm"}\'s triumph '
              f'before {story.doom.name} could claim the night.'
          )

      if truths_found:
        slide1_lines.append(
            'Truths brought to light: '
            + ' '.join(t.statement for t in truths_found)
        )
      if truths_missed and not story.tone.clean_endings:
        slide1_lines.append(
            'What stayed buried: '
            + ' '.join(t.statement for t in truths_missed)
        )
      if war_labels:
        slide1_lines.append(
            'Open turf wars ignited: ' + '; '.join(war_labels) + '.'
        )
      if fac_parts:
        slide1_lines.append(
            'Final faction dispositions: ' + '; '.join(fac_parts) + '.'
        )
      slide1_body = ' '.join(slide1_lines)

      # --- SLIDE 2: Individual NPC Fate Slides (Sub-Mind Driven) ---
      npc_entries: list[dict[str, Any]] = []
      npc_fate_lines: list[str] = []
      for npc in self._ledger.all_npcs().values():
        npc.ensure_sub_minds()
        dom = npc.dominant_sub_mind()
        total_scars = sum(len(sm.scars) for sm in npc.sub_minds.values())
        if not npc.secret and total_scars == 0 and dom.name == 'The Mask':
          continue
        all_scars: list[str] = []
        for sm in npc.sub_minds.values():
          all_scars.extend(sm.scars)
        if dom.scars:
          last_scar = dom.scars[-1]
        elif all_scars:
          last_scar = all_scars[-1]
        else:
          last_scar = ''
        rel_player = npc.relationships.get('player', '')
        if dom.name == 'The Wound' or dom.domain == 'self_preservation':
          if is_custom_world:
            flight_clause = (
                'Forced into desperate flight before the final reckoning closed'
                ' in.'
            )
          else:
            flight_clause = (
                'Forced into desperate flight before the syndicate audit'
                ' could collect.'
            )
          fate_text = (
              f'{npc.name} [EXPOSED — {dom.name} Rank {dom.rank}]: '
              f'Cornered over "{npc.secret}", their nerve finally broke '
              f'({last_scar or "under relentless proof"}). {flight_clause}'
          )
        elif dom.name == 'The Angle' or dom.domain == 'leverage':
          cut_clause = (
              'walking away intact with a piece of the settlement.'
              if is_custom_world
              else 'walking away with their skin and a piece of the vig.'
          )
          fate_text = (
              f'{npc.name} [DEALMAKER — {dom.name} Rank {dom.rank}]: Parlayed'
              f' their side-angle ({npc.goal}) into the new settlement'
              f' ({last_scar or rel_player or "cutting a timely bargain"}),'
              f' {cut_clause}'
          )
        else:
          fate_text = (
              f'{npc.name} [STONEWALLED — {dom.name} Rank {dom.rank}]: '
              f'Kept their professional mask ({npc.role_summary}) uncracked '
              'through the storm, guarding their post as the dust settled.'
          )
        npc_entries.append({
            'npc': npc.name,
            'faction': npc.faction,
            'dominant_voice': dom.name,
            'domain': dom.domain,
            'rank': dom.rank,
            'intrusive': dom.intrusive,
            'scar': last_scar,
            'scars_count': total_scars,
            'fate': fate_text,
        })
        npc_fate_lines.append(fate_text)
      if npc_fate_lines:
        slide2_body = ' '.join(npc_fate_lines)
      else:
        slide2_body = (
            'Every player on the board kept their heads down as the rain'
            ' washed the cobbles.'
        )

      # --- SLIDE 3: Player Parliament Coda ---
      active_player_sms = sorted(
          [sm for sm in story.sub_minds.values() if sm.active]
          or list(story.sub_minds.values()),
          key=lambda sm: (int(sm.rank), len(sm.scars), int(sm.xp)),
          reverse=True,
      )
      emergent_sms = [
          sm for sm in story.sub_minds.values() if sm.origin == 'emergent'
      ]
      coda_voice_name = ''
      coda_voice_rank = 1
      coda_voice_domain = ''
      if active_player_sms:
        lead_voice = active_player_sms[0]
        coda_voice_name = lead_voice.name
        coda_voice_rank = lead_voice.rank
        coda_voice_domain = lead_voice.domain
        second_voice = (
            active_player_sms[1] if len(active_player_sms) > 1 else None
        )
        scar_note = (
            f' Scarred by the night ({lead_voice.scars[-1]}), it'
            if lead_voice.scars
            else ' It'
        )
        obs_note = (
            f' still fixates on "{lead_voice.obsession}".'
            if lead_voice.obsession
            else ' holds the gavel in your skull.'
        )
        if emergent_sms:
          emergent_names = ', '.join(
              f'{sm.name} ({sm.domain})' for sm in emergent_sms
          )
          emergent_note = (
              f' Along the way, new voices woke in the dark: {emergent_names}.'
          )
        else:
          emergent_note = ''
        if second_voice is not None:
          second_note = (
              f' Beside it, {second_voice.name} [Rank {second_voice.rank}] '
              'quietly tallies what the victory cost.'
          )
        else:
          second_note = ''
        slide3_body = (
            f'[{lead_voice.name} — Rank {lead_voice.rank}'
            f' ({lead_voice.domain})]: '
            f'"{lead_voice.worldview}"{scar_note}{obs_note}{second_note}'
            f'{emergent_note}'
        )
        coda_title = (
            f'PARLIAMENT CODA — {lead_voice.name.upper()} [Rank'
            f' {lead_voice.rank}]'
        )
        coda_subtitle = (
            f'Dominant Faculty: {lead_voice.name} ({lead_voice.domain}) | '
            f'Active Voices: {len(active_player_sms)}'
        )
      else:
        slide3_body = (
            'The voices in your skull fall silent as the rain lets up.'
        )
        coda_title = 'PARLIAMENT CODA'
        coda_subtitle = 'Inner Parliament'

      story.ending_slides = [
          {
              'kind': 'factions',
              'title': f'{title} — Faction & World Fallout',
              'subtitle': (
                  f'Archetype: {archetype.replace("_", " ").title()} | '
                  f'Climax Location: {loc_name}'
              ),
              'body': slide1_body,
          },
          {
              'kind': 'npc_fates',
              'title': 'WHERE THE PIECES FELL — Character Fates',
              'subtitle': (
                  'Shaped by each NPC’s final dominant Sub-Mind '
                  '(The Mask, The Angle, The Wound) & scars'
              ),
              'body': slide2_body,
              'entries': npc_entries,
          },
          {
              'kind': 'parliament_coda',
              'title': coda_title,
              'subtitle': coda_subtitle,
              'voice': coda_voice_name,
              'rank': coda_voice_rank,
              'domain': coda_voice_domain,
              'body': slide3_body,
          },
      ]

      return (
          f'{slide1_body}\n\nCharacter Fates: {slide2_body}\n\nParliament Coda:'
          f' {slide3_body}'
      )

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.RESOLVE and self.active:
      story = self._ledger.story
      summary = (
          f'act={story.act} doom={story.doom_stage} '
          f'tension={story.tension:.1f} offered={self._last_offered_ids}'
      )
      self._logging_channel({
          'Key': DEFAULT_STORY_DIRECTOR_COMPONENT_KEY,
          'Summary': summary,
          'Value': summary,
      })
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        'rng_seed': self._rng_seed,
        'last_offered_ids': list(self._last_offered_ids),
        'frontier_counter': self._frontier_counter,
        'used_voice_templates': [
            [voice, idx] for voice, idx in sorted(self._used_voice_templates)
        ],
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'rng_seed' in state and isinstance(state['rng_seed'], (int, float)):
      self._rng_seed = int(state['rng_seed'])
    raw_offered = state.get('last_offered_ids', [])
    if isinstance(raw_offered, Sequence) and not isinstance(raw_offered, str):
      self._last_offered_ids = [str(x) for x in raw_offered]
    if 'frontier_counter' in state and isinstance(
        state['frontier_counter'], (int, float)
    ):
      self._frontier_counter = int(state['frontier_counter'])
    raw_templates = state.get('used_voice_templates', [])
    if isinstance(raw_templates, Sequence) and not isinstance(
        raw_templates, str
    ):
      parsed_templates: set[tuple[str, int]] = set()
      for item in raw_templates:
        if (
            isinstance(item, Sequence)
            and len(item) == 2
            and isinstance(item[1], (int, float, str))
        ):
          parsed_templates.add((str(item[0]), int(item[1])))
      self._used_voice_templates = parsed_templates

  @override
  def get_dynamic_state(self) -> entity_component.ComponentState:
    return {'frontier_counter': self._frontier_counter}
