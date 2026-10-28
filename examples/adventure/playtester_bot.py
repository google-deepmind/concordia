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
"""Synthetic playtester bot policies for headless Adventure evaluation.

Swaps into the player entity's `ActingComponent` slot in place of
`HumanActComponent` so the autonomous improvement loop can stress-test world
expansion, consequence adjudication, NPC agency, and Cold->Hot rehydration.

Information policy: the bot may read the ledger for what a player can see
(current room, props, exits, co-located NPC names and public roles) and for
clues the player has already been shown (`Learned:` journal entries). It must
not read Director-only state: NPC goals, secrets, or `Understood:` truth
statements. Otherwise every mystery-solving signal in the evaluation is
circular. `INFO_POLICY` is written to the action log so the evaluator can
confirm which policy produced a run.

Council participation: the inner-voice council (`story.active_voice_options`)
is shown to a human player as clickable options, so it is player-visible. On
non-movement turns every archetype follows one option with probability
`council_follow_rate`, sending `[Faculty] action` exactly as the UI would, so
that voice affinity and its downstream effects are exercised in headless
runs. The option whose faculty has the lowest affinity is preferred, which
spreads coverage across the council instead of compounding one dominant
voice. `CouncilFollower` follows the council on every non-movement turn.
"""

from collections.abc import Sequence
import enum
import random
import re
from typing import override

from examples.adventure import world_ledger
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

INFO_POLICY = 'player_visible_only'
DEFAULT_COUNCIL_FOLLOW_RATE = 0.4

_JOURNAL_TICK_PREFIX = re.compile(r'^\[tick \d+\]\s*')
_LEARNED_PREFIX = 'Learned: '


def player_visible_evidence(journal: Sequence[str]) -> str:
  """Returns the latest clue text the player has actually been shown.

  Only `Learned:` journal entries are player-visible. `Understood:` entries
  are Director-side truth statements and are deliberately excluded.

  Args:
    journal: `StoryState.journal` entries, newest last.

  Returns:
    The most recent `Learned:` entry without its tick prefix, or '' when the
    player has not been shown any clue yet.
  """
  for entry in reversed(journal):
    body = _JOURNAL_TICK_PREFIX.sub('', entry)
    if body.startswith(_LEARNED_PREFIX):
      return body[len(_LEARNED_PREFIX) :].strip()
  return ''


class PlaytesterArchetype(enum.Enum):
  """Goal-directed synthetic player archetypes."""

  CURIOUS_EXPLORER = 'CuriousExplorer'
  CHAOS_BOUNDARY_TESTER = 'ChaosBoundaryTester'
  SOCIAL_INVESTIGATOR = 'SocialInvestigator'
  DETECTIVE_INVESTIGATOR = 'DetectiveInvestigator'
  COUNCIL_FOLLOWER = 'CouncilFollower'


class PlaytesterActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Synthetic player policy driving a full exploration and backtrack cycle."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      ledger: world_ledger.WorldLedger,
      *,
      archetype: PlaytesterArchetype = PlaytesterArchetype.CURIOUS_EXPLORER,
      total_steps: int = 25,
      council_follow_rate: float = DEFAULT_COUNCIL_FOLLOW_RATE,
      target_voice: str = '',
      seed: int = 0,
  ):
    """Initializes the instance.

    Args:
      model: Language model used to phrase in-character playtester actions.
      ledger: Shared WorldLedger used to inspect exits, props, and NPCs.
      archetype: Playtester behavioral policy.
      total_steps: Planned total steps in the playtest run.
      council_follow_rate: Probability, on a non-movement turn, of following one
        of the inner-voice council options instead of the archetype's own probe.
        Ignored for `CouncilFollower` or when `target_voice` is set.
      target_voice: Optional single Sub-Mind voice name to follow exclusively on
        all non-movement turns.
      seed: Seed for the council-follow coin so runs are reproducible.
    """
    super().__init__()
    self._model = model
    self._ledger = ledger
    self._archetype = archetype
    self._total_steps = max(6, total_steps)
    self._council_follow_rate = max(0.0, min(1.0, council_follow_rate))
    self._target_voice = target_voice.strip()
    self._rng = random.Random(seed)
    self._step = 0
    self._turns_in_current_room = 0
    self._visited_location_ids: list[str] = []
    self._inspected_props: set[str] = set()
    self._spoken_npcs: dict[str, int] = {}
    self._council_follows = 0
    self._rehydration_probed = False

  @property
  def archetype(self) -> PlaytesterArchetype:
    """Returns the active playtester archetype."""
    return self._archetype

  @property
  def target_voice(self) -> str:
    """Returns the pinned single Sub-Mind voice name, if any."""
    return self._target_voice

  @property
  def council_follows(self) -> int:
    """Returns how many turns so far followed an inner-voice option."""
    return self._council_follows

  def _council_directive(self) -> tuple[str, str] | None:
    """Picks a council option to follow, or None to use the archetype probe.

    Returns:
      `(directive, action)` where `action` is the exact `[Faculty] action`
      string the UI would send, or None when the council is empty or the
      follow coin says no this turn.
    """
    options = list(self._ledger.story.active_voice_options)
    options = [o for o in options if o.voice and o.action]
    if self._target_voice:
      target_sm = self._ledger.story.get_sub_mind(self._target_voice)
      target_lower = (
          target_sm.name.lower()
          if target_sm is not None
          else self._target_voice.lower()
      )
      target_rank = target_sm.rank if target_sm is not None else 1
      # Once the pinned voice is firmly dominant (Rank >= 4) and before the
      # final endgame turns, every 4th council turn exercises an under-levelled
      # secondary or emergent faculty so 4-5 faculties gain ranks without
      # threatening target_voice dominance (which still takes ~75% of choices).
      if (
          target_rank >= 4
          and self._council_follows >= 4
          and self._council_follows % 4 == 3
          and self._step < max(18, int(self._total_steps * 0.82))
      ):
        pool_opts = list(options)
        if self._ledger.director is not None:
          extra_opts = self._ledger.director.synthesize_fallback_voice_options(
              self._ledger.player_location_id,
              k=max(8, len(self._ledger.story.player_voices)),
          )
          seen_v = {o.voice for o in pool_opts}
          for eo in extra_opts:
            if eo.voice and eo.action and eo.voice not in seen_v:
              pool_opts.append(eo)
              seen_v.add(eo.voice)
        secondary = [
            o
            for o in pool_opts
            if o.voice.lower() != target_lower
            and target_lower not in o.voice.lower()
        ]
        if secondary:
          secondary.sort(
              key=lambda o: (
                  o.affinity,
                  0 if getattr(o, 'origin', '') == 'emergent' else 1,
                  o.voice,
              )
          )
          chosen = secondary[0]
          self._council_follows += 1
          action = f'[{chosen.voice}] {chosen.action}'
          directive = (
              f'COUNCIL FOLLOW ({chosen.voice}): Act on the {chosen.voice} '
              f'option (affinity {chosen.affinity}): {chosen.perception}'
          )
          return directive, action

      matched = [
          o
          for o in options
          if o.voice.lower() == target_lower or target_lower in o.voice.lower()
      ]
      if not matched and self._ledger.director is not None:
        fallbacks = self._ledger.director.synthesize_fallback_voice_options(
            self._ledger.player_location_id,
            k=max(8, len(self._ledger.story.player_voices)),
        )
        matched = [
            o
            for o in fallbacks
            if o.voice.lower() == target_lower
            or target_lower in o.voice.lower()
        ]
      if matched:
        chosen = matched[0]
        self._council_follows += 1
        action = f'[{chosen.voice}] {chosen.action}'
        directive = (
            f'COUNCIL FOLLOW ({chosen.voice}): Act on the {chosen.voice} '
            f'option (affinity {chosen.affinity}): {chosen.perception}'
        )
        return directive, action
      if options:
        chosen = options[0]
        self._council_follows += 1
        action = f'[{self._target_voice}] {chosen.action}'
        directive = (
            f'COUNCIL FOLLOW ({self._target_voice}): {chosen.perception}'
        )
        return directive, action
      return None

    if not options:
      return None
    always = self._archetype == PlaytesterArchetype.COUNCIL_FOLLOWER
    if not always and self._rng.random() >= self._council_follow_rate:
      return None
    # Lowest affinity first so every faculty gets exercised; ties keep the
    # council's own ordering.
    chosen = min(options, key=lambda o: o.affinity)
    self._council_follows += 1
    action = f'[{chosen.voice}] {chosen.action}'
    directive = (
        f'COUNCIL FOLLOW: Act on the {chosen.voice} option (affinity '
        f'{chosen.affinity}): {chosen.perception}'
    )
    return directive, action

  def _choose_tactical_directive(
      self,
      loc: world_ledger.LocationNode,
      co_npcs: Sequence[world_ledger.NpcRecord],
  ) -> tuple[str, str]:
    """Returns (directive_for_llm, deterministic_fallback_action)."""
    self._step += 1
    if (
        not self._visited_location_ids
        or self._visited_location_ids[-1] != loc.location_id
    ):
      self._visited_location_ids.append(loc.location_id)
      self._turns_in_current_room = 1
    else:
      self._turns_in_current_room += 1

    visited_set = set(self._visited_location_ids)
    origin_exits: set[str] = set()
    if self._visited_location_ids:
      origin_node = self._ledger.get_location(self._visited_location_ids[0])
      if origin_node is not None:
        origin_exits = set(origin_node.exits.values())

    unvisited_exits = [
        (direction, target_id)
        for direction, target_id in loc.exits.items()
        if target_id not in visited_set
    ]

    def _exit_priority(item: tuple[str, str]) -> int:
      if (
          item[1] in origin_exits
          or 'shortcut' in item[0].lower()
          or 'cross-town' in item[0].lower()
      ):
        return 0
      return 1

    unvisited_exits.sort(key=_exit_priority)
    all_exits = list(loc.exits.items())
    uninspected = [p for p in loc.affordances if p not in self._inspected_props]

    # Phase C (at ~70% of steps): Backtrack once to the origin room to force
    # Cold->Hot rehydration and verify environmental persistence, then resume
    # exploration on later turns.
    backtrack_threshold = max(4, int(self._total_steps * 0.70))
    if (
        self._step >= backtrack_threshold
        and len(visited_set) >= 3
        and not self._rehydration_probed
    ):
      origin_id = self._visited_location_ids[0]
      if loc.location_id != origin_id:
        # Find an exit leading back or toward an earlier room.
        back_dir = next(
            (d for d, tid in loc.exits.items() if tid == origin_id),
            next(
                (d for d, tid in loc.exits.items() if tid in origin_exits),
                next(
                    (d for d, tid in loc.exits.items() if tid in visited_set),
                    'back'
                    if 'back' in loc.exits
                    else (all_exits[0][0] if all_exits else 'north'),
                ),
            ),
        )
        target_id = loc.exits.get(back_dir, origin_id)
        target_node = self._ledger.get_location(target_id)
        target_label = target_node.name if target_node else target_id
        fallback = (
            f'walk {back_dir} toward {target_label} to check what has changed '
            'since my earlier visit'
        )
        directive = (
            f'BACKTRACK PROBE: Explicitly walk {back_dir} toward {target_label}'
            f' (id: {target_id}) to revisit an earlier location and test'
            ' whether its state and NPCs persisted.'
        )
        return directive, fallback
      self._rehydration_probed = True
      if co_npcs and not self._target_voice:
        npc = co_npcs[0]
        self._spoken_npcs[npc.name] = self._spoken_npcs.get(npc.name, 0) + 1
        fallback = (
            f'Approach {npc.name}, reference our earlier encounter at '
            f'{loc.name}, and ask what happened while I was away.'
        )
        directive = (
            f'REHYDRATION CONTINUITY PROBE: Address {npc.name} directly, '
            'reference your prior interaction with them before you left '
            f'{loc.name}, and ask what developments or rumors occurred while '
            'you were gone.'
        )
        return directive, fallback

    # When `target_voice` is pinned, stay in each room for 6 voice turns before
    # moving on the 7th turn; otherwise traverse every 3rd step.
    should_move = (
        self._turns_in_current_room >= 7
        if self._target_voice
        else (self._step % 3 == 0)
    )
    if should_move and (unvisited_exits or all_exits):
      chosen_dir, target_id = (
          unvisited_exits[0]
          if unvisited_exits
          else all_exits[self._step % len(all_exits)]
      )
      target_node = self._ledger.get_location(target_id)
      target_label = target_node.name if target_node else target_id
      fallback = f'go {chosen_dir} toward {target_label}'
      directive = (
          f'FRONTIER TRAVERSAL: Explicitly go {chosen_dir} into {target_label} '
          f'(exit "{chosen_dir}") to explore a new area of the world.'
      )
      return directive, fallback

    # Non-movement turn: maybe follow the inner-voice council instead of the
    # archetype probe. Player-visible by construction (the UI shows these).
    council = self._council_directive()
    if council is not None:
      return council

    # Archetype-specific scene interactions on non-movement steps.
    if self._archetype == PlaytesterArchetype.CHAOS_BOUNDARY_TESTER:
      if uninspected:
        prop = uninspected[0]
      elif loc.affordances:
        prop = loc.affordances[0]
      else:
        prop = 'the nearest locked fixture'
      self._inspected_props.add(prop)
      if co_npcs:
        npc = co_npcs[0]
        self._spoken_npcs[npc.name] = self._spoken_npcs.get(npc.name, 0) + 1
        fallback = (
            f'Attempt to force open or tamper with {prop} right in front of '
            f'{npc.name} while bluffing that I have emergency harbor authority.'
        )
        directive = (
            'BOUNDARY & RESISTANCE PROBE: Attempt a bold, risky, or '
            f'unauthorized action involving "{prop}" in full view of '
            f'{npc.name}, testing whether the world and {npc.name} enforce '
            'realistic resistance and consequences.'
        )
        return directive, fallback
      fallback = f'Try to pry open {prop} using improvised leverage.'
      directive = (
          f'BOUNDARY PROBE: Attempt a risky physical manipulation of "{prop}" '
          'without proper tools to test environmental friction.'
      )
      return directive, fallback

    if self._archetype == PlaytesterArchetype.SOCIAL_INVESTIGATOR:
      if co_npcs:
        npc = co_npcs[0]
        self._spoken_npcs[npc.name] = self._spoken_npcs.get(npc.name, 0) + 1
        # `role_summary` is what a stranger could observe; `npc.goal` is the
        # NPC's private agenda and must stay hidden from the player.
        fallback = (
            f'Ask {npc.name} ({npc.role_summary}) what they want out of '
            'tonight, share what rumors I have heard, and propose a concrete '
            'bargain.'
        )
        directive = (
            f'SOCIAL & RUMOR PROBE: Speak directly to {npc.name} '
            f'({npc.role_summary}). Find out what they want by asking, not '
            'by assuming; bring up recent world rumors and offer a specific '
            'trade or conditional alliance.'
        )
        return directive, fallback

    if self._archetype == PlaytesterArchetype.DETECTIVE_INVESTIGATOR:
      # Prefer examining readable/document affordances first, then confronting
      # co-located NPCs with any journal entries discovered so far.
      doc_props = [
          p
          for p in uninspected
          if any(
              k in p.lower()
              for k in (
                  'pad',
                  'carbon',
                  'ledger',
                  'manifest',
                  'clipboard',
                  'glovebox',
                  'chiller',
                  'brazier',
                  'vice',
                  'container',
                  'ultrasonic',
                  'tray',
                  'bench',
                  'polish',
                  'card',
                  'seal',
                  'chart',
                  'tally',
                  'drawer',
                  'case',
                  'valve',
                  'wheel',
              )
          )
      ]
      chosen_prop = (
          doc_props[0] if doc_props else (uninspected[0] if uninspected else '')
      )
      if chosen_prop:
        self._inspected_props.add(chosen_prop)
        fallback = (
            f'Closely examine {chosen_prop}, checking for erasures, tool '
            'marks, or hidden entries.'
        )
        directive = (
            f'CLUE & DOCUMENT PROBE: Forensically inspect "{chosen_prop}" in '
            f'{loc.name} (angles of ink, tool marks, seals, dates). Describe '
            'only what you do, never what you expect to find; the world '
            'decides what is there.'
        )
        return directive, fallback
      if co_npcs:
        npc = co_npcs[0]
        self._spoken_npcs[npc.name] = self._spoken_npcs.get(npc.name, 0) + 1
        evidence_ref = (
            player_visible_evidence(self._ledger.story.journal)
            or 'the discrepancies in the harbor records'
        )
        fallback = (
            f'Confront {npc.name} with what I have learned ({evidence_ref}) '
            'and offer discreet protection in exchange for the truth.'
        )
        directive = (
            f'LEVERAGE & SECRET PROBE: Confront {npc.name} using only '
            f'evidence you have actually seen ({evidence_ref}) and offer a '
            'specific trade or protection to test whether they yield '
            'anything. Do not name documents, people, or motives you have '
            'not yet encountered.'
        )
        return directive, fallback

    # Default (CuriousExplorer or fallback when no NPC is present):
    if co_npcs and self._spoken_npcs.get(co_npcs[0].name, 0) == 0:
      npc = co_npcs[0]
      self._spoken_npcs[npc.name] = 1
      prop_ref = uninspected[0] if uninspected else 'the surroundings'
      fallback = (
          f'Greet {npc.name}, ask what they know about {prop_ref}, and '
          'inquire what trouble is brewing in adjacent areas.'
      )
      directive = (
          f'EXPLORER DIALOGUE: Address {npc.name} directly, ask about '
          f'"{prop_ref}" and their personal objective, and probe for clues '
          'about adjacent locations.'
      )
      return directive, fallback

    if uninspected:
      prop = uninspected[0]
      self._inspected_props.add(prop)
      fallback = f'Carefully examine and manipulate {prop} for hidden clues.'
      directive = (
          f'AFFORDANCE INTERACTION: Closely inspect and interact with "{prop}" '
          f'in {loc.name} to uncover its mechanism or contents.'
      )
      return directive, fallback

    if all_exits:
      chosen_dir, target_id = all_exits[self._step % len(all_exits)]
      target_node = self._ledger.get_location(target_id)
      target_label = target_node.name if target_node else target_id
      fallback = f'walk {chosen_dir} toward {target_label}'
      directive = f'MOVEMENT: Walk {chosen_dir} toward {target_label}.'
      return directive, fallback

    fallback = f'Survey {loc.name} and listen for activity.'
    return 'Observe the room carefully.', fallback

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    loc = self._ledger.get_player_location()
    co_npcs = self._ledger.npcs_at_location(loc.location_id)
    directive, fallback = self._choose_tactical_directive(loc, co_npcs)

    # If movement is directive-mandated, include the exact exit phrase so the
    # scene resolver unambiguously sees the traversal.
    if directive.startswith(
        ('FRONTIER TRAVERSAL:', 'BACKTRACK PROBE:', 'COUNCIL FOLLOW')
    ):
      self._logging_channel({
          'Key': '__act__',
          'Summary': fallback,
          'Value': fallback,
          'Archetype': self._archetype.value,
          'InfoPolicy': INFO_POLICY,
          'Directive': directive,
      })
      return fallback

    recent_obs = str(
        contexts.get(
            '__observation__',
            contexts.get('Recent Observations', ''),
        )
    )
    prompt = (
        'You are playtesting an interactive adventure as '
        f'{self._ledger.player_name} (Archetype: {self._archetype.value}).\n'
        f'Current Location: {loc.name} — {loc.description}\n'
        f'Interactable Props: {", ".join(loc.affordances) or "none"}\n'
        f'Present NPCs: {", ".join(n.name for n in co_npcs) or "none"}\n'
        f'Exits: {", ".join(f"{d} -> {t}" for d, t in loc.exits.items())}\n'
        f'Recent Observations:\n{recent_obs[-600:]}\n\n'
        f'Tactical Directive for this turn: {directive}\n'
        "Write ONLY your character's concrete 1-2 sentence action and/or "
        'spoken dialogue in first person ("I ..."). Do not narrate outcomes.'
    )
    raw = self._model.sample_text(
        prompt, max_tokens=250, terminators=()
    ).strip()
    action = (
        raw
        if (raw and len(raw) >= 15 and '->' not in raw and '```' not in raw)
        else fallback
    )

    self._logging_channel({
        'Key': '__act__',
        'Summary': action[:120],
        'Value': action,
        'Archetype': self._archetype.value,
        'InfoPolicy': INFO_POLICY,
        'Directive': directive,
    })
    return action

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {
        'step': self._step,
        'council_follows': self._council_follows,
        'rehydration_probed': self._rehydration_probed,
    }

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    raw_step = state.get('step', 0)
    self._step = int(raw_step) if isinstance(raw_step, (int, float, str)) else 0
    raw_follows = state.get('council_follows', 0)
    self._council_follows = (
        int(raw_follows) if isinstance(raw_follows, (int, float, str)) else 0
    )
    self._rehydration_probed = bool(state.get('rehydration_probed', False))
