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
"""Three-layer evaluator for Adventure runs.

Layer 1, hard invariants: deterministic architectural checks (LLM call
budgets, tier lifecycle, memory contamination) plus two content checks that
mark a run broken regardless of prose quality: a dormant NPC the player
addressed in their own room was never woken, or an inner voice stated a truth
the player had not yet earned.

Layer 2, deterministic metrics: ledger/prose fidelity (do fired storylets and
revealed clues actually appear in the resolved prose at that tick?), pacing,
beat repetition, and Warm/Cold re-entry pairs. These are computed in code and
passed to the autorater as evidence, so the LLM grades what happened rather
than what the Director recorded.

Layer 3, LLM autorater: grades `adventure_rubric.md` dimensions. Dimensions a
run did not exercise (for example consequence testing with a non-adversarial
playtester, or Cold rehydration when no met NPC was ever rehydrated) are
reported as not applicable and excluded from the weighted total instead of
being guessed.
"""

from collections.abc import Mapping, Sequence
import json
import os
import re
from typing import Any

from examples.adventure import adventure_specific_components
from examples.adventure import story_state
from concordia.language_model import language_model
from concordia.utils import structured_logging

DIMENSION_WEIGHTS: dict[str, float] = {
    'player_parliament': 0.14,
    'npc_parliament': 0.12,
    'narrative_and_dialogue_coherence': 0.12,
    'voice_combat': 0.11,
    'consequence_and_resistance': 0.11,
    'npc_agency': 0.09,
    'persistence_and_rehydration': 0.09,
    'off_screen_emergence': 0.07,
    'ledger_prose_fidelity': 0.06,
    'environmental_affordances': 0.05,
    'mystery_pacing': 0.04,
}

PLAYTEST_META_FILENAME = 'playtest_meta.json'
# Log key under which `NpcSubMindCouncilComponent` records each deliberation.
NPC_COUNCIL_COMPONENT_KEY = (
    adventure_specific_components.DEFAULT_NPC_SUBMIND_COUNCIL_COMPONENT_KEY
)
# Domain of the NPC voice that stonewalls by default ("The Mask" role).
NPC_MASK_DOMAIN = 'stonewalling'

# Archetypes that deliberately attempt risky or unauthorized actions. Only
# these runs exercise the consequence dimension.
BOUNDARY_TESTING_ARCHETYPES = frozenset({
    'ChaosBoundaryTester',
    'CouncilFollower',
})

# Keyword support thresholds. Support is the fraction of a source text's
# content words (>= 6 letters) found in the prose resolved at the same tick.
CLUE_SUPPORT_THRESHOLD = 0.34
BEAT_SUPPORT_THRESHOLD = 0.25
SOFT_GATE_FIDELITY_RATE = 0.60
VERBATIM_LEAK_MIN_WORDS = 8
BEAT_REPETITION_FLAG_FRACTION = 0.60
BEAT_REPETITION_MIN_TURNS = 4
# A voice line that carries this fraction of an undiscovered clue's or truth's
# content words is treated as handing the player the answer. Truth statements
# are long, so a lower fraction already identifies them.
VOICE_CLUE_LEAK_THRESHOLD = 0.40
VOICE_TRUTH_LEAK_THRESHOLD = 0.30
# Alternatively, a line that shares this many distinctive tokens (names,
# numbers, short nouns such as "dyed", "trap") with an undiscovered truth is a
# leak even when the long-word fraction is low.
VOICE_LEAK_MIN_DISTINCT_HITS = 5
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

_NARRATIVE_KINDS = frozenset(
    {'scene_resolution', 'npc_resolution', 'rumor', 'arrival', 'revelation'}
)
# Letters and digits, with apostrophes allowed only inside a word so that
# quoted spans ('folio', wall') tokenize the same as unquoted ones.
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
# Coarse beat classes for behavioural repetition. Vocabulary is deliberately
# broad across settings (harbour noir, gangster drama, ...); a setting-specific
# idiom that is missed here lowers recall, it does not create false flags.
_BEAT_CLASSES: dict[str, re.Pattern[str]] = {
    'threat': re.compile(
        r'\b(irons?|manacles?|brig|cells?|arrest|nick you|haul you|clap|'
        r'flay|drown|toss you|kneecaps?|break your|cut you|shiv|blade|'
        r'shoot|bullet|bury you|end you|hurt you|the river|the thames|'
        r"or i('ll| will)|or you('ll| will) (be|find|wish)|last warning)\b"
    ),
    'demand': re.compile(
        r'\b(hand (it |them )?over|turn (them|it) over|surrender|'
        r'spit (it )?out|produce|state your business|empty your|give (me|it)|'
        r"where('s| is) (the|my)|tell me (where|who)|pay (up|what)|"
        r'cough (it )?up)\b'
    ),
    'block': re.compile(
        r'\b(plants? (himself|herself|a boot|her boot|his boot)|'
        r'steps? (between|in front)|bars? (the|any|your)|blocks?|'
        r'slams? (both|a|her|his) (palms?|hands?)|pins? the|'
        r'puts? (himself|herself) between|squares? up)\b'
    ),
    'bargain_push': re.compile(
        r'\b(slides?|shoves?|pushes?|slaps?|taps?) (the |a |his |her |an )?'
        r'[a-z0-9\- ]{0,25}\b(voucher|slip|manifest|pass|ledger|key|chit|'
        r'card|mug|pad)\b|\b(in exchange for|sign (it|the)|clear my|'
        r'fudge the|fake the|get me on that|put my name on|stamp my|'
        r'my ticket out)\b'
    ),
}


class AutoraterParseError(ValueError):
  """The autorater returned output that does not satisfy the rubric schema."""


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------


def _content_words(text: str, min_len: int = 6) -> set[str]:
  return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= min_len}


def keyword_support(source_text: str, narrative_text: str) -> float:
  """Returns the fraction of `source_text` content words in `narrative_text`.

  Args:
    source_text: Authored text (clue text or storylet beat).
    narrative_text: Prose that should have expressed it.

  Returns:
    Value in [0, 1]; 0 when `source_text` has no content words.
  """
  words = _content_words(source_text)
  if not words:
    return 0.0
  haystack = narrative_text.lower()
  hits = sum(1 for w in words if w in haystack)
  return hits / len(words)


def distinctive_token_hits(source_text: str, narrative_text: str) -> int:
  """Counts distinctive tokens of `source_text` present in `narrative_text`.

  Distinctive means at least four characters or numeric, and not a stopword.
  Complements `keyword_support` for short-token leaks (names, numbers).

  Args:
    source_text: Authored clue or truth text.
    narrative_text: Line that may be restating it.

  Returns:
    Number of distinct source tokens found in the narrative.
  """
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


def directive_support(directive: str, narrative_text: str) -> float:
  """Fraction of a short directive's distinctive tokens found in the prose.

  Council directives are one-line imperatives ("Hand over the key.") with few
  six-letter words, so `keyword_support` would read 0 for a line that enacts
  them exactly. This uses the same distinctive-token rule as leak detection.

  Args:
    directive: One-turn action text urged by a winning sub-mind.
    narrative_text: The line the NPC actually produced.

  Returns:
    Value in [0, 1]; 0 when the directive has no distinctive tokens.
  """
  tokens = {
      t
      for t in set(_WORD_RE.findall(directive.lower()))
      if t not in _LEAK_STOPWORDS and (len(t) >= 4 or t.isdigit())
  }
  if not tokens:
    return 0.0
  return distinctive_token_hits(directive, narrative_text) / len(tokens)


def longest_common_word_run(text_a: str, text_b: str) -> int:
  """Returns the length of the longest shared contiguous word sequence."""
  words_a = _WORD_RE.findall(text_a.lower())
  words_b = _WORD_RE.findall(text_b.lower())
  if not words_a or not words_b:
    return 0
  best = 0
  previous_row = [0] * (len(words_b) + 1)
  for word_a in words_a:
    current_row = [0] * (len(words_b) + 1)
    for j, word_b in enumerate(words_b, start=1):
      if word_a == word_b:
        current_row[j] = previous_row[j - 1] + 1
        best = max(best, current_row[j])
    previous_row = current_row
  return best


def _ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
  words = _WORD_RE.findall(text.lower())
  return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


# -----------------------------------------------------------------------------
# Transcript extraction
# -----------------------------------------------------------------------------


def check_memory_contamination(
    entity_memories: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
  """Checks all entity memories for leaked JSON schema strings."""
  total = 0
  leaked = 0
  samples: list[str] = []
  forbidden_tokens = (
      '"narrative_event":',
      '"output_type":',
      '"call_to_action":',
      '"add_affordances":',
  )
  for entity_name, mems in entity_memories.items():
    for mem in mems:
      total += 1
      text = str(mem)
      if any(tok in text for tok in forbidden_tokens):
        leaked += 1
        if len(samples) < 5:
          samples.append(f'{entity_name}: {text[:140]}')
  rate = (leaked / total * 100.0) if total > 0 else 0.0
  return {
      'total_memories': total,
      'leaked_memories': leaked,
      'leak_rate_percent': round(rate, 3),
      'samples': samples,
  }


def _resolution_record(
    step: int, entity_name: str, res_comp: Mapping[str, Any]
) -> dict[str, Any] | None:
  """Extracts a structured resolution record from a step's resolve payload."""
  text = str(res_comp.get('Value', '')).strip()
  if not text:
    return None
  measurements = res_comp.get('Measurements')
  raw_tick = (
      measurements.get('tick') if isinstance(measurements, Mapping) else None
  )
  resolved_step = (
      int(raw_tick) if isinstance(raw_tick, (int, float)) else int(step)
  )
  tiers_raw = res_comp.get('Tiers')
  tiers: dict[str, list[str]] = {}
  if isinstance(tiers_raw, Mapping):
    for tier_name, names in tiers_raw.items():
      if isinstance(names, Sequence) and not isinstance(names, str):
        tiers[str(tier_name)] = [str(n) for n in names]
  story_delta = res_comp.get('StoryDelta')
  return {
      'step': resolved_step,
      'engine_step': int(step),
      'actor': str(res_comp.get('Actor', entity_name)),
      'location': str(res_comp.get('Location', '')),
      'text': text,
      'tiers': tiers,
      'story_delta': (
          dict(story_delta) if isinstance(story_delta, Mapping) else {}
      ),
  }


def _npc_council_record(
    step: int, entity_name: str, council: Mapping[str, Any]
) -> dict[str, Any]:
  """Extracts one NPC Parliament deliberation from an entity log entry."""
  options = [
      o for o in council.get('ActiveOptions', []) if isinstance(o, Mapping)
  ]
  raw_tick = council.get('Tick')
  resolved_step = (
      int(raw_tick) if isinstance(raw_tick, (int, float)) else int(step)
  )
  return {
      'step': resolved_step,
      'engine_step': int(step),
      'npc': str(council.get('NPC', entity_name)),
      'player_action': str(council.get('PlayerAction', '')),
      'winning_voice': str(council.get('WinningVoice', '')),
      'options': [
          {
              'voice': str(o.get('voice', '')),
              'domain': str(o.get('domain', '')),
              'rank': int(o.get('affinity', 1) or 1),
              'intrusive': bool(o.get('intrusive', False)),
              'action': str(o.get('action', ''))[:200],
          }
          for o in options
      ],
  }


def extract_transcript_from_log(
    sim_log: structured_logging.SimulationLog,
) -> dict[str, Any]:
  """Extracts actions, GM resolutions, tiers, and warm events from a log."""
  interface = structured_logging.AIAgentLogInterface(sim_log)
  overview = interface.get_overview()

  action_lines: list[str] = []
  archetype = ''
  info_policy = ''
  for action in interface.get_component_values():
    step = action.get('step', '?')
    entity_name = action.get('entity_name', '?')
    text = str(action.get('value', '')).strip()
    if text:
      action_lines.append(f'Step {step} [{entity_name}]: {text}')

  resolutions: list[dict[str, Any]] = []
  warm_lines: list[str] = []
  npc_council_turns: list[dict[str, Any]] = []
  for entry in sim_log.entries:
    full = sim_log.reconstruct_value(entry.deduplicated_data)
    value = full.get('value', {})
    if entry.entry_type == 'entity' and isinstance(value, Mapping):
      council = value.get(NPC_COUNCIL_COMPONENT_KEY)
      if isinstance(council, Mapping) and council.get('WinningVoice'):
        npc_council_turns.append(
            _npc_council_record(entry.step, entry.entity_name, council)
        )
      continue
    if entry.entry_type != 'step':
      continue
    act_comp = value.get('__act__') if isinstance(value, Mapping) else None
    if isinstance(act_comp, Mapping) and act_comp.get('Archetype'):
      archetype = archetype or str(act_comp.get('Archetype'))
      info_policy = info_policy or str(act_comp.get('InfoPolicy', ''))
    resolve = value.get('resolve', {}) if isinstance(value, Mapping) else {}
    if not isinstance(resolve, Mapping):
      continue
    res_comp = resolve.get('__resolution__') or resolve.get('FastSceneResolver')
    warm_comp = resolve.get('WarmChronicler')
    if (
        isinstance(res_comp, Mapping)
        and res_comp.get('Key') == 'WarmChronicler'
    ):
      warm_comp = res_comp
      res_comp = None
    if isinstance(res_comp, Mapping):
      record = _resolution_record(entry.step, entry.entity_name, res_comp)
      if record is not None:
        resolutions.append(record)
    if isinstance(warm_comp, Mapping):
      warm_text = str(warm_comp.get('Value', '')).strip()
      if warm_text:
        warm_meas = warm_comp.get('Measurements')
        warm_tick = (
            warm_meas.get('tick') if isinstance(warm_meas, Mapping) else None
        )
        warm_step = (
            int(warm_tick)
            if isinstance(warm_tick, (int, float))
            else entry.step
        )
        warm_lines.append(f'Step {warm_step} [Warm]: {warm_text}')

  log_dict = sim_log.to_dict()
  entity_memories = {
      str(k): [str(m) for m in v]
      for k, v in dict(log_dict.get('entity_memories', {})).items()
  }
  resolution_lines = [
      f"Step {r['step']} [GM -> {r['actor']} @ {r['location']}]: {r['text']}"
      for r in resolutions
  ]
  return {
      'overview': overview,
      'action_lines': action_lines,
      'resolution_lines': resolution_lines,
      'resolutions': resolutions,
      'warm_lines': warm_lines,
      'npc_council_turns': npc_council_turns,
      'entity_memories': entity_memories,
      'archetype': archetype,
      'info_policy': info_policy,
  }


# -----------------------------------------------------------------------------
# Deterministic metrics
# -----------------------------------------------------------------------------


def _narrative_by_tick(ledger_dict: Mapping[str, Any]) -> dict[int, list[str]]:
  by_tick: dict[int, list[str]] = {}
  for entry in ledger_dict.get('narrative_log', []):
    if not isinstance(entry, Mapping):
      continue
    if str(entry.get('kind', '')) not in _NARRATIVE_KINDS:
      continue
    by_tick.setdefault(int(entry.get('tick', -1)), []).append(
        str(entry.get('text', ''))
    )
  return by_tick


def _prose_window(
    by_tick: Mapping[int, Sequence[str]], tick: int, width: int = 2
) -> str:
  """Joins prose from `tick` through `tick + width` inclusive.

  A player resolution lands at tick t; a Warm rumor may land at t+1 and the
  responding NPC line at t+2, so the default window spans all three.

  Args:
    by_tick: Mapping from simulation tick to narrative prose lines.
    tick: Starting tick index.
    width: Number of subsequent ticks to include.

  Returns:
    Newline-joined prose across the tick window.
  """
  parts: list[str] = []
  for t in range(tick, tick + width + 1):
    parts.extend(by_tick.get(t, []))
  return '\n'.join(parts)


def _rate(numerator: int, denominator: int) -> float | None:
  if denominator <= 0:
    return None
  return round(numerator / denominator, 3)


def compute_fidelity_metrics(ledger_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Measures whether recorded story events appear in the resolved prose.

  Two failure modes are measured. A clue or storylet the Director recorded at
  tick t whose content words are absent from the prose at ticks t..t+2 is
  *unsupported*: the ledger says it happened and the player never saw it. A
  clue whose authored text appears as a long verbatim run in the prose is a
  *leak*: the resolver pasted the answer key instead of dramatising it.

  Args:
    ledger_dict: Serialized `WorldLedger`.

  Returns:
    Support rates, the offending ids with their support scores, and whether
    `fired_log` was available (older ledgers only carry fire counts).
  """
  story = ledger_dict.get('story') or {}
  by_tick = _narrative_by_tick(ledger_dict)

  clue_checks: list[dict[str, Any]] = []
  leaks: list[dict[str, Any]] = []
  for clue_id, clue in dict(story.get('clues', {})).items():
    if not isinstance(clue, Mapping) or not clue.get('discovered'):
      continue
    tick = int(clue.get('discovered_tick', -1))
    prose = _prose_window(by_tick, tick)
    support = keyword_support(str(clue.get('text', '')), prose)
    clue_checks.append(
        {'id': str(clue_id), 'tick': tick, 'support': round(support, 3)}
    )
    run = longest_common_word_run(str(clue.get('text', '')), prose)
    if run >= VERBATIM_LEAK_MIN_WORDS:
      leaks.append({'id': str(clue_id), 'tick': tick, 'shared_words': run})

  storylets = dict(story.get('storylets', {}))
  fired_log = [e for e in story.get('fired_log', []) if isinstance(e, Mapping)]
  beat_checks: list[dict[str, Any]] = []
  for fired in fired_log:
    tick = int(fired.get('tick', -1))
    storylet = storylets.get(str(fired.get('storylet_id', '')))
    if tick < 0 or not isinstance(storylet, Mapping):
      continue
    prose = _prose_window(by_tick, tick)
    support = keyword_support(str(storylet.get('beat', '')), prose)
    beat_checks.append({
        'id': str(fired.get('storylet_id', '')),
        'tick': tick,
        'source': str(fired.get('source', '')),
        'support': round(support, 3),
    })

  unsupported_clues = [
      c for c in clue_checks if c['support'] < CLUE_SUPPORT_THRESHOLD
  ]
  unadopted_beats = [
      b for b in beat_checks if b['support'] < BEAT_SUPPORT_THRESHOLD
  ]
  return {
      'clues_checked': len(clue_checks),
      'clue_support_rate': _rate(
          len(clue_checks) - len(unsupported_clues), len(clue_checks)
      ),
      'unsupported_clues': unsupported_clues,
      'clue_text_leaks': leaks,
      'storylets_checked': len(beat_checks),
      'storylet_adoption_rate': _rate(
          len(beat_checks) - len(unadopted_beats), len(beat_checks)
      ),
      'unadopted_storylets': unadopted_beats,
      'fired_log_available': bool(fired_log),
  }


def compute_pacing_metrics(ledger_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Summarises mystery progression against the doom clock and the map."""
  story = ledger_dict.get('story') or {}
  telemetry = ledger_dict.get('telemetry') or {}
  truths = [
      t
      for t in dict(story.get('truths', {})).values()
      if isinstance(t, Mapping)
  ]
  clues = [
      c for c in dict(story.get('clues', {})).values() if isinstance(c, Mapping)
  ]
  truth_ticks = sorted(
      int(t.get('discovered_tick', -1)) for t in truths if t.get('discovered')
  )
  arrival_ticks = [
      int(e.get('tick', 0))
      for e in ledger_dict.get('narrative_log', [])
      if isinstance(e, Mapping) and e.get('kind') == 'arrival'
  ]
  first_departure_tick = min(arrival_ticks) if arrival_ticks else None
  first_truth_tick = truth_ticks[0] if truth_ticks else None
  total_ticks = int(telemetry.get('tick', ledger_dict.get('tick', 0)) or 0)
  truth_before_departure = first_truth_tick is not None and (
      first_departure_tick is None or first_truth_tick < first_departure_tick
  )
  return {
      'total_ticks': total_ticks,
      'truths_total': len(truths),
      'truths_discovered': len(truth_ticks),
      'truth_discovery_ticks': truth_ticks,
      'first_truth_tick': first_truth_tick,
      'first_departure_tick': first_departure_tick,
      'truth_before_first_departure': bool(truth_before_departure),
      'clues_total': len(clues),
      'clues_discovered': sum(1 for c in clues if c.get('discovered')),
      'truths_per_100_ticks': (
          round(len(truth_ticks) * 100.0 / total_ticks, 2)
          if total_ticks
          else None
      ),
      'doom_stage': int(story.get('doom_stage', 0)),
      'act': int(story.get('act', 1)),
      'tension': float(story.get('tension', 0.0)),
      'ended': bool(story.get('ended', False)),
      'ending_archetype': str(story.get('ending_archetype', '')),
      'ending_title': str(story.get('ending_title', '')),
      'ending_slides_count': len(story.get('ending_slides', []) or []),
  }


def compute_dormant_npc_addressability(
    ledger_dict: Mapping[str, Any],
    resolutions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Finds player actions that named a co-located NPC who was never woken.

  A player action at ledger tick t is resolved at step t+1. If the action
  names an NPC whose home location is the player's current location, that NPC
  must be in the Hot tier at step t+1 or t+2. Otherwise the resolver had to
  improvise around an absent character (the "there is no runner named Pell
  here" failure).

  Args:
    ledger_dict: Serialized `WorldLedger`.
    resolutions: Resolution records from `extract_transcript_from_log`.

  Returns:
    Count of addressed NPCs, the violations, and an `ok` flag.
  """
  npcs_raw = ledger_dict.get('npcs', [])
  npc_iter = npcs_raw.values() if isinstance(npcs_raw, Mapping) else npcs_raw
  home_by_name = {
      str(n.get('name')): str(n.get('location_id', ''))
      for n in npc_iter
      if isinstance(n, Mapping) and n.get('name')
  }
  hot_by_step: dict[int, set[str]] = {}
  for record in resolutions:
    hot_by_step[int(record['step'])] = set(record['tiers'].get('hot', []))

  addressed = 0
  violations: list[dict[str, Any]] = []
  for entry in ledger_dict.get('narrative_log', []):
    if not isinstance(entry, Mapping) or entry.get('kind') != 'player_action':
      continue
    tick = int(entry.get('tick', -1))
    text = str(entry.get('text', ''))
    lowered = text.lower()
    location_id = str(entry.get('location_id', ''))
    for name, home in home_by_name.items():
      if home != location_id or name.lower() not in lowered:
        continue
      addressed += 1
      woke = any(
          name in hot_by_step.get(tick + offset, set()) for offset in (1, 2)
      )
      if not woke:
        violations.append({'tick': tick, 'npc': name, 'action': text[:160]})
  return {
      'addressed_count': addressed,
      'violations': violations,
      'ok': not violations,
  }


def compute_beat_repetition(
    resolutions: Sequence[Mapping[str, Any]], player_name: str
) -> dict[str, Any]:
  """Measures behavioural (not lexical) repetition per NPC.

  Paraphrase hides repetition from n-gram metrics; an NPC can threaten the
  player five turns running with 0.1 lexical overlap. This classifies each NPC
  turn into coarse beat classes and reports the dominant class fraction.

  Args:
    resolutions: Resolution records from `extract_transcript_from_log`.
    player_name: Name of the player entity, excluded from the analysis.

  Returns:
    Per-NPC class fractions, lexical overlap, and the flagged NPC list.
  """
  by_actor: dict[str, list[str]] = {}
  for record in resolutions:
    actor = str(record['actor'])
    if actor == player_name:
      continue
    by_actor.setdefault(actor, []).append(str(record['text']))

  per_npc: dict[str, dict[str, Any]] = {}
  flagged: list[str] = []
  for actor, turns in by_actor.items():
    fractions = {
        cls: round(
            sum(1 for t in turns if pattern.search(t.lower())) / len(turns), 3
        )
        for cls, pattern in _BEAT_CLASSES.items()
    }
    grams = [_ngrams(t) for t in turns]
    overlaps: list[float] = []
    for i in range(len(grams)):
      for j in range(i + 1, len(grams)):
        overlaps.append(
            len(grams[i] & grams[j]) / max(1, min(len(grams[i]), len(grams[j])))
        )
    cwords = [_content_words(t, min_len=5) for t in turns]
    content_overlaps: list[float] = []
    for i in range(len(cwords)):
      for j in range(i + 1, len(cwords)):
        if cwords[i] and cwords[j]:
          content_overlaps.append(
              len(cwords[i] & cwords[j])
              / max(1, min(len(cwords[i]), len(cwords[j])))
          )
    dominant = max(fractions.values()) if fractions else 0.0
    max_lex = round(max(overlaps), 3) if overlaps else 0.0
    mean_content = (
        round(sum(content_overlaps) / len(content_overlaps), 3)
        if content_overlaps
        else 0.0
    )
    per_npc[actor] = {
        'turns': len(turns),
        'beat_fractions': fractions,
        'dominant_beat_fraction': dominant,
        'max_lexical_overlap': max_lex,
        'mean_content_overlap': mean_content,
    }
    if len(turns) >= BEAT_REPETITION_MIN_TURNS and (
        dominant >= BEAT_REPETITION_FLAG_FRACTION
        or max_lex >= 0.25
        or mean_content >= 0.35
    ):
      flagged.append(actor)
  return {'per_npc': per_npc, 'flagged_npcs': flagged}


def compute_reentry_pairs(
    resolutions: Sequence[Mapping[str, Any]], player_name: str
) -> list[dict[str, Any]]:
  """Pairs each NPC's last line before leaving Hot with their first line after.

  Only NPCs who had already spoken on-screen count; a first appearance is not
  a re-entry. `kind` is 'cold' when the NPC passed through the Cold tier
  during the gap, otherwise 'warm'.

  Args:
    resolutions: Resolution records from `extract_transcript_from_log`.
    player_name: Name of the player entity.

  Returns:
    Re-entry pairs with before/after quotes, ordered by step.
  """
  last_line: dict[str, tuple[int, str]] = {}
  was_hot: dict[str, bool] = {}
  went_cold: dict[str, bool] = {}
  pairs: list[dict[str, Any]] = []
  for record in resolutions:
    tiers = record['tiers']
    hot = set(tiers.get('hot', []))
    cold = set(tiers.get('cold', []))
    actor = str(record['actor'])
    for name in hot | cold | set(tiers.get('warm', [])):
      if name in cold and name in last_line:
        went_cold[name] = True
      if name in hot and not was_hot.get(name, False) and name in last_line:
        pairs.append({
            'npc': name,
            'kind': 'cold' if went_cold.get(name) else 'warm',
            'left_hot_after_step': last_line[name][0],
            'rehydrated_step': int(record['step']),
            'before': last_line[name][1],
            'after': '',
        })
        went_cold[name] = False
      was_hot[name] = name in hot
    if actor != player_name:
      last_line[actor] = (int(record['step']), str(record['text']))
      for pair in reversed(pairs):
        if pair['npc'] == actor and not pair['after']:
          pair['after'] = str(record['text'])
          break
  return [p for p in pairs if p['after']]


def compute_applicability(
    archetype: str,
    reentry_pairs: Sequence[Mapping[str, Any]],
    voices: Mapping[str, Any] | None = None,
    parliament: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
  """Decides which rubric dimensions the run actually exercised.

  Args:
    archetype: Playtester archetype name, or '' when unknown.
    reentry_pairs: Output of `compute_reentry_pairs`.
    voices: Output of `compute_voice_metrics`; None disables the player
      parliament dimension (older runs without the council).
    parliament: Output of `compute_parliament_metrics`; None or missing sections
      disable the NPC parliament and voice combat dimensions.

  Returns:
    Per-dimension `applicable` flag and a reason the grader is shown.
  """
  applicability = {
      name: {'applicable': True, 'reason': ''} for name in DIMENSION_WEIGHTS
  }
  if voices is None or not voices.get('enabled'):
    applicability['player_parliament'] = {
        'applicable': False,
        'reason': 'Bible defines no player voices or tone disables them.',
    }
  elif not voices.get('choices_exercised'):
    applicability['player_parliament']['reason'] = (
        'No [Faculty] option was ever chosen; grade line quality, coverage, '
        'freshness, and leaks, but not rank/scar consequences.'
    )
  parliament = parliament or {}
  npc_side = parliament.get('npc') or {}
  if not npc_side.get('available'):
    applicability['npc_parliament'] = {
        'applicable': False,
        'reason': (
            'No NPC council deliberations were logged (run predates NPC '
            'Sub-Minds, or no Hot NPC acted).'
        ),
    }
  combat = parliament.get('combat') or {}
  if not combat.get('available'):
    applicability['voice_combat'] = {
        'applicable': False,
        'reason': (
            'No voice-vs-voice combat events were logged (no [Faculty] '
            'action resolved against a co-located NPC, or run predates '
            'StoryDelta logging).'
        ),
    }
  if archetype and archetype not in BOUNDARY_TESTING_ARCHETYPES:
    applicability['consequence_and_resistance'] = {
        'applicable': False,
        'reason': (
            f'Archetype {archetype} does not attempt risky or unauthorized '
            'actions, so world resistance was not tested.'
        ),
    }
  if not reentry_pairs:
    applicability['persistence_and_rehydration'] = {
        'applicable': False,
        'reason': (
            'No previously-met NPC left and re-entered the Hot tier, so '
            'continuity across re-entry was not tested.'
        ),
    }
  else:
    kinds = {p['kind'] for p in reentry_pairs}
    applicability['persistence_and_rehydration']['reason'] = (
        f'Re-entry kinds observed: {sorted(kinds)}. Grade Cold rehydration '
        'only if a cold pair exists; otherwise grade Warm re-entry continuity.'
    )
  return applicability


def _npc_sub_mind_metrics(ledger_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Summarizes NPC Sub-Mind Parliament states, shifts, scars, and councils."""
  npcs = dict(ledger_dict.get('npcs', {}))
  total_npcs = 0
  npcs_with_sub_minds = 0
  npcs_shifted_from_mask = 0
  npcs_deliberated = 0
  total_scars = 0
  summary: dict[str, dict[str, Any]] = {}
  for npc_name, npc in npcs.items():
    if not isinstance(npc, Mapping):
      continue
    total_npcs += 1
    sub_minds = dict(npc.get('sub_minds', {}))
    if not sub_minds:
      continue
    npcs_with_sub_minds += 1
    ranks = {
        str(v_name): int(sm.get('rank', 1))
        for v_name, sm in sub_minds.items()
        if isinstance(sm, Mapping)
    }
    scars_count = sum(
        len(sm.get('scars', []))
        for sm in sub_minds.values()
        if isinstance(sm, Mapping)
    )
    total_scars += scars_count
    last_winning = str(npc.get('last_winning_voice', '') or '')
    if not last_winning and ranks:
      last_winning = max(ranks.items(), key=lambda kv: kv[1])[0]
    if last_winning and last_winning != 'The Mask':
      npcs_shifted_from_mask += 1
    options = list(npc.get('active_voice_options', []))
    if options:
      npcs_deliberated += 1
    summary[str(npc_name)] = {
        'leading_voice': last_winning,
        'ranks': ranks,
        'scars_count': scars_count,
        'deliberated': bool(options),
    }
  return {
      'total_npcs': total_npcs,
      'npcs_with_sub_minds': npcs_with_sub_minds,
      'npcs_shifted_from_mask': npcs_shifted_from_mask,
      'npcs_deliberated': npcs_deliberated,
      'total_scars': total_scars,
      'per_npc': summary,
  }


def compute_voice_metrics(ledger_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Measures the inner-voice council: coverage, freshness, leaks, choices.

  Four questions. Does every faculty get a turn, or does one dominate? Are the
  lines and options new each tick, or recycled? Does any voice state a clue
  or truth before the ledger says the player earned it (the voice as
  answer-key channel)? Did the player ever invoke a faculty, and did affinity
  move as a result?

  Args:
    ledger_dict: Serialized `WorldLedger`.

  Returns:
    Metrics plus `enabled` (bible has voices and tone allows them) and
    `choices_exercised` (at least one `[Faculty]` action occurred).
  """
  npc_sub_minds = _npc_sub_mind_metrics(ledger_dict)
  story = ledger_dict.get('story') or {}
  faculties = {
      str(k): str(v) for k, v in dict(story.get('player_voices', {})).items()
  }
  tone = story.get('tone') or {}
  enabled = bool(faculties) and bool(tone.get('inner_voices', True))
  if not enabled:
    return {
        'enabled': False,
        'choices_exercised': False,
        'npc_sub_minds': npc_sub_minds,
    }

  lines: list[dict[str, Any]] = []
  for entry in ledger_dict.get('narrative_log', []):
    if isinstance(entry, Mapping) and entry.get('kind') == 'inner_voice':
      lines.append({
          'tick': int(entry.get('tick', -1)),
          'actor': str(entry.get('actor', '')),
          'text': str(entry.get('text', '')),
      })
  faculty_counts = {name: 0 for name in faculties}
  unknown_faculty_lines = 0
  for line in lines:
    if line['actor'] in faculty_counts:
      faculty_counts[line['actor']] += 1
    else:
      unknown_faculty_lines += 1
  spoken = sum(faculty_counts.values())
  max_share = max(faculty_counts.values()) / spoken if spoken else 0.0
  silent_faculties = [n for n, c in faculty_counts.items() if c == 0]
  unique_texts = {line['text'].strip().lower() for line in lines}
  duplicate_line_rate = _rate(len(lines) - len(unique_texts), len(lines))

  leaks = _voice_truth_leaks(story, lines)
  option_metrics = _voice_option_metrics(story)
  choices = [
      e for e in story.get('voice_choice_log', []) if isinstance(e, Mapping)
  ]
  choice_counts: dict[str, int] = {}
  for choice in choices:
    name = str(choice.get('voice', ''))
    choice_counts[name] = choice_counts.get(name, 0) + 1
  affinity = {
      str(k): int(v) for k, v in dict(story.get('voice_affinity', {})).items()
  }
  return {
      'enabled': True,
      'faculties': list(faculties),
      'inner_voice_lines': len(lines),
      'faculty_line_counts': faculty_counts,
      'max_faculty_share': round(max_share, 3),
      'silent_faculties': silent_faculties,
      'unknown_faculty_lines': unknown_faculty_lines,
      'duplicate_line_rate': duplicate_line_rate,
      'premature_leaks': leaks,
      'options': option_metrics,
      'choices_exercised': bool(choices),
      'choice_count': len(choices),
      'choice_counts': choice_counts,
      'final_affinity': affinity,
      'dominant_voice': (
          max(affinity.items(), key=lambda item: item[1])[0] if affinity else ''
      ),
      'npc_sub_minds': npc_sub_minds,
  }


def _voice_truth_leaks(
    story: Mapping[str, Any], lines: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
  """Finds voice lines that state a clue or truth before it was discovered."""
  leaks: list[dict[str, Any]] = []
  sources: list[tuple[str, str, str, int]] = []
  for clue_id, clue in dict(story.get('clues', {})).items():
    if isinstance(clue, Mapping):
      sources.append((
          'clue',
          str(clue_id),
          str(clue.get('text', '')),
          int(clue.get('discovered_tick', -1))
          if clue.get('discovered')
          else 10**9,
      ))
  for truth_id, truth in dict(story.get('truths', {})).items():
    if isinstance(truth, Mapping):
      sources.append((
          'truth',
          str(truth_id),
          str(truth.get('statement', '')),
          int(truth.get('discovered_tick', -1))
          if truth.get('discovered')
          else 10**9,
      ))
  for line in lines:
    for kind, source_id, text, discovered_tick in sources:
      if line['tick'] >= discovered_tick:
        continue
      threshold = (
          VOICE_CLUE_LEAK_THRESHOLD
          if kind == 'clue'
          else VOICE_TRUTH_LEAK_THRESHOLD
      )
      support = keyword_support(text, line['text'])
      hits = distinctive_token_hits(text, line['text'])
      if support >= threshold or hits >= VOICE_LEAK_MIN_DISTINCT_HITS:
        leaks.append({
            'tick': line['tick'],
            'voice': line['actor'],
            'kind': kind,
            'id': source_id,
            'support': round(support, 3),
            'distinct_hits': hits,
            'line': line['text'][:200],
        })
  return leaks


def _voice_option_metrics(story: Mapping[str, Any]) -> dict[str, Any]:
  """Freshness and diversity of the per-tick option sets."""
  option_log = [
      e for e in story.get('voice_option_log', []) if isinstance(e, Mapping)
  ]
  if not option_log:
    return {'available': False}
  total_options = 0
  llm_options = 0
  stale_options = 0
  within_set_overlaps: list[float] = []
  faculties_offered: set[str] = set()
  previous_actions: set[str] = set()
  for refresh in option_log:
    options = [o for o in refresh.get('options', []) if isinstance(o, Mapping)]
    actions = [str(o.get('action', '')).strip().lower() for o in options]
    grams = [_ngrams(a, n=3) for a in actions]
    for i, gram_a in enumerate(grams):
      for gram_b in grams[i + 1 :]:
        within_set_overlaps.append(
            len(gram_a & gram_b) / max(1, min(len(gram_a), len(gram_b)))
        )
    for option, action in zip(options, actions, strict=True):
      total_options += 1
      faculties_offered.add(str(option.get('voice', '')))
      if str(option.get('source', '')) == 'llm':
        llm_options += 1
      if action and action in previous_actions:
        stale_options += 1
    previous_actions = set(actions)
  return {
      'available': True,
      'refreshes': len(option_log),
      'total_options': total_options,
      'llm_option_rate': _rate(llm_options, total_options),
      'stale_option_rate': _rate(stale_options, total_options),
      'mean_within_set_overlap': (
          round(sum(within_set_overlaps) / len(within_set_overlaps), 3)
          if within_set_overlaps
          else 0.0
      ),
      'faculties_offered': sorted(faculties_offered),
  }


def _npc_domain_maps(
    ledger_dict: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
  """Returns, per NPC, a map from sub-mind display name to canonical domain."""
  canonical_by_archetype = {
      'facade': 'stonewalling',
      'mask': 'stonewalling',
      'persona': 'stonewalling',
      'stonewalling': 'stonewalling',
      'drive': 'leverage',
      'angle': 'leverage',
      'desire': 'leverage',
      'leverage': 'leverage',
      'wound': 'self_preservation',
      'shadow': 'self_preservation',
      'vulnerability': 'self_preservation',
      'self_preservation': 'self_preservation',
  }
  npcs_raw = ledger_dict.get('npcs', {})
  if isinstance(npcs_raw, Mapping):
    npc_items = npcs_raw.items()
  else:
    npc_items = [
        (str(n.get('name', '')), n) for n in npcs_raw if isinstance(n, Mapping)
    ]
  maps: dict[str, dict[str, str]] = {}
  for npc_name, npc in npc_items:
    if not isinstance(npc, Mapping):
      continue
    by_name: dict[str, str] = {
        'The Mask': 'stonewalling',
        'The Angle': 'leverage',
        'The Wound': 'self_preservation',
        'stonewalling': 'stonewalling',
        'leverage': 'leverage',
        'self_preservation': 'self_preservation',
    }
    raw_sub_minds = dict(npc.get('sub_minds', {}))
    cold_state = npc.get('cold_state')
    if isinstance(cold_state, Mapping):
      ctx_comps = cold_state.get('context_components')
      if isinstance(ctx_comps, Mapping):
        council_state = ctx_comps.get(NPC_COUNCIL_COMPONENT_KEY)
        if isinstance(council_state, Mapping):
          cold_sm = council_state.get('sub_minds')
          if isinstance(cold_sm, Mapping):
            for ck, cv in cold_sm.items():
              if ck not in raw_sub_minds and isinstance(cv, Mapping):
                raw_sub_minds[str(ck)] = cv
    sm_entries = [
        (str(k), sm)
        for k, sm in raw_sub_minds.items()
        if isinstance(sm, Mapping)
    ]
    for idx, (key, sub_mind) in enumerate(sm_entries):
      raw_domain = str(sub_mind.get('domain') or '').strip()
      raw_arch = str(sub_mind.get('role_archetype') or '').strip().lower()
      canonical_domain = (
          canonical_by_archetype.get(raw_domain.lower())
          or canonical_by_archetype.get(raw_arch)
          or (
              ('stonewalling', 'leverage', 'self_preservation')[idx]
              if idx < 3 and len(sm_entries) == 3
              else (raw_domain or key)
          )
      )
      by_name[str(sub_mind.get('name', key))] = canonical_domain
      by_name[str(key)] = canonical_domain
      if raw_domain:
        by_name[raw_domain] = canonical_domain
    for opt in npc.get('active_voice_options', []):
      if isinstance(opt, Mapping):
        v_name = str(opt.get('voice', '')).strip()
        o_dom = str(opt.get('domain', '')).strip()
        if v_name and v_name not in by_name and o_dom:
          by_name[v_name] = canonical_by_archetype.get(o_dom.lower(), o_dom)
    maps[str(npc_name)] = by_name
  return maps


def compute_parliament_metrics(
    ledger_dict: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
  """Measures the Sub-Mind parliaments: player sheet, NPC councils, combat.

  Player side: rank spread, scars, obsessions, intrusive voices, minted
  (custom/emergent) voices. NPC side: which inner voice won each Hot turn,
  how often the stonewalling Mask won, whether that share fell as the run
  progressed, and whether the NPC's line actually enacted the winning
  directive. Combat: outcome distribution of player-voice vs NPC-voice
  clashes recorded in the resolver's `StoryDelta`.

  Args:
    ledger_dict: Serialized `WorldLedger`.
    transcript: Output of `extract_transcript_from_log`.

  Returns:
    Nested metrics with `player`, `npc`, and `combat` sections.
  """
  story = ledger_dict.get('story') or {}
  player = _player_parliament_metrics(story)
  npc = _npc_parliament_metrics(ledger_dict, transcript)
  combat = _voice_combat_metrics(transcript)
  return {'player': player, 'npc': npc, 'combat': combat}


def _player_parliament_metrics(story: Mapping[str, Any]) -> dict[str, Any]:
  """Computes summary metrics for the player's Sub-Mind parliament."""
  sub_minds = [
      sm
      for sm in dict(story.get('sub_minds', {})).values()
      if isinstance(sm, Mapping)
  ]
  if not sub_minds:
    return {'available': False}
  ranks = {str(sm.get('name', '')): int(sm.get('rank', 1)) for sm in sub_minds}
  origins: dict[str, int] = {}
  for sm in sub_minds:
    origin = str(sm.get('origin', 'setting'))
    origins[origin] = origins.get(origin, 0) + 1
  return {
      'available': True,
      'count': len(sub_minds),
      'ranks': ranks,
      'rank_spread': max(ranks.values()) - min(ranks.values()) if ranks else 0,
      'voices_above_rank_1': sum(1 for r in ranks.values() if r > 1),
      'intrusive_voices': [
          str(sm.get('name', '')) for sm in sub_minds if sm.get('intrusive')
      ],
      'total_scars': sum(len(sm.get('scars', [])) for sm in sub_minds),
      'voices_with_obsession': sum(
          1 for sm in sub_minds if str(sm.get('obsession', '')).strip()
      ),
      'origins': origins,
  }


def _pair_npc_council_turns(
    turns: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    ledger_dict: Mapping[str, Any],
) -> list[dict[str, Any]]:
  """Pairs each NPC council turn with its resolved line and preceding player prompt."""
  player_name = str(ledger_dict.get('player_name', 'Traveler'))
  res_by_actor_step: dict[tuple[str, int], Mapping[str, Any]] = {}
  res_by_actor_engine: dict[tuple[str, int], Mapping[str, Any]] = {}
  res_by_actor_list: dict[str, list[Mapping[str, Any]]] = {}
  preceding_player_by_res_id: dict[int, str] = {}

  last_player_text = ''
  for idx, record in enumerate(resolutions):
    actor = str(record['actor'])
    step = int(record['step'])
    eng = int(record.get('engine_step', step))
    res_by_actor_step[(actor, step)] = record
    res_by_actor_engine[(actor, eng)] = record
    if actor == player_name:
      last_player_text = str(record['text'])
    else:
      res_by_actor_list.setdefault(actor, []).append(record)
      if last_player_text:
        preceding_player_by_res_id[idx] = last_player_text

  # Map narrative_log npc_resolution text -> preceding player_action.
  player_action_by_npc_line: dict[str, str] = {}
  recent_player_act = ''
  for entry in ledger_dict.get('narrative_log', []):
    if not isinstance(entry, Mapping):
      continue
    kind = str(entry.get('kind', ''))
    text = str(entry.get('text', '')).strip()
    if kind == 'player_action' and text:
      recent_player_act = text
    elif kind == 'npc_resolution' and text and recent_player_act:
      player_action_by_npc_line[text[:120]] = recent_player_act

  actor_seq_idx: dict[str, int] = {}
  paired: list[dict[str, Any]] = []
  for turn in turns:
    npc = str(turn['npc'])
    step = int(turn['step'])
    eng = int(turn.get('engine_step', step))
    seq_i = actor_seq_idx.get(npc, 0)
    actor_seq_idx[npc] = seq_i + 1

    matched_res = (
        res_by_actor_step.get((npc, step))
        or res_by_actor_step.get((npc, step + 1))
        or res_by_actor_engine.get((npc, eng))
        or res_by_actor_engine.get((npc, eng + 1))
    )
    if matched_res is None:
      npc_res_list = res_by_actor_list.get(npc, [])
      if seq_i < len(npc_res_list):
        matched_res = npc_res_list[seq_i]
    if matched_res is None and resolutions:
      # Skip off-screen Warm turns that have no foreground npc_resolution.
      continue

    npc_line = str(matched_res['text']) if matched_res is not None else ''
    resolved_tick = (
        int(matched_res['step']) if matched_res is not None else step
    )
    player_prompt = str(turn.get('player_action', '')).strip()
    if not player_prompt and npc_line:
      player_prompt = player_action_by_npc_line.get(npc_line[:120], '')
    if not player_prompt and matched_res is not None:
      for idx, r in enumerate(resolutions):
        if r is matched_res:
          player_prompt = preceding_player_by_res_id.get(idx, '')
          break
    paired.append({
        **dict(turn),
        'resolved_tick': resolved_tick,
        'npc_line': npc_line,
        'player_prompt': player_prompt,
    })
  return paired


def _npc_parliament_metrics(
    ledger_dict: Mapping[str, Any], transcript: Mapping[str, Any]
) -> dict[str, Any]:
  """Computes summary metrics for NPC Sub-Mind councils across turns."""
  domain_maps = _npc_domain_maps(ledger_dict)
  raw_turns = list(transcript.get('npc_council_turns', []))
  resolutions = list(transcript.get('resolutions', []))
  if not raw_turns:
    return {'available': False, 'npcs_with_sub_minds': len(domain_maps)}

  paired_turns = _pair_npc_council_turns(raw_turns, resolutions, ledger_dict)

  per_npc: dict[str, dict[str, Any]] = {}
  mask_flags: list[tuple[int, bool]] = []
  supports: list[float] = []
  prior_directives_by_npc: dict[str, list[str]] = {}
  prior_lines_by_npc: dict[str, list[str]] = {}
  stale_directive_flags: list[bool] = []
  unresponsive_flags: list[bool] = []

  for turn in paired_turns:
    npc = str(turn['npc'])
    win_voice = str(turn['winning_voice'])
    opt_domain = next(
        (
            str(o.get('domain', '')).strip()
            for o in turn.get('options', [])
            if isinstance(o, Mapping) and str(o.get('voice', '')) == win_voice
        ),
        '',
    )
    domain = (
        domain_maps.get(npc, {}).get(win_voice)
        or (
            domain_maps.get(npc, {}).get(opt_domain, opt_domain)
            if opt_domain
            else ''
        )
        or win_voice
    )
    is_mask = domain == NPC_MASK_DOMAIN
    mask_flags.append((int(turn['resolved_tick']), is_mask))
    stats = per_npc.setdefault(
        npc,
        {
            'turns': 0,
            'winning_domains': {},
            'mask_share': 0.0,
            'dominant_voice_share': 0.0,
        },
    )
    stats['turns'] += 1
    stats['winning_domains'][domain] = (
        stats['winning_domains'].get(domain, 0) + 1
    )
    winning_action = next(
        (
            o['action']
            for o in turn['options']
            if o['voice'] == turn['winning_voice']
        ),
        '',
    )
    npc_line = str(turn.get('npc_line', ''))
    player_prompt = str(turn.get('player_prompt', ''))
    if winning_action and npc_line:
      supports.append(directive_support(winning_action, npc_line))

    prev_directives = prior_directives_by_npc.setdefault(npc, [])
    is_stale_dir = False
    if winning_action and prev_directives:
      is_stale_dir = any(
          directive_support(prev_d, winning_action) >= 0.25
          or directive_support(winning_action, prev_d) >= 0.25
          or distinctive_token_hits(prev_d, winning_action) >= 2
          for prev_d in prev_directives
      )
      stale_directive_flags.append(is_stale_dir)
    if winning_action:
      prev_directives.append(winning_action)

    prev_lines = prior_lines_by_npc.setdefault(npc, [])
    if npc_line and player_prompt and prev_lines:
      repeats_prior_line = is_stale_dir or any(
          keyword_support(prev_l, npc_line) >= 0.25
          or distinctive_token_hits(prev_l, npc_line) >= 5
          for prev_l in prev_lines
      )
      clean_prompt = re.sub(r'^\[[^\]]+\]\s*', '', player_prompt).strip()
      core_prompt = re.split(
          r'\b(?:before|and watch)\b', clean_prompt, maxsplit=1
      )[0].strip()
      prompt_support = directive_support(core_prompt or clean_prompt, npc_line)
      unresponsive_flags.append(
          bool(repeats_prior_line and prompt_support < 0.20)
      )
    if npc_line:
      prev_lines.append(npc_line)

  for stats in per_npc.values():
    stats['mask_share'] = round(
        stats['winning_domains'].get(NPC_MASK_DOMAIN, 0) / stats['turns'], 3
    )
    stats['dominant_voice_share'] = round(
        max(stats['winning_domains'].values()) / stats['turns'], 3
    )

  mask_flags.sort()
  half = len(mask_flags) // 2
  first = [m for _, m in mask_flags[:half]] if half else []
  second = [m for _, m in mask_flags[half:]]
  overall_mask_share = (
      (sum(1 for _, m in mask_flags if m) / len(mask_flags))
      if mask_flags
      else 0.0
  )
  return {
      'available': True,
      'npcs_with_sub_minds': len(domain_maps),
      'council_turns': len(paired_turns),
      'overall_mask_share': round(overall_mask_share, 3),
      'mask_share_first_half': (
          round(sum(first) / len(first), 3) if first else None
      ),
      'mask_share_second_half': (
          round(sum(second) / len(second), 3) if second else None
      ),
      'npcs_never_leaving_mask': [
          n
          for n, st in per_npc.items()
          if st['turns'] >= 3 and st['mask_share'] >= 0.999
      ],
      'npcs_with_voice_monopoly': [
          n
          for n, st in per_npc.items()
          if st['turns'] >= 4 and st['dominant_voice_share'] >= 0.80
      ],
      'stale_directive_rate': (
          round(sum(stale_directive_flags) / len(stale_directive_flags), 3)
          if stale_directive_flags
          else 0.0
      ),
      'unresponsive_turn_rate': (
          round(sum(unresponsive_flags) / len(unresponsive_flags), 3)
          if unresponsive_flags
          else 0.0
      ),
      'winning_action_support_mean': (
          round(sum(supports) / len(supports), 3) if supports else None
      ),
      'per_npc': per_npc,
  }


def _voice_combat_metrics(transcript: Mapping[str, Any]) -> dict[str, Any]:
  """Computes summary metrics for player vs. NPC Sub-Mind clashes."""
  events: list[dict[str, Any]] = []
  for record in transcript.get('resolutions', []):
    delta = record.get('story_delta') or {}
    for event in delta.get('npc_voice_combat') or []:
      if isinstance(event, Mapping):
        events.append({'step': int(record['step']), **dict(event)})
  if not events:
    return {'available': False}
  outcomes: dict[str, int] = {}
  voices: dict[str, int] = {}
  for event in events:
    outcome = str(event.get('outcome', ''))
    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    voice = str(event.get('player_voice', ''))
    voices[voice] = voices.get(voice, 0) + 1
  total = len(events)
  decisive = outcomes.get('angle_spiked', 0) + outcomes.get('wound_spiked', 0)
  return {
      'available': True,
      'events': total,
      'outcomes': outcomes,
      'decisive_rate': round(decisive / total, 3),
      'stonewalled_rate': round(outcomes.get('mask_vindicated', 0) / total, 3),
      'player_voices_used': voices,
      'single_outcome_only': len(outcomes) == 1 and total >= 4,
      'sample': events[:12],
  }


def compute_narrative_coherence_metrics(
    ledger_dict: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
  """Audits plot, dialogue, and syntax coherence for mechanical or structural defects.

  Detects concrete failure modes that break fiction immersion:
  1. Duplicated actor prefixes or stuttered verbs ('Jonas Reed Jonas...', 'slam
  slammed').
  2. Raw prompt tags ('Answer:', 'Response:', 'Action:') leaked into dialogue.
  3. Mechanical spliced gerund prefixes ('Bristling as...', 'Countering ...
  press over').
  4. Broken fallback action syntax ('the room's', 'across turn 22', 'the
  Mercer's').
  5. Truncated or bare-name Warm events and scars (<=2 words or unbalanced
  parens).
  6. Ghost/duplicate location nodes created with overlapping names.

  Args:
    ledger_dict: Serialized `WorldLedger`.
    transcript: Output of `extract_transcript_from_log`.

  Returns:
    Defect lists by category, `total_defect_count`, and `defect_free`.
  """
  player_name = str(ledger_dict.get('player_name', 'Traveler'))
  npcs_raw = ledger_dict.get('npcs', {})
  npc_names: list[str] = []
  if isinstance(npcs_raw, Mapping):
    npc_names = [str(k) for k in npcs_raw.keys() if k]
  elif isinstance(npcs_raw, Sequence):
    npc_names = [
        str(n.get('name'))
        for n in npcs_raw
        if isinstance(n, Mapping) and n.get('name')
    ]

  duplicated_prefixes: list[str] = []
  raw_prompt_tags: list[str] = []
  spliced_gerunds: list[str] = []
  broken_syntax: list[str] = []
  truncated_or_bare: list[str] = []

  tag_re = re.compile(
      r'(?:^|\]:\s*)(?:[A-Z][a-zA-Z\.\'\-'
      r' ]+\s+)?(?:Answer|Response|Action)\s*:',
      re.IGNORECASE,
  )
  spliced_re = re.compile(
      r'\b(?:Bristling as|Countering [A-Za-z0-9_\-\' ]+ press over|'
      r'Pushing back as|Unmoved as|Refusing to yield as)\b',
      re.IGNORECASE,
  )
  broken_action_re = re.compile(
      r'\b(?:the room[\'’]s|across turn \d+|demand turn \d+|'
      r'inspect the (?:knock|rap|bang|pound|confront|demand|press)|'
      r'and spent and)\b',
      re.IGNORECASE,
  )
  npc_name_tokens: set[str] = {
      'Mercer',
      'Vance',
      'Okafor',
      'Kaelen',
      'Lian',
      'Soren',
      'Slater',
      'Frankie',
      'Desmond',
      'Baptiste',
      'Reuben',
      'Vane',
      'Finch',
      'Ronald',
      'Carla',
      'Kovacs',
  }
  title_stop = {
      'det',
      'insp',
      'dr',
      'capt',
      'cmdr',
      'lt',
      'sgt',
      'the',
      'sir',
      'lord',
      'lady',
      'prof',
      'moat',
      'nine',
      'pint',
      'alderman',
  }
  for full_name in npc_names:
    for raw_tok in re.findall(r'[A-Z][a-z]{2,}', full_name):
      if raw_tok.lower() not in title_stop:
        npc_name_tokens.add(raw_tok)
  proper_possessive_re = re.compile(
      r'\bthe\s+(?:'
      + '|'.join(re.escape(t) for t in sorted(npc_name_tokens))
      + r')[\'’]s\b'
  )

  def _check_npc_prefix(actor: str, text: str, label: str) -> None:
    clean_actor = actor.strip()
    if not clean_actor or clean_actor == player_name:
      return
    stripped = text.strip()
    if stripped.lower().startswith(clean_actor.lower()):
      rest = stripped[len(clean_actor) :].lstrip(' :')
      if rest.lower().startswith(clean_actor.lower()):
        duplicated_prefixes.append(f'{label}: {stripped[:120]}')
        return
      for tok in clean_actor.split():
        if len(tok) >= 3 and rest.lower().startswith(
            (tok.lower() + ' ', tok.lower() + "'", tok.lower() + '’')
        ):
          duplicated_prefixes.append(f'{label}: {stripped[:120]}')
          return
      words = rest.split()
      if len(words) >= 2:
        w0 = re.sub(r'[^a-z]', '', words[0].lower())
        w1 = re.sub(r'[^a-z]', '', words[1].lower())
        if len(w0) >= 4 and len(w1) >= 4 and (w0 == w1 or w1.startswith(w0)):
          duplicated_prefixes.append(f'{label}: {stripped[:120]}')

  for record in transcript.get('resolutions', []):
    if not isinstance(record, Mapping):
      continue
    step = record.get('step', '?')
    actor = str(record.get('actor', ''))
    text = str(record.get('text', '')).strip()
    label = f'step {step} [{actor}]'
    _check_npc_prefix(actor, text, label)
    if tag_re.search(text):
      raw_prompt_tags.append(f'{label}: {text[:120]}')
    if spliced_re.search(text):
      spliced_gerunds.append(f'{label}: {text[:120]}')

  for act_line in transcript.get('action_lines', []):
    line_str = str(act_line).strip()
    if tag_re.search(line_str):
      raw_prompt_tags.append(line_str[:120])
    if spliced_re.search(line_str):
      spliced_gerunds.append(line_str[:120])
    m = re.fullmatch(r'Step\s+(\S+)\s+\[([^\]]+)\]:\s*(.+)', line_str)
    if m:
      step_id, ent_name, body = m.group(1), m.group(2), m.group(3)
      _check_npc_prefix(ent_name, body, f'action step {step_id} [{ent_name}]')
      if ent_name == player_name:
        if broken_action_re.search(body) or proper_possessive_re.search(body):
          broken_syntax.append(f'step {step_id}: {body[:120]}')

  story = ledger_dict.get('story') or {}
  for refresh in story.get('voice_option_log', []):
    if not isinstance(refresh, Mapping):
      continue
    tick = refresh.get('tick', '?')
    for opt in refresh.get('options', []):
      if not isinstance(opt, Mapping):
        continue
      act_text = str(opt.get('action', '')).strip()
      if broken_action_re.search(act_text) or proper_possessive_re.search(
          act_text
      ):
        broken_syntax.append(
            f"tick {tick} [{opt.get('voice', '?')}]: {act_text[:120]}"
        )

  for warm_line in transcript.get('warm_lines', []):
    body = re.sub(r'^Step\s+\S+\s+\[Warm\]:\s*', '', str(warm_line)).strip()
    if not body:
      continue
    words = body.split()
    if len(words) <= 2 or body.rstrip(' .,:;') in npc_names:
      truncated_or_bare.append(f'warm: {body[:120]}')
    elif body.count('(') != body.count(')'):
      truncated_or_bare.append(f'warm (unbalanced parens): {body[:120]}')

  for sm in dict(story.get('sub_minds', {})).values():
    if isinstance(sm, Mapping):
      for scar in sm.get('scars', []):
        s_str = str(scar)
        if s_str.count('(') != s_str.count(')'):
          truncated_or_bare.append(
              f"player scar [{sm.get('name')}]: {s_str[:120]}"
          )

  loc_stopwords = frozenset({
      'the',
      'and',
      'room',
      'area',
      'sector',
      'corridor',
      'hall',
      'chamber',
      'office',
      'station',
      'level',
      'deck',
      'north',
      'south',
      'east',
      'west',
      'upper',
      'lower',
      'main',
      'central',
      'inner',
      'outer',
      'wing',
      'bay',
      'zone',
      'floor',
      'yard',
      'lane',
      'street',
      'arch',
  })
  locations = dict(ledger_dict.get('locations', {}))
  loc_ids = [str(k) for k in locations.keys() if k]
  loc_tokens: dict[str, set[str]] = {}
  for lid in loc_ids:
    loc_tokens[lid] = {
        w
        for w in re.findall(r'[a-z0-9]+', lid.lower())
        if len(w) >= 4 and w not in loc_stopwords
    }
  ghost_locations: list[str] = []
  for i, lid_a in enumerate(loc_ids):
    toks_a = loc_tokens[lid_a]
    if len(toks_a) < 2:
      continue
    for lid_b in loc_ids[i + 1 :]:
      toks_b = loc_tokens[lid_b]
      if len(toks_b) < 2:
        continue
      overlap = len(toks_a & toks_b) / min(len(toks_a), len(toks_b))
      if overlap >= 0.60:
        ghost_locations.append(f'{lid_a} <-> {lid_b}')

  total_defects = (
      len(duplicated_prefixes)
      + len(raw_prompt_tags)
      + len(spliced_gerunds)
      + len(broken_syntax)
      + len(truncated_or_bare)
      + len(ghost_locations)
  )
  return {
      'duplicated_actor_prefixes': duplicated_prefixes[:10],
      'raw_prompt_tags': raw_prompt_tags[:10],
      'spliced_gerund_prefixes': spliced_gerunds[:10],
      'broken_action_syntax': broken_syntax[:10],
      'truncated_or_bare_lines': truncated_or_bare[:10],
      'ghost_locations': ghost_locations[:10],
      'total_defect_count': total_defects,
      'defect_free': total_defects == 0,
  }


def compute_spoiler_metrics(ledger_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Checks tick-0 player-facing text against the run's truths.

  The role briefing, opening narration, and each truth's `question` and
  `stakes` are visible before any clue is found. Each is checked with
  `story_state.truth_leaks_in_text` using a public vocabulary rebuilt from
  the ledger (premise, first visited room, NPC names/roles/traits).

  Args:
    ledger_dict: Serialized `WorldLedger`.

  Returns:
    `findings` (leak/warn entries labelled by text) and `has_leak`.
  """
  story_raw = ledger_dict.get('story') or {}
  truths = [
      story_state.Truth.from_dict({'id': str(tid), **dict(raw)})
      for tid, raw in dict(story_raw.get('truths', {})).items()
      if isinstance(raw, Mapping)
  ]
  if not truths:
    return {'findings': [], 'has_leak': False, 'checked_texts': []}
  # Only the fields `get_player_role_info` reads; a full `StoryState` parse
  # is unnecessary and brittle against partial ledgers.
  story = story_state.StoryState(
      truths={t.id: t for t in truths},
      world_title=str(story_raw.get('world_title', '')),
      world_premise=str(story_raw.get('world_premise', '')),
      world_opening=str(story_raw.get('world_opening', '')),
  )

  parts = [story.world_premise]
  visited = [str(v) for v in story_raw.get('visited_location_ids', [])]
  start_id = (
      visited[0] if visited else str(ledger_dict.get('player_location_id', ''))
  )
  start = dict(ledger_dict.get('locations', {})).get(start_id) or {}
  if isinstance(start, Mapping):
    parts.extend([
        str(start.get('name', '')),
        str(start.get('description', '')),
        ' '.join(str(a) for a in start.get('affordances', [])),
    ])
  npcs_raw = ledger_dict.get('npcs', {})
  npc_iter = npcs_raw.values() if isinstance(npcs_raw, Mapping) else npcs_raw
  for npc in npc_iter:
    if isinstance(npc, Mapping):
      parts.extend([
          str(npc.get('name', '')),
          str(npc.get('role_summary', '')),
          str(npc.get('traits', '')),
      ])
  vocabulary = story_state.distinctive_tokens(' '.join(parts))

  role = story.get_player_role_info()
  texts = {
      'role_briefing': f"{role.get('title', '')}. {role.get('summary', '')}",
      'opening': story.world_opening,
  }
  for truth in truths:
    if truth.question:
      texts[f'truth[{truth.id}].question'] = truth.question
    if truth.stakes:
      texts[f'truth[{truth.id}].stakes'] = truth.stakes
  findings: list[dict[str, Any]] = []
  for label, text in texts.items():
    for finding in story_state.truth_leaks_in_text(text, truths, vocabulary):
      findings.append({'text': label, **finding})
  return {
      'findings': findings,
      'has_leak': any(f['severity'] == 'leak' for f in findings),
      'checked_texts': sorted(texts),
  }


def compute_deterministic_metrics(
    *,
    ledger_dict: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
  """Runs every deterministic metric and returns them keyed by name."""
  player_name = str(ledger_dict.get('player_name', 'Traveler'))
  resolutions = list(transcript.get('resolutions', []))
  return {
      'fidelity': compute_fidelity_metrics(ledger_dict),
      'pacing': compute_pacing_metrics(ledger_dict),
      'dormant_npc_addressability': compute_dormant_npc_addressability(
          ledger_dict, resolutions
      ),
      'beat_repetition': compute_beat_repetition(resolutions, player_name),
      'reentry_pairs': compute_reentry_pairs(resolutions, player_name),
      'voices': compute_voice_metrics(ledger_dict),
      'parliament': compute_parliament_metrics(ledger_dict, transcript),
      'spoilers': compute_spoiler_metrics(ledger_dict),
      'narrative_coherence': compute_narrative_coherence_metrics(
          ledger_dict, transcript
      ),
  }


# -----------------------------------------------------------------------------
# Layer 1: hard invariants
# -----------------------------------------------------------------------------


def evaluate_hard_invariants(
    *,
    telemetry: Mapping[str, Any],
    latency: Mapping[str, Any],
    entity_memories: Mapping[str, Sequence[str]],
    deterministic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Evaluates deterministic architectural and content invariants.

  Args:
    telemetry: `WorldLedger.telemetry_summary()` output.
    latency: `LatencyTracker.summary()` output.
    entity_memories: Final memories per entity.
    deterministic: Output of `compute_deterministic_metrics`. When None (for
      example when called from the CLI on a live ledger without a structured
      log), the content invariants are skipped.

  Returns:
    Per-check booleans, `all_passed`, and the supporting audits.
  """
  mem_audit = check_memory_contamination(entity_memories)
  control_calls = int(latency.get('control_hook_llm_calls', 0))
  max_resolve_calls = int(latency.get('max_llm_calls_per_player_resolve', 1))
  hit_rate = float(telemetry.get('frontier_cache_hit_rate', 1.0))
  hibernations = int(telemetry.get('hibernations_count', 0))
  rehydrations = int(telemetry.get('rehydrations_count', 0))
  warm_events = int(telemetry.get('warm_events_count', 0))
  director_calls = int(telemetry.get('director_llm_calls', 0))

  checks = {
      'zero_control_hook_llm_calls': control_calls == 0,
      'zero_director_llm_calls': director_calls == 0,
      'single_call_player_resolve': max_resolve_calls <= 1,
      'frontier_cache_hit_rate_ok': hit_rate >= 0.80,
      'full_tier_cycle_ok': hibernations >= 1 and rehydrations >= 1,
      'off_screen_liveness_ok': warm_events >= 1,
      'zero_memory_contamination': mem_audit['leaked_memories'] == 0,
  }
  architectural_checks = dict(checks)
  content_checks: dict[str, bool] = {}
  if deterministic is not None:
    addressability = deterministic.get('dormant_npc_addressability', {})
    content_checks['dormant_npc_addressability_ok'] = bool(
        addressability.get('ok', True)
    )
    voices = deterministic.get('voices', {})
    if voices.get('enabled'):
      content_checks['no_inner_voice_truth_leaks'] = not any(
          leak.get('kind') == 'truth'
          for leak in voices.get('premature_leaks', [])
      )
    spoilers = deterministic.get('spoilers')
    if spoilers is not None:
      content_checks['no_briefing_truth_leaks'] = not bool(
          spoilers.get('has_leak', False)
      )
    checks.update(content_checks)
  return {
      'all_passed': all(checks.values()),
      'architectural_passed': all(architectural_checks.values()),
      'content_passed': all(content_checks.values()),
      'checks': checks,
      'memory_audit': mem_audit,
      'telemetry': dict(telemetry),
      'latency': dict(latency),
  }


# -----------------------------------------------------------------------------
# Layer 3: LLM autorater
# -----------------------------------------------------------------------------


def _format_reentry_pairs(pairs: Sequence[Mapping[str, Any]]) -> str:
  """Formats NPC Hot-tier exit and re-entry pairs for the autorater prompt."""
  if not pairs:
    return 'None. No previously-met NPC re-entered the Hot tier.'
  lines: list[str] = []
  for pair in pairs:
    lines.append(
        f"- {pair['npc']} ({pair['kind']} re-entry; last Hot line at step "
        f"{pair['left_hot_after_step']}, rehydrated step "
        f"{pair['rehydrated_step']})\n"
        f"    BEFORE: {pair['before'][:500]}\n"
        f"    AFTER:  {pair['after'][:500]}"
    )
  return '\n'.join(lines)


def _format_fired_beats(
    ledger_dict: Mapping[str, Any], fidelity: Mapping[str, Any]
) -> str:
  """Formats fired storylet beats and their prose support status."""
  story = ledger_dict.get('story') or {}
  storylets = dict(story.get('storylets', {}))
  support_by_key = {
      (b['id'], b['tick']): b['support']
      for b in fidelity.get('unadopted_storylets', [])
  }
  lines: list[str] = []
  for fired in story.get('fired_log', []):
    if not isinstance(fired, Mapping):
      continue
    storylet_id = str(fired.get('storylet_id', ''))
    tick = int(fired.get('tick', -1))
    storylet = storylets.get(storylet_id)
    if not isinstance(storylet, Mapping):
      continue
    status = (
        f'UNSUPPORTED (support={support_by_key[(storylet_id, tick)]})'
        if (storylet_id, tick) in support_by_key
        else 'supported'
    )
    lines.append(
        f'- tick {tick} {storylet_id} [{status}]: '
        f"{str(storylet.get('beat', ''))[:300]}"
    )
  if not lines:
    return 'No per-tick storylet record available for this run.'
  return '\n'.join(lines)


def _format_voice_lines(ledger_dict: Mapping[str, Any]) -> str:
  lines: list[str] = []
  for entry in ledger_dict.get('narrative_log', []):
    if isinstance(entry, Mapping) and entry.get('kind') == 'inner_voice':
      lines.append(
          f"- tick {int(entry.get('tick', -1))}: "
          f"{str(entry.get('text', ''))[:240]}"
      )
  return '\n'.join(lines[-60:]) or 'None'


def _format_voice_options(ledger_dict: Mapping[str, Any]) -> str:
  """Formats recent inner-voice council option refreshes for grading."""
  story = ledger_dict.get('story') or {}
  option_log = [
      e for e in story.get('voice_option_log', []) if isinstance(e, Mapping)
  ]
  if not option_log:
    return 'No per-tick option record available for this run.'
  blocks: list[str] = []
  for refresh in option_log[-6:]:
    rows = [
        f"    [{o.get('source', '?')}] {o.get('voice', '?')}: "
        f"{str(o.get('perception', ''))[:120]} -> "
        f"{str(o.get('action', ''))[:120]}"
        for o in refresh.get('options', [])
        if isinstance(o, Mapping)
    ]
    blocks.append(
        f"- tick {int(refresh.get('tick', -1))} @ "
        f"{refresh.get('location_id', '?')}\n"
        + '\n'.join(rows)
    )
  return '\n'.join(blocks)


def _format_player_parliament(ledger_dict: Mapping[str, Any]) -> str:
  """Formats the player's Sub-Mind parliament for the LLM judge prompt."""
  story = ledger_dict.get('story') or {}
  rows: list[str] = []
  for sm in dict(story.get('sub_minds', {})).values():
    if not isinstance(sm, Mapping):
      continue
    scars = [str(x) for x in sm.get('scars', [])][-2:]
    flags = []
    if sm.get('intrusive'):
      flags.append('INTRUSIVE')
    if str(sm.get('origin', 'setting')) != 'setting':
      flags.append(str(sm.get('origin')))
    flag_text = ', ' + ', '.join(flags) if flags else ''
    rows.append(
        f"- {sm.get('name')} [rank {sm.get('rank', 1)}, "
        f"{sm.get('domain', '?')}{flag_text}]"
        f" obsession: {str(sm.get('obsession', '')) or '-'} | "
        f"blind spot: {str(sm.get('blind_spot', ''))[:100] or '-'} | "
        f"recent scars: {' || '.join(scars) or '-'}"
    )
  return '\n'.join(rows) or 'No Sub-Mind character sheet in this run.'


def _format_npc_council_turns(
    transcript: Mapping[str, Any], ledger_dict: Mapping[str, Any]
) -> str:
  """Formats paired NPC council turns for the LLM judge prompt."""
  domain_maps = _npc_domain_maps(ledger_dict)
  raw_turns = list(transcript.get('npc_council_turns', []))
  resolutions = list(transcript.get('resolutions', []))
  paired = _pair_npc_council_turns(raw_turns, resolutions, ledger_dict)
  rows: list[str] = []
  for turn in paired[-30:]:
    npc = str(turn['npc'])
    voice = str(turn['winning_voice'])
    opt_domain = next(
        (
            str(o.get('domain', '')).strip()
            for o in turn.get('options', [])
            if isinstance(o, Mapping) and str(o.get('voice', '')) == voice
        ),
        '',
    )
    domain = (
        domain_maps.get(npc, {}).get(voice)
        or (
            domain_maps.get(npc, {}).get(opt_domain, opt_domain)
            if opt_domain
            else ''
        )
        or '?'
    )
    action = next(
        (o['action'] for o in turn['options'] if o['voice'] == voice), ''
    )
    line = str(turn.get('npc_line', ''))
    player_prompt = str(turn.get('player_prompt', ''))
    prompt_line = (
        f'\n    PLAYER: {player_prompt[:220]}' if player_prompt else ''
    )
    rows.append(
        f"- step {turn['resolved_tick']} {npc}: winner {voice} ({domain}) urged"
        f' "{action[:140]}"{prompt_line}\n    LINE:'
        f' {line[:360] or "(no line found)"}'
    )
  return '\n'.join(rows) or 'No NPC council deliberations logged.'


def _format_combat_events(parliament: Mapping[str, Any]) -> str:
  """Formats Sub-Mind combat events for the LLM judge prompt."""
  combat = parliament.get('combat') or {}
  if not combat.get('available'):
    return 'None logged.'
  rows = [
      f"- step {e.get('step')} {e.get('player_voice')} vs {e.get('npc')}: "
      f"{e.get('outcome')} (roll {e.get('roll')}); NPC now led by "
      f"{e.get('dominant_voice')} rank {e.get('dominant_rank')}"
      f"{' [intrusive]' if e.get('intrusive') else ''}"
      for e in combat.get('sample', [])
  ]
  rows.append(
      f"outcomes: {combat.get('outcomes')}; decisive_rate "
      f"{combat.get('decisive_rate')}; stonewalled_rate "
      f"{combat.get('stonewalled_rate')}"
  )
  return '\n'.join(rows)


def metric_caps(
    deterministic: Mapping[str, Any], header: Mapping[str, Any]
) -> dict[str, tuple[float, str]]:
  """Derives per-dimension score ceilings from deterministic metrics.

  The rubric states these caps in prose, but graders have repeatedly scored
  86-95 across the board while the metrics block showed flagged NPCs and
  stale councils. Enforcing the caps in code makes the metrics binding.

  Args:
    deterministic: Output of `compute_deterministic_metrics`.
    header: Run header (archetype, info policy).

  Returns:
    Map from dimension name to `(ceiling, reason)`; absent means uncapped.
  """
  caps: dict[str, tuple[float, str]] = {}

  def _cap(dimension: str, ceiling: float, reason: str) -> None:
    current = caps.get(dimension)
    if current is None or ceiling < current[0]:
      caps[dimension] = (ceiling, reason)

  info_policy = str(header.get('info_policy', ''))
  if info_policy not in ('', 'unknown', 'human', 'player_visible_only'):
    for dimension in ('consequence_and_resistance', 'npc_agency'):
      _cap(dimension, 60, f'playtester info_policy={info_policy!r}')

  beats = deterministic.get('beat_repetition') or {}
  if beats.get('flagged_npcs'):
    _cap('npc_agency', 65, f"beat repetition flagged: {beats['flagged_npcs']}")

  voices = deterministic.get('voices') or {}
  if voices.get('enabled'):
    if any(l.get('kind') == 'truth' for l in voices.get('premature_leaks', [])):
      _cap('player_parliament', 40, 'inner voice leaked an undiscovered truth')
    if int(voices.get('unknown_faculty_lines', 0) or 0) > 0:
      _cap('player_parliament', 70, 'lines attributed to unknown faculties')
    options = voices.get('options') or {}
    stale = options.get('stale_option_rate')
    if stale is not None and stale > 0.25:
      _cap('player_parliament', 70, f'stale_option_rate={stale}')

  fidelity = deterministic.get('fidelity') or {}
  for key in ('clue_support_rate', 'storylet_adoption_rate'):
    rate = fidelity.get(key)
    if rate is not None and rate < SOFT_GATE_FIDELITY_RATE:
      _cap('ledger_prose_fidelity', 50, f'{key}={rate}')

  parliament = deterministic.get('parliament') or {}
  npc_side = parliament.get('npc') or {}
  if npc_side.get('available'):
    if npc_side.get('overall_mask_share', 0) >= 0.9:
      _cap(
          'npc_parliament',
          50,
          f"Mask won {npc_side['overall_mask_share']} of council turns; "
          'parliament inert',
      )
    if npc_side.get('npcs_with_voice_monopoly'):
      _cap(
          'npc_parliament',
          55,
          'Single voice monopolized NPC parliament (>=80% across >=4 turns): '
          f"{npc_side['npcs_with_voice_monopoly']}",
      )
    stale_dir = npc_side.get('stale_directive_rate')
    if stale_dir is not None and stale_dir > 0.35:
      _cap(
          'npc_parliament',
          55,
          'NPC sub-mind directives repeated across turns'
          f' (stale_directive_rate={stale_dir})',
      )
    unresp = npc_side.get('unresponsive_turn_rate')
    if unresp is not None and unresp > 0.30:
      for dim in (
          'npc_parliament',
          'npc_agency',
          'narrative_and_dialogue_coherence',
      ):
        _cap(
            dim,
            55,
            'NPC ignored player prompts to repeat prior moves'
            f' (unresponsive_turn_rate={unresp})',
        )
    support = npc_side.get('winning_action_support_mean')
    if support is not None and support < 0.15:
      _cap(
          'npc_parliament',
          60,
          f'NPC lines ignore the winning directive (support {support})',
      )
  combat = parliament.get('combat') or {}
  if combat.get('available') and combat.get('single_outcome_only'):
    _cap('voice_combat', 50, f"every clash resolved {combat.get('outcomes')}")

  coherence = deterministic.get('narrative_coherence') or {}
  if coherence.get('ghost_locations'):
    _cap(
        'narrative_and_dialogue_coherence',
        55,
        f"duplicate/ghost locations created: {coherence['ghost_locations']}",
    )
  defect_count = int(coherence.get('total_defect_count', 0) or 0)
  if defect_count >= 5:
    _cap(
        'narrative_and_dialogue_coherence',
        60,
        f'total_defect_count={defect_count} narrative/syntax defects detected',
    )
  elif defect_count >= 2:
    _cap(
        'narrative_and_dialogue_coherence',
        72,
        f'total_defect_count={defect_count} narrative/syntax defects detected',
    )
  return caps


def _format_header(
    header: Mapping[str, Any], applicability: Mapping[str, Mapping[str, Any]]
) -> str:
  lines = [f'{k}: {v}' for k, v in header.items()]
  lines.append('Dimension applicability:')
  for name, info in applicability.items():
    status = 'APPLICABLE' if info['applicable'] else 'NOT EXERCISED'
    reason = f" — {info['reason']}" if info.get('reason') else ''
    lines.append(f'  - {name}: {status}{reason}')
  return '\n'.join(lines)


def build_autorater_prompt(
    *,
    transcript: Mapping[str, Any],
    ledger_dict: Mapping[str, Any],
    rubric_text: str,
    deterministic: Mapping[str, Any],
    applicability: Mapping[str, Mapping[str, Any]],
    header: Mapping[str, Any],
) -> str:
  """Assembles the grading prompt with every evidence block the rubric needs."""
  story = ledger_dict.get('story') or {}
  journal_lines = '\n'.join(f'- {j}' for j in story.get('journal', []))
  memories_summary = []
  for name, mems in dict(transcript.get('entity_memories', {})).items():
    tail = ' | '.join(mems[-4:])
    memories_summary.append(f'- {name} ({len(mems)} memories): {tail}')
  metrics_for_prompt = {
      'fidelity': deterministic.get('fidelity'),
      'pacing': deterministic.get('pacing'),
      'dormant_npc_addressability': deterministic.get(
          'dormant_npc_addressability'
      ),
      'beat_repetition': deterministic.get('beat_repetition'),
      'voices': deterministic.get('voices'),
      'parliament': deterministic.get('parliament'),
      'narrative_coherence': deterministic.get('narrative_coherence'),
      'spoilers': deterministic.get('spoilers'),
  }
  return (
      'You are a strict, non-sycophantic simulation evaluator.\n'
      'Grade the following Adventure simulation run strictly according to '
      'this rubric:\n\n'
      f'{rubric_text}\n\n'
      '=== RUN HEADER ===\n'
      f'{_format_header(header, applicability)}\n\n'
      '=== DETERMINISTIC METRICS (computed in code; treat as ground truth) '
      '===\n'
      f'{json.dumps(metrics_for_prompt, indent=1)}\n\n'
      '=== ENTITY ACTIONS ===\n'
      + '\n'.join(transcript.get('action_lines', [])[-60:])
      + '\n\n=== GM SCENE RESOLUTIONS (in order) ===\n'
      + '\n'.join(transcript.get('resolution_lines', [])[-80:])
      + '\n\n=== OFF-SCREEN WARM EVENTS ===\n'
      + ('\n'.join(transcript.get('warm_lines', [])) or 'None')
      + '\n\n=== NPC RE-ENTRY PAIRS (before leaving Hot / after return) ===\n'
      + _format_reentry_pairs(deterministic.get('reentry_pairs', []))
      + '\n\n=== STORYLETS FIRED, ALIGNED WITH PROSE AT THAT TICK ===\n'
      + _format_fired_beats(ledger_dict, deterministic.get('fidelity', {}))
      + '\n\n=== INNER VOICE LINES (faculty: line, by tick) ===\n'
      + _format_voice_lines(ledger_dict)
      + '\n\n=== VOICE OPTIONS OFFERED (last 6 refreshes) ===\n'
      + _format_voice_options(ledger_dict)
      + '\n\n=== PLAYER PARLIAMENT (character sheet at end of run) ===\n'
      + _format_player_parliament(ledger_dict)
      + '\n\n=== NPC PARLIAMENT TURNS (winning inner voice, then the line) '
      '===\n'
      + _format_npc_council_turns(transcript, ledger_dict)
      + '\n\n=== VOICE-VS-VOICE COMBAT EVENTS ===\n'
      + _format_combat_events(deterministic.get('parliament', {}))
      + '\n\n=== DIRECTOR JOURNAL (what the ledger claims the player learned) '
      '===\n'
      + (journal_lines or 'None')
      + '\n\n=== FINAL ENTITY MEMORIES (TAIL) ===\n'
      + '\n'.join(memories_summary)
      + '\n\nReturn ONLY a valid JSON object matching the exact schema in the '
      'rubric. Use single quotes inside string values so the JSON stays valid. '
      'For every dimension marked NOT EXERCISED above, set "score" '
      'to null and explain in "evidence_against" why it was not exercised.'
  )


def _parse_dimension(name: str, raw: Any, applicable: bool) -> dict[str, Any]:
  """Validates and normalizes a single graded rubric dimension."""
  if not isinstance(raw, Mapping):
    raise AutoraterParseError(
        f'Dimension {name!r} missing from autorater JSON.'
    )
  score_raw = raw.get('score')
  evidence_for = str(raw.get('evidence_for', '')).strip()
  evidence_against = str(raw.get('evidence_against', '')).strip()
  if not applicable or score_raw is None:
    if applicable:
      raise AutoraterParseError(
          f'Dimension {name!r} is applicable but the autorater gave no score.'
      )
    return {
        'score': None,
        'evidence_for': evidence_for,
        'evidence_against': evidence_against,
    }
  try:
    score = float(score_raw)
  except (TypeError, ValueError) as err:
    raise AutoraterParseError(
        f'Dimension {name!r} score {score_raw!r} is not numeric.'
    ) from err
  if not evidence_for or not evidence_against:
    raise AutoraterParseError(
        f'Dimension {name!r} must cite both evidence_for and evidence_against.'
    )
  return {
      'score': round(max(0.0, min(100.0, score)), 1),
      'evidence_for': evidence_for,
      'evidence_against': evidence_against,
  }


def run_qualitative_autorater(
    model: language_model.LanguageModel,
    *,
    transcript: Mapping[str, Any],
    ledger_dict: Mapping[str, Any],
    rubric_text: str,
    deterministic: Mapping[str, Any],
    applicability: Mapping[str, Mapping[str, Any]],
    header: Mapping[str, Any],
) -> dict[str, Any]:
  """Grades the run against the rubric and renormalises over exercised dims.

  Args:
    model: Grader model.
    transcript: Output of `extract_transcript_from_log`.
    ledger_dict: Serialized `WorldLedger`.
    rubric_text: Contents of `adventure_rubric.md`.
    deterministic: Output of `compute_deterministic_metrics`.
    applicability: Output of `compute_applicability`.
    header: Run metadata (archetype, info policy, steps).

  Returns:
    Per-dimension scores and evidence, weighted total over applicable
    dimensions, lowest applicable dimension, and the improvement target.

  Raises:
    AutoraterParseError: The model's output did not satisfy the schema. The
      caller should surface this instead of recording a default score.
  """
  prompt = build_autorater_prompt(
      transcript=transcript,
      ledger_dict=ledger_dict,
      rubric_text=rubric_text,
      deterministic=deterministic,
      applicability=applicability,
      header=header,
  )
  last_error: AutoraterParseError | None = None
  for attempt in range(3):
    retry_note = ''
    if attempt > 0:
      retry_note = (
          '\n\nIMPORTANT: Your previous response failed JSON validation '
          f'({last_error}). Output valid JSON only, escaping or using single '
          'quotes inside evidence strings.'
      )
    raw = model.sample_text(
        prompt + retry_note, max_tokens=8192, terminators=()
    )
    parsed = adventure_specific_components.extract_json_object(raw)
    if not isinstance(parsed, Mapping):
      last_error = AutoraterParseError(
          f'Autorater returned no JSON object. Raw head: {raw[:300]!r}'
      )
      continue
    dims_raw = parsed.get('dimensions')
    if not isinstance(dims_raw, Mapping):
      last_error = AutoraterParseError(
          'Autorater JSON lacks a "dimensions" mapping.'
      )
      continue
    try:
      dimensions: dict[str, dict[str, Any]] = {}
      caps = metric_caps(deterministic, header)
      caps_applied: dict[str, dict[str, Any]] = {}
      weighted_sum = 0.0
      weight_sum = 0.0
      lowest_dim = ''
      lowest_score = 101.0
      for dim_name, weight in DIMENSION_WEIGHTS.items():
        applicable = bool(
            applicability.get(dim_name, {}).get('applicable', True)
        )
        entry = _parse_dimension(dim_name, dims_raw.get(dim_name), applicable)
        entry['weight'] = weight
        entry['applicable'] = applicable
        dimensions[dim_name] = entry
        if entry['score'] is None:
          continue
        cap = caps.get(dim_name)
        if cap is not None and entry['score'] > cap[0]:
          caps_applied[dim_name] = {
              'grader_score': entry['score'],
              'ceiling': cap[0],
              'reason': cap[1],
          }
          entry['grader_score'] = entry['score']
          entry['score'] = float(cap[0])
        weighted_sum += entry['score'] * weight
        weight_sum += weight
        if entry['score'] < lowest_score:
          lowest_score = entry['score']
          lowest_dim = dim_name
      return {
          'dimensions': dimensions,
          'weighted_total': (
              round(weighted_sum / weight_sum, 2) if weight_sum > 0 else None
          ),
          'applicable_weight': round(weight_sum, 2),
          'caps_applied': caps_applied,
          'lowest_dimension': str(parsed.get('lowest_dimension') or lowest_dim),
          'top_improvement_target': (
              str(parsed.get('top_improvement_target') or '').strip()
          ),
      }
    except AutoraterParseError as err:
      last_error = err
  assert last_error is not None
  raise last_error


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _load_rubric(rubric_path: str | None) -> str:
  candidates = [rubric_path] if rubric_path else []
  candidates.append(story_state.resolve_asset_path('adventure_rubric.md'))
  for candidate in candidates:
    if candidate and os.path.exists(candidate):
      with open(candidate, 'r', encoding='utf-8') as f:
        return f.read()
  raise FileNotFoundError(
      f'adventure_rubric.md not found (tried {candidates}).'
  )


def _load_meta(output_dir: str) -> dict[str, Any]:
  meta_path = os.path.join(output_dir, PLAYTEST_META_FILENAME)
  if not os.path.exists(meta_path):
    return {}
  with open(meta_path, 'r', encoding='utf-8') as f:
    loaded = json.load(f)
  return dict(loaded) if isinstance(loaded, Mapping) else {}


def evaluate_artifacts_directory(
    output_dir: str,
    *,
    model: language_model.LanguageModel,
    rubric_path: str | None = None,
) -> dict[str, Any]:
  """Evaluates all simulation artifacts in `output_dir` and writes a report.

  Args:
    output_dir: Directory containing `simulation_structured.json`,
      `world_ledger.json`, `latency.json`, and optionally `playtest_meta.json`.
    model: Grader model for the qualitative layer.
    rubric_path: Override for the rubric file; defaults to the bundled one.

  Returns:
    The full report, also written to `evaluation_report.json`.
  """
  structured_path = os.path.join(output_dir, 'simulation_structured.json')
  ledger_path = os.path.join(output_dir, 'world_ledger.json')
  latency_path = os.path.join(output_dir, 'latency.json')

  with open(structured_path, 'r', encoding='utf-8') as f:
    sim_log = structured_logging.SimulationLog.from_json(f.read())
  with open(ledger_path, 'r', encoding='utf-8') as f:
    ledger_dict = json.load(f)
  with open(latency_path, 'r', encoding='utf-8') as f:
    latency_dict = json.load(f)
  rubric_text = _load_rubric(rubric_path)
  meta = _load_meta(output_dir)

  transcript = extract_transcript_from_log(sim_log)
  header = {
      'archetype': (
          meta.get('archetype') or transcript['archetype'] or 'unknown'
      ),
      'info_policy': (
          meta.get('info_policy') or transcript['info_policy'] or 'unknown'
      ),
      'steps': transcript['overview'].get('total_steps', len(sim_log.entries)),
      'bible': meta.get('bible', ''),
      'council_follow_rate': meta.get('council_follow_rate', 'unknown'),
      'council_follows': meta.get('council_follows', 'unknown'),
  }
  deterministic = compute_deterministic_metrics(
      ledger_dict=ledger_dict, transcript=transcript
  )
  applicability = compute_applicability(
      str(header['archetype']),
      deterministic['reentry_pairs'],
      deterministic['voices'],
      deterministic['parliament'],
  )
  invariants = evaluate_hard_invariants(
      telemetry=ledger_dict.get('telemetry', {}),
      latency=latency_dict,
      entity_memories=transcript['entity_memories'],
      deterministic=deterministic,
  )
  autorater = run_qualitative_autorater(
      model,
      transcript=transcript,
      ledger_dict=ledger_dict,
      rubric_text=rubric_text,
      deterministic=deterministic,
      applicability=applicability,
      header=header,
  )

  full_report = {
      'header': header,
      'invariants': invariants,
      'deterministic': deterministic,
      'applicability': applicability,
      'autorater': autorater,
      'overview': transcript['overview'],
  }
  report_json_path = os.path.join(output_dir, 'evaluation_report.json')
  with open(report_json_path, 'w', encoding='utf-8') as f:
    json.dump(full_report, f, indent=2)
  return full_report
