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

"""Story-layer data model: world bible, truths, clues, factions, storylets.

The story layer is pure data plus deterministic bookkeeping. Nothing in this
module calls a language model. A `WorldBible` is authored JSON (see
`content/glassharbor.json`); `StoryState` is the mutable runtime projection of
that bible, owned by `WorldLedger` and serialized alongside it.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
import re
from typing import Any

# Condition keys the StoryDirector knows how to evaluate. Validation rejects any
# storylet that uses a key outside this set so typos fail at load time.
CONDITION_KEYS = frozenset({
    'npc_present',
    'npc_absent',
    'npc_tier',
    'location_is',
    'location_visited',
    'inventory_has',
    'inventory_lacks',
    'clue_discovered',
    'clue_undiscovered',
    'clue_at_location',
    'truth_discovered',
    'truth_undiscovered',
    'faction_standing_lt',
    'faction_standing_gt',
    'inter_faction_standing_lte',
    'tension_lt',
    'tension_gt',
    'clock_between',
    'doom_stage_gte',
    'doom_stage_lt',
    'act_is',
    'act_gte',
    'relationship_contains',
    'fired_count_lt',
    'rumor_contains',
    'fact_is',
    'player_status_contains',
})

# Effect keys the StoryDirector knows how to apply.
EFFECT_KEYS = frozenset({
    'tension',
    'faction_shift',
    'inter_faction_shift',
    'reveal_clue',
    'move_clue',
    'relationship',
    'rumor',
    'add_affordance',
    'set_fact',
    'journal',
    'set_act',
    'end_run',
})

TRIGGERS = frozenset(
    {'scene', 'arrival', 'warm', 'doom', 'rehydrate', 'ending'}
)

CLUE_CARRIERS = frozenset(
    {'location', 'npc', 'document', 'moment', 'bond', 'system'}
)


def _str_list(raw: Any) -> list[str]:
  if not isinstance(raw, (list, tuple)):
    return []
  return [str(x).strip() for x in raw if str(x).strip()]


@dataclasses.dataclass
class ToneConfig:
  """Configurable tone knobs read by the StoryDirector and prompt builders.

  Attributes:
    name: Short label, e.g. 'noir', 'heroic', 'grim'.
    prose_register: One-sentence style guidance injected into GM prompts.
    npc_lie_probability: Base chance (0-1) that an NPC lies when cornered about
      their secret.
    reveal_strictness: 'lenient' reveals clues on inspection; 'strict' requires
      leverage, force, or a trade.
    doom_hard: If True the run ends in failure when the doom clock expires.
    clean_endings: If False the Director prefers epilogues with moral residue.
    inner_voices: If True the GM may emit one-line inner-voice asides.
    setting_style: Optional world-generation style guidance for frontier rooms.
    room_palette: Optional list of architectural/location motifs for frontier
      generation.
    npc_archetypes: Optional list of character archetypes for frontier casting.
  """

  name: str = 'noir'
  prose_register: str = (
      'Terse, sensory, rain-and-brass noir. Dialogue carries subtext; nobody '
      'explains themselves fully.'
  )
  npc_lie_probability: float = 0.5
  reveal_strictness: str = 'strict'
  doom_hard: bool = True
  clean_endings: bool = False
  inner_voices: bool = True
  setting_style: str = ''
  room_palette: list[str] = dataclasses.field(default_factory=list)
  npc_archetypes: list[str] = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any] | None) -> 'ToneConfig':
    payload = payload or {}
    default = cls()
    return cls(
        name=str(payload.get('name', default.name)),
        prose_register=str(
            payload.get('prose_register', default.prose_register)
        ),
        npc_lie_probability=float(
            payload.get('npc_lie_probability', default.npc_lie_probability)
        ),
        reveal_strictness=str(
            payload.get('reveal_strictness', default.reveal_strictness)
        ),
        doom_hard=bool(payload.get('doom_hard', default.doom_hard)),
        clean_endings=bool(payload.get('clean_endings', default.clean_endings)),
        inner_voices=bool(payload.get('inner_voices', default.inner_voices)),
        setting_style=str(payload.get('setting_style', default.setting_style)),
        room_palette=_str_list(payload.get('room_palette')),
        npc_archetypes=_str_list(payload.get('npc_archetypes')),
    )

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass
class Truth:
  """A narrative arc, turning point, or hidden truth the player can unlock.

  Attributes:
    id: Stable identifier (e.g. 't_valve').
    statement: The full truth or turning-point realization, never shown to the
      player verbatim before discovery.
    clue_ids: Catalysts/clues that together establish or unlock this arc.
    min_clues_to_discover: How many of `clue_ids` must be found before the
      arc/truth counts as discovered.
    discovered: Runtime flag.
    discovered_tick: Tick at which the truth was discovered.
    arc_kind: Genre-generalized category ('revelation', 'bond', 'betrayal',
      'ideology', 'system_shift', 'reckoning').
    stakes: Optional description of what shifts in relationships or the world
      when this arc resolves. May be shown to the player, so it must pose the
      outcome without naming the answer.
    question: Player-safe label for this thread while it is still unsolved (e.g.
      "Who silenced the station?"). Shown in the UI case board; must not restate
      `statement`.
  """

  id: str
  statement: str
  clue_ids: list[str] = dataclasses.field(default_factory=list)
  min_clues_to_discover: int = 0
  discovered: bool = False
  discovered_tick: int = -1
  arc_kind: str = 'revelation'
  stakes: str = ''
  question: str = ''

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'Truth':
    clue_ids = _str_list(payload.get('clue_ids') or payload.get('catalyst_ids'))
    min_clues = int(
        payload.get('min_clues_to_discover')
        or payload.get('min_catalysts_to_discover')
        or 0
    )
    if min_clues <= 0:
      min_clues = max(1, (len(clue_ids) + 1) // 2) if clue_ids else 0
    return cls(
        id=str(payload['id']),
        statement=str(payload.get('statement', '')),
        clue_ids=clue_ids,
        min_clues_to_discover=min_clues,
        discovered=bool(payload.get('discovered', False)),
        discovered_tick=int(payload.get('discovered_tick', -1)),
        arc_kind=str(payload.get('arc_kind', 'revelation') or 'revelation'),
        stakes=str(payload.get('stakes', '')),
        question=str(payload.get('question', '')),
    )


@dataclasses.dataclass
class Clue:
  """A narrative catalyst, revelation, bond moment, or piece of evidence.

  Attributes:
    id: Stable identifier (e.g. 'c_manifest').
    truth_id: The truth / narrative arc this catalyst supports.
    text: What the player learns or experiences when the catalyst triggers.
    carrier: 'location' (prop/scene beat), 'npc' (confession/bond/secret),
      'document' (readable object), 'moment' (dramatic encounter), 'bond'
      (interpersonal shift), or 'system' (terminal/network node).
    carrier_hint: Guidance for placement: an NPC name, a location name, or a
      description of the kind of place/object that should hold it.
    placed_at: Location id or NPC name once the clue is placed in the world.
    discovered: Runtime flag.
    discovered_tick: Tick at which the player found it.
    movable: Whether off-screen Warm NPCs can physically relocate or steal this
      clue/MacGuffin before the player discovers it.
    catalyst_kind: Genre-generalized category ('evidence', 'confession',
      'intimacy', 'vow', 'betrayal', 'sabotage', 'transmission').
  """

  id: str
  truth_id: str
  text: str
  carrier: str = 'location'
  carrier_hint: str = ''
  placed_at: str | None = None
  discovered: bool = False
  discovered_tick: int = -1
  movable: bool = False
  catalyst_kind: str = 'evidence'

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'Clue':
    placed = payload.get('placed_at')
    carrier = str(payload.get('carrier', 'location'))
    default_movable = carrier == 'document'
    return cls(
        id=str(payload['id']),
        truth_id=str(payload.get('truth_id') or payload.get('arc_id', '')),
        text=str(payload.get('text', '')),
        carrier=carrier,
        carrier_hint=str(payload.get('carrier_hint', '')),
        placed_at=str(placed) if placed else None,
        discovered=bool(payload.get('discovered', False)),
        discovered_tick=int(payload.get('discovered_tick', -1)),
        movable=bool(payload.get('movable', default_movable)),
        catalyst_kind=str(
            payload.get('catalyst_kind', 'evidence') or 'evidence'
        ),
    )


def infer_faction_polarity_delta(text: str) -> int:
  """Infers a lightweight polarity delta in [-2, 2] from natural-language text."""
  low = (text or '').lower()
  if not low:
    return 0
  strong_neg = (
      'open turf war',
      'shoot on sight',
      'scorched earth',
      'warrant',
      'indictment',
      'betrayed',
      'blood feud',
      'tore apart',
  )
  neg_tokens = (
      'hostile',
      'rattled',
      'cornered',
      'strong-armed',
      'compromised',
      'pinned',
      'exposed',
      'reprisal',
      'turf war',
      'feud',
      'cooled',
      'crossed',
      'ambush',
      'burned',
      'raided',
      'stonewalled',
      'threatened',
      'destabilized',
      'antagonized',
      'suspects',
      'distrust',
      'shortfall',
      'skimming',
  )
  strong_pos = (
      'allied',
      'sworn pact',
      'squared the debt',
      'clean sweep',
      'handed control',
      'full backing',
  )
  pos_tokens = (
      'reciprocal',
      'hooked',
      'side-angle',
      'side-deal',
      'deal',
      'bargain',
      'trust',
      'backed',
      'squared',
      'cut in',
      'cut into',
      'warmed',
      'indebted',
      'favored',
      'pact',
      'cooperative',
      'cleared',
      'aligned',
      'vouched',
      'brokered',
  )
  if any(t in low for t in strong_neg):
    return -2
  if any(t in low for t in strong_pos):
    return 2
  neg_hits = sum(1 for t in neg_tokens if t in low)
  pos_hits = sum(1 for t in pos_tokens if t in low)
  if pos_hits > neg_hits:
    return 1
  if neg_hits > pos_hits:
    return -1
  return 0


@dataclasses.dataclass
class Faction:
  """A political actor in the world (may be non-human, e.g. the floodgate).

  Attributes:
    id: Stable identifier (e.g. 'wardens').
    name: Display name.
    aim: What the faction is trying to achieve this run.
    disposition: Natural-language stance toward the player.
    relations: Natural-language stance toward other factions, keyed by faction
      id.
    recent_actions: Chronological ledger of concrete faction shifts/events.
    standing_with_player: Internal polarity score in [-5, 5] for precondition
      compatibility.
    standing: Internal polarity toward other factions, keyed by faction id.
    assets: Things the faction controls (places, objects, people).
    register: How members of this faction tend to speak.
  """

  id: str
  name: str
  aim: str = ''
  standing_with_player: int = 0
  standing: dict[str, int] = dataclasses.field(default_factory=dict)
  assets: list[str] = dataclasses.field(default_factory=list)
  register: str = ''
  disposition: str = ''
  relations: dict[str, str] = dataclasses.field(default_factory=dict)
  recent_actions: list[str] = dataclasses.field(default_factory=list)

  def current_disposition(self) -> str:
    """Returns a natural-language description of stance toward the player."""
    if self.disposition.strip():
      return self.disposition.strip()
    aim_clause = f' ({self.aim})' if self.aim else ''
    if self.standing_with_player >= 3:
      return f'Cut into a working alliance with you{aim_clause}.'
    if self.standing_with_player >= 1:
      return f'Cautiously receptive to your angle{aim_clause}.'
    if self.standing_with_player <= -3:
      return f'Openly hostile and moving to shut you down{aim_clause}.'
    if self.standing_with_player <= -1:
      return f'Suspicious and defensive toward your probing{aim_clause}.'
    return f'Wary neutrality; watching how you play your hand{aim_clause}.'

  def to_dict(self) -> dict[str, Any]:
    data = dataclasses.asdict(self)
    if not data.get('disposition'):
      data['disposition'] = self.current_disposition()
    return data

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'Faction':
    """Deserializes a `Faction` from a dictionary."""
    standing = {
        str(k): int(v) for k, v in dict(payload.get('standing', {})).items()
    }
    raw_relations = payload.get('relations', {})
    if isinstance(raw_relations, Mapping):
      relations = {
          str(k): str(v)
          for k, v in dict(raw_relations).items()
          if str(v).strip()
      }
    else:
      relations = {}
    for other_id, st_val in standing.items():
      if other_id not in relations:
        if st_val <= -4:
          relations[other_id] = (
              'Open turf war: crews trading reprisals and shooting on sight.'
          )
        elif st_val < 0:
          relations[other_id] = 'Simmering turf friction and mutual suspicion.'
        elif st_val > 0:
          relations[other_id] = 'Quiet working understanding.'
    fac = cls(
        id=str(payload['id']),
        name=str(payload.get('name', payload['id'])),
        aim=str(payload.get('aim', '')),
        standing_with_player=int(payload.get('standing_with_player', 0)),
        standing=standing,
        assets=_str_list(payload.get('assets')),
        register=str(payload.get('register', '')),
        disposition=str(payload.get('disposition', '')).strip(),
        relations=relations,
        recent_actions=_str_list(payload.get('recent_actions')),
    )
    if not fac.disposition:
      fac.disposition = fac.current_disposition()
    return fac


@dataclasses.dataclass
class Storylet:
  """An authored narrative fragment with preconditions and effects.

  Attributes:
    id: Stable identifier.
    trigger: One of TRIGGERS; when the Director considers this storylet.
    weight: Relative selection weight among eligible storylets.
    once: If True the storylet can fire at most once per run.
    preconditions: List of single-key dicts, e.g. {'npc_present': 'Orin'}.
    beat: The authored beat, written as guidance for the GM, not player text.
    effects: List of single-key dicts applied when the storylet fires.
    voice: Optional inner-voice faculty name to attach an aside to.
    voice_line: Optional authored inner-voice line.
    tags: Free-form tags for filtering and evaluation.
  """

  id: str
  trigger: str
  beat: str
  weight: float = 1.0
  once: bool = True
  preconditions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  effects: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  voice: str = ''
  voice_line: str = ''
  tags: list[str] = dataclasses.field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'Storylet':
    return cls(
        id=str(payload['id']),
        trigger=str(payload.get('trigger', 'scene')),
        beat=str(payload.get('beat', '')),
        weight=float(payload.get('weight', 1.0)),
        once=bool(payload.get('once', True)),
        preconditions=[
            dict(c)
            for c in payload.get('preconditions', [])
            if isinstance(c, Mapping)
        ],
        effects=[
            dict(e)
            for e in payload.get('effects', [])
            if isinstance(e, Mapping)
        ],
        voice=str(payload.get('voice', '')),
        voice_line=str(payload.get('voice_line', '')),
        tags=_str_list(payload.get('tags')),
    )


def _has_any_word(text: str, words: Sequence[str]) -> bool:
  """Returns True if any token in `words` appears as a whole word/stem in `text`."""
  for w in words:
    if re.search(rf'\b{re.escape(w)}\b', text):
      return True
  return False


def infer_submind_defaults(
    name: str, worldview: str, domain_hint: str = ''
) -> tuple[str, str, list[str]]:
  """Infers domain, blind spot, and triggers from a voice name and worldview."""
  name_low = name.strip().lower()
  combined = f'{name} {worldview}'.lower()
  hint = domain_hint.strip().lower()

  # 1. Canonical Bermondsey voices
  if 'brass' in name_low:
    return (
        'audacity',
        'Bluffs past the point of credibility and provokes entrenched heavies.',
        ['door', 'badge', 'clipboard', 'bar', 'office', 'vip', 'gate'],
    )
  if 'hustle' in name_low:
    return (
        'leverage',
        'Tries to broker a three-way cut even when the room is on fire.',
        ['voucher', 'key', 'pass', 'debt', 'holdall', 'collateral', 'price'],
    )
  if 'book' in name_low:
    return (
        'accounting',
        (
            'Reduces human desperation to arithmetic and misses impending '
            'violence.'
        ),
        ['ledger', 'manifest', 'carbon', 'slip', 'card', 'book', 'scale'],
    )
  if 'copper' in name_low:
    return (
        'forensics',
        'Sees police wires and planted setups behind every shadow.',
        ['shells', 'rover', 'glovebox', 'serial', 'stamp', 'lock', 'trace'],
    )
  if 'naughty' in name_low:
    return (
        'coercion',
        (
            'Treats every locked door or tense silence as an excuse for heavy '
            'iron.'
        ),
        ['lock', 'padlock', 'shutter', 'case', 'brazier', 'valve', 'lever'],
    )
  if 'ear' in name_low and 'ground' in name_low:
    return (
        'empathy',
        'Over-reads nervous tells and hesitates when direct force is needed.',
        ['whisper', 'tell', 'sweat', 'cadence', 'alibi', 'bar', 'glass'],
    )
  if 'guv' in name_low:
    return (
        'hierarchy',
        'Pulls rank too hard and turns neutral crews into turf-war enemies.',
        ['boss', 'manor', 'turf', 'split', 'collector', 'table', 'order'],
    )

  # 2. Canonical Glassharbor voices (Clockwork Maritime Noir)
  if name_low == 'suspicion':
    return (
        'forensics',
        'Reads every kindness as a trap and alienates potential allies.',
        ['seal', 'lock', 'alibi', 'ledger', 'watch', 'key', 'manifest'],
    )
  if name_low == 'empathy':
    return (
        'empathy',
        (
            'Feels the fear behind a lie and hesitates when hard leverage is'
            ' needed.'
        ),
        ['fear', 'daughter', 'debt', 'tremor', 'voice', 'eyes', 'grief'],
    )
  if 'salt-memory' in name_low or 'salt memory' in name_low:
    return (
        'instinct',
        'Gets pulled into dockside nostalgia and old port grudges.',
        ['tide', 'wharf', 'rope', 'skiff', 'sluice', 'lantern', 'salt', 'pier'],
    )
  if name_low == 'ledger':
    return (
        'accounting',
        (
            'Reduces human desperation to arithmetic and misses impending '
            'violence.'
        ),
        [
            'ledger',
            'manifest',
            'folio',
            'date',
            'receipt',
            'tally',
            'chit',
            'chart',
        ],
    )
  if name_low == 'nerve':
    return (
        'coercion',
        (
            'Treats every locked door or tense silence as an excuse for direct '
            'force.'
        ),
        [
            'padlock',
            'door',
            'valve',
            'wheel',
            'shutter',
            'exit',
            'crane',
            'rope',
        ],
    )

  # 3. Canonical Dystopian Sci-Fi voices (Outpost 7 & Sector 9)
  if 'dead-reckoning' in name_low or 'dead reckoning' in name_low:
    return (
        'accounting',
        (
            'Gets absorbed in logbook discrepancies while someone reaches '
            'for a heavy wrench.'
        ),
        ['manifest', 'log', 'weight', 'crate', 'timestamp', 'tally', 'slip'],
    )
  if 'cabin fever' in name_low:
    return (
        'empathy',
        (
            'Sympathizes with an exhausted crewman’s isolation and hesitates '
            'to press hard when stories falter.'
        ),
        ['coffee', 'parka', 'tag', 'hands', 'memory', 'voice', 'knuckle'],
    )
  if 'hard line' in name_low:
    return (
        'tactical_force',
        (
            'Draws iron too early in a cramped room and turns panic into a '
            'bloodbath.'
        ),
        ['cutters', 'latch', 'revolver', 'door', 'wrench', 'holster'],
    )
  if 'company book' in name_low:
    return (
        'hierarchy',
        (
            'Assumes company regulations still protect you six hundred miles '
            'from the nearest settlement.'
        ),
        ['contract', 'company', 'waiver', 'director', 'seal', 'clearance'],
    )
  if 'side-angle' in name_low or 'side angle' in name_low:
    return (
        'leverage',
        (
            'Tries to cut a quiet side-deal with people who cannot afford to '
            'leave a witness alive.'
        ),
        ['sample', 'shuttle', 'seat', 'bonus', 'bribe', 'ticket'],
    )
  if 'biometric lens' in name_low:
    return (
        'cybernetics',
        (
            'Trusts the pocket scanner readout over the human being standing '
            'in front of you.'
        ),
        ['cornea', 'pulse', 'pupil', 'wrist', 'scar', 'scanner'],
    )
  if 'neural lace' in name_low:
    return (
        'cybernetics',
        (
            'Treats human loyalty like a patchable exploit and triggers ICE'
            ' alarms.'
        ),
        ['terminal', 'subnet', 'implant', 'telemetry', 'node', 'cipher', 'hud'],
    )
  if 'chrome reflex' in name_low:
    return (
        'tactical_force',
        'Escalates standoffs into kinetic firefights before extracting data.',
        [
            'holster',
            'servo',
            'airlock',
            'sightline',
            'bulkhead',
            'hatch',
            'lock',
        ],
    )
  if 'cold calculus' in name_low:
    return (
        'accounting',
        'Reduces human survival to oxygen-credit margins and misses mutiny.',
        ['quota', 'oxygen', 'credit', 'audit', 'margin', 'ledger', 'manifest'],
    )
  if 'residual empathy' in name_low:
    return (
        'empathy',
        'Absorbs the trauma of compromised subjects and hesitates to purge.',
        ['memory', 'breathing', 'family', 'fear', 'scars', 'voice', 'tremor'],
    )
  if 'black-market signal' in name_low:
    return (
        'leverage',
        'Tries to broker a black-market cut even while the airlock is venting.',
        ['shard', 'smuggler', 'pass', 'bribe', 'clinic', 'deal', 'keycard'],
    )
  if 'union memory' in name_low:
    return (
        'hierarchy',
        (
            'Trusts shift-crew solidarity even when an informant has sold the'
            ' deck.'
        ),
        ['coolant', 'scrubber', 'shift', 'union', 'valve', 'crew', 'duct'],
    )

  # 4. Canonical Vampire Romance voices
  if name_low == 'the hunger':
    return (
        'predation',
        (
            'Risks losing control in a brightly lit hospital corridor when'
            ' fresh blood is in the air.'
        ),
        ['pulse', 'blood', 'vein', 'throat', 'heartbeat', 'scent', 'bandage'],
    )
  if name_low in ('cover story', 'velvet blade'):
    return (
        'decorum',
        (
            'Tries to maintain a routine cover story even when the secret is'
            ' already out on the table.'
        ),
        ['badge', 'shift', 'cooler', 'protocol', 'rule', 'schedule', 'id'],
    )
  if name_low == 'unspoken vow':
    return (
        'passion',
        (
            'Protects Claire or sympathizes with a desperate promise even when'
            ' it puts everyone in danger.'
        ),
        ['claire', 'watch', 'promise', 'train', 'hand', 'love', 'vow'],
    )
  if name_low in ('paper trail', 'whisper network'):
    return (
        'leverage',
        (
            'Treats grief or love like a bargain that can be settled with cash'
            ' or a train ticket.'
        ),
        ['folder', 'photo', 'file', 'camera', 'record', 'ticket', 'secret'],
    )
  if name_low in ('night chart', 'ancient lore'):
    return (
        'forensics',
        (
            'Focuses on lab numbers and clinical symptoms while emotions in the'
            ' room boil over.'
        ),
        ['binder', 'o-negative', 'pupil', 'fever', 'gauze', 'scar', 'cooler'],
    )
  if name_low == 'mortal echo':
    return (
        'empathy',
        (
            'Hesitates to use threat or force against a mortal who is grieving'
            ' or terrified.'
        ),
        ['mortal', 'breath', 'dawn', 'wife', 'grief', 'warmth', 'mercy'],
    )

  # 5. If an explicit domain_hint was supplied on SubMindState, honor it first.
  domain_defaults: dict[str, tuple[str, list[str]]] = {
      'predation': (
          (
              'Risks shattering courtly cover to feed or dominate a vulnerable'
              ' pulse.'
          ),
          ['pulse', 'goblet', 'throat', 'chalice', 'blood', 'shadow', 'scent'],
      ),
      'passion': (
          (
              'Trusts an alluring rival or forbidden lover past the point of'
              ' betrayal.'
          ),
          ['gaze', 'vow', 'ring', 'letter', 'balcony', 'whisper', 'touch'],
      ),
      'decorum': (
          (
              'Clings to ancestral protocol while rivals prepare a knife in the'
              ' dark.'
          ),
          ['seal', 'crest', 'throne', 'invitation', 'oath', 'title', 'court'],
      ),
      'cybernetics': (
          (
              'Treats human loyalty like a patchable exploit and triggers ICE'
              ' alarms.'
          ),
          [
              'terminal',
              'port',
              'datashard',
              'relay',
              'camera',
              'console',
              'node',
          ],
      ),
      'tactical_force': (
          'Escalates standoffs into kinetic firefights before extracting data.',
          ['bulkhead', 'holster', 'drone', 'turret', 'lock', 'armor', 'hatch'],
      ),
      'forensics': (
          'Sees planted setups and conspiracies behind every shadow.',
          ['shells', 'rover', 'glovebox', 'serial', 'stamp', 'lock', 'trace'],
      ),
      'hierarchy': (
          'Pulls rank too hard and turns neutral factions into enemies.',
          ['boss', 'manor', 'turf', 'split', 'collector', 'table', 'order'],
      ),
      'accounting': (
          (
              'Reduces human desperation to arithmetic and misses impending'
              ' violence.'
          ),
          ['ledger', 'manifest', 'carbon', 'slip', 'card', 'book', 'scale'],
      ),
      'coercion': (
          (
              'Treats every locked door or tense silence as an excuse for'
              ' direct force.'
          ),
          ['lock', 'padlock', 'shutter', 'case', 'brazier', 'valve', 'lever'],
      ),
      'empathy': (
          'Over-reads nervous tells and hesitates when direct force is needed.',
          ['whisper', 'tell', 'sweat', 'cadence', 'alibi', 'bar', 'glass'],
      ),
      'leverage': (
          'Tries to broker a three-way cut even when the room is on fire.',
          ['voucher', 'key', 'pass', 'debt', 'holdall', 'collateral', 'price'],
      ),
      'audacity': (
          'Bluffs empty-handed past the point where someone calls the hand.',
          [
              'door',
              'desk',
              'counter',
              'bluff',
              'freehold',
              'stare',
              'challenge',
          ],
      ),
  }
  if hint in domain_defaults:
    blind, trigs = domain_defaults[hint]
    return hint, blind, list(trigs)

  # 6. Word-boundary genre & general domain inference for custom/minted voices.
  if _has_any_word(
      combined,
      ('thirst', 'hunger', 'vein', 'fang', 'predator', 'bloodlust', 'vitae'),
  ):
    blind, trigs = domain_defaults['predation']
    return 'predation', blind, list(trigs)
  if _has_any_word(
      combined,
      (
          'siren',
          'yearning',
          'devotion',
          'desire',
          'passion',
          'romance',
          'lover',
          'seduce',
      ),
  ):
    blind, trigs = domain_defaults['passion']
    return 'passion', blind, list(trigs)
  if _has_any_word(
      combined,
      (
          'etiquette',
          'masquerade',
          'courtly',
          'lineage',
          'noble',
          'decorum',
          'chivalry',
      ),
  ):
    blind, trigs = domain_defaults['decorum']
    return 'decorum', blind, list(trigs)
  if _has_any_word(
      combined,
      (
          'splicer',
          'cybernetic',
          'cybernetics',
          'neural',
          'subnet',
          'wetware',
          'uplink',
          'datashard',
          'daemon',
          'netrunner',
      ),
  ):
    blind, trigs = domain_defaults['cybernetics']
    return 'cybernetics', blind, list(trigs)
  if _has_any_word(
      combined,
      (
          'overclock',
          'chrome',
          'servo',
          'tactical',
          'ballistics',
          'ballistic',
          'firefight',
      ),
  ):
    blind, trigs = domain_defaults['tactical_force']
    return 'tactical_force', blind, list(trigs)
  if _has_any_word(
      combined, ('copper', 'forensic', 'warrant', 'evidence', 'sensor', 'scan')
  ):
    blind, trigs = domain_defaults['forensics']
    return 'forensics', blind, list(trigs)
  if _has_any_word(
      combined,
      ('guv', 'hierarchy', 'firm', 'manor', 'protocol', 'directive', 'command'),
  ):
    blind, trigs = domain_defaults['hierarchy']
    return 'hierarchy', blind, list(trigs)
  if _has_any_word(
      combined, ('book', 'ledger', 'odds', 'vig', 'account', 'tally', 'audit')
  ):
    blind, trigs = domain_defaults['accounting']
    return 'accounting', blind, list(trigs)
  if _has_any_word(
      combined,
      (
          'naughty',
          'nerve',
          'mechanist',
          'iron',
          'cosh',
          'force',
          'duel',
          'blade',
      ),
  ):
    blind, trigs = domain_defaults['coercion']
    return 'coercion', blind, list(trigs)
  if _has_any_word(
      combined,
      ('ear', 'ground', 'empathy', 'suspicion', 'subtext', 'intuition'),
  ):
    blind, trigs = domain_defaults['empathy']
    return 'empathy', blind, list(trigs)
  if _has_any_word(
      combined,
      ('hustle', 'tide', 'leverage', 'swap', 'trade', 'deal', 'broker'),
  ):
    blind, trigs = domain_defaults['leverage']
    return 'leverage', blind, list(trigs)
  if _has_any_word(combined, ('brass', 'neck', 'audacity', 'bluff', 'daring')):
    blind, trigs = domain_defaults['audacity']
    return 'audacity', blind, list(trigs)
  return (
      'instinct',
      'Fixates on its own obsession at the expense of conventional caution.',
      ['clue', 'secret', 'door', 'lock', 'stranger', 'ledger'],
  )


def infer_submind_combat_profile(
    name: str, worldview: str, domain: str
) -> tuple[str, list[str], list[str], list[str]]:
  """Infers (role_archetype, strong_against, vulnerable_to, favored_tactics)."""
  dom = (domain or '').lower()
  combined = f'{name} {worldview} {dom}'.lower()
  if dom in ('coercion', 'tactical_force', 'predation') or any(
      w in combined for w in ('force', 'iron', 'thirst', 'predator', 'blade')
  ):
    return (
        'force',
        ['leverage', 'angle', 'drive', 'ambition', 'cowardice', 'deceit'],
        ['stonewalling', 'facade', 'hierarchy', 'decorum', 'protocol'],
        ['intimidate', 'corner', 'seize', 'overpower'],
    )
  if dom in ('forensics', 'accounting', 'cybernetics') or any(
      w in combined for w in ('audit', 'evidence', 'ledger', 'neural', 'cipher')
  ):
    return (
        'analytic',
        ['stonewalling', 'facade', 'self_preservation', 'wound', 'protocol'],
        ['coercion', 'tactical_force', 'predation'],
        ['expose', 'cross-examine', 'decrypt', 'audit'],
    )
  if dom in ('leverage', 'passion', 'seduction', 'empathy') or any(
      w in combined for w in ('deal', 'yearning', 'devotion', 'whisper', 'bond')
  ):
    return (
        'social',
        ['leverage', 'angle', 'drive', 'self_preservation', 'wound', 'passion'],
        ['hierarchy', 'decorum', 'protocol'],
        ['tempt', 'confide', 'bargain', 'entice'],
    )
  if dom in ('hierarchy', 'decorum', 'audacity') or any(
      w in combined for w in ('court', 'crown', 'guv', 'brass', 'command')
  ):
    return (
        'authority',
        ['stonewalling', 'facade', 'leverage', 'drive'],
        ['forensics', 'analytic', 'self_preservation'],
        ['command', 'pull rank', 'invoke oath', 'bluff'],
    )
  if dom == 'stonewalling':
    return (
        'facade',
        ['coercion', 'audacity', 'force'],
        ['forensics', 'accounting', 'cybernetics', 'evidence'],
        ['deflect', 'cite protocol', 'stonewall'],
    )
  if dom == 'self_preservation':
    return (
        'wound',
        ['decorum', 'facade'],
        ['forensics', 'coercion', 'passion', 'predation', 'cybernetics'],
        ['conceal', 'panic', 'plead', 'lash out'],
    )
  return (
      'instinct',
      ['facade', 'drive'],
      ['wound'],
      ['probe', 'press', 'improvise'],
  )


@dataclasses.dataclass
class SubMindState:
  """First-class HRL Sub-Mind / inner voice on a character sheet (Player or NPC).

  Attributes:
    name: Display name of the faculty (e.g. "The Book", "Velvet Decorum").
    domain: Core analytical, emotional, or tactical domain.
    worldview: Personality, speech register, and what it notices in a scene.
    blind_spot: How this sub-mind overreaches or misleads when dominant.
    rank: Character-sheet level (1=Whisper, 2=Clear, 3=Sharp, 4+=Dominant).
    xp: Accumulated experience toward rank progression.
    attention_triggers: Prop/scene keywords that wake this sub-mind up.
    scars: Formative memories and lessons learned from past turns.
    obsession: Active hypothesis, grudge, or fixation coloring its perceptions.
    origin: 'setting' | 'custom' | 'emergent' | 'npc'.
    active: Whether this sub-mind is enabled in the parliament.
    intrusive: True when dominant enough to hijack attention or push blind
      spots.
    role_archetype: Structural role ('facade', 'drive', 'wound', 'force',
      'analytic', 'social', 'authority', 'instinct').
    strong_against: Domains or archetypes this voice pierces in social combat.
    vulnerable_to: Domains or archetypes that crack or provoke this voice.
    favored_tactics: Genre-fitted verbs/approaches for procedural actions.
    voice_tone: Optional prose cadence hint for this voice.
  """

  name: str
  domain: str = 'instinct'
  worldview: str = ''
  blind_spot: str = ''
  rank: int = 1
  xp: int = 0
  attention_triggers: list[str] = dataclasses.field(default_factory=list)
  scars: list[str] = dataclasses.field(default_factory=list)
  obsession: str = ''
  origin: str = 'setting'
  active: bool = True
  intrusive: bool = False
  role_archetype: str = ''
  strong_against: list[str] = dataclasses.field(default_factory=list)
  vulnerable_to: list[str] = dataclasses.field(default_factory=list)
  favored_tactics: list[str] = dataclasses.field(default_factory=list)
  voice_tone: str = ''

  def short_role(self) -> str:
    """Returns a concise role label for UI badges and prompts."""
    if '—' in self.worldview:
      return self.worldview.split('—', 1)[0].strip()
    if '.' in self.worldview:
      return self.worldview.split('.', 1)[0].strip()
    return self.domain.replace('_', ' ').title()

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_legacy(
      cls,
      name: str,
      description: str,
      *,
      rank: int = 1,
      origin: str = 'setting',
  ) -> 'SubMindState':
    """Constructs a rich SubMindState from a legacy one-line description."""
    domain, blind_spot, triggers = infer_submind_defaults(name, description)
    role_arch, strong, vuln, tactics = infer_submind_combat_profile(
        name, description, domain
    )
    clean_rank = max(1, min(10, int(rank)))
    return cls(
        name=name.strip(),
        domain=domain,
        worldview=description.strip(),
        blind_spot=blind_spot,
        rank=clean_rank,
        xp=0,
        attention_triggers=triggers,
        scars=[],
        obsession='',
        origin=origin,
        active=True,
        intrusive=clean_rank >= 4,
        role_archetype=role_arch,
        strong_against=strong,
        vulnerable_to=vuln,
        favored_tactics=tactics,
    )

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'SubMindState':
    """Restores a SubMindState from a serialized dictionary."""
    name = str(payload.get('name', 'Sub-Mind')).strip()
    worldview = str(
        payload.get('worldview') or payload.get('description', '')
    ).strip()
    raw_domain = str(payload.get('domain', '')).strip()
    def_domain, def_blind, def_triggers = infer_submind_defaults(
        name, worldview, domain_hint=raw_domain
    )
    domain = raw_domain or def_domain
    def_arch, def_strong, def_vuln, def_tactics = infer_submind_combat_profile(
        name, worldview, domain
    )
    rank = max(1, min(10, int(payload.get('rank', 1))))
    triggers = _str_list(payload.get('attention_triggers')) or def_triggers
    return cls(
        name=name,
        domain=domain,
        worldview=worldview,
        blind_spot=str(payload.get('blind_spot') or def_blind).strip(),
        rank=rank,
        xp=max(0, int(payload.get('xp', 0))),
        attention_triggers=triggers,
        scars=_str_list(payload.get('scars')),
        obsession=str(payload.get('obsession', '')).strip(),
        origin=str(payload.get('origin', 'setting')).strip() or 'setting',
        active=bool(payload.get('active', True)),
        intrusive=bool(payload.get('intrusive', rank >= 4)),
        role_archetype=(str(payload.get('role_archetype') or def_arch).strip()),
        strong_against=_str_list(payload.get('strong_against')) or def_strong,
        vulnerable_to=_str_list(payload.get('vulnerable_to')) or def_vuln,
        favored_tactics=(
            _str_list(payload.get('favored_tactics')) or def_tactics
        ),
        voice_tone=str(payload.get('voice_tone', '')).strip(),
    )


@dataclasses.dataclass
class VoiceOption:
  """A context-aware HRL sub-mind perception and clickable 1-turn action option.

  Attributes:
    voice: Faculty name (e.g. "The Book", "Proper Naughty", "Copper's Nose").
    role: Short domain description of this sub-mind.
    perception: Context-aware insight explaining what this sub-mind notices in
      the current room and why it matters.
    action: Concrete 1-turn tactical action the player can execute or edit.
    affinity: Current dominance/affinity level of this sub-mind (1+).
    domain: Core analytical domain of this sub-mind.
    scar_note: Recent scar or obsession influencing this perception.
    intrusive: True when this option is an over-dominant intrusive impulse.
    origin: 'setting' | 'custom' | 'emergent'.
  """

  voice: str
  role: str = ''
  perception: str = ''
  action: str = ''
  affinity: int = 1
  domain: str = ''
  scar_note: str = ''
  intrusive: bool = False
  origin: str = 'setting'

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'VoiceOption':
    return cls(
        voice=str(payload.get('voice', '')),
        role=str(payload.get('role', '')),
        perception=str(payload.get('perception', '')),
        action=str(payload.get('action', '')),
        affinity=max(1, int(payload.get('affinity', 1))),
        domain=str(payload.get('domain', '')),
        scar_note=str(payload.get('scar_note', '')),
        intrusive=bool(payload.get('intrusive', False)),
        origin=str(payload.get('origin', 'setting')) or 'setting',
    )


@dataclasses.dataclass
class DoomStage:
  """One stage of the world's countdown.

  Attributes:
    index: Stage number, 0 is the starting calm.
    name: Short label used in Director notes.
    at_fraction: Fraction of total doom minutes elapsed when this stage begins.
    world_effect: What visibly changes; rendered into rumors/room variants.
  """

  index: int
  name: str
  at_fraction: float
  world_effect: str = ''

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any], index: int) -> 'DoomStage':
    return cls(
        index=int(payload.get('index', index)),
        name=str(payload.get('name', f'Stage {index}')),
        at_fraction=float(payload.get('at_fraction', 0.0)),
        world_effect=str(payload.get('world_effect', '')),
    )


@dataclasses.dataclass
class DoomConfig:
  """Countdown configuration.

  Attributes:
    name: What the doom is (e.g. 'the king-tide').
    total_minutes: Simulated minutes from run start until the final stage.
    stages: Ordered stages; the last one is the failure state.
  """

  name: str = 'the king-tide'
  total_minutes: int = 16 * 60
  stages: list[DoomStage] = dataclasses.field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
        'name': self.name,
        'total_minutes': self.total_minutes,
        'stages': [s.to_dict() for s in self.stages],
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any] | None) -> 'DoomConfig':
    """Builds a DoomConfig from a serialized dictionary."""
    payload = payload or {}
    raw_stages = payload.get('stages', [])
    stages = [
        DoomStage.from_dict(s, i)
        for i, s in enumerate(raw_stages)
        if isinstance(s, Mapping)
    ]
    if not stages:
      stages = [
          DoomStage(0, 'calm', 0.0, ''),
          DoomStage(1, 'warning signs', 0.5, ''),
          DoomStage(2, 'catastrophe', 1.0, ''),
      ]
    stages.sort(key=lambda s: s.at_fraction)
    return cls(
        name=str(payload.get('name', 'the reckoning')),
        total_minutes=int(payload.get('total_minutes', 16 * 60)),
        stages=stages,
    )


@dataclasses.dataclass
class ActRule:
  """Rule for advancing to a given act.

  Attributes:
    act: Act number this rule unlocks.
    truths_discovered_gte: Minimum truths discovered.
    doom_stage_gte: Minimum doom stage.
    tone_note: One line of guidance the GM receives while in this act.
  """

  act: int
  truths_discovered_gte: int = 0
  doom_stage_gte: int = 0
  tone_note: str = ''

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'ActRule':
    return cls(
        act=int(payload['act']),
        truths_discovered_gte=int(payload.get('truths_discovered_gte', 0)),
        doom_stage_gte=int(payload.get('doom_stage_gte', 0)),
        tone_note=str(payload.get('tone_note', '')),
    )


@dataclasses.dataclass
class RoutineSlot:
  """A time window during which an NPC is normally at a given place.

  Attributes:
    start_hour: Inclusive start hour (0-23).
    end_hour: Exclusive end hour (1-24). May wrap past midnight if < start.
    location_name: Human-readable location name (normalized at runtime).
    activity: What they do there.
  """

  start_hour: int
  end_hour: int
  location_name: str
  activity: str = ''

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'RoutineSlot':
    return cls(
        start_hour=int(payload.get('start_hour', 0)),
        end_hour=int(payload.get('end_hour', 24)),
        location_name=str(payload.get('location_name', '')),
        activity=str(payload.get('activity', '')),
    )

  def covers_hour(self, hour: int) -> bool:
    """Returns True if `hour` (0-23) falls within this slot."""
    hour %= 24
    if self.start_hour <= self.end_hour:
      return self.start_hour <= hour < self.end_hour
    return hour >= self.start_hour or hour < self.end_hour


@dataclasses.dataclass
class SeedNpc:
  """An authored starting NPC.

  Attributes:
    name: Unique character name.
    role_summary: Concise persona and visual description.
    goal: Active objective.
    traits: Speech register and temperament.
    location_name: Where they start.
    tier: 'hot', 'warm', or 'cold'.
    faction: Faction id.
    secret: What they hide.
    wound: Why they hide it and what would make them talk.
    formative_memories: Seed memories.
    routine: Daily schedule slots.
    relationships: Initial disposition notes keyed by entity name.
    sub_minds: Authored NPC Sub-Minds keyed by voice name.
  """

  name: str
  role_summary: str
  goal: str
  traits: str
  location_name: str
  tier: str = 'cold'
  faction: str = ''
  secret: str = ''
  wound: str = ''
  formative_memories: list[str] = dataclasses.field(default_factory=list)
  routine: list[RoutineSlot] = dataclasses.field(default_factory=list)
  relationships: dict[str, str] = dataclasses.field(default_factory=dict)
  sub_minds: dict[str, SubMindState] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'SeedNpc':
    raw_sub_minds = payload.get('sub_minds', {})
    sub_minds: dict[str, SubMindState] = {}
    if isinstance(raw_sub_minds, Mapping):
      for k, v in raw_sub_minds.items():
        if isinstance(v, Mapping):
          sm_dict = dict(v)
          sm_dict.setdefault('name', str(k))
          sm_dict.setdefault('origin', 'npc')
          sub_minds[str(k)] = SubMindState.from_dict(sm_dict)
    return cls(
        name=str(payload['name']),
        role_summary=str(payload.get('role_summary', '')),
        goal=str(payload.get('goal', '')),
        traits=str(payload.get('traits', '')),
        location_name=str(payload.get('location_name', '')),
        tier=str(payload.get('tier', 'cold')).lower(),
        faction=str(payload.get('faction', '')),
        secret=str(payload.get('secret', '')),
        wound=str(payload.get('wound', '')),
        formative_memories=_str_list(payload.get('formative_memories')),
        routine=[
            RoutineSlot.from_dict(r)
            for r in payload.get('routine', [])
            if isinstance(r, Mapping)
        ],
        relationships={
            str(k): str(v)
            for k, v in dict(payload.get('relationships', {})).items()
        },
        sub_minds=sub_minds,
    )


@dataclasses.dataclass
class StartingLocation:
  """The authored starting room."""

  name: str
  description: str = ''
  affordances: list[str] = dataclasses.field(default_factory=list)
  exits: dict[str, str] = dataclasses.field(default_factory=dict)
  clue_ids: list[str] = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'StartingLocation':
    return cls(
        name=str(payload['name']),
        description=str(payload.get('description', '')),
        affordances=_str_list(payload.get('affordances')),
        exits={
            str(k): str(v) for k, v in dict(payload.get('exits', {})).items()
        },
        clue_ids=_str_list(payload.get('clue_ids')),
    )


@dataclasses.dataclass
class WorldBible:
  """Everything authored about a world, loaded from JSON.

  Attributes:
    title: World title.
    premise: World premise and tone paragraph.
    opening: Opening narration template; may reference {start_name}.
    tone: Tone knobs.
    player_voices: Inner-voice faculty names and one-line personalities.
    player_sub_minds: Rich SubMindState definitions keyed by voice name.
    character_presets: Authored character-creation rank presets.
    emergent_voice_pool: Candidate Sub-Minds that can emerge during play.
    starting_location: Authored starting room.
    starting_inventory: Initial items in the player's inventory.
    factions: Factions.
    truths: Hidden truths.
    clues: Clues supporting truths.
    seed_npcs: Starting NPCs.
    storylets: Authored storylets.
    doom: Countdown config.
    acts: Act advancement rules.
  """

  title: str
  premise: str
  opening: str = ''
  tone: ToneConfig = dataclasses.field(default_factory=ToneConfig)
  player_voices: dict[str, str] = dataclasses.field(default_factory=dict)
  player_sub_minds: dict[str, SubMindState] = dataclasses.field(
      default_factory=dict
  )
  character_presets: list[dict[str, Any]] = dataclasses.field(
      default_factory=list
  )
  emergent_voice_pool: list[SubMindState] = dataclasses.field(
      default_factory=list
  )
  starting_location: StartingLocation | None = None
  starting_inventory: list[str] = dataclasses.field(default_factory=list)
  factions: list[Faction] = dataclasses.field(default_factory=list)
  truths: list[Truth] = dataclasses.field(default_factory=list)
  clues: list[Clue] = dataclasses.field(default_factory=list)
  seed_npcs: list[SeedNpc] = dataclasses.field(default_factory=list)
  storylets: list[Storylet] = dataclasses.field(default_factory=list)
  doom: DoomConfig = dataclasses.field(default_factory=DoomConfig)
  acts: list[ActRule] = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'WorldBible':
    """Deserializes a `WorldBible` from a dictionary."""
    start_raw = payload.get('starting_location')
    raw_pv = dict(payload.get('player_voices', {}))
    player_voices: dict[str, str] = {}
    player_sub_minds: dict[str, SubMindState] = {}
    for k, v in raw_pv.items():
      v_name = str(k).strip()
      if isinstance(v, Mapping):
        sm_dict = dict(v)
        sm_dict.setdefault('name', v_name)
        sm_dict.setdefault('origin', 'setting')
        sm = SubMindState.from_dict(sm_dict)
        player_sub_minds[v_name] = sm
        player_voices[v_name] = sm.worldview
      else:
        player_voices[v_name] = str(v)
    raw_psm = payload.get('player_sub_minds') or payload.get('voice_catalog')
    if isinstance(raw_psm, Mapping):
      for k, v in raw_psm.items():
        if isinstance(v, Mapping):
          sm_dict = dict(v)
          sm_dict.setdefault('name', str(k).strip())
          sm_dict.setdefault('origin', 'setting')
          sm = SubMindState.from_dict(sm_dict)
          player_sub_minds[sm.name] = sm
          player_voices.setdefault(sm.name, sm.worldview)
    elif isinstance(raw_psm, Sequence) and not isinstance(raw_psm, str):
      for item in raw_psm:
        if isinstance(item, Mapping) and item.get('name'):
          sm_dict = dict(item)
          sm_dict.setdefault('origin', 'setting')
          sm = SubMindState.from_dict(sm_dict)
          player_sub_minds[sm.name] = sm
          player_voices.setdefault(sm.name, sm.worldview)

    raw_truths = (
        payload.get('truths')
        or payload.get('narrative_arcs')
        or payload.get('turning_points')
        or []
    )
    raw_clues = payload.get('clues') or payload.get('catalysts') or []
    raw_presets = payload.get('character_presets', [])
    character_presets = [dict(p) for p in raw_presets if isinstance(p, Mapping)]
    raw_emergent = payload.get('emergent_voice_pool', [])
    emergent_pool = [
        SubMindState.from_dict({**dict(e), 'origin': 'emergent'})
        for e in raw_emergent
        if isinstance(e, Mapping) and e.get('name')
    ]
    return cls(
        title=str(payload.get('title', 'Untitled World')),
        premise=str(payload.get('premise', '')),
        opening=str(payload.get('opening', '')),
        tone=ToneConfig.from_dict(payload.get('tone')),
        player_voices=player_voices,
        player_sub_minds=player_sub_minds,
        character_presets=character_presets,
        emergent_voice_pool=emergent_pool,
        starting_location=(
            StartingLocation.from_dict(start_raw)
            if isinstance(start_raw, Mapping)
            else None
        ),
        starting_inventory=_str_list(payload.get('starting_inventory')),
        factions=[
            Faction.from_dict(f)
            for f in payload.get('factions', [])
            if isinstance(f, Mapping)
        ],
        truths=[
            Truth.from_dict(t) for t in raw_truths if isinstance(t, Mapping)
        ],
        clues=[Clue.from_dict(c) for c in raw_clues if isinstance(c, Mapping)],
        seed_npcs=[
            SeedNpc.from_dict(n)
            for n in payload.get('seed_npcs', [])
            if isinstance(n, Mapping)
        ],
        storylets=[
            Storylet.from_dict(s)
            for s in payload.get('storylets', [])
            if isinstance(s, Mapping)
        ],
        doom=DoomConfig.from_dict(payload.get('doom')),
        acts=[
            ActRule.from_dict(a)
            for a in payload.get('acts', [])
            if isinstance(a, Mapping)
        ],
    )

  @classmethod
  def from_json_path(cls, path: str) -> 'WorldBible':
    with open(path, 'r', encoding='utf-8') as f:
      payload = json.load(f)
    return cls.from_dict(payload)


# --- Truth-leak detection for player-facing text -----------------------------
#
# Anything the player can read before solving a thread (opening narration, role
# briefing, truth questions and stakes, authored voice lines) must not restate
# the thread's answer. The check is lexical: tokens of a truth statement that
# do not already occur in the world's public layer (premise, starting room,
# NPC names/roles/traits) are that truth's *private vocabulary*; a text that
# reuses enough of it is restating the answer. Paraphrased reveals (clone vs.
# duplicate) escape this check; `author_world` adds a model-based audit when
# a model is available.

_LEAK_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
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
    'three',
    'four',
    'five',
    'man',
    'men',
    'night',
    'morning',
    'hours',
    'days',
    'ago',
    'tonight',
    'whether',
})
LEAK_MIN_PRIVATE_HITS = 3
LEAK_MIN_PRIVATE_FRACTION = 0.4


def distinctive_tokens(text: str) -> set[str]:
  """Lower-cased tokens of `text` that carry content (>= 4 chars or digits)."""
  return {
      t
      for t in _LEAK_WORD_RE.findall(text.lower())
      if t not in _LEAK_STOPWORDS and (len(t) >= 4 or t.isdigit())
  }


def public_vocabulary(bible: WorldBible) -> set[str]:
  """Tokens the player legitimately knows at tick 0.

  Premise, starting room (name, description, affordances), and every seed
  NPC's name, public role, and traits. Secrets, goals, wounds, clue texts, and
  truth statements are excluded.

  Args:
    bible: Authored world.

  Returns:
    Set of distinctive tokens.
  """
  parts = [bible.premise]
  if bible.starting_location is not None:
    parts.append(bible.starting_location.name)
    parts.append(bible.starting_location.description)
    parts.extend(bible.starting_location.affordances)
  for npc in bible.seed_npcs:
    parts.extend([npc.name, npc.role_summary, npc.traits])
  return distinctive_tokens(' '.join(parts))


def truth_leaks_in_text(
    text: str,
    truths: Sequence[Truth],
    vocabulary: set[str],
) -> list[dict[str, Any]]:
  """Finds truths whose private vocabulary `text` reuses.

  Args:
    text: Player-facing text to check.
    truths: Truths to protect.
    vocabulary: Public-layer tokens (see `public_vocabulary`); statement tokens
      in this set are not counted as private.

  Returns:
    One dict per flagged truth: `truth_id`, `hits` (private tokens found),
    `fraction` of private vocabulary reused, and `severity` in
    {'leak', 'warn'}. 'leak' means >= LEAK_MIN_PRIVATE_HITS tokens or
    >= LEAK_MIN_PRIVATE_FRACTION; 'warn' means exactly two tokens.
  """
  text_tokens = distinctive_tokens(text)
  findings: list[dict[str, Any]] = []
  for truth in truths:
    private = distinctive_tokens(truth.statement) - vocabulary
    if not private:
      continue
    hits = sorted(private & text_tokens)
    fraction = len(hits) / len(private)
    if (
        len(hits) >= LEAK_MIN_PRIVATE_HITS
        or fraction >= LEAK_MIN_PRIVATE_FRACTION
    ):
      severity = 'leak'
    elif len(hits) >= 2:
      severity = 'warn'
    else:
      continue
    findings.append({
        'truth_id': truth.id,
        'hits': hits,
        'fraction': round(fraction, 3),
        'severity': severity,
    })
  return findings


def player_facing_texts(bible: WorldBible) -> dict[str, str]:
  """Texts shown to the player before any truth is solved, by label."""
  texts = {'opening': bible.opening}
  for truth in bible.truths:
    if truth.question:
      texts[f'truth[{truth.id}].question'] = truth.question
    if truth.stakes:
      texts[f'truth[{truth.id}].stakes'] = truth.stakes
  for storylet in bible.storylets:
    if storylet.voice_line:
      texts[f'storylet[{storylet.id}].voice_line'] = storylet.voice_line
  return texts


def find_bible_leaks(
    bible: WorldBible, *, extra_texts: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
  """Runs `truth_leaks_in_text` over every player-facing text in the bible.

  Args:
    bible: Authored world.
    extra_texts: Additional label -> text pairs to check (e.g. the role
      briefing, which lives outside the bible).

  Returns:
    Findings with an added `text` label, 'leak' severity first.
  """
  vocab = public_vocabulary(bible)
  texts = dict(player_facing_texts(bible))
  texts.update(extra_texts or {})
  findings: list[dict[str, Any]] = []
  for label, text in texts.items():
    for finding in truth_leaks_in_text(text, bible.truths, vocab):
      findings.append({'text': label, **finding})
  findings.sort(key=lambda f: (f['severity'] != 'leak', f['text']))
  return findings


def validate_bible(bible: WorldBible) -> list[str]:
  """Returns a list of human-readable problems; empty means valid."""
  problems: list[str] = []
  if not bible.premise.strip():
    problems.append('premise is empty')
  if bible.starting_location is None:
    problems.append('starting_location is missing')

  faction_ids = {f.id for f in bible.factions}
  truth_ids = {t.id for t in bible.truths}
  clue_ids = {c.id for c in bible.clues}
  npc_names = {n.name for n in bible.seed_npcs}
  storylet_ids: set[str] = set()

  for f in bible.factions:
    for other in f.standing:
      if other not in faction_ids:
        problems.append(f'faction {f.id} has standing toward unknown {other}')

  for c in bible.clues:
    if c.truth_id not in truth_ids:
      problems.append(f'clue {c.id} references unknown truth {c.truth_id}')
    if c.carrier not in CLUE_CARRIERS:
      problems.append(f'clue {c.id} has unknown carrier {c.carrier!r}')
    if (
        c.carrier == 'npc'
        and c.carrier_hint
        and c.carrier_hint not in npc_names
    ):
      # Allowed: the hint may name an NPC to be generated later. Warn only.
      pass

  for t in bible.truths:
    for cid in t.clue_ids:
      if cid not in clue_ids:
        problems.append(f'truth {t.id} references unknown clue {cid}')
    if t.clue_ids and t.min_clues_to_discover > len(t.clue_ids):
      problems.append(
          f'truth {t.id} requires {t.min_clues_to_discover} clues but has '
          f'{len(t.clue_ids)}'
      )
  for c in bible.clues:
    owner = next((t for t in bible.truths if c.id in t.clue_ids), None)
    if owner is None:
      problems.append(f'clue {c.id} is not listed by any truth')

  for n in bible.seed_npcs:
    if n.faction and n.faction not in faction_ids:
      problems.append(f'seed npc {n.name} has unknown faction {n.faction}')
    if n.tier not in ('hot', 'warm', 'cold'):
      problems.append(f'seed npc {n.name} has invalid tier {n.tier!r}')

  for s in bible.storylets:
    if s.id in storylet_ids:
      problems.append(f'duplicate storylet id {s.id}')
    storylet_ids.add(s.id)
    if s.trigger not in TRIGGERS:
      problems.append(f'storylet {s.id} has unknown trigger {s.trigger!r}')
    if not s.beat.strip():
      problems.append(f'storylet {s.id} has an empty beat')
    for cond in s.preconditions:
      if len(cond) != 1:
        problems.append(f'storylet {s.id} has a malformed condition {cond}')
        continue
      key = next(iter(cond))
      if key not in CONDITION_KEYS:
        problems.append(f'storylet {s.id} uses unknown condition {key!r}')
      value = cond[key]
      if (
          key in ('clue_discovered', 'clue_undiscovered')
          and value not in clue_ids
      ):
        problems.append(f'storylet {s.id} references unknown clue {value}')
      if key == 'clue_at_location':
        if not (isinstance(value, Sequence) and len(value) == 2):
          problems.append(
              f'storylet {s.id} malformed clue_at_location: {value}'
          )
        elif value[0] not in clue_ids:
          problems.append(f'storylet {s.id} references unknown clue {value[0]}')
      if (
          key in ('truth_discovered', 'truth_undiscovered')
          and value not in truth_ids
      ):
        problems.append(f'storylet {s.id} references unknown truth {value}')
      if key in ('faction_standing_lt', 'faction_standing_gt'):
        if not (isinstance(value, Sequence) and len(value) == 2):
          problems.append(f'storylet {s.id} malformed {key}: {value}')
        elif value[0] not in faction_ids:
          problems.append(
              f'storylet {s.id} references unknown faction {value[0]}'
          )
      if key == 'inter_faction_standing_lte':
        if not (isinstance(value, Sequence) and len(value) == 3):
          problems.append(f'storylet {s.id} malformed {key}: {value}')
        elif value[0] not in faction_ids or value[1] not in faction_ids:
          problems.append(
              f'storylet {s.id} references unknown faction in {value}'
          )
    for eff in s.effects:
      if len(eff) != 1:
        problems.append(f'storylet {s.id} has a malformed effect {eff}')
        continue
      key = next(iter(eff))
      if key not in EFFECT_KEYS:
        problems.append(f'storylet {s.id} uses unknown effect {key!r}')
      value = eff[key]
      if key == 'reveal_clue' and value not in clue_ids:
        problems.append(f'storylet {s.id} reveals unknown clue {value}')
      if key == 'move_clue':
        if not (isinstance(value, Sequence) and len(value) == 2):
          problems.append(f'storylet {s.id} malformed move_clue: {value}')
        elif value[0] not in clue_ids:
          problems.append(f'storylet {s.id} moves unknown clue {value[0]}')
      if key == 'faction_shift':
        if not (isinstance(value, Sequence) and len(value) == 2):
          problems.append(f'storylet {s.id} malformed faction_shift: {value}')
        elif value[0] not in faction_ids:
          problems.append(f'storylet {s.id} shifts unknown faction {value[0]}')
      if key == 'inter_faction_shift':
        if not (isinstance(value, Sequence) and len(value) == 3):
          problems.append(
              f'storylet {s.id} malformed inter_faction_shift: {value}'
          )
        elif value[0] not in faction_ids or value[1] not in faction_ids:
          problems.append(f'storylet {s.id} shifts unknown faction in {value}')
    if s.voice and bible.player_voices and s.voice not in bible.player_voices:
      problems.append(f'storylet {s.id} uses unknown voice {s.voice!r}')

  if bible.doom.stages and abs(bible.doom.stages[-1].at_fraction - 1.0) > 1e-6:
    problems.append('last doom stage must have at_fraction == 1.0')

  for a in bible.acts:
    if a.act < 1:
      problems.append(f'act rule has invalid act number {a.act}')

  for finding in find_bible_leaks(bible):
    if finding['severity'] == 'leak':
      problems.append(
          f"{finding['text']} restates truth {finding['truth_id']} "
          f"(private tokens {finding['hits']})"
      )

  return problems


@dataclasses.dataclass
class StoryState:
  """Mutable runtime story state, owned by `WorldLedger`.

  Attributes:
    tone: Tone knobs.
    player_voices: Inner-voice faculties.
    voice_affinity: Dominance level (1-10) per faculty.
    sub_minds: First-class HRL Sub-Mind objects keyed by faculty name.
    character_initialized: True once the player's Parliament of Sub-Minds has
      been explicitly initialized.
    active_voice_options: Current sub-mind council options offered to the
      player.
    voice_option_log: Per-refresh record of the options offered: dicts with
      `tick`, `location_id`, and `options` (each with `voice`, `action`,
      `perception`, `source` in {'llm', 'fallback'}). Lets the evaluator grade
      option freshness and diversity after the run.
    voice_choice_log: Record of player actions that invoked a faculty: dicts
      with `tick`, `voice`, and `via` in {'prefix', 'option_match'}.
    truths: Truths keyed by id.
    clues: Clues keyed by id.
    factions: Factions keyed by id.
    storylets: Storylets keyed by id.
    fired: Count of times each storylet has fired.
    fired_log: Chronological record of storylet firings, one dict per firing
      with keys `storylet_id`, `tick`, and `source`. Used by the evaluator to
      align fired beats with the prose resolved at the same tick.
    tension: Current dramatic tension (0-10).
    doom: Countdown config.
    doom_start_minutes: Clock minutes when the countdown began.
    doom_stage: Current stage index.
    act: Current act number.
    acts: Act rules.
    journal: Player-facing log of what has been learned.
    visited_location_ids: Locations the player has entered.
    clues_relocated_count: Count of off-screen clue relocations.
    ended: True once an ending has fired.
    ending_summary: Final epilogue text, if ended.
    ending_archetype: Machine-readable ending archetype id.
    ending_title: Display title of the triggered ending.
    ending_slides: Epilogue slides summarizing world and faction fates.
    world_title: Title of the active world bible.
    world_premise: Premise text of the active world bible.
    world_opening: Formatted opening narration of the active world bible.
    character_presets: Character-creation rank presets for the active world.
    emergent_voice_pool: Candidate Sub-Minds that can emerge during play.
  """

  tone: ToneConfig = dataclasses.field(default_factory=ToneConfig)
  player_voices: dict[str, str] = dataclasses.field(default_factory=dict)
  voice_affinity: dict[str, int] = dataclasses.field(default_factory=dict)
  sub_minds: dict[str, SubMindState] = dataclasses.field(default_factory=dict)
  character_initialized: bool = False
  active_voice_options: list[VoiceOption] = dataclasses.field(
      default_factory=list
  )
  voice_option_log: list[dict[str, Any]] = dataclasses.field(
      default_factory=list
  )
  voice_choice_log: list[dict[str, Any]] = dataclasses.field(
      default_factory=list
  )
  truths: dict[str, Truth] = dataclasses.field(default_factory=dict)
  clues: dict[str, Clue] = dataclasses.field(default_factory=dict)
  factions: dict[str, Faction] = dataclasses.field(default_factory=dict)
  storylets: dict[str, Storylet] = dataclasses.field(default_factory=dict)
  fired: dict[str, int] = dataclasses.field(default_factory=dict)
  fired_log: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  tension: float = 2.0
  doom: DoomConfig = dataclasses.field(default_factory=DoomConfig)
  doom_start_minutes: int = 0
  doom_stage: int = 0
  act: int = 1
  acts: list[ActRule] = dataclasses.field(default_factory=list)
  journal: list[str] = dataclasses.field(default_factory=list)
  visited_location_ids: list[str] = dataclasses.field(default_factory=list)
  clues_relocated_count: int = 0
  ended: bool = False
  ending_summary: str = ''
  ending_archetype: str = ''
  ending_title: str = ''
  ending_slides: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  world_title: str = ''
  world_premise: str = ''
  world_opening: str = ''
  character_presets: list[dict[str, Any]] = dataclasses.field(
      default_factory=list
  )
  emergent_voice_pool: list[SubMindState] = dataclasses.field(
      default_factory=list
  )

  def get_player_role_info(self) -> dict[str, str]:
    """Returns `{title, summary}` explaining the player's role and objective."""
    blob = (
        f'{self.tone.name} {self.world_title} {self.world_premise} '
        f'{self.world_opening}'
    ).lower()
    if (
        't_inside_job' in self.truths
        or 'bermondsey' in blob
        or 'corrigan' in blob
    ):
      return {
          'title': 'Independent Bermondsey Fixer',
          'summary': (
              'Summoned by East End governor "The Alderman" Arthur Corrigan '
              'with his empty pigskin holdall. Before the 11:00 PM last-orders '
              'bell, uncover who stitched up the Rotherhithe van job, track '
              "down the engraved presentation pistols, the Butcher's Bill "
              'bribe ledger, and the prize greyhound Duchess—and decide who '
              'takes the fall.'
          ),
      }
    if 't_valve' in self.truths or 'glassharbor' in blob:
      return {
          'title': 'Maritime Inquiry Inspector',
          'summary': (
              'Summoned to Glassharbor with an unsealed inquiry writ after '
              'the port magistrate vanished. Before the midnight king-tide '
              '(12:00 AM), find out why the floodgate is failing and who '
              'profits from it, then decide whether to expose the '
              'conspirators to the Watch, broker a backroom deal, or find '
              'another way to spill the tide.'
          ),
      }
    if (
        't_oxygen_cull' in self.truths
        or 't_purge_manifest' in self.truths
        or 'dystopian_scifi' in blob
        or 'outpost 7' in blob
        or 'sector 9' in blob
        or 'eidolon' in blob
    ):
      return {
          'title': 'Outpost Relief Marshal',
          'summary': (
              'Arrived at four-person weather station Outpost 7 just as the '
              'supply crate revealed a frozen corpse wearing the winter parka '
              'and brass transit tag of chief mechanic Jonas Reed—who is '
              'sitting in the mess hall drinking black coffee with a three-day '
              'gap in his memory. The roof radio is dead. Before the 06:00 AM '
              'company transport lands, work out who the dead man in Jonas’s '
              'parka is, why Jonas lost three days, who silenced the station, '
              'and who walks onto the morning shuttle alive.'
          ),
      }
    if (
        't_forbidden_bond' in self.truths
        or 'vampire_romance' in blob
        or 'st. jude' in blob
        or 'house valerius' in blob
        or 'solstice masque' in blob
    ):
      return {
          'title': 'Night-Shift Blood Bank Tech',
          'summary': (
              'You and Claire have survived for forty years by working '
              'hospital graveyard shifts, living off expiring O-negative '
              'bags, and moving cities every five years. Tonight at St. '
              'Jude’s Hospital, three blood bags are missing, young ER '
              'resident Dr. David Park has a bandaged neck and keeps asking '
              'for Claire, and retired detective Arthur Finch is at the '
              'security desk with forty years of your old ID photos. Before '
              'sunrise and the 06:00 AM shift change, decide how to handle '
              'Finch and who boards the morning train.'
          ),
      }
    # Extract role and objective from `world_opening` if present.
    m = re.search(
        r'You are (?:an?\s+)?([^.—]+)[.—]\s*(.+)',
        self.world_opening,
        re.IGNORECASE,
    )
    if m:
      return {
          'title': m.group(1).strip().title(),
          'summary': m.group(2).strip(),
      }
    return {
        'title': 'Protagonist Investigator',
        'summary': (
            self.world_opening.strip()
            or self.world_premise.strip()
            or 'Uncover the hidden turning points before the countdown runs out.'
        ),
    }

  @classmethod
  def _normalize_preset_dict(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Ensures every preset has `id`, `key`, `name`, `label`, `tagline`, `description`, `ranks`."""
    del cls
    p = dict(raw)
    pid = str(p.get('id') or p.get('key') or 'preset').strip()
    pname = str(p.get('name') or p.get('label') or pid).strip()
    p['id'] = pid
    p['key'] = pid
    p['name'] = pname
    p['label'] = pname
    p.setdefault('tagline', pname)
    p.setdefault('description', '')
    return p

  def get_character_presets(self) -> list[dict[str, Any]]:
    """Returns authored or dynamically synthesized character creation presets."""
    if self.character_presets:
      return [self._normalize_preset_dict(p) for p in self.character_presets]
    voices = list(self.sub_minds.values())
    if not voices:
      return []
    tone_low = (
        f'{self.tone.name} {self.world_title} {self.world_premise}'.lower()
    )
    if any(
        w in tone_low for w in ('bermondsey', 'caper', 'corrigan', 'snooker')
    ):
      return [
          self._normalize_preset_dict({
              'id': 'balanced',
              'name': 'Balanced Fixer',
              'tagline': "Brass Neck 2 · The Book 2 · Copper's Nose 2",
              'description': (
                  'All-round Southwark fixer who can bluff a doorman, audit a '
                  'bent layoff pad, and read Flying Squad procedure.'
              ),
              'ranks': {
                  'Brass Neck': 2,
                  'The Book': 2,
                  "Copper's Nose": 2,
                  'Proper Naughty': 1,
                  'Ear to the Ground': 1,
                  'The Hustle': 1,
                  "The Guv'nor": 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'heavy',
              'name': 'Heavy Muscle & Brass',
              'tagline': "Proper Naughty 4 · Brass Neck 2 · The Guv'nor 1",
              'description': (
                  'Enforcer build: jemmies padlocks, backs heavies into '
                  'corners, and pulls syndicate rank.'
              ),
              'ranks': {
                  'Proper Naughty': 4,
                  'Brass Neck': 2,
                  "The Guv'nor": 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'auditor',
              'name': 'Forensic Bookkeeper',
              'tagline': "The Book 4 · Copper's Nose 2 · Ear to the Ground 1",
              'description': (
                  'Follows the paper trail, spots shaved margins and police '
                  'wires, and catches liars on arithmetic.'
              ),
              'ranks': {
                  'The Book': 4,
                  "Copper's Nose": 2,
                  'Ear to the Ground': 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'hustler',
              'name': 'Street Hustler',
              'tagline': 'The Hustle 4 · Ear to the Ground 2 · Brass Neck 1',
              'description': (
                  'Plays rival crews off against each other and brokers '
                  'three-way swaps before the bell.'
              ),
              'ranks': {
                  'The Hustle': 4,
                  'Ear to the Ground': 2,
                  'Brass Neck': 1,
              },
          }),
      ]
    if 'glassharbor' in tone_low or 't_valve' in self.truths:
      return [
          self._normalize_preset_dict({
              'id': 'balanced_inspector',
              'name': 'Balanced Inquiry Inspector',
              'tagline': (
                  'Ledger 2 · Suspicion 2 · Empathy 2 · Salt-Memory 1 · Nerve 1'
              ),
              'description': (
                  'Well-rounded maritime investigator trained to audit port'
                  ' ledgers, read nervous witnesses, and weather dockside'
                  ' danger.'
              ),
              'ranks': {
                  'Ledger': 2,
                  'Suspicion': 2,
                  'Empathy': 2,
                  'Salt-Memory': 1,
                  'Nerve': 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'port_auditor',
              'name': 'Forensic Port Auditor',
              'tagline': 'Ledger 4 · Suspicion 3 · Salt-Memory 1',
              'description': (
                  'Obsessive archive and manifest examiner who spots scraped '
                  'tide-tables, replaced brass seals, and forged signatures.'
              ),
              'ranks': {
                  'Ledger': 4,
                  'Suspicion': 3,
                  'Salt-Memory': 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'pier_confessor',
              'name': 'Salt-Pier Confessor',
              'tagline': 'Empathy 4 · Salt-Memory 3 · Suspicion 1',
              'description': (
                  'Raised on salt-piers like Glassharbor; hears the family fear'
                  " behind a Warden's lie and speaks the smugglers' cadence."
              ),
              'ranks': {
                  'Empathy': 4,
                  'Salt-Memory': 3,
                  'Suspicion': 1,
              },
          }),
          self._normalize_preset_dict({
              'id': 'watch_inquisitor',
              'name': 'Harbor Watch Enforcer',
              'tagline': 'Nerve 4 · Suspicion 3 · Ledger 1',
              'description': (
                  'Direct, unflinching investigator who forces bolted shed '
                  'doors and backs saboteurs against the seawall.'
              ),
              'ranks': {
                  'Nerve': 4,
                  'Suspicion': 3,
                  'Ledger': 1,
              },
          }),
      ]
    names = [v.name for v in voices]
    presets: list[dict[str, Any]] = []
    bal_ranks = {n: 2 if i < 3 else 1 for i, n in enumerate(names[:4])}
    presets.append(
        self._normalize_preset_dict({
            'id': 'balanced',
            'name': 'Balanced Protagonist',
            'tagline': 'Balanced across top faculties',
            'description': (
                'Evenly weighted inner parliament across your core faculties.'
            ),
            'ranks': bal_ranks,
        })
    )
    for idx, v in enumerate(voices[:3]):
      role_label = v.short_role() or v.name
      if len(role_label) > 26:
        role_label = v.name
      second = names[(idx + 1) % len(names)]
      third = names[(idx + 2) % len(names)] if len(names) > 2 else second
      spec_ranks: dict[str, int] = {v.name: 4}
      if second != v.name:
        spec_ranks[second] = 2
      if third not in spec_ranks:
        spec_ranks[third] = 1
      presets.append(
          self._normalize_preset_dict({
              'id': f'specialist_{idx + 1}',
              'name': role_label,
              'tagline': f'{v.name} 4 · {second} 2',
              'description': v.worldview,
              'ranks': spec_ranks,
          })
      )
    return presets

  def __post_init__(self) -> None:
    if self.player_voices and not self.sub_minds:
      for k, desc in self.player_voices.items():
        rank = max(1, int(self.voice_affinity.get(k, 1)))
        self.sub_minds[k] = SubMindState.from_legacy(k, desc, rank=rank)
    elif self.sub_minds and not self.player_voices:
      self.sync_voice_maps()

  def sync_voice_maps(self) -> None:
    """Synchronizes `player_voices`, `voice_affinity`, and `intrusive` flags."""
    if not self.sub_minds:
      return
    active_ranks = [sm.rank for sm in self.sub_minds.values() if sm.active] or [
        1
    ]
    min_rank = min(active_ranks)
    max_rank = max(active_ranks)
    new_voices: dict[str, str] = {}
    new_affinity: dict[str, int] = {}
    for name, sm in self.sub_minds.items():
      sm.rank = max(1, min(10, int(sm.rank)))
      sm.intrusive = sm.active and (
          sm.rank >= 4 or (sm.rank == max_rank and sm.rank - min_rank >= 3)
      )
      if sm.active:
        new_voices[name] = sm.worldview
        new_affinity[name] = sm.rank
    if new_voices:
      self.player_voices = new_voices
      self.voice_affinity = new_affinity

  def get_sub_mind(self, voice_name: str) -> SubMindState | None:
    """Looks up a SubMindState by case-insensitive name."""
    cleaned = voice_name.strip()
    if not cleaned:
      return None
    if cleaned in self.sub_minds:
      return self.sub_minds[cleaned]
    lower = cleaned.lower()
    for k, sm in self.sub_minds.items():
      if k.lower() == lower or k.lower() in lower:
        return sm
    alias_map = {
        "punter's heart": 'Ear to the Ground',
        'punters heart': 'Ear to the Ground',
        'old bill': "Copper's Nose",
    }
    alias_target = alias_map.get(lower, '')
    if alias_target and alias_target in self.sub_minds:
      return self.sub_minds[alias_target]
    return None

  def register_sub_mind(self, sub_mind: SubMindState) -> SubMindState:
    """Adds or updates a SubMindState on the character sheet."""
    self.sub_minds[sub_mind.name] = sub_mind
    self.sync_voice_maps()
    return sub_mind

  def configure_parliament(
      self,
      specs: Sequence[Mapping[str, Any]] = (),
      *,
      active_ranks: Mapping[str, int] | None = None,
      custom_voices: Sequence[Mapping[str, Any] | SubMindState] = (),
      custom_sub_minds: Sequence[Mapping[str, Any] | SubMindState] = (),
      tick: int = 0,
  ) -> list[SubMindState]:
    """Configures the player's Sub-Mind parliament from character creation."""
    for sm in self.sub_minds.values():
      sm.active = False

    combined_specs: list[dict[str, Any]] = [
        {str(k): v for k, v in s.items()}
        for s in specs
        if isinstance(s, Mapping)
    ]
    if active_ranks:
      for v_name, v_rank in active_ranks.items():
        r = int(v_rank)
        combined_specs.append(
            {'name': str(v_name), 'rank': max(1, r), 'active': r > 0}
        )
    for item in list(custom_voices) + list(custom_sub_minds):
      if isinstance(item, SubMindState):
        self.register_sub_mind(item)
      elif isinstance(item, Mapping):
        combined_specs.append({str(k): v for k, v in item.items()})

    for raw in combined_specs:
      name = str(raw.get('name', '')).strip()
      if not name:
        continue
      active = bool(raw.get('active', True))
      rank = max(1, min(10, int(raw.get('rank', 1))))
      existing = self.get_sub_mind(name)
      if existing is not None:
        existing.active = active
        existing.rank = rank
        existing.intrusive = rank >= 4
        if raw.get('worldview'):
          existing.worldview = str(raw['worldview']).strip()
        if raw.get('blind_spot'):
          existing.blind_spot = str(raw['blind_spot']).strip()
        if raw.get('obsession'):
          existing.obsession = str(raw['obsession']).strip()
        for scar in _str_list(raw.get('scars')):
          if scar not in existing.scars:
            existing.scars.append(scar)
      else:
        worldview = str(
            raw.get('worldview')
            or raw.get('description')
            or 'Custom inner voice guiding your instincts.'
        ).strip()
        def_dom, def_blind, def_trig = infer_submind_defaults(name, worldview)
        sm = SubMindState(
            name=name,
            domain=str(raw.get('domain') or def_dom).strip(),
            worldview=worldview,
            blind_spot=str(raw.get('blind_spot') or def_blind).strip(),
            rank=rank,
            xp=0,
            attention_triggers=(
                _str_list(raw.get('attention_triggers')) or def_trig
            ),
            scars=_str_list(raw.get('scars')),
            obsession=str(raw.get('obsession', '')).strip(),
            origin=str(raw.get('origin', 'custom')).strip() or 'custom',
            active=active,
            intrusive=rank >= 4,
        )
        self.sub_minds[name] = sm
    if not any(sm.active for sm in self.sub_minds.values()):
      for sm in list(self.sub_minds.values())[:4]:
        sm.active = True
    self.character_initialized = True
    self.sync_voice_maps()
    active_list = [sm for sm in self.sub_minds.values() if sm.active]
    summary_names = ', '.join(f'{sm.name} (Lv {sm.rank})' for sm in active_list)
    self.journal.append(
        f'[tick {tick}] Character Psyche Initialized: {summary_names}'
    )
    return active_list

  @classmethod
  def from_bible(
      cls, bible: WorldBible, *, start_clock_minutes: int = 0
  ) -> 'StoryState':
    """Projects an authored bible into fresh runtime state."""
    sub_minds: dict[str, SubMindState] = {}
    for k, desc in bible.player_voices.items():
      if k in bible.player_sub_minds:
        sub_minds[k] = dataclasses.replace(
            bible.player_sub_minds[k],
            attention_triggers=list(
                bible.player_sub_minds[k].attention_triggers
            ),
            scars=list(bible.player_sub_minds[k].scars),
            strong_against=list(bible.player_sub_minds[k].strong_against),
            vulnerable_to=list(bible.player_sub_minds[k].vulnerable_to),
            favored_tactics=list(bible.player_sub_minds[k].favored_tactics),
        )
      else:
        sub_minds[k] = SubMindState.from_legacy(
            k, desc, rank=1, origin='setting'
        )
    for k, sm in bible.player_sub_minds.items():
      if k not in sub_minds:
        sub_minds[k] = dataclasses.replace(
            sm,
            attention_triggers=list(sm.attention_triggers),
            scars=list(sm.scars),
            strong_against=list(sm.strong_against),
            vulnerable_to=list(sm.vulnerable_to),
            favored_tactics=list(sm.favored_tactics),
        )
    player_voices = {k: sm.worldview for k, sm in sub_minds.items()} or dict(
        bible.player_voices
    )
    start_room_name = (
        bible.starting_location.name
        if bible.starting_location is not None
        else 'the room'
    )
    if bible.opening:
      world_opening = bible.opening.replace('{start_name}', start_room_name)
    else:
      world_opening = ''
    state = cls(
        tone=dataclasses.replace(
            bible.tone,
            room_palette=list(bible.tone.room_palette),
            npc_archetypes=list(bible.tone.npc_archetypes),
        ),
        player_voices=player_voices,
        voice_affinity={k: sm.rank for k, sm in sub_minds.items()},
        sub_minds=sub_minds,
        truths={t.id: dataclasses.replace(t) for t in bible.truths},
        clues={c.id: dataclasses.replace(c) for c in bible.clues},
        factions={
            f.id: dataclasses.replace(
                f,
                standing=dict(f.standing),
                assets=list(f.assets),
                disposition=f.disposition or f.current_disposition(),
                relations=dict(f.relations),
                recent_actions=list(f.recent_actions),
            )
            for f in bible.factions
        },
        storylets={s.id: dataclasses.replace(s) for s in bible.storylets},
        doom=DoomConfig.from_dict(bible.doom.to_dict()),
        doom_start_minutes=start_clock_minutes,
        acts=[dataclasses.replace(a) for a in bible.acts],
        world_title=bible.title,
        world_premise=bible.premise,
        world_opening=world_opening,
        character_presets=[
            cls._normalize_preset_dict(p) for p in bible.character_presets
        ],
        emergent_voice_pool=[
            dataclasses.replace(
                e,
                attention_triggers=list(e.attention_triggers),
                scars=list(e.scars),
                strong_against=list(e.strong_against),
                vulnerable_to=list(e.vulnerable_to),
                favored_tactics=list(e.favored_tactics),
            )
            for e in bible.emergent_voice_pool
        ],
    )
    if not state.character_presets:
      state.character_presets = state.get_character_presets()
    return state

  # --- Derived queries -------------------------------------------------------

  def truths_discovered(self) -> int:
    return sum(1 for t in self.truths.values() if t.discovered)

  def clues_discovered(self) -> int:
    return sum(1 for c in self.clues.values() if c.discovered)

  def storylets_fired(self) -> int:
    return sum(self.fired.values())

  def doom_fraction(self, clock_minutes: int) -> float:
    if self.doom.total_minutes <= 0:
      return 0.0
    elapsed = max(0, clock_minutes - self.doom_start_minutes)
    return min(1.0, elapsed / float(self.doom.total_minutes))

  def doom_minutes_remaining(self, clock_minutes: int) -> int:
    elapsed = max(0, clock_minutes - self.doom_start_minutes)
    return max(0, self.doom.total_minutes - elapsed)

  def stage_for_fraction(self, fraction: float) -> DoomStage:
    current = self.doom.stages[0]
    for stage in self.doom.stages:
      if fraction + 1e-9 >= stage.at_fraction:
        current = stage
    return current

  def unplaced_clues(self) -> list[Clue]:
    return [c for c in self.clues.values() if c.placed_at is None]

  def clues_at(self, placed_at: str) -> list[Clue]:
    return [c for c in self.clues.values() if c.placed_at == placed_at]

  def faction(self, faction_id: str) -> Faction | None:
    return self.factions.get(faction_id)

  def inter_faction_standing(self, fac_a: str, fac_b: str) -> int:
    """Returns the standing between two factions (default 0)."""
    fa = self.factions.get(fac_a)
    fb = self.factions.get(fac_b)
    if fa is not None and fac_b in fa.standing:
      return int(fa.standing[fac_b])
    if fb is not None and fac_a in fb.standing:
      return int(fb.standing[fac_a])
    return 0

  def inter_faction_relation(self, fac_a: str, fac_b: str) -> str:
    """Returns the natural-language relation between two factions."""
    fa = self.factions.get(fac_a)
    fb = self.factions.get(fac_b)
    if fa is not None and fac_b in fa.relations and fa.relations[fac_b].strip():
      return fa.relations[fac_b].strip()
    if fb is not None and fac_a in fb.relations and fb.relations[fac_a].strip():
      return fb.relations[fac_a].strip()
    val = self.inter_faction_standing(fac_a, fac_b)
    if val <= -4:
      return 'Open turf war: crews trading reprisals and shooting on sight.'
    if val < 0:
      return 'Simmering turf friction and mutual suspicion.'
    if val > 0:
      return 'Quiet working understanding.'
    return 'Uneasy stand-off.'

  def active_turf_wars(
      self, *, threshold: int = -4
  ) -> list[tuple[str, str, int]]:
    """Returns pairs `(fac_a, fac_b, standing)` in active open turf war."""
    wars: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    fids = list(self.factions.keys())
    for i, fa in enumerate(fids):
      for fb in fids[i + 1 :]:
        pair = (min(fa, fb), max(fa, fb))
        if pair in seen:
          continue
        seen.add(pair)
        val = self.inter_faction_standing(fa, fb)
        rel_low = self.inter_faction_relation(fa, fb).lower()
        is_nl_war = any(
            w in rel_low
            for w in (
                'open turf war',
                'open war',
                'shooting on sight',
                'shoot on sight',
                'blood feud',
                'active reprisals',
            )
        )
        if val <= threshold or is_nl_war:
          wars.append((fa, fb, min(val, threshold) if is_nl_war else val))
    return wars

  def describe_turf_wars(self, *, threshold: int = -4) -> list[str]:
    """Returns natural-language descriptions of active inter-faction wars."""
    out: list[str] = []
    for fa, fb, _ in self.active_turf_wars(threshold=threshold):
      na = self.factions[fa].name if fa in self.factions else fa
      nb = self.factions[fb].name if fb in self.factions else fb
      rel = self.inter_faction_relation(fa, fb)
      out.append(f'{na} vs {nb} ({rel.rstrip(".")})')
    return out

  # --- Mutations (pure bookkeeping; the Director decides when) ---------------

  def mark_clue_discovered(self, clue_id: str, tick: int) -> Truth | None:
    """Marks a clue found; returns the truth if this discovery completed it."""
    clue = self.clues.get(clue_id)
    if clue is None or clue.discovered:
      return None
    clue.discovered = True
    clue.discovered_tick = tick
    self.journal.append(f'[tick {tick}] Learned: {clue.text}')
    truth = self.truths.get(clue.truth_id)
    if truth is None or truth.discovered:
      return None
    found = sum(
        1
        for cid in truth.clue_ids
        if self.clues.get(cid, Clue('', '', '')).discovered
    )
    if truth.min_clues_to_discover and found >= truth.min_clues_to_discover:
      truth.discovered = True
      truth.discovered_tick = tick
      self.journal.append(f'[tick {tick}] Understood: {truth.statement}')
      return truth
    return None

  def shift_faction(
      self,
      faction_id: str,
      delta: int | str,
      *,
      reason: str = '',
      tick: int = -1,
  ) -> int | None:
    """Updates a faction's natural-language disposition and internal polarity."""
    fac = self.factions.get(faction_id)
    if fac is None:
      return None
    nl_note = reason.strip()
    num_delta = 0
    if isinstance(delta, (int, float)):
      num_delta = int(delta)
    else:
      raw_str = str(delta).strip()
      try:
        num_delta = int(raw_str)
      except ValueError:
        nl_note = raw_str
        num_delta = infer_faction_polarity_delta(raw_str)
    fac.standing_with_player = max(
        -5, min(5, fac.standing_with_player + num_delta)
    )
    if nl_note:
      fac.disposition = nl_note
      entry = (
          f'[tick {tick}] {nl_note}'
          if tick >= 0 and not nl_note.startswith('[')
          else nl_note
      )
      if not fac.recent_actions or fac.recent_actions[-1] != entry:
        fac.recent_actions.append(entry)
        if len(fac.recent_actions) > 8:
          fac.recent_actions = fac.recent_actions[-8:]
    elif not fac.disposition or fac.disposition.startswith((
        'Wary neutrality',
        'Cautiously receptive',
        'Cut into a working alliance',
        'Suspicious and defensive',
        'Openly hostile',
    )):
      fac.disposition = fac.current_disposition()
    return fac.standing_with_player

  def shift_inter_faction(
      self,
      fac_a: str,
      fac_b: str,
      delta: int | str,
      *,
      reason: str = '',
      tick: int = -1,
  ) -> int | None:
    """Updates mutual natural-language relation and polarity between two factions."""
    if not fac_a or not fac_b or fac_a == fac_b:
      return None
    fa = self.factions.get(fac_a)
    fb = self.factions.get(fac_b)
    if fa is None or fb is None:
      return None
    nl_note = reason.strip()
    if isinstance(delta, (int, float)):
      num_delta = int(delta)
    else:
      raw_str = str(delta).strip()
      try:
        num_delta = int(raw_str)
      except ValueError:
        nl_note = raw_str
        num_delta = infer_faction_polarity_delta(raw_str) or -1
    current = self.inter_faction_standing(fac_a, fac_b)
    updated = max(-5, min(5, current + num_delta))
    fa.standing[fac_b] = updated
    fb.standing[fac_a] = updated
    if not nl_note:
      if updated <= -4:
        nl_note = (
            f'Open turf war between {fa.name} and {fb.name}: crews trading '
            'reprisals and shooting on sight.'
        )
      elif updated < 0:
        nl_note = f'Escalating turf friction between {fa.name} and {fb.name}.'
      elif updated > 0:
        nl_note = f'Working truce between {fa.name} and {fb.name}.'
    if nl_note:
      fa.relations[fac_b] = nl_note
      fb.relations[fac_a] = nl_note
      entry = (
          f'[tick {tick}] {nl_note}'
          if tick >= 0 and not nl_note.startswith('[')
          else nl_note
      )
      for f_obj in (fa, fb):
        if not f_obj.recent_actions or f_obj.recent_actions[-1] != entry:
          f_obj.recent_actions.append(entry)
          if len(f_obj.recent_actions) > 8:
            f_obj.recent_actions = f_obj.recent_actions[-8:]
    return updated

  def shift_tension(self, delta: float) -> float:
    self.tension = max(0.0, min(10.0, self.tension + delta))
    return self.tension

  def note_fired(
      self, storylet_id: str, *, tick: int = -1, source: str = ''
  ) -> None:
    """Records one firing of `storylet_id`.

    Args:
      storylet_id: Storylet that fired.
      tick: Ledger tick at firing time; -1 when the caller has no tick.
      source: Who triggered the firing (acting entity, 'StoryDirector', ...).
    """
    self.fired[storylet_id] = self.fired.get(storylet_id, 0) + 1
    self.fired_log.append(
        {'storylet_id': storylet_id, 'tick': int(tick), 'source': source}
    )

  def note_visited(self, location_id: str) -> None:
    if location_id not in self.visited_location_ids:
      self.visited_location_ids.append(location_id)

  def bump_voice_affinity(self, voice_name: str, delta: int = 1) -> int | None:
    """Increments the dominance/affinity level of a sub-mind faculty."""
    cleaned = voice_name.strip()
    if not cleaned:
      return None
    canonical = None
    for k in self.player_voices:
      if k.lower() == cleaned.lower():
        canonical = k
        break
    if canonical is None:
      sm_match = self.get_sub_mind(cleaned)
      if sm_match is not None:
        canonical = sm_match.name
        sm_match.active = True
    if canonical is None:
      return None
    cur = int(self.voice_affinity.get(canonical, 1))
    updated = max(1, min(10, cur + int(delta)))
    self.voice_affinity[canonical] = updated
    sm = self.sub_minds.get(canonical)
    if sm is not None:
      sm.rank = updated
      sm.xp += max(0, int(delta))
    self.sync_voice_maps()
    for opt in self.active_voice_options:
      if opt.voice == canonical:
        opt.affinity = updated
    return updated

  def record_voice_outcome(
      self,
      voice_name: str,
      *,
      tick: int = 0,
      location_name: str = 'the scene',
      payoff: bool = False,
      backfired: bool | None = None,
      summary: str = '',
      scar: str = '',
      obsession: str = '',
  ) -> SubMindState | None:
    """Records a formative scar and updates obsession/intrusiveness on a voice."""
    sm = self.get_sub_mind(voice_name)
    if sm is None:
      return None
    if backfired is not None:
      payoff = not backfired
    if scar.strip():
      scar_entry = scar.strip()
    else:
      kind_label = 'Vindicated' if payoff else 'Burned'
      clean_sum = summary.strip().rstrip('.')
      if len(clean_sum) > 96:
        clean_sum = clean_sum[:93].rsplit(' ', 1)[0] + '...'
        if clean_sum.count('(') > clean_sum.count(')'):
          clean_sum += ')'
      scar_entry = f'[Tick {tick} @ {location_name}] {kind_label}: {clean_sum}.'
    if scar_entry not in sm.scars:
      sm.scars.append(scar_entry)
      if len(sm.scars) > 6:
        sm.scars = sm.scars[-6:]
    if payoff:
      sm.xp += 1
    if obsession.strip():
      sm.obsession = obsession.strip()
    elif not payoff:
      sm.obsession = f'Settle the score at {location_name} after being stalled.'
    elif payoff and not sm.obsession:
      sm.obsession = f'Press the {sm.domain} angle opened at {location_name}.'
    self.sync_voice_maps()
    # Mark neglected rank-1 voices as resentful when a dominant voice hits 4+.
    max_rank = max(
        (m.rank for m in self.sub_minds.values() if m.active), default=1
    )
    if max_rank >= 4:
      dom_names = [
          m.name
          for m in self.sub_minds.values()
          if m.active and m.rank == max_rank
      ]
      dom_label = dom_names[0] if dom_names else sm.name
      domain_resentments = {
          'deduction': (
              f'Cataloging the physical trace evidence {dom_label} rushed '
              f'past at {location_name}.'
          ),
          'forensics': (
              f'Reconstructing the timeline discrepancy {dom_label} '
              f'overlooked at {location_name}.'
          ),
          'authority': (
              f'Watching {dom_label} yield tactical ground at '
              f'{location_name}; ready to force a hard breach.'
          ),
          'tactical_force': (
              f'Measuring the sightlines and lock-points {dom_label} left '
              f'unguarded at {location_name}.'
          ),
          'empathy': (
              f'Tracking the unspoken terror {dom_label} is ignoring at '
              f'{location_name}.'
          ),
          'paranoia': (
              f'Listening for the trap behind the walls at {location_name} '
              f'while {dom_label} fixates on the obvious.'
          ),
          'accounting': (
              f'Auditing the off-ledger liabilities {dom_label} is racking '
              f'up at {location_name}.'
          ),
          'hierarchy': (
              f'Logging the protocol violations {dom_label} triggered at '
              f'{location_name}.'
          ),
          'hustle': (
              f'Pricing the side-angle {dom_label} left on the table at '
              f'{location_name}.'
          ),
          'leverage': (
              f'Waiting to turn the wedge {dom_label} missed at '
              f'{location_name} into a private cut.'
          ),
          'synthesis': (
              f'Scanning for the physiological tell {dom_label} missed at '
              f'{location_name}.'
          ),
          'cybernetics': (
              f'Cross-checking the biometric drift at {location_name} that '
              f'{dom_label} refused to read.'
          ),
      }
      for idx_other, other in enumerate(self.sub_minds.values()):
        if other.active and other.rank == 1 and not other.obsession:
          dom_key = (other.domain or '').strip().lower()
          other.obsession = domain_resentments.get(
              dom_key,
              f"Tracking what {dom_label}'s blind spot missed at "
              f'{location_name} ({other.domain or f"angle #{idx_other + 1}"}).',
          )
    return sm

  def recompute_act(self) -> int:
    """Advances `act` according to the act rules; never regresses."""
    discovered = self.truths_discovered()
    best = self.act
    for rule in self.acts:
      if (
          discovered >= rule.truths_discovered_gte
          and self.doom_stage >= rule.doom_stage_gte
          and rule.act > best
      ):
        best = rule.act
    self.act = best
    return self.act

  def current_act_note(self) -> str:
    for rule in self.acts:
      if rule.act == self.act:
        return rule.tone_note
    return ''

  # --- Serialization ---------------------------------------------------------

  def to_dict(self) -> dict[str, Any]:
    return {
        'tone': self.tone.to_dict(),
        'player_voices': dict(self.player_voices),
        'voice_affinity': dict(self.voice_affinity),
        'sub_minds': {k: v.to_dict() for k, v in self.sub_minds.items()},
        'character_initialized': self.character_initialized,
        'active_voice_options': [
            v.to_dict() for v in self.active_voice_options
        ],
        'voice_option_log': [dict(e) for e in self.voice_option_log],
        'voice_choice_log': [dict(e) for e in self.voice_choice_log],
        'truths': {k: v.to_dict() for k, v in self.truths.items()},
        'clues': {k: v.to_dict() for k, v in self.clues.items()},
        'factions': {k: v.to_dict() for k, v in self.factions.items()},
        'storylets': {k: v.to_dict() for k, v in self.storylets.items()},
        'fired': dict(self.fired),
        'fired_log': [dict(entry) for entry in self.fired_log],
        'tension': self.tension,
        'doom': self.doom.to_dict(),
        'doom_start_minutes': self.doom_start_minutes,
        'doom_stage': self.doom_stage,
        'act': self.act,
        'acts': [a.to_dict() for a in self.acts],
        'journal': list(self.journal),
        'visited_location_ids': list(self.visited_location_ids),
        'clues_relocated_count': self.clues_relocated_count,
        'ended': self.ended,
        'ending_summary': self.ending_summary,
        'ending_archetype': self.ending_archetype,
        'ending_title': self.ending_title,
        'ending_slides': [dict(s) for s in self.ending_slides],
        'world_title': self.world_title,
        'world_premise': self.world_premise,
        'world_opening': self.world_opening,
        'player_role': self.get_player_role_info(),
        'character_presets': [dict(p) for p in self.get_character_presets()],
        'emergent_voice_pool': [e.to_dict() for e in self.emergent_voice_pool],
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any] | None) -> 'StoryState':
    """Restores a StoryState from a serialized dictionary."""
    if not payload:
      return cls()
    player_voices = {
        str(k): str(v)
        for k, v in dict(payload.get('player_voices', {})).items()
    }
    raw_affinity = dict(payload.get('voice_affinity', {}))
    voice_affinity = {
        k: max(1, int(raw_affinity.get(k, 1))) for k in player_voices
    }
    for k, v in raw_affinity.items():
      if str(k) not in voice_affinity:
        voice_affinity[str(k)] = max(1, int(v))
    raw_sub_minds = dict(payload.get('sub_minds', {}))
    sub_minds: dict[str, SubMindState] = {}
    for k, v in raw_sub_minds.items():
      if isinstance(v, Mapping):
        sub_minds[str(k)] = SubMindState.from_dict(v)
    if not sub_minds and player_voices:
      for k, desc in player_voices.items():
        sub_minds[k] = SubMindState.from_legacy(
            k, desc, rank=voice_affinity.get(k, 1)
        )
    return cls(
        tone=ToneConfig.from_dict(payload.get('tone')),
        player_voices=player_voices,
        voice_affinity=voice_affinity,
        sub_minds=sub_minds,
        character_initialized=bool(payload.get('character_initialized', False)),
        active_voice_options=[
            VoiceOption.from_dict(v)
            for v in payload.get('active_voice_options', [])
            if isinstance(v, Mapping)
        ],
        voice_option_log=[
            dict(e)
            for e in payload.get('voice_option_log', [])
            if isinstance(e, Mapping)
        ],
        voice_choice_log=[
            dict(e)
            for e in payload.get('voice_choice_log', [])
            if isinstance(e, Mapping)
        ],
        truths={
            str(k): Truth.from_dict(v)
            for k, v in dict(payload.get('truths', {})).items()
        },
        clues={
            str(k): Clue.from_dict(v)
            for k, v in dict(payload.get('clues', {})).items()
        },
        factions={
            str(k): Faction.from_dict(v)
            for k, v in dict(payload.get('factions', {})).items()
        },
        storylets={
            str(k): Storylet.from_dict(v)
            for k, v in dict(payload.get('storylets', {})).items()
        },
        fired={
            str(k): int(v) for k, v in dict(payload.get('fired', {})).items()
        },
        fired_log=[
            {
                'storylet_id': str(entry.get('storylet_id', '')),
                'tick': int(entry.get('tick', -1)),
                'source': str(entry.get('source', '')),
            }
            for entry in payload.get('fired_log', [])
            if isinstance(entry, Mapping)
        ],
        tension=float(payload.get('tension', 2.0)),
        doom=DoomConfig.from_dict(payload.get('doom')),
        doom_start_minutes=int(payload.get('doom_start_minutes', 0)),
        doom_stage=int(payload.get('doom_stage', 0)),
        act=int(payload.get('act', 1)),
        acts=[
            ActRule.from_dict(a)
            for a in payload.get('acts', [])
            if isinstance(a, Mapping)
        ],
        journal=_str_list(payload.get('journal')),
        visited_location_ids=_str_list(payload.get('visited_location_ids')),
        clues_relocated_count=int(payload.get('clues_relocated_count', 0)),
        ended=bool(payload.get('ended', False)),
        ending_summary=str(payload.get('ending_summary', '')),
        ending_archetype=str(payload.get('ending_archetype', '')),
        ending_title=str(payload.get('ending_title', '')),
        ending_slides=[
            dict(s)
            for s in payload.get('ending_slides', [])
            if isinstance(s, Mapping)
        ],
        world_title=str(payload.get('world_title', '')),
        world_premise=str(payload.get('world_premise', '')),
        world_opening=str(payload.get('world_opening', '')),
        character_presets=[
            cls._normalize_preset_dict(p)
            for p in payload.get('character_presets', [])
            if isinstance(p, Mapping)
        ],
        emergent_voice_pool=[
            SubMindState.from_dict(e)
            for e in payload.get('emergent_voice_pool', [])
            if isinstance(e, Mapping)
        ],
    )


_SHIPPED_BIBLES: dict[str, str] = {
    'glassharbor': 'glassharbor.json',
    'bermondsey': 'bermondsey.json',
    'bermondsey_drop': 'bermondsey.json',
    'snatch': 'bermondsey.json',
}


def resolve_asset_path(*parts: str) -> str:
  """Resolves an Adventure asset path in both open-source and monorepo layouts."""
  here = os.path.dirname(os.path.abspath(__file__))
  local_path = os.path.join(here, *parts)
  if os.path.exists(local_path):
    return local_path
  opensource_path = os.path.normpath(
      os.path.join(
          here, '..', '..', 'opensource', 'examples', 'adventure', *parts
      )
  )
  if os.path.exists(opensource_path):
    return opensource_path
  return local_path


def resolve_bible_path(spec: str) -> str:
  """Resolves a shipped bible key ('bermondsey', 'glassharbor') or file path."""
  key = spec.strip().lower()
  if key in _SHIPPED_BIBLES:
    return resolve_asset_path('content', _SHIPPED_BIBLES[key])
  return spec


def load_shipped_bible(name: str) -> WorldBible:
  """Loads one of the shipped world bibles ('bermondsey' or 'glassharbor')."""
  return WorldBible.from_json_path(resolve_bible_path(name))
