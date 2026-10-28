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
"""Thread-safe world state ledger for the infinite adventure simulation.

Stores the location graph (expanded rooms and frontier exit stubs), the
simulated world clock, global facts/rumors, player state, and the three-tier
NPC registry (Hot, Warm, Cold), bridged to a ReactiveX TopicBus.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import enum
import re
import threading
import time
from typing import Any, cast, override

from concordia.contrib.pubsub import topics
from examples.adventure import story_state
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

DEFAULT_WORLD_LEDGER_COMPONENT_KEY = '__world_ledger__'
DEFAULT_STORY_DIRECTOR_COMPONENT_KEY = '__story_director__'


class NpcTier(enum.Enum):
  """Execution tier for an NPC in the adventure simulation."""

  HOT = 'hot'
  WARM = 'warm'
  COLD = 'cold'


def normalize_location_id(name: str) -> str:
  """Converts a human-readable location name into a canonical slug ID."""
  cleaned = re.sub(r'[^a-z0-9]+', '_', name.strip().lower()).strip('_')
  return cleaned or 'unknown_location'


_RAW_CLUE_TAG_RE = re.compile(
    r'\s*(?:displaying|concealing|containing|holding|revealing|marked)?\s*'
    r'[\[\(]c_[a-z0-9_]+[\]\)]',
    re.IGNORECASE,
)

_CLUE_ITEM_LABELS: dict[str, str] = {
    'c_valve_scratches': 'rubbing of the valve housing tool scratches',
    'c_pressure_chart': "Soren Vale's ciphered pressure chart",
    'c_soren_commission': 'Compact commission chit',
    'c_warden_count': 'warden floodgate key-ring log',
    'c_manifest': 'copied 2 AM customs manifest entry',
    'c_ilsa_debt': "note of Ilsa Vance's tavern tally debt",
    'c_seal_replaced': 'replaced brass customs seal',
    'c_vance_confession': "Captain Vance's signed statement",
    'c_schematic_stub': 'torn Folio 9 index card',
    'c_guild_receipt': 'twelve-year-old guild receipt',
    'c_bypass_wheel': 'notes on the bricked west-wall wheelhouse',
    'c_orin_knows': 'Folio 9 manual bypass schematics',
    'c_tide_ledger': 'scraped tide-ledger entry notes',
    'c_boathouse_prep': 'notes on the empty Compact boats',
    'c_runner_slip': "Compact runner's folded slip",
    'c_layoff_carbon': "Frankie's carbon-copy Catford layoff slip",
    'c_frankie_pager': 'Frankie Slater’s pager message log',
    'c_van_key_wax': 'wax impression of the Corrigan van ignition key',
    'c_blagger_confession': (
        'sworn word from the locked-up Rotherhithe blaggers'
    ),
    'c_cue_case_shells': 'pair of brass-hulled .455 target cartridges',
    'c_chiller_manifest': 'Spitalfields chiller-van delivery docket',
    'c_gun_oil_rag': 'rangoon-oiled gunsmith rag from the chiller floor',
    'c_reuben_appraisal': "Reuben Vane's diamond-swap valuation slip",
    'c_jersey_sort_code': 'torn Butcher’s Bill page 14 bank sort code',
    'c_rover_radio_log': 'Flying Squad Rover SD1 call-sign log',
    'c_scorched_folio': 'half-charred Butcher’s Bill ledger cover',
    'c_finch_admission': 'DI Finch’s taped confession on the Jersey account',
    'c_boot_polish_tin': (
        'tin of Parade Gloss black boot polish and ear stencil'
    ),
    'c_catford_trap_card': 'altered Catford Trap 4 race-card for "Biscuit"',
    'c_vet_dockets': 'Deptford kennel sedation & dye dockets',
    'c_desmond_slip': 'Desmond’s 12-to-1 syndicate betting voucher',
}


def clean_affordance_text(text: str) -> str:
  """Strips internal clue IDs like `[c_...]` from visible affordance strings."""
  cleaned = _RAW_CLUE_TAG_RE.sub('', text.strip())
  cleaned = re.sub(r'\bc_[a-z0-9_]+\b:?\s*', '', cleaned, flags=re.IGNORECASE)
  return re.sub(r'\s+', ' ', cleaned).strip(' -—,:;')


@dataclasses.dataclass
class LocationNode:
  """A node in the world location graph.

  Attributes:
    location_id: Canonical slug identifier (e.g. 'rusty_anchor_tavern').
    name: Human-readable title (e.g. 'The Rusty Anchor Tavern').
    description: Sensory and architectural description of the location.
    affordances: Specific interactable props, clues, or mechanisms present.
    exits: Mapping from exit label/direction to target location_id.
    expanded: Whether FrontierArchitect has fully generated this location.
    recent_events: Bounded list of recent narrative events in this location.
    clue_ids: Story clues physically placed in this location.
    doom_variants: Description overrides keyed by doom stage index (as str).
    faction_lean: Faction id that dominates this location, if any.
    image_url: Relative URL for the location's generated establishing shot.
  """

  location_id: str
  name: str
  description: str = ''
  affordances: list[str] = dataclasses.field(default_factory=list)
  exits: dict[str, str] = dataclasses.field(default_factory=dict)
  expanded: bool = False
  recent_events: list[str] = dataclasses.field(default_factory=list)
  clue_ids: list[str] = dataclasses.field(default_factory=list)
  doom_variants: dict[str, str] = dataclasses.field(default_factory=dict)
  faction_lean: str = ''
  image_url: str = ''

  def to_dict(self) -> dict[str, Any]:
    """Serializes the location node to a dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'LocationNode':
    """Restores a LocationNode from a serialized dictionary."""
    return cls(
        location_id=str(payload['location_id']),
        name=str(payload['name']),
        description=str(payload.get('description', '')),
        affordances=[str(x) for x in payload.get('affordances', [])],
        exits={
            str(k): str(v) for k, v in dict(payload.get('exits', {})).items()
        },
        expanded=bool(payload.get('expanded', False)),
        recent_events=[str(x) for x in payload.get('recent_events', [])],
        clue_ids=[str(x) for x in payload.get('clue_ids', [])],
        doom_variants={
            str(k): str(v)
            for k, v in dict(payload.get('doom_variants', {})).items()
        },
        faction_lean=str(payload.get('faction_lean', '')),
        image_url=str(payload.get('image_url', '')),
    )


_NPC_STOPWORDS = frozenset({
    'about',
    'after',
    'against',
    'around',
    'before',
    'behind',
    'between',
    'could',
    'every',
    'from',
    'have',
    'into',
    'keep',
    'knows',
    'make',
    'more',
    'must',
    'never',
    'only',
    'other',
    'over',
    'same',
    'should',
    'some',
    'still',
    'such',
    'take',
    'than',
    'that',
    'their',
    'them',
    'then',
    'there',
    'these',
    'they',
    'this',
    'those',
    'through',
    'under',
    'until',
    'very',
    'want',
    'wants',
    'what',
    'when',
    'where',
    'which',
    'while',
    'with',
    'would',
    'your',
})


def _extract_npc_keywords(text: str, limit: int = 5) -> list[str]:
  """Extracts distinctive lowercase 4+ letter keywords from NPC metadata."""
  tokens = re.findall(r'[a-z]{4,}', (text or '').lower())
  out: list[str] = []
  for tok in tokens:
    if tok not in _NPC_STOPWORDS and tok not in out:
      out.append(tok)
      if len(out) >= limit:
        break
  return out


_NPC_ROLE_DOMAIN_ALIASES: dict[str, str] = {
    'the mask': 'stonewalling',
    'mask': 'stonewalling',
    'stonewalling': 'stonewalling',
    'facade': 'stonewalling',
    'persona': 'stonewalling',
    'decorum': 'stonewalling',
    'protocol': 'stonewalling',
    'the angle': 'leverage',
    'angle': 'leverage',
    'leverage': 'leverage',
    'drive': 'leverage',
    'desire': 'leverage',
    'passion': 'leverage',
    'ambition': 'leverage',
    'the wound': 'self_preservation',
    'wound': 'self_preservation',
    'self_preservation': 'self_preservation',
    'vulnerability': 'self_preservation',
    'shadow': 'self_preservation',
    'thirst': 'self_preservation',
}

_NPC_ROLE_ARCHETYPE_ALIASES: dict[str, str] = {
    'the mask': 'facade',
    'mask': 'facade',
    'stonewalling': 'facade',
    'facade': 'facade',
    'persona': 'facade',
    'the angle': 'drive',
    'angle': 'drive',
    'leverage': 'drive',
    'drive': 'drive',
    'desire': 'drive',
    'the wound': 'wound',
    'wound': 'wound',
    'self_preservation': 'wound',
    'vulnerability': 'wound',
    'shadow': 'wound',
}


class NpcSubMindMap(dict[str, story_state.SubMindState]):
  """Dictionary of an NPC's Sub-Minds with role archetype and domain lookup."""

  def _resolve_alias(self, key: Any) -> story_state.SubMindState | None:
    """Resolves a voice name, role archetype, or domain alias to a SubMindState."""
    if not isinstance(key, str):
      return None
    norm = key.strip().lower()
    if not norm:
      return None
    # 1. Case-insensitive voice name or direct domain match.
    for k, sm in dict.items(self):
      if (
          k.lower() == norm
          or sm.name.lower() == norm
          or sm.domain.lower() == norm
      ):
        return sm
    # 2. Structural role archetype match ('facade', 'drive', 'wound').
    target_arch = _NPC_ROLE_ARCHETYPE_ALIASES.get(norm)
    if target_arch:
      for sm in dict.values(self):
        if (sm.role_archetype or '').lower() == target_arch:
          return sm
    # 3. Legacy domain alias match ('stonewalling', 'leverage',
    # 'self_preservation').
    target_domain = _NPC_ROLE_DOMAIN_ALIASES.get(norm)
    if target_domain:
      for sm in dict.values(self):
        if sm.domain == target_domain:
          return sm
    # 4. Positional fallback for 3-slot NPC parliaments when custom domains
    # are used.
    vals = list(dict.values(self))
    if target_arch == 'facade' and len(vals) >= 1:
      return vals[0]
    if target_arch == 'drive' and len(vals) >= 2:
      return vals[1]
    if target_arch == 'wound' and len(vals) >= 3:
      return vals[2]
    return None

  @override
  def __contains__(self, key: object) -> bool:
    if super().__contains__(key):
      return True
    return self._resolve_alias(key) is not None

  @override
  def __getitem__(self, key: str) -> story_state.SubMindState:
    if super().__contains__(key):
      return super().__getitem__(key)
    resolved = self._resolve_alias(key)
    if resolved is not None:
      return resolved
    raise KeyError(key)

  @override
  def get(
      self, key: str, default: Any = None
  ) -> story_state.SubMindState | Any:
    if super().__contains__(key):
      return super().__getitem__(key)
    resolved = self._resolve_alias(key)
    if resolved is not None:
      return resolved
    return default


def _derive_npc_voice_names_and_ranks(
    *,
    name: str,
    role_summary: str,
    goal: str,
    traits: str,
    faction: str,
    secret: str,
    wound: str,
) -> tuple[tuple[str, int], tuple[str, int], tuple[str, int]]:
  """Derives character-specific voice names and starting ranks across any genre."""
  name_low = (name or '').strip().lower()
  combined = f'{name} {role_summary} {traits} {faction}'.lower()
  goal_lower = (goal or '').lower()
  wound_lower = f'{secret} {wound}'.lower()

  # Explicit canonical NPC profiles for Glassharbor, Sci-Fi, and Vampire Romance
  if 'mara vance' in name_low:
    return ("The Warden's Seal", 4), ('Quiet Settlement', 2), ("Ilsa's Debt", 2)
  if 'soren vale' in name_low:
    return (
        ('Clockwork Pride', 2),
        ('Safe Passage Out', 3),
        ('The Bleed-Valve Guilt', 4),
    )
  if 'archivist orin' in name_low or name_low == 'orin':
    return (
        ('Guild Procedure', 4),
        ("The Archive's Good Name", 2),
        ('Folio 9 Scandal', 3),
    )
  if 'ilsa vance' in name_low:
    return (
        ('Tavern Bravado', 2),
        ('Wiping the Tally', 3),
        ("Her Mother's Badge", 4),
    )
  if name_low == 'pell':
    return (
        ("Compact Runner's Grin", 2),
        ('The Midnight Boat-Cut', 4),
        ('Empty Crates on the Pier', 2),
    )
  if 'kaelen vos' in name_low:
    return (
        ('Directive Protocol', 4),
        ('Clean Containment Log', 2),
        ('Sister in the Clinic', 3),
    )
  if 'jonas reed' in name_low:
    return (
        ('Wrench & Coffee', 3),
        ('Seat on the Shuttle', 4),
        ('Three-Day Blank', 2),
    )
  if 'mara lin' in name_low:
    return (
        ('Clinical Calm', 4),
        ('Clean Transfer Log', 3),
        ('The Stopped Watch', 2),
    )
  if 'cole mercer' in name_low:
    return (
        ('Radio Static', 2),
        ('The Core Sample Buyer', 4),
        ('Bolt-Cutter Guilt', 3),
    )
  if 'splicer nyx' in name_low or name_low == 'nyx':
    return (
        ('Neural Firewall', 2),
        ('Sub-Grid Uplink', 4),
        ('Memory Wipe Trauma', 3),
    )
  if 'foreman rook' in name_low or name_low == 'rook':
    return (
        ('Sub-Deck Solidarity', 4),
        ('The Scrubber Spine', 3),
        ('Lost Parley Crew', 2),
    )
  if 'claire moreau' in name_low:
    return (
        ('Night-Shift Calm', 3),
        ('Save David Before Dawn', 4),
        ('Forty Years of Hiding', 2),
    )
  if 'david park' in name_low:
    return (
        ('ER Resident Composure', 2),
        ('A Ticket on the Train', 4),
        ('The Slowing Pulse', 3),
    )
  if 'arthur finch' in name_low or 'detective finch' in name_low:
    return (
        ('Tired Cop Routine', 4),
        ('The 1986 Photo File', 3),
        ('Room 412 Oncology', 2),
    )
  if 'julian vance' in name_low:
    return (
        ('Diplomatic Poise', 2),
        ('Forbidden Yearning', 4),
        ('Forsaken Heart', 3),
    )
  if 'seraphine valerius' in name_low:
    return (
        ('Velvet Decorum', 3),
        ('Forbidden Yearning', 4),
        ('The Starving Vein', 2),
    )
  if 'chamberlain malakor' in name_low or 'malakor' in name_low:
    return (
        ('Velvet Decorum', 4),
        ('Crimson Ambition', 3),
        ('The Starving Vein', 2),
    )

  is_scifi = any(
      w in combined or w in goal_lower or w in wound_lower
      for w in (
          'android',
          'bionic',
          'synth',
          'directorate',
          'orbital',
          'neural',
          'subnet',
          'wetware',
          'splicer',
          'netrunner',
          'airlock',
          'scrubber',
          'datashard',
          'engram',
          'eidolon',
          'sub-deck',
          'subdeck',
      )
  )
  is_vampire = not is_scifi and any(
      w in combined or w in goal_lower or w in wound_lower
      for w in (
          'vampire',
          'countess',
          'sire',
          'bloodline',
          'masquerade',
          'coven',
          'reliquary',
          'chalice',
          'vitae',
          'blood-moon',
          'valerius',
      )
  )

  # 1. Character-specific Facade / Mask voice names across genres.
  if is_scifi:
    if any(
        w in combined
        for w in (
            'splicer',
            'hacker',
            'netrunner',
            'courier',
            'technician',
            'clinic',
        )
    ):
      mask_name = 'Neural Firewall'
    elif any(
        w in combined
        for w in ('union', 'foreman', 'scrubber', 'rig', 'breacher')
    ):
      mask_name = 'Sub-Deck Solidarity'
    else:
      mask_name = 'Directive Protocol'
  elif is_vampire:
    if any(
        w in combined
        for w in (
            'inquisitor',
            'paladin',
            'templar',
            'priest',
            'cleric',
            'vow',
            'chivalry',
            'oath',
        )
    ):
      mask_name = 'Sacred Vow'
    elif 'mortal' in combined and 'envoy' in combined:
      mask_name = 'Diplomatic Poise'
    else:
      mask_name = 'Velvet Decorum'
  elif any(
      w in combined
      for w in ('inspector', 'sergeant', 'detective', 'squad', 'police', 'cib')
  ):
    mask_name = 'The Warrant Card'
  elif any(w in combined for w in ('warden', 'captain', 'harbor', 'watch')):
    mask_name = "The Warden's Seal"
  elif any(
      w in combined for w in ('patter', 'bookie', 'turf accountant', 'tout')
  ):
    mask_name = 'The Patter'
  elif any(
      w in combined
      for w in ('broker', 'diamond', 'jeweller', 'cutter', 'vault', 'vane')
  ):
    mask_name = 'Velvet & Glass'
  elif any(
      w in combined for w in ('archivist', 'scribe', 'chancery', 'keeper')
  ):
    mask_name = 'Guild Procedure'
  elif any(
      w in combined
      for w in ('veterinary', 'surgeon', 'doctor', 'clinical', 'kennel')
  ):
    mask_name = 'Clinical Steel'
  elif any(
      w in combined
      for w in ('scrap', 'heavy', 'enforcer', 'knuckle', 'boxer', 'brixton')
  ):
    mask_name = 'Iron Plate'
  elif any(
      w in combined
      for w in (
          'vampire',
          'countess',
          'lord',
          'lady',
          'sire',
          'bloodline',
          'masquerade',
          'coven',
          'courtier',
          'noble',
          'duchess',
          'prince',
      )
  ):
    mask_name = 'Velvet Decorum'
  elif any(
      w in combined
      for w in (
          'inquisitor',
          'paladin',
          'templar',
          'priest',
          'cleric',
          'vow',
          'chivalry',
          'oath',
      )
  ):
    mask_name = 'Sacred Vow'
  else:
    mask_name = 'The Mask'

  # 2. Character-specific Drive / Angle voice names across genres.
  if is_scifi:
    if any(
        w in goal_lower or w in combined
        for w in (
            'off-world',
            'shuttle',
            'orbit',
            'escape',
            'clearance',
            'colony',
            'contain',
        )
    ):
      angle_name = 'Off-World Transit'
    elif any(
        w in goal_lower or w in combined
        for w in (
            'datashard',
            'decrypt',
            'subnet',
            'upload',
            'broadcast',
            'firmware',
            'implant',
            'eidolon',
        )
    ):
      angle_name = 'Sub-Grid Uplink'
    else:
      angle_name = 'The Scrubber Spine'
  elif is_vampire:
    if any(
        w in goal_lower or w in combined
        for w in (
            'love',
            'lover',
            'romance',
            'heart',
            'desire',
            'seduce',
            'elope',
            'tryst',
            'passion',
            'devotion',
            'escape',
        )
    ):
      angle_name = 'Forbidden Yearning'
    else:
      angle_name = 'Crimson Ambition'
  elif any(
      w in goal_lower or w in combined
      for w in ('layoff', 'odds', 'greyhound', 'dog-track', 'betting', 'tote')
  ):
    angle_name = 'The Layoff'
  elif any(
      w in goal_lower or w in combined
      for w in ('boart', 'carat', 'diamond', 'wholesale', 'appraisal', 'margin')
  ):
    angle_name = 'The Wholesale Margin'
  elif any(
      w in goal_lower or w in combined
      for w in ('shakedown', 'retainer', 'kickback', 'pension', 'patrol')
  ):
    angle_name = 'The Shakedown'
  elif any(
      w in goal_lower or w in combined
      for w in ('fee', 'second half', 'kovacs', 'veterinary')
  ):
    angle_name = 'The Second Half'
  elif any(
      w in goal_lower or w in combined
      for w in ('pistol', 'shotgun', 'swap', 'carve', 'unload', 'handoff')
  ):
    angle_name = 'The Carve-Up'
  elif any(
      w in goal_lower or w in combined
      for w in (
          'love',
          'lover',
          'romance',
          'heart',
          'desire',
          'seduce',
          'elope',
          'tryst',
          'passion',
          'devotion',
      )
  ):
    angle_name = 'Forbidden Yearning'
  elif any(
      w in goal_lower or w in combined
      for w in (
          'throne',
          'succession',
          'coven',
          'crown',
          'blood-pact',
          'court',
          'heir',
      )
  ):
    angle_name = 'Crimson Ambition'
  else:
    angle_name = 'The Angle'

  # 3. Character-specific Wound / Shadow voice names across genres.
  if is_scifi:
    if any(
        w in wound_lower
        for w in (
            'wipe',
            'memory',
            'engram',
            'reset',
            'decommission',
            'glitch',
            'virus',
            'implant',
            'eidolon',
        )
    ):
      wound_name = 'Memory Wipe Trauma'
    elif 'sister' in wound_lower or 'family' in wound_lower:
      wound_name = 'Sister in the Clinic'
    else:
      wound_name = 'Sub-Deck Guilt'
  elif is_vampire:
    if 'mortal' in combined and 'envoy' in combined:
      wound_name = 'Forsaken Heart'
    elif any(
        w in wound_lower
        for w in (
            'thirst',
            'hunger',
            'starving',
            'frenzy',
            'feed',
            'vein',
            'blood',
            'sunrise',
            'execution',
        )
    ):
      wound_name = 'The Starving Vein'
    else:
      wound_name = 'Forsaken Heart'
  elif any(w in wound_lower for w in ('prison', 'cib', 'sort code', 'pension')):
    wound_name = 'Ford Open Prison'
  elif any(
      w in wound_lower for w in ('shortfall', 'audit', '60,000', 'bounced')
  ):
    wound_name = 'The Shortfall'
  elif any(
      w in wound_lower for w in ('parole', 'brother', 'container', 'crusher')
  ):
    wound_name = 'Blood & Parole'
  elif any(
      w in wound_lower for w in ('insurance', 'fire', 'torch', 'underwriter')
  ):
    wound_name = "The Underwriter's Fire"
  elif any(
      w in wound_lower for w in ('alibi', 'kennel', 'duchess', 'turpentine')
  ):
    wound_name = 'Kennel 3 Alibi'
  elif any(
      w in wound_lower
      for w in ('thirst', 'hunger', 'starving', 'frenzy', 'feed', 'vein')
  ):
    wound_name = 'The Starving Vein'
  elif any(
      w in wound_lower
      for w in (
          'betray',
          'forsaken',
          'jealous',
          'heartbreak',
          'vow',
          'forbidden',
      )
  ):
    wound_name = 'Forsaken Heart'
  else:
    wound_name = 'The Wound'

  # Ensure all 3 keys are distinct inside NpcSubMindMap.
  if angle_name == mask_name:
    angle_name = f'{angle_name} Drive'
  if wound_name in (mask_name, angle_name):
    wound_name = f'{wound_name} Scar'

  # 4. Personality-fitted starting ranks across genres.
  if any(
      w in combined
      for w in (
          'inspector',
          'warden',
          'captain',
          'sergeant',
          'bully',
          'deadpan',
          'menace',
          'heavy',
          'enforcer',
          'elder',
          'inquisitor',
          'commander',
          'director',
          'noble',
          'countess',
          'prince',
      )
  ):
    ranks = (4, 2, 1)
  elif any(
      w in combined
      for w in (
          'accountant',
          'bookie',
          'broker',
          'merchant',
          'patter',
          'margin',
          'fixer',
          'fence',
          'courtier',
          'siren',
          'diplomat',
          'splicer',
          'smuggler',
      )
  ):
    ranks = (2, 4, 2)
  elif any(
      w in combined or w in wound_lower
      for w in (
          'terrified',
          'sweating',
          'vet',
          'trainer',
          'cornered',
          'trembling',
          'alibi',
          'starving',
          'heartbroken',
          'glitched',
          'infected',
          'hunted',
      )
  ):
    ranks = (2, 3, 4)
  else:
    ranks = (3, 2, 1)

  return (
      (mask_name, ranks[0]),
      (angle_name, ranks[1]),
      (wound_name, ranks[2]),
  )


def synthesize_npc_sub_minds(
    *,
    name: str,
    role_summary: str,
    goal: str,
    traits: str,
    faction: str = '',
    secret: str = '',
    wound: str = '',
) -> NpcSubMindMap:
  """Synthesizes a 3-voice Parliament of Sub-Minds for an NPC."""
  role_clean = (role_summary or 'Local figure').strip().rstrip('.')
  traits_clean = (traits or 'guarded and calculating').strip().rstrip('.')
  goal_clean = (goal or 'secure an advantage and protect their own').strip()
  secret_clean = (secret or 'unspoken secret').strip()
  wound_clean = (wound or 'fears exposure and betrayal').strip()
  turf_label = faction.strip() if faction.strip() else 'personal'

  (mask_name, mask_rank), (angle_name, angle_rank), (wound_name, wound_rank) = (
      _derive_npc_voice_names_and_ranks(
          name=name,
          role_summary=role_clean,
          goal=goal_clean,
          traits=traits_clean,
          faction=faction,
          secret=secret_clean,
          wound=wound_clean,
      )
  )

  mask_triggers = [
      'authority',
      'turf',
      'protocol',
      'alibi',
      'witness',
      'honor',
      'vow',
  ]
  mask_triggers.extend(_extract_npc_keywords(f'{role_clean} {faction}', 4))

  angle_triggers = [
      'deal',
      'ledger',
      'cut',
      'trade',
      'rival',
      'swap',
      'price',
      'desire',
      'heart',
      'alliance',
      'pact',
  ]
  angle_triggers.extend(_extract_npc_keywords(goal_clean, 5))

  wound_triggers = [
      'secret',
      'proof',
      'evidence',
      'warrant',
      'betrayal',
      'exposure',
      'blood',
      'memory',
      'oath',
      'sleep',
      'slept',
      'afraid',
      'terrified',
      'hiding',
      'blank',
  ]
  wound_triggers.extend(
      _extract_npc_keywords(f'{secret_clean} {wound_clean}', 6)
  )

  return NpcSubMindMap({
      mask_name: story_state.SubMindState(
          name=mask_name,
          domain='stonewalling',
          worldview=(
              f'{role_clean} — {traits_clean}. Upholds {turf_label} facade and '
              'resists unleveraged pressure.'
          ),
          blind_spot=(
              'Overplays status or protocol and underestimates direct emotional'
              ' or evidentiary proof.'
          ),
          rank=mask_rank,
          xp=0,
          attention_triggers=list(dict.fromkeys(mask_triggers)),
          scars=[],
          obsession=(
              f'Uphold {faction.strip()} standing'
              if faction.strip()
              else 'Yield nothing without undeniable leverage'
          ),
          origin='npc',
          active=True,
          intrusive=mask_rank >= 4,
          role_archetype='facade',
          strong_against=['coercion', 'audacity', 'tactical_force', 'force'],
          vulnerable_to=[
              'forensics',
              'accounting',
              'cybernetics',
              'decorum',
              'hierarchy',
          ],
          favored_tactics=['deflect', 'invoke protocol', 'stonewall'],
      ),
      angle_name: story_state.SubMindState(
          name=angle_name,
          domain='leverage',
          worldview=(
              f'Inner Drive — Actively pursues: {goal_clean}. '
              'Seeks a reciprocal pact, temptation, alliance, or side-deal.'
          ),
          blind_spot=(
              'Reaches for what it desires even when the surrounding situation '
              'is on the verge of collapse.'
          ),
          rank=angle_rank,
          xp=0,
          attention_triggers=list(dict.fromkeys(angle_triggers)),
          scars=[],
          obsession=goal_clean,
          origin='npc',
          active=True,
          intrusive=angle_rank >= 4,
          role_archetype='drive',
          strong_against=['decorum', 'facade', 'hierarchy'],
          vulnerable_to=[
              'leverage',
              'passion',
              'seduction',
              'empathy',
              'social',
          ],
          favored_tactics=['bargain', 'entice', 'propose pact', 'maneuver'],
      ),
      wound_name: story_state.SubMindState(
          name=wound_name,
          domain='self_preservation',
          worldview=(
              f'Hidden Wound — Guards secret ({secret_clean}). '
              f'Vulnerability: {wound_clean}.'
          ),
          blind_spot=(
              'Panics, lashes out, or confesses damning truths when its core '
              'vulnerability is pierced.'
          ),
          rank=wound_rank,
          xp=0,
          attention_triggers=list(dict.fromkeys(wound_triggers)),
          scars=[],
          obsession=wound_clean if wound else secret_clean,
          origin='npc',
          active=True,
          intrusive=wound_rank >= 4,
          role_archetype='wound',
          strong_against=['decorum', 'protocol'],
          vulnerable_to=[
              'forensics',
              'coercion',
              'passion',
              'empathy',
              'intuition',
              'predation',
              'cybernetics',
          ],
          favored_tactics=['conceal', 'confess under pressure', 'lash out'],
      ),
  })


@dataclasses.dataclass
class NpcRecord:
  """Metadata and lifecycle state for a simulated NPC.

  Attributes:
    name: Unique character name.
    role_summary: Concise persona and visual description.
    goal: Active personal motivation or objective.
    traits: Speech register, temperament, and behavioral quirks.
    location_id: Current location_id where the NPC resides.
    tier: Current lifecycle tier (HOT, WARM, or COLD).
    last_active_tick: Simulation tick when the NPC last acted.
    last_action_wall_time: Wall-clock timestamp when the NPC last acted.
    directly_addressed: True if the latest scene event addressed this NPC.
    relationships: Map of entity name to relationship/disposition note.
    formative_memories: Initial memories seeded when the NPC was created.
    cold_state: Serialized EntityAgent state dictionary when hibernated.
    faction: Story faction id this NPC belongs to, if any.
    secret: What the NPC hides. Never shown to the player directly.
    wound: Why they hide it, and what leverage would make them talk.
    routine: Daily schedule slots (start_hour, end_hour, location, activity).
    known_clue_ids: Story clues this NPC can reveal when given leverage.
    sub_minds: Competing inner voices (`The Mask`, `The Angle`, `The Wound`).
    active_voice_options: Latest deliberated proposals from the NPC's council.
    last_winning_voice: Name of the Sub-Mind currently leading the NPC's mind.
    portrait_url: Relative URL for the NPC's generated portrait image.
  """

  name: str
  role_summary: str
  goal: str
  traits: str
  location_id: str
  tier: NpcTier = NpcTier.COLD
  last_active_tick: int = 0
  last_action_wall_time: float = 0.0
  directly_addressed: bool = False
  relationships: dict[str, str] = dataclasses.field(default_factory=dict)
  formative_memories: list[str] = dataclasses.field(default_factory=list)
  cold_state: dict[str, Any] | None = None
  faction: str = ''
  secret: str = ''
  wound: str = ''
  routine: list[story_state.RoutineSlot] = dataclasses.field(
      default_factory=list
  )
  known_clue_ids: list[str] = dataclasses.field(default_factory=list)
  sub_minds: dict[str, story_state.SubMindState] = dataclasses.field(
      default_factory=dict
  )
  active_voice_options: list[story_state.VoiceOption] = dataclasses.field(
      default_factory=list
  )
  last_winning_voice: str = ''
  portrait_url: str = ''

  def __post_init__(self) -> None:
    self.ensure_sub_minds()

  def ensure_sub_minds(self) -> None:
    """Ensures the NPC has a populated 3-voice Parliament of Sub-Minds."""
    if not self.sub_minds:
      self.sub_minds = synthesize_npc_sub_minds(
          name=self.name,
          role_summary=self.role_summary,
          goal=self.goal,
          traits=self.traits,
          faction=self.faction,
          secret=self.secret,
          wound=self.wound,
      )
    elif not isinstance(self.sub_minds, NpcSubMindMap):
      self.sub_minds = NpcSubMindMap(self.sub_minds)
    if not self.last_winning_voice or (
        self.last_winning_voice not in self.sub_minds
    ):
      self.last_winning_voice = self.dominant_sub_mind().name

  def dominant_sub_mind(self) -> story_state.SubMindState:
    """Returns the currently dominant Sub-Mind in this NPC's inner council."""
    if not self.sub_minds:
      self.sub_minds = synthesize_npc_sub_minds(
          name=self.name,
          role_summary=self.role_summary,
          goal=self.goal,
          traits=self.traits,
          faction=self.faction,
          secret=self.secret,
          wound=self.wound,
      )
    elif not isinstance(self.sub_minds, NpcSubMindMap):
      self.sub_minds = NpcSubMindMap(self.sub_minds)
    domain_priority = {
        'self_preservation': 3,
        'wound': 3,
        'leverage': 2,
        'drive': 2,
        'stonewalling': 1,
        'facade': 1,
    }
    candidates = [sm for sm in self.sub_minds.values() if sm.active] or list(
        self.sub_minds.values()
    )
    return max(
        candidates,
        key=lambda sm: (
            int(sm.rank),
            1 if sm.intrusive else 0,
            1 if sm.name == self.last_winning_voice else 0,
            max(
                domain_priority.get(sm.domain, 0),
                domain_priority.get(sm.role_archetype, 0),
            ),
        ),
    )

  def apply_voice_clash(
      self,
      *,
      mask_delta: int = 0,
      angle_delta: int = 0,
      wound_delta: int = 0,
      scar_target: str = '',
      scar_note: str = '',
      obsession_target: str = '',
      obsession_note: str = '',
  ) -> story_state.SubMindState:
    """Applies a social-combat rank/scar/obsession shift to NPC Sub-Minds."""
    self.ensure_sub_minds()
    deltas = {
        'stonewalling': mask_delta,
        'leverage': angle_delta,
        'self_preservation': wound_delta,
    }
    for domain_or_alias, delta in deltas.items():
      sm = self.sub_minds.get(domain_or_alias)
      if sm is not None and delta != 0:
        sm.rank = max(1, min(6, sm.rank + int(delta)))
        sm.intrusive = sm.rank >= 4
    if scar_target and scar_note:
      sm = self.sub_minds.get(scar_target)
      if sm is not None:
        cleaned_scar = scar_note.strip()
        if cleaned_scar and cleaned_scar not in sm.scars:
          sm.scars.append(cleaned_scar)
          if len(sm.scars) > 6:
            del sm.scars[:-6]
    if obsession_target and obsession_note:
      sm = self.sub_minds.get(obsession_target)
      if sm is not None and obsession_note.strip():
        sm.obsession = obsession_note.strip()
    dom = self.dominant_sub_mind()
    if not self.last_winning_voice:
      self.last_winning_voice = dom.name
    return dom

  def to_dict(self) -> dict[str, Any]:
    """Serializes the NPC record to a JSON-compatible dictionary."""
    self.ensure_sub_minds()
    result = dataclasses.asdict(self)
    result['tier'] = self.tier.value
    result['routine'] = [slot.to_dict() for slot in self.routine]
    result['sub_minds'] = {k: v.to_dict() for k, v in self.sub_minds.items()}
    result['active_voice_options'] = [
        opt.to_dict() for opt in self.active_voice_options
    ]
    return result

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> 'NpcRecord':
    """Restores an NpcRecord from a serialized dictionary."""
    cold_raw = payload.get('cold_state')
    cold_state = dict(cold_raw) if isinstance(cold_raw, Mapping) else None
    raw_sub_minds = payload.get('sub_minds', {})
    sub_minds: NpcSubMindMap = NpcSubMindMap()
    if isinstance(raw_sub_minds, Mapping):
      for k, v in raw_sub_minds.items():
        if isinstance(v, Mapping):
          sm = story_state.SubMindState.from_dict(v)
          sub_minds[str(k)] = sm
    raw_opts = payload.get('active_voice_options', [])
    active_opts = [
        story_state.VoiceOption.from_dict(o)
        for o in raw_opts
        if isinstance(o, Mapping)
    ]
    return cls(
        name=str(payload['name']),
        role_summary=str(payload.get('role_summary', '')),
        goal=str(payload.get('goal', '')),
        traits=str(payload.get('traits', '')),
        location_id=str(payload.get('location_id', '')),
        tier=NpcTier(str(payload.get('tier', NpcTier.COLD.value))),
        last_active_tick=int(payload.get('last_active_tick', 0)),
        last_action_wall_time=float(payload.get('last_action_wall_time', 0.0)),
        directly_addressed=bool(payload.get('directly_addressed', False)),
        relationships={
            str(k): str(v)
            for k, v in dict(payload.get('relationships', {})).items()
        },
        formative_memories=[
            str(x) for x in payload.get('formative_memories', [])
        ],
        cold_state=cold_state,
        faction=str(payload.get('faction', '')),
        secret=str(payload.get('secret', '')),
        wound=str(payload.get('wound', '')),
        routine=[
            story_state.RoutineSlot.from_dict(r)
            for r in payload.get('routine', [])
            if isinstance(r, Mapping)
        ],
        known_clue_ids=[str(x) for x in payload.get('known_clue_ids', [])],
        sub_minds=sub_minds,
        active_voice_options=active_opts,
        last_winning_voice=str(payload.get('last_winning_voice', '')),
        portrait_url=str(payload.get('portrait_url', '')),
    )

  def routine_location_for_hour(self, hour: int) -> str | None:
    """Returns the routine location name covering `hour`, if any."""
    for slot in self.routine:
      if slot.covers_hour(hour):
        return slot.location_name
    return None


class WorldLedger(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Thread-safe world graph, clock, and NPC registry backed by a TopicBus."""

  def __init__(
      self,
      *,
      premise: str,
      player_name: str = 'Traveler',
      starting_location_name: str = 'The Ashfall Crossroads',
      starting_location_description: str = '',
      starting_affordances: Sequence[str] = (),
      starting_exits: Mapping[str, str] | None = None,
      starting_inventory: Sequence[str] = (),
      start_clock_minutes: int = 8 * 60,
      bus: topics.TopicBus | None = None,
      max_recent_events: int = 12,
      story: story_state.StoryState | None = None,
  ):
    """Initializes the instance.

    Args:
      premise: Overall world premise and tone.
      player_name: Name of the player entity.
      starting_location_name: Name of the initial location node.
      starting_location_description: Description of the initial location.
      starting_affordances: Initial interactable objects/affordances.
      starting_exits: Initial mapping of exit label to adjacent location name.
      starting_inventory: Initial items carried by the player.
      start_clock_minutes: Starting time in minutes from Day 1 00:00.
      bus: Optional TopicBus instance. If None, a new TopicBus is created.
      max_recent_events: Max events retained per location node.
      story: Optional runtime story state (truths, clues, factions, storylets).
        If None, an empty StoryState is used and the story layer is inert.
    """
    super().__init__()
    self._lock = threading.RLock()
    self._premise = premise
    self._player_name = player_name
    self._tick = 0
    self._clock_minutes = start_clock_minutes
    self._max_recent_events = max_recent_events
    self._story = story or story_state.StoryState()
    if story is not None and story.doom_start_minutes == 0:
      self._story.doom_start_minutes = start_clock_minutes
    self._director: Any | None = None

    self._bus = bus or topics.TopicBus(clock=self.clock_float)
    self._bus.declare(
        'world.clock',
        retention=topics.Retention.LATEST,
        description='Current simulated time.',
    )
    self._bus.declare(
        'world.rumors',
        retention=topics.Retention.REPLAY,
        replay_size=20,
        description='Off-screen events and circulating rumors.',
    )
    self._bus.declare(
        'world.frontier',
        retention=topics.Retention.LATEST,
        description='Frontier expansion requests.',
    )
    self._bus.declare(
        'world.casting',
        retention=topics.Retention.LATEST,
        description='NPC spawn and rehydration requests.',
    )
    self._bus.declare(
        'world.image',
        retention=topics.Retention.LATEST,
        description='Generated location and NPC portrait image updates.',
    )

    self._locations: dict[str, LocationNode] = {}
    self._npcs: dict[str, NpcRecord] = {}
    self._world_facts: dict[str, str] = {}
    self._rumors: list[str] = []
    self._narrative_log: list[dict[str, Any]] = []
    self._next_narrative_id = 1
    self._player_inventory: list[str] = [
        str(x).strip() for x in starting_inventory if str(x).strip()
    ]
    self._player_status: str = 'Alert and unharmed'
    self._prose_mode: str = 'literary'
    self._pending_spawn_requests: list[dict[str, str]] = []

    # Telemetry counters for hard invariant verification.
    self._frontier_traversals = 0
    self._frontier_cache_hits = 0
    self._warm_events_count = 0
    self._hibernations_count = 0
    self._rehydrations_count = 0

    start_id = normalize_location_id(starting_location_name)
    self._player_location_id = start_id
    self._story.note_visited(start_id)
    exits_map: dict[str, str] = {}
    if starting_exits:
      for direction, target_name in starting_exits.items():
        target_id = self.ensure_stub_location(target_name)
        exits_map[direction] = target_id
      spoke_ids = list(dict.fromkeys(exits_map.values()))
      if len(spoke_ids) >= 2:
        for idx, spoke_id in enumerate(spoke_ids):
          next_spoke_id = spoke_ids[(idx + 1) % len(spoke_ids)]
          spoke_node = self._locations.get(spoke_id)
          if spoke_node is not None and next_spoke_id != spoke_id:
            spoke_node.exits['cross-town shortcut'] = next_spoke_id

    self._locations[start_id] = LocationNode(
        location_id=start_id,
        name=starting_location_name,
        description=starting_location_description or premise,
        affordances=[
            clean_affordance_text(a)
            for a in starting_affordances
            if clean_affordance_text(a)
        ],
        exits=exits_map,
        expanded=bool(starting_location_description),
    )
    self._bus.declare(
        f'scene.{start_id}',
        retention=topics.Retention.REPLAY,
        replay_size=10,
        description=f'Events in {starting_location_name}.',
    )
    self._bus.publish('world.clock', self.format_clock(), source='WorldLedger')

  @property
  def bus(self) -> topics.TopicBus:
    """Returns the shared TopicBus."""
    return self._bus

  @property
  def premise(self) -> str:
    """Returns the world premise."""
    with self._lock:
      return self._premise

  @property
  def story(self) -> story_state.StoryState:
    """Returns the mutable runtime story state (truths, clues, factions)."""
    return self._story

  @property
  def lock(self) -> threading.RLock:
    """Returns the ledger's re-entrant lock for compound story mutations."""
    return self._lock

  @property
  def director(self) -> Any | None:
    """Returns the attached or GM-mounted `StoryDirector`, if any."""
    if self._director is not None:
      return self._director
    if self._entity is not None:
      try:
        comp = self._entity.get_component(DEFAULT_STORY_DIRECTOR_COMPONENT_KEY)
        if comp is not None:
          self._director = comp
          return comp
      except KeyError:
        pass
    return None

  def set_director(self, director: Any | None) -> None:
    """Attaches or detaches a `StoryDirector`."""
    with self._lock:
      self._director = director

  @property
  def player_name(self) -> str:
    """Returns the human or synthetic player entity name."""
    return self._player_name

  @property
  def player_location_id(self) -> str:
    """Returns the location_id where the player is currently located."""
    with self._lock:
      return self._player_location_id

  @property
  def player_inventory(self) -> list[str]:
    """Returns a copy of the player's current inventory."""
    with self._lock:
      return list(self._player_inventory)

  @property
  def player_status(self) -> str:
    """Returns the player's current status summary."""
    with self._lock:
      return self._player_status

  @property
  def prose_mode(self) -> str:
    """Returns the active GM prose verbosity mode ('concise' or 'literary')."""
    with self._lock:
      return self._prose_mode

  def set_prose_mode(self, mode: str) -> str:
    """Sets the active GM prose verbosity mode ('concise' or 'literary')."""
    clean = str(mode or '').strip().lower()
    if clean not in ('concise', 'literary'):
      clean = 'concise'
    with self._lock:
      self._prose_mode = clean
      return self._prose_mode

  @property
  def tick(self) -> int:
    """Returns the current simulation step count."""
    with self._lock:
      return self._tick

  @property
  def clock_minutes(self) -> int:
    """Returns the elapsed simulated time in minutes."""
    with self._lock:
      return self._clock_minutes

  def clock_float(self) -> float:
    """Returns simulated clock minutes as a float for TopicBus timestamps."""
    with self._lock:
      return float(self._clock_minutes)

  def format_clock(self) -> str:
    """Formats the current simulated clock as 'Day D, HH:MM AM/PM'."""
    with self._lock:
      day = (self._clock_minutes // (24 * 60)) + 1
      mins_in_day = self._clock_minutes % (24 * 60)
      hour_24 = mins_in_day // 60
      minute = mins_in_day % 60
      period = 'AM' if hour_24 < 12 else 'PM'
      hour_12 = hour_24 % 12
      if hour_12 == 0:
        hour_12 = 12
      return f'Day {day}, {hour_12:02d}:{minute:02d} {period}'

  def advance_clock(self, minutes: int = 5, increment_tick: bool = True) -> str:
    """Advances the simulated clock and publishes on `world.clock`."""
    with self._lock:
      if increment_tick:
        self._tick += 1
      if minutes > 0:
        self._clock_minutes += minutes
      formatted = self.format_clock()
      self._bus.publish('world.clock', formatted, source='WorldLedger')
      director = self.director
      if director is not None:
        director.on_clock()
      return formatted

  def _match_existing_location_id(self, location_name: str) -> str | None:
    """Matches a location name or alias against existing locations in the ledger."""
    norm_id = normalize_location_id(location_name)
    if norm_id in self._locations:
      return norm_id
    raw_clean = location_name.strip().lower()
    if not raw_clean or raw_clean.startswith('inner reaches of'):
      return None
    ignore_words = {
        'the',
        'a',
        'an',
        'and',
        'of',
        'at',
        'in',
        'on',
        'to',
        'unheated',
        'heated',
        'outpost',
        'station',
        '7',
    }

    def _core_tokens(text: str) -> set[str]:
      return {
          w
          for w in re.findall(r'[a-z0-9]+', text.lower())
          if w not in ignore_words and len(w) >= 3
      }

    cand_tokens = _core_tokens(raw_clean)
    head_part = (
        re.split(r'\s*(?:&|\band\b)\s*', raw_clean, maxsplit=1)[0].strip()
        if ('&' in raw_clean or ' and ' in raw_clean)
        else ''
    )
    head_tokens = _core_tokens(head_part) if head_part else set()

    for existing_id, node in self._locations.items():
      if node.name.strip().lower() == raw_clean:
        return existing_id
      ex_tokens = _core_tokens(node.name) | _core_tokens(existing_id)
      if len(cand_tokens) >= 2 and cand_tokens <= ex_tokens:
        return existing_id
      if len(head_tokens) >= 2 and head_tokens <= ex_tokens:
        return existing_id
    return None

  def ensure_stub_location(self, location_name: str) -> str:
    """Creates an unexpanded stub location if it does not already exist."""
    with self._lock:
      matched_id = self._match_existing_location_id(location_name)
      if matched_id is not None:
        return matched_id
      loc_id = normalize_location_id(location_name)
      if loc_id not in self._locations:
        self._locations[loc_id] = LocationNode(
            location_id=loc_id,
            name=location_name.strip(),
            description=f'An unexplored area known as {location_name.strip()}.',
            expanded=False,
        )
        self._bus.declare(
            f'scene.{loc_id}',
            retention=topics.Retention.REPLAY,
            replay_size=10,
            description=f'Events in {location_name.strip()}.',
        )
      return loc_id

  def expand_location(
      self,
      location_id_or_name: str,
      *,
      name: str | None = None,
      description: str,
      affordances: Sequence[str] = (),
      exits: Mapping[str, str] | None = None,
      backlink_from_id: str | None = None,
      backlink_label: str = 'back',
      clue_ids: Sequence[str] = (),
      faction_lean: str = '',
  ) -> LocationNode:
    """Populates a location node with full details and stub exits."""
    with self._lock:
      loc_id = self._match_existing_location_id(
          location_id_or_name
      ) or normalize_location_id(location_id_or_name)
      node = self._locations.get(loc_id)
      display_name = name or (node.name if node else location_id_or_name)
      resolved_exits: dict[str, str] = dict(node.exits) if node else {}
      if exits:
        for direction, target_name in exits.items():
          target_id = self.ensure_stub_location(target_name)
          if target_id != loc_id:
            resolved_exits[str(direction)] = target_id
      if backlink_from_id and backlink_from_id in self._locations:
        if (
            backlink_from_id != loc_id
            and backlink_from_id not in resolved_exits.values()
        ):
          resolved_exits[backlink_label] = backlink_from_id

      cleaned_affordances = [
          clean_affordance_text(a)
          for a in affordances
          if clean_affordance_text(a)
      ]
      if node is None:
        node = LocationNode(
            location_id=loc_id,
            name=display_name,
            description=description,
            affordances=list(cleaned_affordances),
            exits=resolved_exits,
            expanded=True,
        )
        self._locations[loc_id] = node
        self._bus.declare(
            f'scene.{loc_id}',
            retention=topics.Retention.REPLAY,
            replay_size=10,
            description=f'Events in {display_name}.',
        )
      else:
        node.name = display_name
        node.description = description
        if cleaned_affordances:
          for item in cleaned_affordances:
            if item not in node.affordances:
              node.affordances.append(item)
        node.exits.update(resolved_exits)
        node.expanded = True
      for cid in clue_ids:
        if cid and cid not in node.clue_ids:
          node.clue_ids.append(cid)
          clue = self._story.clues.get(cid)
          if clue is not None:
            clue.placed_at = loc_id
      if faction_lean:
        node.faction_lean = faction_lean
      return copy.deepcopy(node)

  def get_location(self, location_id: str) -> LocationNode | None:
    """Returns a copy of a LocationNode, or None if not found."""
    with self._lock:
      node = self._locations.get(normalize_location_id(location_id))
      return copy.deepcopy(node) if node is not None else None

  def set_location_image_url(self, location_id: str, image_url: str) -> None:
    """Assigns a generated establishing-shot URL to `location_id`."""
    with self._lock:
      norm_id = normalize_location_id(location_id)
      node = self._locations.get(norm_id)
      if node is None:
        return
      if node.image_url == image_url:
        return
      node.image_url = image_url
      self._bus.publish(
          'world.image',
          f'location:{norm_id}:{image_url}',
          source='SceneImageGenerator',
      )

  def set_narrative_entry_image(
      self,
      entry_id: int,
      image_url: str,
      *,
      shot_caption: str = '',
      location_id: str = '',
  ) -> None:
    """Attaches a Photography Director panel URL and caption to a log beat."""
    with self._lock:
      changed = False
      target_loc_id = normalize_location_id(
          location_id or self._player_location_id
      )
      for entry in reversed(self._narrative_log):
        if entry.get('id') == entry_id:
          if (
              entry.get('image_url') != image_url
              or entry.get('shot_caption') != shot_caption
          ):
            entry['image_url'] = image_url
            if shot_caption:
              entry['shot_caption'] = shot_caption
            changed = True
          entry_loc = entry.get('location_id')
          if isinstance(entry_loc, str) and entry_loc:
            target_loc_id = normalize_location_id(entry_loc)
          break
      node = self._locations.get(target_loc_id)
      if node is not None and node.image_url != image_url:
        node.image_url = image_url
        changed = True
      cur_node = self._locations.get(self._player_location_id)
      if cur_node is not None and cur_node.image_url != image_url:
        cur_node.image_url = image_url
        changed = True
      if changed:
        self._bus.publish(
            'world.image',
            f'entry:{entry_id}:{image_url}',
            source='PhotographyDirector',
        )

  def set_npc_portrait_url(self, name: str, portrait_url: str) -> None:
    """Assigns a generated portrait URL to an NPC by `name`."""
    with self._lock:
      npc = self._npcs.get(name)
      if npc is None:
        return
      if npc.portrait_url == portrait_url:
        return
      npc.portrait_url = portrait_url
      self._bus.publish(
          'world.image',
          f'npc:{name}:{portrait_url}',
          source='SceneImageGenerator',
      )

  def get_player_location(self) -> LocationNode:
    """Returns a copy of the player's current LocationNode."""
    with self._lock:
      return copy.deepcopy(self._locations[self._player_location_id])

  def all_locations(self) -> dict[str, LocationNode]:
    """Returns a deep copy of all locations in the graph."""
    with self._lock:
      return copy.deepcopy(self._locations)

  def unexpanded_adjacent_locations(
      self, center_location_id: str | None = None
  ) -> list[LocationNode]:
    """Returns unexpanded stub locations reachable from `center_location_id`."""
    with self._lock:
      loc_id = center_location_id or self._player_location_id
      center = self._locations.get(loc_id)
      if center is None:
        return []
      stubs = []
      for target_id in center.exits.values():
        target = self._locations.get(target_id)
        if target is not None and not target.expanded:
          stubs.append(copy.deepcopy(target))
      return stubs

  def register_npc(self, record: NpcRecord) -> NpcRecord:
    """Registers or updates an NPC in the ledger."""
    with self._lock:
      self.ensure_stub_location(record.location_id)
      record.location_id = normalize_location_id(record.location_id)
      self._npcs[record.name] = copy.deepcopy(record)
      return copy.deepcopy(record)

  def get_npc(self, name: str) -> NpcRecord | None:
    """Returns a copy of an NpcRecord by name, or None if unknown."""
    with self._lock:
      rec = self._npcs.get(name)
      return copy.deepcopy(rec) if rec is not None else None

  def all_npcs(self) -> dict[str, NpcRecord]:
    """Returns a copy of all registered NPC records."""
    with self._lock:
      return copy.deepcopy(self._npcs)

  def npcs_at_location(self, location_id: str) -> list[NpcRecord]:
    """Returns all NPC records currently at `location_id`."""
    with self._lock:
      norm_id = normalize_location_id(location_id)
      return [
          copy.deepcopy(npc)
          for npc in self._npcs.values()
          if npc.location_id == norm_id
      ]

  def npcs_in_tier(self, tier: NpcTier) -> list[NpcRecord]:
    """Returns all NPC records currently in `tier`."""
    with self._lock:
      return [
          copy.deepcopy(npc) for npc in self._npcs.values() if npc.tier == tier
      ]

  def set_npc_tier(
      self,
      name: str,
      tier: NpcTier,
      *,
      cold_state: dict[str, Any] | None = None,
  ) -> NpcRecord | None:
    """Updates an NPC's tier and records hibernation/rehydration telemetry."""
    with self._lock:
      rec = self._npcs.get(name)
      if rec is None:
        return None
      previous_tier = rec.tier
      rec.tier = tier
      if tier == NpcTier.COLD and cold_state is not None:
        rec.cold_state = copy.deepcopy(cold_state)
        if previous_tier != NpcTier.COLD:
          self._hibernations_count += 1
      elif tier in (NpcTier.HOT, NpcTier.WARM):
        if previous_tier == NpcTier.COLD and rec.cold_state is not None:
          self._rehydrations_count += 1
      return copy.deepcopy(rec)

  def mark_npc_acted(self, name: str) -> None:
    """Updates `last_active_tick`, wall timestamp, and clears address flag."""
    with self._lock:
      rec = self._npcs.get(name)
      if rec is not None:
        rec.last_active_tick = self._tick
        rec.last_action_wall_time = time.time()
        rec.directly_addressed = False

  def set_npc_addressed(self, name: str, addressed: bool = True) -> None:
    """Marks whether an NPC was directly addressed in the current scene."""
    with self._lock:
      rec = self._npcs.get(name)
      if rec is not None:
        rec.directly_addressed = addressed

  def update_npc_sub_minds(
      self,
      name: str,
      *,
      sub_minds: Mapping[str, story_state.SubMindState] | None = None,
      active_voice_options: Sequence[story_state.VoiceOption] | None = None,
      last_winning_voice: str | None = None,
  ) -> NpcRecord | None:
    """Updates an NPC's Parliament of Sub-Minds and active voice options."""
    with self._lock:
      rec = self._npcs.get(name)
      if rec is None:
        return None
      rec.ensure_sub_minds()
      if sub_minds is not None:
        rec.sub_minds = NpcSubMindMap(
            {str(k): copy.deepcopy(v) for k, v in sub_minds.items()}
        )
      if active_voice_options is not None:
        rec.active_voice_options = [
            copy.deepcopy(opt) for opt in active_voice_options
        ]
      if last_winning_voice is not None and last_winning_voice.strip():
        rec.last_winning_voice = last_winning_voice.strip()
      else:
        rec.last_winning_voice = rec.dominant_sub_mind().name
      return copy.deepcopy(rec)

  def apply_npc_voice_clash(
      self,
      name: str,
      *,
      mask_delta: int = 0,
      angle_delta: int = 0,
      wound_delta: int = 0,
      scar_target: str = '',
      scar_note: str = '',
      obsession_target: str = '',
      obsession_note: str = '',
  ) -> NpcRecord | None:
    """Applies a voice-vs-voice rank/scar/obsession clash to an NPC's council."""
    with self._lock:
      rec = self._npcs.get(name)
      if rec is None:
        return None
      rec.apply_voice_clash(
          mask_delta=mask_delta,
          angle_delta=angle_delta,
          wound_delta=wound_delta,
          scar_target=scar_target,
          scar_note=scar_note,
          obsession_target=obsession_target,
          obsession_note=obsession_note,
      )
      return copy.deepcopy(rec)

  def move_entity(
      self, entity_name: str, target_location_id_or_name: str
  ) -> str:
    """Moves the player or an NPC to a target location and returns its ID."""
    with self._lock:
      if entity_name == self._player_name:
        current_loc_id = self._player_location_id
      elif entity_name in self._npcs:
        current_loc_id = self._npcs[entity_name].location_id
      else:
        current_loc_id = self._player_location_id
      current_node = self._locations.get(current_loc_id)
      resolved_id = None
      candidate = target_location_id_or_name.strip()

      # First check if candidate matches an exit direction or target ID/name.
      if current_node is not None:
        for direction, exit_target_id in current_node.exits.items():
          if candidate.lower() in (
              direction.lower(),
              exit_target_id.lower(),
          ):
            resolved_id = exit_target_id
            break
          target_node = self._locations.get(exit_target_id)
          if target_node and candidate.lower() == target_node.name.lower():
            resolved_id = exit_target_id
            break

      if resolved_id is None:
        norm_candidate = normalize_location_id(candidate)
        if norm_candidate in self._locations:
          resolved_id = norm_candidate
        else:
          matched_existing = self._match_existing_location_id(candidate)
          if matched_existing:
            resolved_id = matched_existing
          else:
            resolved_id = self.ensure_stub_location(candidate)
          if current_node is not None and resolved_id != current_loc_id:
            current_node.exits[candidate.lower()] = resolved_id
            self._locations[resolved_id].exits['back'] = current_loc_id

      if entity_name == self._player_name:
        if resolved_id != self._player_location_id:
          self._frontier_traversals += 1
          target_node = self._locations.get(resolved_id)
          if target_node is not None and target_node.expanded:
            self._frontier_cache_hits += 1
          self._player_location_id = resolved_id
          self._story.note_visited(resolved_id)
          if self._director is not None and self._director.active:
            self._director.refresh_voice_options(resolved_id)
          self._bus.publish(
              'world.frontier',
              resolved_id,
              source=self._player_name,
          )
      elif entity_name in self._npcs:
        self._npcs[entity_name].location_id = resolved_id

      return resolved_id

  def break_verbatim_clue_runs(self, text: str) -> str:
    """Breaks 6+ word verbatim clue runs while keeping all content words."""
    if not self._story.clues or not text:
      return text
    word_re = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
    func_swaps = {
        'of': 'from',
        'in': 'within',
        'on': 'across',
        'to': 'toward',
        'at': 'beside',
        'with': 'bearing',
        'and': 'alongside',
        'from': 'out of',
        'under': 'beneath',
        'inside': 'within',
        'into': 'down into',
        'for': 'covering',
        'by': 'near',
        'a': 'that',
        'an': 'that',
        'the': 'this',
    }
    adjectives = ('registered ', 'marked ', 'logged ', 'recorded ')
    result = text
    for clue in self._story.clues.values():
      clue_words = word_re.findall(clue.text.lower())
      if len(clue_words) < 6:
        continue
      for step_idx in range(4):
        matches = list(word_re.finditer(result.lower()))
        if len(matches) < 6:
          break
        res_words = [m.group(0) for m in matches]
        replaced = False
        for i in range(len(res_words) - 5):
          window = res_words[i : i + 6]
          matched_run = any(
              window == clue_words[j : j + 6]
              for j in range(len(clue_words) - 5)
          )
          if not matched_run:
            continue
          # Prefer swapping a stopword/preposition inside the 6-word window
          # so all content words remain intact and the prose reads naturally.
          swap_idx = -1
          for k in range(i + 1, i + 5):
            if res_words[k] in func_swaps:
              swap_idx = k
              break
          if swap_idx >= 0:
            m_tok = matches[swap_idx]
            orig_tok = result[m_tok.start() : m_tok.end()]
            rep_tok = func_swaps[res_words[swap_idx]]
            if orig_tok and orig_tok[0].isupper():
              rep_tok = rep_tok.capitalize()
            result = result[: m_tok.start()] + rep_tok + result[m_tok.end() :]
          else:
            ins_idx = i + 2
            for k in range(i + 1, i + 5):
              tok_str = result[matches[k].start() : matches[k].end()]
              if tok_str and tok_str[0].islower() and tok_str.isalpha():
                ins_idx = k
                break
            ins_pos = matches[ins_idx].start()
            adj = adjectives[step_idx % len(adjectives)]
            result = result[:ins_pos] + adj + result[ins_pos:]
          replaced = True
          break
        if not replaced:
          break
    return result

  def append_narrative_entry(
      self,
      *,
      kind: str,
      actor: str,
      text: str,
      location_id: str | None = None,
      headline: str = '',
      chips: Sequence[Mapping[str, str]] | None = None,
  ) -> dict[str, Any]:
    """Appends a deduplicated entry to the canonical ordered narrative log."""
    with self._lock:
      cleaned = text.strip()
      if kind in (
          'scene_resolution',
          'npc_resolution',
          'rumor',
          'arrival',
          'revelation',
      ):
        cleaned = self.break_verbatim_clue_runs(cleaned)
      norm_loc = normalize_location_id(location_id or self._player_location_id)
      node = self._locations.get(norm_loc)
      loc_name = node.name if node is not None else norm_loc

      for recent in self._narrative_log[-4:]:
        if recent.get('kind') == kind and recent.get('text') == cleaned:
          if headline.strip() and not recent.get('headline'):
            recent['headline'] = headline.strip()
          if chips and not recent.get('chips'):
            recent['chips'] = [dict(c) for c in chips]
          return dict(recent)

      entry: dict[str, Any] = {
          'id': self._next_narrative_id,
          'kind': kind,
          'actor': actor,
          'location_id': norm_loc,
          'location_name': loc_name,
          'clock': self.format_clock(),
          'tick': self._tick,
          'text': cleaned,
      }
      if headline.strip():
        entry['headline'] = headline.strip()
      if chips:
        entry['chips'] = [dict(c) for c in chips]
      self._next_narrative_id += 1
      self._narrative_log.append(entry)
      if len(self._narrative_log) > 2000:
        del self._narrative_log[:-2000]
      return dict(entry)

  def get_narrative_log(self) -> list[dict[str, Any]]:
    """Returns a copy of the canonical narrative log."""
    with self._lock:
      return [dict(e) for e in self._narrative_log]

  def record_intro(
      self, intro_text: str, *, location_id: str | None = None
  ) -> dict[str, Any]:
    """Records the opening scene setup card at the start of the adventure."""
    with self._lock:
      norm_id = normalize_location_id(location_id or self._player_location_id)
      node = self._locations.get(norm_id)
      if node is not None and intro_text.strip() not in node.recent_events:
        node.recent_events.insert(0, intro_text.strip())
      return self.append_narrative_entry(
          kind='intro',
          actor='Opening Scene',
          text=intro_text,
          location_id=norm_id,
      )

  def record_player_action(
      self, action_text: str, *, location_id: str | None = None
  ) -> dict[str, Any]:
    """Records a player action in the canonical narrative log and broadcasts."""
    with self._lock:
      if self._story.ended:
        return {}
      cleaned = action_text.strip()
      for recent in self._narrative_log[-4:]:
        if (
            recent.get('kind') == 'player_action'
            and recent.get('text') == cleaned
            and recent.get('tick') == self._tick
        ):
          return dict(recent)
      director = self.director
      if director is not None and director.active:
        director.on_player_action(action_text)
      entry = self.append_narrative_entry(
          kind='player_action',
          actor=self._player_name,
          text=action_text,
          location_id=location_id or self._player_location_id,
      )
      self._bus.publish(
          'world.clock', self.format_clock(), source=self._player_name
      )
      return entry

  def record_arrival(
      self, arrival_text: str, *, location_id: str | None = None
  ) -> dict[str, Any]:
    """Records a room arrival description in the canonical narrative log."""
    with self._lock:
      norm_loc = normalize_location_id(location_id or self._player_location_id)
      node = self._locations.get(norm_loc)
      loc_name = node.name if node is not None else norm_loc
      return self.append_narrative_entry(
          kind='arrival',
          actor='New Location',
          text=arrival_text,
          location_id=norm_loc,
          headline=f'Arrived at {loc_name}',
      )

  def record_scene_event(
      self,
      location_id: str,
      event_text: str,
      *,
      source: str = 'GameMaster',
      record_in_narrative_log: bool = True,
  ) -> None:
    """Appends an event to a location's history and publishes to `scene.<id>`."""
    with self._lock:
      norm_id = normalize_location_id(location_id)
      cleaned = event_text.strip()
      if not cleaned:
        return
      node = self._locations.get(norm_id)
      if node is not None:
        if not node.recent_events or node.recent_events[-1] != cleaned:
          node.recent_events.append(cleaned)
        if len(node.recent_events) > self._max_recent_events:
          del node.recent_events[: -self._max_recent_events]
      if record_in_narrative_log and norm_id == self._player_location_id:
        kind = (
            'scene_resolution'
            if source in (self._player_name, 'GameMaster')
            else 'npc_resolution'
        )
        self.append_narrative_entry(
            kind=kind,
            actor=source,
            text=cleaned,
            location_id=norm_id,
        )
      self._bus.publish(f'scene.{norm_id}', cleaned, source=source)

  def record_rumor(
      self,
      rumor_text: str,
      *,
      source: str = 'Chronicler',
      is_warm: bool = True,
      reliability: str = 'true',
  ) -> None:
    """Records an off-screen rumor/event and publishes to `world.rumors`."""
    with self._lock:
      rel_tag = (
          f' ({reliability} rumor via {source})'
          if reliability in ('distorted', 'false')
          else f' (via {source})'
      )
      stamped = f'[{self.format_clock()}{rel_tag}] {rumor_text}'
      self._rumors.append(stamped)
      if len(self._rumors) > 30:
        del self._rumors[:-30]
      if is_warm:
        self._warm_events_count += 1
      self.append_narrative_entry(
          kind='rumor',
          actor=source,
          text=stamped,
          location_id=self._player_location_id,
      )
      self._bus.publish('world.rumors', stamped, source=source)

  def get_rumors(self, limit: int = 20) -> list[str]:
    """Returns up to `limit` most recent rumors."""
    with self._lock:
      return list(self._rumors[-limit:]) if limit > 0 else list(self._rumors)

  def set_npc_relationship(
      self, npc_name: str, target_entity: str, note: str
  ) -> None:
    """Updates `npc_name`'s relationship note toward `target_entity`."""
    with self._lock:
      rec = self._npcs.get(npc_name)
      if rec is not None and note.strip():
        rec.relationships[target_entity] = note.strip()

  def add_location_affordance(self, location_id: str, prop: str) -> None:
    """Adds an affordance to `location_id`, replacing prior traces from the same NPC."""
    with self._lock:
      node = self._locations.get(normalize_location_id(location_id))
      cleaned = clean_affordance_text(prop)
      if node is None or not cleaned:
        return
      if cleaned.lower().startswith('physical spillover from '):
        node.affordances = [
            a
            for a in node.affordances
            if not a.lower().startswith('physical spillover from ')
        ]
      for prefix_marker in (
          'recent activity (',
          'reprisal prepared by ',
          'physical spillover from ',
      ):
        if prefix_marker in cleaned and '(' in cleaned:
          prefix = cleaned.split('(', 1)[0].strip()
          node.affordances = [
              a for a in node.affordances if not a.startswith(prefix)
          ]
      if cleaned not in node.affordances:
        node.affordances.append(cleaned)

  def relocate_clue(
      self,
      clue_id: str,
      target_location_or_npc: str,
      *,
      by_npc: str = '',
  ) -> str | None:
    """Relocates an undiscovered clue/MacGuffin to a new location or NPC.

    Args:
      clue_id: ID of the clue to relocate.
      target_location_or_npc: Destination location name/ID or NPC name.
      by_npc: Optional NPC name who moved or stole the clue.

    Returns:
      Human-readable description of the relocation, or None if not moved.
    """
    with self._lock:
      clue = self._story.clues.get(clue_id)
      if clue is None or clue.discovered:
        return None
      target_spec = target_location_or_npc.strip()
      if not target_spec:
        return None

      old_placed = clue.placed_at
      # Remove from previous location or NPC.
      if old_placed:
        old_loc = self._locations.get(normalize_location_id(old_placed))
        if old_loc is not None and clue_id in old_loc.clue_ids:
          old_loc.clue_ids.remove(clue_id)
        old_npc = self._npcs.get(old_placed)
        if old_npc is not None and clue_id in old_npc.known_clue_ids:
          old_npc.known_clue_ids.remove(clue_id)

      label = _CLUE_ITEM_LABELS.get(
          clue_id,
          clue.carrier_hint.split(',')[-1].strip() or clue_id,
      )
      actor_tag = f' by {by_npc}' if by_npc else ''

      if target_spec in self._npcs:
        dest_npc = self._npcs[target_spec]
        if clue.placed_at == dest_npc.name:
          return None
        if clue_id not in dest_npc.known_clue_ids:
          dest_npc.known_clue_ids.append(clue_id)
        clue.placed_at = dest_npc.name
        dest_display = dest_npc.name
      else:
        dest_loc_id = self.ensure_stub_location(target_spec)
        if old_placed and normalize_location_id(old_placed) == dest_loc_id:
          # Restore if same location.
          dest_node = self._locations.get(dest_loc_id)
          if dest_node is not None and clue_id not in dest_node.clue_ids:
            dest_node.clue_ids.append(clue_id)
          return None
        dest_node = self._locations[dest_loc_id]
        if clue_id not in dest_node.clue_ids:
          dest_node.clue_ids.append(clue_id)
        clue.placed_at = dest_loc_id
        dest_display = dest_node.name
        self.add_location_affordance(
            dest_loc_id,
            f'newly stashed {label} brought in{actor_tag}',
        )

      if old_placed:
        old_loc = self._locations.get(normalize_location_id(old_placed))
        if old_loc is not None:
          self.add_location_affordance(
              old_loc.location_id,
              f'fresh jimmy marks where {label} was lifted{actor_tag} '
              f'toward {dest_display}',
          )
          self.record_consequence(
              old_loc.location_id,
              f'Fresh jimmy marks show {label} was shifted{actor_tag} '
              f'toward {dest_display}.',
          )

      self._story.clues_relocated_count += 1
      return f'{label} moved{actor_tag} to {dest_display}'

  def set_world_fact(self, key: str, value: str) -> None:
    """Records or updates a global world fact."""
    with self._lock:
      if key.strip():
        self._world_facts[key.strip()] = value.strip()

  def add_clue_to_inventory(self, clue_id: str) -> None:
    """Adds the human-readable item/note for a discovered clue to inventory."""
    with self._lock:
      label = _CLUE_ITEM_LABELS.get(clue_id.strip())
      if label and label not in self._player_inventory:
        self._player_inventory.append(label)

  def record_consequence(self, location_id: str, consequence_text: str) -> None:
    """Records a legible cause->effect line in a location's recent_events."""
    with self._lock:
      node = self._locations.get(normalize_location_id(location_id))
      cleaned = consequence_text.strip()
      if node is not None and cleaned:
        if cleaned.startswith('[Spillover from adjacent '):
          node.recent_events = [
              ev
              for ev in node.recent_events
              if not ev.startswith('[Spillover from adjacent ')
          ]
        node.recent_events.append(cleaned)
        if len(node.recent_events) > self._max_recent_events:
          del node.recent_events[: -self._max_recent_events]

  def pop_spawn_requests(self) -> list[dict[str, str]]:
    """Drains and returns any pending NPC spawn requests."""
    with self._lock:
      items = list(self._pending_spawn_requests)
      self._pending_spawn_requests.clear()
      return items

  def apply_delta(
      self,
      delta: Mapping[str, Any],
      *,
      acting_entity: str,
      narrative_event: str,
      record_in_narrative_log: bool = True,
  ) -> dict[str, Any]:
    """Applies a structured resolution delta to the world ledger.

    Args:
      delta: Structured dictionary of world updates produced by the resolver.
      acting_entity: Name of the entity whose action was resolved.
      narrative_event: Resolved narrative description of what happened.
      record_in_narrative_log: Whether to append narrative_event to
        narrative_log.

    Returns:
      Summary dictionary of applied state changes.
    """
    with self._lock:
      if acting_entity == self._player_name:
        origin_loc_id = self._player_location_id
      elif acting_entity in self._npcs:
        origin_loc_id = self._npcs[acting_entity].location_id
      else:
        origin_loc_id = self._player_location_id
      minutes = int(delta.get('advance_minutes', 2))
      self.advance_clock(minutes=max(1, minutes), increment_tick=True)

      moved_to = None
      move_target = delta.get('move_to')
      if isinstance(move_target, str) and move_target.strip():
        moved_to = self.move_entity(acting_entity, move_target.strip())

      active_loc_id = moved_to or origin_loc_id
      node = self._locations.get(active_loc_id)
      if acting_entity == self._player_name and moved_to:
        for entry in reversed(self._narrative_log):
          if entry.get('kind') == 'player_action':
            entry['location_id'] = active_loc_id
            entry['location_name'] = node.name if node else active_loc_id
            break

      if node is not None:
        for added_prop in delta.get('add_affordances', []) or []:
          prop_str = clean_affordance_text(str(added_prop))
          if not prop_str:
            continue
          if prop_str.lower().startswith('physical spillover from '):
            node.affordances = [
                a
                for a in node.affordances
                if not a.lower().startswith('physical spillover from ')
            ]
          for prefix_marker in (
              'recent activity (',
              'reprisal prepared by ',
              'physical spillover from ',
          ):
            if prefix_marker in prop_str and '(' in prop_str:
              prefix = prop_str.split('(', 1)[0].strip()
              node.affordances = [
                  a for a in node.affordances if not a.startswith(prefix)
              ]
          if prop_str not in node.affordances:
            node.affordances.append(prop_str)
        for removed_prop in delta.get('remove_affordances', []) or []:
          prop_str = str(removed_prop).strip()
          cleaned_prop = clean_affordance_text(prop_str)
          if prop_str in node.affordances:
            node.affordances.remove(prop_str)
          elif cleaned_prop in node.affordances:
            node.affordances.remove(cleaned_prop)
        new_exits = delta.get('new_exits')
        if isinstance(new_exits, Mapping):
          for direction, target_name in new_exits.items():
            if str(direction).strip() and str(target_name).strip():
              target_id = self.ensure_stub_location(str(target_name).strip())
              node.exits[str(direction).strip()] = target_id

      implied_clues: list[str] = []
      for item in delta.get('add_inventory', []) or []:
        raw_item = str(item).strip()
        if not raw_item:
          continue
        if raw_item in self._story.clues:
          implied_clues.append(raw_item)
        item_str = _CLUE_ITEM_LABELS.get(
            raw_item, clean_affordance_text(raw_item) or raw_item
        )
        if item_str and item_str not in self._player_inventory:
          self._player_inventory.append(item_str)
      if implied_clues:
        delta = dict(delta)
        existing_clues = list(delta.get('clues_revealed') or [])
        for cid in implied_clues:
          if cid not in existing_clues:
            existing_clues.append(cid)
        delta['clues_revealed'] = existing_clues
      for item in delta.get('remove_inventory', []) or []:
        raw_item = str(item).strip()
        mapped_item = _CLUE_ITEM_LABELS.get(raw_item, raw_item)
        if raw_item in self._player_inventory:
          self._player_inventory.remove(raw_item)
        elif mapped_item in self._player_inventory:
          self._player_inventory.remove(mapped_item)

      player_status = delta.get('player_status')
      if isinstance(player_status, str) and player_status.strip():
        self._player_status = player_status.strip()

      world_facts = delta.get('world_facts')
      if isinstance(world_facts, Mapping):
        for k, v in world_facts.items():
          if str(k).strip():
            self._world_facts[str(k).strip()] = str(v).strip()

      relationships = delta.get('relationship_updates')
      if isinstance(relationships, Mapping):
        for npc_name, note in relationships.items():
          rec = self._npcs.get(str(npc_name))
          if rec is not None and str(note).strip():
            rec.relationships[acting_entity] = str(note).strip()

      if acting_entity == self._player_name:
        for npc in self.npcs_at_location(active_loc_id):
          self.set_npc_addressed(npc.name, False)

      addressed_list = delta.get('npcs_addressed', []) or []
      for npc_name in addressed_list:
        self.set_npc_addressed(str(npc_name), True)

      if acting_entity == self._player_name:
        generic_titles = {
            'captain',
            'archivist',
            'engineer',
            'innkeeper',
            'warden',
            'seneschal',
            'master',
            'ferryman',
            'inspector',
        }
        putative_lower = str(delta.get('_putative_action', '')).lower()
        explicitly_named = False
        if putative_lower:
          for npc in self.npcs_at_location(active_loc_id):
            distinct_tokens = [
                t.lower()
                for t in npc.name.split()
                if len(t) >= 3 and t.lower() not in generic_titles
            ]
            if npc.name.lower() in putative_lower or any(
                re.search(rf'\b{re.escape(tok)}\b', putative_lower)
                for tok in distinct_tokens
            ):
              self.set_npc_addressed(npc.name, True)
              explicitly_named = True
        if not explicitly_named and not addressed_list:
          for npc in self.npcs_at_location(active_loc_id):
            first_token = npc.name.split()[0].lower()
            if (
                npc.name.lower() in narrative_event.lower()
                or first_token in narrative_event.lower()
            ):
              self.set_npc_addressed(npc.name, True)

      spawn_req = delta.get('spawn_npc')
      if isinstance(spawn_req, Mapping) and spawn_req.get('name'):
        req_copy = {str(k): str(v) for k, v in spawn_req.items() if v}
        req_copy.setdefault('location_id', active_loc_id)
        self._pending_spawn_requests.append(req_copy)
        self._bus.publish(
            'world.casting',
            req_copy['name'],
            source=acting_entity,
        )

      if narrative_event.strip():
        self.record_scene_event(
            active_loc_id,
            narrative_event.strip(),
            source=acting_entity,
            record_in_narrative_log=record_in_narrative_log,
        )
        if moved_to and moved_to != origin_loc_id:
          self.record_scene_event(
              origin_loc_id,
              f'{acting_entity} departed toward'
              f' {node.name if node else moved_to}.',
              source=acting_entity,
          )

      story_summary: dict[str, Any] = {}
      director = self.director
      if director is not None and narrative_event.strip():
        story_summary = director.on_delta(
            delta,
            acting_entity=acting_entity,
            narrative_event=narrative_event,
            active_location_id=active_loc_id,
        )

      if record_in_narrative_log and narrative_event.strip():
        self._annotate_latest_resolution_entry(
            acting_entity=acting_entity,
            narrative_event=narrative_event,
            active_loc_id=active_loc_id,
            moved_to=moved_to,
            delta=delta,
            story_summary=story_summary,
        )

      return {
          'location_id': active_loc_id,
          'moved_to': moved_to,
          'clock': self.format_clock(),
          'story': story_summary,
      }

  def _annotate_latest_resolution_entry(
      self,
      *,
      acting_entity: str,
      narrative_event: str,
      active_loc_id: str,
      moved_to: str | None,
      delta: Mapping[str, Any],
      story_summary: Mapping[str, Any],
  ) -> None:
    """Attaches a scannable `headline` and `chips` list to the resolution entry."""
    expected_kind = (
        'scene_resolution'
        if acting_entity in (self._player_name, 'GameMaster')
        else 'npc_resolution'
    )
    target_entry: dict[str, Any] | None = None
    for entry in reversed(self._narrative_log):
      if (
          entry.get('kind') == expected_kind
          and entry.get('actor') == acting_entity
          and entry.get('tick') == self._tick
      ):
        target_entry = entry
        break
    if target_entry is None:
      return

    node = self._locations.get(active_loc_id)
    loc_title = node.name if node is not None else active_loc_id

    raw_headline = str(
        delta.get('outcome_headline') or delta.get('headline') or ''
    ).strip()
    raw_headline = re.sub(r'^["\'`*#\-—\s]+|["\'`*.\s]+$', '', raw_headline)
    if not raw_headline:
      if moved_to:
        raw_headline = f'Moved to {loc_title}'
      else:
        clean_ev = re.sub(r'\s+', ' ', narrative_event.strip())
        if expected_kind == 'npc_resolution' and '"' in clean_ev:
          pre_quote = clean_ev.split('"', 1)[0].strip().rstrip(',:—-')
          words = pre_quote.split()
          if 3 <= len(words) <= 12:
            raw_headline = pre_quote
        if not raw_headline:
          first_clause = re.split(r'(?<=[.!?])\s+|\s*[—;]\s*', clean_ev)[
              0
          ].strip()
          words = first_clause.split()
          if len(words) > 11:
            raw_headline = ' '.join(words[:10]).rstrip(',;:') + '…'
          else:
            raw_headline = first_clause.rstrip('.!?')
    else:
      words = raw_headline.split()
      if len(words) > 14:
        raw_headline = ' '.join(words[:12]).rstrip(',;:') + '…'

    chips: list[dict[str, str]] = []
    for cid in story_summary.get('clues_revealed') or []:
      cid_str = str(cid).strip()
      c_obj = self._story.clues.get(cid_str)
      c_label = _CLUE_ITEM_LABELS.get(cid_str)
      if not c_label and c_obj is not None and c_obj.carrier_hint:
        c_label = c_obj.carrier_hint.split(',')[-1].strip()
      c_label = c_label or cid_str
      chips.append({'kind': 'clue', 'label': f'🔍 Clue: {c_label[:42]}'})

    for tid in story_summary.get('truths_completed') or []:
      t_obj = self._story.truths.get(str(tid))
      t_stmt = t_obj.statement if t_obj is not None else str(tid)
      short_stmt = ' '.join(t_stmt.split()[:7]).rstrip('.,;:')
      chips.append({'kind': 'truth', 'label': f'💡 Mystery: {short_stmt}…'})

    for item in delta.get('add_inventory') or []:
      raw_item = str(item).strip()
      if not raw_item:
        continue
      label = _CLUE_ITEM_LABELS.get(
          raw_item, clean_affordance_text(raw_item) or raw_item
      )
      short_item = ' '.join(label.split()[:6]).rstrip('.,;:')
      chips.append({'kind': 'item', 'label': f'🎒 +{short_item}'})

    for item in delta.get('remove_inventory') or []:
      raw_item = str(item).strip()
      if not raw_item:
        continue
      label = _CLUE_ITEM_LABELS.get(raw_item, raw_item)
      short_item = ' '.join(label.split()[:6]).rstrip('.,;:')
      chips.append({'kind': 'item', 'label': f'🎒 -{short_item}'})

    new_exits = delta.get('new_exits')
    if isinstance(new_exits, Mapping):
      for d_name, t_name in list(new_exits.items())[:2]:
        if str(d_name).strip() and str(t_name).strip():
          chips.append({
              'kind': 'exit',
              'label': (
                  f'🚪 Exit: {str(d_name).strip()} → {str(t_name).strip()}'
              ),
          })

    for prop in (delta.get('add_affordances') or [])[:2]:
      p_clean = clean_affordance_text(str(prop))
      if not p_clean or p_clean.lower().startswith((
          'physical spillover from ',
          'evidence of ',
          'reprisal prepared by ',
          'fresh jimmy marks ',
          'recent activity (',
      )):
        continue
      short_prop = ' '.join(p_clean.split()[:6]).rstrip('.,;:')
      chips.append({'kind': 'prop', 'label': f'🔧 {short_prop}'})

    fac_shifts = story_summary.get('faction_shifts')
    if isinstance(fac_shifts, Mapping):
      for fid, disp in list(fac_shifts.items())[:2]:
        fac_obj = self._story.factions.get(str(fid))
        fac_name = fac_obj.name if fac_obj is not None else str(fid)
        disp_str = str(disp).strip()
        short_disp = ' '.join(disp_str.split()[:5]).rstrip('.,;:')
        chips.append(
            {'kind': 'faction', 'label': f'⚖️ {fac_name}: {short_disp}'}
        )

    for combat in (story_summary.get('npc_voice_combat') or [])[:2]:
      if not isinstance(combat, Mapping):
        continue
      npc_n = str(combat.get('npc') or 'NPC')
      dom_v = str(combat.get('dominant_voice') or 'Sub-Mind')
      dom_r = combat.get('dominant_rank', 1)
      out_lbl = str(combat.get('outcome') or '')
      if out_lbl == 'wound_spiked':
        chips.append({
            'kind': 'submind',
            'label': f'🧠 {npc_n}: {dom_v} exposed (Lv {dom_r})',
        })
      elif out_lbl == 'angle_spiked':
        chips.append({
            'kind': 'submind',
            'label': f'🧠 {npc_n}: {dom_v} hooked (Lv {dom_r})',
        })
      elif out_lbl == 'mask_vindicated':
        chips.append({
            'kind': 'submind',
            'label': f'🛡️ {npc_n}: Stonewalled ({dom_v} Lv {dom_r})',
        })

    if raw_headline:
      target_entry['headline'] = raw_headline
    if chips:
      target_entry['chips'] = chips

  def build_scene_context(
      self, location_id: str | None = None, *, is_arrival: bool = False
  ) -> str:
    """Formats a compact local context string for `FastSceneResolver`."""
    with self._lock:
      loc_id = normalize_location_id(location_id or self._player_location_id)
      node = self._locations.get(loc_id)
      if node is None:
        return f'Time: {self.format_clock()}\nUnknown location: {loc_id}'

      co_npcs = self.npcs_at_location(loc_id)
      exits_str = (
          ', '.join(
              f'{dir_name} -> {self._locations[tid].name} ({tid})'
              if tid in self._locations
              else f'{dir_name} -> {tid}'
              for dir_name, tid in node.exits.items()
          )
          or 'None visible'
      )
      affordances_str = ', '.join(node.affordances) or 'None noted'
      npc_lines = []
      for npc in co_npcs:
        rel = npc.relationships.get(self._player_name, 'Neutral')
        fac_tag = f' [{npc.faction}]' if npc.faction else ''
        dom = npc.dominant_sub_mind()
        intrusive_tag = ', INTRUSIVE' if dom.intrusive else ''
        npc_lines.append(
            f'- {npc.name}{fac_tag} ({npc.role_summary}) | Goal: {npc.goal} | '
            f'Traits: {npc.traits} | Attitude to {self._player_name}: {rel} | '
            f'Dominant Inner Voice: {dom.name} [Rank {dom.rank}'
            f'{intrusive_tag}] ({dom.obsession or dom.short_role()})'
        )
      if npc_lines:
        npcs_block = '\n'.join(npc_lines)
      else:
        npcs_block = (
            f'None present ({self._player_name} is completely ALONE in '
            f'{node.name}; NO assailant, rival, or NPC is in this room).'
        )
      recent_block = (
          '\n'.join(f'- {ev}' for ev in node.recent_events[-6:])
          if node.recent_events
          else 'No recent events in this location.'
      )
      rumors_block = (
          '\n'.join(f'- {r}' for r in self._rumors[-3:])
          if self._rumors
          else 'None.'
      )
      inv_str = ', '.join(self._player_inventory) or 'Empty'

      base = (
          f'World Premise: {self._premise}\n'
          f'Current Time: {self.format_clock()} (Tick {self._tick})\n'
          f'Active Location: {node.name} (id: {node.location_id})\n'
          f'Description: {node.description}\n'
          f'Interactable Props & Affordances: {affordances_str}\n'
          f'Exits: {exits_str}\n'
          f'Player ({self._player_name}) Status: {self._player_status} | '
          f'Inventory: {inv_str}\n'
          f'Co-located NPCs Physically Present in {node.name} (ONLY these '
          'characters are in this room; off-screen NPCs are NOT here):\n'
          f'{npcs_block}\n'
          f'Recent Local Events:\n{recent_block}\n'
          'Off-Screen World Rumors (distant events, NOT in this room):\n'
          f'{rumors_block}'
      )
      director = self.director
      if director is not None:
        notes = director.notes_for_scene(loc_id, is_arrival=is_arrival)
        if notes:
          return f'{base}\n\n{notes}'
      return base

  def telemetry_summary(self) -> dict[str, Any]:
    """Returns deterministic invariant and performance counters."""
    with self._lock:
      hit_rate = (
          (self._frontier_cache_hits / self._frontier_traversals)
          if self._frontier_traversals > 0
          else 1.0
      )
      tier_counts = {
          tier.value: len(self.npcs_in_tier(tier)) for tier in NpcTier
      }
      return {
          'tick': self._tick,
          'clock': self.format_clock(),
          'total_locations': len(self._locations),
          'expanded_locations': sum(
              1 for n in self._locations.values() if n.expanded
          ),
          'frontier_traversals': self._frontier_traversals,
          'frontier_cache_hits': self._frontier_cache_hits,
          'frontier_cache_hit_rate': hit_rate,
          'warm_events_count': self._warm_events_count,
          'hibernations_count': self._hibernations_count,
          'rehydrations_count': self._rehydrations_count,
          'tier_counts': tier_counts,
          'storylets_fired': self._story.storylets_fired(),
          'clues_discovered': self._story.clues_discovered(),
          'clues_relocated': self._story.clues_relocated_count,
          'truths_discovered': self._story.truths_discovered(),
          'active_turf_wars': len(self._story.active_turf_wars()),
          'tension': self._story.tension,
          'doom_stage': self._story.doom_stage,
          'act': self._story.act,
          'director_llm_calls': 0,
      }

  def to_dict(self) -> dict[str, Any]:
    """Serializes the entire WorldLedger for checkpointing and UI inspection."""
    with self._lock:
      return {
          'premise': self._premise,
          'player_name': self._player_name,
          'player_location_id': self._player_location_id,
          'player_inventory': list(self._player_inventory),
          'player_status': self._player_status,
          'prose_mode': self._prose_mode,
          'tick': self._tick,
          'clock_minutes': self._clock_minutes,
          'clock_formatted': self.format_clock(),
          'locations': {
              loc_id: node.to_dict() for loc_id, node in self._locations.items()
          },
          'npcs': {name: npc.to_dict() for name, npc in self._npcs.items()},
          'world_facts': dict(self._world_facts),
          'rumors': list(self._rumors),
          'narrative_log': [dict(entry) for entry in self._narrative_log],
          'telemetry': self.telemetry_summary(),
          'story': self._story.to_dict(),
          'bus_state': self._bus.get_state(),
      }

  def load_dict(self, state: Mapping[str, Any]) -> None:
    """Restores the WorldLedger from a dictionary produced by `to_dict()`."""
    with self._lock:
      self._premise = str(state.get('premise', self._premise))
      self._player_name = str(state.get('player_name', self._player_name))
      self._player_location_id = str(
          state.get('player_location_id', self._player_location_id)
      )
      self._player_inventory = [
          str(x) for x in state.get('player_inventory', [])
      ]
      self._player_status = str(state.get('player_status', self._player_status))
      if 'prose_mode' in state:
        self.set_prose_mode(str(state.get('prose_mode', self._prose_mode)))
      self._tick = int(state.get('tick', 0))
      self._clock_minutes = int(state.get('clock_minutes', 8 * 60))
      self._locations = {
          str(k): LocationNode.from_dict(v)
          for k, v in dict(state.get('locations', {})).items()
      }
      self._npcs = {
          str(k): NpcRecord.from_dict(v)
          for k, v in dict(state.get('npcs', {})).items()
      }
      self._world_facts = {
          str(k): str(v) for k, v in dict(state.get('world_facts', {})).items()
      }
      self._rumors = [str(x) for x in state.get('rumors', [])]
      raw_narrative = state.get('narrative_log', [])
      if isinstance(raw_narrative, list):
        self._narrative_log = [
            dict(x) for x in raw_narrative if isinstance(x, Mapping)
        ]
        max_id = max(
            (int(x.get('id', 0)) for x in self._narrative_log), default=0
        )
        self._next_narrative_id = max_id + 1
      telemetry = state.get('telemetry', {})
      if isinstance(telemetry, Mapping):
        self._frontier_traversals = int(telemetry.get('frontier_traversals', 0))
        self._frontier_cache_hits = int(telemetry.get('frontier_cache_hits', 0))
        self._warm_events_count = int(telemetry.get('warm_events_count', 0))
        self._hibernations_count = int(telemetry.get('hibernations_count', 0))
        self._rehydrations_count = int(telemetry.get('rehydrations_count', 0))
      story_raw = state.get('story')
      if isinstance(story_raw, Mapping):
        self._story = story_state.StoryState.from_dict(story_raw)
      bus_state = state.get('bus_state')
      if isinstance(bus_state, Mapping):
        self._bus.set_state(bus_state)

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    summary = (
        f'{self.format_clock()} @ {self.player_location_id} '
        f'(tick={self.tick}, status={self.player_status})'
    )
    self._logging_channel({
        'Key': DEFAULT_WORLD_LEDGER_COMPONENT_KEY,
        'Summary': summary,
        'Value': summary,
    })
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    return cast(entity_component.ComponentState, self.to_dict())

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    if state:
      self.load_dict(state)

  @override
  def get_dynamic_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'player_location_id': self._player_location_id,
          'player_status': self._player_status,
          'player_inventory': list(self._player_inventory),
          'clock_minutes': self._clock_minutes,
      }
