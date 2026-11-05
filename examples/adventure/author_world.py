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
"""Offline WorldBible generator (`author_world.py`).

Drafts a complete, schema-validated `WorldBible` JSON from a short world
premise and tone preset ('noir', 'heroic', or 'grim'), then runs
`story_state.validate_bible` and repairs any dangling IDs deterministically so
the output is immediately playable with `--world_bible=<output_path>`.
"""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from examples.adventure import adventure_specific_components
from examples.adventure import story_state
from concordia.language_model import language_model


def _define_flags() -> None:
  """Defines CLI flags when running author_world as a standalone script."""
  flags.DEFINE_string(
      'title', 'The Clockwork Quarantine', 'Title of the generated world.'
  )
  flags.DEFINE_string(
      'premise',
      'A walled alpine observatory-city is under quarantine after the brass '
      'pendulum that regulates its steam-locks began losing time. Three rival '
      'guilds blame one another while the mountain pass freezes shut.',
      'One-paragraph world premise.',
  )
  flags.DEFINE_string(
      'tone',
      'noir',
      'Tone/genre preset or custom style (e.g. noir, heroic, grim, caper, '
      'vampire_romance, dystopian_scifi).',
  )
  flags.DEFINE_string(
      'output',
      '/tmp/authored_world.json',
      'Output file path for the validated WorldBible JSON.',
  )


TONE_PRESETS: dict[str, dict[str, Any]] = {
    'noir': {
        'name': 'noir',
        'prose_register': (
            'Terse, sensory, rain-and-brass noir. Dialogue carries subtext; '
            'nobody explains themselves fully. Consequences are specific.'
        ),
        'npc_lie_probability': 0.5,
        'reveal_strictness': 'strict',
        'doom_hard': True,
        'clean_endings': False,
        'inner_voices': True,
        'setting_style': (
            'Rain-slicked industrial port and clockwork guild districts with '
            'ledger vaults, customs sheds, and lantern-lit alleys.'
        ),
        'room_palette': [
            'padlocked customs weigh-station smelling of wet iron and coal',
            'subterranean archive vault lined with chained folios',
            'pressurized steam-lock gallery with weeping brass gauges',
        ],
        'npc_archetypes': [
            'harbor watch officer nursing an unrecorded family debt',
            'contraband broker who trades favors instead of coin',
            'anxious guild archivist hiding a missing folio',
        ],
    },
    'heroic': {
        'name': 'heroic',
        'prose_register': (
            'Vivid, high-stakes adventure prose. Bold risks earn clear turns, '
            'allies remember loyalty, and villains overplay their hand.'
        ),
        'npc_lie_probability': 0.25,
        'reveal_strictness': 'lenient',
        'doom_hard': False,
        'clean_endings': True,
        'inner_voices': True,
        'setting_style': (
            'High-stakes frontier citadel with bannered halls, signal towers, '
            'and sky-bridges.'
        ),
        'room_palette': [
            'wind-swept signal bastion overlooking the mountain pass',
            'guild armory hung with notched banners and dispatch maps',
            'great clock-tower gallery where counterweights swing',
        ],
        'npc_archetypes': [
            'veteran gate captain testing strangers for courage',
            'daring courier with a half-burned cipher',
            'disgraced artificer seeking a chance to redeem their name',
        ],
    },
    'grim': {
        'name': 'grim',
        'prose_register': (
            'Sparse, material, survival-focused prose. Scarcity, cold, and '
            'fatigue weigh on every bargain; every rescue leaves a debt.'
        ),
        'npc_lie_probability': 0.65,
        'reveal_strictness': 'strict',
        'doom_hard': True,
        'clean_endings': False,
        'inner_voices': True,
        'setting_style': (
            'Snow-choked quarantine ward and rationed boiler-tunnels where '
            'fuel and passes are scarce.'
        ),
        'room_palette': [
            'soot-blackened coal bunker with ration tallies on the wall',
            'quarantine triage post smelling of carbolic soap and wet wool',
            'frozen sluice-gate catwalk coated in rime ice',
        ],
        'npc_archetypes': [
            'exhausted ration clerk who measures every word in coal chits',
            'desperate tunnel runner willing to sell anyone for a pass',
            'hard-eyed watch corporal enforcing curfew at iron-point',
        ],
    },
    'caper': {
        'name': 'caper',
        'prose_register': (
            'Punchy, fast-talking London underworld caper prose with dry dark '
            'humour, Cockney/Soho idioms, and sharp physical consequences. '
            'Everyone has an angle, a vig, or a double-cross in play.'
        ),
        'npc_lie_probability': 0.55,
        'reveal_strictness': 'strict',
        'doom_hard': True,
        'clean_endings': False,
        'inner_voices': True,
        'setting_style': (
            'South London underworld caper across Bermondsey railway '
            'arches, Soho cellars, Hatton Garden vaults, unlicensed boxing '
            'gyms, chop shops, and greyhound tracks.'
        ),
        'room_palette': [
            'unlicensed bare-knuckle boxing gym under a dripping brick viaduct',
            'backroom bookmaker settlement cage with scratched armoured glass',
            'Hatton Garden basement diamond workshop with ultrasonic cleaners',
            'Rotherhithe panel-beater chop shop smelling of cellulose spray',
            (
                'lock-in Victorian pub back room with a baize table and fruit'
                ' machine'
            ),
            'Spitalfields meat chiller lockup with false-floor crates',
        ],
        'npc_archetypes': [
            'fast-talking ticket tout up to his neck in dog-track vigorish',
            'retired cruiserweight turned polite East End debt collector',
            (
                'bespectacled diamond valuer with a jeweller loupe and a pocket'
                ' ledger'
            ),
            'bent Flying Squad detective sergeant nursing a pension shortfall',
            'veterinary chemist who doctors greyhounds before the traps open',
        ],
    },
    'vampire_romance': {
        'name': 'vampire_romance',
        'prose_register': (
            'Quiet, grounded nocturnal thriller and intimate love drama. '
            'Plain spoken dialogue, fluorescent hospital hallways, cold snow '
            'outside, thermos flasks, and forty years of shared survival—no '
            'gothic court jargon.'
        ),
        'npc_lie_probability': 0.45,
        'reveal_strictness': 'strict',
        'doom_hard': True,
        'clean_endings': False,
        'inner_voices': True,
        'setting_style': (
            'Basement blood bank and hematology lab of St. Jude’s City '
            'Hospital at 03:00 AM during a winter snowstorm: stainless-steel '
            'blood cooler, breakroom with a ticking wall clock, ambulance '
            'loading dock, and night-shift security desk.'
        ),
        'room_palette': [
            (
                'fluorescent-lit basement blood bank with a humming '
                'stainless-steel cooler and red sign-out binder'
            ),
            (
                'staff breakroom smelling of stale drip coffee, wet winter '
                'coats, and antiseptic soap'
            ),
            (
                'snow-blown ambulance loading bay with heavy plastic strip '
                'curtains and a cold sedan'
            ),
            (
                'quiet hospital records and security corridor with frosted '
                'wire-glass doors'
            ),
        ],
        'npc_archetypes': [
            (
                'your partner of forty years in a lab coat with a packed '
                'canvas duffel bag under the bench'
            ),
            (
                'bandaged young ER resident sitting in the breakroom whose '
                'eyes hurt under the fluorescent lights'
            ),
            (
                'retired police detective at the security desk holding a '
                'folder of forty-year-old hospital ID photos'
            ),
        ],
    },
    'dystopian_scifi': {
        'name': 'dystopian_scifi',
        'prose_register': (
            'Grounded, physical, paranoid frontier sci-fi prose. Plain spoken'
            ' dialogue, cold iron, diesel heaters, coffee mugs, and quiet'
            ' dread—no technobabble.'
        ),
        'npc_lie_probability': 0.55,
        'reveal_strictness': 'strict',
        'doom_hard': True,
        'clean_endings': False,
        'inner_voices': True,
        'setting_style': (
            'Isolated four-person company weather and relay station on frozen '
            'salt flats: mess hall with a clanking heater, unheated cargo '
            'hangar with a dry-ice crate, medical bay, and radio shack.'
        ),
        'room_palette': [
            (
                'cramped station mess hall smelling of burnt coffee, wet wool,'
                ' and diesel heat'
            ),
            (
                'unheated cargo hangar where frost coats the supply sled and'
                ' packing crates'
            ),
            (
                'compact medical bay with a stainless examination table and'
                ' locked chemical cabinet'
            ),
            (
                'wind-rattled radio and generator shack with severed antenna'
                ' coaxial cables'
            ),
        ],
        'npc_archetypes': [
            (
                'exhausted station mechanic drinking black coffee with a'
                ' three-day gap in his memory'
            ),
            (
                'composed company medic hiding a sedative log and a clearance'
                ' directive'
            ),
            (
                'nervous radio operator with bolt cutters and a side deal on'
                ' core samples'
            ),
        ],
    },
}


def _resolve_genre_mode(tone_name: str, title: str, premise: str) -> str:
  """Resolves the genre/tone mode from explicit tone_name or premise keywords."""
  low_tone = tone_name.strip().lower()
  if low_tone in TONE_PRESETS and low_tone not in ('noir',):
    return low_tone
  combined = f'{low_tone} {title.lower()} {premise.lower()}'
  if any(
      kw in combined
      for kw in (
          'vampire',
          'immortal',
          'coven',
          'sire',
          'crimson court',
          'blood-pact',
          'gothic romance',
          'vampire romance',
          'love drama',
      )
  ):
    return 'vampire_romance'
  if any(
      kw in combined
      for kw in (
          'dystop',
          'scifi',
          'sci-fi',
          'cyber',
          'orbital',
          'neural',
          'wetware',
          'megacorp',
          'synth',
          'airlock',
          'starship',
      )
  ):
    return 'dystopian_scifi'
  if any(
      kw in combined
      for kw in (
          'gangster',
          'bermondsey',
          'soho',
          'cockney',
          'hatton garden',
          'caper',
          'underworld firm',
      )
  ):
    return 'caper'
  return low_tone if low_tone in TONE_PRESETS else 'noir'


def _build_vampire_romance_template_dict(
    *,
    title: str,
    premise: str,
    tone: dict[str, Any],
) -> dict[str, Any]:
  """Builds the grounded vampire love drama WorldBible dictionary."""
  voices = {
      'The Hunger': (
          'Thirst & Pulse-Sense — Hears a racing heartbeat through a closed '
          'door, smells fresh blood under gauze, and knows when your own '
          'thirst or someone else’s is about to slip.'
      ),
      'Cover Story': (
          'Forty Years of Clean Aliases — Speaks in calm hospital shift-talk, '
          'keeps the ID badges and cooler logs straight, and knows survival '
          'depends on never making a scene.'
      ),
      'Unspoken Vow': (
          'Forty Years with Claire — Knows every hitch in Claire’s voice '
          'after four decades together, and hears the reckless devotion in '
          'someone who thinks forever is a gift.'
      ),
      'Paper Trail': (
          'Old IDs & Back-Room Deals — Tracks payroll archives, security '
          'camera blind spots, and what a tired man with a cold-case folder '
          'actually wants.'
      ),
      'Night Chart': (
          'Blood Cooler & Triage Signs — Reads fridge sign-out sheets, needle '
          'gauges, pupil dilation, and the early fever that means vampire '
          'blood is in a human bloodstream.'
      ),
      'Mortal Echo': (
          'Remembered Daylight — Remembers what it felt like to grow old, to '
          'breathe cold air that warmed in your lungs, and refuses to treat '
          'human lives as disposable.'
      ),
  }
  player_sub_minds = {
      'The Hunger': {
          'description': voices['The Hunger'],
          'rank': 3,
          'domain': 'predation',
          'blind_spot': (
              'Risks losing control in a brightly lit hospital corridor when '
              'fresh blood is in the air.'
          ),
          'role_archetype': 'instinct',
          'attention_triggers': [
              'pulse',
              'blood',
              'vein',
              'throat',
              'heartbeat',
              'scent',
              'bandage',
          ],
          'strong_against': ['decorum', 'hierarchy', 'stonewalling'],
          'vulnerable_to': ['passion', 'empathy', 'forensics'],
          'favored_tactics': [
              (
                  'step close and listen to the jump in their pulse when you'
                  ' ask about the ambulance bay'
              ),
              'let your calm slip just enough that they remember what you are',
          ],
          'voice_tone': (
              'Low, quiet, intensely physical—counts heartbeats per minute.'
          ),
      },
      'Cover Story': {
          'description': voices['Cover Story'],
          'rank': 3,
          'domain': 'decorum',
          'blind_spot': (
              'Tries to maintain a routine cover story even when the secret '
              'is already out on the table.'
          ),
          'role_archetype': 'authority',
          'attention_triggers': [
              'badge',
              'shift',
              'cooler',
              'protocol',
              'rule',
              'schedule',
          ],
          'strong_against': ['audacity', 'coercion', 'stonewalling'],
          'vulnerable_to': ['predation', 'passion', 'leverage'],
          'favored_tactics': [
              (
                  'use calm hospital procedure to back them into a corner'
                  ' without raising your voice'
              ),
              'remind them that the 06:00 AM day shift arrives in three hours',
          ],
          'voice_tone': (
              'Measured, composed, sounds like a veteran night-shift'
              ' supervisor.'
          ),
      },
      'Unspoken Vow': {
          'description': voices['Unspoken Vow'],
          'rank': 3,
          'domain': 'passion',
          'blind_spot': (
              'Protects Claire or sympathizes with a desperate promise even '
              'when it puts everyone in danger.'
          ),
          'role_archetype': 'empathy',
          'attention_triggers': [
              'claire',
              'watch',
              'promise',
              'train',
              'hand',
              'love',
          ],
          'strong_against': ['self_preservation', 'predation', 'stonewalling'],
          'vulnerable_to': ['decorum', 'leverage', 'accounting'],
          'favored_tactics': [
              (
                  'lower your voice so only they can hear and name the promise'
                  ' they are trying to keep'
              ),
              (
                  'reach for Claire’s hand by the centrifuge and ask her what'
                  ' happened in the ambulance bay'
              ),
          ],
          'voice_tone': 'Intimate, tired, fiercely loyal.',
      },
      'Paper Trail': {
          'description': voices['Paper Trail'],
          'rank': 2,
          'domain': 'leverage',
          'blind_spot': (
              'Treats grief or love like a bargain that can be settled with '
              'cash or a train ticket.'
          ),
          'role_archetype': 'transactional',
          'attention_triggers': [
              'folder',
              'photo',
              'file',
              'camera',
              'record',
              'ticket',
          ],
          'strong_against': ['decorum', 'self_preservation', 'hierarchy'],
          'vulnerable_to': ['predation', 'passion'],
          'favored_tactics': [
              (
                  'ask what it will take to keep that manila folder off the'
                  ' morning supervisor’s desk'
              ),
              (
                  'check whether the security desk logged an official report or'
                  ' kept the file private'
              ),
          ],
          'voice_tone': (
              'Pragmatic, streetwise, always looking for the quiet exit.'
          ),
      },
      'Night Chart': {
          'description': voices['Night Chart'],
          'rank': 2,
          'domain': 'forensics',
          'blind_spot': (
              'Focuses on lab numbers and clinical symptoms while emotions in '
              'the room boil over.'
          ),
          'role_archetype': 'analytical',
          'attention_triggers': [
              'binder',
              'o-negative',
              'pupil',
              'fever',
              'gauze',
              'scar',
          ],
          'strong_against': ['stonewalling', 'leverage'],
          'vulnerable_to': ['predation', 'passion'],
          'favored_tactics': [
              (
                  'check the O-negative sign-out binder against the empty hooks'
                  ' in the cooler'
              ),
              (
                  'shine a penlight into their pupils and check the bandage on'
                  ' their neck'
              ),
          ],
          'voice_tone': (
              'Clinical, observant, notices every missing bag and dilated'
              ' pupil.'
          ),
      },
      'Mortal Echo': {
          'description': voices['Mortal Echo'],
          'rank': 2,
          'domain': 'empathy',
          'blind_spot': (
              'Hesitates to use threat or force against a mortal who is '
              'grieving or terrified.'
          ),
          'role_archetype': 'empathy',
          'attention_triggers': [
              'mortal',
              'breath',
              'dawn',
              'wife',
              'grief',
              'warmth',
          ],
          'strong_against': ['self_preservation', 'passion'],
          'vulnerable_to': ['predation', 'coercion'],
          'favored_tactics': [
              (
                  'speak to them plainly about what forty years without'
                  ' sunlight actually costs'
              ),
              'offer a humane way out before anyone else gets hurt tonight',
          ],
          'voice_tone': 'Warm, honest, carries forty years of quiet regret.',
      },
  }
  character_presets = [
      {
          'id': 'crimson_heir',
          'name': 'The Careful Partner',
          'tagline': 'Cover Story 4 · Night Chart 3 — Calm & Methodical',
          'description': (
              'You kept the two of you alive since 1986 by checking the blood '
              'cooler logs twice, paying rent in cash, and moving cities '
              'every five years before anyone noticed.'
          ),
          'ranks': {
              'Cover Story': 4,
              'Night Chart': 3,
              'Unspoken Vow': 2,
              'Paper Trail': 2,
              'The Hunger': 2,
              'Mortal Echo': 1,
          },
          'inventory': [
              'stainless blood-cooler key',
              'two 06:00 AM sleeper tickets to Montreal',
              'thirty-year-old photo booth strip of you and Claire',
          ],
      },
      {
          'id': 'blood_bound_envoy',
          'name': 'The Devoted Companion',
          'tagline': 'Unspoken Vow 4 · Mortal Echo 3 — Loyal & Heart-Torn',
          'description': (
              'Forty years of graveyard shifts haven’t dulled what Claire '
              'means to you—or the ache of remembering the human lives you '
              'both left behind.'
          ),
          'ranks': {
              'Unspoken Vow': 4,
              'Mortal Echo': 3,
              'Paper Trail': 3,
              'Cover Story': 2,
              'Night Chart': 1,
              'The Hunger': 1,
          },
          'inventory': [
              'thirty-year-old photo booth strip of you and Claire',
              'two 06:00 AM sleeper tickets to Montreal',
              'thermos of warmed O-negative',
          ],
      },
      {
          'id': 'night_inquisitor',
          'name': 'The Cornered Immortal',
          'tagline': 'The Hunger 4 · Paper Trail 3 — Sharp & Territorial',
          'description': (
              'You can hear a heartbeat spike from down the corridor, and you '
              'will not let a retired detective or a panicked resident stop '
              'you and Claire from making that morning train.'
          ),
          'ranks': {
              'The Hunger': 4,
              'Paper Trail': 3,
              'Night Chart': 3,
              'Cover Story': 2,
              'Unspoken Vow': 1,
              'Mortal Echo': 1,
          },
          'inventory': [
              'stainless blood-cooler key',
              'forged Montreal driver’s licenses',
              'two 06:00 AM sleeper tickets to Montreal',
          ],
      },
  ]
  if (
      not title.strip()
      or 'House Valerius' in title
      or 'Solstice Masque' in title
      or title.strip().lower() == 'vampire_romance'
  ):
    default_title = "The Night Shift at St. Jude's"
  else:
    default_title = title.strip()
  if (
      len(premise.strip()) > 160
      and 'Obsidian Court' not in premise
      and 'Solstice Masque' not in premise
      and 'House Valerius' not in premise
  ):
    rich_premise = premise.strip()
  else:
    rich_premise = (
        'For forty years, you and Claire have survived by keeping one rule: '
        'work the graveyard shift, live off expiring O-negative bags from '
        'hospital blood banks, and change cities every five years before '
        'anyone notices neither of you has aged. Tonight is supposed to be '
        'your last shift at St. Jude’s Hospital before catching the 06:00 '
        'AM train north. Instead, three bags of O-negative are missing from '
        'the locked cooler, a twenty-six-year-old ER resident named Dr. '
        'David Park is sitting in the breakroom with a gauze bandage taped '
        'over his neck and a feverish devotion in his eyes, and retired '
        'detective Arthur Finch is at the security desk holding a manila '
        'folder of Claire’s hospital ID photos from 1986, 1998, and tonight.'
    )
  return {
      'title': default_title,
      'premise': rich_premise,
      'opening': (
          'You step into The Basement Blood Bank & Lab at 03:00 AM, shaking '
          'wet snow off your coat. In your pocket sit the stainless blood-'
          'cooler key, two one-way sleeper tickets on the 06:00 AM train to '
          'Montreal, and a thirty-year-old photo booth strip of you and '
          'Claire. You are the Night-Shift Blood Bank Tech—and Claire’s '
          'partner of forty years. Before sunrise and the 06:00 AM shift '
          'change (3 hours from now), you must find out what happened to '
          'young Dr. David Park in the ambulance bay and what Claire had to '
          'do with it, deal with Detective Arthur Finch before his photo '
          'file reaches the morning supervisor, and decide who gets on that '
          'morning train.'
      ),
      'tone': tone,
      'player_voices': voices,
      'player_sub_minds': player_sub_minds,
      'character_presets': character_presets,
      'emergent_voice_pool': [{
          'name': 'The Obsessive Heart',
          'description': (
              'Emerged after forty years of hiding broke open — refuses to '
              'board the 06:00 AM train if it means leaving someone behind to '
              'face the sunrise alone.'
          ),
          'domain': 'passion',
          'role_archetype': 'empathy',
          'rank': 3,
          'keywords': [
              'kiss',
              'embrace',
              'vow',
              'blood',
              'forgive',
              'claire',
              'david',
          ],
          'min_Count': 1,
          'attention_triggers': [
              'pulse',
              'vow',
              'gaze',
              'devotion',
              'train',
              'bandage',
          ],
          'strong_against': ['self_preservation', 'predation'],
          'vulnerable_to': ['decorum', 'leverage'],
          'favored_tactics': [
              (
                  'put the train tickets on the bench and swear nobody gets'
                  ' left behind'
              ),
              (
                  'tell them plainly what you are willing to give up before'
                  ' sunrise'
              ),
          ],
      }],
      'starting_inventory': [
          'stainless blood-cooler key',
          'two 06:00 AM sleeper tickets to Montreal',
          'thirty-year-old photo booth strip of you and Claire',
      ],
      'starting_location': {
          'name': 'The Basement Blood Bank & Lab',
          'description': (
              'Fluorescent tubes hum overhead in the windowless basement '
              'hematology lab. Condensation beads on the glass door of the '
              'stainless-steel blood cooler, a red transfusion sign-out '
              'binder lies open beside the centrifuge, and Claire’s zipped '
              'canvas duffel bag is tucked underneath the steel lab bench.'
          ),
          'affordances': [
              (
                  'stainless-steel blood cooler with three empty hooks on the'
                  ' O-negative rack'
              ),
              'red transfusion sign-out binder beside the centrifuge',
              'Claire’s zipped canvas duffel bag tucked under the lab bench',
          ],
          'exits': {
              'north': 'The Night-Shift Breakroom',
              'east': 'The Records & Security Corridor',
              'west': 'The Ambulance Loading Bay',
          },
          'clue_ids': ['c_chalice_ash', 'c_ribbon_vow'],
      },
      'factions': [
          {
              'id': 'house_valerius',
              'name': 'The Forty-Year Pact (You & Claire)',
              'aim': (
                  'Make the 06:00 AM train to Montreal with your cover intact '
                  'and decide what becomes of young Dr. David Park.'
              ),
              'standing_with_player': 0,
              'standing': {'thorn_covenant': -3, 'mortal_embassy': 1},
              'assets': [
                  'The Basement Blood Bank & Lab',
                  '06:00 AM Montreal sleeper tickets',
              ],
              'register': (
                  'Quiet, intimate shorthand built over forty years of night '
                  'shifts and shared secrets.'
              ),
          },
          {
              'id': 'thorn_covenant',
              'name': 'Finch’s Cold-Case File',
              'aim': (
                  'Use forty years of matching hospital ID photos to force a '
                  'confession—and a bargain—before the 06:00 AM shift change.'
              ),
              'standing_with_player': 0,
              'standing': {'house_valerius': -3, 'mortal_embassy': -2},
              'assets': [
                  'The Records & Security Corridor',
                  '1986–2026 payroll photo folder',
              ],
              'register': (
                  'Tired, plain-spoken detective persistence driven by personal'
                  ' grief rather than official duty.'
              ),
          },
          {
              'id': 'mortal_embassy',
              'name': 'Dr. David Park (The Night Resident)',
              'aim': (
                  'Convince Claire to take him on the 06:00 AM train before '
                  'the vampire blood in his veins finishes turning him.'
              ),
              'standing_with_player': 0,
              'standing': {'house_valerius': 1, 'thorn_covenant': -2},
              'assets': [
                  'The Night-Shift Breakroom',
                  'three stolen O-negative bags',
              ],
              'register': (
                  'Shivering, earnest, half-terrified and half-intoxicated by '
                  'what Claire did to save his life.'
              ),
          },
      ],
      'truths': [
          {
              'id': 't_forbidden_bond',
              'statement': (
                  'Claire gave Dr. David Park her own blood in the ambulance '
                  'bay an hour ago after a violent patient severed his carotid '
                  'artery—and David hid the three missing O-negative bags in '
                  'the breakroom because his human pulse is already slowing.'
              ),
              'arc_kind': 'bond',
              'stakes': (
                  'Whether Dr. David Park boards the 06:00 AM train with you '
                  'and Claire, stays behind, or never gets the choice'
              ),
              'question': 'What happened to Dr. Park in the ambulance bay?',
              'clue_ids': [
                  'c_ribbon_vow',
                  'c_parapet_embrace',
                  'c_seraphine_confession',
              ],
              'min_clues_to_discover': 2,
          },
          {
              'id': 't_crimson_betrayal',
              'statement': (
                  'Detective Arthur Finch has not alerted the police—he '
                  'tracked your 1986 and 1998 hospital ID photos on his own '
                  'after his wife died in St. Jude’s oncology ward, and he '
                  'wants a vial of your blood in exchange for the file.'
              ),
              'arc_kind': 'betrayal',
              'stakes': (
                  'Whether Detective Finch’s forty-year photo file is '
                  'bargained away, destroyed, or handed to the morning shift'
              ),
              'question': 'What does Detective Finch want?',
              'clue_ids': ['c_chalice_ash', 'c_reliquary_ledger'],
              'min_clues_to_discover': 2,
          },
      ],
      'clues': [
          {
              'id': 'c_chalice_ash',
              'truth_id': 't_crimson_betrayal',
              'carrier': 'location',
              'catalyst_kind': 'evidence',
              'carrier_hint': (
                  'The Basement Blood Bank & Lab, red transfusion sign-out'
                  ' binder'
              ),
              'text': (
                  'Pressed inside the transfusion sign-out binder is a '
                  'security inquiry slip signed by Arthur Finch requesting '
                  'archived personnel photos for Claire Moreau from St. '
                  'Mary’s (1986) and Mercy General (1998).'
              ),
          },
          {
              'id': 'c_ribbon_vow',
              'truth_id': 't_forbidden_bond',
              'carrier': 'bond',
              'catalyst_kind': 'vow',
              'carrier_hint': (
                  'The Basement Blood Bank & Lab, Claire’s canvas duffel bag'
              ),
              'text': (
                  'Inside Claire’s zipped duffel bag is a blood-spotted ER '
                  'trauma coat from an hour ago and a note in her handwriting: '
                  '"His artery was cut open on the bay floor. I couldn’t just '
                  'watch him die."'
              ),
          },
          {
              'id': 'c_parapet_embrace',
              'truth_id': 't_forbidden_bond',
              'carrier': 'moment',
              'catalyst_kind': 'intimacy',
              'carrier_hint': 'The Night-Shift Breakroom',
              'text': (
                  'In Dr. David Park’s breakroom locker sit the three missing '
                  'bags of O-negative—warmed against the radiator—and beneath '
                  'his neck gauze, a deep knife wound has already sealed into '
                  'a pale scar.'
              ),
          },
          {
              'id': 'c_seraphine_confession',
              'truth_id': 't_forbidden_bond',
              'carrier': 'npc',
              'catalyst_kind': 'confession',
              'carrier_hint': 'Claire Moreau',
              'text': (
                  'Claire admits she gave David her blood when his heart '
                  'stopped in the ambulance bay, and says she cannot board '
                  'the 06:00 AM train if it means leaving him alone for his '
                  'first sunrise.'
              ),
          },
          {
              'id': 'c_reliquary_ledger',
              'truth_id': 't_crimson_betrayal',
              'carrier': 'document',
              'catalyst_kind': 'betrayal',
              'carrier_hint': 'The Records & Security Corridor',
              'text': (
                  'Finch’s manila folder on the security desk holds your 1986 '
                  'and 1998 hospital ID badges clipped beside a St. Jude’s '
                  'oncology chart for his late wife, Helen Finch—no official '
                  'police report has been filed.'
              ),
          },
      ],
      'seed_npcs': [
          {
              'name': 'Claire Moreau',
              'role_summary': (
                  'Your partner of forty years and fellow night-shift lab '
                  'tech, standing by the centrifuge with her winter coat on'
              ),
              'goal': (
                  'Convince you to help her get young Dr. David Park safely '
                  'out of St. Jude’s before his fever turns or Detective '
                  'Finch locks the exits'
              ),
              'traits': (
                  'Calm voice, tired eyes that look twenty-eight, keeps '
                  'touching the silver wristwatch you gave her in 1991'
              ),
              'location_name': 'The Basement Blood Bank & Lab',
              'tier': 'hot',
              'faction': 'house_valerius',
              'secret': (
                  'Gave Dr. David Park her own blood in the ambulance bay an '
                  'hour ago after a violent patient severed his carotid '
                  'artery—knowing it would start the turn.'
              ),
              'wound': (
                  'After forty years of living like ghosts and leaving every '
                  'five years, she couldn’t bear watching a kid who was kind '
                  'to her bleed to death on the concrete.'
              ),
              'formative_memories': [
                  'Held David’s neck in the snow-blown ambulance bay until his '
                  'wound closed.'
              ],
              'routine': [
                  {
                      'start_hour': 18,
                      'end_hour': 6,
                      'location_name': 'The Basement Blood Bank & Lab',
                      'activity': 'running the night-shift hematology bench',
                  },
                  {
                      'start_hour': 6,
                      'end_hour': 18,
                      'location_name': 'The Night-Shift Breakroom',
                      'activity': 'resting behind blacked-out shutters',
                  },
              ],
              'relationships': {
                  'Detective Arthur Finch': (
                      'Saw him asking the records clerk for 1990s employee '
                      'files an hour ago.'
                  ),
                  'Dr. David Park': (
                      'Feels responsible for the life she bound to hers '
                      'tonight.'
                  ),
              },
          },
          {
              'name': 'Detective Arthur Finch',
              'role_summary': (
                  'Retired police investigator working night security, holding '
                  'a manila folder of hospital ID photos from 1986, 1998, and '
                  'tonight'
              ),
              'goal': (
                  'Confirm what you and Claire are and trade his forty-year '
                  'photo file for the secret of how you never age or die'
              ),
              'traits': (
                  'Sixty-eight, frayed wool overcoat, speaks quietly because '
                  'he came here for a bargain, not an arrest'
              ),
              'location_name': 'The Records & Security Corridor',
              'tier': 'warm',
              'faction': 'thorn_covenant',
              'secret': (
                  'Has not told the police or hospital board what he found—his '
                  'wife Helen died in St. Jude’s oncology ward last winter, '
                  'and he kept the file private.'
              ),
              'wound': (
                  'Watched his wife die under hospital fluorescent lights and '
                  'cannot accept that you two get decades untouched while '
                  'ordinary people run out of time.'
              ),
              'formative_memories': [
                  'Matched Claire’s 1986 St. Mary’s ID badge to her St. Jude’s '
                  'security badge tonight.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Records & Security Corridor',
                  'activity': (
                      'watching the basement stairwell and reviewing the '
                      'manila folder'
                  ),
              }],
              'relationships': {
                  'Claire Moreau': (
                      'Has been tracing her old aliases across three hospitals.'
                  )
              },
          },
          {
              'name': 'Dr. David Park',
              'role_summary': (
                  'Twenty-six-year-old first-year ER resident sitting in the '
                  'breakroom with gauze taped to his neck, shivering under a '
                  'scrub jacket'
              ),
              'goal': (
                  'Convince Claire and you to take him on the 06:00 AM train '
                  'before the change finishes taking hold'
              ),
              'traits': (
                  'Earnest, dilated pupils, squinting against the fluorescent '
                  'lights as his human pulse slows down'
              ),
              'location_name': 'The Night-Shift Breakroom',
              'tier': 'cold',
              'faction': 'mortal_embassy',
              'secret': (
                  'Used Claire’s keycard to take the three missing O-negative '
                  'bags from the cooler because the smell of breakroom coffee '
                  'made him violently sick.'
              ),
              'wound': (
                  'Terrified that you will force Claire to leave him behind in '
                  'the hospital when sunrise comes.'
              ),
              'formative_memories': [
                  'Saw Detective Finch photographing the blood cooler sign-out '
                  'sheet.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Night-Shift Breakroom',
                  'activity': (
                      'pressing a hand to the bandage on his neck and watching '
                      'the clock'
                  ),
              }],
              'relationships': {
                  'Claire Moreau': (
                      'Believes she saved his life and refuses to let her go '
                      'alone.'
                  )
              },
          },
      ],
      'doom': {
          'name': 'sunrise & 06:00 AM shift change',
          'total_minutes': 720,
          'stages': [
              {
                  'index': 0,
                  'name': '03:00 AM snowstorm over St. Jude’s',
                  'at_fraction': 0.0,
                  'world_effect': (
                      'Snow blows against the ambulance bay curtains while the '
                      'blood cooler alarm blinks amber.'
                  ),
              },
              {
                  'index': 1,
                  'name': 'security lockdown on the basement floor',
                  'at_fraction': 0.5,
                  'world_effect': (
                      'Detective Finch keys the night-latch on the corridor '
                      'doors and asks to see the cooler inventory.'
                  ),
              },
              {
                  'index': 2,
                  'name': '06:00 AM day shift & first light',
                  'at_fraction': 1.0,
                  'world_effect': (
                      'Grey morning light reaches the ground-floor windows as '
                      'the day-shift hematology crew keys into the elevator.'
                  ),
              },
          ],
      },
      'acts': [
          {
              'act': 1,
              'tone_note': (
                  'Act 1: three O-negative bags are missing, David is bandaged '
                  'in the breakroom, and Claire has her coat on.'
              ),
          },
          {
              'act': 2,
              'truths_discovered_gte': 1,
              'tone_note': (
                  'Act 2: what happened in the ambulance bay and Finch’s '
                  'forty-year photo file are out in the open.'
              ),
          },
          {
              'act': 3,
              'truths_discovered_gte': 2,
              'tone_note': (
                  'Act 3: the 06:00 AM train is boarding soon; decide who '
                  'leaves St. Jude’s and what happens to Finch’s file.'
              ),
          },
      ],
      'storylets': [
          {
              'id': 's_seraphine_tests_glance',
              'trigger': 'scene',
              'weight': 4,
              'once': True,
              'preconditions': [
                  {'npc_present': 'Claire Moreau'},
                  {'act_is': 1},
              ],
              'beat': (
                  'Claire rests her hand on the centrifuge lid and glances '
                  'toward the breakroom door, waiting to see if you noticed '
                  'the three empty hooks in the O-negative cooler.'
              ),
              'effects': [{'tension': 1}],
              'voice': 'Unspoken Vow',
              'voice_line': (
                  'She’s touching the silver watch from 1991. She only does '
                  'that when she’s afraid you’re going to walk out on her.'
              ),
          },
          {
              'id': 's_chalice_inspected',
              'trigger': 'scene',
              'weight': 3,
              'once': False,
              'preconditions': [
                  {'location_is': 'The Basement Blood Bank & Lab'},
                  {'clue_undiscovered': 'c_chalice_ash'},
              ],
              'beat': (
                  'If the player inspects the red transfusion sign-out binder '
                  'or the blood cooler, reveal c_chalice_ash.'
              ),
              'effects': [],
              'voice': 'Night Chart',
              'voice_line': (
                  'Three bags of O-negative gone since 02:00 AM, and there’s a '
                  'records request slip stuck in the back of the binder.'
              ),
          },
          {
              'id': 'a_salon_arrival',
              'trigger': 'arrival',
              'weight': 4,
              'once': True,
              'preconditions': [
                  {'location_is': 'The Records & Security Corridor'}
              ],
              'beat': (
                  'Detective Arthur Finch looks up from the security desk lamp,'
                  ' one hand resting on a manila folder marked with hospital'
                  ' stamps from 1986 and 1998.'
              ),
              'effects': [],
          },
          {
              'id': 'w_malakor_whispers',
              'trigger': 'warm',
              'weight': 4,
              'once': True,
              'preconditions': [{'truth_undiscovered': 't_crimson_betrayal'}],
              'beat': (
                  'Down the hall, Detective Finch pulls the 1998 Mercy General '
                  'personnel microfiche from his coat pocket.'
              ),
              'effects': [{
                  'rumor': {
                      'text': (
                          'Finch never logged his inquiry with police '
                          'dispatch—he pulled the old hospital files himself.'
                      ),
                      'source': 'night-shift switchboard',
                      'reliability': 'distorted',
                  }
              }],
          },
          {
              'id': 'd_stage1_edict',
              'trigger': 'doom',
              'weight': 10,
              'once': True,
              'preconditions': [{'doom_stage_gte': 1}],
              'beat': (
                  'The magnetic night-locks click shut along the basement '
                  'corridor as the 04:30 AM security check begins.'
              ),
              'effects': [{'tension': 1}],
          },
          {
              'id': 'r_seraphine_recalls',
              'trigger': 'rehydrate',
              'weight': 5,
              'once': False,
              'preconditions': [{'npc_present': 'Claire Moreau'}],
              'beat': (
                  'Claire steps closer to the bench and quietly asks what '
                  'Finch or David said to you down the hall.'
              ),
              'effects': [],
          },
          {
              'id': 'e_crimson_reconciliation',
              'trigger': 'ending',
              'weight': 50,
              'once': True,
              'preconditions': [
                  {'truth_discovered': 't_forbidden_bond'},
                  {'truth_discovered': 't_crimson_betrayal'},
              ],
              'beat': (
                  'With the truth about David’s turn and Finch’s forty-year '
                  'photo file on the table, the night shift reaches its '
                  'breaking point before the 06:00 AM train.'
              ),
              'effects': [{'end_run': 'success'}],
          },
      ],
  }


def _build_dystopian_scifi_template_dict(
    *,
    title: str,
    premise: str,
    tone: dict[str, Any],
) -> dict[str, Any]:
  """Builds 'The Sabotage at Outpost 7' paranoid frontier sci-fi WorldBible."""
  voices = {
      'Dead-Reckoning': (
          'Station Logs & Supply Tallies — Counts ration tins, diesel burn '
          'rates, radio timestamps, and weight discrepancies on the cargo '
          'manifest.'
      ),
      'Cabin Fever': (
          'Small-Talk & Micro-Tells — Knows what twelve months at a '
          'four-person station does to a person’s hands, coffee habits, and '
          'stories about home.'
      ),
      'Hard Line': (
          'Sidearm, Sightlines & Bolts — Measures who is closest to the bolt '
          'cutters, which bulkhead door latches from the inside, and how many '
          'rounds are left in your revolver.'
      ),
      'Company Book': (
          'Contract Law & Liability — Knows how frontier mining corporations '
          'write hazard waivers, non-disclosure clauses, and quiet '
          'clean-up orders.'
      ),
      'Side-Angle': (
          'Backroom Deals & Shuttle Seats — Knows what a cornered crew member '
          'will trade for a clean report and a guaranteed seat on the 06:00 '
          'transport.'
      ),
      'Biometric Lens': (
          'Hand-Scanner & Medical Forensics — Reads corneal gas-burns, pupil '
          'dilation, chemical residue on parka cuffs, and the three-day '
          'memory gap from a clinical sedative.'
      ),
  }
  player_sub_minds = {
      'Dead-Reckoning': {
          'description': voices['Dead-Reckoning'],
          'rank': 3,
          'domain': 'accounting',
          'blind_spot': (
              'Gets absorbed in logbook discrepancies while someone reaches '
              'for a heavy wrench.'
          ),
          'role_archetype': 'analytical',
          'attention_triggers': [
              'manifest',
              'log',
              'weight',
              'crate',
              'timestamp',
              'tally',
          ],
          'strong_against': ['leverage', 'audacity', 'stonewalling'],
          'vulnerable_to': ['tactical_force', 'empathy'],
          'favored_tactics': [
              'check the cargo sled weight slip against the food tins',
              'compare the handwriting in the generator logbook',
          ],
          'voice_tone': (
              'Dry, methodical, measures everything in pounds and hours.'
          ),
      },
      'Cabin Fever': {
          'description': voices['Cabin Fever'],
          'rank': 3,
          'domain': 'empathy',
          'blind_spot': (
              'Sympathizes with an exhausted crewman’s isolation and hesitates '
              'to press hard when stories falter.'
          ),
          'role_archetype': 'empathy',
          'attention_triggers': [
              'coffee',
              'parka',
              'tag',
              'hands',
              'memory',
              'voice',
          ],
          'strong_against': ['self_preservation', 'stonewalling'],
          'vulnerable_to': ['tactical_force', 'accounting'],
          'favored_tactics': [
              (
                  'ask about the last shift he clearly remembers before the'
                  ' storm hit'
              ),
              (
                  'watch how he holds his coffee mug with that scarred left'
                  ' knuckle'
              ),
          ],
          'voice_tone': 'Quiet, observant, worn down by long winters.',
      },
      'Hard Line': {
          'description': voices['Hard Line'],
          'rank': 3,
          'domain': 'tactical_force',
          'blind_spot': (
              'Draws iron too early in a cramped room and turns panic into a '
              'bloodbath.'
          ),
          'role_archetype': 'instinct',
          'attention_triggers': [
              'cutters',
              'latch',
              'revolver',
              'door',
              'wrench',
              'holster',
          ],
          'strong_against': ['leverage', 'cybernetics', 'stonewalling'],
          'vulnerable_to': ['accounting', 'hierarchy'],
          'favored_tactics': [
              'kick the bolt cutters out of reach and thumb the hammer back',
              'check which side of the airlock door was wedged shut',
          ],
          'voice_tone': 'Blunt, steady, keeps its back to the bulkhead.',
      },
      'Company Book': {
          'description': voices['Company Book'],
          'rank': 2,
          'domain': 'hierarchy',
          'blind_spot': (
              'Assumes company regulations still protect you six hundred miles '
              'from the nearest settlement.'
          ),
          'role_archetype': 'authority',
          'attention_triggers': [
              'contract',
              'company',
              'waiver',
              'director',
              'seal',
              'clearance',
          ],
          'strong_against': ['stonewalling', 'leverage'],
          'vulnerable_to': ['tactical_force', 'cybernetics'],
          'favored_tactics': [
              'cite the station charter and demand the sealed medical binder',
              'remind them who signs the boarding manifest at 06:00 AM',
          ],
          'voice_tone': 'Clipped, official, fluent in company fine print.',
      },
      'Side-Angle': {
          'description': voices['Side-Angle'],
          'rank': 2,
          'domain': 'leverage',
          'blind_spot': (
              'Tries to cut a quiet side-deal with people who cannot afford to '
              'leave a witness alive.'
          ),
          'role_archetype': 'transactional',
          'attention_triggers': [
              'sample',
              'shuttle',
              'seat',
              'bonus',
              'bribe',
              'ticket',
          ],
          'strong_against': ['self_preservation', 'hierarchy'],
          'vulnerable_to': ['tactical_force', 'accounting'],
          'favored_tactics': [
              'offer to leave their name off the incident report for the truth',
              'trade a guaranteed seat on the morning transport for the key',
          ],
          'voice_tone': 'Pragmatic, low-voiced, always counting the angles.',
      },
      'Biometric Lens': {
          'description': voices['Biometric Lens'],
          'rank': 2,
          'domain': 'cybernetics',
          'blind_spot': (
              'Trusts the pocket scanner readout over the human being standing '
              'in front of you.'
          ),
          'role_archetype': 'analytical',
          'attention_triggers': [
              'cornea',
              'pulse',
              'pupil',
              'wrist',
              'scar',
              'scanner',
          ],
          'strong_against': ['stonewalling', 'hierarchy', 'accounting'],
          'vulnerable_to': ['tactical_force', 'empathy'],
          'favored_tactics': [
              (
                  'shine the pocket scanner across his cornea to check for'
                  ' chemical haze'
              ),
              'scan the puncture mark on his forearm for sedative residue',
          ],
          'voice_tone': 'Cool, clinical, reads bodies like diagnostic charts.',
      },
  }
  character_presets = [
      {
          'id': 'relief_marshal',
          'name': 'The Veteran Relief Marshal',
          'tagline': 'Hard Line 4 · Dead-Reckoning 3 — Steady & Methodical',
          'description': (
              'Twenty years policing remote mining towns and weather rigs. You '
              'keep your thumb on the hammer, count the coffee mugs, and check '
              'the cargo weights twice.'
          ),
          'ranks': {
              'Hard Line': 4,
              'Dead-Reckoning': 3,
              'Cabin Fever': 2,
              'Company Book': 2,
              'Biometric Lens': 2,
              'Side-Angle': 1,
          },
          'inventory': [
              'dead man’s brass locker key from the supply crate',
              'Polaroid snapshot of the corpse in the crate',
              'marshal’s service revolver (three rounds left)',
          ],
      },
      {
          'id': 'clearance_auditor',
          'name': 'The Clearance Auditor',
          'tagline': 'Cabin Fever 4 · Biometric Lens 3 — Reads Every Tell',
          'description': (
              'Trained to investigate remote station cover-ups by'
              ' cross-checking shift logs, chemical exposure signs, and the'
              ' small habits an rehearsed alibi gets wrong.'
          ),
          'ranks': {
              'Cabin Fever': 4,
              'Biometric Lens': 3,
              'Dead-Reckoning': 3,
              'Company Book': 2,
              'Side-Angle': 1,
              'Hard Line': 1,
          },
          'inventory': [
              'pocket corneal scanner',
              'Polaroid snapshot of the corpse in the crate',
              'dead man’s brass locker key from the supply crate',
          ],
      },
      {
          'id': 'company_fixer',
          'name': 'The Company Troubleshooter',
          'tagline': 'Side-Angle 4 · Company Book 3 — Pragmatic & Sharp',
          'description': (
              'You know the company never buries an inspection report '
              'without a balance-sheet reason, and you know how to walk onto '
              'the morning shuttle with a bonus.'
          ),
          'ranks': {
              'Side-Angle': 4,
              'Company Book': 3,
              'Dead-Reckoning': 2,
              'Hard Line': 2,
              'Biometric Lens': 2,
              'Cabin Fever': 1,
          },
          'inventory': [
              'company winter handover clipboard',
              'dead man’s brass locker key from the supply crate',
              'Polaroid snapshot of the corpse in the crate',
          ],
      },
  ]
  default_title = (
      'The Sabotage at Outpost 7'
      if not title.strip() or 'Sector 9' in title
      else title.strip()
  )
  if len(premise.strip()) > 160 and 'Sector 9' not in premise:
    rich_premise = premise.strip()
  else:
    rich_premise = (
        'Outpost 7 is a four-person weather and deep-bore relay station on the '
        'frozen salt flats of Kepler-442b, six hundred miles from the nearest '
        'settlement. An hour ago, the monthly unmanned supply sled dropped '
        'out of the storm and skidded into the hangar. Inside the main '
        'cargo crate—packed in dry ice underneath the canned peaches and '
        'diesel filters—was a dead man wearing Jonas Reed’s stitched Outpost 7 '
        'parka and brass transit tag. He has been dead for three days, while '
        'Jonas Reed himself, the station’s chief mechanic, is sitting in the '
        'mess hall drinking black coffee with a three-day blank in his memory. '
        'Someone has cut the roof radio wire with bolt cutters, and the '
        'company retrieval transport lands at 06:00 AM (12 hours from now).'
    )
  return {
      'title': default_title,
      'premise': rich_premise,
      'opening': (
          'You step out of the hangar airlock into The Outpost 7 Mess Hall, '
          'stomping salt-crust off your boots. In your coat pocket sit the '
          'dead man’s brass locker key, a Polaroid you just snapped of the '
          'corpse in the crate, and your handheld biometric scanner. You are '
          'the Relief Marshal sent out to oversee the station’s winter '
          'handover. The roof radio is dead. Before the company transport '
          'lands at 06:00 AM (12 hours from now), you must find out who the '
          'dead man in Jonas Reed’s parka is, why Jonas is missing three days '
          'of memory, who silenced the station, and decide who walks onto that '
          'morning transport alive.'
      ),
      'tone': tone,
      'player_voices': voices,
      'player_sub_minds': player_sub_minds,
      'character_presets': character_presets,
      'emergent_voice_pool': [{
          'name': 'The Mirror Test',
          'description': (
              'Emerged after piecing together the station’s chemical logs — '
              'checks every dilated pupil and tremor for signs of a memory '
              'purge.'
          ),
          'domain': 'cybernetics',
          'role_archetype': 'analytical',
          'rank': 3,
          'keywords': [
              'purge',
              'sedative',
              'corpse',
              'wrist',
              'mirror',
              'scanner',
          ],
          'min_Count': 1,
          'attention_triggers': ['tag', 'scar', 'wrist', 'photo', 'memory'],
          'strong_against': ['stonewalling', 'hierarchy'],
          'vulnerable_to': ['tactical_force', 'empathy'],
          'favored_tactics': [
              'lay the Polaroid of the corpse next to his coffee mug',
              'ask him why his brass transit tag was in the cargo crate',
          ],
      }],
      'starting_inventory': [
          'dead man’s brass locker key from the supply crate',
          'Polaroid snapshot of the corpse in the crate',
          'handheld pocket biometric scanner',
      ],
      'starting_location': {
          'name': 'The Outpost 7 Mess Hall',
          'description': (
              'Rime frost creeps along the bottom edge of the double-paned '
              'portholes while a diesel heater clanks steadily in the corner. '
              'A half-eaten pot of beef stew sits on the aluminum table '
              'beside a scarred cribbage board, a rack of wet parkas by the '
              'airlock door, and a locked steel medicine cabinet bolted to '
              'the bulkhead.'
          ),
          'affordances': [
              'pot of beef stew and two coffee mugs on the aluminum table',
              (
                  'rack of mud-caked winter parkas and heavy bolt cutters by'
                  ' the door'
              ),
              (
                  'scarred cribbage board with a torn radio dispatch slip'
                  ' underneath'
              ),
              'locked steel medicine cabinet bolted to the bulkhead',
          ],
          'exits': {
              'north': 'The Cargo Hangar & Cold Locker',
              'east': 'The Medical & Triage Bay',
              'west': 'The Radio & Generator Shack',
          },
          'clue_ids': ['c_airlock_telemetry', 'c_duct_splice'],
      },
      'factions': [
          {
              'id': 'helix_corp',
              'name': 'Kepler Mining & Relay Corp',
              'aim': (
                  'Have a compliant Jonas Reed sign the clean winter'
                  ' drill-safety certificate before the 06:00 AM transport'
                  ' lands, and bury any record of the dead inspector.'
              ),
              'standing_with_player': 0,
              'standing': {'splice_underground': -3, 'subdeck_union': -2},
              'assets': [
                  'The Medical & Triage Bay',
                  '06:00 AM retrieval transport',
              ],
              'register': (
                  'Plain, polite company liability language that treats crew '
                  'members like replaceable spare parts.'
              ),
          },
          {
              'id': 'splice_underground',
              'name': 'The Station Crew (Jonas & Mara)',
              'aim': (
                  'Survive the night until the 06:00 AM transport without being'
                  ' framed for sabotage or arrested for the corpse in the'
                  ' crate.'
              ),
              'standing_with_player': 0,
              'standing': {'helix_corp': -3, 'subdeck_union': 1},
              'assets': ['The Outpost 7 Mess Hall', 'station tool lockers'],
              'register': (
                  'Tired, tight-lipped blue-collar station talk where everyone '
                  'is watching each other’s hands.'
              ),
          },
          {
              'id': 'subdeck_union',
              'name': 'Independent Prospectors (Cole’s Buyers)',
              'aim': (
                  'Smuggle the stolen lithium-brine core samples onto the'
                  ' morning transport before the marshal searches the radio'
                  ' shack.'
              ),
              'standing_with_player': 0,
              'standing': {'helix_corp': -2, 'splice_underground': 1},
              'assets': [
                  'The Radio & Generator Shack',
                  'hidden sample case under the floorboards',
              ],
              'register': (
                  'Nervous, transactional, eager to point fingers elsewhere.'
              ),
          },
      ],
      'truths': [
          {
              'id': 't_oxygen_cull',
              'statement': (
                  'The corpse in the supply crate is undercover safety auditor '
                  'Elias Ward. When Ward discovered toxic mercury-brine '
                  'leaking from the deep-bore drill, Dr. Mara Lin sedated him '
                  'on company orders—and he froze to death after being hidden '
                  'in the outgoing crate.'
              ),
              'arc_kind': 'system_shift',
              'stakes': (
                  'Whether what the company did at Outpost 7 reaches the '
                  'mainland at 06:00 AM or is buried with the crate'
              ),
              'question': (
                  'Who is the dead man in Jonas Reed’s parka, and why did he'
                  ' die?'
              ),
              'clue_ids': [
                  'c_airlock_telemetry',
                  'c_duct_splice',
                  'c_vos_override_log',
              ],
              'min_clues_to_discover': 2,
          },
          {
              'id': 't_eidolon_ghost',
              'statement': (
                  'Jonas Reed swapped his parka and transit tag with auditor '
                  'Elias Ward to help him smuggle the brine-leak notebook out, '
                  'before Dr. Mara Lin dosed Jonas with neuro-scrubber purge '
                  'gas (Protocol 7-REED-B) in the infirmary to erase his '
                  'memory of the swap.'
              ),
              'arc_kind': 'revelation',
              'stakes': (
                  'Whether Jonas Reed is cleared as a witness or framed '
                  'when the transport lands'
              ),
              'question': 'Why is Jonas Reed missing three days of memory?',
              'clue_ids': ['c_nyx_memory_shard', 'c_spine_core_handshake'],
              'min_clues_to_discover': 2,
          },
          {
              'id': 't_scrubber_bypass',
              'statement': (
                  'Radio operator Cole Mercer cut the roof antenna with bolt '
                  'cutters and wedged the hangar door because auditor Elias '
                  'Ward caught him selling stolen lithium-brine core '
                  'samples—trapping the sedated auditor in the freezing hangar.'
              ),
              'arc_kind': 'revelation',
              'stakes': (
                  'Whether whoever silenced the station answers for it '
                  'before the transport lands'
              ),
              'question': 'Who silenced the station?',
              'clue_ids': ['c_duct_splice', 'c_spine_core_handshake'],
              'min_clues_to_discover': 2,
          },
      ],
      'clues': [
          {
              'id': 'c_airlock_telemetry',
              'truth_id': 't_oxygen_cull',
              'carrier': 'system',
              'catalyst_kind': 'transmission',
              'carrier_hint': (
                  'The Outpost 7 Mess Hall, cribbage board dispatch slip'
              ),
              'text': (
                  'A torn company dispatch slip under the cribbage board orders'
                  ' Dr. Lin to "execute Protocol 7-REED-B and detain the'
                  ' visiting auditor if the winter brine-drill clearance is'
                  ' refused."'
              ),
          },
          {
              'id': 'c_duct_splice',
              'truth_id': 't_oxygen_cull',
              'carrier': 'location',
              'catalyst_kind': 'sabotage',
              'carrier_hint': 'The Outpost 7 Mess Hall, bolt cutters by door',
              'text': (
                  'The heavy bolt cutters by the airlock door have shreds of '
                  'red roof-antenna coaxial insulation caught in their jaws '
                  'and grease from Cole Mercer’s radio rig.'
              ),
          },
          {
              'id': 'c_vos_override_log',
              'truth_id': 't_oxygen_cull',
              'carrier': 'npc',
              'catalyst_kind': 'confession',
              'carrier_hint': 'Jonas Reed',
              'text': (
                  'Jonas rolls up his flannel sleeve with a trembling hand to'
                  ' show a bruised IV puncture stamped "7-REED-B" on his wrist'
                  ' and admits he lent his parka to a visiting safety auditor'
                  ' on Tuesday night before waking up on the medical bay cot.'
              ),
          },
          {
              'id': 'c_nyx_memory_shard',
              'truth_id': 't_eidolon_ghost',
              'carrier': 'document',
              'catalyst_kind': 'evidence',
              'carrier_hint': 'The Cargo Hangar & Cold Locker',
              'text': (
                  'Inside the dead man’s borrowed parka pocket in the supply '
                  'crate is a laminated inspector credential for "Elias Ward, '
                  'Regional Safety Auditor" and a notebook logging illegal '
                  'mercury-brine leaks signed jointly by Ward and Jonas Reed.'
              ),
          },
          {
              'id': 'c_spine_core_handshake',
              'truth_id': 't_eidolon_ghost',
              'carrier': 'system',
              'catalyst_kind': 'transmission',
              'carrier_hint': 'The Medical & Triage Bay',
              'text': (
                  'The locked chemical cabinet in the back of the medical bay '
                  'logs a three-day neuro-scrubber sedation cycle (Protocol '
                  '7-REED-B) administered seventy-two hours ago, alongside two '
                  'empty vials of clinical sedative.'
              ),
          },
      ],
      'seed_npcs': [
          {
              'name': 'Jonas Reed',
              'role_summary': (
                  'Station chief mechanic sitting at the mess table with a mug'
                  ' of black coffee, missing his winter parka and brass transit'
                  ' tag'
              ),
              'goal': (
                  'Piece together his three-day memory gap and secure a seat '
                  'on the 06:00 AM transport home without being framed for the '
                  'body in the crate'
              ),
              'traits': (
                  'Calloused hands, quiet blue-collar dignity, keeps rubbing '
                  'the scar on his left knuckle as his memory skips'
              ),
              'location_name': 'The Outpost 7 Mess Hall',
              'tier': 'hot',
              'faction': 'splice_underground',
              'secret': (
                  'Woke up on the medical bay cot three days ago with a blank '
                  'spot in his memory and found the protocol stamp 7-REED-B '
                  'inked over an IV bruise on his wrist an hour ago.'
              ),
              'wound': (
                  'Remembers promising to help a visiting safety inspector '
                  'check the deep-bore brine valves on Tuesday night, and is '
                  'terrified the company drugged him to frame him for murder.'
              ),
              'formative_memories': [
                  'Found a dead man wearing his stitched Outpost 7 parka '
                  'packed in dry ice inside the hangar supply crate an hour '
                  'ago.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Outpost 7 Mess Hall',
                  'activity': 'nursing black coffee and watching the door',
              }],
              'relationships': {
                  'Dr. Mara Lin': (
                      'Realizes she gave him a "vitamin shot" three mornings '
                      'ago when he woke up on her cot.'
                  ),
                  'Cole Mercer': (
                      'Remembers arguing with Cole about the radio shack '
                      'before his memory went blank.'
                  ),
              },
          },
          {
              'name': 'Dr. Mara Lin',
              'role_summary': (
                  'Outpost 7 station medic examining the dry-ice supply crate '
                  'in the unheated cargo hangar'
              ),
              'goal': (
                  'Destroy the neuro-scrubber sedation log and get Jonas Reed '
                  'to sign the company clearance before the 06:00 AM transport'
              ),
              'traits': (
                  'Composed, immaculate white parka, keeps one hand in her '
                  'coat pocket holding the dead man’s stopped wristwatch'
              ),
              'location_name': 'The Cargo Hangar & Cold Locker',
              'tier': 'warm',
              'faction': 'helix_corp',
              'secret': (
                  'Sedated undercover auditor Elias Ward on company orders and '
                  'put Jonas Reed under a three-day neuro-scrubber memory '
                  'purge when Jonas tried to help Ward.'
              ),
              'wound': (
                  'Never meant for Elias Ward to freeze to death—she hid the '
                  'sedated auditor in the supply crate to ship out alive '
                  'before Cole locked the unheated hangar door.'
              ),
              'formative_memories': [
                  'Hid the empty sedative ampoules inside the locked steel '
                  'medicine cabinet.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Cargo Hangar & Cold Locker',
                  'activity': 'inspecting the cargo crate and cold locker',
              }],
              'relationships': {
                  'Jonas Reed': (
                      'Cannot look Jonas in the eye when he asks why his wrist '
                      'is bruised.'
                  )
              },
          },
          {
              'name': 'Cole Mercer',
              'role_summary': (
                  'Station radio operator nursing a bruised jaw beside the '
                  'dead shortwave rig'
              ),
              'goal': (
                  'Blame the cut radio antenna on Jonas and smuggle '
                  'his stolen core samples onto the morning transport'
              ),
              'traits': (
                  'Restless, chain-smoking, flinches whenever the wind bangs '
                  'the hangar doors'
              ),
              'location_name': 'The Radio & Generator Shack',
              'tier': 'cold',
              'faction': 'subdeck_union',
              'secret': (
                  'Cut the roof radio antenna with bolt cutters and wedged the '
                  'hangar door from the inside after auditor Elias Ward caught '
                  'him packing stolen lithium core samples.'
              ),
              'wound': (
                  'Thought the man in Jonas’s parka was just locked in the '
                  'hangar with a space heater, not sedated in a dry-ice crate.'
              ),
              'formative_memories': [
                  'Left red coaxial insulation on the mess hall bolt cutters.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Radio & Generator Shack',
                  'activity': 'pretending to repair the severed shortwave rig',
              }],
              'relationships': {
                  'Dr. Mara Lin': (
                      'Suspects she is hiding something in the locked cabinet '
                      'behind the medical bay.'
                  )
              },
          },
      ],
      'doom': {
          'name': 'the 06:00 AM company retrieval transport',
          'total_minutes': 720,
          'stages': [
              {
                  'index': 0,
                  'name': 'blizzard over the salt flats',
                  'at_fraction': 0.0,
                  'world_effect': (
                      'Salt-wind hammers the portholes while the corpse in the '
                      'hangar crate begins to thaw.'
                  ),
              },
              {
                  'index': 1,
                  'name': 'transport beacon lock',
                  'at_fraction': 0.5,
                  'world_effect': (
                      'The landing beacon on the hangar roof begins flashing '
                      'green as the company shuttle enters descent range.'
                  ),
              },
              {
                  'index': 2,
                  'name': 'company transport touchdown',
                  'at_fraction': 1.0,
                  'world_effect': (
                      'Landing thrusters roar outside the hangar doors as the '
                      'company security team steps onto the salt flat.'
                  ),
              },
          ],
      },
      'acts': [
          {
              'act': 1,
              'tone_note': (
                  'Act 1: a dead man is in the crate wearing Jonas’s parka and '
                  'nobody admits how the body got there.'
              ),
          },
          {
              'act': 2,
              'truths_discovered_gte': 1,
              'tone_note': (
                  'Act 2: the company cover-up is out in the open; '
                  'fear turns the four-person station into a standoff.'
              ),
          },
          {
              'act': 3,
              'truths_discovered_gte': 2,
              'tone_note': (
                  'Act 3: landing lights sweep the salt flat; decide who boards'
                  ' the 06:00 AM transport and whose name goes on the warrant.'
              ),
          },
      ],
      'storylets': [
          {
              'id': 's_vos_scans_clearance',
              'trigger': 'scene',
              'weight': 4,
              'once': True,
              'preconditions': [
                  {'npc_present': 'Jonas Reed'},
                  {'act_is': 1},
              ],
              'beat': (
                  'Jonas wraps both hands around his tin coffee mug; you spot '
                  'the faded ink stamp on his wrist cuff just as he realizes '
                  'his winter parka is on the corpse in the hangar.'
              ),
              'effects': [{'tension': 1}],
              'voice': 'Cabin Fever',
              'voice_line': (
                  'He’s blowing on black coffee that went cold twenty minutes '
                  'ago. Ask him when he last slept.'
              ),
          },
          {
              'id': 's_terminal_inspected',
              'trigger': 'scene',
              'weight': 3,
              'once': False,
              'preconditions': [
                  {'location_is': 'The Outpost 7 Mess Hall'},
                  {'clue_undiscovered': 'c_airlock_telemetry'},
              ],
              'beat': (
                  'If the player inspects the cribbage board or the torn radio '
                  'dispatch slip, reveal c_airlock_telemetry.'
              ),
              'effects': [],
              'voice': 'Dead-Reckoning',
              'voice_line': (
                  'That dispatch slip under the cribbage board has a company '
                  'authorization code on it.'
              ),
          },
          {
              'id': 'a_clinic_arrival',
              'trigger': 'arrival',
              'weight': 4,
              'once': True,
              'preconditions': [
                  {'location_is': 'The Cargo Hangar & Cold Locker'}
              ],
              'beat': (
                  'Sub-zero wind whistles through the hangar seal as Dr. Mara '
                  'Lin stands over the open supply crate, where frost clings '
                  'to the dead man’s borrowed parka.'
              ),
              'effects': [],
          },
          {
              'id': 'w_drone_sweep',
              'trigger': 'warm',
              'weight': 4,
              'once': True,
              'preconditions': [{'truth_undiscovered': 't_oxygen_cull'}],
              'beat': (
                  'From the corridor comes the metallic clatter of someone '
                  'trying to force the medical bay cabinet latch.'
              ),
              'effects': [{
                  'rumor': {
                      'text': (
                          'Dr. Lin locked the back room of the medical bay '
                          'three nights ago and claimed the chemical vent '
                          'had malfunctioned.'
                      ),
                      'source': 'station logbook',
                      'reliability': 'distorted',
                  }
              }],
          },
          {
              'id': 'd_stage1_klaxon',
              'trigger': 'doom',
              'weight': 10,
              'once': True,
              'preconditions': [{'doom_stage_gte': 1}],
              'beat': (
                  'On the hangar roof, the automated green approach strobe '
                  'begins pulsing through the snow—the 06:00 AM company '
                  'transport has locked onto the station beacon.'
              ),
              'effects': [{'tension': 1}],
          },
          {
              'id': 'r_vos_recalls',
              'trigger': 'rehydrate',
              'weight': 5,
              'once': False,
              'preconditions': [{'npc_present': 'Jonas Reed'}],
              'beat': (
                  'Jonas looks up from the aluminum table, his voice unsteady, '
                  'and asks what you found in the hangar crate.'
              ),
              'effects': [],
          },
          {
              'id': 'e_eidolon_uplink',
              'trigger': 'ending',
              'weight': 50,
              'once': True,
              'preconditions': [
                  {'truth_discovered': 't_oxygen_cull'},
                  {'truth_discovered': 't_eidolon_ghost'},
              ],
              'beat': (
                  'As the company transport’s landing lights flood the mess '
                  'hall portholes, the truth about the Outpost 7 cover-up is '
                  'laid on the table.'
              ),
              'effects': [{'end_run': 'success'}],
          },
      ],
  }


def build_template_bible_dict(
    *,
    title: str,
    premise: str,
    tone_name: str = 'noir',
) -> dict[str, Any]:
  """Builds a complete, valid baseline WorldBible dictionary from a premise."""
  genre_mode = _resolve_genre_mode(tone_name, title, premise)
  tone = dict(TONE_PRESETS.get(genre_mode, TONE_PRESETS['noir']))
  if genre_mode == 'vampire_romance':
    return _build_vampire_romance_template_dict(
        title=title, premise=premise, tone=tone
    )
  if genre_mode == 'dystopian_scifi':
    return _build_dystopian_scifi_template_dict(
        title=title, premise=premise, tone=tone
    )
  if genre_mode == 'caper':
    voices = {
        'Brass Neck': (
            'Audacity & Bluff — Fronts out impossible situations and talks '
            'like you own the freehold when your pockets are lint.'
        ),
        'The Book': (
            'Forensic Odds & Ledgers — Runs the vig, the odds, and the '
            'double-cross maths. Spots a shaved margin or a bent ledger.'
        ),
        'Proper Naughty': (
            'Threat & Mechanics — Measures bottle, reach, locks, and '
            'concealed iron. Knows who fancies a row and who is all mouth.'
        ),
        'Ear to the Ground': (
            'Street Subtext & Tells — Reads pub subtext, rhyming slang, and '
            'who just looked at the back door.'
        ),
        'The Hustle': (
            'Leverage & Arbitrage — Spots the swap, the blackmail angle, and '
            'what a cornered punter will trade to save their skin.'
        ),
        "Copper's Nose": (
            'Old Bill Procedure & Forensics — Clocks warrant cards, radio '
            'logs, powder burns, and how bent detectives cover their tracks.'
        ),
        "The Guv'nor": (
            'Underworld Hierarchy & Rep — Reads Firm protocol, turf tribute, '
            'and when to invoke the boss’s name with weight.'
        ),
    }
    starting_inventory = [
        'monogrammed empty pigskin holdall',
        'pink Catford greyhound betting slip',
        'brass knuckle-duster lighter',
    ]
  else:
    voices = {
        'Suspicion': 'Reads every kindness as a hook. Short, flat sentences.',
        'Empathy': 'Notices what people are afraid of.',
        'Ledger': 'Counts dates, seals, and discrepancies.',
        'Nerve': 'Measures exits, distances, and danger.',
        'Mechanist': (
            'Reads valves, locks, pressure, and structural weak points.'
        ),
        'Jurisdiction': (
            'Knows harbor law, writs, charters, and official procedure.'
        ),
    }
    starting_inventory = [
        'unsealed inquiry writ',
        "inspector's pocket ledger",
    ]
  first_voice = 'Brass Neck' if genre_mode == 'caper' else 'Suspicion'
  second_voice = 'The Book' if genre_mode == 'caper' else 'Ledger'
  default_title = (
      'The Brass-bound Ledger of Deptford Creek'
      if genre_mode == 'caper'
      else 'The Iron Sluice of Glassharbor'
  )
  effective_title = (title or default_title).strip()
  effective_premise = (premise or '').strip()
  if not effective_premise:
    effective_premise = (
        f'In {effective_title}, the harbor clock is running down toward '
        'a midnight inspection while the Harbor Wardens and the Dockyard '
        'Collective trade accusations over a tampered sluice and a '
        'missing customs manifest.'
    )
  return {
      'title': effective_title,
      'premise': effective_premise,
      'opening': (
          'You arrive at The Lower Gatehouse carrying an unsealed inquiry'
          ' commission '
          'and your pocket ledger. You are an independent Investigator & '
          'Fixer summoned into the heart of the crisis with no faction '
          'leash. Before the final bell runs out, you must piece together the '
          'hidden truths behind the conspiracy, pit the rival factions '
          'against each other, and decide who pays the price.'
      ),
      'tone': tone,
      'player_voices': voices,
      'starting_inventory': starting_inventory,
      'starting_location': {
          'name': 'The Lower Gatehouse',
          'description': (
              'Rain-slick stone arches frame the gatehouse, smelling of wet '
              'iron, coal smoke, and lamp oil while brass pressure gauges '
              'hum along the inner sluice wall.'
          ),
          'affordances': [
              'chained entry ledger on the watch desk',
              'tampered brass seal on the inner sluice door',
              'chalk tally board behind the stove',
          ],
          'exits': {
              'north': 'The Lantern Commons',
              'east': 'The Pendulum Engine Hall',
              'west': 'The Sealed Chancery',
          },
          'clue_ids': ['c_ledger_entry', 'c_seal_scratch'],
      },
      'factions': [
          {
              'id': 'watch',
              'name': 'The Gate Watch',
              'aim': 'Keep order and confiscate unauthorized keys.',
              'standing_with_player': 0,
              'standing': {'syndicate': -3, 'scribes': 0},
              'assets': ['the gatehouse', 'the night patrol'],
              'register': 'Clipped, procedural, suspicious of unsealed papers.',
          },
          {
              'id': 'syndicate',
              'name': 'The Night Syndicate',
              'aim': 'Profit from the panic and slip contraband past the lock.',
              'standing_with_player': 0,
              'standing': {'watch': -3, 'scribes': -1},
              'assets': ['the commons back room', 'the runners'],
              'register': 'Warm, transactional, every answer is a trade.',
          },
          {
              'id': 'scribes',
              'name': 'The Chancery Scribes',
              'aim': 'Preserve the charters and hide past complicity.',
              'standing_with_player': 0,
              'standing': {'watch': 0, 'syndicate': -1},
              'assets': ['the Sealed Chancery', 'Folio 4'],
              'register': 'Formal, cautious, hides behind procedure.',
          },
      ],
      'truths': [
          {
              'id': 't_sabotage',
              'statement': (
                  'The mechanism was deliberately bled by a disgraced engineer '
                  'hired by the Night Syndicate.'
              ),
              'clue_ids': [
                  'c_ledger_entry',
                  'c_seal_scratch',
                  'c_syndicate_chit',
              ],
              'min_clues_to_discover': 2,
          },
          {
              'id': 't_bypass',
              'statement': (
                  'The Chancery holds Folio 4 showing a manual relief channel '
                  'that can avert catastrophe.'
              ),
              'clue_ids': ['c_index_card', 'c_scribe_admission'],
              'min_clues_to_discover': 2,
          },
      ],
      'clues': [
          {
              'id': 'c_ledger_entry',
              'truth_id': 't_sabotage',
              'carrier': 'document',
              'carrier_hint': 'The Lower Gatehouse, chained entry ledger',
              'text': (
                  'A midnight entry signed out replacement weights that never '
                  'reached the engine hall.'
              ),
          },
          {
              'id': 'c_seal_scratch',
              'truth_id': 't_sabotage',
              'carrier': 'location',
              'carrier_hint': 'The Lower Gatehouse, inner sluice door',
              'text': (
                  'Fresh tool marks under the grease show the seal was swapped '
                  'within the week.'
              ),
          },
          {
              'id': 'c_syndicate_chit',
              'truth_id': 't_sabotage',
              'carrier': 'document',
              'carrier_hint': 'The Lantern Commons',
              'text': (
                  'A promissory chit from the Syndicate offering safe passage '
                  'once the lock fails.'
              ),
          },
          {
              'id': 'c_index_card',
              'truth_id': 't_bypass',
              'carrier': 'document',
              'carrier_hint': 'The Sealed Chancery',
              'text': (
                  'A torn index card names Folio 4: the manual relief channel.'
              ),
          },
          {
              'id': 'c_scribe_admission',
              'truth_id': 't_bypass',
              'carrier': 'npc',
              'carrier_hint': 'Archivist Venn',
              'text': (
                  'Archivist Venn admits Folio 4 is hidden behind the west '
                  'wainscoting.'
              ),
          },
      ],
      'seed_npcs': [
          {
              'name': 'Warden Kesh',
              'role_summary': 'Gate Watch commander checking passes',
              'goal': 'Find who tampered with the inner sluice seal',
              'traits': 'Terse, observant, respects hard proof',
              'location_name': 'The Lower Gatehouse',
              'tier': 'hot',
              'faction': 'watch',
              'secret': (
                  'Looked the other way when a family debt was called in.'
              ),
              'wound': 'Will deal if the family name is kept out of the log.',
              'formative_memories': [
                  'The inner sluice seal was replaced during the midnight'
                  ' shift.'
              ],
              'routine': [
                  {
                      'start_hour': 6,
                      'end_hour': 18,
                      'location_name': 'The Lower Gatehouse',
                      'activity': 'inspecting passes',
                  },
                  {
                      'start_hour': 18,
                      'end_hour': 6,
                      'location_name': 'The Lantern Commons',
                      'activity': 'off-duty watch',
                  },
              ],
              'relationships': {'Broker Tamsin': 'Suspects her of smuggling.'},
          },
          {
              'name': 'Broker Tamsin',
              'role_summary': 'Syndicate fixer holding court in the commons',
              'goal': 'Move the contraband crates before the lock freezes',
              'traits': 'Quick-witted, bargaining, never answers for free',
              'location_name': 'The Lantern Commons',
              'tier': 'warm',
              'faction': 'syndicate',
              'secret': 'Paid for the weights to be removed from the engine.',
              'wound': 'Fears being trapped inside if the pass closes early.',
              'formative_memories': [
                  'Hid the promissory chit behind the commons wainscoting.'
              ],
              'routine': [{
                  'start_hour': 0,
                  'end_hour': 24,
                  'location_name': 'The Lantern Commons',
                  'activity': 'brokering trades',
              }],
              'relationships': {
                  'Warden Kesh': 'Watches every move Kesh makes.'
              },
          },
          {
              'name': 'Archivist Venn',
              'role_summary': 'Keeper of the Sealed Chancery',
              'goal': 'Protect the old charters from confiscation',
              'traits': 'Formal, anxious, cites procedure to stall strangers',
              'location_name': 'The Sealed Chancery',
              'tier': 'cold',
              'faction': 'scribes',
              'secret': 'Hid Folio 4 to conceal an old sale to the Syndicate.',
              'wound': (
                  'Yields when shown the torn index card or a signed writ.'
              ),
              'formative_memories': ['Folio 4 describes the relief channel.'],
              'routine': [
                  {
                      'start_hour': 7,
                      'end_hour': 20,
                      'location_name': 'The Sealed Chancery',
                      'activity': 'cataloguing',
                  },
                  {
                      'start_hour': 20,
                      'end_hour': 7,
                      'location_name': 'The Lantern Commons',
                      'activity': 'taking late tea',
                  },
              ],
              'relationships': {'Warden Kesh': 'Owes the watch a favor.'},
          },
      ],
      'doom': {
          'name': 'the final lock-freeze',
          'total_minutes': 720,
          'stages': [
              {
                  'index': 0,
                  'name': 'uneasy calm',
                  'at_fraction': 0.0,
                  'world_effect': 'Frost gathers on the brass gauges.',
              },
              {
                  'index': 1,
                  'name': 'the pendulum falters',
                  'at_fraction': 0.5,
                  'world_effect': 'Warning bells sound across the commons.',
              },
              {
                  'index': 2,
                  'name': 'the lock freezes shut',
                  'at_fraction': 1.0,
                  'world_effect': 'The main mechanism seizes completely.',
              },
          ],
      },
      'acts': [
          {
              'act': 1,
              'tone_note': 'Act 1: everyone is polite and nobody is helpful.',
          },
          {
              'act': 2,
              'truths_discovered_gte': 1,
              'tone_note': 'Act 2: the first truth is out; bargains sharpen.',
          },
          {
              'act': 3,
              'truths_discovered_gte': 2,
              'tone_note': 'Act 3: endings are in reach for anyone with proof.',
          },
      ],
      'storylets': [
          {
              'id': 's_kesh_tests_commission',
              'trigger': 'scene',
              'weight': 4,
              'once': True,
              'preconditions': [{'npc_present': 'Warden Kesh'}, {'act_is': 1}],
              'beat': (
                  "Kesh inspects the player's unsealed commission and notes "
                  'that an unsealed paper buys no favors here.'
              ),
              'effects': [{'tension': 1}],
              'voice': first_voice,
              'voice_line': 'He checked the seal before he checked your name.',
          },
          {
              'id': 's_ledger_inspected',
              'trigger': 'scene',
              'weight': 3,
              'once': False,
              'preconditions': [
                  {'location_is': 'The Lower Gatehouse'},
                  {'clue_undiscovered': 'c_ledger_entry'},
              ],
              'beat': (
                  'If the player reads the chained entry ledger closely, reveal'
                  ' c_ledger_entry.'
              ),
              'effects': [],
              'voice': second_voice,
              'voice_line': 'Midnight weights signed out, never delivered.',
          },
          {
              'id': 'a_commons_arrival',
              'trigger': 'arrival',
              'weight': 4,
              'once': True,
              'preconditions': [{'location_is': 'The Lantern Commons'}],
              'beat': 'Conversations drop when a stranger enters the commons.',
              'effects': [],
          },
          {
              'id': 'w_tamsin_moves_crates',
              'trigger': 'warm',
              'weight': 4,
              'once': True,
              'preconditions': [{'truth_undiscovered': 't_sabotage'}],
              'beat': (
                  'Off-screen, Broker Tamsin shifts crates under tarp toward '
                  'the service tunnel.'
              ),
              'effects': [{
                  'rumor': {
                      'text': (
                          'Broker Tamsin was seen moving heavy crates '
                          'after dusk.'
                      ),
                      'source': 'porters',
                      'reliability': 'distorted',
                  }
              }],
          },
          {
              'id': 'd_stage1_warning',
              'trigger': 'doom',
              'weight': 10,
              'once': True,
              'preconditions': [{'doom_stage_gte': 1}],
              'beat': 'The warning bells ring once across the gatehouse.',
              'effects': [{'tension': 1}],
          },
          {
              'id': 'r_kesh_recalls',
              'trigger': 'rehydrate',
              'weight': 5,
              'once': False,
              'preconditions': [{'npc_present': 'Warden Kesh'}],
              'beat': (
                  'Kesh opens by naming where the watch spotted the player '
                  'since they last spoke.'
              ),
              'effects': [],
          },
          {
              'id': 'e_relief_opened',
              'trigger': 'ending',
              'weight': 50,
              'once': True,
              'preconditions': [{'truth_discovered': 't_bypass'}],
              'beat': (
                  'With Folio 4 understood, the manual relief channel can be '
                  'turned before the lock seizes.'
              ),
              'effects': [{'end_run': 'success'}],
          },
      ],
  }


def repair_bible_dict(raw: dict[str, Any]) -> dict[str, Any]:
  """Repairs common LLM authoring omissions so `validate_bible` passes."""
  data: dict[str, Any] = dict(raw)
  factions: list[dict[str, Any]] = [
      dict(f) for f in data.get('factions', []) if isinstance(f, dict)
  ]
  data['factions'] = factions
  faction_ids = {str(f.get('id', '')) for f in factions if f.get('id')}
  for f in factions:
    standing = f.get('standing')
    if isinstance(standing, dict):
      f['standing'] = {
          k: int(v) for k, v in standing.items() if k in faction_ids
      }

  truths: list[dict[str, Any]] = [
      dict(t) for t in data.get('truths', []) if isinstance(t, dict)
  ]
  clues: list[dict[str, Any]] = [
      dict(c) for c in data.get('clues', []) if isinstance(c, dict)
  ]
  truth_ids = {str(t.get('id', '')) for t in truths if t.get('id')}
  if not truths:
    default_t: dict[str, Any] = {
        'id': 't_main',
        'statement': 'Something is rotten.',
        'clue_ids': [],
    }
    truths = [default_t]
    truth_ids = {'t_main'}
  data['truths'] = truths
  data['clues'] = clues

  default_truth_id = next(iter(truth_ids))
  clue_ids: set[str] = set()
  for c in clues:
    cid = str(c.get('id', '')).strip()
    if not cid:
      continue
    clue_ids.add(cid)
    if c.get('truth_id') not in truth_ids:
      c['truth_id'] = default_truth_id
    if c.get('carrier') not in story_state.CLUE_CARRIERS:
      c['carrier'] = 'location'

  # Ensure bidirectional truth <-> clue membership.
  listed: set[str] = set()
  for t in truths:
    t['clue_ids'] = [cid for cid in t.get('clue_ids', []) if cid in clue_ids]
    listed.update(t['clue_ids'])
  for c in clues:
    cid = str(c.get('id', ''))
    if cid and cid not in listed:
      owner = next(
          (t for t in truths if t.get('id') == c.get('truth_id')), truths[0]
      )
      owner.setdefault('clue_ids', []).append(cid)
  for t in truths:
    cids = t.get('clue_ids', [])
    raw_min = t.get('min_clues_to_discover', 0)
    min_c = int(raw_min) if isinstance(raw_min, (int, float, str)) else 0
    if min_c > len(cids):
      t['min_clues_to_discover'] = max(1, len(cids))

  for n in data.get('seed_npcs', []):
    if isinstance(n, dict):
      if n.get('faction') and n['faction'] not in faction_ids:
        n['faction'] = next(iter(faction_ids), '')
      if str(n.get('tier', 'cold')).lower() not in ('hot', 'warm', 'cold'):
        n['tier'] = 'cold'

  doom = data.get('doom')
  if isinstance(doom, dict):
    stages = doom.get('stages')
    if isinstance(stages, list) and stages and isinstance(stages[-1], dict):
      stages[-1]['at_fraction'] = 1.0

  voices = data.get('player_voices') or {}
  seen_sids: set[str] = set()
  clean_storylets = []
  for idx, s in enumerate(data.get('storylets', [])):
    if not isinstance(s, dict):
      continue
    sid = str(s.get('id') or f's_auto_{idx}')
    if sid in seen_sids:
      sid = f'{sid}_{idx}'
    seen_sids.add(sid)
    s['id'] = sid
    if s.get('trigger') not in story_state.TRIGGERS:
      s['trigger'] = 'scene'
    if not str(s.get('beat', '')).strip():
      s['beat'] = 'A tense beat passes in the room.'
    if s.get('voice') and voices and s['voice'] not in voices:
      s['voice'] = next(iter(voices))
    s['preconditions'] = [
        c
        for c in s.get('preconditions', [])
        if isinstance(c, dict)
        and len(c) == 1
        and next(iter(c)) in story_state.CONDITION_KEYS
    ]
    s['effects'] = [
        e
        for e in s.get('effects', [])
        if isinstance(e, dict)
        and len(e) == 1
        and next(iter(e)) in story_state.EFFECT_KEYS
    ]
    clean_storylets.append(s)
  data['storylets'] = clean_storylets
  return data


def _player_facing_fields(
    bible_dict: Mapping[str, Any],
) -> dict[str, tuple[str, str, int | None]]:
  """Maps a label to `(section, key, index)` for each player-facing text."""
  fields: dict[str, tuple[str, str, int | None]] = {
      'opening': ('opening', '', None)
  }
  for idx, truth in enumerate(bible_dict.get('truths', [])):
    if not isinstance(truth, Mapping):
      continue
    for key in ('question', 'stakes'):
      if truth.get(key):
        fields[f"truth[{truth.get('id', idx)}].{key}"] = ('truths', key, idx)
  for idx, storylet in enumerate(bible_dict.get('storylets', [])):
    if isinstance(storylet, Mapping) and storylet.get('voice_line'):
      fields[f"storylet[{storylet.get('id', idx)}].voice_line"] = (
          'storylets',
          'voice_line',
          idx,
      )
  return fields


def _read_field(
    bible_dict: Mapping[str, Any], location: tuple[str, str, int | None]
) -> str:
  section, key, idx = location
  if idx is None:
    return str(bible_dict.get(section, ''))
  return str(bible_dict[section][idx].get(key, ''))


def _write_field(
    bible_dict: dict[str, Any],
    location: tuple[str, str, int | None],
    value: str,
) -> None:
  section, key, idx = location
  if idx is None:
    bible_dict[section] = value
  else:
    bible_dict[section][idx][key] = value


def scrub_spoilers_with_model(
    bible_dict: Mapping[str, Any],
    *,
    model: language_model.LanguageModel,
) -> dict[str, Any]:
  """Rewrites player-facing texts that reveal a truth, using the model.

  Two detectors feed one rewrite call. The lexical detector in `story_state`
  catches texts that reuse a truth's private vocabulary; the model is asked,
  in addition, to flag paraphrased reveals (clone vs. duplicate) that share no
  tokens. Texts the model does not return are left unchanged. A second
  lexical pass confirms the rewrite; if it still leaks, the text is replaced
  with a neutral placeholder so validation fails loudly rather than shipping
  a spoiler.

  Args:
    bible_dict: Repaired bible dictionary (mutated copy is returned).
    model: Language model used for the audit and rewrite.

  Returns:
    New bible dictionary with scrubbed player-facing texts.
  """
  result = json.loads(json.dumps(bible_dict))
  bible = story_state.WorldBible.from_dict(result)
  fields = _player_facing_fields(result)
  lexical = story_state.find_bible_leaks(bible)
  flagged = {f['text'] for f in lexical if f['severity'] == 'leak'}
  truths_block = '\n'.join(f'- {t.id}: {t.statement}' for t in bible.truths)
  texts_block = '\n'.join(
      f'[{label}] {_read_field(result, loc)}' for label, loc in fields.items()
  )
  prompt = (
      'You are auditing an interactive mystery for spoilers.\n'
      'HIDDEN TRUTHS (the player must earn these; never state or paraphrase '
      'them):\n'
      f'{truths_block}\n\n'
      'PLAYER-FACING TEXTS shown before any truth is solved:\n'
      f'{texts_block}\n\n'
      'A text spoils a truth if a reader could infer the explanation (who, '
      'why, how, or the nature of the twist) rather than only the inciting '
      'facts and the open question. Already flagged by a lexical check: '
      f'{sorted(flagged) or "none"}.\n'
      'Return ONLY a JSON object mapping each label that spoils a truth to a '
      'rewritten version of the same length and register that keeps the '
      'inciting facts and the question but removes the explanation. Omit '
      'labels that are fine.'
  )
  raw_text = model.sample_text(prompt, max_tokens=2000, terminators=())
  rewrites = adventure_specific_components.extract_json_object(raw_text)
  if isinstance(rewrites, Mapping):
    for label, new_text in rewrites.items():
      loc = fields.get(str(label))
      if loc is not None and isinstance(new_text, str) and new_text.strip():
        _write_field(result, loc, new_text.strip())
  bible_after = story_state.WorldBible.from_dict(result)
  for finding in story_state.find_bible_leaks(bible_after):
    if finding['severity'] != 'leak':
      continue
    loc = fields.get(finding['text'])
    if loc is None:
      continue
    logging.warning(
        'Spoiler scrub could not clean %s (still restates %s); blanking it.',
        finding['text'],
        finding['truth_id'],
    )
    _write_field(result, loc, '')
  return result


def draft_world_bible(
    *,
    title: str,
    premise: str,
    tone_name: str = 'noir',
    model: language_model.LanguageModel | None = None,
) -> story_state.WorldBible:
  """Drafts and validates a `WorldBible` from a premise (with optional LLM)."""
  genre_mode = _resolve_genre_mode(tone_name, title, premise)
  template = build_template_bible_dict(
      title=title, premise=premise, tone_name=tone_name
  )
  if model is not None:
    prompt = (
        'You are a lead narrative designer authoring a WorldBible JSON for an '
        'interactive RPG engine.\n'
        f"World Title: {template['title']}\n"
        f"World Premise: {template['premise']}\n"
        f'Tone / Genre: {tone_name} (resolved archetype: {genre_mode})\n\n'
        'Return ONLY a single valid JSON object matching the exact top-level '
        'keys and schema of this reference template, tailored to the premise '
        'with bespoke player_voices, player_sub_minds, character_presets, '
        'factions, hidden truths (with arc_kind, stakes, and a player-safe '
        'question), clues (with catalyst_kind and carrier), seed NPCs (with '
        'secret, wound, routine), doom stages, and storylets.\n'
        'Spoiler rule: `opening`, every truth `question` and `stakes`, and '
        'every storylet `voice_line` are shown to the player BEFORE the '
        'truths are solved. They must pose the mystery (what happened, who, '
        'why) without stating or paraphrasing any truth `statement`. Name the '
        'inciting facts the player can see; never the explanation.\n'
        f'{json.dumps(template, indent=2)}'
    )
    raw_text = model.sample_text(prompt, max_tokens=4000, terminators=())
    parsed = adventure_specific_components.extract_json_object(raw_text)
    if isinstance(parsed, Mapping) and parsed.get('premise'):
      candidate = dict(template)
      candidate.update(parsed)
      base_tone = dict(TONE_PRESETS.get(genre_mode, TONE_PRESETS['noir']))
      if isinstance(parsed.get('tone'), Mapping):
        for k, v in parsed['tone'].items():
          if v is not None and v:
            base_tone[k] = v
      candidate['tone'] = base_tone
      repaired = repair_bible_dict(candidate)
      repaired = scrub_spoilers_with_model(repaired, model=model)
      bible = story_state.WorldBible.from_dict(repaired)
      if not story_state.validate_bible(bible):
        return bible

  repaired = repair_bible_dict(template)
  bible = story_state.WorldBible.from_dict(repaired)
  problems = story_state.validate_bible(bible)
  if problems:
    raise ValueError('Template bible validation failed: ' + '; '.join(problems))
  return bible


def write_world_bible(bible: story_state.WorldBible, output_path: str) -> str:
  """Writes a validated `WorldBible` to `output_path` as formatted JSON."""
  problems = story_state.validate_bible(bible)
  if problems:
    raise ValueError('Cannot write invalid bible: ' + '; '.join(problems))
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  starting_loc_dict = None
  if bible.starting_location is not None:
    starting_loc_dict = {
        'name': bible.starting_location.name,
        'description': bible.starting_location.description,
        'affordances': list(bible.starting_location.affordances),
        'exits': dict(bible.starting_location.exits),
        'clue_ids': list(bible.starting_location.clue_ids),
    }
  seed_npc_dicts: list[dict[str, Any]] = []
  for n in bible.seed_npcs:
    npc_dict: dict[str, Any] = {
        'name': n.name,
        'role_summary': n.role_summary,
        'goal': n.goal,
        'traits': n.traits,
        'location_name': n.location_name,
        'tier': n.tier,
        'faction': n.faction,
        'secret': n.secret,
        'wound': n.wound,
        'formative_memories': list(n.formative_memories),
        'routine': [r.to_dict() for r in n.routine],
        'relationships': dict(n.relationships),
    }
    if getattr(n, 'sub_minds', None):
      npc_dict['sub_minds'] = {
          k: dict(v) if isinstance(v, Mapping) else v.to_dict()
          for k, v in n.sub_minds.items()
      }
    seed_npc_dicts.append(npc_dict)
  payload: dict[str, Any] = {
      'title': bible.title,
      'premise': bible.premise,
      'opening': bible.opening,
      'tone': bible.tone.to_dict(),
      'player_voices': dict(bible.player_voices),
      'starting_inventory': list(bible.starting_inventory),
      'starting_location': starting_loc_dict,
      'factions': [f.to_dict() for f in bible.factions],
      'truths': [t.to_dict() for t in bible.truths],
      'clues': [c.to_dict() for c in bible.clues],
      'seed_npcs': seed_npc_dicts,
      'doom': bible.doom.to_dict(),
      'acts': [a.to_dict() for a in bible.acts],
      'storylets': [s.to_dict() for s in bible.storylets],
  }
  if bible.player_sub_minds:
    payload['player_sub_minds'] = {
        k: dict(v) if isinstance(v, Mapping) else v.to_dict()
        for k, v in bible.player_sub_minds.items()
    }
  if bible.character_presets:
    payload['character_presets'] = list(bible.character_presets)
  if bible.emergent_voice_pool:
    payload['emergent_voice_pool'] = [
        dict(v) if isinstance(v, Mapping) else v.to_dict()
        for v in bible.emergent_voice_pool
    ]
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
  return output_path


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  bible = draft_world_bible(
      title=flags.FLAGS.title,
      premise=flags.FLAGS.premise,
      tone_name=flags.FLAGS.tone,
  )
  out = write_world_bible(bible, flags.FLAGS.output)
  logging.info('Wrote validated WorldBible %r to %s', bible.title, out)


if __name__ == '__main__':
  _define_flags()
  app.run(main)
