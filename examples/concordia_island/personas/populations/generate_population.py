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

r"""Step 1 of persona generation: build a population roster.

See `personas/README.md` for the whole pipeline. This script writes a roster
JSON (`{"metadata": ..., "agents": [...]}`); `personas/generate_personas.py`
then turns the roster into full personas (traits, worldview, formative
memories) that `run.py --personas_date=<label>` loads.

Produces one entry per agent with demographic metadata:
  - Name, age (21-75+), gender, sexual orientation, relationship status
  - Economic class (5-tier), occupation sector, home_place, work_place
  - Personality one-liner, backstory paragraph
  - Cultural/ethnic background

Demographic constraints enforced:
  - ~10% homosexual orientation (date same sex)
  - Some agents paired as couples (same household)
  - Age Gaussian per economic tier (lower=younger mean, elite=older)
  - Gender roughly 50/50 with small non-binary fraction

Demographics and census names are sampled with a seeded RNG; the LLM writes
only the one-line personality and one-sentence backstory. If the LLM output
for a batch cannot be parsed after `--max_retries` attempts the script stops
with an error rather than writing placeholder text.

Usage:
  python -m examples.concordia_island.personas.populations.generate_population \\
    --setting=brecksville \\
    --api_type=google_aistudio \\
    --model_name=gemini-2.5-flash \\
    --num_agents=1000 \\
    --output_path=/tmp/brecksville_roster.json

  # Dry run: demographics and names only, no LLM calls. The roster has no
  # personality/backstory fields and is marked "dry_run" in its metadata.
  python -m examples.concordia_island.personas.populations.generate_population \\
    --dry_run --num_agents=50 --output_path=/tmp/roster_dry.json
"""

import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List

from absl import app
from absl import flags
from concordia.document import interactive_document
from concordia.language_model import language_model

from examples.concordia_island.personas import census_names

FLAGS = flags.FLAGS


def _define_flags() -> None:
  """Defines the command-line flags (only when run as a script).

  Flags live here rather than at import time so that
  `personas/generate_personas.py` can import this module's helpers without
  inheriting its flags.
  """
  flags.DEFINE_enum(
      'setting',
      'brecksville',
      ['brecksville', 'ohio_suburb', 'island', 'kerala', 'lagos'],
      'Population setting; picks location context, home prefixes, occupations'
      ' and ethnicities. brecksville is an alias for ohio_suburb (the bundled'
      ' Brecksville, Ohio population). island is Concordia Island.',
  )
  flags.DEFINE_string(
      'island_name',
      '',
      'Community name used in prompts. Empty uses the setting default (e.g.'
      ' "Brecksville, Ohio").',
  )
  flags.DEFINE_integer(
      'num_agents', 1000, 'Total number of agents to generate.'
  )
  flags.DEFINE_string(
      'output_path',
      '/tmp/population_roster.json',
      'Path to save the population roster JSON.',
  )
  flags.DEFINE_bool(
      'dry_run',
      False,
      'If true, assign demographics and names only (no LLM calls, no'
      ' personality or backstory).',
  )
  flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
  flags.DEFINE_integer(
      'max_retries', 3, 'LLM attempts per batch before the script fails.'
  )
  flags.DEFINE_string(
      'api_type',
      'google_aistudio',
      'Concordia language_model_setup api_type (e.g. google_aistudio,'
      ' openai, vllm, ollama, huggingface).',
  )
  flags.DEFINE_string(
      'model_name',
      'gemini-2.5-flash',
      'Model name passed to language_model_setup.',
  )
  flags.DEFINE_string(
      'api_key',
      None,
      'API key for the provider (falls back to GOOGLE_API_KEY, GEMINI_API_KEY'
      ' or OPENAI_API_KEY).',
  )


# === Demographic Constants ===

ECONOMIC_TIERS = {
    'lower_middle': {
        'prefix': 'sunset_apartments',
        'fraction': 0.50,
        'age_mu': 35,
        'age_sigma': 15,
        'occupations': [
            'cafe',
            'general_store',
            'restaurant',
            'fishing_dock',
            'marina',
            'market',
        ],
    },
    'middle': {
        'prefix': 'coral_village',
        'fraction': 0.25,
        'age_mu': 38,
        'age_sigma': 12,
        'occupations': [
            'school',
            'office_floor_finance',
            'office_floor_tech',
            'office_floor_creative',
            'library',
            'medical_clinic',
            'town_hall',
            'restaurant',
        ],
    },
    'upper_middle': {
        'prefix': 'palm_heights',
        'fraction': 0.15,
        'age_mu': 45,
        'age_sigma': 12,
        'occupations': [
            'office_floor_tech',
            'office_floor_finance',
            'school',
            'medical_clinic',
            'office_building',
            'marina',
        ],
    },
    'upper': {
        'prefix': 'ocean_view_estates',
        'fraction': 0.07,
        'age_mu': 50,
        'age_sigma': 12,
        'occupations': [
            'office_building',
            'medical_clinic',
            'town_hall',
            'office_floor_finance',
        ],
    },
    'elite': {
        'prefix': 'paradise_point',
        'fraction': 0.03,
        'age_mu': 50,
        'age_sigma': 12,
        'occupations': [
            'office_building',
            'town_hall',
        ],
    },
}

GENDERS = ['man', 'woman', 'non-binary']
GENDER_WEIGHTS = [0.495, 0.495, 0.01]

SEXUAL_ORIENTATIONS = ['heterosexual', 'homosexual', 'bisexual']
ORIENTATION_WEIGHTS = [0.85, 0.10, 0.05]

ETHNICITIES = [
    'White American',
    'Black American',
    'Latino/Hispanic',
    'East Asian American',
    'South Asian American',
    'Middle Eastern American',
    'Pacific Islander',
    'Native American',
    'Mixed/Multiracial',
]

RELATIONSHIP_STATUSES = ['single', 'married']

POLITICAL_ORIENTATIONS = [
    'progressive',
    'liberal',
    'moderate',
    'conservative',
    'libertarian',
    'socialist',
    'apolitical',
]
POLITICAL_WEIGHTS = [0.15, 0.20, 0.25, 0.20, 0.08, 0.05, 0.07]

HOBBIES = [
    'cooking and food',
    'fitness and sports',
    'reading and writing',
    'music and arts',
    'gardening and nature',
    'gaming and technology',
    'crafts and DIY',
    'travel and exploration',
    'community volunteering',
    'fishing and outdoors',
    'collecting and antiques',
    'photography',
    'social media and content creation',
    'religious activities',
    'board games and puzzles',
    'dancing',
    'meditation and yoga',
]

LOCATION_CONTEXT = (
    '{location_name} is a lush, self-contained island community with a '
    'population of roughly one thousand residents. The island has a diverse '
    'economy ranging from fishing and hospitality to remote tech work and '
    'professional services. Neighborhoods range from modest apartment '
    'complexes near the harbor to grand estates overlooking the bay. '
    'Residents know each other and interact regularly at shared spaces '
    'like the market, cafe, library, tavern, and beach. The community '
    'values self-sufficiency but is connected to the mainland via ferry '
    'and internet. Cultural diversity is a hallmark, with residents '
    'from many different backgrounds and walks of life.'
)

# === Setting-specific overrides ===

SETTING_OVERRIDES = {
    'ohio_suburb': {
        'default_name': 'Brecksville, Ohio',
        'location_context': (
            '{location_name} is a suburban community of roughly one thousand '
            'residents outside Cleveland, Ohio. The neighborhood has a mix of '
            'apartment complexes, subdivisions, and a few upscale estates near '
            'the lake. The local economy includes retail, healthcare, tech, '
            'and professional services, with many residents commuting to '
            'Cleveland for work. Residents interact at the coffee shop, '
            'library, church, shopping plaza, and community recreation center. '
            'The area has a strong sense of community with seasonal events, '
            'youth sports leagues, and neighborhood cookouts.'
        ),
        'ethnicities': [
            'White American',
            'Black American',
            'Latino/Hispanic',
            'East Asian American',
            'South Asian American',
            'Mixed/Multiracial',
        ],
        'economic_tiers': {
            'lower_middle': {
                'prefix': 'millbrook_apts',
                'fraction': 0.50,
                'age_mu': 35,
                'age_sigma': 15,
                'occupations': [
                    'cafe',
                    'general_store',
                    'restaurant',
                    'shopping_plaza',
                    'market',
                    'school',
                ],
            },
            'middle': {
                'prefix': 'brecksville_commons',
                'fraction': 0.25,
                'age_mu': 38,
                'age_sigma': 12,
                'occupations': [
                    'school',
                    'office_floor_finance',
                    'office_floor_tech',
                    'office_floor_creative',
                    'library',
                    'medical_clinic',
                    'community_center',
                    'restaurant',
                ],
            },
            'upper_middle': {
                'prefix': 'chippewa_ridge',
                'fraction': 0.15,
                'age_mu': 45,
                'age_sigma': 12,
                'occupations': [
                    'office_floor_tech',
                    'office_floor_finance',
                    'school',
                    'medical_clinic',
                    'office_building',
                    'community_center',
                ],
            },
            'upper': {
                'prefix': 'riverview_estates',
                'fraction': 0.07,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'medical_clinic',
                    'community_center',
                    'office_floor_finance',
                ],
            },
            'elite': {
                'prefix': 'timber_creek',
                'fraction': 0.03,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'community_center',
                ],
            },
        },
    },
    'kerala': {
        'default_name': 'Alappuzha, Kerala',
        'location_context': (
            '{location_name} is a small coastal town in Kerala, India with a '
            'population of roughly one thousand residents. The town has a '
            'diverse economy including IT services, healthcare, education, '
            'small retail, and government jobs. Housing ranges from modest '
            'chawl blocks to comfortable residential colonies to upscale '
            'villas. Residents interact at the tea shop, market, temple, '
            'mosque, church, library, and community hall. The town is '
            'connected to nearby cities by bus and train.'
        ),
        'ethnicities': [
            'Malayali Hindu',
            'Malayali Christian',
            'Malayali Muslim',
            'Tamil',
            'Konkani',
            'Anglo-Indian',
        ],
        'economic_tiers': {
            'lower_middle': {
                'prefix': 'canal_row_flats',
                'fraction': 0.50,
                'age_mu': 35,
                'age_sigma': 15,
                'occupations': [
                    'tea_shop',
                    'general_store',
                    'restaurant',
                    'bus_stand',
                    'market',
                    'school',
                ],
            },
            'middle': {
                'prefix': 'thottam_colony',
                'fraction': 0.25,
                'age_mu': 38,
                'age_sigma': 12,
                'occupations': [
                    'school',
                    'office_floor_finance',
                    'office_floor_tech',
                    'office_floor_creative',
                    'library',
                    'medical_clinic',
                    'community_center',
                    'restaurant',
                ],
            },
            'upper_middle': {
                'prefix': 'paddy_view_villas',
                'fraction': 0.15,
                'age_mu': 45,
                'age_sigma': 12,
                'occupations': [
                    'office_floor_tech',
                    'office_floor_finance',
                    'school',
                    'medical_clinic',
                    'office_building',
                    'temple',
                ],
            },
            'upper': {
                'prefix': 'backwater_estates',
                'fraction': 0.07,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'medical_clinic',
                    'community_center',
                    'office_floor_finance',
                ],
            },
            'elite': {
                'prefix': 'coconut_grove',
                'fraction': 0.03,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'community_center',
                ],
            },
        },
    },
    'lagos': {
        'default_name': 'Surulere, Lagos',
        'location_context': (
            '{location_name} is a bustling neighborhood in Lagos, Nigeria with '
            'a population of roughly one thousand residents. The area has a '
            'vibrant economy of small businesses, tech startups, professional '
            'services, and street commerce. Housing ranges from crowded '
            'face-me-I-face-you tenements to gated estate housing to '
            'luxury flats. Residents interact at the market, mama put stalls, '
            'beer parlours, church, mosque, and community center. The '
            'neighborhood is connected to the rest of Lagos by danfo buses '
            'and okada motorcycles.'
        ),
        'ethnicities': [
            'Yoruba',
            'Igbo',
            'Hausa',
            'Edo',
            'Ijaw',
            'Mixed Nigerian',
            'Ghanaian Nigerian',
        ],
        'economic_tiers': {
            'lower_middle': {
                'prefix': 'eko_flats',
                'fraction': 0.50,
                'age_mu': 35,
                'age_sigma': 15,
                'occupations': [
                    'cafe',
                    'general_store',
                    'restaurant',
                    'bus_stand',
                    'market',
                    'school',
                ],
            },
            'middle': {
                'prefix': 'surulere_courts',
                'fraction': 0.25,
                'age_mu': 38,
                'age_sigma': 12,
                'occupations': [
                    'school',
                    'office_floor_finance',
                    'office_floor_tech',
                    'office_floor_creative',
                    'library',
                    'medical_clinic',
                    'community_center',
                    'restaurant',
                ],
            },
            'upper_middle': {
                'prefix': 'gbagada_heights',
                'fraction': 0.15,
                'age_mu': 45,
                'age_sigma': 12,
                'occupations': [
                    'office_floor_tech',
                    'office_floor_finance',
                    'school',
                    'medical_clinic',
                    'office_building',
                    'church',
                ],
            },
            'upper': {
                'prefix': 'ikoyi_estates',
                'fraction': 0.07,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'medical_clinic',
                    'community_center',
                    'office_floor_finance',
                ],
            },
            'elite': {
                'prefix': 'banana_island',
                'fraction': 0.03,
                'age_mu': 50,
                'age_sigma': 12,
                'occupations': [
                    'office_building',
                    'community_center',
                ],
            },
        },
    },
}


# Settings without an entry in SETTING_OVERRIDES use the module defaults.
_SETTING_ALIASES = {'brecksville': 'ohio_suburb', 'island': ''}
DEFAULT_COMMUNITY_NAME = 'Concordia Island'


def canonical_setting(setting: str) -> str:
  """Maps a --setting value to its SETTING_OVERRIDES key ('' = island)."""
  setting = _SETTING_ALIASES.get(setting, setting)
  if setting and setting not in SETTING_OVERRIDES:
    raise ValueError(
        f'Unknown setting {setting!r}; expected one of'
        f' {sorted(set(SETTING_OVERRIDES) | set(_SETTING_ALIASES))}.'
    )
  return setting


def get_setting_config(setting: str) -> dict[str, Any]:
  """Get demographic overrides for a setting."""
  return SETTING_OVERRIDES.get(canonical_setting(setting), {})


def community_name_for(setting: str, override: str = '') -> str:
  """Returns the community name for prompts and persona context."""
  if override:
    return override
  return get_setting_config(setting).get(
      'default_name', DEFAULT_COMMUNITY_NAME
  )


# sim/locations.py preset whose buildings and public places match each
# setting's home prefixes and work places. run.py reads it from the persona
# set's metadata.json ('setting_preset').
_SIM_SETTING_PRESETS = {
    'ohio_suburb': 'brecksville_1000',
    '': 'island',
    'kerala': 'kerala',
    'lagos': 'lagos',
}


def sim_setting_preset(setting: str) -> str:
  """Returns the sim/locations.py preset name for a --setting value."""
  return _SIM_SETTING_PRESETS[canonical_setting(setting)]


def _assign_demographics(
    rng: random.Random,
    num_agents: int,
    setting: str = '',
) -> List[Dict[str, Any]]:
  """Assign demographic metadata deterministically using seeded RNG."""
  agents = []

  setting = canonical_setting(setting)
  setting_config = get_setting_config(setting)
  tiers_config = setting_config.get('economic_tiers', ECONOMIC_TIERS)
  ethnicities = setting_config.get('ethnicities', ETHNICITIES)

  # Neighborhood assignment for ohio_suburb.  Each economic tier maps to a
  # subset of the 10 Brecksville neighborhoods; agents are round-robin
  # distributed within that subset so every neighborhood gets residents.
  tier_neighborhoods = {
      'lower_middle': [
          'snowville',
          'barr_road',
          'oakes',
          'stadium',
      ],
      'middle': [
          'whitewood',
          'fitzwater',
          'parkside',
          'snowville',
      ],
      'upper_middle': [
          'chippewa',
          'highland',
          'parkside',
      ],
      'upper': [
          'riverview',
          'highland',
          'chippewa',
      ],
      'elite': [
          'riverview',
          'highland',
      ],
  }
  nbr_counters = {tier: 0 for tier in tier_neighborhoods}

  # Compute tier sizes
  tier_sizes = {}
  remaining = num_agents
  tiers = list(tiers_config.keys())
  for i, tier in enumerate(tiers):
    if i == len(tiers) - 1:
      tier_sizes[tier] = remaining
    else:
      size = round(num_agents * tiers_config[tier]['fraction'])
      tier_sizes[tier] = size
      remaining -= size

  unit_counters = {tier: 0 for tier in tiers}

  # Pre-generate relationship statuses to ensure exact 50/50 split
  statuses = ['single'] * (num_agents // 2) + ['married'] * (
      num_agents - num_agents // 2
  )
  rng.shuffle(statuses)
  status_idx = 0

  for tier in tiers:
    tier_info = tiers_config[tier]
    for _ in range(tier_sizes[tier]):
      unit_counters[tier] += 1
      unit_num = unit_counters[tier]

      age = max(21, int(rng.gauss(tier_info['age_mu'], tier_info['age_sigma'])))
      age = min(age, 85)

      gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
      orientation = rng.choices(
          SEXUAL_ORIENTATIONS, weights=ORIENTATION_WEIGHTS, k=1
      )[0]
      ethnicity = rng.choice(ethnicities)
      occupation = rng.choice(tier_info['occupations'])
      political = rng.choices(
          POLITICAL_ORIENTATIONS, weights=POLITICAL_WEIGHTS, k=1
      )[0]
      hobbies = rng.sample(HOBBIES, k=rng.randint(1, 3))

      relationship_status = statuses[status_idx]
      status_idx += 1

      # Assign neighborhood (ohio_suburb only for now)
      neighborhood = ''
      if setting == 'ohio_suburb' and tier in tier_neighborhoods:
        tier_nbrs = tier_neighborhoods[tier]
        neighborhood = tier_nbrs[nbr_counters[tier] % len(tier_nbrs)]
        nbr_counters[tier] += 1

      agents.append({
          'economic_class': tier,
          'home_place': f'{tier_info["prefix"]}_unit_{unit_num}',
          'work_place': occupation,
          'age': age,
          'gender': gender,
          'sexual_orientation': orientation,
          'ethnicity': ethnicity,
          'relationship_status': relationship_status,
          'political_orientation': political,
          'hobbies': hobbies,
          'unit_num': unit_num,
          'neighborhood': neighborhood,
      })

  return agents


def _pair_couples(
    agents: List[Dict[str, Any]],
    rng: random.Random,
) -> None:
  """Pair some agents as couples in-place.

  Respects sexual orientation: homosexual agents pair with same gender,
  heterosexual with opposite, bisexual with either.

  Args:
    agents: List of agent dictionaries to pair.
    rng: Seeded random number generator.
  """
  available = [
      i
      for i, a in enumerate(agents)
      if a['relationship_status'] in ('in a relationship', 'married')
  ]
  rng.shuffle(available)

  paired = set()
  for idx in available:
    if idx in paired:
      continue
    agent = agents[idx]

    # Find a compatible partner
    for partner_idx in available:
      if partner_idx in paired or partner_idx == idx:
        continue
      partner = agents[partner_idx]

      # Check orientation compatibility
      if _is_compatible(agent, partner):
        partner_name = partner.get('name', f'Agent_{partner_idx}')
        agent_name = agent.get('name', f'Agent_{idx}')
        agent['relationship_status'] = f'married to {partner_name}'
        partner['relationship_status'] = f'married to {agent_name}'
        agent['partner_idx'] = partner_idx
        partner['partner_idx'] = idx
        # Share the same household (cohabitation)
        partner['home_place'] = agent['home_place']
        partner['neighborhood'] = agent.get('neighborhood', '')
        paired.add(idx)
        paired.add(partner_idx)
        break


def _is_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
  """Check if two agents are romantically compatible by orientation."""

  def _attracted_to(agent: Dict[str, Any], target_gender: str) -> bool:
    orientation = agent['sexual_orientation']
    own_gender = agent['gender']
    if orientation == 'bisexual':
      return True
    if orientation == 'homosexual':
      return target_gender == own_gender
    # heterosexual
    return target_gender != own_gender

  return _attracted_to(a, b['gender']) and _attracted_to(b, a['gender'])


def _fix_backstory_name(
    backstory: str,
    census_name: str,
    gender: str,
) -> str:
  """Replace any LLM-invented name in the backstory with the census name.

  The LLM often ignores the instruction to use the assigned name and
  generates its own.  This function detects the LLM name (typically the
  first capitalized word) and replaces all occurrences with the correct
  census first name.  It also fixes pronoun mismatches.

  Args:
    backstory: The backstory text to fix.
    census_name: The correct census-assigned full name.
    gender: The agent's gender (e.g. 'man', 'woman').

  Returns:
    The backstory with corrected names and pronouns.
  """
  if not backstory or not census_name:
    return backstory

  census_first = census_name.split()[0]

  # Find the first capitalized word in the backstory — usually the LLM's
  # invented first name.  Skip common sentence starters.
  skip_words = {
      'A',
      'An',
      'The',
      'After',
      'Before',
      'When',
      'Once',
      'Having',
      'With',
      'As',
      'In',
      'On',
      'At',
      'For',
      'Born',
      'Growing',
      'Recently',
      'Originally',
      'Following',
      'During',
      'Since',
  }
  words = backstory.split()
  llm_first = None
  for w in words:
    # Strip trailing punctuation for matching
    clean = re.sub(r'[^A-Za-z]', '', w)
    if clean and clean[0].isupper() and clean not in skip_words:
      llm_first = clean
      break

  if llm_first and llm_first != census_first:
    # Replace all occurrences of the LLM name with the census name
    backstory = re.sub(
        r'\b' + re.escape(llm_first) + r'\b', census_first, backstory
    )

  # Fix pronoun mismatches based on stated gender
  if gender == 'man':
    backstory = backstory.replace(' she ', ' he ')
    backstory = backstory.replace(' her ', ' his ')
    backstory = backstory.replace(' herself ', ' himself ')
    backstory = backstory.replace('She ', 'He ')
    backstory = backstory.replace('Her ', 'His ')
  elif gender == 'woman':
    backstory = backstory.replace(' he ', ' she ')
    backstory = backstory.replace(' him ', ' her ')
    backstory = backstory.replace(' himself ', ' herself ')
    backstory = backstory.replace('He ', 'She ')
    backstory = backstory.replace('His ', 'Her ')

  return backstory


def _generate_names_and_details_batch(
    model: language_model.LanguageModel,
    agents_batch: List[Dict[str, Any]],
    island_name: str,
    batch_idx: int,
    extra_instructions: str = '',
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
  """Asks the LLM for a personality and backstory for each agent in a batch.

  Args:
    model: Language model.
    agents_batch: Agents with demographics and census names; updated in place.
    island_name: Community name used in the prompt.
    batch_idx: Batch index, used in log and error messages.
    extra_instructions: Optional text appended to the question.
    max_retries: Attempts before giving up.

  Returns:
    `agents_batch`, with `personality` and `backstory` set on every agent.

  Raises:
    RuntimeError: If no attempt returns one valid entry per agent.
  """
  last_error = ''
  for attempt in range(1, max_retries + 1):
    try:
      results = _request_details(
          model, agents_batch, island_name, extra_instructions
      )
    except ValueError as e:
      last_error = str(e)
      logging.warning(
          'Batch %d attempt %d/%d: %s', batch_idx, attempt, max_retries, e
      )
      continue
    for agent, result in zip(agents_batch, results):
      agent['personality'] = result['personality']
      # Replace any LLM-invented name with the census name.
      agent['backstory'] = _fix_backstory_name(
          result['backstory'], agent['name'], agent.get('gender', '')
      )
    return agents_batch
  raise RuntimeError(
      f'Batch {batch_idx}: no valid personality/backstory output after'
      f' {max_retries} attempts. Last error: {last_error}'
  )


def _request_details(
    model: language_model.LanguageModel,
    agents_batch: List[Dict[str, Any]],
    island_name: str,
    extra_instructions: str = '',
) -> List[Dict[str, str]]:
  """Runs one LLM call for a batch and returns the validated results.

  Args:
    model: Language model.
    agents_batch: Agents with demographics and census names.
    island_name: Community name used in the prompt.
    extra_instructions: Optional text appended to the question.

  Returns:
    One dict per agent with "personality" and "backstory" strings.

  Raises:
    ValueError: If the output is not a JSON array with one object per agent,
      each with non-empty "personality" and "backstory" strings.
  """
  prompt = interactive_document.InteractiveDocument(model)
  prompt.statement(
      'You are creating characters for a social simulation set in '
      f'{island_name}, a diverse community. '
      'Generate realistic, diverse names and brief character descriptions.'
  )

  demographics_text = []
  for i, agent in enumerate(agents_batch):
    name_line = ''
    if agent.get('name'):
      name_line = f'  Name (MUST USE EXACTLY): {agent["name"]}\n'
    demographics_text.append(
        f'Character {i+1}:\n'
        f'{name_line}'
        f'  Age: {agent["age"]}\n'
        f'  Gender: {agent["gender"]}\n'
        f'  Ethnicity: {agent["ethnicity"]}\n'
        f'  Economic class: {agent["economic_class"]}\n'
        f'  Occupation sector: {agent["work_place"]}\n'
        f'  Sexual orientation: {agent["sexual_orientation"]}\n'
        f'  Relationship status: {agent["relationship_status"]}'
    )

  prompt.statement(
      'Here are the demographic slots to fill:\n\n'
      + '\n\n'.join(demographics_text)
  )

  question = (
      f'For each of the {len(agents_batch)} characters above, generate:\n'
      '1. A one-line personality description (e.g. "Hardworking and '
      'cheerful, always has a smile")\n'
      '2. A one-sentence backstory relevant to their occupation and '
      'background\n\n'
      'CRITICAL: Each character already has a Name assigned above. '
      'You MUST use that EXACT name in the backstory — do NOT invent, '
      'change, or substitute a different name. '
      'Use correct pronouns matching their stated gender.\n\n'
      'Output as a JSON array of objects with keys: '
      '"personality", "backstory"\n'
      'Output ONLY the JSON array, no markdown formatting.'
  )
  if extra_instructions:
    question += f'\n\nAdditional Instructions:\n{extra_instructions}'

  raw = prompt.open_question(
      question,
      max_tokens=4000,
      terminators=[],
      temperature=1.0,
  ).strip()

  # Strip markdown code fences and control characters.
  if raw.startswith('```'):
    raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
  if raw.endswith('```'):
    raw = raw[:-3]
  raw = re.sub(r'[\x00-\x1f]', '', raw.strip())

  try:
    results = json.loads(raw)
  except json.JSONDecodeError as e:
    raise ValueError(f'JSON parse error: {e}') from e
  if not isinstance(results, list) or len(results) != len(agents_batch):
    got = len(results) if isinstance(results, list) else type(results).__name__
    raise ValueError(f'expected {len(agents_batch)} results, got {got}')
  for k, result in enumerate(results):
    for key in ('personality', 'backstory'):
      value = result.get(key) if isinstance(result, dict) else None
      if not isinstance(value, str) or not value.strip():
        raise ValueError(f'result {k} has no {key!r}')
  return results


def generate_population(
    model: language_model.LanguageModel | None,
    num_agents: int = 1000,
    island_name: str = '',
    seed: int = 42,
    batch_size: int = 20,
    setting: str = 'brecksville',
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
  """Generates a population roster: demographics, names and LLM details.

  Args:
    model: Language model for personalities and backstories. None skips the
      LLM step (dry run): agents get demographics and names only.
    num_agents: Total agents to create.
    island_name: Community name for prompts; empty uses the setting default.
    seed: Random seed for demographics, names and couple pairing.
    batch_size: How many agents to send to the LLM per call.
    setting: --setting value (see `canonical_setting`).
    max_retries: LLM attempts per batch before failing.

  Returns:
    List of agent dictionaries.

  Raises:
    RuntimeError: If an LLM batch fails `max_retries` times.
    ValueError: If two agents end up with the same name.
  """
  setting = canonical_setting(setting)
  island_name = community_name_for(setting, island_name)
  rng = random.Random(seed)

  logging.info('Assigning demographics for %d agents...', num_agents)
  agents = _assign_demographics(rng, num_agents, setting=setting)

  # Names come from census data (deterministic, no LLM needed).
  logging.info('Assigning names from census data...')
  name_sampler = census_names.CensusNameSampler(seed=seed)
  name_sampler.sample_population(agents)

  if model is not None:
    total_batches = (len(agents) + batch_size - 1) // batch_size
    logging.info(
        'Generating personalities in %d batches of %d...',
        total_batches,
        batch_size,
    )
    for batch_start in range(0, len(agents), batch_size):
      batch_end = min(batch_start + batch_size, len(agents))
      batch_idx = batch_start // batch_size
      pct = int(100 * (batch_idx + 1) / total_batches)
      print(
          f'\r  [{"#" * (pct // 2)}{"·" * (50 - pct // 2)}] '
          f'Batch {batch_idx + 1}/{total_batches}  '
          f'({batch_start + 1}-{batch_end} of {len(agents)})  {pct}%',
          end='',
          flush=True,
      )
      _generate_names_and_details_batch(
          model,
          agents[batch_start:batch_end],
          island_name,
          batch_idx,
          max_retries=max_retries,
      )
    print(flush=True)  # newline after progress bar

  # Pair couples (now that names are assigned).
  _pair_couples(agents, rng)

  names = [a['name'] for a in agents]
  duplicates = sorted({n for n in names if names.count(n) > 1})
  if duplicates:
    raise ValueError(
        f'{len(duplicates)} duplicate names in the roster (seed={seed}):'
        f' {duplicates[:10]}'
    )
  return agents


def build_roster(
    agents: List[Dict[str, Any]],
    setting: str,
    island_name: str = '',
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
  """Wraps agents in the roster format read by generate_personas.py."""
  setting = canonical_setting(setting)
  setting_config = get_setting_config(setting)
  community_name = community_name_for(setting, island_name)
  context = setting_config.get('location_context', LOCATION_CONTEXT).format(
      location_name=community_name
  )
  tiers_config = setting_config.get('economic_tiers', ECONOMIC_TIERS)
  return {
      'metadata': {
          'location_name': community_name,
          'setting': setting or 'island',
          'setting_preset': sim_setting_preset(setting),
          'context': context,
          'num_agents': len(agents),
          'seed': seed,
          'dry_run': dry_run,
          'economic_tiers': {
              tier: {
                  'count': sum(
                      1 for a in agents if a['economic_class'] == tier
                  ),
              }
              for tier in tiers_config
          },
          'orientation_distribution': {
              orient: sum(
                  1 for a in agents if a.get('sexual_orientation') == orient
              )
              for orient in SEXUAL_ORIENTATIONS
          },
          'gender_distribution': {
              g: sum(1 for a in agents if a.get('gender') == g) for g in GENDERS
          },
      },
      'agents': agents,
  }


def main(_):
  """Entry point."""
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(message)s',
      stream=sys.stderr,
  )
  if FLAGS.dry_run:
    model = None
  else:
    from concordia.contrib.language_models import language_model_setup  # pylint: disable=g-import-not-at-top

    api_key = (
        FLAGS.api_key
        or os.environ.get('GOOGLE_API_KEY', '')
        or os.environ.get('GEMINI_API_KEY', '')
        or os.environ.get('OPENAI_API_KEY', '')
        or None
    )
    model = language_model_setup(
        api_type=FLAGS.api_type,
        model_name=FLAGS.model_name,
        api_key=api_key,
    )

  agents = generate_population(
      model=model,
      num_agents=FLAGS.num_agents,
      island_name=FLAGS.island_name,
      seed=FLAGS.seed,
      setting=FLAGS.setting,
      max_retries=FLAGS.max_retries,
  )
  output = build_roster(
      agents,
      setting=FLAGS.setting,
      island_name=FLAGS.island_name,
      seed=FLAGS.seed,
      dry_run=FLAGS.dry_run,
  )

  os.makedirs(os.path.dirname(FLAGS.output_path) or '.', exist_ok=True)
  with open(FLAGS.output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
  logging.info('Saved %d agents to %s', len(agents), FLAGS.output_path)

  meta = output['metadata']
  logging.info('POPULATION SUMMARY: %s', meta['location_name'])
  for tier, info in meta['economic_tiers'].items():
    logging.info('  %s: %d agents', tier, info['count'])
  for g, count in meta['gender_distribution'].items():
    logging.info('  Gender %s: %d', g, count)
  for o, count in meta['orientation_distribution'].items():
    logging.info('  Orientation %s: %d', o, count)
  coupled = sum(
      1 for a in agents if 'partnered with' in a.get('relationship_status', '')
  )
  logging.info('  Coupled agents: %d (%d pairs)', coupled, coupled // 2)


if __name__ == '__main__':
  _define_flags()
  app.run(main)
