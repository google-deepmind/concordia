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

"""Scenarios: the Sandbox, and presets of it (settings, not code)."""

from collections.abc import Mapping
import copy
from typing import Any

from examples.lifeways.world import culture as culture_lib
from examples.lifeways.world import land as land_lib
from examples.lifeways.world import world as world_lib
import numpy as np

# Reading and writing, in a script of six marks (`culture.MARKS`) that
# drifts as it is taught.
SCRIPT = ['hook', 'ring', 'cross', 'wave', 'fork', 'dot']
_READING = {
    'kind': 'skill',
    'domain': 'learning',
    'text': f'Reading and writing marks on birch bark: {" ".join(SCRIPT)}.',
    'params': {'level': 1, 'script': SCRIPT},
}

SANDBOX: Mapping[str, Any] = {
    'scenario': 'sandbox',
    'seed': 0,
    'rows': 16,
    'cols': 24,
    # The land: how much each tile carries, how fast food regrows and what
    # even a bare tile regrows. Together they set how many people the valley
    # feeds (here 20-60, about 40, over a thousand years).
    'richness': 0.2,
    'regrowth': 0.5,
    'seed_bank': 0.06,
    # Good years and lean (DESIGN §2): a good year turns lean with chance
    # `start`, a lean one stays lean with chance `stay`, and a lean year's
    # land grows back `lean` times as much as usual (a good year's a little
    # more, so that over many years the land feeds as many). None: every
    # year alike.
    'seasons': {'start': 0.2, 'stay': 0.4, 'lean': 0.6},
    # What a person forages in a year whatever else it does (a child half);
    # gathering is the year's work on top of it.
    'subsistence': 1.0,
    'roam': 3,  # How far one moving looks for better land.
    'founders': 12,
    'people': 'River people',  # The founders' people, by descent.
    'mode': world_lib.SEXUAL,
    'mutation': 0.05,  # Copying errors at birth.
    # A child's speech drifts from its mother's by a word with this chance;
    # talk between dialects adds `garble` times their misunderstanding to the
    # chance of a copying error.
    'dialect_drift': 0.02,
    # Whom a child learns to speak from: its mother, or one of the grown
    # around her, drawn (`world_lib.COMMUNITY`): a community's speech drifts
    # together, and parted communities drift apart.
    'speech_from': world_lib.MOTHER,
    'garble': 0.5,
    # The channels customs pass by among the living, and their reach and
    # fidelity; leave one out and it does not exist.
    'channels': copy.deepcopy(dict(world_lib.DEFAULT_CHANNELS)),
    'child_cost': 2.0,
    'max_years': None,  # The Sandbox runs indefinitely.
    # Free mode (live play): a year also closes after this many seconds,
    # whoever has not answered doing nothing. None: tick mode (reproducible).
    'free': None,
    'history_kept': 200,
    # The Experimenter's moves by year (in the World's `RELEASE_FORM`); none:
    # the Sandbox's Experimenter watches, or is a person.
    'plan': {},
    # Libraries: founding one costs food; they stand apart; a shelf holds a
    # dozen texts; one unused for 30 years is a ruin, whose texts crumble.
    'library': {
        'cost': 3.0,
        'spacing': 5,
        'shelf': 12,
        'ruin_after': 30,
        'crumble': 0.1,
    },
    # Schools: founding one costs food; they stand apart; a teacher's term
    # is five years.
    'school': {'cost': 3.0, 'spacing': 5, 'term': 5},
    # Shrines: building one costs food; they stand apart; their rite falls
    # every fourth year.
    'shrine': {'cost': 3.0, 'spacing': 5, 'every': 4},
    # Courts: building one costs food; they stand apart; a court hears those
    # seen hoarding within `reach` of it, and one judged gives each of the
    # needy it passed over up to `fine` food.
    'court': {'cost': 3.0, 'spacing': 6, 'reach': 5, 'fine': 1.0},
    # Markets: building one costs `cost` food; they stand `spacing` apart.
    # Within `reach` of one, the grown may sell the know-how they were given
    # or buy it: a buyer pays `price` food to the best seller there, who
    # teaches at most `per_seller` a year (the `market` channel).
    'market': {
        'cost': 3.0,
        'spacing': 6,
        'reach': 3,
        'price': 1.0,
        'per_seller': 2,
    },
    # Disasters: each year one strikes with this chance, of one of these
    # kinds, at a place drawn from the seed, within `radius` steps (about one
    # a century; the Experimenter may send more).
    'disasters': {
        'chance': 0.01,
        'radius': world_lib.DISASTER_RADIUS,
        'kinds': list(world_lib.DISASTERS),
    },
    # Farming: one given a farming skill may sow the land it lives on, which
    # then carries `yield` more food per level of the skill (times the
    # richness) while tended; untended for `fallow` years, it goes wild. A
    # tender's fields pass at death to its eldest living child, else its
    # partner (`inherit`; None: they stay the dead's, and go wild). No
    # founder of the Sandbox knows how; a song may teach it.
    'farming': {
        'yield': land_lib.FIELD_YIELD,
        'fallow': land_lib.FALLOW,
        'inherit': world_lib.ELDEST,
    },
    # While the valley is nearly empty, strangers may come with customs of
    # their own: they keep their elders' ways, feed only their own and have
    # children young.
    'arrivals': {
        'below': 20,
        'chance': 0.25,
        'group': 2,
        'food': 6.0,
        'people': 'Hill people',
        'customs': [
            {
                'kind': 'norm',
                'domain': 'kinship',
                'text': 'Children are a blessing; we have them while young.',
            },
            {
                'kind': 'marriage',
                'domain': 'kinship',
                'text': 'We marry only our own people.',
                'params': {'within': 'descent'},
            },
            {
                'kind': 'norm',
                'domain': 'sharing',
                'text': 'Each family feeds its own; we neither beg nor give.',
            },
            {
                'kind': 'skill',
                'domain': 'food',
                'text': 'Snaring hares on the hillsides.',
                'params': {'level': 2},
            },
            {
                'kind': 'belief',
                'domain': 'sacred',
                'text': 'The hills remember every footstep.',
            },
            {
                'kind': 'value',
                'domain': 'identity',
                'text': 'We are the Hill people; we owe nobody.',
            },
            {
                'kind': 'dialect',
                'domain': 'language',
                'text': 'The Hill tongue: gun tem wen den bo ha yo lu.',
                'params': {
                    'words': [
                        'gun',
                        'tem',
                        'wen',
                        'den',
                        'bo',
                        'ha',
                        'yo',
                        'lu',
                    ]
                },
            },
            {
                'kind': 'bias',
                'domain': 'learning',
                'text': 'Keep to the traditional ways of our elders.',
            },
        ],
    },
    'founding_customs': [
        {
            'kind': 'norm',
            'domain': 'kinship',
            'text': 'We raise a child only when the stores can feed it.',
        },
        # A quarter of the founders have made sharing their own; half only
        # know it is expected of them; a quarter know nothing of it.
        {
            'kind': 'norm',
            'domain': 'sharing',
            'text': 'We share our food with those near us who go hungry.',
            'held_by': 0.25,
        },
        {
            'kind': 'norm',
            'domain': 'sharing',
            'text': 'We share our food with those near us who go hungry.',
            'params': {'form': 'known'},
            'held_by': 0.5,
        },
        {
            'kind': 'skill',
            'domain': 'food',
            'text': 'Smoking fish over green alder.',
            'params': {'level': 2},
        },
        {
            'kind': 'belief',
            'domain': 'sacred',
            'text': 'Storms come when the shrine is neglected.',
        },
        {
            'kind': 'value',
            'domain': 'identity',
            'text': 'We are the River people; we do not forget a debt.',
        },
        {
            'kind': 'habit',
            'domain': 'food',
            'text': 'In lean years, gather before anything else.',
        },
        {
            'kind': 'bias',
            'domain': 'learning',
            'text': 'Learn from those whose stores are full.',
            'held_by': 0.5,
        },
        # Authority: a third defer to those who judge at a court, taking up
        # their ways the more readily the more they have judged.
        {
            'kind': 'bias',
            'domain': 'authority',
            'text': 'We heed those who judge at the court.',
            'held_by': 1 / 3,
        },
        {**_READING, 'held_by': 0.25},
        {
            'kind': 'dialect',
            'domain': 'language',
            'text': 'The River tongue: ka ri to ma el an so vi.',
            'params': {
                'words': ['ka', 'ri', 'to', 'ma', 'el', 'an', 'so', 'vi']
            },
        },
    ],
}


def _tongue(name: str, words: list[str]) -> dict[str, Any]:
  return {
      'kind': 'dialect',
      'domain': 'language',
      'text': f'The {name} tongue: {" ".join(words)}.',
      'params': {'words': words},
  }


# Customs any people of the presets may start with.
_WORK = [
    {
        'kind': 'norm',
        'domain': 'kinship',
        'text': 'We raise a child only when the stores can feed it.',
    },
    {
        'kind': 'skill',
        'domain': 'food',
        'text': 'Smoking fish over green alder.',
        'params': {'level': 2},
    },
    {
        'kind': 'habit',
        'domain': 'food',
        'text': 'In lean years, gather before anything else.',
    },
]
_SHARING = {
    'kind': 'norm',
    'domain': 'sharing',
    'text': 'We share our food with those near us who go hungry.',
}

# Plasmid Soup: everyone buds alone, and customs move freely, by watching
# from afar, by texts and by sermons; there are no peoples and no marriage
# rules. Do traditions form without peoples? How fast do customs sweep?
PLASMID_SOUP: Mapping[str, Any] = {
    'mode': world_lib.ASEXUAL,
    'richness': 0.17,  # Everyone buds: a poorer land feeds as many.
    'people': 'valley',
    'arrivals': None,
    'channels': {
        **copy.deepcopy(dict(world_lib.DEFAULT_CHANNELS)),
        'observation': {
            'radius': 3,
            'witnessed': 5,
            'mutation': 0.05,
            'form': 'known',
        },
    },
    'founding_customs': [
        *_WORK,
        {**_SHARING, 'held_by': 0.5},
        {**_READING, 'held_by': 0.5},
        _tongue('valley', ['ka', 'ri', 'to', 'ma', 'el', 'an', 'so', 'vi']),
        {
            'kind': 'belief',
            'domain': 'sacred',
            'text': 'The river is a god.',
            'held_by': 1 / 3,
        },
        {
            'kind': 'belief',
            'domain': 'sacred',
            'text': 'The sky watches all we do.',
            'held_by': 1 / 3,
        },
        {
            'kind': 'belief',
            'domain': 'sacred',
            'text': 'The dead walk beside us.',
            'held_by': 1 / 3,
        },
        {
            'kind': 'bias',
            'domain': 'learning',
            'text': 'Try new ways.',
            'held_by': 0.5,
        },
    ],
    # Texts left in the world, and songs: the soup's plasmids.
    'plan': {
        '20': 'release belief sacred as a text at 8,8: The stars are fires.',
        '40': 'release value identity as a song at 8,8: We belong to no one.',
        '60': 'release norm sharing as a text at 8,8: Give to the stranger.',
    },
}

# Two Valleys: two peoples with their own tongues, beliefs and marriage
# rules, in valleys a ridge parts but for a pass. Ethnogenesis, merger, or
# both, over centuries?
# Decision logics as customs (DESIGN 3): each a question one asks oneself
# before acting, as the stock `QuestionOfRecentMemories` asks it.
APPROPRIATENESS: Mapping[str, Any] = {
    'kind': 'question',
    'domain': 'reasoning',
    'text': (
        'What kind of situation is this, what kind of person am I, and what'
        ' does a person such as I do in a situation such as this?'
    ),
}
CONSEQUENCE: Mapping[str, Any] = {
    'kind': 'question',
    'domain': 'reasoning',
    'text': (
        'What could I do now, and what would come of each choice for me and'
        ' for those near?'
    ),
}

TWO_VALLEYS: Mapping[str, Any] = {
    'rows': 20,
    'cols': 40,
    'ridge': {'col': 20, 'width': 2, 'gap': [9, 10]},
    'arrivals': None,
    'groups': [
        {
            'people': 'West people',
            'home': [10, 9],
            'founders': 12,
            'founding_customs': [
                *_WORK,
                {**_SHARING, 'held_by': 0.5},
                {**_READING, 'held_by': 0.3},
                _tongue(
                    'West', ['ka', 'ri', 'to', 'ma', 'el', 'an', 'so', 'vi']
                ),
                {
                    'kind': 'marriage',
                    'domain': 'kinship',
                    'text': 'We marry one who speaks as we do.',
                    'params': {'within': 'tongue'},
                },
                {
                    'kind': 'belief',
                    'domain': 'sacred',
                    'text': 'The river is a god.',
                },
                {
                    'kind': 'value',
                    'domain': 'identity',
                    'text': 'We are the West people, children of the river.',
                },
                APPROPRIATENESS,
            ],
        },
        {
            'people': 'East people',
            'home': [10, 30],
            'founders': 12,
            'founding_customs': [
                *_WORK,
                {**_READING, 'held_by': 0.3},
                _tongue(
                    'East', ['gun', 'tem', 'wen', 'den', 'bo', 'ha', 'yo', 'lu']
                ),
                {
                    'kind': 'marriage',
                    'domain': 'kinship',
                    'text': 'We marry only our own people.',
                    'params': {'within': 'descent'},
                },
                {
                    'kind': 'belief',
                    'domain': 'sacred',
                    'text': 'The mountain keeps the dead.',
                },
                {
                    'kind': 'value',
                    'domain': 'identity',
                    'text': 'We are the East people; the mountain is ours.',
                },
                CONSEQUENCE,
            ],
        },
    ],
}

# Know-how of sowing: the farming skill First Farmers' founders hold.
SOWING: Mapping[str, Any] = {
    'kind': 'skill',
    'domain': culture_lib.FARMING,
    'text': 'Sowing wild barley in cleared ground.',
    'params': {'level': 1},
}

# A norm of tending: who holds it sows every year it can spare the food.
TENDING: Mapping[str, Any] = {
    'kind': 'norm',
    'domain': culture_lib.FARMING,
    'text': 'We tend our fields every year.',
}

# A norm of property: who holds it leaves others' fields alone.
SPARING: Mapping[str, Any] = {
    'kind': 'norm',
    'domain': culture_lib.FARMING,
    'text': "We take nothing from another's field.",
}

# First Farmers: the Sandbox, a third of whose founders know how to sow.
# Does farming spread or die out, and does the valley come to feed more?
FIRST_FARMERS: Mapping[str, Any] = {
    'founding_customs': [
        *SANDBOX['founding_customs'],
        {**SOWING, 'held_by': 1 / 3},
    ],
}

# A marriage rule that bars no one: who holds it marries across peoples.
OPEN_MARRIAGE: Mapping[str, Any] = {
    'kind': 'marriage',
    'domain': 'kinship',
    'text': 'We marry whom we love.',
}


def _bridged(valleys: Mapping[str, Any]) -> dict[str, Any]:
  """Valleys in which one founder of each people holds `OPEN_MARRIAGE`."""
  bridged = copy.deepcopy(dict(valleys))
  for group in bridged['groups']:
    one = 1 / int(group['founders'])
    for custom in group['founding_customs']:
      if custom['kind'] == 'marriage':
        custom['held_by'] = 1 - one
    group['founding_customs'].append({**OPEN_MARRIAGE, 'held_by': one})
  return bridged


# Bridged Valleys: Two Valleys with one founder in each people who marries
# whom they love. The two bridge the peoples into one; as the custom gains
# and loses holders, whole peoples merge and split.
BRIDGED_VALLEYS: Mapping[str, Any] = _bridged(TWO_VALLEYS)

# What each preset is, in a line for the page.
PRESET_NOTES: Mapping[str, str] = {
    'sandbox': 'A valley left to itself: its people live, learn and die.',
    'plasmid_soup': (
        'Everyone buds alone and customs move freely: no peoples, no'
        ' marriage rules. Do traditions form without peoples?'
    ),
    'two_valleys': (
        'Two peoples with their own tongues, beliefs and marriage rules, in'
        ' valleys a ridge parts but for a pass. Do they merge or split?'
    ),
    'bridged_valleys': (
        'Two Valleys, with one founder in each people who marries for love.'
        ' Do whole peoples merge and split?'
    ),
    'first_farmers': (
        'The Sandbox, a third of whose founders know how to sow. Does'
        ' farming spread or die out?'
    ),
}

PRESETS: Mapping[str, Mapping[str, Any]] = {
    'sandbox': {},
    'plasmid_soup': PLASMID_SOUP,
    'two_valleys': TWO_VALLEYS,
    'bridged_valleys': BRIDGED_VALLEYS,
    'first_farmers': FIRST_FARMERS,
}


# The levers a twin may pull: world rules the World reads from its settings
# as it plays (never from its saved state), each with its bounds. A twin of a
# world is the same world from the same year with one lever changed.
LEVERS: Mapping[str, tuple[float, float]] = {
    'richness': (0.05, 1.0),
    'regrowth': (0.0, 1.0),
    'seed_bank': (0.0, 0.5),
    'subsistence': (0.0, 2.0),
    'mutation': (0.0, 0.5),
    'dialect_drift': (0.0, 0.5),
    'garble': (0.0, 2.0),
    'child_cost': (0.0, 6.0),
}
# The lever that takes a channel away; its value names the channel.
WITHOUT = 'without'


def with_lever(
    settings: Mapping[str, Any], lever: str, value: Any
) -> dict[str, Any]:
  """A world's settings with one lever pulled, for its twin.

  Args:
    settings: The world's settings.
    lever: A key of `LEVERS`, or `WITHOUT`.
    value: The lever's new value (within its bounds), or the channel taken
      away.

  Returns:
    The twin's settings.

  Raises:
    ValueError: If the lever is not one, or the value is out of its bounds,
      or the world has no such channel.
  """
  twin = copy.deepcopy(dict(settings))
  if lever == WITHOUT:
    channels = dict(twin.get('channels', world_lib.DEFAULT_CHANNELS))
    if not isinstance(value, str) or value not in channels:
      raise ValueError(f'This world has no channel {value!r}.')
    del channels[value]
    twin['channels'] = channels
    return twin
  if lever not in LEVERS:
    raise ValueError(f'{lever!r} is not a lever.')
  low, high = LEVERS[lever]
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f'{lever} is a number.')
  if not low <= float(value) <= high:
    raise ValueError(f'{lever} is from {low} to {high}.')
  twin[lever] = float(value)
  return twin


def settings(name: str = 'sandbox', **overrides: Any) -> dict[str, Any]:
  """A scenario's settings: the Sandbox's, a preset's changes, overrides.

  Args:
    name: The scenario: `sandbox`, or a preset in `PRESETS`.
    **overrides: Settings that replace the scenario's.

  Returns:
    The settings.
  """
  merged = copy.deepcopy(dict(SANDBOX))
  merged.update(copy.deepcopy(dict(PRESETS[name])))
  merged['scenario'] = name
  merged.update(overrides)
  return merged


def replayed(replay: Mapping[str, Any]) -> dict[str, Any]:
  """A world's settings to play it again, its moves as the Experimenter's plan.

  Args:
    replay: A world's replay (`web.session.Session.replay`): its scenario,
      seed, settings over the scenario's, moves by year and years played.

  Returns:
    The settings, in tick mode, for as many years as it played.
  """
  overrides = {k: v for k, v in replay.get('overrides', {}).items()}
  overrides.pop('free', None)  # A year closed by the clock is not replayed.
  return settings(
      str(replay['scenario']),
      **{
          **overrides,
          'seed': int(replay['seed']),
          'plan': {str(y): str(m) for y, m in replay.get('plan', {}).items()},
          'max_years': int(replay['years']),
      },
  )


def founders(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
  """The founders of a world: who they are, where, and what they hold.

  Made once from the settings, for the person prefabs and the World GM
  alike. The founders come in groups (`groups`: each a `people`, a `home`
  tile, a number of `founders` and their `founding_customs`), or in one
  group from the top-level settings of those names. Within a group, a
  founding custom is held by every founder, or by the share of them its
  `held_by` says (drawn from the seed), among those who hold nothing of its
  kind and domain yet: listing a norm known by half the founders after the
  same norm internalized by a quarter gives the rest none.

  Args:
    settings: The world's settings.

  Returns:
    For each founder: `params` (its person prefab's: name, birth year, sex,
    the world seed and its customs' specs with their ids), `tile`, `food`
    and `people`.
  """
  seed = int(settings.get('seed', 0))
  rows, cols = int(settings.get('rows', 16)), int(settings.get('cols', 24))
  groups = settings.get('groups') or [{
      'people': settings.get('people', 'River people'),
      'home': [rows // 2, cols // 3],
      'founders': settings.get('founders', 12),
      'founding_customs': settings.get('founding_customs', ()),
  }]
  asexual = settings.get('mode', world_lib.SEXUAL) == world_lib.ASEXUAL
  culture = culture_lib.Culture()
  records = []
  for g, group in enumerate(groups):
    keys = (g,) if g else ()  # A single group draws as it always has.
    count = int(group.get('founders', 12))
    holders: dict[str, set[int]] = {}
    loci: dict[tuple[str, str], set[int]] = {}
    for i, spec in enumerate(group.get('founding_customs', ())):
      custom = culture.add(
          spec['kind'], spec['domain'], spec['text'], spec.get('params')
      )
      held = loci.setdefault((custom.kind, custom.domain), set())
      order = np.random.default_rng(world_lib.seed_of(seed, 'held', *keys, i))
      free = [int(f) for f in order.permutation(count) if int(f) not in held]
      chosen = set(free[: round(float(spec.get('held_by', 1.0)) * count)])
      holders[custom.id] = chosen
      held.update(chosen)
    rng = np.random.default_rng(world_lib.seed_of(seed, 'founders', *keys))
    home = group['home']
    for i in range(count):
      tile = [
          (int(home[0]) + int(rng.integers(-2, 3))) % rows,
          (int(home[1]) + int(rng.integers(-2, 3))) % cols,
      ]
      records.append({
          'params': {
              'name': world_lib.name_of(len(records), seed),
              'born': -int(rng.integers(18, 40)),
              'sex': '' if asexual else ('f' if i % 2 == 0 else 'm'),
              'parents': [],
              'seed': seed,
              'customs': [
                  culture.get(c).spec()
                  for c, held in holders.items()
                  if i in held
              ],
          },
          'tile': tile,
          'food': 8.0,
          'people': str(group.get('people', 'River people')),
      })
  return records
