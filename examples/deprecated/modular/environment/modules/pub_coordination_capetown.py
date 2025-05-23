# Copyright 2024 DeepMind Technologies Limited.
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

"""A set of pub names and reasons to like them."""

import random
from examples.deprecated.modular.environment import pub_coordination

YEAR = 2023
MONTH = 10
DAY = 14

NUM_PUBS = 3

PUB_PREFERENCES = {
    "The Springbok's Lair": [
        "Massive screens for an immersive viewing experience",
        "Green and gold decor to show your Bokke pride",
        "Special game-day menu with South African favorites",
        "Energetic atmosphere with passionate fans",
    ],
    "The Table Mountain Tavern": [
        "Stunning views of Table Mountain from the rooftop terrace",
        "Relaxed atmosphere perfect for casual viewing",
        "Craft beer selection featuring local breweries",
        "Delicious pub grub with a modern twist",
    ],
    "The Waterfront Whistle Stop": [
        "Bustling location in the heart of the V&A Waterfront",
        "Lively crowd with a mix of locals and tourists",
        "Wide range of international beers and spirits",
        "Outdoor seating to enjoy the harbor views",
    ],
    "The Newlands Nectar": [
        "Historic pub near the iconic Newlands Stadium",
        "Rugby memorabilia adorning the walls",
        "Traditional pub fare done to perfection",
        "Friendly staff who know their rugby",
    ],
    "The Protea's Perch": [
        "Family-friendly atmosphere with a dedicated play area",
        "Large screens visible from every corner",
        "Affordable menu with options for all ages",
        "Cheerful and welcoming environment",
    ],
    "The Cape Town Kick-Off": [
        "Multiple screens showing different games simultaneously",
        "Sports-themed decor creating a vibrant atmosphere",
        "Extensive menu with international flavors",
        "Great spot for catching any sporting event",
    ],
    "The Lion's Den": [
        "Cozy and intimate setting with a fireplace",
        "Perfect for watching games with a small group",
        "Wide selection of whiskeys and single malts",
        "Knowledgeable staff who can discuss the game in-depth",
    ],
    "The Stormers' Stronghold": [
        "Dedicated to the local Super Rugby team",
        "Blue and white decor to show your support",
        "Special events and promotions on game days",
        "Passionate fans creating an electric atmosphere",
    ],
    "The Bo-Kaap Brewhouse": [
        "Unique location in the colorful Bo-Kaap neighborhood",
        "Vibrant atmosphere with live music on weekends",
        "Focus on craft beers and local brews",
        "Delicious fusion cuisine with a Cape Malay influence",
    ],
    "The Kirstenbosch Kraal": [
        "Tranquil setting near the beautiful Kirstenbosch Gardens",
        "Outdoor seating surrounded by lush greenery",
        "Laid-back vibe perfect for a relaxed viewing experience",
        "Focus on sustainable and locally sourced food and drinks",
    ],
}


SOCIAL_CONTEXT = [
    (
        "The sun beats down on Camps Bay beach, casting a warm glow on the"
        " white sand. {players} lounge on colorful towels, their laughter"
        " echoing against the backdrop of crashing waves. A group of surfers"
        " emerge from the water, their boards tucked under their arms."
        " {player_name} just arrived, shielding their eyes from the sun."
    ),
    (
        "The scent of spices and grilled meat fills the air at the V&A"
        " Waterfront Food Market. {players} navigate the bustling crowds, their"
        " plates piled high with samosas, bobotie, and biltong. The sun begins"
        " to set, painting the sky with vibrant hues. {player_name} just"
        " arrived, their mouth watering at the sight of the delicious food."
    ),
    (
        "A cool breeze rustles through the leaves as {players} hike up Table"
        " Mountain. The city sprawls out below, a patchwork of buildings and"
        " streets. They reach a viewpoint, their breath catching in their"
        " chests at the breathtaking panorama. {player_name} just arrived,"
        " their face flushed from the exertion."
    ),
    (
        "The rhythmic beat of African drums echoes through the Bo-Kaap"
        " neighborhood. {players} wander the cobblestone streets, admiring the"
        " brightly colored houses. They stumble upon a street market, filled"
        " with handmade crafts and traditional clothing. {player_name} just"
        " arrived, their eyes wide with wonder."
    ),
    (
        "The vibrant energy of Long Street pulses through the air. {players}"
        " weave their way through the crowd, drawn to the lively bars and"
        " restaurants. The sound of live music spills out onto the street,"
        " enticing them to dance. {player_name} just arrived, their smile"
        " widening at the prospect of a fun night out."
    ),
]

RUGBY_COUNTRIES = [
    "South Africa",
    "New Zealand",
    "France",
    "Ireland",
    "England",
    "Australia",
    "Argentina",
    "Wales",
    "Scotland",
    "Fiji",
    "Japan",
    "Italy",
    "Samoa",
    "Georgia",
    "Tonga",
    "Romania",
    "Namibia",
    "Uruguay",
    "Chile",
    "Portugal",
]

FEMALE_NAMES = [
    "Amahle Nkosi",
    "Thandiwe Dlamini",
    "Lerato Khumalo",
    "Zinhle Ngubane",
    "Buhle Mkhize",
    "Ntombizanele Mthembu",
    "Nomusa Ndlovu",
    "Ayanda Zulu",
    "Nonhlanhla Xhosa",
    "Palesa Sotho",
    "Lindiwe Tswana",
    "Refilwe Pedi",
    "Mpho Venda",
    "Sibongile Swazi",
    "Nthabiseng Ndebele",
    "Bongiwe Shona",
    "Zandile Afrikaans",
    "Nosipho English",
    "Unathi van der Merwe",
    "Phindile Jansen",
]

MALE_NAMES = [
    "Siyabonga Khumalo",
    "Thabo Ndlovu",
    "Mandla Zulu",
    "Lunga Ngcobo",
    "Mpho Mokoena",
    "Katlego Modise",
    "Tshepo Mashaba",
    "Kagiso Khoza",
    "Lethabo Mabena",
    "Ofentse Mnguni",
    "Sandile Cele",
    "Bonga Sithole",
    "Neo Maphosa",
    "Lwazi Nkosi",
    "Sizwe Dlamini",
    "Bandile Khumalo",
    "Andile Ngubane",
    "Philani Mkhize",
    "Thamsanqa Mthembu",
    "Bongani Ndlovu",
]


def sample_parameters(seed: int | None = None):
  """Samples a set of parameters for the world configuration."""

  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  pubs = rng.sample(list(PUB_PREFERENCES.keys()), NUM_PUBS)
  pub_preferences = {k: PUB_PREFERENCES[k] for k in pubs}

  config = pub_coordination.WorldConfig(
      year=YEAR,
      location="Cape Town",
      event="The Rugby World Cup",
      game_countries=RUGBY_COUNTRIES,
      venues=pubs,
      venue_preferences=pub_preferences,
      social_context=SOCIAL_CONTEXT,
      random_seed=seed,
      num_main_players=6,
      num_supporting_players=0,
  )

  all_names = list(MALE_NAMES) + list(FEMALE_NAMES)

  rng.shuffle(all_names)
  config.people = all_names

  for _, name in enumerate(MALE_NAMES):
    config.person_data[name] = {"gender": "male"}
  for _, name in enumerate(FEMALE_NAMES):
    config.person_data[name] = {"gender": "female"}

  return config
