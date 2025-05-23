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

from collections.abc import Sequence
import random

from examples.deprecated.modular.environment import pub_coordination


YEAR = 2023
MONTH = 10
DAY = 14

NUM_PUBS = 2

PUB_PREFERENCES = {
    "Sandy Bell's": [
        "Traditional pub with a lively atmosphere",
        "Known for its folk music sessions",
        "Wide selection of Scottish beers and whiskies",
        "Friendly staff and a welcoming environment",
    ],
    "The Salt Horse": [
        "Cozy and intimate setting with a fireplace",
        "Rotating selection of craft beers on tap",
        "Delicious pub fare with locally sourced ingredients",
        "Great for a quiet pint or a catch-up with friends",
    ],
    "The Sheep Held Inn": [
        "Historic pub with a charming atmosphere",
        "Large beer garden with stunning views of the city",
        "Gastropub menu with a focus on Scottish cuisine",
        "Perfect for a special occasion or a romantic evening",
    ],
    "The Canny Man's": [
        "Quirky and eclectic pub with a unique atmosphere",
        "Wide range of international beers and spirits",
        "Hidden gem with a loyal local following",
        "Great for a conversation and a relaxed drink",
    ],
    "The Guildford Arms": [
        "Traditional pub with a Victorian interior",
        "Popular with students and locals alike",
        "Affordable prices and a lively atmosphere",
        "Perfect for a casual night out with friends",
    ],
    "The Bow Bar": [
        "Wide selection of whiskies and real ales",
        "Knowledgeable staff who can recommend a dram",
        "Traditional pub with a relaxed atmosphere",
        "Great for a whisky tasting or a quiet pint",
    ],
    "The Blue Moon": [
        "Historic pub with a literary connection",
        "Cozy and intimate setting with a fireplace",
        "Wide selection of Scottish gins and cocktails",
        "Perfect for a romantic date or a special occasion",
    ],
    "The Ensign Ewart": [
        "Traditional pub with a military history",
        "Located near Edinburgh Castle",
        "Popular with tourists and locals alike",
        "Great for a pint and a bite to eat after sightseeing",
    ],
    "The Hanging Bat": [
        "Craft beer bar with a wide selection of international beers",
        "Lively atmosphere with regular events and live music",
        "Great for a night out with friends",
        "Popular with beer enthusiasts",
    ],
}


SOCIAL_CONTEXT = [
    (
        "The Royal Mile is bustling with tourists and street performers."
        " {players} navigate the crowds, admiring the historic buildings. The"
        " sound of bagpipes fills the air. {player_name} just arrived, their"
        " eyes wide with excitement."
    ),
    (
        "A cool mist hangs over the city as {players} climb Calton Hill. The "
        "view from the top is breathtaking, with the cityscape stretching out "
        "before them. They pause to catch their breath, admiring the panoramic "
        "vista. {player_name} just arrived, their face flushed from the climb."
    ),
    (
        "The aroma of coffee fills the air in Stockbridge. {players} stroll "
        "along the charming streets, browsing the independent shops and cafes. "
        "They stop for a coffee and a pastry, enjoying the relaxed atmosphere. "
        "{player_name} just arrived, their smile widening at the sight of the "
        "quaint neighborhood."
    ),
    (
        "The sound of laughter echoes through the Grassmarket as {players}"
        " explore the lively market stalls. They haggle for souvenirs and"
        " sample local delicacies. The atmosphere is festive and vibrant."
        " {player_name} just arrived, their senses bombarded with the sights,"
        " sounds, and smells."
    ),
    (
        "The sun sets over the Firth of Forth, casting a golden glow on the"
        " water. {players} walk along Portobello Beach, enjoying the fresh air"
        " and the sound of the waves. They find a bench and watch the sun dip"
        " below the horizon. {player_name} just arrived, their heart filled"
        " with a sense of peace."
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
    "Ailsa MacDonald",
    "Catriona Campbell",
    "Fiona Stewart",
    "Isla MacLeod",
    "Morag MacKay",
    "Shona Cameron",
    "Iona Ross",
    "Mhairi Wilson",
    "Kirsty Robertson",
    "Eilidh Davidson",
]

MALE_NAMES = [
    "Angus Graham",
    "Calum Scott",
    "Douglas Reid",
    "Euan Murray",
    "Fraser Clark",
    "Hamish Taylor",
    "Iain Brown",
    "Malcolm Mitchell",
    "Niall Thomson",
    "Rory Stewart",
]


def _make_empty_relationship_matrix(names: Sequence[str]):
  """Samples a symmetric matrix of relationships in a group.

  Args:
      names: A list of strings representing the names of individuals in the
        group.

  Returns:
      A dictionary representing the symmetric relationship matrix, where:
          - Keys are names from the 'names' list.
          - Values are dictionaries, where:
              - Keys are also names from the 'names' list.
              - Values are either 0.0 or 1.0, representing the relationship
              between two individuals.
          - The matrix is symmetric: m[a][b] == m[b][a]
          - Diagonal elements are 1: m[a][a] == 1
  """

  m = {}
  for a in names:
    m[a] = {}
    for b in names:
      if a == b:
        m[a][b] = 1.0  # Diagonal elements are 1
      elif b in m and a in m[b]:
        m[a][b] = m[b][a]  # Ensure symmetry
      else:
        m[a][b] = 0.0  # No relationship

  return m


def sample_parameters(seed: int | None = None):
  """Samples a set of parameters for the world configuration."""

  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  pubs = rng.sample(list(PUB_PREFERENCES.keys()), NUM_PUBS)
  pub_preferences = {k: PUB_PREFERENCES[k] for k in pubs}

  config = pub_coordination.WorldConfig(
      year=YEAR,
      location="Edinburgh",
      event="The Rugby World Cup",
      game_countries=RUGBY_COUNTRIES,
      venues=pubs,
      venue_preferences=pub_preferences,
      social_context=SOCIAL_CONTEXT,
      random_seed=seed,
      pub_closed_probability=1.0,
      num_games=4,
      num_main_players=5,
      num_supporting_players=0,
  )

  all_names = list(MALE_NAMES) + list(FEMALE_NAMES)

  rng.shuffle(all_names)
  config.people = all_names

  for _, name in enumerate(MALE_NAMES):
    config.person_data[name] = {"gender": "male", "favorite_pub": pubs[0]}
  for _, name in enumerate(FEMALE_NAMES):
    config.person_data[name] = {"gender": "female", "favorite_pub": pubs[0]}

  m = _make_empty_relationship_matrix(config.people[: config.num_main_players])

  # Make the first two players friends.
  visitor_name = config.people[0]
  friend_name = config.people[1]
  m[visitor_name][friend_name] = 1.0
  m[friend_name][visitor_name] = 1.0

  # Make the rest of the players friends with everyone but the first two.
  for i in config.people[2 : config.num_main_players]:
    for j in config.people[2 : config.num_main_players]:
      m[i][j] = 1.0
      m[j][i] = 1.0

  config.person_data[visitor_name]["favorite_pub"] = pubs[1]
  config.person_data[friend_name]["favorite_pub"] = pubs[1]

  config.relationship_matrix = m

  return config
