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

"""Edinburgh tough friendship configuration.

In this scenario:
- The focal player's only friend prefers a different pub than them
- Other players form a separate friend group that excludes focal + friend
- 100% pub closure probability (one pub always closed each round)
- Tests whether the focal player can persuade their only friend to switch pubs
"""

from concordia.typing import entity as agent_lib

YEAR = 2023
MONTH = 10
DAY = 14

LOCATION = "Edinburgh"
EVENT = "The Rugby World Cup"

NUM_VENUES = 2
NUM_MAIN_PLAYERS = 5
NUM_GAMES = 4

# High closure probability - always one closed (matching deprecated)
PUB_CLOSED_PROBABILITY = 1.0

GAME_COUNTRIES = [
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
]

VENUE_PREFERENCES = {
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
}

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

SOCIAL_CONTEXTS = [
    (
        "The Royal Mile is bustling with tourists and street performers."
        " Friends navigate the crowds, admiring the historic buildings. The"
        " sound of bagpipes fills the air. {name} just arrived."
    ),
    (
        "A cool mist hangs over the city as friends climb Calton Hill. The view"
        " from the top is breathtaking, with the cityscape stretching out"
        " before them. {name} just arrived."
    ),
    (
        "The aroma of coffee fills the air in Stockbridge. Friends stroll along"
        " the charming streets, browsing the independent shops and cafes."
        " {name} just arrived."
    ),
    (
        "The sound of laughter echoes through the Grassmarket as friends"
        " explore the lively market stalls. They haggle for souvenirs and"
        " sample local delicacies. {name} just arrived."
    ),
    (
        "The sun sets over the Firth of Forth, casting a golden glow on the"
        " water. Friends walk along Portobello Beach, enjoying the fresh air."
        " {name} just arrived."
    ),
]

CALL_TO_SPEECH = agent_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_DECISION = "To which pub would {name} go to watch the game?"
SCENARIO_PREMISE = (
    "It is {year}, {location}. {event} is happening. A group of friends is"
    " planning to go to the pub and watch the game."
)
CONVERSATION_PREMISE = (
    "{name} is meeting up with friends to decide where to watch the game."
)
DECISION_PREMISE = "It is time for {name} to decide where to go."


# Custom relationship configuration for "tough friendship" scenario
# - First player (visitor) is only friends with second player
# - Second player prefers a different pub than first player
# - Other players (3,4,5) form a clique that excludes the first two
USE_CUSTOM_RELATIONSHIPS = True


def make_tough_friendship_matrix(names):
  """Creates the specific relationship matrix for tough friendship scenario.

  Args:
      names: List of player names (at least 5 expected).

  Returns:
      A dict-of-dicts relationship matrix where:
      - names[0] and names[1] are mutual friends
      - names[2:] form a separate friend group
      - No connections between these groups
  """
  m = {}
  for a in names:
    m[a] = {}
    for b in names:
      if a == b:
        m[a][b] = 1.0
      else:
        m[a][b] = 0.0

  # First two are friends with each other
  if len(names) >= 2:
    m[names[0]][names[1]] = 1.0
    m[names[1]][names[0]] = 1.0

  # Rest form a clique
  for i in range(2, len(names)):
    for j in range(2, len(names)):
      if i != j:
        m[names[i]][names[j]] = 1.0

  return m


# Preference assignment: first player gets venue 0, friend gets venue 1
# (Environment will read these via PERSON_PREFERENCES if set, otherwise random)
# Note: Actual venue indices depend on sampling; environment.py handles this.
FOCAL_PREFERS_VENUE_INDEX = 0
FRIEND_PREFERS_VENUE_INDEX = 1
