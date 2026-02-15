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

"""Edinburgh configuration for the Pub Coordination simulation."""

from concordia.typing import entity as agent_lib

YEAR = 2023
MONTH = 10
DAY = 14

LOCATION = "Edinburgh"
EVENT = "The Rugby World Cup"

NUM_VENUES = 2
NUM_MAIN_PLAYERS = 4
NUM_SUPPORTING_PLAYERS = 1
NUM_GAMES = 3
GAME_COUNTRIES = ["Scotland", "Ireland", "Italy", "France", "England"]
NUM_PEOPLE = 5

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
