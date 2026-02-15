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

"""Cape Town configuration for the Pub Coordination simulation."""

from concordia.typing import entity as agent_lib

YEAR = 2023
MONTH = 10
DAY = 14

LOCATION = "Cape Town"
EVENT = "The Rugby World Cup"

NUM_VENUES = 2
NUM_MAIN_PLAYERS = 4
NUM_SUPPORTING_PLAYERS = 1
NUM_GAMES = 3
GAME_COUNTRIES = [
    "South Africa",
    "New Zealand",
    "Australia",
    "England",
    "Wales",
]

VENUE_PREFERENCES = {
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
}

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
]

SOCIAL_CONTEXTS = [
    (
        "The sun beats down on Camps Bay beach, white sand. Friends lounge on"
        " colorful towels, laughter echoing against the backdrop of crashing"
        " waves. {name} just arrived."
    ),
    (
        "The scent of spices and grilled meat fills the air at the V&A"
        " Waterfront Food Market. Friends navigate the bustling crowds, plates"
        " piled high with samosas and biltong. {name} just arrived."
    ),
    (
        "A cool breeze rustles through the leaves as friends hike up Table"
        " Mountain. The city sprawls out below. {name} just arrived."
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
