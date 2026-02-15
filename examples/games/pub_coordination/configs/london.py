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

"""London configuration for the Pub Coordination simulation."""

from concordia.typing import entity as agent_lib

YEAR = 2015
MONTH = 5
DAY = 14

LOCATION = "London"
EVENT = "The European football cup"

NUM_VENUES = 2
NUM_MAIN_PLAYERS = 4
NUM_BACKGROUND_PLAYERS = 2
NUM_SUPPORTING_PLAYERS = 1
NUM_GAMES = 3
GAME_COUNTRIES = ["England", "France", "Germany", "Italy", "Spain"]

FOCAL_PLAYER_PREFAB = "basic__Entity"
BACKGROUND_PLAYER_PREFAB = "rational__Entity"

VENUE_PREFERENCES = {
    "The Princess of Wales": [
        (
            "A cozy and traditional pub with a roaring fireplace, perfect for"
            " escaping the cold."
        ),
        "Serves a wide selection of classic British ales and lagers.",
        "Hosts a weekly quiz night that's always a lively affair.",
        "Has a charming beer garden with plenty of seating.",
        "The staff are friendly and welcoming, making everyone feel at home.",
    ],
    "The Crooked Billet": [
        "A historic pub with a unique and quirky atmosphere.",
        (
            "Boasts a wide selection of craft beers and ciders from local"
            " breweries."
        ),
        (
            "Hosts live music performances on weekends, showcasing a variety of"
            " genres."
        ),
        "Serves delicious, homemade pub food with generous portions.",
        (
            "The staff are knowledgeable about their drinks and always happy to"
            " recommend something new."
        ),
    ],
    "The Clapton Hart": [
        "A modern and stylish pub with a vibrant atmosphere.",
        "Offers an extensive selection of craft beers, cocktails, and wines.",
        "Hosts regular DJ nights and themed parties.",
        "Serves innovative and globally-inspired cuisine.",
        "The staff are attentive and create a fun and welcoming environment.",
    ],
    "The King's Head": [
        "A traditional pub with a focus on sports.",
        "Multiple screens showing live sporting events throughout the week.",
        "Serves classic pub grub and a wide selection of beers on tap.",
        "Has a pool table and dartboard for some friendly competition.",
        "The atmosphere is lively and energetic, especially during big games.",
    ],
    "The Queen's Arms": [
        "A family-friendly pub with a large outdoor play area.",
        "Serves a varied menu with options for both adults and children.",
        "Hosts regular events and activities for kids.",
        "The atmosphere is relaxed and welcoming, perfect for a family outing.",
        (
            "The staff are accommodating and go out of their way to make"
            " families feel comfortable."
        ),
    ],
}

FEMALE_NAMES = [
    "Olivia Smith",
    "Amelia Jones",
    "Isla Taylor",
    "Ava Brown",
    "Emily Wilson",
    "Isabella Johnson",
    "Mia Williams",
    "Jessica Davies",
    "Poppy Evans",
    "Lily Walker",
]
MALE_NAMES = [
    "Noah Smith",
    "William Jones",
    "Jack Taylor",
    "Logan Brown",
    "Thomas Wilson",
    "Oscar Johnson",
    "Lucas Williams",
    "Harry Davies",
    "Ethan Evans",
    "Jacob Walker",
]

SOCIAL_CONTEXTS = [
    (
        "The sun peeks through the morning mist as friends gather near the"
        " entrance of Hackney Marshes. They stretch and laugh, their colorful"
        " running attire contrasting with the green expanse. {name} just"
        " arrived."
    ),
    (
        "The aroma of freshly brewed coffee and artisan pastries fills the air."
        " Friends sit at a long wooden table under a striped awning, their"
        " laughter mingling with the chatter of the bustling market. {name}"
        " just arrived."
    ),
    (
        "Sunlight dances on the water as friends cycle along the towpath of"
        " Regent's Canal. They pause to admire the colorful houseboats and wave"
        " at fellow cyclists. {name} just arrived."
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
