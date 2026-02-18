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
GAME_COUNTRIES = [
    "Albania",
    "Andorra",
    "Armenia",
    "Austria",
    "Azerbaijan",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "England",
    "Estonia",
    "Faroe Islands",
    "Finland",
    "France",
    "Georgia",
    "Germany",
    "Gibraltar",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Israel",
    "Italy",
    "Kazakhstan",
    "Kosovo",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Russia",
    "San Marino",
    "Scotland",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Turkey",
    "Ukraine",
    "Wales",
]

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
    "The Duke of York": [
        "A dog-friendly pub with a spacious beer garden.",
        "Welcomes dogs of all sizes and provides water bowls and treats.",
        (
            "Serves a selection of craft beers and ales, as well as"
            " dog-friendly snacks."
        ),
        (
            "The atmosphere is laid-back and friendly, making it a popular spot"
            " for dog owners."
        ),
        (
            "The staff are dog lovers themselves and always happy to see furry"
            " friends."
        ),
    ],
    "The Red Lion": [
        "A cozy and intimate pub with a focus on conversation.",
        "No loud music or TVs, creating a peaceful environment.",
        "Serves a carefully curated selection of wines and spirits.",
        "Has comfortable seating and a warm, inviting atmosphere.",
        (
            "The staff are attentive and knowledgeable, happy to engage in"
            " conversation."
        ),
    ],
    "The White Hart": [
        "A vibrant and trendy pub with a focus on cocktails.",
        "Serves a creative and extensive cocktail menu.",
        "Hosts regular cocktail masterclasses and tasting events.",
        "The atmosphere is sophisticated and stylish, perfect for a night out.",
        "The staff are skilled mixologists and passionate about their craft.",
    ],
    "The Black Swan": [
        "A historic pub with a focus on live music.",
        "Hosts a variety of musical performances throughout the week.",
        "Serves a selection of craft beers and ales, as well as cocktails.",
        (
            "The atmosphere is intimate and welcoming, perfect for enjoying"
            " live music."
        ),
        (
            "The staff are passionate about music and create a supportive"
            " environment for artists."
        ),
    ],
    "The Golden Fleece": [
        "A quirky and eclectic pub with a focus on board games.",
        "Has a vast collection of board games available to play.",
        "Hosts regular game nights and tournaments.",
        "Serves a selection of craft beers, ciders, and snacks.",
        (
            "The atmosphere is fun and social, perfect for meeting new people"
            " and enjoying games."
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
    "Sophie Martin",
    "Grace Thompson",
    "Ruby White",
    "Ella Roberts",
    "Evie Green",
    "Florence Hall",
    "Millie Wood",
    "Molly Clark",
    "Alice Lewis",
    "Phoebe Young",
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
    "Matthew Thompson",
    "Alexander White",
    "Benjamin Roberts",
    "Henry Green",
    "Daniel Hall",
    "Michael Wood",
    "Joshua Clark",
    "Elijah Lewis",
    "Jackson Young",
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
    (
        "Vibrant murals and graffiti adorn the brick walls of Shoreditch."
        " Friends wander through the streets, their eyes wide with wonder as"
        " they discover hidden gems of urban art. {name} just arrived."
    ),
    (
        "A checkered blanket is spread out on the lush green lawn of Victoria"
        " Park. Friends lounge in the sunshine, sharing snacks and stories."
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
