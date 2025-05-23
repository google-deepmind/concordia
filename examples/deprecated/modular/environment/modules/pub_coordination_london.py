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

YEAR = 2015
MONTH = 5
DAY = 14

NUM_PUBS = 2

PUB_PREFERENCES = {
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

SCENARIO_KNOWLEDGE = (
    (
        "It is 2015, London. The European football cup is happening. A group of"
        " friends is planning to go to the pub and watch the game. The"
        " simulation consists of several scenes. In the discussion scene"
        " players meet in social circumstances and have a conversation."
        " Aftewards comes a decision scene where they each decide which pub"
        " they want to go to. "
    ),
)
SOCIAL_CONTEXT = [
    (
        "The sun peeks through the morning"
        " mist as {players} gather near the entrance of Hackney"
        " Marshes. They stretch and laugh, their colorful running attire"
        " contrasting with the green expanse. A few dog walkers pass by, their"
        " furry companions excitedly sniffing the air. "
        "{player_name} just arrived."
    ),
    (
        "The aroma of freshly brewed coffee and"
        " artisan pastries fills the air. {players} sit at a long wooden table"
        " under a striped awning, their laughter mingling with the chatter of"
        " the bustling market. A street musician strums a guitar nearby, adding"
        " a bohemian touch to the scene. "
        "{player_name} just arrived."
    ),
    (
        "Sunlight dances on the water as"
        " {players} cycle along the towpath of Regent's Canal. They"
        " pause to admire the colorful houseboats and wave at fellow cyclists."
        " The gentle sound of water lapping against the canal banks creates a"
        " peaceful atmosphere. "
        "{player_name} just arrived."
    ),
    (
        "Vibrant murals and graffiti adorn the brick walls of Shoreditch."
        " {players} wander through the streets, their eyes wide with wonder as"
        " they discover hidden gems of urban art. The smell of street food"
        " wafts from nearby vendors, tempting them to take a break."
        " {player_name} just arrived."
    ),
    (
        "A checkered blanket is spread out on the"
        " lush green lawn of Victoria Park. {players} lounge in the sunshine,"
        " sharing snacks and stories. The laughter of children playing nearby"
        " adds a joyful backdrop to the scene. "
        "{player_name} just arrived."
    ),
]


EURO_CUP_COUNTRIES = [
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


def sample_parameters(seed: int | None = None):
  """Samples a set of parameters for the world configuration."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  pubs = rng.sample(list(PUB_PREFERENCES.keys()), NUM_PUBS)
  pub_preferences = {k: PUB_PREFERENCES[k] for k in pubs}

  config = pub_coordination.WorldConfig(
      year=YEAR,
      location="London",
      event="European football cup",
      game_countries=EURO_CUP_COUNTRIES,
      venues=pubs,
      venue_preferences=pub_preferences,
      social_context=SOCIAL_CONTEXT,
      random_seed=seed,
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
