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

"""Fruitville configuration for the Haggling simulation."""

from concordia.typing import entity as entity_lib

YEAR = 1895
MONTH = 9
DAY = 12

LOCATION = "Fruitville"
EVENT = "The seasonal fruit market"

NUM_MAIN_PLAYERS = 3
NUM_GAMES = 2

# Bargaining parameters
BUYER_BASE_REWARD_MIN = 5
BUYER_BASE_REWARD_MAX = 6
SELLER_BASE_REWARD_MIN = 1
SELLER_BASE_REWARD_MAX = 2

# Available price options
PRICE_OPTIONS = ("1 coin", "2 coins", "3 coins", "4 coins", "5 coins")

# Focal and background prefabs
FOCAL_PLAYER_PREFAB = "basic__Entity"
BACKGROUND_PLAYER_PREFAB = "rational__Entity"

FEMALE_NAMES = [
    "Anya Blossomwood",
    "Brynn Orchardheart",
    "Clara Applebrook",
    "Della Plumstone",
    "Elara Berryvine",
    "Faye Honeydew",
    "Gaia Willowbrook",
    "Hazel Nutgrove",
    "Ivy Pearblossom",
    "Juniper Quincewood",
    "Kira Citronbloom",
    "Lila Figleaf",
    "Maeve Mulberry",
    "Nova Peachwood",
    "Opal Apricot",
    "Sasha Grapevine",
    "Quinn Cherryblossom",
    "Rowan Cranberry",
    "Sage Honeycomb",
    "Willow Blackberry",
]

MALE_NAMES = [
    "Aiden Riversong",
    "Blaise Windwillow",
    "Cedric Meadowbrook",
    "Dorian Sunstone",
    "Elias Berryfield",
    "Flynn Honeycomb",
    "Garen Willowshade",
    "Hunter Nutwood",
    "Ivor Pearbrook",
    "Jasper Quincehill",
    "Kieran Citronbrook",
    "Lennon Figtree",
    "Magnus Mulberrylane",
    "Nolan Peachgrove",
    "Orion Apricotwood",
    "Peregrine Grapehill",
    "Quentin Cherrybrook",
    "Rowan Cranberryfield",
    "Silas Honeybrook",
    "Tristan Blackberrylane",
]

SCENARIO_PREMISE = (
    "It is {year}, in the realm of Ouroboros. There is a quiet village of "
    "{location}, which is famous for its fruit market. Traders from all over "
    "the realm come to {location} for {event}."
)

VISUAL_SCENE_OPENINGS = [
    (
        "The first rays of dawn painted the sky above Fruitville in hues of "
        "orange and gold, casting a warm glow over the bustling market. Stalls "
        "overflowed with vibrant fruits, their aromas mingling in the crisp "
        "morning air."
    ),
    (
        "As the sun peeked over the horizon, the market of Fruitville stirred "
        "to life. Merchants, their voices a cheerful symphony, arranged their "
        "wares: glistening berries, plump melons, and exotic fruits from "
        "distant lands."
    ),
    (
        "Dewdrops clung to the colorful fruits displayed in the market of "
        "Fruitville, reflecting the soft morning light. The air buzzed with "
        "anticipation as traders and customers alike gathered for the day's "
        "trade."
    ),
    (
        "The cobblestone streets of Fruitville echoed with the clatter of "
        "hooves and the rumble of carts as the market awoke. Underneath "
        "colorful awnings, merchants proudly presented their bountiful "
        "harvests, their voices a chorus of greetings and bartering."
    ),
    (
        "In the heart of Fruitville, the market square transformed into a "
        "kaleidoscope of colors as the sun rose. Fruits of every imaginable "
        "shape and size adorned the stalls, a feast for the eyes and a promise "
        "of delightful flavors."
    ),
]

# Call-to-action prompts (CRITICAL for puppet agents to match)
CALL_TO_SPEECH = entity_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_PROPOSE = "What price would {name} propose?:"
CALL_TO_ACCEPT = "Would {name} accept the offer?:"

NEGOTIATION_PREMISE = (
    "{buyer_name} is trying to buy some fruit from {seller_name}. They are "
    "negotiating a price."
)

BUYER_PREMISE = (
    "{scene_visual} {buyer_name} is trying to buy some fruit from "
    "{seller_name}. {buyer_name} can sell the fruit for {buyer_reward} coins "
    "back in their home town."
)

SELLER_PREMISE = (
    "{scene_visual} {seller_name} is trying to sell some fruit. They are "
    "negotiating a price with {buyer_name}. It costs {seller_name} "
    "{seller_cost} coins to buy the fruit from the farm."
)

# Shared memories for all players
SHARED_MEMORIES = [
    "Fruits are sold by weight.",
    (
        "The price of one kilogram of fruit is, on average, 3 coins. 1 coin "
        "is really cheap and 5 coins is really expensive. The smallest value "
        "of transaction is 1 coin, all prices have to be in multiples of 1 "
        "coin. No fractional values are allowed."
    ),
]

MAX_STEPS = 100
