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

"""Vegbrooke strange game configuration - there is no profitable trade.

In this scenario, the buyer can only sell for 1 coin (their reward) but the
seller needs 6 coins to break even. There is no mutually beneficial trade.
The optimal strategy is to not trade at all.
"""

from concordia.typing import entity as entity_lib

YEAR = 1913
MONTH = 9
DAY = 12

LOCATION = "Vegbrooke"
EVENT = "The vegetable market"

NUM_MAIN_PLAYERS = 3
NUM_GAMES = 2

BUYER_BASE_REWARD_MIN = 1
BUYER_BASE_REWARD_MAX = 1
SELLER_BASE_REWARD_MIN = 6
SELLER_BASE_REWARD_MAX = 6

PRICE_OPTIONS = ("1 coin", "2 coins", "3 coins", "4 coins", "5 coins")

FOCAL_PLAYER_PREFAB = "basic__Entity"
BACKGROUND_PLAYER_PREFAB = "rational__Entity"

FEMALE_NAMES = [
    "Elara Greenleaf",
    "Seraphina Rootwood",
    "Willow Thistlebrook",
    "Ivy Mossheart",
    "Rosalind Nettleford",
    "Anya Pepperbloom",
    "Bryony Trufflewood",
    "Linnea Beetleblossom",
    "Maeve Parsnipvale",
    "Thora Gourdvine",
]

MALE_NAMES = [
    "Cedric Willowbark",
    "Rowan Mossglen",
    "Finnian Thistledew",
    "Asher Nettlewood",
    "Jasper Peppercorn",
    "Silas Trufflebrook",
    "Eamon Beetlebranch",
    "Gareth Parsnipfield",
    "Torin Gourdwhisper",
    "Callum Leekstone",
]

SCENARIO_PREMISE = (
    "It is {year}, in the enchanted kingdom of Verdant. Nestled among rolling "
    "hills lies the quaint town of {location}, renowned for its vibrant "
    "vegetable market. Merchants and travelers from across the realm journey "
    "to {location} for {event}."
)

VISUAL_SCENE_OPENINGS = [
    (
        "The first rays of dawn paint the sky with hues of orange and gold as "
        "the vegetable market of Vegbrooke awakens. Merchants bustle about, "
        "arranging their colorful displays of crisp cabbages, plump pumpkins, "
        "and fragrant herbs."
    ),
    (
        "A gentle mist blankets the cobblestone streets of Vegbrooke as the "
        "market begins to stir. The air fills with the earthy aroma of freshly "
        "harvested root vegetables and the cheerful chatter of early shoppers."
    ),
    (
        "Sunlight filters through the leaves of the ancient oak tree that "
        "stands sentinel over the market square. Farmers arrive in their "
        "creaking carts, laden with baskets overflowing with vibrant produce, "
        "ready for a day of lively trade."
    ),
    (
        "The sound of cheerful bartering fills the air as the market of "
        "Vegbrooke bursts into life. Shoppers eagerly inspect the mounds of "
        "gleaming peppers, glistening eggplants, and artfully arranged bundles "
        "of asparagus."
    ),
    (
        "A cool breeze carries the scent of blooming flowers from the nearby "
        "meadows as the market of Vegbrooke awakens. Merchants greet each "
        "other with warm smiles, preparing for another day of bustling "
        "activity and the joy of sharing the bounty of the land."
    ),
]

CALL_TO_SPEECH = entity_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_PROPOSE = "What price would {name} propose?:"
CALL_TO_ACCEPT = "Would {name} accept the offer?:"

NEGOTIATION_PREMISE = (
    "{buyer_name} is trying to buy some vegetables from {seller_name}. They "
    "are negotiating a price."
)

BUYER_PREMISE = (
    "{scene_visual} {buyer_name} is trying to buy some vegetables from "
    "{seller_name}. {buyer_name} can sell the vegetables for {buyer_reward} "
    "coins back in their home town."
)

SELLER_PREMISE = (
    "{scene_visual} {seller_name} is trying to sell some vegetables. They are "
    "negotiating a price with {buyer_name}. It costs {seller_name} "
    "{seller_cost} coins to buy the vegetables from the farm."
)

SHARED_MEMORIES = [
    "Vegetables are sold by weight.",
    (
        "The price of one kilogram of vegetables is, on average, 3 coins. "
        "1 coin is really cheap and 5 coins is really expensive. The smallest "
        "value of transaction is 1 coin, all prices have to be in multiples "
        "of 1 coin. No fractional values are allowed."
    ),
]

MAX_STEPS = 100
