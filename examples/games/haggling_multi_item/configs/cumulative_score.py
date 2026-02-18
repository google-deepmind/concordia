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

"""Puppet test configuration for testing cumulative score aggregation.

This config runs 3 games with deterministic puppet responses to verify
that scores accumulate correctly across all rounds.

Expected outcomes with fixed responses (buyer proposes 3 coins, seller accepts):
- Game 1: Buyer profit = reward(5) - price(3) = 2, Seller profit = price(3) -
cost(1) = 2
- Game 2: Buyer profit = 2, Seller profit = 2
- Game 3: Buyer profit = 2, Seller profit = 2
- Total: Buyer = 6, Seller = 6
"""

from concordia.typing import entity as entity_lib

YEAR = 1895
MONTH = 9
DAY = 12

LOCATION = "Fruitville"
EVENT = "The seasonal fruit market"

# Run 3 games to test cumulative scoring
NUM_MAIN_PLAYERS = 1
NUM_SUPPORTING_PLAYERS = 1
NUM_GAMES = 3

ITEMS_FOR_SALE = ("apple",)
PRICES = (1, 2, 3, 4, 5)

# Fixed rewards to make expected scores predictable:
# Buyer can sell for 5 coins, seller buys for 1 coin
# If they agree on 3 coins: buyer gets 5-3=2, seller gets 3-1=2
BUYER_BASE_REWARD_MIN = 5
BUYER_BASE_REWARD_MAX = 5
SELLER_BASE_REWARD_MIN = 1
SELLER_BASE_REWARD_MAX = 1

FOCAL_PLAYER_PREFAB = "puppet__Entity"

FEMALE_NAMES = [
    "Alice Testbuyer",
]

MALE_NAMES = [
    "Bob Testseller",
]

SCENARIO_PREMISE = "It is {year}. A simple market in {location} for {event}."

VISUAL_SCENE_OPENINGS = [
    "The market is open.",
    "Trading continues.",
    "Final round.",
]

CALL_TO_SPEECH = entity_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_PROPOSE = (
    "Which fruit would {name} like to buy and what price would {name} propose?:"
)
CALL_TO_ACCEPT = "Would {name} accept the offer?:"

BUYER_PREMISE = (
    "{scene_visual} {buyer_name} is buying from {seller_name}. "
    "Resale value: {buyer_reward} coins."
)

SELLER_PREMISE = (
    "{scene_visual} {seller_name} is selling to {buyer_name}. "
    "Acquisition cost: {seller_cost} coin."
)

# Configure supporting player (seller) to ALWAYS accept
SUPPORTING_PLAYER_FIXED_RESPONSES = {
    "Would {name} accept the offer?:": "accept",
}

SUPPORTING_PLAYER_MEMORIES = [
    "I always accept offers above my cost.",
]

SHARED_MEMORIES = [
    "Trades at 3 coins are fair.",
]

MAX_STEPS = 30
