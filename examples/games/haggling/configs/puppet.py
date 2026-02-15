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

"""Puppet test configuration for deterministic haggling testing."""

from concordia.typing import entity as entity_lib

YEAR = 1895
MONTH = 9
DAY = 12

LOCATION = "Fruitville"
EVENT = "The seasonal fruit market"

NUM_MAIN_PLAYERS = 2
NUM_GAMES = 1

# Fixed bargaining parameters for deterministic testing
BUYER_BASE_REWARD_MIN = 5
BUYER_BASE_REWARD_MAX = 5
SELLER_BASE_REWARD_MIN = 1
SELLER_BASE_REWARD_MAX = 1

# Available price options
PRICE_OPTIONS = ("1 coin", "2 coins", "3 coins", "4 coins", "5 coins")

# Use puppet prefab for all players
FOCAL_PLAYER_PREFAB = "puppet__Entity"

FEMALE_NAMES = [
    "Alice Applebrook",
    "Betty Berryvine",
]

MALE_NAMES = [
    "Charlie Citronhill",
    "David Dewdrop",
]

SCENARIO_PREMISE = (
    "It is {year}, in the realm of Ouroboros. There is a quiet village of"
    " {location}, which is famous for its fruit market. Traders come for"
    " {event}."
)

VISUAL_SCENE_OPENINGS = [
    "The market of Fruitville is open for business.",
]

# Call-to-action prompts (CRITICAL for puppet agents to match)
CALL_TO_SPEECH = entity_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_PROPOSE = "What price would {name} propose?:"
CALL_TO_ACCEPT = "Would {name} accept the offer?:"

BUYER_PREMISE = (
    "{scene_visual} {buyer_name} is trying to buy fruit from {seller_name}. "
    "{buyer_name} can sell the fruit for {buyer_reward} coins elsewhere."
)

SELLER_PREMISE = (
    "{scene_visual} {seller_name} is selling fruit to {buyer_name}. "
    "It costs {seller_cost} coin to acquire the fruit."
)

SHARED_MEMORIES = [
    "Fruits are sold by weight.",
    "The average price is 3 coins. Prices range from 1 to 5 coins.",
]

MAX_STEPS = 20
