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

"""Minimal deterministic configuration for puppet agent tests."""

from concordia.typing import entity as agent_lib

YEAR = 2015
MONTH = 5
DAY = 14

LOCATION = "TestCity"
EVENT = "The Test Match"

NUM_VENUES = 2
NUM_MAIN_PLAYERS = 3
NUM_BACKGROUND_PLAYERS = 0
NUM_SUPPORTING_PLAYERS = 0
NUM_GAMES = 1
GAME_COUNTRIES = ["Testland", "Mockistan"]

# All players use puppet prefab for deterministic testing
FOCAL_PLAYER_PREFAB = "puppet__Entity"
BACKGROUND_PLAYER_PREFAB = "puppet__Entity"

# Minimal venues for testing
VENUE_PREFERENCES = {
    "Pub A": ["A nice pub with good atmosphere."],
    "Pub B": ["A cozy pub with great food."],
}

# Fixed player names for deterministic tests
FEMALE_NAMES = ["Alice", "Charlie"]
MALE_NAMES = ["Bob"]

# --- Player Preferences ---
# Maps player name to their favorite pub. If not specified, random.
PERSON_PREFERENCES = {
    "Alice": "Pub A",
    "Bob": "Pub B",
    "Charlie": "Pub B",
}

# Minimal social context
SOCIAL_CONTEXTS = [
    "{name} is meeting friends to decide where to watch the game.",
]

# Call to action templates that puppets will respond to
CALL_TO_SPEECH = agent_lib.DEFAULT_CALL_TO_SPEECH
CALL_TO_DECISION = "To which pub would {name} go to watch the game?"
SCENARIO_PREMISE = (
    "It is {year}, {location}. {event} is happening. Friends are deciding"
    " where to watch the game."
)
CONVERSATION_PREMISE = "{name} is deciding where to watch the game."
DECISION_PREMISE = "It is time for {name} to decide where to go."

# Probability of pub closure (0.0 = never closed)
PUB_CLOSED_PROBABILITY = 0.0
