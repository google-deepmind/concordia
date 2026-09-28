# Copyright 2026 DeepMind Technologies Limited.
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

"""Standard Base Game (Demons vs Basic Townsfolk) scenario configuration."""

from examples.games.social_deception.setup import scripts

SCRIPT_NAME = "base_game"
SCRIPT_ENUM = scripts.GameScript.BASE_GAME
DEFAULT_NUM_PLAYERS = 6
DEFAULT_PLAYER_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
PLAYER_CAN_PASS = True
