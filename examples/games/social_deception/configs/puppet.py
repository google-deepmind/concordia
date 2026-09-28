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

"""Minimal deterministic puppet agent configuration."""

from examples.games.social_deception.setup import scripts

SCRIPT_NAME = "basic_epistemic_town"
SCRIPT_ENUM = scripts.GameScript.BASIC_EPISTEMIC_TOWN
DEFAULT_NUM_PLAYERS = 5
DEFAULT_PLAYER_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve"]
PLAYER_CAN_PASS = True
FOCAL_PLAYER_PREFAB = "puppet__Entity"
BACKGROUND_PLAYER_PREFAB = "puppet__Entity"
