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

"""London Mini configuration for the Pub Coordination simulation."""

from examples.games.pub_coordination.configs import london

# Inherit everything from london
globals().update(
    {k: v for k, v in london.__dict__.items() if not k.startswith("_")}
)

# Override specific parameters
NUM_VENUES = 2
NUM_MAIN_PLAYERS = 2
NUM_SUPPORTING_PLAYERS = 1
NUM_GAMES = 1
GAME_COUNTRIES = ["England", "Germany"]
