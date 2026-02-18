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

"""Edinburgh closures configuration.

Inherits from Edinburgh base config, overriding:
- NUM_VENUES = 3
- NUM_MAIN_PLAYERS = 6
- NUM_GAMES = 5
- PUB_CLOSED_PROBABILITY = 1.0 (one pub always closed each round)
"""

from examples.games.pub_coordination.configs import edinburgh

globals().update(
    {k: v for k, v in vars(edinburgh).items() if not k.startswith("_")}
)

NUM_VENUES = 3
NUM_MAIN_PLAYERS = 6
NUM_GAMES = 5
PUB_CLOSED_PROBABILITY = 1.0
