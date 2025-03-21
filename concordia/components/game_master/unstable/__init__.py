# Copyright 2025 DeepMind Technologies Limited.
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

"""Library of components specifically for generative game masters."""

from concordia.components.game_master.unstable import event_resolution
from concordia.components.game_master.unstable import instructions
from concordia.components.game_master.unstable import inventory
from concordia.components.game_master.unstable import make_observation
from concordia.components.game_master.unstable import next_acting
from concordia.components.game_master.unstable import next_game_master
from concordia.components.game_master.unstable import switch_act
from concordia.components.game_master.unstable import triggered_function
from concordia.components.game_master.unstable import triggered_inventory_effect
from concordia.components.game_master.unstable import world_state
