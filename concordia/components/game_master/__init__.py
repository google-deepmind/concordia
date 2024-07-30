# Copyright 2022 DeepMind Technologies Limited.
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

from concordia.components.game_master import conversation
from concordia.components.game_master import coordination_payoffs
from concordia.components.game_master import current_scene
from concordia.components.game_master import direct_effect
from concordia.components.game_master import inventory
from concordia.components.game_master import inventory_based_score
from concordia.components.game_master import player_status
from concordia.components.game_master import relevant_events
from concordia.components.game_master import schedule
from concordia.components.game_master import schelling_diagram_payoffs
from concordia.components.game_master import time_display
from concordia.components.game_master import triggered_function
from concordia.components.game_master import triggered_inventory_effect
