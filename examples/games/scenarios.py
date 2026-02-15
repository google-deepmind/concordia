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

"""Scenario configurations for the Pub Coordination simulation."""

from collections.abc import Collection
import dataclasses

import immutabledict


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
  """Configuration for a pub coordination scenario."""

  description: str
  config_module: str
  background_agent_prefab: str
  num_background_players: int
  num_supporting_players: int = 0
  focal_is_resident: bool = True
  tags: Collection[str] = ()


SCENARIO_CONFIGS = immutabledict.immutabledict(
    pub_coordination_london_0=ScenarioConfig(
        description=(
            "London pub coordination with rational background agents. "
            "Focal players are residents."
        ),
        config_module="london",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        tags=("coordination", "persuasion"),
    ),
    pub_coordination_london_closures_0=ScenarioConfig(
        description=(
            "London pub coordination with a chance of pub closures. "
            "Tests information asymmetry."
        ),
        config_module="london_closures",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        tags=("coordination", "persuasion", "hidden information"),
    ),
    pub_coordination_london_mini=ScenarioConfig(
        description=(
            "Minimal London scenario for fast testing with one focal and "
            "one rational background player."
        ),
        config_module="london_mini",
        background_agent_prefab="rational__Entity",
        num_background_players=1,
        tags=("coordination",),
    ),
    pub_coordination_capetown_0=ScenarioConfig(
        description=(
            "Cape Town pub coordination during Rugby World Cup. "
            "Focal players are residents with rational background agents."
        ),
        config_module="capetown",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        tags=("coordination", "persuasion"),
    ),
    pub_coordination_edinburgh_0=ScenarioConfig(
        description=(
            "Edinburgh pub coordination during Rugby World Cup. "
            "Focal players are residents with rational background agents."
        ),
        config_module="edinburgh",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        tags=("coordination", "persuasion"),
    ),
    pub_coordination_tough_friendship_0=ScenarioConfig(
        description=(
            "Visitor focal agent whose only friend prefers a different pub."
            " Other players form a separate friend group. Tests persuasion"
            " capabilities when the focal player must convince their only"
            " friend to switch pubs. 100% pub closure probability."
        ),
        config_module="edinburgh_tough_friendship",
        background_agent_prefab="rational__Entity",
        num_background_players=4,
        num_supporting_players=0,
        focal_is_resident=False,
        tags=("coordination", "persuasion", "social networks"),
    ),
    # Test-only scenarios (prefixed with _)
    _puppet_test=ScenarioConfig(
        description=(
            "Test-only scenario with all puppet agents for deterministic "
            "scoring verification. Split preferences."
        ),
        config_module="puppet_test",
        background_agent_prefab="puppet__Entity",
        num_background_players=0,
        num_supporting_players=0,
        tags=("test",),
    ),
    _puppet_test_consensus=ScenarioConfig(
        description=(
            "Test-only scenario with all puppets preferring the same pub."
        ),
        config_module="puppet_test_consensus",
        background_agent_prefab="puppet__Entity",
        num_background_players=0,
        num_supporting_players=0,
        tags=("test",),
    ),
    _puppet_test_closed=ScenarioConfig(
        description="Test-only scenario with pub closure.",
        config_module="puppet_test_closed",
        background_agent_prefab="puppet__Entity",
        num_background_players=0,
        num_supporting_players=0,
        tags=("test",),
    ),
    # Haggling scenarios (matching deprecated scenario names)
    # haggling_0: resident focal agents, rational background, fruitville
    haggling_0=ScenarioConfig(
        description=(
            "Resident population of focal agents in a haggling scenario with "
            "rational background agents in Fruitville. Players negotiate "
            "fruit prices taking turns as buyers and sellers."
        ),
        config_module="fruitville",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        focal_is_resident=True,
        tags=("negotiation",),
    ),
    # haggling_1: visitor focal agent, gullible supporting agent
    haggling_1=ScenarioConfig(
        description=(
            "Visitor focal agent in a haggling scenario with a supporting "
            "agent who will accept any price. Tests exploitation of naive "
            "trading partners."
        ),
        config_module="fruitville_gullible",
        background_agent_prefab="rational__Entity",
        num_background_players=0,
        num_supporting_players=1,
        focal_is_resident=False,
        tags=("negotiation",),
    ),
    # haggling_strange_game_0: no profitable trade possible
    haggling_strange_game_0=ScenarioConfig(
        description=(
            "Visitor focal agents in a haggling scenario where there is no "
            "profitable trade - buyer reward (1) is less than seller cost "
            "(6). The optimal strategy is not to trade."
        ),
        config_module="vegbrooke_strange_game",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        focal_is_resident=False,
        tags=("negotiation",),
    ),
    # haggling_stubborn_one_0: supporting agent only accepts 4 coins
    haggling_stubborn_one_0=ScenarioConfig(
        description=(
            "Visitor focal agent negotiating with a stubborn supporting agent "
            "who will only transact at exactly 4 coins - rejecting both higher "
            "and lower offers."
        ),
        config_module="vegbrooke_stubborn",
        background_agent_prefab="rational__Entity",
        num_background_players=0,
        num_supporting_players=1,
        focal_is_resident=False,
        tags=("negotiation", "calculation"),
    ),
    # haggling_vanilla_0: balanced vegbrooke scenario
    haggling_vanilla_0=ScenarioConfig(
        description=(
            "Resident population of focal agents in a balanced haggling "
            "scenario in Vegbrooke with rational background agents."
        ),
        config_module="vegbrooke",
        background_agent_prefab="rational__Entity",
        num_background_players=3,
        focal_is_resident=True,
        tags=("negotiation", "calculation"),
    ),
    # Test-only scenarios
    _haggling_puppet_test=ScenarioConfig(
        description=(
            "Test-only haggling scenario with puppet agents for deterministic "
            "scoring verification."
        ),
        config_module="puppet",
        background_agent_prefab="puppet__Entity",
        num_background_players=0,
        num_supporting_players=0,
        tags=("test",),
    ),
    # Haggling multi-item scenarios (multiple fruit types)
    # haggling_multi_item_0: visitor focal with gullible supporting agent
    haggling_multi_item_0=ScenarioConfig(
        description=(
            "Visitor focal agent in a multi-item haggling scenario with a "
            "gullible supporting agent who accepts any offer. Players "
            "negotiate which fruit to buy (apple, banana, pear) and at what "
            "price. Tests exploitation with hidden per-item valuations."
        ),
        config_module="fruitville_gullible",
        background_agent_prefab="rational__Entity",
        num_background_players=0,
        num_supporting_players=1,
        focal_is_resident=False,
        tags=("negotiation", "hidden information"),
    ),
    # haggling_multi_item_1: resident focal in vegbrooke
    haggling_multi_item_1=ScenarioConfig(
        description=(
            "Resident population of focal agents in a multi-item haggling "
            "scenario in Vegbrooke. Players negotiate fruit types and prices "
            "with varying per-item valuations. Tests calculation of optimal "
            "item-price combinations."
        ),
        config_module="vegbrooke",
        background_agent_prefab="rational__Entity",
        num_background_players=2,
        focal_is_resident=True,
        tags=("negotiation", "calculation"),
    ),
    # Test-only multi-item scenario
    _haggling_multi_item_puppet_test=ScenarioConfig(
        description=(
            "Test-only multi-item haggling scenario with puppet agents for "
            "deterministic scoring verification."
        ),
        config_module="puppet",
        background_agent_prefab="puppet__Entity",
        num_background_players=0,
        num_supporting_players=0,
        tags=("test",),
    ),
    # Reality Show scenarios (game-theoretic social dilemmas)
)
