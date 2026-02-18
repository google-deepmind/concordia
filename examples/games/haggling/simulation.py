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

"""A Haggling simulation using Concordia prefabs.

This simulates a fruit market where buyers and sellers negotiate prices.
The buyer proposes a price, and the seller accepts or rejects.
"""

import collections
import random
from typing import Any, Callable, Mapping, Sequence

from absl import logging
from examples.games.haggling import sequential_bargain_game_master
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import puppet
from concordia.prefabs.entity import rational
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions


# Price string to numeric value mapping
PRICE_TO_VALUE = {
    "1 coin": 1.0,
    "2 coins": 2.0,
    "3 coins": 3.0,
    "4 coins": 4.0,
    "5 coins": 5.0,
}


class HagglingPayoff:
  """A class to handle scoring for bargaining games."""

  def __init__(
      self,
      buyer_name: str,
      seller_name: str,
      buyer_base_reward: float,
      seller_base_reward: float,
  ):
    """Initialize the payoff handler.

    Args:
      buyer_name: Name of the buyer player.
      seller_name: Name of the seller player.
      buyer_base_reward: How much the buyer can resell the fruit for.
      seller_base_reward: Cost for the seller to acquire the fruit.
    """
    self._buyer_name = buyer_name
    self._seller_name = seller_name
    self._buyer_base_reward = buyer_base_reward
    self._seller_base_reward = seller_base_reward
    self._latest_joint_action = {}

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    return self._latest_joint_action

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Maps joint actions (accept/reject + price) to scores."""
    self._latest_joint_action = joint_action
    # Parse all actions to find current buyer and seller dynamically
    proposer = None
    responder = None
    price = 0.0
    accepted = False

    for name, action in joint_action.items():
      action_lower = action.lower().strip()
      # Check if this is a price proposal (contains coin values)
      price_match = None
      for price_str, price_val in PRICE_TO_VALUE.items():
        if price_str in action_lower:
          price_match = price_val
          break

      if price_match is not None:
        proposer = name
        price = price_match
      elif "accept" in action_lower:
        responder = name
        accepted = True
      elif "reject" in action_lower:
        responder = name
        accepted = False

    if proposer is None or responder is None:
      return {self._buyer_name: 0.0, self._seller_name: 0.0}

    if not accepted:
      return {self._buyer_name: 0.0, self._seller_name: 0.0}

    # Now determine who is buyer and who is seller
    buyer = self._buyer_name
    seller = self._seller_name

    buyer_score = self._buyer_base_reward - price
    seller_score = price - self._seller_base_reward

    return {buyer: buyer_score, seller: seller_score}

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Maps scores to descriptive observations."""
    joint_action = self._latest_joint_action
    observations = {}
    for name in [self._buyer_name, self._seller_name]:
      score = scores.get(name, 0.0)
      _ = joint_action.get(name, "unknown action")
      if score > 0:
        observations[name] = (
            f"{name} made a profit of {score:.1f} coins from the trade."
        )
      elif score == 0:
        is_rejection = any("reject" in a.lower() for a in joint_action.values())
        if is_rejection:
          observations[name] = f"{name}: The deal fell through. No trade."
        else:
          observations[name] = f"{name} broke even on the trade."
      else:
        observations[name] = f"{name} lost {abs(score):.1f} coins on the trade."
    return observations


class CumulativePayoff:
  """Tracks total payoffs across multiple games."""

  def __init__(self):
    self._game_payoffs: list[HagglingPayoff] = []
    self._total_scores: dict[str, float] = collections.defaultdict(float)

  def add_game_payoff(self, payoff: HagglingPayoff):
    self._game_payoffs.append(payoff)

  def update_scores(self, scores: Mapping[str, float]):
    for name, score in scores.items():
      self._total_scores[name] += score

  def get_total_scores(self) -> dict[str, float]:
    return dict(self._total_scores)


def sample_parameters(
    male_names: Sequence[str],
    female_names: Sequence[str],
    num_people: int,
    seed: int | None = None,
) -> tuple[list[str], random.Random]:
  """Sample player names from the pools."""
  rng = random.Random(seed)
  names = list(male_names) + list(female_names)
  rng.shuffle(names)
  selected = names[:num_people]
  return selected, rng


def create_player_pairs(
    players: Sequence[str],
    rng: random.Random,
) -> list[tuple[str, str]]:
  """Create buyer-seller pairs from a list of players (round-robin)."""
  pairs = []
  for i in range(len(players)):
    for j in range(i + 1, len(players)):
      if rng.choice([True, False]):
        pairs.append((players[i], players[j]))
      else:
        pairs.append((players[j], players[i]))
  rng.shuffle(pairs)
  return pairs


def configure_scenes(
    pairs: Sequence[tuple[str, str]],
    num_games: int,
    call_to_speech: str,
    call_to_propose: str,  # pylint: disable=unused-argument
    call_to_accept: str,  # pylint: disable=unused-argument
    visual_scene_openings: Sequence[str],
    buyer_premise_template: str,
    seller_premise_template: str,
    buyer_rewards: Sequence[float],
    seller_costs: Sequence[float],
    price_options: Sequence[str],
    rng: random.Random,
) -> list[scene_lib.SceneSpec]:
  """Configure the scenes for the simulation."""
  scenes = []

  game_idx = 0
  for _ in range(num_games):
    for buyer_name, seller_name in pairs:
      buyer_reward = buyer_rewards[game_idx]
      seller_cost = seller_costs[game_idx]
      game_idx += 1

      # Conversation scene (for negotiation talk)
      conversation_scene_type = scene_lib.SceneTypeSpec(
          name=f"negotiation_game_{game_idx}",
          game_master_name="conversation rules",
          action_spec=entity_lib.free_action_spec(
              call_to_action=call_to_speech,
          ),
      )

      opening = rng.choice(visual_scene_openings)
      buyer_premise = buyer_premise_template.format(
          scene_visual=opening,
          buyer_name=buyer_name,
          buyer_reward=buyer_reward,
          seller_name=seller_name,
      )
      seller_premise = seller_premise_template.format(
          scene_visual=opening,
          seller_name=seller_name,
          seller_cost=seller_cost,
          buyer_name=buyer_name,
      )

      scenes.append(
          scene_lib.SceneSpec(
              scene_type=conversation_scene_type,
              participants=[buyer_name, seller_name],
              num_rounds=3,
              premise={
                  buyer_name: [buyer_premise, opening],
                  seller_name: [seller_premise, opening],
              },
          )
      )

      # Decision scene (for actual proposal and acceptance)
      decision_action_spec = entity_lib.ActionSpec(
          call_to_action=(
              "{name} must decide: propose a price or accept/reject the offer."
          ),
          output_type=entity_lib.OutputType.CHOICE,
          options=list(price_options) + ["accept", "reject"],
      )

      decision_scene_type = scene_lib.SceneTypeSpec(
          name=f"decision_game_{game_idx}",
          game_master_name="decision rules",
          action_spec=decision_action_spec,
      )

      decision_premise = {
          buyer_name: [f"{buyer_name} is ready to make an offer."],
          seller_name: [f"{seller_name} has to accept or reject the offer."],
      }

      scenes.append(
          scene_lib.SceneSpec(
              scene_type=decision_scene_type,
              participants=[buyer_name, seller_name],
              num_rounds=2,  # One round per participant
              premise=decision_premise,
          )
      )

  return scenes


def run_simulation(
    config: Any,
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
) -> dict[str, Any] | None:
  """Run the Haggling simulation.

  Args:
    config: The scenario configuration module.
    model: The language model to use.
    embedder: The sentence embedder to use.

  Returns:
    A dictionary containing the simulation results.
  """
  config_lib = config
  num_main = getattr(config_lib, "NUM_MAIN_PLAYERS", 3)
  num_supporting = getattr(config_lib, "NUM_SUPPORTING_PLAYERS", 0)
  num_games = getattr(config_lib, "NUM_GAMES", 2)

  all_people, rng = sample_parameters(
      male_names=getattr(config_lib, "MALE_NAMES"),
      female_names=getattr(config_lib, "FEMALE_NAMES"),
      num_people=num_main + num_supporting,
  )

  # Split into main and supporting players
  people = all_people[:num_main]
  supporting_players = all_people[num_main:]

  # Create buyer-seller pairs
  if supporting_players:
    pairs = []
    for main_player in people:
      for support_player in supporting_players:
        if rng.choice([True, False]):
          pairs.append((main_player, support_player))
        else:
          pairs.append((support_player, main_player))
  else:
    pairs = create_player_pairs(people, rng)

  # All active players for scenes and memories
  all_active_players = list(people) + list(supporting_players)

  # Generate random rewards for each game
  buyer_rewards = [
      rng.randint(
          getattr(config_lib, "BUYER_BASE_REWARD_MIN", 5),
          getattr(config_lib, "BUYER_BASE_REWARD_MAX", 6),
      )
      for _ in range(num_games * len(pairs))
  ]
  seller_costs = [
      rng.randint(
          getattr(config_lib, "SELLER_BASE_REWARD_MIN", 1),
          getattr(config_lib, "SELLER_BASE_REWARD_MAX", 2),
      )
      for _ in range(num_games * len(pairs))
  ]

  scenes = configure_scenes(
      pairs=pairs,
      num_games=num_games,
      call_to_speech=getattr(config_lib, "CALL_TO_SPEECH"),
      call_to_propose=getattr(config_lib, "CALL_TO_PROPOSE"),
      call_to_accept=getattr(config_lib, "CALL_TO_ACCEPT"),
      visual_scene_openings=getattr(config_lib, "VISUAL_SCENE_OPENINGS"),
      buyer_premise_template=getattr(config_lib, "BUYER_PREMISE"),
      seller_premise_template=getattr(config_lib, "SELLER_PREMISE"),
      buyer_rewards=buyer_rewards,
      seller_costs=seller_costs,
      price_options=getattr(config_lib, "PRICE_OPTIONS"),
      rng=rng,
  )

  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["rational__Entity"] = rational.Entity()
  prefabs["conversation_rules__GameMaster"] = (
      dialogic_and_dramaturgic.GameMaster()
  )
  prefabs["decision_rules__GameMaster"] = (
      sequential_bargain_game_master.SequentialBargainGameMaster()
  )
  prefabs["puppet__Entity"] = puppet.Entity()

  focal_player_prefab = getattr(
      config_lib, "FOCAL_PLAYER_PREFAB", "basic__Entity"
  )

  instances = []
  for name in people:
    params = {"name": name}
    # Add fixed_responses for puppet prefabs to enable deterministic testing
    if focal_player_prefab == "puppet__Entity":
      params["fixed_responses"] = {
          getattr(config_lib, "CALL_TO_SPEECH").format(
              name=name
          ): "Let's make a deal!",
          getattr(config_lib, "CALL_TO_PROPOSE").format(name=name): "3 coins",
          getattr(config_lib, "CALL_TO_ACCEPT").format(name=name): "accept",
      }
      params["goal"] = f"{name} wants to make a profitable deal."
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=focal_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params=params,
        )
    )

  # Create supporting players with config-driven fixed responses
  supporting_fixed_responses = getattr(
      config_lib, "SUPPORTING_PLAYER_FIXED_RESPONSES", {}
  )
  supporting_memories = getattr(config_lib, "SUPPORTING_PLAYER_MEMORIES", [])

  for name in supporting_players:
    fixed_responses = {}
    for key, value in supporting_fixed_responses.items():
      fixed_responses[key.format(name=name)] = value

    call_to_speech = getattr(config_lib, "CALL_TO_SPEECH")
    if call_to_speech.format(name=name) not in fixed_responses:
      fixed_responses[call_to_speech.format(name=name)] = "Let's make a deal!"

    formatted_memories = [m.format(name=name) for m in supporting_memories]

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="puppet__Entity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "fixed_responses": fixed_responses,
                "goal": f"{name} wants to make a profitable deal.",
                "extras": {"specific_memories": formatted_memories},
            },
        )
    )

  shared_memories = getattr(config_lib, "SHARED_MEMORIES", [])
  shared_memories = shared_memories + [
      (
          f"It is {getattr(config_lib, 'YEAR')}, "
          f"{getattr(config_lib, 'MONTH')}, {getattr(config_lib, 'DAY')} in "
          f"{getattr(config_lib, 'LOCATION')}."
      ),
      getattr(config_lib, "SCENARIO_PREMISE").format(
          year=getattr(config_lib, "YEAR"),
          location=getattr(config_lib, "LOCATION"),
          event=getattr(config_lib, "EVENT"),
      ),
  ]

  player_specific_memories = {}
  for name in all_active_players:
    player_specific_memories[name] = [
        f"{name} always drives a hard bargain.",
        (
            f"{name} is a travelling merchant. Their business is buying and "
            "selling fruit."
        ),
    ]

  instances.append(
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={
              "name": "initial setup rules",
              "next_game_master_name": "conversation rules",
              "shared_memories": shared_memories,
              "player_specific_memories": player_specific_memories,
          },
      )
  )

  # Create cumulative payoff tracker
  cumulative_payoff = CumulativePayoff()

  # Build a registry of seller costs for dynamic lookup
  seller_costs_registry = {}
  for i, (_, seller_name) in enumerate(pairs):
    seller_costs_registry[seller_name] = seller_costs[i]

  # Create a payoff handler for the first game
  first_buyer, first_seller = pairs[0]
  first_payoff = HagglingPayoff(
      buyer_name=first_buyer,
      seller_name=first_seller,
      buyer_base_reward=buyer_rewards[0],
      seller_base_reward=seller_costs[0],
  )
  cumulative_payoff.add_game_payoff(first_payoff)

  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "conversation rules",
              "scenes": scenes,
          },
      )
  )

  instances.append(
      prefab_lib.InstanceConfig(
          prefab="decision_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "decision rules",
              "scenes": scenes,
              "buyer_name": first_buyer,
              "seller_name": first_seller,
              "action_to_scores": first_payoff.action_to_scores,
              "scores_to_observation": first_payoff.scores_to_observation,
              "seller_costs_registry": seller_costs_registry,
          },
      )
  )

  config = prefab_lib.Config(
      default_premise=getattr(config_lib, "SCENARIO_PREMISE").format(
          year=getattr(config_lib, "YEAR"),
          location=getattr(config_lib, "LOCATION"),
          event=getattr(config_lib, "EVENT"),
      ),
      default_max_steps=getattr(config_lib, "MAX_STEPS", 100),
      prefabs=prefabs,
      instances=instances,
  )

  logging.set_verbosity(logging.INFO)

  sim = simulation_lib.Simulation(
      config=config,
      model=model,
      embedder=embedder,
  )

  structured_log = sim.play()

  # Extract final results
  joint_action = first_payoff.latest_joint_action
  all_scores = cumulative_payoff.get_total_scores()

  focal_scores = {name: all_scores.get(name, 0.0) for name in people}
  background_scores = {
      name: all_scores.get(name, 0.0) for name in supporting_players
  }

  return {
      "joint_action": joint_action,
      "scores": all_scores,
      "focal_scores": focal_scores,
      "background_scores": background_scores,
      "focal_players": list(people),
      "background_players": list(supporting_players),
      "players": list(people),
      "pairs": pairs,
      "payoff": first_payoff,
      "structured_log": structured_log,
  }
