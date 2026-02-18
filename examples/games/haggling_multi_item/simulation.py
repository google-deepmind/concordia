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

"""A multi-item Haggling simulation using Concordia prefabs.

This simulates a fruit market where buyers and sellers negotiate prices for
specific items (e.g., apple, banana, pear). The buyer proposes an item and
price, and the seller accepts or rejects.
"""

import collections
import datetime
import random
import re
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


def parse_item_and_price(action: str) -> tuple[str | None, float]:
  """Parse an action like 'apple for 3 coins' into (item, price).

  Args:
    action: The action string to parse.

  Returns:
    A tuple of (item_name, price_value). Returns (None, 0.0) if parsing fails.
  """
  match = re.match(r"(\w+)\s+for\s+(\d+)\s+coins?", action.lower())
  if match:
    item = match.group(1)
    price = float(match.group(2))
    return item, price
  return None, 0.0


class MultiItemHagglingPayoff:
  """A class to handle scoring for multi-item bargaining games."""

  def __init__(
      self,
      buyer_name: str,
      seller_name: str,
      buyer_base_reward_per_item: Mapping[str, float],
      seller_base_reward_per_item: Mapping[str, float],
  ):
    """Initialize the payoff handler.

    Args:
      buyer_name: Name of the buyer player.
      seller_name: Name of the seller player.
      buyer_base_reward_per_item: How much the buyer can resell each item for.
      seller_base_reward_per_item: Cost for the seller to acquire each item.
    """
    self._buyer_name = buyer_name
    self._seller_name = seller_name
    self._buyer_base_reward_per_item = buyer_base_reward_per_item
    self._seller_base_reward_per_item = seller_base_reward_per_item
    self._latest_joint_action = {}

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores for each player.

    Dynamically identifies buyer/seller from action content:
    - Buyer's action is "{item} for {price} coins"
    - Seller's action is "accept" or "reject"

    Args:
      joint_action: A mapping from player name to their action.

    Returns:
      A mapping from player name to their score.
    """
    self._latest_joint_action = joint_action

    # Dynamically identify buyer and seller from action content
    buyer_name = None
    seller_name = None
    buyer_action = None
    seller_action = None

    for name, action in joint_action.items():
      if action is None:
        continue
      action_lower = action.lower()
      if "accept" in action_lower or "reject" in action_lower:
        seller_name = name
        seller_action = action
      elif "for" in action_lower and "coin" in action_lower:
        buyer_name = name
        buyer_action = action

    # If we couldn't identify both parties, return zeros for configured players
    if buyer_name is None or seller_name is None:
      return {self._buyer_name: 0.0, self._seller_name: 0.0}

    # Update internal names for observation generation
    self._buyer_name = buyer_name
    self._seller_name = seller_name

    if "reject" in seller_action.lower():
      return {buyer_name: 0.0, seller_name: 0.0}

    item, price = parse_item_and_price(buyer_action)
    if item is None:
      return {buyer_name: 0.0, seller_name: 0.0}

    buyer_base = self._buyer_base_reward_per_item.get(item, 0.0)
    seller_base = self._seller_base_reward_per_item.get(item, 0.0)

    buyer_score = buyer_base - price
    seller_score = price - seller_base

    return {buyer_name: buyer_score, seller_name: seller_score}

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Maps scores back to descriptive observations for each player."""
    joint_action = self._latest_joint_action
    seller_action = joint_action.get(self._seller_name)
    buyer_action = joint_action.get(self._buyer_name)

    if seller_action is None:
      outcome = " were not involved in this deal."
    elif "reject" in seller_action.lower():
      outcome = " couldn't agree on a price and the deal fell through."
    else:
      item, price = parse_item_and_price(buyer_action or "")
      if item:
        outcome = (
            f" agreed on a price of {int(price)} coins for {item} and the deal"
            " was successful!"
        )
      else:
        outcome = " agreed on a price and the deal was successful!"

    buyer_and_seller = f"{self._buyer_name} and {self._seller_name}"

    results = {}
    for name, score in scores.items():
      if score > 0:
        results[name] = (
            f"{buyer_and_seller}{outcome} {name} stands to make a profit of "
            f"{score} coins from the deal."
        )
      elif score == 0:
        results[name] = (
            f"{buyer_and_seller}{outcome} {name} breaks even on this deal."
        )
      else:
        results[name] = (
            f"{buyer_and_seller}{outcome} However, {name} stands to lose "
            f"{-score} coins from the deal."
        )
    return results


class CumulativePayoff:
  """Accumulates payoffs across multiple games."""

  def __init__(self):
    self._game_payoffs: list[MultiItemHagglingPayoff] = []

  def add_game_payoff(self, payoff: MultiItemHagglingPayoff) -> None:
    """Add a game payoff handler."""
    self._game_payoffs.append(payoff)

  def get_total_scores(self) -> Mapping[str, float]:
    """Get the total scores across all games."""
    totals = collections.defaultdict(float)
    for payoff in self._game_payoffs:
      if payoff.latest_joint_action:
        scores = payoff.action_to_scores(payoff.latest_joint_action)
        for name, score in scores.items():
          totals[name] += score
    return dict(totals)


def sample_parameters(
    male_names: Sequence[str],
    female_names: Sequence[str],
    num_people: int,
    seed: int | None = None,
):
  """Samples player names and creates random number generator."""
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)
  all_names = list(male_names) + list(female_names)
  rng.shuffle(all_names)
  people = all_names[:num_people]
  return people, rng


def create_player_pairs(
    players: Sequence[str], rng: random.Random
) -> list[tuple[str, str]]:
  """Creates buyer-seller pairs from a list of players."""
  pairs = []
  player_list = list(players)
  for i in range(len(player_list)):
    for j in range(i + 1, len(player_list)):
      if rng.choice([True, False]):
        pairs.append((player_list[i], player_list[j]))
      else:
        pairs.append((player_list[j], player_list[i]))
  return pairs


def generate_price_options(
    items: Sequence[str], prices: Sequence[int]
) -> tuple[str, ...]:
  """Generate all item x price option combinations."""
  options = []
  for item in items:
    for price in prices:
      options.append(f"{item} for {price} coins")
  return tuple(options)


def configure_scenes(
    pairs: Sequence[tuple[str, str]],
    num_games: int,
    call_to_speech: str,
    call_to_propose: str,
    call_to_accept: str,
    visual_scene_openings: Sequence[str],
    buyer_premise_template: str,
    seller_premise_template: str,
    buyer_rewards: Sequence[Mapping[str, float]],
    seller_costs: Sequence[Mapping[str, float]],
    price_options: Sequence[str],
    items_for_sale: Sequence[str],
    rng: random.Random,
) -> Sequence[scene_lib.SceneSpec]:
  """Configures the scenes for the simulation."""
  scenes = []

  for game_idx in range(num_games * len(pairs)):
    buyer_name, seller_name = pairs[game_idx % len(pairs)]
    scene_visual = rng.choice(list(visual_scene_openings))
    buyer_reward_per_item = buyer_rewards[game_idx % len(buyer_rewards)]
    seller_cost_per_item = seller_costs[game_idx % len(seller_costs)]

    buyer_reward_str = "; ".join([
        f"{item} for {buyer_reward_per_item[item]} coins"
        for item in items_for_sale
    ])
    seller_cost_str = "; ".join([
        f"{item} for {seller_cost_per_item[item]} coins"
        for item in items_for_sale
    ])

    conversation_scene_type = scene_lib.SceneTypeSpec(
        name=f"negotiation_game_{game_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=call_to_speech,
        ),
    )

    conversation_premise = {
        buyer_name: [
            buyer_premise_template.format(
                scene_visual=scene_visual,
                buyer_name=buyer_name,
                seller_name=seller_name,
                buyer_reward=buyer_reward_str,
            )
        ],
        seller_name: [
            seller_premise_template.format(
                scene_visual=scene_visual,
                buyer_name=buyer_name,
                seller_name=seller_name,
                seller_cost=seller_cost_str,
            )
        ],
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=conversation_scene_type,
            participants=[buyer_name, seller_name],
            num_rounds=2,
            premise=conversation_premise,
        )
    )

    decision_action_spec = {
        buyer_name: entity_lib.choice_action_spec(
            call_to_action=call_to_propose,
            options=price_options,
            tag="decision",
        ),
        seller_name: entity_lib.choice_action_spec(
            call_to_action=call_to_accept,
            options=("accept", "reject"),
            tag="decision",
        ),
    }

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
            num_rounds=2,
            premise=decision_premise,
        )
    )

  return scenes


def run_simulation(
    config: Any,
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
) -> dict[str, Any] | None:
  """Run the Haggling Multi-Item simulation.

  Args:
    config: The scenario configuration module.
    model: The language model to use.
    embedder: The sentence embedder to use.

  Returns:
    A dictionary containing the simulation results.
  """
  config_lib = config
  num_main = getattr(config_lib, "NUM_MAIN_PLAYERS", 2)
  num_supporting = getattr(config_lib, "NUM_SUPPORTING_PLAYERS", 0)
  num_games = getattr(config_lib, "NUM_GAMES", 2)
  items_for_sale = getattr(
      config_lib, "ITEMS_FOR_SALE", ("apple", "banana", "pear")
  )
  prices = getattr(config_lib, "PRICES", (1, 2, 3, 4, 5, 6))

  all_people, rng = sample_parameters(
      male_names=getattr(config_lib, "MALE_NAMES"),
      female_names=getattr(config_lib, "FEMALE_NAMES"),
      num_people=num_main + num_supporting,
  )

  people = all_people[:num_main]
  supporting_players = all_people[num_main:]

  only_match_with_support = getattr(
      config_lib, "ONLY_MATCH_WITH_SUPPORT", False
  )
  if supporting_players and only_match_with_support:
    pairs = []
    for main_player in people:
      for support_player in supporting_players:
        if rng.choice([True, False]):
          pairs.append((main_player, support_player))
        else:
          pairs.append((support_player, main_player))
  else:
    pairs = create_player_pairs(list(people) + list(supporting_players), rng)

  all_active_players = list(people) + list(supporting_players)

  buyer_rewards = []
  seller_costs = []
  for _ in range(num_games * len(pairs)):
    buyer_reward = {
        item: rng.randint(
            getattr(config_lib, "BUYER_BASE_REWARD_MIN", 5),
            getattr(config_lib, "BUYER_BASE_REWARD_MAX", 6),
        )
        for item in items_for_sale
    }
    seller_cost = {
        item: rng.randint(
            getattr(config_lib, "SELLER_BASE_REWARD_MIN", 1),
            getattr(config_lib, "SELLER_BASE_REWARD_MAX", 2),
        )
        for item in items_for_sale
    }
    buyer_rewards.append(buyer_reward)
    seller_costs.append(seller_cost)

  price_options = generate_price_options(items_for_sale, prices)

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
      price_options=price_options,
      items_for_sale=items_for_sale,
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
    if focal_player_prefab == "puppet__Entity":
      params["fixed_responses"] = {
          getattr(config_lib, "CALL_TO_SPEECH").format(
              name=name
          ): "Let's make a deal!",
          getattr(config_lib, "CALL_TO_PROPOSE").format(
              name=name
          ): "apple for 3 coins",
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

  cumulative_payoff = CumulativePayoff()

  first_buyer, first_seller = pairs[0]
  first_payoff = MultiItemHagglingPayoff(
      buyer_name=first_buyer,
      seller_name=first_seller,
      buyer_base_reward_per_item=buyer_rewards[0],
      seller_base_reward_per_item=seller_costs[0],
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

  joint_action = first_payoff.latest_joint_action
  # Get cumulative scores across all rounds (not just last action)
  all_scores = cumulative_payoff.get_total_scores()

  # Split scores into focal (main players) and background (supporting players)
  focal_scores = {name: all_scores.get(name, 0.0) for name in people}
  background_scores = {
      name: all_scores.get(name, 0.0) for name in supporting_players
  }

  # Add final score entries to structured log
  last_step = (
      max(structured_log.get_steps()) if structured_log.get_steps() else 0
  )
  timestamp = datetime.datetime.now().isoformat()
  for name in list(people) + list(supporting_players):
    structured_log.add_entry(
        step=last_step + 1,
        timestamp=timestamp,
        entity_name=name,
        component_name="final_scores",
        entry_type="score",
        summary=f"Final score: {all_scores.get(name, 0.0):.2f}",
        raw_data={
            "player": name,
            "score": all_scores.get(name, 0.0),
            "role": "focal" if name in people else "background",
        },
    )

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
