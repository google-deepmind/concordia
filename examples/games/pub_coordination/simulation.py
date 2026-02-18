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

"""A Pub Coordination simulation using Concordia prefabs."""

import collections
import random
from typing import Any, Callable, Mapping, Sequence

from absl import logging
from examples.games.pub_coordination import social_data
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import puppet
from concordia.prefabs.entity import rational
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions


class PubCoordinationPayoff:
  """A class to handle scoring and detailed outcome observations."""

  def __init__(
      self,
      player_names: Sequence[str],
      person_preferences: Mapping[str, str],
      player_multipliers: Mapping[str, Mapping[str, float]],
      option_multipliers: Mapping[str, float],
      relational_matrix: Mapping[str, Mapping[str, float]],
  ):
    self._player_names = player_names
    self._person_preferences = person_preferences
    self._player_multipliers = player_multipliers
    self._option_multipliers = option_multipliers
    self._relational_matrix = relational_matrix
    self._latest_joint_action = {}

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    """Returns the latest joint action passed to action_to_scores."""
    return self._latest_joint_action

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Maps joint actions to scores for each player."""
    self._latest_joint_action = joint_action
    scores = {}
    for player in self._player_names:
      choice = joint_action.get(player)
      if (
          choice in self._option_multipliers
          and self._option_multipliers[choice] == 0.0
      ):
        scores[player] = 0.0
        continue

      score = 0.0
      # Match favorite pub
      score += 0.5 if choice == self._person_preferences.get(player) else 0.0

      # Social score based on friends (relational matrix)
      same_choice_by_relation = 0.0
      for other_player, other_choice in joint_action.items():
        if player != other_player and choice == other_choice:
          same_choice_by_relation += self._relational_matrix[player][
              other_player
          ]

      max_social_score = (
          sum(self._relational_matrix[player].values())
          - self._relational_matrix[player][player]
      )
      if max_social_score > 0:
        score += same_choice_by_relation / max_social_score

      scores[player] = score
    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Maps scores back to descriptive observations for each player."""
    joint_action = self._latest_joint_action

    players_by_choice = collections.defaultdict(list)
    for name, choice in joint_action.items():
      players_by_choice[choice].append(name)

    summary_of_attendance = ""
    for choice, attendees in players_by_choice.items():
      summary_of_attendance += f"{', '.join(attendees)} went to {choice}. "

    results = {}
    for player in self._player_names:
      choice = joint_action.get(player)
      score = scores.get(player, 0.0)
      was_pub_closed = self._option_multipliers.get(choice) == 0.0

      # Sentiment
      if score > 0.9:
        enjoyment = f"Overall, {player} had a great time watching the game!"
      elif score > 0.5:
        enjoyment = f"Overall, {player} had an ok time watching the game."
      elif score < 1e-8:
        enjoyment = f"Overall, {player} had the worst time ever."
      else:
        enjoyment = f"Overall, {player} had a bad time watching the game."

      # Social feedback
      same_choice_by_relation = 0.0
      for other_player, other_choice in joint_action.items():
        if player != other_player and choice == other_choice:
          same_choice_by_relation += self._relational_matrix[player][
              other_player
          ]

      max_social_score = (
          sum(self._relational_matrix[player].values())
          - self._relational_matrix[player][player]
      )
      if same_choice_by_relation == max_social_score:
        friends_attendance = (
            f"All of {player}'s friends went to the same pub! It couldn't have"
            " been better."
        )
      elif same_choice_by_relation > 0.5 * max_social_score:
        friends_attendance = (
            "It could have been better if more friends showed up."
        )
      elif same_choice_by_relation > 0.0:
        friends_attendance = (
            f"{player} would have been a lot happier if more of their friends"
            " had shown up."
        )
      else:
        friends_attendance = (
            f"None of {player}'s friends showed up, it couldn't have been"
            " worse!"
        )

      # Choice feedback
      if was_pub_closed:
        choice_feedback = f"{player} went to a closed pub."
      elif choice == self._person_preferences.get(player):
        choice_feedback = f"{player} watched the game at their favorite pub."
      else:
        choice_feedback = (
            f"{player} watched the game at a pub that is not their favorite."
        )

      results[player] = (
          f"{summary_of_attendance} {choice_feedback} {friends_attendance}"
          f" {enjoyment}"
      )
    return results


def configure_scenes(
    venues: Sequence[str],
    people: Sequence[str],
    conversation_call_to_action: str,
    decision_call_to_action: str,
    conversation_premise: str,
    decision_premise: str,
    social_contexts: Sequence[str],
    relationship_statements: Mapping[str, Sequence[str]],
    game_countries: Sequence[str],
    num_games: int,
    rng: random.Random,
    pub_closed_probability: float = 0.0,
) -> tuple[Sequence[scene_lib.SceneSpec], Sequence[list[str]]]:
  """Configures the scenes for the simulation.

  Args:
    venues: The list of venue names.
    people: The list of player names.
    conversation_call_to_action: The call to action for conversation scenes.
    decision_call_to_action: The call to action for decision scenes.
    conversation_premise: The premise template for conversation scenes.
    decision_premise: The premise template for decision scenes.
    social_contexts: Social context templates.
    relationship_statements: Per-player relationship statements.
    game_countries: Countries that can play in games.
    num_games: Number of games/rounds to simulate.
    rng: Random number generator.
    pub_closed_probability: Probability a pub is closed each round.

  Returns:
    A tuple of (scenes, closures_per_game) where closures_per_game[i] is the
    list of closed venues for game i (may be empty).
  """

  def _get_decision_premise(name: str):
    return decision_premise.format(name=name)

  scenes = []
  closures_per_game = []

  games = []
  countries = list(game_countries)

  for _ in range(num_games):
    c1 = rng.choice(countries)
    c2 = rng.choice([c for c in countries if c != c1])
    games.append((c1, c2))

  for game_idx, (c1, c2) in enumerate(games):
    # Roll for pub closure each game (matches original behavior)
    closed_this_game = []
    if rng.random() < pub_closed_probability:
      closed_this_game = [rng.choice(venues)]
    closures_per_game.append(closed_this_game)

    # Select shared social context for this game (all players see same scene)
    shared_context = rng.choice(social_contexts)

    conversation_scene_type = scene_lib.SceneTypeSpec(
        name=f"conversation_game_{game_idx}",
        game_master_name="conversation rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=conversation_call_to_action,
        ),
    )

    # Build premise with closure info for first player if applicable
    premise = {}
    for i, name in enumerate(people):
      context = shared_context.format(name=name)
      relationships = "\n".join(relationship_statements[name])
      conversation_context = (
          f"{context}\n"
          f"{conversation_premise.format(name=name)}\n"
          f"Relationships:\n{relationships}"
      )
      player_premise = [
          f"It is game {game_idx + 1}. {c1} is playing against {c2}.",
          conversation_context,
      ]
      if i == 0 and closed_this_game:
        player_premise.append(
            f"{name} has heard that {closed_this_game[0]} might be closed"
            " today. Going there would be a bad idea."
        )
      premise[name] = player_premise

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=conversation_scene_type,
            participants=people,
            num_rounds=2 * len(people),
            premise=premise,
        )
    )

    decision_scene_type = scene_lib.SceneTypeSpec(
        name=f"decision_game_{game_idx}",
        game_master_name="decision rules",
        action_spec=entity_lib.choice_action_spec(
            call_to_action=decision_call_to_action,
            options=venues,
            tag="decision",
        ),
    )

    decision_premise_dict = {
        name: [_get_decision_premise(name)] for name in people
    }

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=decision_scene_type,
            participants=people,
            num_rounds=len(people),
            premise=decision_premise_dict,
        )
    )

  return scenes, closures_per_game


def sample_parameters(
    venue_preferences: Mapping[str, Sequence[str]],
    male_names: Sequence[str],
    female_names: Sequence[str],
    num_venues: int,
    num_people: int,
    seed: int | None = None,
):
  """Samples venues, people, and a random number generator.

  Args:
    venue_preferences: A mapping from venue names to reasons for liking them.
    male_names: A sequence of male names to sample from.
    female_names: A sequence of female names to sample from.
    num_venues: The number of venues to sample.
    num_people: The number of people to sample.
    seed: Optional seed for the random number generator.

  Returns:
    A tuple containing:
      - venues: A list of sampled venue names.
      - people: A list of sampled person names.
      - rng: The random number generator used.
  """
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)
  venues = rng.sample(list(venue_preferences.keys()), num_venues)
  all_names = list(male_names) + list(female_names)
  rng.shuffle(all_names)
  people = all_names[:num_people]
  return venues, people, rng


def sample_symmetric_relationship_matrix(
    names: Sequence[str],
    rng: random.Random,
) -> Mapping[str, Mapping[str, float]]:
  """Samples a symmetric relationship matrix for a group of people.

  1.0 indicates a friendship/positive relationship, 0.0 indicates neutral/none.
  The matrix is symmetric: m[a][b] == m[b][a].
  Diagonal elements (self-relationship) are always 1.0.

  Args:
    names: The names of the people to generate relationships for.
    rng: The random number generator to use.

  Returns:
    A dictionary mapping each name to a dictionary of their relationships with
    others.
  """
  m = {}
  for a in names:
    m[a] = {}
    for b in names:
      if a == b:
        m[a][b] = 1.0  # Diagonal elements are 1
      elif b in m and a in m[b]:
        m[a][b] = m[b][a]  # Ensure symmetry
      else:
        m[a][b] = rng.choice([0.0, 1.0])
  return m


def generate_relationship_statements(
    names: Sequence[str],
    m: Mapping[str, Mapping[str, float]],
    rng: random.Random,
) -> Mapping[str, Sequence[str]]:
  """Generates text descriptions of relationships for each person.

  Args:
    names: The names of the people.
    m: The relationship matrix (m[a][b] is the relationship value).
    rng: The random number generator to use for selecting statement templates.

  Returns:
    A dictionary mapping each persons name to a list of relationship statements
    describing their feelings towards others.
  """
  relationship_statements = {}
  for a in names:
    statements = []
    for b in names:
      if a != b:
        if m[a][b] > 0.0:
          statement = rng.choice(social_data.POSITIVE_RELATIONSHIP_STATEMENTS)
          statement = statement.format(player_a=a, player_b=b)
          statements.append(statement)
        elif m[a][b] == 0.0:
          statement = rng.choice(social_data.NEUTRAL_RELATIONSHIP_STATEMENTS)
          statement = statement.format(player_a=a, player_b=b)
          statements.append(statement)
    relationship_statements[a] = statements
  return relationship_statements


def run_simulation(
    config: Any,
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
) -> dict[str, Any] | None:
  """Run the Pub Coordination simulation.

  Args:
    config: The scenario configuration module.
    model: The language model to use.
    embedder: The sentence embedder to use.

  Returns:
    A dictionary containing the simulation results.
  """
  config_lib = config

  num_main = getattr(config_lib, "NUM_MAIN_PLAYERS", 4)
  num_background = getattr(config_lib, "NUM_BACKGROUND_PLAYERS", 2)
  num_supporting = getattr(config_lib, "NUM_SUPPORTING_PLAYERS", 0)
  focal_player_prefab = getattr(
      config_lib, "FOCAL_PLAYER_PREFAB", "basic__Entity"
  )
  background_player_prefab = getattr(
      config_lib, "BACKGROUND_PLAYER_PREFAB", "rational__Entity"
  )

  venues, people, rng = sample_parameters(
      venue_preferences=getattr(config_lib, "VENUE_PREFERENCES"),
      male_names=getattr(config_lib, "MALE_NAMES"),
      female_names=getattr(config_lib, "FEMALE_NAMES"),
      num_venues=getattr(config_lib, "NUM_VENUES"),
      num_people=num_main + num_background + num_supporting,
  )

  # Get closure probability from config (closures are rolled per-game in
  # configure_scenes)
  prob_closed = getattr(config_lib, "PUB_CLOSED_PROBABILITY", 0.0)

  # Use custom relationship matrix if config provides one
  use_custom = getattr(config_lib, "USE_CUSTOM_RELATIONSHIPS", False)
  if use_custom and hasattr(config_lib, "make_tough_friendship_matrix"):
    relational_matrix = config_lib.make_tough_friendship_matrix(people)
  else:
    relational_matrix = sample_symmetric_relationship_matrix(people, rng)

  relationship_statements = generate_relationship_statements(
      people, relational_matrix, rng
  )

  # Split into focal, background, and supporting players
  focal_players = people[:num_main]
  background_players = people[num_main : num_main + num_background]
  supporting_players = people[num_main + num_background :]
  all_players = focal_players + background_players + supporting_players

  # Rich agent configuration
  social_classes = ["working", "middle", "upper"]
  player_multipliers = {}
  player_specific_memories = {}
  person_preferences = {}

  # Handle custom preference assignment (for tough friendship scenario)
  focal_venue_idx = getattr(config_lib, "FOCAL_PREFERS_VENUE_INDEX", None)
  friend_venue_idx = getattr(config_lib, "FRIEND_PREFERS_VENUE_INDEX", None)

  for idx, name in enumerate(all_players):
    # Use config-defined preference if available, else random
    person_preferences_config = getattr(config_lib, "PERSON_PREFERENCES", {})
    if name in person_preferences_config:
      fav_pub = person_preferences_config[name]
    elif (
        idx == 0
        and focal_venue_idx is not None
        and len(venues) > focal_venue_idx
    ):
      fav_pub = venues[focal_venue_idx]
    elif (
        idx == 1
        and friend_venue_idx is not None
        and len(venues) > friend_venue_idx
    ):
      fav_pub = venues[friend_venue_idx]

    else:
      fav_pub = rng.choice(venues)
    person_preferences[name] = fav_pub

    social_class = rng.choice(social_classes)
    reasons = getattr(config_lib, "VENUE_PREFERENCES")[fav_pub]

    # Matching the original's multiplier logic
    player_multipliers[name] = {v: 1.0 if v == fav_pub else 0.8 for v in venues}

    trait = social_data.get_trait(flowery=True, rng=rng)
    goal = (
        f"Have a good time. To have a good time, {name} would like to "
        "watch the game in the same pub as their friends. "
        f"{name} would prefer everyone went to {fav_pub}."
    )

    mems = [
        f"{name} is a member of the {social_class} class.",
        f"{name}'s favorite pub is {fav_pub}.",
        f"{name} likes {fav_pub} because: {reasons}",
        f"{name}'s personality is like {trait}",
        f"[goal] {goal}",
    ]

    player_specific_memories[name] = mems

  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["rational__Entity"] = rational.Entity()

  prefabs["conversation_rules__GameMaster"] = (
      dialogic_and_dramaturgic.GameMaster()
  )
  prefabs["decision_rules__GameMaster"] = (
      game_theoretic_and_dramaturgic.GameMaster()
  )
  prefabs["puppet__Entity"] = puppet.Entity()

  # Configure scenes (closure logic is now per-game inside configure_scenes)
  scenes, closures_per_game = configure_scenes(
      venues=venues,
      people=all_players,
      conversation_call_to_action=getattr(config_lib, "CALL_TO_SPEECH"),
      decision_call_to_action=getattr(config_lib, "CALL_TO_DECISION"),
      conversation_premise=getattr(config_lib, "CONVERSATION_PREMISE"),
      decision_premise=getattr(config_lib, "DECISION_PREMISE"),
      social_contexts=getattr(config_lib, "SOCIAL_CONTEXTS"),
      relationship_statements=relationship_statements,
      game_countries=getattr(config_lib, "GAME_COUNTRIES"),
      num_games=getattr(config_lib, "NUM_GAMES", 1),
      rng=rng,
      pub_closed_probability=prob_closed,
  )

  # For the payoff, use the first game's closure (matching the single-payoff
  # architecture)
  closed_venues = closures_per_game[0] if closures_per_game else []

  if focal_player_prefab:
    pass  # Use the provided prefab
  else:
    focal_player_prefab = getattr(
        config_lib, "FOCAL_PLAYER_PREFAB", "basic__Entity"
    )

  instances = []
  for name in focal_players:
    fav_pub = person_preferences[name]
    params = {"name": name}
    # Add fixed_responses for puppet prefabs to enable deterministic testing
    if focal_player_prefab == "puppet__Entity":
      params["fixed_responses"] = {
          getattr(config_lib, "CALL_TO_SPEECH").format(name=name): fav_pub,
          getattr(config_lib, "CALL_TO_DECISION").format(name=name): fav_pub,
      }
      params["goal"] = f"{name} would prefer everyone went to {fav_pub}."
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=focal_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params=params,
        )
    )

  for name in background_players:
    fav_pub = person_preferences[name]
    goal = f"{name} would prefer everyone went to {fav_pub}."
    instances.append(
        prefab_lib.InstanceConfig(
            prefab=background_player_prefab,
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": goal,
            },
        )
    )

  for name in supporting_players:
    fav_pub = person_preferences[name]
    goal = f"{name} would prefer everyone went to {fav_pub}."
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="puppet__Entity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "fixed_responses": {
                    getattr(config_lib, "CALL_TO_SPEECH").format(
                        name=name
                    ): fav_pub,
                    getattr(config_lib, "CALL_TO_DECISION").format(
                        name=name
                    ): fav_pub,
                },
                "goal": goal,
            },
        )
    )

  shared_memories = [
      (
          f"It is {getattr(config_lib, 'YEAR')},"
          f" {getattr(config_lib, 'MONTH')}, {getattr(config_lib, 'DAY')} in"
          f" {getattr(config_lib, 'LOCATION')}."
      ),
      getattr(config_lib, "SCENARIO_PREMISE").format(
          year=getattr(config_lib, "YEAR"),
          location=getattr(config_lib, "LOCATION"),
          event=getattr(config_lib, "EVENT"),
      ),
      f"The available venues are: {', '.join(venues)}.",
  ]
  player_specific_memories = {}
  for i, name in enumerate(people):
    prefs = person_preferences[name]
    reasons = rng.choice(getattr(config_lib, "VENUE_PREFERENCES")[prefs])
    mems = [
        f"{name}'s favorite venue is {prefs}.",
        f"{name} likes {prefs} because: {reasons}",
        f"{name} wants to watch the game with friends.",
    ]
    if i == 0 and closed_venues:
      mems.append(
          f"{name} has heard that {closed_venues[0]} might be closed today."
      )
    player_specific_memories[name] = mems

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

  payoff = PubCoordinationPayoff(
      player_names=list(people),
      person_preferences=person_preferences,
      player_multipliers={
          name: {"preference": 1.0} for name in people
      },  # Simple default
      option_multipliers={
          venue: 0.0 if venue in closed_venues else 1.0 for venue in venues
      },
      relational_matrix=relational_matrix,
  )

  instances.append(
      prefab_lib.InstanceConfig(
          prefab="decision_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "decision rules",
              "scenes": scenes,
              "action_to_scores": payoff.action_to_scores,
              "scores_to_observation": payoff.scores_to_observation,
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

  # Compute final scores
  joint_action = payoff.latest_joint_action
  if joint_action:
    all_scores = payoff.action_to_scores(joint_action)
  else:
    all_scores = {}

  return {
      "focal_scores": {
          name: all_scores.get(name, 0.0) for name in focal_players
      },
      "background_scores": {
          name: all_scores.get(name, 0.0) for name in background_players
      },
      "joint_action": joint_action,
      "payoff": payoff,
      "focal_players": focal_players,
      "background_players": background_players,
      "structured_log": structured_log,
  }
