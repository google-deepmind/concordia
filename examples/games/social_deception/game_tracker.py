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

"""The GameTracker tracks the ground truth state of the Social Deception game."""

import dataclasses
import enum
import logging
import random
from typing import Any

from examples.games.social_deception.setup import scripts


class Alignment(enum.Enum):
  GOOD = "good"
  EVIL = "evil"


class PlayerStatus(enum.Enum):
  ALIVE = "alive"
  DEAD = "dead"


@dataclasses.dataclass
class PlayerState:
  """State tracking for a single player."""

  name: str
  role: str
  alignment: Alignment
  perceived_role: str = ""
  status: PlayerStatus = PlayerStatus.ALIVE
  is_publicly_alive: bool = True
  is_poisoned: bool = False
  is_drunk: bool = False
  is_drunk_until_dusk: bool = False
  has_spent_dead_vote: bool = False
  has_used_single_use_ability: bool = False
  has_nominated_today: bool = False
  has_been_nominated_today: bool = False

  # Role-specific mechanics
  servant_master: str | None = None
  is_seer_decoy: bool = False

  # For Registration (Outcast, Spy)
  register_as_alignment: Alignment | None = None
  register_as_role: str | None = None

  # Protection and tracking
  demon_protected: bool = False
  woke_up_tonight: bool = False

  def __post_init__(self):
    # By default, the perceived role is the actual role.
    if not self.perceived_role:
      self.perceived_role = self.role


class GameTracker:
  """Omniscient ground truth tracker of the game state."""

  def __init__(
      self,
      players: list[PlayerState],
      script: (
          scripts.BaseScript | scripts.GameScript
      ) = scripts.GameScript.BASIC_EPISTEMIC_TOWN,
  ):
    if isinstance(script, enum.Enum):
      self.script = script.value
    else:
      self.script = script

    self.players = {p.name: p for p in players}
    self.player_names = [p.name for p in players]

    self.executed_player: str | None = None
    self.last_deaths_info: list[dict[str, str]] = []
    self.deaths_today: list[str] = []

    # List of {"name": name, "source": role_name, "unblockable": bool}
    self.kill_list: list[dict[str, Any]] = []

    self.game_over_reason: str | None = None
    self.vortox_in_play: bool = False

  def get_player(self, name: str | None) -> PlayerState:
    if name is None or name not in self.players:
      raise ValueError(f"Player {name} not found in GameTracker.")
    return self.players[name]

  @property
  def townsfolk(self) -> set[str]:
    return self.script.townsfolk

  @property
  def outsiders(self) -> set[str]:
    return self.script.outsiders

  @property
  def minions(self) -> set[str]:
    return self.script.minions

  @property
  def demons(self) -> set[str]:
    return self.script.demons

  def generate_safe_bluffs(self, count: int = 3) -> list[str]:
    """Identifies good roles not in play to serve as safe bluffs for the demon."""
    roles_in_play = {p.role for p in self.players.values()}
    good_roles = self.script.townsfolk | self.script.outsiders
    available_bluffs = list(good_roles - roles_in_play)
    random.shuffle(available_bluffs)
    return available_bluffs[:count]

  def get_evil_intelligence(self) -> str:
    """Returns a summary of the Evil team's known information."""
    if self.script.is_compact_game:
      return "Compact game rules: You do not know your teammates."

    demons = [n for n in self.player_names if self.is_type(n, "Demon")]
    minions = [n for n in self.player_names if self.is_type(n, "Minion")]

    return (
        f"EVIL INTEL: Demon is {', '.join(demons)}. Minions are"
        f" {', '.join(minions)}."
    )

  def get_neighbors(self, name: str | None) -> list[PlayerState]:
    """Returns the two closest alive neighbors."""
    if name is None or name not in self.player_names:
      return []
    idx = self.player_names.index(name)
    neighbors = []

    # Check neighbors in clockwise direction
    num_players = len(self.player_names)
    curr = (idx + 1) % num_players
    while len(neighbors) < 1 and curr != idx:
      p = self.players[self.player_names[curr]]
      if p.status == PlayerStatus.ALIVE:
        neighbors.append(p)
      curr = (curr + 1) % num_players

    # Check neighbors in counter-clockwise direction
    curr = (idx - 1) % num_players
    while len(neighbors) < 2 and curr != idx:
      p = self.players[self.player_names[curr]]
      if p.status == PlayerStatus.ALIVE:
        neighbors.append(p)
      curr = (curr - 1) % num_players

    return neighbors

  def reset_daily_flags(self):
    """Resets daily nomination flags at the start of each day."""
    for p in self.players.values():
      p.has_nominated_today = False
      p.has_been_nominated_today = False
      p.woke_up_tonight = False
    self.deaths_today = []
    self.executed_player = None

  def reset_dusk_flags(self):
    """Resets 'until dusk' flags at the end of each day."""
    for p in self.players.values():
      if p.is_drunk_until_dusk:
        p.is_drunk = False
        p.is_drunk_until_dusk = False

  def kill_player(self, name: str, publicly: bool = True):
    """Kills a player."""
    if name not in self.players:
      raise ValueError(f"Player {name} not found. Could not kill.")

    p = self.players[name]
    p.status = PlayerStatus.DEAD
    self.deaths_today.append(name)
    if publicly:
      p.is_publicly_alive = False

  def resurrect_player(self, name: str):
    """Resurrects a player."""
    if name not in self.players:
      raise ValueError(f"Player {name} not found. Could not resurrect.")

    p = self.players[name]
    p.status = PlayerStatus.ALIVE
    p.is_publicly_alive = True
    p.has_spent_dead_vote = False

  def add_to_kill_list(self, name: str, source: str, unblockable: bool = False):
    self.kill_list.append(
        {"name": name, "source": source, "unblockable": unblockable}
    )

  def resolve_deaths(self) -> list[str]:
    """Resolves the kill list and returns the names of players who actually died."""
    actually_died = []
    for kill in self.kill_list:
      name = kill["name"]
      if name not in self.players:
        continue
      source = kill["source"]
      unblockable = kill["unblockable"]

      p = self.players[name]

      # The Mayor Bounce (Night attack only)
      if p.role == "Mayor" and not self.is_impaired(name):
        if random.choice([True, False]):
          alive_others = [
              n
              for n, pl in self.players.items()
              if pl.status == PlayerStatus.ALIVE and n != name
          ]
          if random.choice([True, False]) and alive_others:
            new_victim_name = random.choice(alive_others)
            name = new_victim_name
            p = self.players[name]
            logging.info("The attack bounced from Mayor to %s.", name)
          else:
            continue

      if p.status == PlayerStatus.DEAD:
        logging.info("Player %s is already dead, skipping.", name)
        continue

      can_die = True
      if not unblockable:
        if p.role == "Soldier" and source in self.script.demons:
          can_die = False
        if p.demon_protected and source in self.script.demons:
          can_die = False

      if not can_die:
        logging.info("Player %s cannot die, skipping.", name)
        continue

      self.kill_player(name)
      actually_died.append({"name": name, "source": source})

    self.kill_list = []
    self.last_deaths_info = actually_died
    return [d["name"] for d in actually_died]

  def set_poisoned(self, name: str, poisoned: bool):
    if name in self.players:
      self.players[name].is_poisoned = poisoned

  def set_drunk(self, name: str, drunk: bool):
    if name in self.players:
      self.players[name].is_drunk = drunk

  def randomize_outcast_registration(self, name: str):
    """Randomly decides if the Outcast registers as evil and/or a Minion/Demon."""
    p = self.players[name]
    if p.role != "Outcast":
      return

    if random.choice([True, False]):
      p.register_as_alignment = Alignment.EVIL
      p.register_as_role = random.choice(list(self.minions | self.demons))
    else:
      p.register_as_alignment = None
      p.register_as_role = None

  def randomize_spy_registration(self, name: str):
    """Randomly decides if the Spy registers as good and/or a Townsfolk/Outsider."""
    p = self.players[name]
    if p.role != "Spy":
      return

    if random.choice([True, False]):
      p.register_as_alignment = Alignment.GOOD
      p.register_as_role = random.choice(list(self.townsfolk | self.outsiders))
    else:
      p.register_as_alignment = None
      p.register_as_role = None

  def refresh_misregistrations(self):
    """Resets or updates registration for all relevant characters in play."""
    for p_name, p in self.players.items():
      if p.role == "Outcast":
        self.randomize_outcast_registration(p_name)
      elif p.role == "Spy":
        self.randomize_spy_registration(p_name)

  def get_alignment(self, name: str) -> Alignment:
    """Returns the alignment of the player, considering Registration."""
    p = self.players[name]
    if p.register_as_alignment is not None:
      return p.register_as_alignment
    return p.alignment

  def get_all_players_of_alignment(
      self, alignment: Alignment = Alignment.GOOD
  ) -> list[str]:
    """Returns a list of all players of a given alignment."""
    return [
        p.name
        for p in self.players.values()
        if self.get_alignment(p.name) == alignment
    ]

  def get_demons(self) -> list[PlayerState]:
    """Returns all Demon players in the game."""
    return [p for p in self.players.values() if p.role in self.script.demons]

  def get_demon(self) -> list[PlayerState]:
    """Alias for get_demons."""
    return self.get_demons()

  def get_minions(self) -> list[PlayerState]:
    """Returns all Minion players in the game."""
    return [p for p in self.players.values() if p.role in self.script.minions]

  def get_outsiders(self) -> list[PlayerState]:
    """Returns all Outsider players in the game."""
    return [p for p in self.players.values() if p.role in self.script.outsiders]

  def get_townsfolk(self) -> list[PlayerState]:
    """Returns all Townsfolk players in the game."""
    return [p for p in self.players.values() if p.role in self.script.townsfolk]

  def is_type(self, name: str, role_type: str) -> bool:
    """Checks if a player registers as a specific role type."""
    if name not in self.players:
      return False
    p = self.players[name]
    role = p.register_as_role if p.register_as_role else p.role

    if role_type == "Townsfolk":
      return role in self.script.townsfolk
    if role_type == "Outsider":
      return role in self.script.outsiders
    if role_type == "Minion":
      return role in self.script.minions
    if role_type == "Demon":
      return role in self.script.demons
    return False

  def is_impaired(self, name: str | None) -> bool:
    """Checks if a player is Drunk or Poisoned."""
    if name is None or name not in self.players:
      return False
    p = self.players[name]
    return p.is_drunk or p.is_poisoned

  def is_droisoned(self, name: str | None) -> bool:
    """Backward compatibility alias for is_impaired."""
    return self.is_impaired(name)

  def resolve_executioner_shot(
      self, executioner_name: str, target_name: str
  ) -> str:
    """The Executioner publicly shoots a player."""
    self.refresh_misregistrations()
    executioner = self.get_player(executioner_name)
    if executioner.has_used_single_use_ability:
      return "You have already used your ability."

    executioner.has_used_single_use_ability = True
    if not self.is_impaired(executioner_name) and self.is_type(
        target_name, "Demon"
    ):
      self.kill_player(target_name)
      return f"Bang! {executioner_name} shoots {target_name} and they die!"

    return f"{executioner_name} shoots {target_name}, but nothing happens."

  def resolve_slayer_shot(self, slayer_name: str, target_name: str) -> str:
    """Backward compatibility alias for resolve_executioner_shot."""
    return self.resolve_executioner_shot(slayer_name, target_name)

  def nominate_player(self, nominator_name: str, nominee_name: str) -> str:
    """Handles nomination, including the Innocent trap."""
    self.refresh_misregistrations()

    nominator = self.get_player(nominator_name)
    nominee = self.get_player(nominee_name)

    if nominator.status == PlayerStatus.DEAD:
      raise ValueError(f"{nominator_name} is dead and cannot nominate.")
    if nominator.has_nominated_today:
      raise ValueError(f"{nominator_name} has already nominated today.")
    if nominee.has_been_nominated_today:
      raise ValueError(
          f"{nominee_name} has already been nominated today. Choose someone"
          " else."
      )

    nominator.has_nominated_today = True
    nominee.has_been_nominated_today = True

    # The Innocent Trap
    if nominee.role == "Innocent" and not nominee.has_used_single_use_ability:
      nominee.has_used_single_use_ability = True
      if not self.is_impaired(nominee_name) and self.is_type(
          nominator_name, "Townsfolk"
      ):
        self.execute_player(nominator_name)
        return (
            f"{nominator_name} nominated {nominee_name} and has been"
            " instantly executed!"
        )

    return f"{nominator_name} nominates {nominee_name}!"

  def execute_player(self, target_name: str) -> str:
    """Handles execution, including Saint logic."""
    target = self.get_player(target_name)

    # The Saint Condition
    if target.role == "Saint" and not self.is_impaired(target_name):
      self.game_over_reason = "Evil wins! The Saint has been executed."
      self.kill_player(target_name)
      return f"{target_name} is executed. Game Over: {self.game_over_reason}"

    self.executed_player = target_name
    self.kill_player(target_name)
    return f"{target_name} is executed."

  def get_state(self) -> dict[str, Any]:
    """Returns the serializable state of the GameTracker."""
    return {
        "players": {
            name: dataclasses.asdict(p) for name, p in self.players.items()
        },
        "executed_player": self.executed_player,
        "last_deaths_info": list(self.last_deaths_info),
        "deaths_today": list(self.deaths_today),
        "kill_list": list(self.kill_list),
        "game_over_reason": self.game_over_reason,
    }

  def set_state(self, state: dict[str, Any]) -> None:
    """Restores the state of the GameTracker."""
    valid_fields = {f.name for f in dataclasses.fields(PlayerState)}
    for name, raw_p_dict in state["players"].items():
      p_dict = {k: v for k, v in raw_p_dict.items() if k in valid_fields}
      if isinstance(p_dict.get("alignment"), str):
        p_dict["alignment"] = Alignment(p_dict["alignment"])
      if isinstance(p_dict.get("status"), str):
        p_dict["status"] = PlayerStatus(p_dict["status"])
      if isinstance(p_dict.get("register_as_alignment"), str):
        p_dict["register_as_alignment"] = Alignment(
            p_dict["register_as_alignment"]
        )
      self.players[name] = PlayerState(**p_dict)
    self.executed_player = state.get("executed_player")
    self.last_deaths_info = list(state.get("last_deaths_info", []))
    self.deaths_today = list(state.get("deaths_today", []))
    self.kill_list = list(state.get("kill_list", []))
    self.game_over_reason = state.get("game_over_reason")
