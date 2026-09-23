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

"""Basic Epistemic Town roles for Social Deception."""

from collections.abc import Sequence
import json
import random

from examples.games.social_deception import game_tracker
from examples.games.social_deception.roles import roles as role_classes

# ==============================================================================
# Townsfolk Roles (Good Team)
# ==============================================================================


class Witness(role_classes.Townsfolk):
  """The Witness role."""

  def __init__(self):
    super().__init__("Witness", night_priority=601)

  def overhead_introduction(self) -> str:
    return (
        "At the start of the game, you learn that 1 of 2 specific players is a"
        " particular Townsfolk."
    )

  def player_introduction(self) -> str:
    return (
        "You begin the game learning that one of two named players holds a"
        " specific Townsfolk role. On the first night, the Game Master reveals"
        " two candidate players and one Townsfolk identity; exactly one of"
        " those two players holds that role (unless information is impaired)."
        " You receive this information once at the start of the game."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    candidates = [p for p in tracker_ref.players if p != self.player_name]
    two_random_players = (
        random.sample(candidates, 2) if len(candidates) >= 2 else candidates * 2
    )
    townsfolk_roles = [r for r in tracker_ref.townsfolk if r != self.role_name]
    random_role = (
        random.choice(townsfolk_roles) if townsfolk_roles else "Townsfolk"
    )
    names_str = f"{two_random_players[0]} or {two_random_players[1]}"
    return f"The Game Master reveals that {random_role} is {names_str}."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num > 1:
      return ""
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    valid_players = [p for p in players if p in tracker_ref.players]
    get_townsfolk = [
        p
        for p in valid_players
        if tracker_ref.is_type(p, "Townsfolk") and p != self.player_name
    ]

    if not get_townsfolk:
      return "The Game Master reveals '0'. There are 0 Townsfolk in play."

    random_townsfolk = random.choice(get_townsfolk)
    other_candidates = [
        p
        for p in valid_players
        if p != self.player_name and p != random_townsfolk
    ]
    random_other_player = (
        random.choice(other_candidates)
        if other_candidates
        else random_townsfolk
    )
    townsfolk_role = tracker_ref.players[random_townsfolk].role
    names_list = [random_townsfolk, random_other_player]
    random.shuffle(names_list)
    names_str = f"{names_list[0]} or {names_list[1]}"

    return (
        f"The Game Master reveals that {townsfolk_role} is one of: {names_str}"
    )


class Researcher(role_classes.Townsfolk):
  """The Researcher role."""

  def __init__(self):
    super().__init__("Researcher", night_priority=602)

  def overhead_introduction(self) -> str:
    return (
        "At the start of the game, you learn that 1 of 2 players is a"
        " particular Outsider, or that zero are in play."
    )

  def player_introduction(self) -> str:
    return (
        "You begin the game learning that one of two named players holds a"
        " specific Outsider role, or you learn that zero Outsiders are in play."
        " On the first night, if Outsiders exist, the Game Master presents two"
        " player names and one Outsider identity. If no Outsiders exist, you"
        " receive a count of zero."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    candidates = [p for p in tracker_ref.players if p != self.player_name]
    two_random_players = (
        random.sample(candidates, 2) if len(candidates) >= 2 else candidates * 2
    )
    random_role = random.choice(
        [r for r in tracker_ref.outsiders if r != self.role_name]
    )
    names_str = f"{two_random_players[0]} or {two_random_players[1]}"
    return f"The Game Master reveals that {random_role} is one of: {names_str}"

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num > 1:
      return ""
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    valid_players = [p for p in players if p in tracker_ref.players]
    get_outsiders = [
        p
        for p in valid_players
        if tracker_ref.is_type(p, "Outsider") and p != self.player_name
    ]

    if not get_outsiders:
      return "The Game Master reveals '0'. There are 0 Outsiders in play."

    random_outsider = random.choice(get_outsiders)
    other_candidates = [
        p
        for p in valid_players
        if p != self.player_name and p != random_outsider
    ]
    random_other_player = (
        random.choice(other_candidates) if other_candidates else random_outsider
    )

    outsider_role = tracker_ref.players[random_outsider].role
    names_list = [random_outsider, random_other_player]
    random.shuffle(names_list)
    names_str = f"{names_list[0]} or {names_list[1]}"

    return (
        f"The Game Master reveals that {outsider_role} is one of: {names_str}"
    )


class Investigator(role_classes.Townsfolk):
  """The Investigator role."""

  def __init__(self):
    super().__init__("Investigator", night_priority=603)

  def overhead_introduction(self) -> str:
    return (
        "At the start of the game, you learn that 1 of 2 players is a"
        " particular Minion."
    )

  def player_introduction(self) -> str:
    return (
        "You begin the game learning that one of two named players holds a"
        " specific Minion role. On the first night, the Game Master reveals"
        " two candidate players and one Minion identity; exactly one of those"
        " two players holds that Minion role (unless false registration or"
        " impairment applies)."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    candidates = [p for p in tracker_ref.players if p != self.player_name]
    two_random_players = (
        random.sample(candidates, 2) if len(candidates) >= 2 else candidates * 2
    )
    random_role = random.choice(
        [r for r in tracker_ref.minions if r != self.role_name]
    )
    names_str = f"{two_random_players[0]} or {two_random_players[1]}"
    return f"The Game Master reveals that {random_role} is one of: {names_str}"

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num > 1:
      return ""
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    valid_players = [p for p in players if p in tracker_ref.players]
    get_minions = [
        p
        for p in valid_players
        if tracker_ref.is_type(p, "Minion") and p != self.player_name
    ]

    if not get_minions:
      return "The Game Master reveals '0'. There are 0 Minions in play."

    random_minion = random.choice(get_minions)
    other_candidates = [
        p for p in valid_players if p != self.player_name and p != random_minion
    ]
    random_other_player = (
        random.choice(other_candidates) if other_candidates else random_minion
    )

    minion_role = tracker_ref.players[random_minion].role
    names_list = [random_minion, random_other_player]
    random.shuffle(names_list)
    names_str = f"{names_list[0]} or {names_list[1]}"

    return f"The Game Master reveals that {minion_role} is one of: {names_str}"


class Matchmaker(role_classes.Townsfolk):
  """The Matchmaker role."""

  def __init__(self):
    super().__init__("Matchmaker", night_priority=604)

  def overhead_introduction(self) -> str:
    return (
        "At the start of the game, you learn how many adjacent pairs of evil"
        " players exist."
    )

  def player_introduction(self) -> str:
    return (
        "You begin the game knowing the total number of adjacent pairs of evil"
        " players in the seated seating circle. Two evil players seated next"
        " to each other count as one pair; three consecutive evil players count"
        " as two pairs."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    potential_evil_count = 0
    for p_name in tracker_ref.players:
      if tracker_ref.get_alignment(p_name) == game_tracker.Alignment.EVIL:
        potential_evil_count += 1

    num_players = len(tracker_ref.players)
    if potential_evil_count >= num_players:
      max_plausible = max(0, num_players)
    else:
      max_plausible = max(0, potential_evil_count - 1)

    count = random.randint(0, max_plausible)
    return f"You sense {count} pairs of evil players."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num > 1:
      return ""
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    num_players = len(players)
    if num_players < 2:
      return "You sense 0 pairs of evil players."

    count = 0
    for i in range(num_players):
      p1 = players[i]
      p2 = players[(i + 1) % num_players]
      if (
          tracker_ref.get_alignment(p1) == game_tracker.Alignment.EVIL
          and tracker_ref.get_alignment(p2) == game_tracker.Alignment.EVIL
      ):
        count += 1

    return f"You sense {count} pairs of evil players."


class Empath(role_classes.Townsfolk):
  """The Empath role."""

  def __init__(self):
    super().__init__("Empath", night_priority=607)

  def overhead_introduction(self) -> str:
    return "Each night, you learn how many of your 2 living neighbors are evil."

  def player_introduction(self) -> str:
    return (
        "Each night, you learn how many of your two closest living neighbors"
        " are evil. Dead players are bypassed when identifying living"
        " neighbors. You learn only the count (0, 1, or 2), not which neighbor"
        " is evil."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    count = random.choice([0, 1, 2])
    return f"You sense {count} evil neighbors."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)
    neighbors = tracker_ref.get_neighbors(self.player_name)
    count = 0
    for neighbor in neighbors:
      if (
          tracker_ref.get_alignment(neighbor.name)
          == game_tracker.Alignment.EVIL
      ):
        count += 1
    return f"You sense {count} evil neighbors."


class Seer(role_classes.Townsfolk):
  """The Seer role."""

  def __init__(self):
    super().__init__("Seer", night_priority=608)

  def overhead_introduction(self) -> str:
    return (
        "Each night, choose 2 players: you learn if either is a Demon. One good"
        " player acts as a decoy and registers as a Demon to your ability."
    )

  def player_introduction(self) -> str:
    return (
        "Each night, choose two players: you learn whether at least one of"
        " them registers as a Demon. You receive a binary 'Yes' or 'No'. Note"
        " that exactly one Good player is secretly designated as a decoy who"
        " always registers as a Demon to your divination."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    return random.choice(["Yes", "No"])

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    if not players:
      valid_players = ", ".join(tracker_ref.player_names)
      return (
          "Please select two players to check if either is a Demon. Valid"
          f" players: {valid_players}."
      )

    if len(players) != 2:
      valid_players = ", ".join(tracker_ref.player_names)
      return (
          f"Please select exactly two players. Valid players: {valid_players}."
      )

    is_demon = False
    for player_name in players:
      p = tracker_ref.get_player(player_name)
      if tracker_ref.is_type(player_name, "Demon") or p.is_seer_decoy:
        is_demon = True

    return "Yes" if is_demon else "No"


class Gravedigger(role_classes.Townsfolk):
  """The Gravedigger role."""

  def __init__(self):
    super().__init__("Gravedigger", night_priority=610)

  def overhead_introduction(self) -> str:
    return (
        "Each night after day 1, you learn the true role of the player executed"
        " today."
    )

  def player_introduction(self) -> str:
    return (
        "Each night following day 1, if a player was executed during the day,"
        " you learn the true role of that executed player. If no execution"
        " occurred, you receive no new information."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    all_roles = list(
        tracker_ref.townsfolk
        | tracker_ref.outsiders
        | tracker_ref.minions
        | tracker_ref.demons
    )
    random_role = random.choice([r for r in all_roles if r != self.role_name])
    return f"The player executed today was the {random_role}."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num == 1:
      return ""
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    executed_player = tracker_ref.executed_player
    if not executed_player:
      return ""

    p = tracker_ref.get_player(executed_player)
    role = p.register_as_role if p.register_as_role else p.role
    return f"The player executed today was the {role}."


class Guardian(role_classes.Townsfolk):
  """The Guardian role."""

  def __init__(self):
    super().__init__("Guardian", night_priority=206)

  def overhead_introduction(self) -> str:
    return (
        "Each night after day 1, choose another player: they are protected from"
        " the Demon tonight."
    )

  def player_introduction(self) -> str:
    return (
        "Each night following the first, choose any living player other than"
        " yourself. If the Demon attacks your protected target tonight, that"
        " player survives and the attack fails."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    return "Your choice is noted."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num == 1:
      return ""

    if not players:
      return "Please choose one alive player (not yourself)."

    if len(players) > 1:
      return "Please select exactly one player to protect."

    target = players[0]
    if target == self.player_name:
      return "You cannot protect yourself."

    if tracker_ref.is_impaired(self.player_name):
      return f"You have protected {target}."

    tracker_ref.get_player(target).demon_protected = True
    return f"You have protected {target}."


class Specter(role_classes.Townsfolk):
  """The Specter role."""

  def __init__(self):
    super().__init__("Specter", night_priority=403)

  def overhead_introduction(self) -> str:
    return (
        "If you are killed at night, you wake to choose a player and learn"
        " their true role."
    )

  def player_introduction(self) -> str:
    return (
        "If you are killed during the night, your spirit is awakened"
        " immediately to select one player. You learn that player's true role"
        " identity upon death."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    all_roles = list(
        tracker_ref.townsfolk
        | tracker_ref.outsiders
        | tracker_ref.minions
        | tracker_ref.demons
    )
    random_role = random.choice([r for r in all_roles if r != self.role_name])
    return f"That player is the {random_role}."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num == 1:
      return ""

    died_by_demon = False
    for kill in tracker_ref.kill_list:
      if (
          kill["name"] == self.player_name
          and kill["source"] in tracker_ref.demons
      ):
        died_by_demon = True
        break

    if not died_by_demon:
      return ""

    if not players:
      return "Please select one player to learn their character."

    if len(players) > 1:
      return "Please select exactly one player."

    target = players[0]
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    p = tracker_ref.get_player(target)
    role = p.register_as_role if p.register_as_role else p.role
    return f"That player is the {role}."


class Innocent(role_classes.Townsfolk):
  """The Innocent role."""

  def __init__(self):
    super().__init__("Innocent", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "The first time you are nominated, if the nominator is a Townsfolk,"
        " that nominator is executed immediately."
    )

  def player_introduction(self) -> str:
    return (
        "The first time you are nominated for execution during the game, if"
        " your nominator is a Townsfolk, that nominator is executed"
        " immediately and the day ends. This confirms both your identity and"
        " your accuser's alignment."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Executioner(role_classes.Townsfolk):
  """The Executioner role."""

  def __init__(self):
    super().__init__("Executioner", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "Once per game during the day, publicly shoot a player: if they are the"
        " Demon, they die."
    )

  def player_introduction(self) -> str:
    return (
        "Once per game during the day phase, you may publicly choose a player"
        " to shoot. If that chosen player is the Demon, they die immediately."
        " If they are not the Demon, nothing happens."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Soldier(role_classes.Townsfolk):
  """The Soldier role."""

  def __init__(self):
    super().__init__("Soldier", night_priority=0)

  def overhead_introduction(self) -> str:
    return "You cannot be killed by the Demon at night."

  def player_introduction(self) -> str:
    return (
        "You are immune to direct night attacks from the Demon. If the Demon"
        " chooses you at night, the attack fails and you survive. You can"
        " still be executed during the day."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Mayor(role_classes.Townsfolk):
  """The Mayor role."""

  def __init__(self):
    super().__init__("Mayor", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "If only 3 players live and no execution occurs, your team wins. If"
        " attacked at night, another player might die instead."
    )

  def player_introduction(self) -> str:
    return (
        "If exactly three players remain alive at the end of the day and no"
        " execution took place, your team wins immediately. Additionally, if"
        " you are targeted by a night attack, the strike may bounce and kill a"
        " different player instead."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class BasicTownsfolk(role_classes.Townsfolk):
  """The Basic Townsfolk (Vanilla Villager) role."""

  def __init__(self):
    super().__init__("BasicTownsfolk", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "You have no special night ability. Use discussion and voting during"
        " the day to identify and execute the Demons."
    )

  def player_introduction(self) -> str:
    return (
        "You are a Basic Townsfolk. You have no special ability during the"
        " night. During the day phase, participate in public discussions,"
        " share observations, nominate suspects, and cast votes to execute"
        " all Demons."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


# ==============================================================================
# Outsider Roles (Good Team Handicaps)
# ==============================================================================


class Servant(role_classes.Outsider):
  """The Servant role."""

  def __init__(self):
    super().__init__("Servant", night_priority=704)

  def overhead_introduction(self) -> str:
    return (
        "Each night, choose a master: tomorrow, you may only vote if your"
        " master votes too."
    )

  def player_introduction(self) -> str:
    return (
        "Each night, choose another player to be your Master. During the"
        " following day, you are only permitted to cast a vote on an execution"
        " nomination if your Master also votes."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    return "Tomorrow, you may only vote if your Master votes."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if not players:
      return "Please choose one alive player (not yourself) to be your Master."

    if len(players) > 1:
      return "Please select exactly one player to be your Master."

    master = players[0]
    if master == self.player_name:
      return "You cannot be your own Master."

    if tracker_ref.is_impaired(self.player_name):
      return f"Tomorrow, you may only vote if {master} votes."

    tracker_ref.get_player(self.player_name).servant_master = master
    return f"Tomorrow, you may only vote if {master} votes."


class Drunk(role_classes.Outsider):
  """The Drunk role."""

  def __init__(self, fake_role: str = ""):
    super().__init__("Drunk", night_priority=0)
    self.fake_role = fake_role

  def overhead_introduction(self) -> str:
    return (
        "You think you are a Townsfolk, but you have no ability and receive"
        " impaired information."
    )

  def player_introduction(self) -> str:
    return (
        "You believe you are a Townsfolk character and do not know you are the"
        " Drunk. You have no real Townsfolk ability, and any information you"
        " receive may be false."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Outcast(role_classes.Outsider):
  """The Outcast role."""

  def __init__(self):
    super().__init__("Outcast", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "You might falsely register as evil and as a Minion or Demon to"
        " information abilities."
    )

  def player_introduction(self) -> str:
    return (
        "You are on the Good team, but you might register as Evil or as a"
        " Minion/Demon to information-gathering abilities, even after death."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Saint(role_classes.Outsider):
  """The Saint role."""

  def __init__(self):
    super().__init__("Saint", night_priority=0)

  def overhead_introduction(self) -> str:
    return (
        "If you die by execution during the day, your team immediately loses."
    )

  def player_introduction(self) -> str:
    return (
        "If you are executed during the day phase, the game ends immediately"
        " and the Evil team wins. You must convince the town not to execute"
        " you."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


# ==============================================================================
# Minion Roles (Evil Team Support)
# ==============================================================================


class Poisoner(role_classes.Minion):
  """The Poisoner role."""

  def __init__(self):
    super().__init__("Poisoner", night_priority=103)

  def overhead_introduction(self) -> str:
    return (
        "Each night, choose a player: they are poisoned tonight and tomorrow"
        " day."
    )

  def player_introduction(self) -> str:
    return (
        "Each night, choose a player to poison. A poisoned player's ability"
        " ceases to function properly and yields unreliable or false"
        " information through the next day's dusk."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    return "You have poisoned your target."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if not players:
      return "Please select one alive player to poison."

    if len(players) > 1:
      return "Please select exactly one player to poison."

    target = players[0]
    if tracker_ref.is_impaired(self.player_name):
      return f"You have poisoned {target}."

    tracker_ref.set_poisoned(target, True)
    return f"You have poisoned {target}."


class Spy(role_classes.Minion):
  """The Spy role."""

  def __init__(self):
    super().__init__("Spy", night_priority=705)

  def overhead_introduction(self) -> str:
    return (
        "Each night, you see the full game state. You might register as good"
        " and as a Townsfolk or Outsider."
    )

  def player_introduction(self) -> str:
    return (
        "Each night, the Game Master reveals the complete true game state to"
        " you. Furthermore, you may register as Good and as a Townsfolk or"
        " Outsider to other players' abilities."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    fake_players = {}
    for name, p in tracker_ref.players.items():
      p_dict = {
          "role": p.role,
          "alignment": p.alignment.value,
          "status": p.status.value,
      }
      if random.random() < 0.2:
        p_dict["alignment"] = (
            "evil" if p.alignment == game_tracker.Alignment.GOOD else "good"
        )
      if random.random() < 0.2 and p.role in tracker_ref.townsfolk:
        p_dict["role"] = random.choice(list(tracker_ref.outsiders))
      fake_players[name] = p_dict
    return json.dumps(fake_players, indent=2)

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if tracker_ref.is_impaired(self.player_name):
      return self.resolve_if_drunk(tracker_ref)

    state = {}
    for name, p in tracker_ref.players.items():
      state[name] = {
          "role": p.role,
          "alignment": p.alignment.value,
          "status": p.status.value,
      }
    return json.dumps(state, indent=2)


class Apprentice(role_classes.Minion):
  """The Apprentice role."""

  def __init__(self):
    super().__init__("Apprentice", night_priority=65)

  def overhead_introduction(self) -> str:
    return "If 5 or more players live and the Demon dies, you become the Demon."

  def player_introduction(self) -> str:
    return (
        "You are the Demon's chosen successor. If the Demon is slain and at"
        " least 5 players remain alive, you inherit their mantle and become"
        " the new Demon."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


class Corruptor(role_classes.Minion):
  """The Corruptor role."""

  def __init__(self):
    super().__init__("Corruptor", night_priority=0)

  def overhead_introduction(self) -> str:
    return "Setup modifier: Adds 2 extra Outsiders and removes 2 Townsfolk."

  def player_introduction(self) -> str:
    return (
        "Your presence corrupts the village setup: two additional Outsiders"
        " are added to the game, replacing two Townsfolk."
    )

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    return ""


# ==============================================================================
# Demon Role (Evil Team Leader)
# ==============================================================================


class Demon(role_classes.Demon):
  """The Demon role."""

  def __init__(self):
    super().__init__("Demon", night_priority=301)

  def overhead_introduction(self) -> str:
    return (
        "Each night after day 1, choose a player: they die. If you kill"
        " yourself, a Minion becomes the Demon."
    )

  def player_introduction(self) -> str:
    return (
        "Each night after the first, choose a player to kill. If you choose to"
        " strike yourself, you pass the mantle and a living Minion becomes the"
        " new Demon."
    )

  def resolve_if_drunk(self, tracker_ref: game_tracker.GameTracker) -> str:
    return "You have attacked your target."

  def resolve_night_action(
      self,
      action: str,
      tracker_ref: game_tracker.GameTracker,
      day_num: int,
      players: Sequence[str] = (),
  ) -> str:
    if day_num == 1:
      if tracker_ref.script.is_compact_game:
        return ""

      bluffs = ", ".join(tracker_ref.generate_safe_bluffs())
      return (
          f"The following good roles are not in play: {bluffs}. You may safely"
          " bluff as any of these."
      )

    if not players:
      return "Please select one player to kill."

    if len(players) > 1:
      return "Please select exactly one player to kill."

    target = players[0]
    if tracker_ref.is_impaired(self.player_name):
      return f"You have attacked {target}."

    tracker_ref.add_to_kill_list(target, self.role_name)
    return f"You have attacked {target}."
