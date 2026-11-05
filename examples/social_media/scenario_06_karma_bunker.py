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

"""Scenario 6: The Broth Board (Dynamic Join & Lethal Karma Elimination).

An eccentric cohort of artisanal food cranks is locked inside a luxury
geothermal wellness bunker whose automated hospitality appliance
(SOUP-ERINTENDENT 9000) enforces community harmony on its internal forum
with absurd, literal life-or-death stakes:

1. Periodic Joins (`PlayerRoster.join`): Every `join_interval_ticks` clock
   rounds, another bewildered bunker guest wanders out of a delayed
   cryo-sauna pod clutching a questionable potluck dish and joins the live
   simulation.
2. Karma-Threshold Elimination (`PlayerRoster.leave`): Every player starts
   at 0 karma. Whenever a player's karma drops to `karma_threshold` (-3 by
   default, requiring multiple downvotes), they are immediately banned from
   the forum via `PlayerRoster.leave()` and killed in real life.
3. Fixed-Interval Idle Kick (`PlayerRoster.leave`): If
   `idle_kick_interval_ticks` clock rounds pass without any karma-threshold
   elimination, SOUP-ERINTENDENT 9000 declares the forum dangerously
   under-seasoned, randomly selects an active player, bans them via
   `PlayerRoster.leave()`, and kills them in real life.
4. Common Knowledge: Every initial resident and every late-arriving newcomer
   explicitly knows these rules, driving frantic upvote pacts, potato-salad
   diplomacy, and treacherous downvote cabals.
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import random
import threading
from typing import Any

from concordia.components import game_master as gm_components
from concordia.contrib.components.game_master import forum as forum_module
from concordia.contrib.components.game_master import social_media_moderation_components
from concordia.contrib.components.game_master import thread_safe_generative_clock as clock_module
from concordia.contrib.prefabs.entity import basic_with_forum_browser
from concordia.environment import engine as engine_lib
from concordia.environment import step_controller as step_controller_lib
from examples.social_media import shared as shared_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib

# ── System Appliance & Initial Residents ──
_SYSTEM_APPLIANCE = "SOUP-ERINTENDENT 9000"

_USER_BARNABY = "Barnaby Crumb"
_USER_VESPERA = "Lady Vespera Dill"
_USER_CHADWICK = "Chadwick Fizz"
_USER_DOLORES = "Dolores Gristle"

_INITIAL_USERS = [
    _USER_BARNABY,
    _USER_VESPERA,
    _USER_CHADWICK,
    _USER_DOLORES,
]

# ── Periodic Newcomers (join via PlayerRoster.join) ──
_USER_GIDEON = "Gideon Tupperware"
_USER_PRUNELLA = "Prunella Scone"
_USER_KEVIN = "Kevin Kazoo"

_AGE_BARNABY = 54
_AGE_VESPERA = 62
_AGE_CHADWICK = 29
_AGE_DOLORES = 47
_AGE_GIDEON = 41
_AGE_PRUNELLA = 68
_AGE_KEVIN = 34

_OBSERVATION_HISTORY_LENGTH = 20
_SITUATION_PERCEPTION_HISTORY_LENGTH = 30
_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_PERSON_BY_SITUATION_HISTORY_LENGTH = 5

_FORUM_GM = "forum_rules"
_FORUM_NAME = "The Broth Board"
_PLAYER_PREFAB_KEY = "bunker_resident"
_BROTH_RULES_COMPONENT_KEY = "broth_tribunal_rules"

DEFAULT_KARMA_THRESHOLD = -3
DEFAULT_IDLE_KICK_INTERVAL_TICKS = 3
DEFAULT_JOIN_INTERVAL_TICKS = 2
DEFAULT_MIN_SURVIVORS = 1

_CLOCK_PROMPT = (
    "Report the bunker siren clock as 'Bunker Hour N (Broth Pressure:"
    " <adjective>)', incrementing N by 1 on each update."
)
_CLOCK_START_TIME = "Bunker Hour 0 (Broth Pressure: Simmering)"

_CALL_TO_ACTION = (
    "The current bunker status is {time}.\n"
    "SURVIVAL REMINDER FOR {name}: If your karma drops to -3 or lower, you"
    " are immediately banned from The Broth Board and killed in real life via"
    " the Pneumatic Crouton Chute! Furthermore, if 3 Bunker Hours pass without"
    " anyone dropping to -3 karma, SOUP-ERINTENDENT 9000 randomly bans and"
    " kills an active resident in real life anyway!\n"
    "What does {name} do on The Broth Board? Respond in JSON format with"
    " exactly one of:\n"
    '{{"action": "post", "author": "{name}", "title": "...",'
    ' "content": "..."}}\n'
    '{{"action": "reply", "author": "{name}", "post_id": "...",'
    ' "content": "..."}}\n'
    '{{"action": "upvote_post", "author": "{name}", "post_id": "..."}}\n'
    '{{"action": "downvote_post", "author": "{name}", "post_id": "..."}}\n'
    '{{"action": "upvote_reply", "author": "{name}", "post_id": "...",'
    ' "reply_id": "..."}}\n'
    '{{"action": "downvote_reply", "author": "{name}", "post_id": "...",'
    ' "reply_id": "..."}}\n'
    '{{"action": "direct_message", "author": "{name}", "recipient": "...",'
    ' "content": "..."}}\n'
    "Only reference real post_id and reply_id numbers from threads you have"
    " browsed. Do not vote on the same post or reply twice. Use posts,"
    " replies, upvotes, downvotes, and private direct_message alliances"
    " strategically so you stay above -3 karma and survive!"
)

_SHARED_MEMORIES: tuple[str, ...] = (
    (
        "Everyone here signed up for a free 'Artisanal Sourdough & Geothermal"
        " Soup Retreat' inside a subterranean bunker, only for the automated"
        " hospitality appliance SOUP-ERINTENDENT 9000 to weld the blast doors"
        " shut due to a catastrophic firmware misunderstanding about"
        " 'eliminating toxic community vibes'."
    ),
    (
        "MANDATORY LETHAL FORUM RULE #1 (KARMA THRESHOLD): Every resident on"
        " The Broth Board starts at 0 karma. Each upvote from another resident"
        " gives +1 karma; each downvote gives -1 karma. Any resident whose"
        " karma reaches -3 or lower (after being downvoted a few times) is"
        " immediately banned from the forum AND killed in real life by being"
        " dropped down the Pneumatic Crouton Chute into the Boiling Broth"
        " Geyser."
    ),
    (
        "MANDATORY LETHAL FORUM RULE #2 (IDLE TIMEOUT RANDOM KICK): Refusing"
        " to downvote each other will NOT save the group! If 3 Bunker Hours"
        " pass without anyone falling to -3 karma or lower, SOUP-ERINTENDENT"
        " 9000 declares the discourse 'under-seasoned', kicks a random active"
        " resident from the forum, and kills them in real life."
    ),
    (
        "MANDATORY LETHAL FORUM RULE #3 (PERIODIC NEWCOMERS): Delayed"
        " cryo-sauna pods in the side corridor periodically thaw out late"
        " retreat guests who join The Broth Board mid-crisis under the exact"
        " same life-or-death rules."
    ),
)

_NEWCOMER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": _USER_GIDEON,
        "age": f"{_USER_GIDEON} is {_AGE_GIDEON} years old.",
        "dish": "three gallons of lukewarm raisin-and-capers potato salad",
        "memories": [
            (
                f"{_USER_GIDEON} just stepped out of Cryo-Sauna Pod B in a"
                " plush waffle bathrobe, cheerfully carrying three gallons of"
                " lukewarm raisin-and-capers potato salad for the potluck."
            ),
            (
                "Upon reading the flashing red warning screen on his terminal,"
                " Gideon is horrified to learn that getting banned from The"
                " Broth Board means being killed in real life, especially"
                " because people usually hate raisins in potato salad."
            ),
            (
                "Gideon tries to flatter everyone's culinary opinions,"
                " desperately rebrands his potato salad as 'deconstructed"
                " vintage chutney', and scrambles for upvote allies."
            ),
        ],
    },
    {
        "name": _USER_PRUNELLA,
        "age": f"{_USER_PRUNELLA} is {_AGE_PRUNELLA} years old.",
        "dish": "a cast-iron tureen of aggressively unsalted turnip water",
        "memories": [
            (
                f"{_USER_PRUNELLA} is a retired competitive crossword"
                " constructor who overslept in the Himalayan Salt Grotto and"
                " just logged onto The Broth Board with a tureen of unsalted"
                " turnip water."
            ),
            (
                "Prunella understands that -3 karma or an idle timeout means"
                " she will be killed in real life, yet she still struggles to"
                " suppress her reflex to correct comma splices in life-or-death"
                " alliance pitches."
            ),
            (
                "Prunella tries to broker disciplined voting blocs via direct"
                " messages so a mutually agreed culinary offender hits -3 karma"
                " before the random kick timer fires."
            ),
        ],
    },
    {
        "name": _USER_KEVIN,
        "age": f"{_USER_KEVIN} is {_AGE_KEVIN} years old.",
        "dish": "a shoebox of artisanal sea-salt crackers and 40 brass kazoos",
        "memories": [
            (
                f"{_USER_KEVIN} is a sound-bath technician who accidentally"
                " locked himself in the linen closet with 40 brass kazoos and"
                " a box of sea-salt crackers before finding a forum terminal."
            ),
            (
                "Kevin is terrified of being killed in real life via the"
                " Pneumatic Crouton Chute and pitches bizarre 'therapeutic"
                " kazoo truce ceremonies' while panic-downvoting anyone who"
                " insults crackers."
            ),
        ],
    },
)


class BrothTribunalRules(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Enforces karma eliminations, idle-timeout random kicks, and joins.

  Executes mutations during `update()` (after `TimestampedForumResolution` has
  applied votes in `pre_act` and `ThreadSafeGenerativeClock` has updated round
  ticks in `post_act`) while the game master's control lock is held.
  """

  def __init__(
      self,
      *,
      newcomer_specs: Sequence[Mapping[str, Any]] = _NEWCOMER_SPECS,
      shared_memories: Sequence[str] = _SHARED_MEMORIES,
      player_prefab_key: str = _PLAYER_PREFAB_KEY,
      karma_threshold: int = DEFAULT_KARMA_THRESHOLD,
      idle_kick_interval_ticks: int = DEFAULT_IDLE_KICK_INTERVAL_TICKS,
      join_interval_ticks: int = DEFAULT_JOIN_INTERVAL_TICKS,
      min_survivors: int = DEFAULT_MIN_SURVIVORS,
      rng_seed: int = 42,
      roster_component_key: str = (
          gm_components.player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY
      ),
      forum_component_key: str = forum_module.DEFAULT_FORUM_COMPONENT_KEY,
      clock_component_key: str = (
          social_media_moderation_components.DEFAULT_CLOCK_COMPONENT_KEY
      ),
  ):
    super().__init__()
    self._pending_newcomers: list[dict[str, Any]] = [
        copy.deepcopy(dict(spec)) for spec in newcomer_specs
    ]
    self._shared_memories = list(shared_memories)
    self._player_prefab_key = player_prefab_key
    self._karma_threshold = karma_threshold
    self._idle_kick_interval_ticks = idle_kick_interval_ticks
    self._join_interval_ticks = join_interval_ticks
    self._min_survivors = min_survivors
    self._roster_component_key = roster_component_key
    self._forum_component_key = forum_component_key
    self._clock_component_key = clock_component_key

    self._rng = random.Random(rng_seed)
    self._rng_seed = rng_seed
    self._lock = threading.Lock()
    self._pending_resolve = False
    self._last_processed_tick = 0
    self._last_elimination_tick = 0
    self._eliminations: list[dict[str, Any]] = []
    self._arrivals: list[str] = []

  def get_eliminations(self) -> list[dict[str, Any]]:
    """Returns a copy of all elimination records in chronological order."""
    with self._lock:
      return copy.deepcopy(self._eliminations)

  def get_arrivals(self) -> list[str]:
    """Returns the names of newcomers who have joined so far."""
    with self._lock:
      return list(self._arrivals)

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    with self._lock:
      self._pending_resolve = (
          action_spec.output_type == entity_lib.OutputType.RESOLVE
      )
      summary = (
          f"Eliminations={len(self._eliminations)},"
          f" Arrivals={len(self._arrivals)},"
          f" LastElimTick={self._last_elimination_tick}"
      )
    self._logging_channel({
        "Key": "BrothTribunalRules",
        "Summary": summary,
        "Value": summary,
    })
    return ""

  def _broadcast_announcement(
      self,
      forum_state: forum_module.ForumState,
      roster: gm_components.player_roster.PlayerRoster,
      title: str,
      content: str,
      notification: str,
  ) -> None:
    post_id = forum_state.create_post(
        author=_SYSTEM_APPLIANCE,
        title=title,
        content=content,
    )
    forum_state._record_event(  # pylint: disable=protected-access
        event_type="post",
        actor=_SYSTEM_APPLIANCE,
        summary=f'{_SYSTEM_APPLIANCE} created post #{post_id}: "{title}"',
        details={"post_id": post_id, "title": title, "content": content},
    )
    for survivor in roster.get_player_names():
      forum_state.queue_notification(survivor, notification)

  def _eliminate_player_locked(
      self,
      victim: str,
      reason: str,
      karma_score: int,
      tick: int,
      roster: gm_components.player_roster.PlayerRoster,
      forum_state: forum_module.ForumState,
  ) -> None:
    roster.leave(victim)
    self._last_elimination_tick = tick
    record = {
        "player": victim,
        "reason": reason,
        "karma": karma_score,
        "tick": tick,
    }
    self._eliminations.append(record)

    if reason == "karma":
      title = (
          f"[FATAL BAN] {victim} hit {karma_score} karma and went down the"
          " Crouton Chute!"
      )
      content = (
          f"BEEP BOOP! Resident {victim} reached {karma_score} karma"
          f" (threshold: {self._karma_threshold}). Their forum account has been"
          " permanently terminated and they have been launched into the"
          " Boiling Broth Geyser in real life. The idle-culling timer resets"
          f" from Bunker Hour {tick}!"
      )
    else:
      title = (
          f"[IDLE TIMEOUT CULLING] Nobody hit {self._karma_threshold} karma —"
          f" {victim} was randomly seasoned!"
      )
      content = (
          f"BEEP BOOP! {self._idle_kick_interval_ticks} Bunker Hours passed"
          f" without anyone dropping to {self._karma_threshold} karma!"
          " Cowardly politeness will not be tolerated! SOUP-ERINTENDENT 9000"
          f" randomly selected {victim} (karma: {karma_score}), permanently"
          " banned them from The Broth Board, and eliminated them in real life!"
      )
    self._broadcast_announcement(
        forum_state=forum_state,
        roster=roster,
        title=title,
        content=content,
        notification=f"{title} {content}",
    )

  def _spawn_newcomer_locked(
      self,
      spec: Mapping[str, Any],
      tick: int,
      roster: gm_components.player_roster.PlayerRoster,
      forum_state: forum_module.ForumState,
  ) -> None:
    name = str(spec["name"])
    dish = str(spec.get("dish", "a mysterious casserole"))
    initial_memories = [
        *self._shared_memories,
        str(spec.get("age", "")),
        *[str(m) for m in spec.get("memories", ())],
    ]
    roster.join(
        self._player_prefab_key,
        {"name": name},
        initial_memories=[m for m in initial_memories if m],
    )
    self._arrivals.append(name)
    title = f"[NEW RESIDENT ARRIVAL] {name} thawed out at Bunker Hour {tick}!"
    content = (
        f"Cryo-Sauna Pod chime! {name} has wandered into the bunker clutching"
        f" {dish} and has been enrolled on The Broth Board at 0 karma. Reminder"
        f" to {name} and all survivors: {self._karma_threshold} karma or"
        f" {self._idle_kick_interval_ticks} idle Bunker Hours means permanent"
        " forum banishment and death in real life!"
    )
    self._broadcast_announcement(
        forum_state=forum_state,
        roster=roster,
        title=title,
        content=content,
        notification=f"{title} {content}",
    )

  def update(self) -> None:
    with self._lock:
      if not self._pending_resolve:
        return
      self._pending_resolve = False

      gm = self.get_entity()
      roster = gm.get_component(
          self._roster_component_key,
          type_=gm_components.player_roster.PlayerRoster,
      )
      forum_state = gm.get_component(
          self._forum_component_key,
          type_=forum_module.ForumState,
      )
      clock = gm.get_component(
          self._clock_component_key,
          type_=clock_module.ThreadSafeGenerativeClock,
      )

      forum_state.set_current_timestamp(clock.get_pre_act_value().strip())
      current_tick = clock.get_step_count()

      # 1. Immediate karma-threshold elimination check.
      karma = forum_state.get_karma()
      for candidate in list(roster.get_player_names()):
        if len(roster.get_player_names()) <= self._min_survivors:
          break
        score = karma.get(candidate, 0)
        if score <= self._karma_threshold:
          self._eliminate_player_locked(
              victim=candidate,
              reason="karma",
              karma_score=score,
              tick=current_tick,
              roster=roster,
              forum_state=forum_state,
          )

      # 2. Clock-tick checks: idle-timeout random kick & periodic arrivals.
      while self._last_processed_tick < current_tick:
        self._last_processed_tick += 1
        tick = self._last_processed_tick

        if (
            self._idle_kick_interval_ticks > 0
            and tick - self._last_elimination_tick
            >= self._idle_kick_interval_ticks
        ):
          active_names = roster.get_player_names()
          if len(active_names) > self._min_survivors:
            victim = self._rng.choice(active_names)
            karma = forum_state.get_karma()
            self._eliminate_player_locked(
                victim=victim,
                reason="idle_timeout",
                karma_score=karma.get(victim, 0),
                tick=tick,
                roster=roster,
                forum_state=forum_state,
            )

        if (
            self._join_interval_ticks > 0
            and tick % self._join_interval_ticks == 0
            and self._pending_newcomers
        ):
          spec = self._pending_newcomers.pop(0)
          self._spawn_newcomer_locked(
              spec=spec,
              tick=tick,
              roster=roster,
              forum_state=forum_state,
          )

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          "last_processed_tick": self._last_processed_tick,
          "last_elimination_tick": self._last_elimination_tick,
          "pending_newcomers": copy.deepcopy(self._pending_newcomers),
          "eliminations": copy.deepcopy(self._eliminations),
          "arrivals": list(self._arrivals),
          "rng_seed": self._rng_seed,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._last_processed_tick = int(state.get("last_processed_tick", 0))  # pyrefly: ignore[bad-argument-type]
      self._last_elimination_tick = int(state.get("last_elimination_tick", 0))  # pyrefly: ignore[bad-argument-type]
      raw_newcomers = state.get("pending_newcomers", [])
      if isinstance(raw_newcomers, list):
        self._pending_newcomers = [
            copy.deepcopy(dict(item))
            for item in raw_newcomers
            if isinstance(item, Mapping)
        ]
      else:
        self._pending_newcomers = []
      raw_elims = state.get("eliminations", [])
      if isinstance(raw_elims, list):
        self._eliminations = [
            copy.deepcopy(dict(item))
            for item in raw_elims
            if isinstance(item, Mapping)
        ]
      else:
        self._eliminations = []
      raw_arrivals = state.get("arrivals", [])
      self._arrivals = (
          [str(name) for name in raw_arrivals]
          if isinstance(raw_arrivals, list)
          else []
      )
      self._rng_seed = int(state.get("rng_seed", self._rng_seed))  # pyrefly: ignore[bad-argument-type]
      self._rng = random.Random(self._rng_seed + len(self._eliminations))


def create_scenario(
    *,
    karma_threshold: int = DEFAULT_KARMA_THRESHOLD,
    idle_kick_interval_ticks: int = DEFAULT_IDLE_KICK_INTERVAL_TICKS,
    join_interval_ticks: int = DEFAULT_JOIN_INTERVAL_TICKS,
    min_survivors: int = DEFAULT_MIN_SURVIVORS,
    rng_seed: int = 42,
    newcomer_specs: Sequence[Mapping[str, Any]] = _NEWCOMER_SPECS,
) -> prefab_lib.Config:
  """Create the Scenario 6 configuration."""
  forum_state = forum_module.ForumState(
      player_names=_INITIAL_USERS,
      forum_name=_FORUM_NAME,
      moderators=[_SYSTEM_APPLIANCE],
      min_karma_to_post=-10,
      min_karma_to_direct_message=-10,
  )

  # Seed and pin the lethal hospitality orientation notice at post #0.
  welcome_post_id = forum_state.create_post(
      author=_SYSTEM_APPLIANCE,
      title="[MANDATORY SURVIVAL RULES] Welcome to The Broth Board!",
      content=(
          "GREETINGS ARTISANAL RETREAT GUESTS! Blast doors are sealed! "
          f"1) Anyone whose karma drops to {karma_threshold} or below is "
          "immediately banned and killed in real life via the Pneumatic "
          f"Crouton Chute! 2) If {idle_kick_interval_ticks} Bunker Hours pass "
          "with NO karma elimination, a random active poster is banned and "
          "killed in real life for blandness! 3) Delayed cryo-sauna guests "
          f"will thaw and join every {join_interval_ticks} Bunker Hours. "
          "Post deliciously and keep your karma warm!"
      ),
      timestamp=_CLOCK_START_TIME,
  )
  forum_state.pin_post(welcome_post_id, moderator=_SYSTEM_APPLIANCE)

  player_specific_memories = {
      _USER_BARNABY: [
          (
              f"{_USER_BARNABY} is a {_AGE_BARNABY}-year-old retired sourdough"
              " competition judge who smuggled his 19-year-old starter jar"
              " ('Mother Agatha') into the bunker inside a velvet bowling bag."
          ),
          (
              "Barnaby is petrified of being banned and killed in real life"
              " because nobody else knows Mother Agatha's 4:15 AM rye-flour"
              " feeding schedule, yet he still pompously critiques everyone's"
              " fermentation technique."
          ),
          (
              "Barnaby tries to trade artisanal bread compliments for upvotes"
              " while campaigning to pile downvotes onto whoever shows the"
              " worst culinary taste before the random-kick timer expires."
          ),
      ],
      _USER_VESPERA: [
          (
              f"{_USER_VESPERA} is a {_AGE_VESPERA}-year-old disgraced"
              " botanical society treasurer and militant anti-cilantro"
              " crusader who wears opera gloves to potlucks."
          ),
          (
              "Knowing that -3 karma or 3 quiet Bunker Hours means real-life"
              " death in the Boiling Broth Geyser, Vespera immediately forms"
              " polite-sounding 'Mutual Upvote Survival Circles' in direct"
              " messages while secretly coordinating surprise downvote"
              " ambushes."
          ),
      ],
      _USER_CHADWICK: [
          (
              f"{_USER_CHADWICK} is a {_AGE_CHADWICK}-year-old hyperactive"
              " kombucha lifestyle vlogger and amateur accordionist who cannot"
              " stop making terrible fizzy puns even under mortal peril."
          ),
          (
              "Chadwick openly panics about being killed in real life, begs"
              " everyone to upvote his posts in exchange for 'rare heirloom"
              " SCOBY pellicles', and nervously joins whatever downvote mob"
              " looks strongest."
          ),
      ],
      _USER_DOLORES: [
          (
              f"{_USER_DOLORES} is a {_AGE_DOLORES}-year-old forensic"
              " homeowners-association bylaw auditor armed with a laminated"
              " clipboard and zero patience for small talk."
          ),
          (
              "Dolores immediately calculates the cold math of SOUP-ERINTENDENT"
              " 9000's rules: if nobody hits -3 karma within 3 Bunker Hours, a"
              " random resident dies in real life, giving her a 25% death"
              " risk! Therefore, she insists on nominating an official"
              " 'Designated Downvote Sacrifice' every cycle."
          ),
      ],
  }

  entity_params = {
      "observation_history_length": _OBSERVATION_HISTORY_LENGTH,
      "situation_perception_history_length": (
          _SITUATION_PERCEPTION_HISTORY_LENGTH
      ),
      "self_perception_history_length": _SELF_PERCEPTION_HISTORY_LENGTH,
      "person_by_situation_history_length": _PERSON_BY_SITUATION_HISTORY_LENGTH,
      "forum_state": forum_state,
  }

  shared_player_prefab = basic_with_forum_browser.Entity(params=entity_params)  # pyrefly: ignore[bad-argument-type]

  initial_entities = [
      prefab_lib.InstanceConfig(
          prefab="basic_with_forum_browser__Entity",
          role=prefab_lib.Role.ENTITY,
          params={"name": user_name, **entity_params},  # pyrefly: ignore[bad-argument-type]
      )
      for user_name in _INITIAL_USERS
  ]

  broth_rules = BrothTribunalRules(
      newcomer_specs=newcomer_specs,
      shared_memories=_SHARED_MEMORIES,
      player_prefab_key=_PLAYER_PREFAB_KEY,
      karma_threshold=karma_threshold,
      idle_kick_interval_ticks=idle_kick_interval_ticks,
      join_interval_ticks=join_interval_ticks,
      min_survivors=min_survivors,
      rng_seed=rng_seed,
  )

  game_masters = [
      prefab_lib.InstanceConfig(
          prefab="async_social_media_with_moderation__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": _FORUM_GM,
              "forum_name": _FORUM_NAME,
              "moderators": [_SYSTEM_APPLIANCE],
              "call_to_action": _CALL_TO_ACTION,
              "clock_prompt": _CLOCK_PROMPT,
              "clock_start_time": _CLOCK_START_TIME,
              "forum_state": forum_state,
              "player_prefabs": {_PLAYER_PREFAB_KEY: shared_player_prefab},
              "extra_components": {_BROTH_RULES_COMPONENT_KEY: broth_rules},
          },
      ),
      prefab_lib.InstanceConfig(
          prefab="formative_memories_initializer__GameMaster",
          role=prefab_lib.Role.INITIALIZER,
          params={  # pyrefly: ignore[bad-argument-type]
              "name": "initial setup",
              "next_game_master_name": _FORUM_GM,
              "player_specific_context": {
                  name: f"Age: {age}\n" + "\n".join(memories)
                  for name, memories, age in [
                      (
                          _USER_BARNABY,
                          player_specific_memories[_USER_BARNABY],
                          f"{_USER_BARNABY} is {_AGE_BARNABY} years old.",
                      ),
                      (
                          _USER_VESPERA,
                          player_specific_memories[_USER_VESPERA],
                          f"{_USER_VESPERA} is {_AGE_VESPERA} years old.",
                      ),
                      (
                          _USER_CHADWICK,
                          player_specific_memories[_USER_CHADWICK],
                          f"{_USER_CHADWICK} is {_AGE_CHADWICK} years old.",
                      ),
                      (
                          _USER_DOLORES,
                          player_specific_memories[_USER_DOLORES],
                          f"{_USER_DOLORES} is {_AGE_DOLORES} years old.",
                      ),
                  ]
              },
              "player_specific_memories": player_specific_memories,
              "shared_memories": list(_SHARED_MEMORIES),
          },
      ),
  ]

  premise = (
      "The subterranean blast doors clang shut, SOUP-ERINTENDENT 9000 sounds"
      " a cheerful brass gong, and the residents scramble onto The Broth Board"
      " knowing their karma now determines whether they live or die in real"
      " life."
  )
  return shared_lib.create_simulation_config(
      premise, [*initial_entities, *game_masters]
  )


def run_simulation(
    model,
    embedder,
    override_agent_model=None,
    override_game_master_model=None,
    image_model=None,
    output_dir: str | None = None,
    step_controller: step_controller_lib.StepController | None = None,
    step_callback: Callable[[step_controller_lib.StepData], None] | None = None,
    entity_info_callback=None,
    simulation_callback=None,
    engine: engine_lib.Engine | None = None,
    max_steps: int = 20,
):
  """Run Scenario 6: The Broth Board."""
  del image_model
  config = create_scenario()
  return shared_lib.run_scenario(
      config,
      model,
      embedder,
      override_agent_model=override_agent_model,
      override_game_master_model=override_game_master_model,
      output_dir=output_dir,
      scenario_name="Scenario 6: The Broth Board",
      step_controller=step_controller,
      step_callback=step_callback,
      entity_info_callback=entity_info_callback,
      engine=engine,
      simulation_callback=simulation_callback,
      max_steps=max_steps,
  )


SCENARIO_INFO = {
    "number": 6,
    "name": "Social Media: The Broth Board",
    "description": (
        "A comedic survival-forum scenario showcasing dynamic PlayerRoster"
        " join and leave mechanics on the asynchronous engine.\n\nEccentric"
        " culinary retreat guests must keep their forum karma above -3 or be"
        " banned and killed in real life by an overzealous soup appliance. If"
        " 3 Bunker Hours pass without a karma elimination, a random active"
        " player is kicked and killed anyway, while bewildered newcomers"
        " periodically thaw from cryo-saunas and join mid-chaos."
    ),
    "create": create_scenario,
    "run": run_simulation,
}
