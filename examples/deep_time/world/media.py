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

"""The Gazette: broadsheets, credibility and consent for coal.

A people that knows printing may spend paper on a broadsheet. Its press (a
slow mind) writes it, arguing for clean air or for coal and progress. The
broadsheet is posted on a contrib `forum.ForumState`, the Gazette, which
every people reads. Each other people judges it against what the public
instruments show (is the air filling, are lands restless, is the author fed?)
and votes it up or down, so an author's forum karma is its credibility.
Broadsheets sway every people's consent for coal by their credibility; the
land striking a people also lowers its consent (the smoke is blamed), and
consent drifts back as memories fade. Coal mines yield in proportion to the
consent of the miners.
"""

from collections.abc import Mapping
import dataclasses
import re
from typing import Any

from concordia.contrib.components.game_master import forum
from examples.deep_time.world import civ as civ_lib


@dataclasses.dataclass(frozen=True)
class Evidence:
  """What the public instruments show when a broadsheet is judged."""

  relative_co2: float  # Air carbon over the starting level.
  restless: float  # The worst restless land, as a share of the breaking point.
  attacked: frozenset[str]  # Peoples the land struck recently.
  fed: Mapping[str, float]  # Polity -> share of its people fed.


@dataclasses.dataclass(frozen=True)
class Broadsheet:
  author: str
  stance: str
  title: str
  post_id: int
  votes: int
  # Or 'pamphlet' (paid for by the coal guilds), or 'omen' (the printers'
  # retelling of the Steward's omen).
  kind: str = 'broadsheet'


def parse_stance(text: str, stances: Mapping[str, str]) -> str | None:
  """The stance whose keyword appears first in `text` (None if none does)."""
  lowered = text.lower()
  found = [
      (lowered.find(word), stance)
      for stance, word in stances.items()
      if word in lowered
  ]
  return min(found)[1] if found else None


def without_stance(text: str, stances: Mapping[str, str]) -> str:
  """A broadsheet's body: the text after its first stance keyword."""
  body = text.strip()
  stance = parse_stance(body, stances)
  if stance is None:
    return body
  start = body.lower().find(stances[stance]) + len(stances[stance])
  return body[start:].lstrip(' :.-—').strip()


def headline(body: str, limit: int = 80) -> str:
  """The first line or sentence, without markdown or a 'Headline:' label."""
  plain = re.sub(r'[*_#`"“”]', '', body).strip()
  plain = re.sub(r'^headline\s*:\s*', '', plain, flags=re.IGNORECASE)
  first = re.split(r'\n|(?<=[.!?])\s', plain, maxsplit=1)[0]
  return first.strip().rstrip('.!?').strip()[:limit]


class Gazette:
  """Broadsheets on a contrib forum, and each people's consent for coal."""

  def __init__(self, config: Mapping[str, Any], names: Mapping[str, str]):
    self.paper_cost = float(config['paper_cost'])
    self.cooldown = int(config['cooldown_ticks'])
    self.latency = int(config['press_latency_ticks'])
    self.sway = float(config['sway'])
    self.relax = float(config['relax'])
    self.blame = float(config['blame'])
    self.min_consent = float(config['min_consent'])
    self.karma_weight = float(config['karma_weight'])
    self.credibility_range = tuple(float(x) for x in config['credibility'])
    self.clean_co2 = float(config['clean_air_credible']['co2'])
    self.clean_restless = float(config['clean_air_credible']['restless'])
    self.stances = {k: str(v).lower() for k, v in config['stances'].items()}
    self.question = str(config['question']).strip()
    pamphlets = config['pamphlets']
    self.pamphlet_silver = float(pamphlets['silver'])
    self.pamphlet_stance = str(pamphlets['stance'])
    self.pamphlet_sway = float(pamphlets.get('sway_factor', 1.0))
    self.pamphlet_question = str(pamphlets['question']).strip()
    # The presses' owners retell the Steward's omens (relay_mind.py).
    self.relay = dict(config.get('relay') or {})
    self._names = dict(names)
    self.forum = forum.ForumState(
        player_names=sorted(self._names.values()),
        forum_name=str(config['name']),
    )
    # Polity -> what its press is writing: 'broadsheet' or 'pamphlet'.
    self.pending: dict[str, str] = {}
    self.last_printed: dict[str, int] = {}
    self.printed: list[Broadsheet] = []

  def can_print(self, polity: str, tick: int, paper: float) -> bool:
    last = self.last_printed.get(polity)
    return (
        polity not in self.pending
        and paper >= self.paper_cost - 1e-9
        and (last is None or tick - last >= self.cooldown)
    )

  def can_pamphlet(self, polity: str, tick: int, silver: float) -> bool:
    """Whether a people can pay the coal guilds for pamphlets now."""
    last = self.last_printed.get(polity)
    return (
        polity not in self.pending
        and silver >= self.pamphlet_silver - 1e-9
        and (last is None or tick - last >= self.cooldown)
    )

  def commission(
      self, polity: str, tick: int, kind: str = 'broadsheet'
  ) -> None:
    """The people's press starts writing a broadsheet (or a pamphlet)."""
    self.pending[polity] = kind
    self.last_printed[polity] = tick

  def question_for(self, polity: str) -> str:
    """The press's question for what it is writing."""
    if self.pending.get(polity) == 'pamphlet':
      return self.pamphlet_question
    return self.question

  def credibility(self, polity: str) -> float:
    karma = self.forum.get_karma().get(self._names[polity], 0)
    low, high = self.credibility_range
    return min(high, max(low, 1.0 + self.karma_weight * karma))

  def _credible(self, stance: str, author: str, evidence: Evidence) -> bool:
    if stance == 'clean':
      return (
          evidence.relative_co2 >= self.clean_co2
          or evidence.restless >= self.clean_restless
          or bool(evidence.attacked)
      )
    return (
        evidence.fed.get(author, 0.0) >= 1.0 and author not in evidence.attacked
    )

  def publish(
      self,
      polity: str,
      text: str,
      timestamp: str,
      evidence: Evidence,
      civ: civ_lib.Civilization,
      kind: str | None = None,
  ) -> Broadsheet | None:
    """Posts a broadsheet, has the other peoples judge it, and sways them.

    Args:
      polity: The author's key.
      text: The press's answer: a stance keyword, a headline and the body.
      timestamp: Forum timestamp (the in-game date).
      evidence: What the public instruments show now.
      civ: The peoples, whose consent the broadsheet sways.
      kind: 'omen' for the printers' retelling of an omen (nothing is
        commissioned); otherwise what the author's press was writing.

    Returns:
      The broadsheet, or None if nothing was commissioned or no stance taken.
    """
    if kind is None:
      kind = self.pending.pop(polity, None)
    if kind is None:
      return None
    stance = parse_stance(text, self.stances)
    if kind == 'pamphlet':
      stance = self.pamphlet_stance  # The guilds pay for coal's side.
    if stance is None:
      return None
    body = without_stance(text, self.stances)
    title = headline(body) or self.stances[stance]
    author = self._names[polity]
    post_id = self.forum.create_post(author, title, body, timestamp)
    credible = self._credible(stance, polity, evidence)
    for reader in sorted(k for k in self._names if k != polity):
      vote = self.forum.upvote if credible else self.forum.downvote
      vote(post_id, voter=self._names[reader])
    weight = self.sway * self.credibility(polity)
    if kind == 'pamphlet':
      weight *= self.pamphlet_sway
    elif kind == 'omen':
      weight *= float(self.relay.get('sway_factor', 1.0))
    direction = -1.0 if stance == 'clean' else 1.0
    for people in civ.polities.values():
      people.consent = min(
          1.0, max(self.min_consent, people.consent + direction * weight)
      )
    votes = self.forum.get_recent_posts(1)[0].votes
    sheet = Broadsheet(author, stance, title, post_id, votes, kind)
    self.printed.append(sheet)
    return sheet

  def blame_smoke(self, polity: str, civ: civ_lib.Civilization) -> None:
    """The land struck this people: it blames the smoke."""
    people = civ.polities[polity]
    people.consent = max(self.min_consent, people.consent - self.blame)

  def step(self, civ: civ_lib.Civilization) -> None:
    """Memories fade: consent drifts back toward full."""
    for people in civ.polities.values():
      people.consent += self.relax * (1.0 - people.consent)

  def payload(
      self, civ: civ_lib.Civilization, evidence: Evidence, n: int = 5
  ) -> dict[str, Any]:
    """The Gazette's front page: broadsheets, the sky watch, opinion."""
    names = {v: k for k, v in self._names.items()}
    words = self.stances
    return {
        'sky': {
            'relative_co2': round(evidence.relative_co2, 3),
            'restless': round(evidence.restless, 3),
        },
        'broadsheets': [
            {
                'author': post.author,
                'title': post.title,
                'stance': next(
                    (
                        b.stance
                        for b in self.printed
                        if b.post_id == post.post_id
                    ),
                    '',
                ),
                'votes': post.votes,
            }
            for post in self.forum.get_recent_posts(n)
        ],
        'credibility': {
            name: round(self.credibility(key), 2)
            for name, key in sorted(names.items())
        },
        'consent': {
            p.name: round(p.consent, 3) for _, p in sorted(civ.polities.items())
        },
        'stances': dict(words),
    }

  def get_state(self) -> dict[str, Any]:
    return {
        'forum': self.forum.get_state(),
        'pending': dict(sorted(self.pending.items())),
        'last_printed': dict(self.last_printed),
        'printed': [dataclasses.asdict(b) for b in self.printed],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.forum.set_state(state['forum'])
    pending = state.get('pending', {})
    self.pending = (
        {str(k): str(v) for k, v in pending.items()}
        if isinstance(pending, Mapping)
        else {str(k): 'broadsheet' for k in pending}
    )
    self.last_printed = {
        k: int(v) for k, v in state.get('last_printed', {}).items()
    }
    self.printed = [Broadsheet(**b) for b in state.get('printed', [])]
