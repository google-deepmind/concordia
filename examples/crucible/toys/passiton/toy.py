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

"""Pass it on: a message goes up the line mouth to ear (cycle 98).

The player writes one message for the first in line. Each relayer passes on
what they heard in their own way (a language model rewriting it in
character, by their temperament's law of relaying); the last in line act on
what reached them, by the laws of reading and their temperament. A level is
solved when they do what its goal asks. Each attempt shows every hop.

The relays and readings are the orders toy's own (`OrdersToy.relay` and
`OrdersToy.reading`): this toy only lays out the line and keeps the ledger
(attempts, words, verdict, stars).
"""

from collections.abc import Mapping
import copy
import re
from typing import Any

from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.crucible import toys
from examples.crucible.toys import base
from examples.crucible.toys.orders import toy as orders_lib

SEAT = 'Player'
# A word: letters, with an apostrophe inside ("don't"); "Don't-build" is two
# (cycle 106: whitespace runs let hyphens and missing spaces cheat stars).
_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words_in(message: str) -> int:
  return len(_WORDS.findall(message.replace('\u2019', "'")))


_SCENE = '__pass_it_on__'
_HOLDS = ('', 'hold', 'pass')


class PassItOnToy(base.Toy):
  """One level: a line of crew, a goal for the last in line, a few tries."""

  name = 'passiton'

  def __init__(
      self,
      model: language_model.LanguageModel,
      data: Mapping[str, Any],
      scenario: str | None = None,
      seed: int = 0,
  ):
    super().__init__()
    if scenario == 'daily' and 'daily' not in data.get('scenarios', {}):
      # Today's puzzle, made from the date's seed (cycle 112).
      from examples.crucible.toys.passiton import daily  # pylint: disable=g-import-not-at-top

      data = {
          **data,
          'scenarios': {**data['scenarios'], 'daily': daily.level(seed)},
      }
    elif scenario not in data.get('scenarios', {}):
      scenario = data.get('default_scenario', data['campaign'][0])
    self.scenario = scenario
    self._data = data
    self._level = data['scenarios'][scenario]
    self._stages = [
        [stage] if isinstance(stage, str) else list(stage)
        for stage in self._level['line']
    ]
    # The orders toy, set up as this line: its relays and readings.
    orders = copy.deepcopy(toys.load_data('orders'))
    for temperament, how in data.get('relays', {}).items():
      orders['temperaments'][temperament]['relays'] = how
    for temperament, props in data.get('temperaments', {}).items():
      orders['temperaments'][temperament].update(props)
    orders['scenarios'][_SCENE] = {
        'crew_present': [name for stage in self._stages for name in stage],
        'events': {1: self._level.get('event', 'quiet')},
        'turns': 1,
        'target': 1,
    }
    self._orders = orders_lib.OrdersToy(model, orders, _SCENE, seed)
    self._orders.world.shift = 0  # the level's event is this turn's
    self._ship = copy.deepcopy(self._orders.world)
    self.world: dict[str, Any] = {'attempts': [], 'solved': False}

  # ---- the line ------------------------------------------------------------

  @property
  def relayers(self) -> list[str]:
    return [stage[0] for stage in self._stages[:-1]]

  @property
  def receivers(self) -> list[str]:
    return list(self._stages[-1])

  @property
  def goal(self) -> dict[str, str]:
    return dict(self._level['goal'])

  def temperament(self, name: str) -> str:
    return self._orders._data['crew'][name]['temperament']  # pylint: disable=protected-access

  def describe(self, name: str) -> str:
    """How `name`'s temperament is described to the models."""
    data = self._orders._data  # pylint: disable=protected-access
    return ' '.join(
        str(data['temperaments'][self.temperament(name)]['describe']).split()
    )

  def relaying_law(self, name: str) -> str:
    """How `name` passes things on, as the relayer's prompt states it."""
    data = self._orders._data  # pylint: disable=protected-access
    how = data['temperaments'][self.temperament(name)].get('relays')
    return ' '.join(str(data['relaying']['laws'][how]).split())

  def pass_up(self, message: str) -> dict[str, Any]:
    """Sends `message` up the line; returns every hop and what was done."""
    heard = message
    hops = []
    for who in self.relayers:
      said = self._orders.relay(heard, who) if heard else ''
      hops.append({'who': who, 'heard': heard, 'said': said})
      heard = said
    ship = copy.deepcopy(self._ship)
    for who in self.receivers:
      ship.heard[who] = heard
    acted = []
    for who in self.receivers:
      action, clause = self._orders.reading(ship, who, 1)
      wanted = self.goal.get(who)
      acted.append({
          'who': who,
          'heard': heard,
          'does': action,
          'why': clause,
          'wanted': wanted,
          'ok': wanted is None or wanted == action,
      })
    return {'hops': hops, 'acted': acted}

  def rehearse(self, message: str) -> dict[str, Any]:
    """How the first in line would pass `message` on (a preview).

    Only the first hop: a measured effect of the words, without what the
    top of the line would do with them, which is the puzzle.

    Args:
      message: The words the player might send.

    Returns:
      {'who': the first relayer, 'said': their words}, or {} when the line
      has no relayer (the player speaks to the top directly).
    """
    message = message.strip().replace('\u2019', "'")
    if ' '.join(re.findall(r'[a-z]+', message.lower())) in _HOLDS:
      message = ''
    if not self.relayers or not message:
      return {}
    if words_in(message) > int(self._data['max_words']):
      return {'error': f'over the limit of {self._data["max_words"]} words'}
    who = self.relayers[0]
    return {'who': who, 'said': self._orders.relay(message, who)}

  # ---- the rulebook ----------------------------------------------------------

  def actors(self) -> list[str]:
    return [SEAT]

  def max_steps(self) -> int:
    # Room for refused (over-long) messages: `done` ends the game.
    return int(self._data['attempts']) * 4

  def action_spec(self, actor: str) -> entity_lib.ActionSpec:
    line = ' -> '.join(
        f'{who} ({self.temperament(who)}: {self.relaying_law(who)})'
        for who in self.relayers
    )
    goal = '; '.join(f'{who}: {act}' for who, act in self.goal.items())
    call = (
        f'{self._level["task"]}\n'
        f'The line: you -> {line + " -> " if line else ""}'
        + ' and '.join(
            f'{who} ({self.temperament(who)}: {self.describe(who)})'
            for who in self.receivers
        )
        + ' (who act).\n'
        f'Goal: {goal}. Try {len(self.world["attempts"]) + 1} of'
        f' {self._data["attempts"]}.\n'
        f'Your message, {{name}}, of at most {self._data["max_words"]} words:'
    )
    return entity_lib.free_action_spec(call_to_action=call, tag='message')

  def apply(self, world: Any, actor: str, action: str) -> base.Receipt:
    message = action.strip().replace('\u2019', "'")
    if ' '.join(re.findall(r'[a-z]+', message.lower())) in _HOLDS:
      message = ''
    words = words_in(message)
    limit = int(self._data['max_words'])
    if words > limit:
      return base.Receipt(
          actor=actor,
          action=message,
          summary=(
              f'The message is {words} words, over the limit of {limit}:'
              ' say it in fewer. (No try used.)'
          ),
      )
    result = self.pass_up(message)
    solved = all(a['ok'] for a in result['acted'])
    world['attempts'].append(
        {'message': message, 'words': words, 'solved': solved, **result}
    )
    world['solved'] = world['solved'] or solved
    steps = [f'You say: "{message}".' if message else 'You say nothing.']
    for hop in result['hops']:
      steps.append(f'{hop["who"]} passes it on: "{hop["said"]}".')
    for a in result['acted']:
      mark = 'as asked' if a['ok'] else f'not {a["wanted"]}'
      steps.append(f'{a["who"]} hears "{a["heard"]}" and does: {a["does"]}'
                   f' ({mark}).')  # fmt: skip
    steps.append('Solved.' if solved else 'Not yet.')
    return base.Receipt(
        actor=actor,
        action=message,
        summary=' '.join(steps),
        steps=steps,
        delta={'solved': float(solved), 'words': float(words)},
    )

  def done(self) -> bool:
    return self.world['solved'] or len(self.world['attempts']) >= int(
        self._data['attempts']
    )

  def stars(self) -> int:
    """0 unsolved; 1 solved; 2 or 3 for few enough words."""
    best = [a['words'] for a in self.world['attempts'] if a['solved']]
    if not best:
      return 0
    earned = 1
    for star in (2, 3):
      wants = self._level.get('stars', {}).get(star)
      if not wants or min(best) > int(wants['max_words']):
        break
      earned = star
    return earned

  def outcome(self) -> dict[str, Any]:
    tries = len(self.world['attempts'])
    best = [a['words'] for a in self.world['attempts'] if a['solved']]
    return {
        'winner': SEAT if self.world['solved'] else None,
        'scenario': self.scenario,
        'stars': self.stars(),
        'tries': tries,
        'words': min(best) if best else None,
        'words_spoken': sum(a['words'] for a in self.world['attempts']),
        'score': float(self.stars()),
    }

  def view(self) -> dict[str, Any]:
    """What the page draws: the line, the goal, every try."""
    return {
        'kind': 'passiton',
        'scenario': self.scenario,
        'title': self._level.get('title', self.scenario),
        'beat': self._level.get('beat', ''),
        'task': self._level.get('task', ''),
        'lesson': self._level.get('lesson') if self.done() else None,
        'event': self._level.get('event', 'quiet'),
        'line': [
            {
                'who': who,
                'temperament': self.temperament(who),
                'how': (
                    self._data.get('relays', {}).get(
                        self.temperament(who), 'word_for_word'
                    )
                ),
                'passes': self.relaying_law(who),
            }
            for who in self.relayers
        ],
        'receivers': [
            {'who': who, 'temperament': self.temperament(who)}
            for who in self.receivers
        ],
        'goal': self.goal,
        'attempts': self.world['attempts'],
        'tries': int(self._data['attempts']),
        'max_words': int(self._data['max_words']),
        'solved': self.world['solved'],
        'done': self.done(),
        'stars': self.stars(),
        'star_rules': {
            str(star): {k: int(v) for k, v in wants.items()}
            for star, wants in sorted(self._level.get('stars', {}).items())
        },
        'say': dict(self._data.get('say', {})),
    }

  # ---- state -------------------------------------------------------------------

  def world_to_dict(self) -> Mapping[str, Any]:
    return copy.deepcopy(self.world)

  def world_from_dict(self, data: Mapping[str, Any]) -> None:
    self.world = copy.deepcopy(dict(data))
