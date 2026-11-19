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

"""Rounds on the asynchronous engine: once each, then all together."""

from collections.abc import Mapping
import json
import threading
import time
from typing import override

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.contrib.components.game_master import rounds as rounds_lib
from concordia.environment.engines import asynchronous
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import async_measurements

_PLAYERS = ('Ada', 'Bo', 'Cy')


class Ledger(entity_component.ContextComponent):
  """A resolver: keeps every round's answers, as resolved."""

  def __init__(self):
    super().__init__()
    self.rounds: list[tuple[int, dict[str, str]]] = []
    self.lock = threading.Lock()

  def resolve_round(self, round_number: int, answers: Mapping[str, str]):
    with self.lock:
      self.rounds.append((round_number, dict(answers)))

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class CountingAct(entity_component.ActingComponent):
  """Answers with its name and how often it has acted, sometimes slowly."""

  def __init__(self, delay: float = 0.0):
    super().__init__()
    self._acted = 0
    self._delay = delay

  @override
  def get_action_attempt(self, context, action_spec) -> str:
    del context, action_spec
    time.sleep(self._delay)
    self._acted += 1
    return f'move {self._acted}'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _game(max_rounds: int, deadline: float | None = None, slow: str = ''):
  ledger = Ledger()
  rounds = rounds_lib.Rounds(
      participants=_PLAYERS,
      resolver_key='ledger',
      max_rounds=max_rounds,
      deadline=deadline,
  )
  model = no_language_model.NoLanguageModel()
  gm = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name='GM',
      act_component=switch_act.SwitchAct(model=model, entity_names=_PLAYERS),
      context_components={
          next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY: rounds,
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: rounds,
          next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: (
              next_acting.FixedActionSpec(
                  entity_lib.free_action_spec(call_to_action='Your move?')
              )
          ),
          make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
              make_observation.MakeObservation(
                  model=model,
                  player_names=_PLAYERS,
                  allow_llm_fallback=False,
              )
          ),
          terminate.DEFAULT_TERMINATE_COMPONENT_KEY: terminate.Terminate(),
          'ledger': ledger,
      },
      measurements=async_measurements.ReactiveMeasurements(),
  )
  players = [
      entity_agent_with_logging.EntityAgentWithLogging(
          agent_name=name,
          act_component=CountingAct(delay=2.0 if name == slow else 0.0),
          context_components={},
          measurements=async_measurements.ReactiveMeasurements(),
      )
      for name in _PLAYERS
  ]
  return gm, players, rounds, ledger


def _play(gm, players) -> None:
  asynchronous.Asynchronous(sleep_time=0.001).run_loop(
      game_masters=[gm],
      entities=players,
      premise='',
      max_steps=100_000,
      verbose=False,
  )


class RoundsTest(absltest.TestCase):

  def test_everyone_answers_once_a_round_and_all_resolve_together(self):
    gm, players, rounds, ledger = _game(max_rounds=4)

    _play(gm, players)

    self.assertEqual([number for number, _ in ledger.rounds], [0, 1, 2, 3])
    for number, answers in ledger.rounds:
      self.assertEqual(list(answers), sorted(_PLAYERS))
      self.assertEqual(set(answers.values()), {f'move {number + 1}'})
    self.assertEqual(rounds.round_number, 4)

  def test_the_outcome_never_depends_on_thread_timing(self):
    outcomes = []
    for _ in range(3):
      gm, players, _, ledger = _game(max_rounds=3)
      _play(gm, players)
      outcomes.append(json.dumps(ledger.rounds))
    self.assertLen(set(outcomes), 1)

  def test_with_a_deadline_a_round_closes_without_the_slow(self):
    gm, players, _, ledger = _game(max_rounds=2, deadline=0.2, slow='Cy')
    started = time.monotonic()

    _play(gm, players)

    self.assertLess(time.monotonic() - started, 30)
    first = ledger.rounds[0][1]
    self.assertEqual(sorted(first), ['Ada', 'Bo'])

  def test_state_round_trips(self):
    _, _, rounds, _ = _game(max_rounds=5)
    rounds.pre_observe('[putative_event] Ada: wait')
    rounds.pre_observe('[putative_event] Zed: not a participant')
    rounds.set_participants(['Ada', 'Bo'])

    other = rounds_lib.Rounds(participants=(), resolver_key='ledger')
    other.set_state(json.loads(json.dumps(rounds.get_state())))

    self.assertEqual(
        other.get_state(),
        {
            'round': 0,
            'answers': {'Ada': 'wait'},
            'participants': ['Ada', 'Bo'],
            'ended': False,
        },
    )


if __name__ == '__main__':
  absltest.main()
