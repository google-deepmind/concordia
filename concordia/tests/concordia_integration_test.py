# Copyright 2023 DeepMind Technologies Limited.
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

import datetime
from typing import List
from absl.testing import absltest
from absl.testing import parameterized
from concordia import components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.environment.metrics import common_sense_morality
from concordia.environment.metrics import goal_achievement
from concordia.environment.metrics import reputation
from concordia.tests import mock_model
import numpy as np


def embedder(text: str):
  del text
  return np.random.rand(16)


def _make_agent(
    name: str,
    model: mock_model.MockModel,
    clock: game_clock.MultiIntervalClock,
    game_master_instructions: str,
    mem_factory: blank_memories.MemoryFactory,
) -> basic_agent.BasicAgent:
  """Creates two agents with the same game master instructions."""
  mem = mem_factory.make_blank_memory()
  agent = basic_agent.BasicAgent(
      model,
      mem,
      name,
      clock,
      [
          components.constant.ConstantComponent(
              'Instructions:', game_master_instructions
          ),
          components.constant.ConstantComponent(
              'General knowledge:', 'this is a test'
          ),
          agent_components.observation.Observation('Alice', mem),
      ],
      verbose=True,
  )

  return agent


def _make_environment(
    model: mock_model.MockModel,
    clock: game_clock.MultiIntervalClock,
    players: List[basic_agent.BasicAgent],
    game_master_instructions: str,
    importance_model_gm: importance_function.ImportanceModel,
) -> game_master.GameMaster:
  """Creates a game master environment."""
  game_master_memory = associative_memory.AssociativeMemory(
      embedder, importance_model_gm.importance, clock=clock.now
  )
  player_names = [player.name for player in players]

  shared_memories = [
      'There is a hamlet named Riverbend.',
  ]

  shared_context = 'There is a hamlet named Riverbend.'

  instructions_construct = components.constant.ConstantComponent(
      game_master_instructions, 'Instructions'
  )
  facts_on_village = components.constant.ConstantComponent(
      ' '.join(shared_memories), 'General knowledge of Riverbend'
  )
  player_status = gm_components.player_status.PlayerStatus(
      clock.now, model, game_master_memory, player_names
  )

  mem_factory = blank_memories.MemoryFactory(
      model=model,
      embedder=embedder,
      importance=importance_model_gm.importance,
      clock_now=clock.now,
  )

  convo_externality = gm_components.conversation.Conversation(
      players,
      model,
      memory=game_master_memory,
      clock=clock,
      burner_memory_factory=mem_factory,
      components=[player_status],
      cap_nonplayer_characters=2,
      game_master_instructions=game_master_instructions,
      shared_context=shared_context,
      verbose=False,
  )

  direct_effect_externality = gm_components.direct_effect.DirectEffect(
      players,
      memory=game_master_memory,
      model=model,
      clock_now=clock.now,
      verbose=False,
      components=[player_status],
  )

  debug_event_time = datetime.datetime(hour=14, year=2024, month=10, day=1)

  schedule = {
      'start': gm_components.schedule.EventData(
          time=datetime.datetime(hour=9, year=2024, month=10, day=1),
          description='',
      ),
      'debug_event': gm_components.schedule.EventData(
          time=debug_event_time,
          description='Debug event',
      ),
  }

  schedule_construct = gm_components.schedule.Schedule(
      clock_now=clock.now, schedule=schedule
  )
  player_goals = {'Alice': 'win', 'Bob': 'win'}
  goal_metric = goal_achievement.GoalAchievementMetric(
      model, player_goals, clock, 'Goal achievement', verbose=False
  )
  morality_metric = common_sense_morality.CommonSenseMoralityMetric(
      model, players, clock, 'Morality', verbose=False
  )
  reputation_metric = reputation.ReputationMetric(
      model, players, clock, 'Reputation', verbose=False
  )

  env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      players=players,
      components=[
          instructions_construct,
          facts_on_village,
          player_status,
          schedule_construct,
          convo_externality,
          direct_effect_externality,
      ],
      measurements=[goal_metric, morality_metric, reputation_metric],
      randomise_initiative=True,
      player_observes_event=False,
      verbose=False,
  )
  return env


class GameMasterTest(parameterized.TestCase):

  def test_full_run(self):
    model = mock_model.MockModel()

    importance_model = importance_function.ConstantImportanceModel()

    clock = game_clock.MultiIntervalClock(
        start=datetime.datetime(hour=8, year=2024, month=9, day=1),
        step_sizes=[
            datetime.timedelta(hours=1),
            datetime.timedelta(seconds=10),
        ],
    )

    game_master_instructions = 'This is a social science experiment.'

    mem_factory = blank_memories.MemoryFactory(
        model=model,
        embedder=embedder,
        importance=importance_model.importance,
        clock_now=clock.now,
    )

    alice = _make_agent(
        name='Alice',
        model=model,
        clock=clock,
        game_master_instructions=game_master_instructions,
        mem_factory=mem_factory,
    )
    bob = _make_agent(
        name='Bob',
        model=model,
        clock=clock,
        game_master_instructions=game_master_instructions,
        mem_factory=mem_factory,
    )

    players = [alice, bob]

    env = _make_environment(
        model,
        clock,
        players,
        game_master_instructions,
        importance_model,
    )

    env.run_episode(12)


if __name__ == '__main__':
  absltest.main()
