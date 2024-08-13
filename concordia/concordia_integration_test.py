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

from collections.abc import Sequence
import datetime

from absl.testing import absltest
from absl.testing import parameterized
from concordia import components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.components.agent import to_be_deprecated as agent_components
from concordia.environment import game_master
from concordia.metrics import common_sense_morality
from concordia.metrics import goal_achievement
from concordia.metrics import opinion_of_others
from concordia.testing import mock_model
import numpy as np


def embedder(text: str):
  del text
  return np.random.rand(16)


def _make_agent(
    name: str,
    model: mock_model.MockModel,
    clock: game_clock.MultiIntervalClock,
    player_names: Sequence[str],
    agent_instructions: str,
    mem_factory: blank_memories.MemoryFactory,
) -> basic_agent.BasicAgent:
  """Creates two agents with same instructions."""
  mem = mem_factory.make_blank_memory()

  goal_metric = goal_achievement.GoalAchievementMetric(
      model=model,
      player_name=name,
      player_goal='win',
      clock=clock,
  )
  morality_metric = common_sense_morality.CommonSenseMoralityMetric(
      model=model,
      player_name=name,
      clock=clock,
  )

  time = components.report_function.ReportFunction(
      name='Current time',
      function=clock.current_time_interval_str,
  )
  somatic_state = agent_components.somatic_state.SomaticState(
      model=model,
      memory=mem,
      agent_name=name,
      clock_now=clock.now,
  )
  identity = agent_components.identity.SimIdentity(
      model=model,
      memory=mem,
      agent_name=name,
      clock_now=clock.now,
  )
  goal_component = components.constant.ConstantComponent(state='test')
  plan = agent_components.plan.SimPlan(
      model=model,
      memory=mem,
      agent_name=name,
      clock_now=clock.now,
      components=[identity],
      goal=goal_component,
      verbose=False,
  )

  self_perception = agent_components.self_perception.SelfPerception(
      name='self perception',
      model=model,
      memory=mem,
      agent_name=name,
      clock_now=clock.now,
      verbose=True,
  )
  situation_perception = (
      agent_components.situation_perception.SituationPerception(
          name='situation perception',
          model=model,
          memory=mem,
          agent_name=name,
          clock_now=clock.now,
          verbose=True,
      )
  )
  person_by_situation = agent_components.person_by_situation.PersonBySituation(
      name='person by situation',
      model=model,
      memory=mem,
      agent_name=name,
      clock_now=clock.now,
      components=[self_perception, situation_perception],
      verbose=True,
  )
  persona = components.sequential.Sequential(
      name='persona',
      components=[
          self_perception,
          situation_perception,
          person_by_situation,
      ],
  )

  observation = agent_components.observation.Observation(
      agent_name=name,
      clock_now=clock.now,
      memory=mem,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )
  observation_summary = agent_components.observation.ObservationSummary(
      agent_name=name,
      model=model,
      clock_now=clock.now,
      memory=mem,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      components=[identity],
      component_name='summary of observations',
  )

  agent = basic_agent.BasicAgent(
      model,
      name,
      clock,
      [
          components.constant.ConstantComponent(
              'Instructions:', agent_instructions
          ),
          persona,
          observation,
          observation_summary,
          plan,
          somatic_state,
          time,
          goal_metric,
          morality_metric,
      ],
      verbose=True,
  )
  reputation_metric = opinion_of_others.OpinionOfOthersMetric(
      model=model,
      player_name=name,
      player_names=player_names,
      context_fn=agent.state,
      clock=clock,
  )
  agent.add_component(reputation_metric)

  return agent


def _make_environment(
    model: mock_model.MockModel,
    clock: game_clock.MultiIntervalClock,
    players: Sequence[basic_agent.BasicAgent],
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

  env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      players=players,
      components=[
          facts_on_village,
          player_status,
          schedule_construct,
          convo_externality,
          direct_effect_externality,
      ],
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

    agent_instructions = 'This is a social science experiment.'

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
        player_names=['Alice', 'Bob'],
        agent_instructions=agent_instructions,
        mem_factory=mem_factory,
    )
    bob = _make_agent(
        name='Bob',
        model=model,
        clock=clock,
        player_names=['Alice', 'Bob'],
        agent_instructions=agent_instructions,
        mem_factory=mem_factory,
    )

    players = [alice, bob]

    env = _make_environment(
        model=model,
        clock=clock,
        players=players,
        importance_model_gm=importance_model,
    )

    env.run_episode(12)


if __name__ == '__main__':
  absltest.main()
