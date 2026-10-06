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

"""Tests for the simultaneous resolution game master prefab."""

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.components import agent as agent_components
from concordia.components.game_master import next_acting
from concordia.contrib.prefabs.game_master import simultaneous_resolution_gm
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
import numpy as np

_PLAYERS = ('Old Tom', 'Hilda')


def _build(**params):
  model = no_language_model.NoLanguageModel()
  players = [
      entity_agent_with_logging.EntityAgentWithLogging(
          agent_name=name,
          act_component=agent_components.concat_act_component.ConcatActComponent(
              model=model
          ),
          context_components={},
      )
      for name in _PLAYERS
  ]
  prefab = simultaneous_resolution_gm.GameMasterSimultaneous()
  prefab.params = {**prefab.params, **params}
  prefab.entities = players
  return prefab.build(
      model=model,
      memory_bank=associative_memory.AssociativeMemoryBank(
          sentence_embedder=lambda _: np.ones(3)
      ),
  )


def _clock(**kwargs):
  return simultaneous_resolution_gm.FixedIncrementClock(
      model=no_language_model.NoLanguageModel(), **kwargs
  )


class SimultaneousResolutionGameMasterTest(absltest.TestCase):

  def test_builds_with_default_parameters(self):
    game_master = _build()
    clock = game_master.get_component(
        'generative_clock', type_=simultaneous_resolution_gm.FixedIncrementClock
    )
    self.assertEqual(
        clock.get_state()['start_time'], 'Tuesday, March 03, 2026 at 08:30 AM'
    )

  def test_missing_start_time_uses_default_time(self):
    clock = _clock(start_time=None)
    self.assertEqual(
        clock.get_state()['start_time'], 'Tuesday, March 03, 2026 at 08:30 AM'
    )

  def test_fixed_increments_advance_by_the_configured_period(self):
    clock = _clock(
        start_time='Friday, March 20, 2026 at 06:00 AM',
        increment_minutes=1440,
        use_variable_increments=False,
    )
    clock.advance_by_minutes(1440)
    self.assertIn('March 21, 2026 at 06:00 AM', clock.get_pre_act_value())

  def test_call_to_action_parameter_replaces_default_plan_request(self):
    question = 'How many fish does {name} catch today?'
    game_master = _build(call_to_action=question)
    action_spec = game_master.get_component(
        next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY,
        type_=next_acting.FixedActionSpec,
    ).pre_act(
        entity_lib.ActionSpec(
            call_to_action='',
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
    self.assertIn(question, action_spec)
    self.assertNotIn('detailed plan', action_spec)


if __name__ == '__main__':
  absltest.main()
