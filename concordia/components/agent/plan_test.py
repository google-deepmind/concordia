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

"""Tests for restoring planning components."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.components.agent import constant
from concordia.components.agent import plan
from concordia.language_model import language_model
from concordia.typing import entity_component


class PlanTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('defaults', None, False, 2),
      ('goal_with_automatic_horizon', 'Goal', False, 2),
      ('goal_with_explicit_horizon', 'Goal', '24 hours', 1),
  )
  def test_act_after_json_state_roundtrip(
      self,
      goal_key: str | None,
      horizon: str | bool,
      expected_model_calls: int,
  ):
    model = mock.create_autospec(language_model.LanguageModel, instance=True)
    model.sample_text.return_value = 'Make a plan.'
    original = plan.Plan(
        model, goal_component_key=goal_key, force_time_horizon=horizon
    )
    restored = plan.Plan(model)
    restored.set_state(json.loads(json.dumps(original.get_state())))

    components: dict[str, entity_component.ContextComponent] = {'Plan': restored}
    if goal_key is not None:
      components[goal_key] = constant.Constant('Find food.')
    acting = mock.create_autospec(
        entity_component.ActingComponent, instance=True
    )
    acting.get_action_attempt.return_value = 'Take action.'
    agent = entity_agent.EntityAgent('Alice', acting, components)

    self.assertEqual(agent.act(), 'Take action.')
    # Automatic horizons require a model question before generating the plan.
    self.assertLen(model.sample_text.call_args_list, expected_model_calls)
    planning_prompt = model.sample_text.call_args_list[-1].kwargs['prompt']
    if horizon is False:
      self.assertNotIn(
          'time horizon over which to plan is False', planning_prompt
      )
    else:
      self.assertIn(
          'time horizon over which to plan is 24 hours', planning_prompt
      )


if __name__ == '__main__':
  absltest.main()
