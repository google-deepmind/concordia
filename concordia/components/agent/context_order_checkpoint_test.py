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

"""Acting context order survives public agent checkpoint restoration."""

import json
from unittest import mock

from absl.testing import parameterized
from concordia.agents import entity_agent
from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.agent import image_text_act_component
from concordia.components.game_master import switch_act
from concordia.language_model import no_language_model
from concordia.typing import entity


class ContextOrderCheckpointTest(parameterized.TestCase):

  def make_agent(self, kind, order):
    model = mock.Mock(wraps=no_language_model.NoLanguageModel())
    reader = mock.Mock(return_value='LOOK')
    if kind == 'switch':
      actor = switch_act.SwitchAct(
          model, entity_names=('Ada',), component_order=order
      )
    elif kind == 'human':
      actor = human_act_component.HumanActComponent(
          reader, component_order=order
      )
    else:
      cls = (
          concat_act_component.ConcatActComponent
          if kind == 'concat'
          else image_text_act_component.ImageTextActComponent
      )
      actor = cls(model, component_order=order, randomize_choices=False)
    agent = entity_agent.EntityAgent(
        'Ada',
        actor,
        {
            'z': constant.Constant('Z', prefix_pre_act=False),
            'a': constant.Constant('A', prefix_pre_act=False),
        },
    )
    return agent, model, reader

  def act_context(self, kind, agent, model, reader):
    agent.act(
        entity.choice_action_spec(
            call_to_action='Choose', options=('LOOK', 'WAIT')
        )
    )
    if kind == 'human':
      return reader.call_args.args[0].context
    return (
        model.sample_choice.call_args.kwargs['prompt']
        .split('\nQuestion:')[0]
        .rstrip('\n')
    )

  @parameterized.product(
      kind=('concat', 'human', 'image', 'switch'),
      order=(None, [], ['a']),
  )
  def test_json_checkpoint_preserves_public_prompt(self, kind, order):
    agent, model, reader = self.make_agent(kind, order)
    expected = 'Z\nA' if order is None else 'A\nZ'
    self.assertEqual(self.act_context(kind, agent, model, reader), expected)
    checkpoint = json.loads(json.dumps(agent.get_state()))
    self.assertEqual(checkpoint['act_component']['component_order'], order)
    agent.set_state(checkpoint)
    self.assertEqual(self.act_context(kind, agent, model, reader), expected)

  @parameterized.product(
      kind=('concat', 'human', 'image', 'switch'),
      order=(None, [], ['a']),
  )
  def test_partial_state_restores_order_and_owns_list(self, kind, order):
    agent, model, reader = self.make_agent(kind, ['z'])
    restored = None if order is None else list(order)
    agent.set_state({'act_component': {'component_order': restored}})
    if restored is not None:
      restored.append('missing')
    expected = 'Z\nA' if order is None else 'A\nZ'
    self.assertEqual(self.act_context(kind, agent, model, reader), expected)
    agent.set_state({'act_component': {}})
    self.assertEqual(self.act_context(kind, agent, model, reader), expected)
    snapshot = agent.get_state()['act_component']['component_order']
    if snapshot is not None:
      snapshot.append('missing')
    self.assertEqual(self.act_context(kind, agent, model, reader), expected)
