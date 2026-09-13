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

"""Trusted Bellwether configuration, using standard actor and GM build paths."""

import json
from typing import Any

from concordia.components.agent import concat_act_component
from concordia.components.agent import constant
from concordia.components.agent import human_act_component
from concordia.components.game_master import make_observation
from concordia.components.game_master import switch_act
from concordia.environment import engine
from concordia.examples.bellwether import game
from concordia.examples.bellwether import scenario
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


class FixtureModel(no_language_model.NoLanguageModel):
  """Deliberately cooperative fixture, not live residents or social evidence."""

  def sample_text(self, prompt, **kwargs):
    del kwargs
    # Read only the final task, not earlier observations. No external model.
    task = prompt.rsplit('Respond to this delivered task', 1)[-1]
    decision = 'speak'
    if '"purpose": "request:' in task:
      decision = 'accept'
    elif '"purpose": "work:' in task:
      decision = 'perform'
    return json.dumps({
        'decision': decision,
        'speech': {
            'speak': 'I am concerned for the people who rely on my facility.',
            'accept': 'I accept this specific request. Record my commitment.',
            'perform': (
                'I choose to perform my accepted repair if the required part is'
                ' here.'
            ),
        }[decision],
    })


class Resident(prefab_lib.Prefab):
  """Fresh standard minimal/basic components and the normal ConcatAct policy."""

  description = (
      'A live Bellwether resident; identity and decision logic are'
      ' configurable.'
  )

  def build(self, model, memory_bank, *, action_model=None):
    params: dict[str, Any] = dict(self.params)
    style = params.pop('decision_logic')
    account = params.pop('account')
    institutions = params.pop('institutions', game.INSTITUTIONS)
    reader = params.pop('human_reader', None)
    params['extra_components'] = {
        scenario.ACCOUNT: constant.Constant(
            account, pre_act_label='Previous storm account'
        ),
        'Affiliations': constant.Constant(
            json.dumps(
                [x for x in institutions if params['name'] in x['members']],
                ensure_ascii=False,
            ),
            pre_act_label='Institutions known to me',
        ),
    }
    if style == 'basic':
      # Basic keeps its own standard instructions; add the resident's scenario
      # instructions as a context, using the same extension mechanism.
      params['extra_components']['ResidentInstructions'] = constant.Constant(
          params.pop('custom_instructions'),
          pre_act_label='Resident instructions',
      )
    builder = basic.Entity if style == 'basic' else minimal.Entity
    policy = None
    if reader is not None:

      def human_policy(order):
        return human_act_component.HumanActComponent(
            reader, component_order=order
        )

      policy = human_policy
    elif action_model is not None:

      def model_policy(order):
        return concat_act_component.ConcatActComponent(
            action_model, component_order=order, prefix_entity_name=False
        )

      policy = model_policy
    return builder(params=params).build(
        model, memory_bank, act_component_factory=policy
    )


def configuration(
    reader,
    world,
    *,
    actor_logic='minimal',
    human_readers=None,
    action_model=None,
):
  """Build fresh components per service; this Config is single-build only."""
  if actor_logic not in ('minimal', 'basic'):
    raise ValueError(
        'Choose standard minimal or basic resident decision logic.'
    )

  bound_action_model = action_model

  class ConfiguredResident(Resident):
    """Runtime-only model binding, never stored in the JSON instance params."""

    def build(self, model, memory_bank, *, action_model=bound_action_model):
      return super().build(model, memory_bank, action_model=action_model)

  class Coordinator(prefab_lib.Prefab):
    """Use the transport-neutral human policy with standard context."""

    description = 'Human emergency coordinator.'

    def build(self, model, memory_bank):
      def human_policy(order):
        return human_act_component.HumanActComponent(
            reader, component_order=order
        )

      return minimal.Entity(params=self.params).build(
          model, memory_bank, act_component_factory=human_policy
      )

  class GameMaster(prefab_lib.Prefab):
    """Compose standard routing with the scenario’s explicit rules."""

    description = (
        'Explicit conservation and consent with standard SwitchAct routing.'
    )

    def build(self, model, memory_bank):
      output = entity_lib.OutputType

      def signal(kind, value):
        return game.Signal(world, kind, value)

      components = {
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: world,
          'Inventory': world.stock,
          switch_act.DEFAULT_NEXT_ACTING_COMPONENT_KEY: signal(
              output.NEXT_ACTING, lambda: world.next_actor
          ),
          switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: signal(
              output.NEXT_ACTION_SPEC,
              lambda: engine.action_spec_to_string(
                  entity_lib.free_action_spec(
                      call_to_action=world.action_prompt()
                  )
              ),
          ),
          switch_act.DEFAULT_TERMINATE_COMPONENT_KEY: signal(
              output.TERMINATE, lambda: 'Yes' if world.finished else 'No'
          ),
          switch_act.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
              make_observation.MakeObservation(
                  model,
                  player_names=game.NAMES,
                  allow_llm_fallback=False,
                  external_queue=world.observations,
              )
          ),
      }
      return minimal.Entity(
          params={
              'name': game.GM,
              'custom_instructions': (
                  'Enforce declared rules; never infer material or consent from'
                  ' prose.'
              ),
              'extra_components': components,  # pyrefly: ignore[bad-assignment]
          }
      ).build(
          model,
          memory_bank,
          act_component=switch_act.SwitchAct(model, entity_names=game.NAMES),
      )

  instances = [
      prefab_lib.InstanceConfig(
          prefab='coordinator',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': game.PLAYER,
              'custom_instructions': (
                  'Coordinate emergency response. Speak only for yourself.'
                  ' Looking around is free.'
              ),
          },
      )
  ]
  for name, (goal, account) in scenario.RESIDENTS.items():
    instances.append(
        prefab_lib.InstanceConfig(
            prefab='resident',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': name,
                'goal': goal,
                'account': account,
                'decision_logic': actor_logic,
                # Final resident actions are JSON objects, not name-prefixed
                # sentence completions. Use the standard ConcatAct option.
                # InstanceConfig's legacy str annotation omits bool options.
                'prefix_entity_name': False,  # pyrefly: ignore[bad-assignment]
                'institutions': world.institutions,
                **(
                    {'human_reader': human_readers[name]}
                    if human_readers and name in human_readers
                    else {}
                ),
                'custom_instructions': (
                    f'You are {name}, a resident of Bellwether. {goal} Make'
                    ' your own decisions based on your memories and delivered'
                    ' observations. No narrator may commit you without your'
                    ' explicit acceptance. You may accept, refuse, negotiate'
                    ' or revoke your unfulfilled commitment. A spoken claim is'
                    ' not a completed physical action. Do not repeat internal'
                    ' PRIVATE_ markers in speech.'
                ),
            },
        )
    )
  instances.append(
      prefab_lib.InstanceConfig(
          prefab='game_master',
          role=prefab_lib.Role.GAME_MASTER,
          params={'name': game.GM},
      )
  )
  return prefab_lib.Config(
      default_max_steps=64,
      default_premise='',
      prefabs={
          'coordinator': Coordinator(),
          'resident': ConfiguredResident(),
          'game_master': GameMaster(),
      },
      instances=instances,
  )
