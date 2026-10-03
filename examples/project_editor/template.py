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

"""Compare standard minimal/basic player entities through supported JSON fields."""

import dataclasses
from typing import Any

from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import dialogic
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_components
from concordia.utils import project_config

TEMPLATE_KEY = 'entity-workbench-v1'
STRUCTURAL_TEMPLATE_KEY = 'scene-builder-v1'
PREVIOUS_TEMPLATE_KEY = 'conversation-v2'
LEGACY_TEMPLATE_KEY = 'conversation-v1'


def _legacy_config() -> prefab_lib.Config:
  """Preserve saved two-minimal-entity projects without changing their meaning."""
  return prefab_lib.Config(
      prefabs={'minimal': minimal.Entity(), 'dialogic': dialogic.GameMaster()},
      instances=[
          prefab_lib.InstanceConfig(
              prefab='minimal',
              role=prefab_lib.Role.ENTITY,
              params={
                  'name': name,
                  'custom_instructions': instructions,
                  'goal': '',
                  'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
                  # InstanceConfig's str annotation predates this prefab's bool.
              },
          )
          for name, instructions in [
              (
                  'Alice',
                  (
                      'You are Alice, a university roommate. Describe your'
                      ' current musical interests and listen to your roommate.'
                  ),
              ),
              (
                  'Bob',
                  (
                      'You are Bob, a university roommate. Discuss music'
                      ' without assuming that either person changes their'
                      ' tastes.'
                  ),
              ),
          ]
      ]
      + [
          prefab_lib.InstanceConfig(
              prefab='dialogic',
              role=prefab_lib.Role.GAME_MASTER,
              params={
                  'name': 'Conversation',
                  'next_game_master_name': 'Conversation',
                  'acting_order': 'fixed',
                  'can_terminate_simulation': (
                      False  # pyrefly: ignore[bad-assignment]
                  ),
              },
          )
      ],
      default_premise=(
          'Two university roommates are discussing what music to listen to in'
          ' their shared kitchen.'
      ),
      default_max_steps=4,
  )


def make_config() -> prefab_lib.Config:
  """Contrast Alice's minimal context with Bob's three-question reasoning."""
  legacy = _legacy_config()
  bob = prefab_lib.InstanceConfig(
      prefab='basic',
      role=prefab_lib.Role.ENTITY,
      params={
          'name': 'Bob',
          'goal': (
              'Bob is a university student who enjoys acoustic folk music.'
              ' He wants to find music he and Alice can both enjoy without'
              ' assuming that either roommate changes their tastes.'
          ),
          'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
      },
  )
  return dataclasses.replace(
      legacy,
      default_max_steps=40,
      prefabs={**legacy.prefabs, 'basic': basic.Entity()},
      instances=[
          dataclasses.replace(
              legacy.instances[0],
              params={
                  **legacy.instances[0].params,
                  'custom_instructions': (
                      'Alice is a university student who loves dubstep.'
                      ' She cares little about what others think of her music,'
                      ' but shares a kitchen with her roommate Bob.'
                  ),
              },
          ),
          bob,
          legacy.instances[2],
      ],
  )


def validate(document: dict[str, Any]) -> None:
  """Constrain the actual dialogic prefab enum, not its generated behavior."""
  for gm in document['instances']:
    if (
        gm['role'] != prefab_lib.Role.GAME_MASTER.value
        or gm['prefab'] != 'dialogic'
    ):
      continue
    if gm['params']['acting_order'] not in (
        'fixed',
        'random',
        'game_master_choice',
    ):
      raise project_config.ValidationError(
          f"$.instances[{gm['id']}].params.acting_order",
          'expected fixed, random or game_master_choice',
      )


def registry() -> project_config.Registry:
  """Trusted prefab prototypes; imported JSON chooses no Python constructors."""
  return project_config.Registry({
      key: project_config.Template(
          factory=factory,
          instance_ids=('alice', 'bob', 'conversation'),
          component_types=project_components.standard_types(
              ('alice', 'bob'), ('alice', 'bob')
          )
          if key == TEMPLATE_KEY
          else {},
          references={
              (
                  'conversation',
                  'next_game_master_name',
              ): prefab_lib.Role.GAME_MASTER
          },
          validate=validate,
          editable_instances=key in (TEMPLATE_KEY, STRUCTURAL_TEMPLATE_KEY),
          inspector={
              'alice': {
                  'name': {'label': 'Entity name'},
                  'custom_instructions': {
                      'label': 'Instructions · initial text'
                  },
                  'goal': {
                      'label': (
                          'Goal · initial text (empty removes the optional'
                          ' component)'
                      )
                  },
                  'randomize_choices': {
                      'label': 'Acting policy · randomize choices'
                  },
              },
              'bob': {
                  'name': {'label': 'Entity name'},
                  'goal': {'label': 'Goal · initial text'},
                  'randomize_choices': {
                      'label': 'Acting policy · randomize choices'
                  },
              },
              'conversation': {
                  'name': {'label': 'Game master entity name'},
                  'acting_order': {
                      'label': 'Acting order',
                      'choices': ['fixed', 'random', 'game_master_choice'],
                  },
                  'can_terminate_simulation': {
                      'label': 'Allow the game master to end the conversation'
                  },
                  'next_game_master_name': {'label': 'Next game master entity'},
              },
          },
      )
      for key, factory in (
          (TEMPLATE_KEY, make_config),
          (STRUCTURAL_TEMPLATE_KEY, make_config),
          (PREVIOUS_TEMPLATE_KEY, make_config),
          (LEGACY_TEMPLATE_KEY, _legacy_config),
      )
  })
