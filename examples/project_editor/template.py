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

"""Ordinary JSON fields on existing minimal actors and dialogic GM."""

from typing import Any

from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import dialogic
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_config

TEMPLATE_KEY = 'conversation-v1'


def make_config() -> prefab_lib.Config:
  """Fresh prefab definitions, no pre-bound or serialized components."""
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


def validate(document: dict[str, Any]) -> None:
  """Constrain the actual dialogic prefab enum, not its generated behavior."""
  gm = next(
      item for item in document['instances'] if item['id'] == 'conversation'
  )
  if gm['params']['acting_order'] not in (
      'fixed',
      'random',
      'game_master_choice',
  ):
    raise project_config.ValidationError(
        '$.instances[conversation].params.acting_order',
        'expected fixed, random or game_master_choice',
    )


def registry() -> project_config.Registry:
  """Caller-owned fixed template registry; imported JSON chooses no code."""
  return project_config.Registry({
      TEMPLATE_KEY: project_config.Template(
          factory=make_config,
          instance_ids=('alice', 'bob', 'conversation'),
          references={
              (
                  'conversation',
                  'next_game_master_name',
              ): prefab_lib.Role.GAME_MASTER
          },
          validate=validate,
      )
  })
