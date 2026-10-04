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

"""Tests that grouped entity cards identify the same inspector instance."""

import itertools
import xml.etree.ElementTree as ET

from absl.testing import absltest
from absl.testing import parameterized
from concordia.typing import prefab as prefab_lib
from concordia.utils import visual_interface

_ROLES = (
    prefab_lib.Role.ENTITY,
    prefab_lib.Role.GAME_MASTER,
    prefab_lib.Role.INITIALIZER,
)
_NS = {"svg": "http://www.w3.org/2000/svg"}


class EntityIdentityTest(parameterized.TestCase):

  @parameterized.parameters(
      *itertools.permutations(_ROLES),
      (_ROLES[2], _ROLES[0], _ROLES[1], _ROLES[0], _ROLES[2], _ROLES[1]),
      (_ROLES[1], _ROLES[2], _ROLES[0], _ROLES[1], _ROLES[0], _ROLES[2]),
      *[(role,) for role in _ROLES],
      (),
  )
  def test_card_matches_inspector_after_role_grouping(self, *roles):
    instances = [
        prefab_lib.InstanceConfig(
            prefab=f"prefab_{index}",
            role=role,
            params={"name": f"Instance {index}", "goal": f"Goal {index}"},
        )
        for index, role in enumerate(roles)
    ]
    config = prefab_lib.Config(prefabs={}, instances=instances)
    checkpoint = {"entities": {}, "game_masters": {}}
    for instance in instances:
      group = (
          "entities"
          if instance.role == prefab_lib.Role.ENTITY
          else "game_masters"
      )
      checkpoint[group][instance.params["name"]] = {
          "component_info": {
              "context_components": {
                  "Marker": {
                      "class_name": "Constant",
                      "state": {
                          "state": instance.params["goal"],
                      },
                  },
              }
          },
      }

    svg, data = visual_interface.visualize_config(config, checkpoint)

    root = ET.fromstring(svg)
    cards = root.findall("svg:g[@class='entity-card']", _NS)
    self.assertLen(cards, len(instances))
    ids = [card.attrib["data-entity-id"] for card in cards]
    self.assertCountEqual(ids, data)
    expected_order = [
        instance
        for role in _ROLES
        for instance in instances
        if instance.role == role
    ]
    # The visual role grouping and order within each group must not change.
    self.assertEqual(
        [card.attrib["data-entity-name"] for card in cards],
        [instance.params["name"] for instance in expected_order],
    )
    for card, instance in zip(cards, expected_order):
      inspector = data[card.attrib["data-entity-id"]]
      self.assertEqual(inspector["name"], card.attrib["data-entity-name"])
      self.assertEqual(inspector["name"], instance.params["name"])
      self.assertEqual(inspector["prefab"], instance.prefab)
      self.assertEqual(inspector["role"], instance.role.value)
      self.assertEqual(
          inspector["component_info"]["context_components"]["Marker"]["state"][
              "state"
          ],
          instance.params["goal"],
      )


if __name__ == "__main__":
  absltest.main()
