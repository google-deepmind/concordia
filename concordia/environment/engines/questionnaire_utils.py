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

"""Shared helpers for questionnaire engines."""

from collections.abc import Mapping

from absl import logging
from concordia.typing import entity as entity_lib


def parse_next_acting_entities(
    player_names_str: str,
    entities_by_name: Mapping[str, entity_lib.Entity],
    *,
    engine_name: str,
) -> list[entity_lib.Entity]:
  """Parses a NEXT_ACTING model response into known entities.

  Args:
    player_names_str: Raw comma-separated player names from the game master.
    entities_by_name: Known entities keyed by name.
    engine_name: Name of the engine using this parser, for logging context.

  Returns:
    A list of known entities in the same order returned by the game master.
    Empty names and unknown names are dropped.
  """
  next_entity_names = [
      name.strip() for name in player_names_str.split(',') if name.strip()
  ]
  invalid_entity_names = [
      name for name in next_entity_names if name not in entities_by_name
  ]
  if invalid_entity_names:
    logging.warning(
        '[%s] Ignoring unknown entity names from game master: %s',
        engine_name,
        invalid_entity_names,
    )
  return [
      entities_by_name[name]
      for name in next_entity_names
      if name in entities_by_name
  ]
