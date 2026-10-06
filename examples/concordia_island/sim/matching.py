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

"""Matching logic for dating feature in Halo simulation."""

from typing import Dict, List, Tuple


def compute_mutual_matches(selections: Dict[str, str]) -> List[Tuple[str, str]]:
  """Computes mutual matches from a dictionary of selections.

  Args:
    selections: A dict mapping agent name -> selected partner name.

  Returns:
    A list of tuples, each containing a matched pair of agent names.
  """
  matches = []
  seen = set()
  for selector, target in selections.items():
    if target in selections and selections[target] == selector:
      pair = tuple(sorted([selector, target]))
      if pair not in seen:
        matches.append(pair)
        seen.add(pair)
  return matches


def get_unmatched_singles(
    all_singles: List[str], matches: List[Tuple[str, str]]
) -> List[str]:
  """Returns a list of singles who were not matched.

  Args:
    all_singles: List of all single agent names.
    matches: List of matched pairs.

  Returns:
    List of unmatched agent names.
  """
  matched_agents = set()
  for p1, p2 in matches:
    matched_agents.add(p1)
    matched_agents.add(p2)

  return [name for name in all_singles if name not in matched_agents]
