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

r"""Matrix audit: counts the interaction matrices and checks their evidence.

Reads data/orders_matrix.yaml. Every system cell and reading cell must name
a test that exists (a `def test_...` in examples/crucible) or golden cases
that exist; every cell's systems, operators and properties must be declared.
Prints the counts quoted in cycle reports.

  python -m examples.crucible.audit
"""

import glob
import json
import os
import re
from typing import Any

from examples.crucible import toys

_HERE = os.path.dirname(__file__)


def test_names() -> set[str]:
  names = set()
  for path in glob.glob(os.path.join(_HERE, '**', '*_test.py'), recursive=True):
    with open(path) as f:
      names.update(re.findall(r'def (test_\w+)\(', f.read()))
  return names


def audit() -> dict[str, Any]:
  matrix = toys.load_data('orders_matrix')
  tests = test_names()
  golden_ids = {case['id'] for case in matrix['golden']}
  problems = []
  systems = matrix['systems']
  for cell in matrix['cells']:
    for end in ('from', 'to'):
      if cell[end] not in systems:
        problems.append(f'unknown system {cell[end]!r}')
  reading = matrix['reading']
  for cell in reading['cells']:
    if cell['operator'] not in reading['operators']:
      problems.append(f'unknown operator {cell["operator"]!r}')
    if cell['property'] not in reading['properties']:
      problems.append(f'unknown property {cell["property"]!r}')
  for cell in [*matrix['cells'], *reading['cells']]:
    test, golden = cell.get('test'), cell.get('golden', [])
    if not test and not golden:
      problems.append(f'cell without evidence: {cell}')
    if test and test not in tests:
      problems.append(f'missing test {test}')
    problems.extend(
        f'missing golden {g}' for g in golden if g not in golden_ids
    )
  # Every campaign scene names the cells it teaches ('from->to' for system
  # cells, 'operator x property' for reading cells) and its lesson.
  refs = {f"{c['from']}->{c['to']}" for c in matrix['cells']} | {
      f"{c['operator']} x {c['property']}" for c in reading['cells']
  }
  data = toys.load_data('orders')
  taught = set()
  for name in data.get('campaign', []):
    scene = data['scenarios'][name]
    if not scene.get('lesson'):
      problems.append(f'scene {name!r} has no lesson')
    for ref in scene.get('teaches', []):
      if ref not in refs:
        problems.append(f'scene {name!r} teaches unknown cell {ref!r}')
      taught.add(ref)
  pairs = {(c['from'], c['to']) for c in matrix['cells']}
  return {
      'counts': {
          'systems': len(systems),
          'system_cells': len(matrix['cells']),
          'system_pairs': len(pairs),
          'reading_cells': len(reading['cells']),
          'matrix_cells': len(matrix['cells']) + len(reading['cells']),
          'operators': len(reading['operators']),
          'properties': len(reading['properties']),
          'golden_cases': len(golden_ids),
          'hard_golden_cases': sum(
              bool(c.get('hard')) for c in matrix['golden']
          ),
          # Cells checked both ways: a model's reading and the rule's test.
          'cell_refs': len(refs),
          'cell_refs_taught_by_scenes': len(taught & refs),
          'doubly_evidenced_cells': sum(
              bool(c.get('test') and c.get('golden'))
              for c in [*matrix['cells'], *reading['cells']]
          ),
      },
      'untaught': sorted(refs - taught),
      'problems': problems,
  }


if __name__ == '__main__':
  print(json.dumps(audit()))
