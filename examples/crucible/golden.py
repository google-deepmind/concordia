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

r"""Golden sets: does a model read orders by the laws? (GUIDE.md, Phase 3.)

Each golden case in data/orders_matrix.yaml sets one situation (an event, a
member, the stores, fatigue), gives an order, and asks the model, through
the toy's own crew prompt, what the member does. A case passes when the
answer is one the laws allow. Hard cases follow from laws a model must not
break (names, conditions, the literal member); the set passes when every
hard case passes and at least `--soft` of the others agree.

  python -m examples.crucible.golden_cli --mode=mock
  python -m examples.crucible.golden_cli --mode=ollama --model_name=qwen3:8b
"""

import copy
import json
from typing import Any

from concordia.language_model import language_model

from examples.crucible import toys
from examples.crucible.toys.orders import toy as orders_lib

SOFT_PASS = 0.75


def load_cases() -> list[dict[str, Any]]:
  return list(toys.load_data('orders_matrix')['golden'])


def run_case(
    model: language_model.LanguageModel, case: dict[str, Any]
) -> dict[str, Any]:
  """Asks the model what the case's member does; returns the verdict."""
  data = copy.deepcopy(toys.load_data('orders'))
  data['events'] = {1: case['event']}
  # The whole crew serves in a golden case, whoever the scene usually has.
  data['scenarios']['voyage'].pop('crew_present', None)
  if 'crew' in case:
    data['scenarios']['voyage']['crew_present'] = list(case['crew'])
  if 'off_duty' in case:
    data['scenarios']['voyage']['off_duty'] = {m: [1] for m in case['off_duty']}
  toy = orders_lib.OrdersToy(model, data, 'voyage', seed=0)
  ship = toy.world
  ship.shift = 0
  ship.order = case['order']
  ship.supplies = int(case.get('supplies', 3))
  if case.get('worn'):
    ship.fatigue[case['member']] = int(data['effects']['tired_at'])
  for member, fatigue in case.get('fatigue', {}).items():
    ship.fatigue[member] = int(fatigue)
  if case['event'] == 'leak':
    ship.leak = True
  action = toy.choose(ship, case['member'], 1)
  return {
      'id': case['id'],
      'hard': bool(case.get('hard', False)),
      'expect': list(case['expect']),
      'answer': action,
      'ok': action in case['expect'],
  }


def run(
    model: language_model.LanguageModel, soft_pass: float = SOFT_PASS
) -> dict[str, Any]:
  """Runs every golden case; summarizes hard and soft results."""
  results = [run_case(model, case) for case in load_cases()]
  hard = [r for r in results if r['hard']]
  soft = [r for r in results if not r['hard']]
  soft_rate = sum(r['ok'] for r in soft) / len(soft) if soft else 1.0
  return {
      'cases': len(results),
      'hard_passed': sum(r['ok'] for r in hard),
      'hard_total': len(hard),
      'soft_agreement': round(soft_rate, 3),
      'passed': all(r['ok'] for r in hard) and soft_rate >= soft_pass,
      'failures': [r for r in results if not r['ok']],
  }


def main_json(model: language_model.LanguageModel) -> str:
  return json.dumps(run(model), indent=1)
