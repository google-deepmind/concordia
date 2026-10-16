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

"""Golden sets for Pass it on (cycle 102): reads and relays.

The cases live in `data/passiton_matrix.yaml` (`golden`). A read case asks
the model what one member does with a message (`PassItOnToy.pass_up` with
the member alone in the line). A relay case asks the model how a relayer
passes a message on, then reads what came out for a probe member by the
laws (the mock), so the case tests the relay alone.
"""

from collections.abc import Sequence
import copy
import json
import time
from typing import Any

from absl import logging
from concordia.contrib.testing import golden_set
from concordia.language_model import language_model
from examples.crucible import model_setup
from examples.crucible import toys
from examples.crucible.toys.passiton import toy as toy_lib

SOFT_PASS = 0.75
_LEVEL = '__golden__'


def _toy(
    model: language_model.LanguageModel,
    line: list[Any],
    goal: dict[str, str],
    event: str,
) -> toy_lib.PassItOnToy:
  data = copy.deepcopy(toys.load_data('passiton'))
  data['scenarios'][_LEVEL] = {
      'line': line,
      'goal': goal,
      'event': event,
      'title': 'golden',
      'task': '',
      'stars': {},
  }
  return toy_lib.PassItOnToy(model, data, _LEVEL)


def run_case(model: language_model.LanguageModel, case: dict[str, Any]) -> str:
  """The action the case's reader (or probe) ends up taking."""
  event = case.get('event', 'quiet')
  if case['kind'] == 'read':
    toy = _toy(
        model, [case['reader']], {case['reader']: case['expect'][0]}, event
    )
    return toy.pass_up(case['message'])['acted'][0]['does']
  relayer, probe = case['relayer'], case['probe']
  toy = _toy(model, [relayer, probe], {probe: case['expect'][0]}, event)
  said = toy.pass_up(case['message'])['hops'][0]['said']
  reader = _toy(model_setup.mock_model(), [probe], {}, event)
  return reader.pass_up(said)['acted'][0]['does']


def cases() -> list[golden_set.Case]:
  return [
      golden_set.Case(
          c['id'], expect=tuple(c['expect']), hard=bool(c.get('hard')), data=c
      )
      for c in toys.load_data('passiton_matrix')['golden']
  ]


def run(
    model: language_model.LanguageModel,
    soft_pass: float = SOFT_PASS,
    repeats: int = 1,
    only: Sequence[str] = (),
) -> dict[str, Any]:
  """Runs the golden set `repeats` times; the same report as golden.run."""
  golden_cases = [c for c in cases() if not only or c.id in only]

  def answer(case: golden_set.Case) -> str:
    began = time.time()
    reply = run_case(model, case.data)
    logging.info('golden %s: %r in %.0f s', case.id, reply, time.time() - began)
    return reply

  report = golden_set.run(
      golden_cases, answer, repeats=repeats, soft_pass=soft_pass
  )
  last = report.runs[-1]
  return {
      'cases': len(golden_cases),
      'hard_passed': last.hard_passed,
      'hard_total': last.hard_total,
      'soft_agreement': last.soft_agreement,
      'passed': report.passed,
      'failures': [
          {'id': c.id, 'hard': c.hard, 'expect': list(c.expect),
           'answer': last.answers[c.id]}  # fmt: skip
          for c in golden_cases
          if last.answers[c.id] not in c.expect
      ],
      'runs': [
          {'hard_passed': r.hard_passed, 'soft_agreement': r.soft_agreement,
           'passed': r.passed}  # fmt: skip
          for r in report.runs
      ],
      'flaky': report.flaky,
  }


def main_json(
    model: language_model.LanguageModel,
    repeats: int = 1,
    only: Sequence[str] = (),
) -> str:
  return json.dumps(run(model, repeats=repeats, only=only), indent=1)
