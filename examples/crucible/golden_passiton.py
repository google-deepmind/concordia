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

"""Golden sets for Pass it on (cycle 102) and the Night Wire: reads, relays.

The cases live in `data/<toy>_matrix.yaml` (`golden`). A read case asks
the model what one member does with a message (`pass_up` with the member at
the end of the line, alone or `with` others who hear it too). A relay case
asks the model how a relayer passes a message on, then reads what came out
for a probe member by the laws (the mock), so the case tests the relay
alone (in the Night Wire, read by the same model: its relayers paraphrase).
A Night Wire case names the `chapter` whose jobs are on offer.
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

SOFT_PASS = 0.75
_LEVEL = '__golden__'


def _toy(
    model: language_model.LanguageModel,
    line: list[Any],
    goal: dict[str, str],
    case: dict[str, Any],
    toy: str,
) -> Any:
  """A one-off level: the case's line, on the case's chapter if it names one."""
  data = copy.deepcopy(toys.load_data(toy))
  chapter = data['scenarios'].get(case.get('chapter', ''), {})
  data['scenarios'][_LEVEL] = {
      **chapter,
      'line': line,
      'goal': goal,
      'event': case.get('event', 'quiet'),
      'title': 'golden',
      'task': '',
      'stars': {},
  }
  return toys.registry()[toy].build(model, data, _LEVEL)


def run_case(
    model: language_model.LanguageModel,
    case: dict[str, Any],
    toy: str = 'passiton',
) -> str:
  """The action the case's reader (or probe) ends up taking."""
  if case['kind'] == 'read':
    reader = case['reader']
    end = [reader, *case.get('with', [])]
    level = _toy(model, [end], {reader: case['expect'][0]}, case, toy)
    if hasattr(level, 'read'):  # the Night Wire asks the one reader only
      return level.read(case['message'], reader)[0]
    acted = level.pass_up(case['message'])['acted']
    return next(a['does'] for a in acted if a['who'] == reader)
  relayer, probe = case['relayer'], case['probe']
  level = _toy(model, [relayer, probe], {probe: case['expect'][0]}, case, toy)
  said = level.pass_up(case['message'])['hops'][0]['said']
  if hasattr(level, 'read'):
    # The Night Wire's relayers are people (cycle 121): what they pass on is
    # read by the same model, not the mock, which knows no paraphrase.
    return level.read(said, probe)[0]
  mock = _toy(model_setup.mock_model(), [probe], {}, case, toy)
  return mock.pass_up(said)['acted'][0]['does']


def cases(toy: str = 'passiton') -> list[golden_set.Case]:
  return [
      golden_set.Case(
          c['id'], expect=tuple(c['expect']), hard=bool(c.get('hard')), data=c
      )
      for c in toys.load_data(f'{toy}_matrix')['golden']
  ]


def run(
    model: language_model.LanguageModel,
    soft_pass: float = SOFT_PASS,
    repeats: int = 1,
    only: Sequence[str] = (),
    toy: str = 'passiton',
) -> dict[str, Any]:
  """Runs the golden set `repeats` times; the same report as golden.run."""
  golden_cases = [c for c in cases(toy) if not only or c.id in only]

  def answer(case: golden_set.Case) -> str:
    began = time.time()
    reply = run_case(model, case.data, toy)
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
    toy: str = 'passiton',
) -> str:
  return json.dumps(run(model, repeats=repeats, only=only, toy=toy), indent=1)
