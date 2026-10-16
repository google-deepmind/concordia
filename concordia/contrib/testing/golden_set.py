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

"""Golden sets: does a language model judge the way a game's laws say?

A game that hands judgements to a language model (who a rule applies to,
what a character does under an order) can state, for fixed situations, the
answers its laws allow. A golden set is those cases, run against a model:

- a **hard** case follows from a law the model must not break; any miss
  fails the set;
- a **soft** case measures agreement with intended judgement (a
  temperament, a style); the set needs `soft_pass` of them;
- providers are not deterministic even at temperature 0, so the set is run
  `repeats` times, every run must pass, and a case answered differently
  across runs is reported as **flaky**.

The game supplies the answers: `answer(case)` builds the situation, asks
its model and returns the answer, so the module is independent of any
prompt format. A provider failure inside `answer` fails that case, not the
run.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Case:
  """One fixed situation and the answers the laws allow.

  Attributes:
    id: A short name, stable across runs.
    expect: The acceptable answers.
    hard: Whether a miss fails the whole set.
    data: Whatever the game needs to build the situation.
  """

  id: str
  expect: Sequence[str]
  hard: bool = False
  data: Any = None


@dataclasses.dataclass(frozen=True)
class Run:
  """One pass over the cases."""

  answers: dict[str, str]
  hard_passed: int
  hard_total: int
  soft_agreement: float
  passed: bool


@dataclasses.dataclass(frozen=True)
class Report:
  """Every run, whether all passed, and the cases that changed answer."""

  runs: list[Run]
  passed: bool
  failures: dict[str, list[str]]  # case id -> wrong answers seen
  flaky: list[str]

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def compare(
    cases: Sequence[Case], before: Report, after: Report
) -> dict[str, list[str]]:
  """What a change did to a golden set: cases fixed, broken, still failing.

  A case passes in a report only if every run of it passed.

  Args:
    cases: The cases both reports ran.
    before: The report before the change (a prompt, a model, a wrapper).
    after: The report after it.

  Returns:
    Case ids under 'fixed', 'broken', 'still_failing' and 'flaky_after'.
  """

  def passed(report: Report, case: Case) -> bool:
    return all(r.answers.get(case.id) in case.expect for r in report.runs)

  out: dict[str, list[str]] = {
      'fixed': [], 'broken': [], 'still_failing': [], 'flaky_after': []
  }  # fmt: skip
  for case in cases:
    was, now = passed(before, case), passed(after, case)
    if now and not was:
      out['fixed'].append(case.id)
    elif was and not now:
      out['broken'].append(case.id)
    elif not was and not now:
      out['still_failing'].append(case.id)
  out['flaky_after'] = list(after.flaky)
  return out


def run_once(
    cases: Sequence[Case],
    answer: Callable[[Case], str],
    soft_pass: float = 0.75,
) -> Run:
  """Runs every case once; a case whose answer raises is a miss."""
  answers = {}
  for case in cases:
    try:
      answers[case.id] = answer(case)
    except Exception as e:  # pylint: disable=broad-exception-caught
      answers[case.id] = f'error: {type(e).__name__}'
  hard = [c for c in cases if c.hard]
  soft = [c for c in cases if not c.hard]
  hard_passed = sum(answers[c.id] in c.expect for c in hard)
  soft_ok = sum(answers[c.id] in c.expect for c in soft)
  soft_agreement = soft_ok / len(soft) if soft else 1.0
  return Run(
      answers=answers,
      hard_passed=hard_passed,
      hard_total=len(hard),
      soft_agreement=round(soft_agreement, 3),
      passed=hard_passed == len(hard) and soft_agreement >= soft_pass,
  )


def run(
    cases: Sequence[Case],
    answer: Callable[[Case], str],
    repeats: int = 2,
    soft_pass: float = 0.75,
) -> Report:
  """Runs the set `repeats` times; it passes only if every run passes.

  Args:
    cases: The golden cases (ids must be unique).
    answer: Builds a case's situation, asks the model, returns its answer.
    repeats: How many times to run the whole set.
    soft_pass: The share of soft cases a run must agree with.

  Returns:
    The report.

  Raises:
    ValueError: If two cases share an id.
  """
  ids = [c.id for c in cases]
  if len(set(ids)) != len(ids):
    raise ValueError('golden case ids must be unique')
  runs = [run_once(cases, answer, soft_pass) for _ in range(repeats)]
  failures: dict[str, list[str]] = {}
  for r in runs:
    for case in cases:
      got = r.answers[case.id]
      if got not in case.expect:
        failures.setdefault(case.id, []).append(got)
  flaky = [c.id for c in cases if len({r.answers[c.id] for r in runs}) > 1]
  return Report(
      runs=runs,
      passed=all(r.passed for r in runs),
      failures=failures,
      flaky=flaky,
  )
