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

"""The page's lab: experiments over seeds, played in the background.

One job at a time, of two kinds, each a set of MOCK runs in tick mode (which
is exact), played by `sweep` in worker processes while the world plays on:
- `compare`: a world's settings against the same with one lever pulled,
  seed by seed (`sweep.summarize`);
- `replicate`: a world's settings played again from the start on other
  seeds, to show the spread a result has (`sweep.spread`).

The lab only builds, plays and reads stock simulations, as `experiment.py`
does; it never touches the world on the page.
"""

from collections.abc import Mapping, Sequence
import multiprocessing
import threading
import time
from typing import Any

from examples.lifeways import scenarios
from examples.lifeways import simulation
from examples.lifeways import sweep

COMPARE, REPLICATE = 'compare', 'replicate'
# The most seeds and years a job may ask for (a job is runs of seeds x arms).
SEEDS_MOST = 10
YEARS_MOST = 300
# Settings a lab run never takes from a world: free mode closes years by the
# clock, so its runs would not be exact.
NOT_COPIED = ('free',)


# How long the lab waits for a run at a time before it looks whether it was
# closed.
WAIT_SECONDS = 1.0


def _play(
    job: tuple[str, Mapping[str, Any], Any, sweep.Condition, int],
) -> tuple[sweep.Condition, list[dict[str, Any]]]:
  """One run, quietly (in a worker process): its condition and rows."""
  scenario, overrides, lever, condition, years = job
  with simulation.quiet():
    return condition, sweep.play(scenario, lever, condition, years, overrides)


class Lab:
  """Runs one experiment at a time in the background, and keeps the last.

  Thread-safe: the page polls `status` while a job plays.
  """

  def __init__(self, workers: int = 2):
    """Initializes the lab.

    Args:
      workers: Runs played at once, each in a process of its own (0: in the
        lab's own thread, one at a time; for tests).
    """
    self._workers = workers
    self._lock = threading.Lock()
    self._job: dict[str, Any] | None = None
    self._thread: threading.Thread | None = None
    self._closed = threading.Event()

  def start(
      self,
      kind: str,
      scenario: str,
      overrides: Mapping[str, Any],
      seeds: int,
      years: int,
      lever: tuple[str, Any] | None = None,
      first_seed: int = 0,
  ) -> dict[str, Any]:
    """Starts a job in the background.

    Args:
      kind: `COMPARE` or `REPLICATE`.
      scenario: The world's scenario.
      overrides: The world's own settings over the scenario's.
      seeds: How many seeds.
      years: Years each run plays.
      lever: For `COMPARE`: the lever pulled and its value.
      first_seed: The first seed (the others follow it).

    Returns:
      The job as `status` gives it.

    Raises:
      ValueError: If a job is playing, or the job is not one the lab plays.
    """
    if kind not in (COMPARE, REPLICATE):
      raise ValueError(f'A job is {COMPARE} or {REPLICATE}.')
    if not 1 <= seeds <= SEEDS_MOST or not 1 <= years <= YEARS_MOST:
      raise ValueError(
          f'A job plays 1 to {SEEDS_MOST} seeds of 1 to {YEARS_MOST} years.'
      )
    overrides = {k: v for k, v in overrides.items() if k not in NOT_COPIED}
    if kind == COMPARE:
      if lever is None:
        raise ValueError('A comparison pulls a lever.')
      # Refused here, not in a worker, if the lever cannot be pulled.
      scenarios.with_lever(scenarios.settings(scenario, **overrides), *lever)
    conditions = [
        sweep.Condition(seed, treated)
        for seed in range(first_seed, first_seed + seeds)
        for treated in ((False, True) if kind == COMPARE else (False,))
    ]
    with self._lock:
      if self._job is not None and self._job['state'] == 'playing':
        raise ValueError('An experiment is playing; wait for it to end.')
      self._job = {
          'kind': kind,
          'scenario': scenario,
          'lever': list(lever) if lever else None,
          'seeds': [first_seed, first_seed + seeds - 1],
          'years': years,
          'runs': len(conditions),
          'done': 0,
          'state': 'playing',
          'started': time.time(),
          'seconds': None,
          'result': None,
          'error': '',
      }
      job = self._job
    runs = [(scenario, overrides, lever, c, years) for c in conditions]
    self._thread = threading.Thread(
        target=self._run,
        args=(job, runs),
        name='lifeways-lab',
        daemon=True,
    )
    self._thread.start()
    return self.status() or {}

  def _run(self, job: dict[str, Any], runs: Sequence[tuple[Any, ...]]) -> None:
    """Plays a job's runs and keeps what they show."""
    rows: dict[sweep.Condition, list[dict[str, Any]]] = {}
    pool = None
    try:
      if self._workers:
        pool = multiprocessing.get_context('spawn').Pool(self._workers)
        playing = pool.imap_unordered(_play, runs)
        while len(rows) < len(runs):
          try:
            condition, run = playing.next(timeout=WAIT_SECONDS)
          except multiprocessing.TimeoutError:
            if self._closed.is_set():
              return
            continue
          rows[condition] = run
          with self._lock:
            job['done'] += 1
      else:
        for run in runs:
          if self._closed.is_set():
            return
          condition, played = _play(run)
          rows[condition] = played
          with self._lock:
            job['done'] += 1
      if job['kind'] == COMPARE:
        result = sweep.summarize(rows)
      else:
        result = sweep.spread({c.seed: run for c, run in rows.items()})
      with self._lock:
        job['result'] = result
        job['state'] = 'done'
    except Exception as error:  # pylint: disable=broad-exception-caught
      with self._lock:
        job['error'] = f'{type(error).__name__}: {error}'
        job['state'] = 'failed'
    finally:
      if pool is not None:
        pool.terminate()  # No worker outlives its job.
        pool.join()
      with self._lock:
        job['seconds'] = round(time.time() - job['started'])
        if job['state'] == 'playing':  # Closed before it ended.
          job['state'] = 'stopped'

  def status(self) -> dict[str, Any] | None:
    """The job playing or last played (None if none), as the page shows it."""
    with self._lock:
      if self._job is None:
        return None
      return {k: v for k, v in self._job.items() if k != 'started'}

  def close(self, timeout: float = 5.0) -> None:
    """Stops a job playing: its runs are dropped, its workers ended."""
    self._closed.set()
    if self._thread is not None:
      self._thread.join(timeout)
