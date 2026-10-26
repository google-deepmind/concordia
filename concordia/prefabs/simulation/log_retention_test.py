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

"""Tests for keeping only recent raw log entries in the generic simulation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from absl.testing import absltest
from concordia.environment import engine as engine_lib
from concordia.environment import step_controller as step_controller_lib
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
import numpy as np


class _CountingEngine(engine_lib.Engine):
  """Logs one entry a step, checkpoints, then reports the step."""

  def make_observation(self, game_master, entity) -> str:
    return ''

  def next_acting(self, game_master, entities):
    raise NotImplementedError()

  def resolve(self, game_master, event: str) -> None:
    pass

  def terminate(self, game_master) -> bool:
    return False

  def next_game_master(self, game_master, game_masters):
    return game_master

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str,
      max_steps: int,
      verbose: bool,
      log: list[Mapping[str, Any]] | None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_controller: step_controller_lib.StepController | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
  ):
    for step in range(1, max_steps + 1):
      if log is not None:
        log.append({'Step': step, 'Summary': f'Step {step}'})
      if checkpoint_callback is not None:
        checkpoint_callback(step)
      if step_callback is not None:
        step_callback(
            step_controller_lib.StepData(
                step=step,
                acting_entity='',
                action='',
                entity_actions={},
                entity_logs={},
            )
        )


def _simulation() -> simulation_lib.Simulation:
  return simulation_lib.Simulation(
      config=prefab_lib.Config(prefabs={}, instances=[]),
      model=no_language_model.NoLanguageModel(),
      embedder=lambda text: np.ones(3),
      engine=_CountingEngine(),
  )


class LogRetentionTest(absltest.TestCase):

  def test_only_the_most_recent_entries_are_kept(self):
    sim = _simulation()
    checkpoint_logs = []

    sim.play(
        max_steps=10,
        log_retention=3,
        get_state_callback=lambda data: checkpoint_logs.append(
            [entry['Step'] for entry in data['raw_log']]
        ),
    )

    self.assertEqual([entry['Step'] for entry in sim.get_raw_log()], [8, 9, 10])
    self.assertTrue(all(len(steps) <= 3 for steps in checkpoint_logs))
    self.assertEqual(checkpoint_logs[-1], [8, 9, 10])

  def test_without_retention_every_entry_is_kept(self):
    sim = _simulation()
    sim.play(max_steps=10)
    self.assertLen(sim.get_raw_log(), 10)

  def test_retention_must_not_be_negative(self):
    with self.assertRaises(ValueError):
      _simulation().play(max_steps=1, log_retention=-1)


if __name__ == '__main__':
  absltest.main()
