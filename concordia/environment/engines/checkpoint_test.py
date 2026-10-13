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

"""Tests for checkpoints at completed engine steps."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.environment.engines import simultaneous
from concordia.typing import entity as entity_lib


class CheckpointTest(parameterized.TestCase):

  @parameterized.product(
      engine_type=(sequential.Sequential, simultaneous.Simultaneous),
      skipped_steps=((), (1,), (2,), (1, 2)),
  )
  def test_checkpoint_records_each_completed_step(
      self, engine_type: type[engine_lib.Engine], skipped_steps: tuple[int, ...]
  ):
    player = mock.create_autospec(entity_lib.Entity, instance=True)
    player.name = 'Alice'
    player.act.return_value = 'waits'
    game_master = mock.create_autospec(entity_lib.Entity, instance=True)
    game_master.name = 'GM'
    action_specs = iter(
        entity_lib.skip_this_step_action_spec()
        if step in skipped_steps
        else entity_lib.DEFAULT_ACTION_SPEC
        for step in range(1, 4)
    )

    def game_master_act(action_spec: entity_lib.ActionSpec) -> str:
      if action_spec.output_type == entity_lib.OutputType.TERMINATE:
        return entity_lib.BINARY_OPTIONS['negative']
      if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
        return player.name
      if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
        return engine_lib.action_spec_to_string(next(action_specs))
      if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
        return 'The scene is quiet.'
      if action_spec.output_type == entity_lib.OutputType.RESOLVE:
        return 'Alice waits.'
      raise AssertionError(f'Unexpected action spec: {action_spec}')

    game_master.act.side_effect = game_master_act
    checkpoints = []
    completed_steps = []

    def checkpoint(step: int) -> None:
      checkpoints.append(
          (step, player.observe.call_count, player.act.call_count)
      )

    engine_type().run_loop(
        game_masters=[game_master],
        entities=[player],
        premise='',
        max_steps=3,
        verbose=False,
        log=None,
        checkpoint_callback=checkpoint,
        step_callback=lambda data: completed_steps.append(data.step),
    )

    self.assertEqual(completed_steps, [1, 2, 3])
    self.assertEqual(
        checkpoints,
        [
            (
                step,
                step,
                sum(i not in skipped_steps for i in range(1, step + 1)),
            )
            for step in range(1, 4)
        ],
    )


if __name__ == '__main__':
  absltest.main()
