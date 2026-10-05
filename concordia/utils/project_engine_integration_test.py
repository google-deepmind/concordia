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

"""Bounded standard-engine integration using only NoLanguageModel and no listener."""

import threading
from unittest import mock

from concordia.environment import engine as engine_lib
from concordia.environment import step_controller
from concordia.environment.engines import asynchronous
from concordia.environment.engines import sequential
from concordia.environment.engines import simultaneous
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.utils import project_test_support as fixtures
from concordia.utils import simulation_server
import numpy as np
import pytest


@pytest.mark.parametrize(
    'engine_type', [sequential.Sequential, simultaneous.Simultaneous]
)
def test_scene_free_editor_standard_engine_two_steps(engine_type):
  registry = fixtures.builder_registry()
  engine = engine_type()
  assert isinstance(engine, engine_lib.Engine)
  document = registry.default_document('builder-v1')
  assert not {'scenes', 'scene_types', 'groups'} & document.keys()
  server = simulation_server.SimulationServer(port=0)

  def build(config):
    return generic.Simulation(
        config=config,
        model=no_language_model.NoLanguageModel(),
        embedder=lambda _: np.ones(8),
        engine=engine,
    )

  def runner(config, requested_steps):
    simulation = build(config)
    server.set_simulation(simulation)
    server.broadcast_entity_info(simulation.make_checkpoint_data())
    simulation.play(
        max_steps=requested_steps,
        step_controller=server.step_controller,
        step_callback=server.broadcast_step,
    )

  with mock.patch.object(
      server, 'start', side_effect=AssertionError('No listener')
  ):
    server.configure_project(
        registry,
        document,
        run_with_steps=runner,
        integrated=True,
        preview=build,
    )
    editor = server._project_editor
    assert editor is not None
    fixtures.dispatch(
        editor, 'project.run', {'revision': 0, 'requested_steps': 2}
    )
    fixtures.eventually(
        lambda: editor.state() not in ('starting', 'running', 'pausing')
    )
    assert editor.state() == 'completed', server.get_project()['run']
    assert len(editor.steps) == 2
    assert editor.runtime_view is not None
    assert editor.runtime_view['engine']['name'] == engine_type.__name__
    assert server.simulation._engine is engine


def test_asynchronous_pause_waits_for_every_inflight_worker():
  controller = step_controller.StepController(start_paused=False)
  players = [
      fixtures.MockEntity('entity_0'),
      fixtures.MockEntity('entity_1'),
  ]
  gm = fixtures.MockEntity('game_master')
  started = [threading.Event(), threading.Event()]
  release = [threading.Event(), threading.Event()]
  completed = []
  failures = []
  originals = [player.act for player in players]

  def act(index, *args, **kwargs):
    started[index].set()
    assert release[index].wait(3)
    return originals[index](*args, **kwargs)

  def callback(data):
    completed.append(data)
    controller.complete_step()

  def run():
    try:
      asynchronous.Asynchronous().run_loop(
          game_masters=[gm],
          entities=players,
          max_steps=3,
          step_controller=controller,
          step_callback=callback,
      )
    except BaseException as error:  # pylint: disable=broad-exception-caught
      failures.append(error)

  with mock.patch.object(
      players[0], 'act', side_effect=lambda *a, **k: act(0, *a, **k)
  ), mock.patch.object(
      players[1], 'act', side_effect=lambda *a, **k: act(1, *a, **k)
  ):
    worker = threading.Thread(target=run)
    worker.start()
    try:
      assert all(event.wait(3) for event in started)
      controller.pause()
      release[1].set()
      fixtures.eventually(lambda: len(completed) == 1)
      assert not controller.at_pause_boundary
      with pytest.raises(ValueError, match='acknowledge pause'):
        with controller.paused_boundary():
          pytest.fail('An active worker cannot be edited')
      release[0].set()
      fixtures.eventually(lambda: controller.at_pause_boundary)
      assert len(completed) == 2
      controller.step()
      fixtures.eventually(
          lambda: len(completed) == 3 and controller.at_pause_boundary
      )
    finally:
      controller.stop()
      for event in release:
        event.set()
      worker.join(5)
    assert not worker.is_alive()
    assert not failures


if __name__ == '__main__':
  raise SystemExit(pytest.main([__file__]))
