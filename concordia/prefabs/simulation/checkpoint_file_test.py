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

"""Checkpoint file integrity under failed writes; no simulation execution."""

import json
from unittest import mock

from concordia.language_model import no_language_model
from concordia.prefabs.simulation import checkpoint_test
from concordia.prefabs.simulation import generic
import numpy as np
import pytest


@pytest.fixture(name='simulation')
def simulation_fixture():
  model = no_language_model.NoLanguageModel()
  with (
      mock.patch.object(
          generic.Simulation,
          'play',
          side_effect=AssertionError('No simulation'),
      ),
      mock.patch.object(
          model, 'sample_text', side_effect=AssertionError('No model')
      ),
      mock.patch.object(
          model, 'sample_choice', side_effect=AssertionError('No model')
      ),
  ):
    simulation = generic.Simulation(
        config=checkpoint_test._make_config(),  # pylint: disable=protected-access
        model=model,
        embedder=lambda _: np.ones(3),
    )
    # play normally initializes this callback; do not run an engine to test IO.
    simulation._get_state_callback = None  # pylint: disable=protected-access
    yield simulation


def test_old_checkpoint_remains_readable_until_complete_replacement(
    simulation, tmp_path
):
  target = tmp_path / 'step_4_checkpoint.json'
  old = {'old': 'complete'}
  target.write_text(json.dumps(old), encoding='utf-8')
  original = json.dump
  seen = []

  def partial_write(data, stream, **kwargs):
    stream.write('{"entities":')
    stream.flush()
    seen.append(json.loads(target.read_text(encoding='utf-8')))
    stream.seek(0)
    stream.truncate()
    return original(data, stream, **kwargs)

  with mock.patch.object(generic.json, 'dump', side_effect=partial_write):
    simulation.save_checkpoint(4, str(tmp_path))
  assert seen == [old]
  saved = json.loads(target.read_text(encoding='utf-8'))
  assert set(saved['entities']) == {'Alice', 'Bob'}
  assert saved['checkpoint_counter'] == 0
  simulation.load_from_checkpoint(saved)
  assert set(p.name for p in tmp_path.iterdir()) == {target.name}


@pytest.mark.parametrize('preexisting', [False, True])
def test_partial_io_failure_never_publishes_broken_json(
    simulation, tmp_path, preexisting
):
  target = tmp_path / 'step_5_checkpoint.json'
  old = '{"old": "valid"}'
  if preexisting:
    target.write_text(old, encoding='utf-8')

  def broken_write(data, stream, **kwargs):
    del data, kwargs
    stream.write('{"broken":')
    stream.flush()
    raise OSError('Injected local write failure')

  with mock.patch.object(generic.json, 'dump', side_effect=broken_write):
    simulation.save_checkpoint(5, str(tmp_path))
  if preexisting:
    assert target.read_text(encoding='utf-8') == old
  else:
    assert not target.exists()
  assert set(p.name for p in tmp_path.iterdir()) == (
      {target.name} if preexisting else set()
  )


def test_serialization_failure_preserves_prior_file_and_cleans_staging(
    simulation, tmp_path
):
  target = tmp_path / 'step_6_checkpoint.json'
  old = '{"old": "valid"}'
  target.write_text(old, encoding='utf-8')
  # Raw logs can contain arbitrary component values. Preserve the existing
  # serialization exception contract rather than publishing a partial file.
  simulation._raw_log.append({'non_json': object()})  # pylint: disable=protected-access
  with pytest.raises(TypeError):
    simulation.save_checkpoint(6, str(tmp_path))
  assert target.read_text(encoding='utf-8') == old
  assert list(tmp_path.iterdir()) == [target]


def test_replace_failure_preserves_prior_file(simulation, tmp_path):
  target = tmp_path / 'step_7_checkpoint.json'
  old = '{"old": "valid"}'
  target.write_text(old, encoding='utf-8')
  with mock.patch.object(
      generic.os, 'replace', side_effect=PermissionError('Blocked')
  ):
    simulation.save_checkpoint(7, str(tmp_path))
  assert target.read_text(encoding='utf-8') == old
  assert list(tmp_path.iterdir()) == [target]


def test_callback_and_counter_still_work_without_a_file(simulation):
  received = []
  simulation._get_state_callback = received.append  # pylint: disable=protected-access
  simulation.save_checkpoint(8, '')
  simulation.save_checkpoint(9, '')
  assert [row['checkpoint_counter'] for row in received] == [0, 1]
  assert set(received[0]['entities']) == {'Alice', 'Bob'}
