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

"""Local vectors, standard memory retrieval and placeholder metadata."""

from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import local_embeddings
from concordia.examples.bellwether import run
from concordia.prefabs.simulation import generic
from concordia.utils import profiler
import numpy as np
import pytest


@pytest.fixture(name='client')
def client_fixture():
  with mock.patch.object(local_embeddings.ollama, 'Client') as factory:
    factory.return_value.show.return_value = {'capabilities': ['embedding']}
    yield factory


def test_bounded_local_normalized_vectors_and_profile_without_text(client):
  profile = profiler.ProfilerContext()
  profile.enable()
  client.return_value.embed.return_value = {'embeddings': [[3.0, 4.0]]}
  embed = local_embeddings.OllamaEmbedder(
      'installed-embed', timeout=2.5, profiler=profile
  )
  client.assert_called_once_with(
      host='http://127.0.0.1:11434', timeout=2.5, trust_env=False
  )
  assert np.allclose(embed('Private fixture memory'), [0.6, 0.8])
  client.return_value.embed.assert_called_once_with(
      model='installed-embed',
      input='Private fixture memory',
      truncate=False,
      keep_alive='5m',
  )
  stats = profile.get_stats()
  assert stats['counters']['embedding.requests'] == 1
  assert 'Private fixture memory' not in str(stats)
  assert 'embedding' in stats['timings']
  client.return_value.pull.assert_not_called()


@pytest.mark.parametrize(
    'rows',
    [
        [],
        [[], []],
        [[]],
        [[float('nan')]],
        [[float('inf')]],
        [[0.0, 0.0]],
        [[[1, 2]]],
    ],
)
def test_malformed_vectors_fail_without_placeholder_or_dimension_commit(
    client, rows
):
  client.return_value.embed.side_effect = [
      {'embeddings': rows},
      {'embeddings': [[0.0, 1.0]]},
  ]
  profile = profiler.ProfilerContext()
  profile.enable()
  embed = local_embeddings.OllamaEmbedder('installed-embed', profiler=profile)
  with pytest.raises(ValueError):
    embed('fixture')
  assert np.allclose(embed('valid fixture'), [0, 1])
  assert profile.get_stats()['counters']['embedding.failures'] == 1


def test_dimensions_cannot_change_and_invalid_input_never_reaches_model(client):
  client.return_value.embed.side_effect = [
      {'embeddings': [[1, 0]]},
      {'embeddings': [[1, 0, 0]]},
  ]
  embed = local_embeddings.OllamaEmbedder('installed-embed')
  embed('one')
  with pytest.raises(ValueError, match='dimensions changed'):
    embed('two')
  with pytest.raises(ValueError, match='must be text'):
    embed(None)  # pyrefly: ignore[bad-argument-type] -- invalid-input test
  assert client.return_value.embed.call_count == 2


def test_model_capability_required_without_download_or_memory_transmission(
    client,
):
  client.return_value.show.return_value = {'capabilities': ['completion']}
  with pytest.raises(ValueError, match='does not support embeddings'):
    local_embeddings.OllamaEmbedder('chat-only')
  client.return_value.embed.assert_not_called()
  client.return_value.pull.assert_not_called()


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_request_limits_reject_before_client_creation(client, timeout):
  with pytest.raises(ValueError):
    local_embeddings.OllamaEmbedder('installed', timeout=timeout)
  client.assert_not_called()


def test_game_uses_standard_bank_with_supplied_callable_and_keeps_default(
    tmp_path,
):
  seen = []

  def embed(text):
    seen.append(text)
    return np.array([1.0, 0.0]) if 'fuel' in text else np.array([0.0, 1.0])

  with mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No simulation')
  ):
    game = game_service.Game(
        tmp_path / 'supplied', embedder=embed, actor_logic='basic'
    )
    default = game_service.Game(tmp_path / 'default')
    try:
      bank = game.simulation.game_master_memory_bank
      bank.add('A violin is rehearsing.')
      bank.add('The shelter needs fuel.')
      assert bank.retrieve_associative('fuel', k=1) == [
          'The shelter needs fuel.'
      ]
      assert len(seen) == 3
      assert game.developer_view()['embedding'] == {'kind': 'provided_callable'}
      assert default.developer_view()['embedding'] == {
          'kind': 'constant_placeholder',
          'dimensions': 8,
      }
      assert 'embedding' not in game.player_view()
    finally:
      game.close()
      default.close()


def test_cli_cannot_silently_ignore_embedding_choice_in_slice(client):
  with pytest.raises(SystemExit) as error:
    run.main(['--embedding-model', 'installed'])
  assert error.value.code == 2
  client.assert_not_called()


def test_cli_passes_explicit_embedder_to_existing_game_and_records_model(
    client, tmp_path
):
  built = []
  original = game_service.Game

  class RecordedGame(original):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      built.append(self)

  with mock.patch.object(game_service, 'Game', RecordedGame), mock.patch.object(
      generic.Simulation, 'play', side_effect=AssertionError('No run')
  ), mock.patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt):
    run.main([
        '--mode',
        'fixture',
        '--embedding-model',
        'installed',
        '--editor-port',
        '0',
        '--player-port',
        '0',
        '--output',
        str(tmp_path),
    ])
  assert len(built) == 1
  assert built[0].embedding == {'kind': 'local_ollama', 'model': 'installed'}
  client.return_value.show.assert_called_once_with('installed')
  client.return_value.embed.assert_not_called()


def test_transport_failure_leaves_standard_bank_unchanged_and_retries(client):
  profile = profiler.ProfilerContext()
  profile.enable()
  client.return_value.embed.side_effect = [
      TimeoutError('local request expired'),
      {'embeddings': [[1.0, 0.0]]},
  ]
  embed = local_embeddings.OllamaEmbedder('installed', profiler=profile)
  bank = basic_associative_memory.AssociativeMemoryBank(embed)
  before = bank.get_state()
  with pytest.raises(TimeoutError):
    bank.add('one memory')
  assert bank.get_state() == before
  bank.add('one memory')
  assert list(bank.get_data_frame()['text']) == ['one memory']
  assert profile.get_stats()['counters']['embedding.failures'] == 1
  assert profile.get_stats()['counters']['embedding.requests'] == 2
