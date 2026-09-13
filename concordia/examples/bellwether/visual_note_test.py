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

"""Pixels, local requests and literal review artifacts without runs."""

import io
import json
from unittest import mock

from concordia.examples.bellwether import visual_note
from concordia.utils import profiler
import httpx
from PIL import Image
from PIL import PngImagePlugin
import pytest


def image_bytes(color='red', size=(20, 10), format_name='PNG', **options):
  output = io.BytesIO()
  Image.new('RGB', size, color).save(output, format=format_name, **options)
  return output.getvalue()


@pytest.fixture(name='client')
def client_fixture():
  with mock.patch.object(visual_note.ollama, 'Client') as factory:
    factory.return_value.show.return_value = {
        'capabilities': ['completion', 'vision', 'thinking']
    }
    factory.return_value.chat.return_value = {
        'message': {'content': 'A red rectangle.'},
        'done': True,
        'done_reason': 'stop',
    }
    yield factory


def test_orientation_metadata_stripping_and_fresh_pixel_provenance():
  tags = PngImagePlugin.PngInfo()
  tags.add_text('Private-fixture', 'Not-for-model')
  exif = Image.Exif()
  exif[274] = 6
  exif[270] = 'Private EXIF description'
  raw = image_bytes(pnginfo=tags, exif=exif)
  prepared = visual_note.prepare_image(raw)
  with Image.open(io.BytesIO(prepared.pixels)) as decoded:
    assert decoded.mode == 'RGB'
    assert decoded.size == (10, 20)
    assert not decoded.getexif()
    assert not decoded.info
  assert b'Not-for-model' not in prepared.pixels
  assert prepared.provenance['original_size'] == [20, 10]
  assert prepared.provenance['transmitted_size'] == [10, 20]
  assert prepared.provenance['original_sha256'] != (
      prepared.provenance['transmitted_sha256']
  )


def test_alpha_on_white_and_bounded_long_edge():
  output = io.BytesIO()
  Image.new('RGBA', (1500, 10), (255, 0, 0, 0)).save(output, format='PNG')
  prepared = visual_note.prepare_image(output.getvalue())
  with Image.open(io.BytesIO(prepared.pixels)) as decoded:
    assert decoded.width == visual_note.MAX_EDGE
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    'data',
    [
        b'',
        b'<svg><script>untrusted</script></svg>',
        b'not an image',
        image_bytes(format_name='GIF'),
        b'x' * (visual_note.MAX_BYTES + 1),
    ],
)
def test_unsupported_or_oversized_bytes_reject(data):
  with pytest.raises(ValueError):
    visual_note.prepare_image(data)


def test_pixel_and_animation_limits_before_inference(monkeypatch):
  monkeypatch.setattr(visual_note, 'MAX_PIXELS', 100)
  with pytest.raises(ValueError, match='megapixel'):
    visual_note.prepare_image(image_bytes(size=(20, 10)))
  output = io.BytesIO()
  Image.new('RGB', (5, 5), 'red').save(
      output,
      format='PNG',
      save_all=True,
      append_images=[Image.new('RGB', (5, 5), 'blue')],
  )
  with pytest.raises(ValueError, match='animation'):
    visual_note.prepare_image(output.getvalue())


def test_actual_sanitized_pixels_local_bounds_and_profile(client):
  prepared = visual_note.prepare_image(image_bytes())
  profile = profiler.ProfilerContext()
  profile.enable()
  provider = visual_note.OllamaVisualDraft(
      'installed-vision', timeout=5, max_output_tokens=128, profiler=profile
  )
  result = provider.describe(prepared, 'Which color is visible?')
  client.assert_called_once_with(
      host='http://127.0.0.1:11434', timeout=5, trust_env=False
  )
  request = client.return_value.chat.call_args.kwargs
  assert request['messages'][1]['images'] == [prepared.pixels]
  assert request['options'] == {'temperature': 0, 'num_predict': 128}
  assert request['keep_alive'] == 0
  assert request['think'] is False
  assert result['text'] == 'A red rectangle.'
  assert result['review'] == 'required_before_use'
  assert 'pixels' not in json.dumps(result)
  result['source']['original_size'][0] = 999
  assert prepared.provenance['original_size'][0] == 20
  stats = profile.get_stats()
  assert stats['counters']['vision.requests'] == 1
  assert 'vision' in stats['timings']
  assert 'Which color' not in json.dumps(stats)
  client.return_value.pull.assert_not_called()


def test_chat_only_model_rejected_without_images(client):
  client.return_value.show.return_value = {'capabilities': ['completion']}
  with pytest.raises(ValueError, match='image input'):
    visual_note.OllamaVisualDraft('chat-only')
  client.return_value.chat.assert_not_called()
  client.return_value.pull.assert_not_called()


@pytest.mark.parametrize('tokens', [0, -1, 1025, True, 1.5])
def test_invalid_limits_before_client_creation(client, tokens):
  with pytest.raises(ValueError):
    visual_note.OllamaVisualDraft('vision', max_output_tokens=tokens)
  client.assert_not_called()


@pytest.mark.parametrize('question', ['', ' ', 'x' * 2001, None])
def test_invalid_question_never_reaches_model(client, question):
  provider = visual_note.OllamaVisualDraft('vision')
  with pytest.raises(ValueError):
    provider.describe(visual_note.prepare_image(image_bytes()), question)
  client.return_value.chat.assert_not_called()


@pytest.mark.parametrize(
    'response',
    [
        {'done': False, 'message': {'content': 'partial'}},
        {'done': True, 'message': {'content': ''}},
        {'done': True, 'message': {'content': 'x' * 16001}},
        {'done': True, 'message': {'content': 'hello', 'tool_calls': [{}]}},
    ],
)
def test_invalid_model_output_is_not_a_valid_note(client, response):
  client.return_value.chat.return_value = response
  profile = profiler.ProfilerContext()
  profile.enable()
  provider = visual_note.OllamaVisualDraft('vision', profiler=profile)
  with pytest.raises(ValueError):
    provider.describe(visual_note.prepare_image(image_bytes()), 'Describe.')
  assert profile.get_stats()['counters']['vision.failures'] == 1


def test_nonthinking_model_and_truncated_draft_are_explicit(client):
  client.return_value.show.return_value = {'capabilities': ['vision']}
  client.return_value.chat.return_value['done_reason'] = 'length'
  result = visual_note.OllamaVisualDraft('vision').describe(
      visual_note.prepare_image(image_bytes()), 'Describe.'
  )
  assert client.return_value.chat.call_args.kwargs['think'] is None
  assert result['stop_reason'] == 'length'
  assert result['review'] == 'required_before_use'


def test_cli_literal_json_no_pixels_path_or_auto_action(client, tmp_path):
  source = tmp_path / 'private-fixture-name.png'
  source.write_bytes(image_bytes())
  output = tmp_path / 'draft.json'
  text = '</script><img src=x onerror=unsafe()> & quoted "é"'
  client.return_value.chat.return_value['message']['content'] = text
  args = [
      '--image',
      str(source),
      '--question',
      'Describe.',
      '--model',
      'installed',
      '--output',
      str(output),
  ]
  visual_note.main(args)
  note = json.loads(output.read_text())
  assert note['text'] == text
  assert note['kind'] == 'model_observation_draft'
  assert str(source) not in output.read_text()
  assert 'base64' not in output.read_text()
  assert note['profile']['counters']['vision.requests'] == 1
  before = output.read_bytes()
  with pytest.raises(SystemExit):
    visual_note.main(args)
  assert output.read_bytes() == before
  assert client.return_value.chat.call_count == 1


def test_cli_bad_image_before_client_or_output(client, tmp_path):
  source = tmp_path / 'bad.png'
  source.write_text('invalid')
  output = tmp_path / 'note.json'
  with pytest.raises(SystemExit) as error:
    visual_note.main([
        '--image',
        str(source),
        '--question',
        'Describe.',
        '--model',
        'installed',
        '--output',
        str(output),
    ])
  assert error.value.code == 2
  assert not output.exists()
  client.assert_not_called()


def test_cli_transport_failure_never_writes_note(client, tmp_path):
  source = tmp_path / 'image.png'
  source.write_bytes(image_bytes())
  output = tmp_path / 'note.json'
  client.return_value.chat.side_effect = httpx.ReadTimeout('fixture timeout')
  with pytest.raises(SystemExit) as error:
    visual_note.main([
        '--image',
        str(source),
        '--question',
        'Describe.',
        '--model',
        'installed',
        '--output',
        str(output),
    ])
  assert error.value.code == 2
  assert not output.exists()
