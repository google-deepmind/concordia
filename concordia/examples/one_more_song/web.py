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

"""Launch One More Song, reusing the existing human browser transport."""

import argparse
import datetime
import json
import pathlib

from concordia.contrib.language_models.ollama import ollama_model
from concordia.examples.astral_canticle import web as human_web
from concordia.examples.one_more_song import game
from concordia.language_model import no_language_model
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import uvicorn


class FixtureModel(no_language_model.NoLanguageModel):
  """Deterministic UI fixture, NOT evidence of intelligent or real-model play."""

  def sample_text(self, prompt, **kwargs):
    del kwargs
    if 'You are Maya.' in prompt:
      return 'I want everyone to leave smiling. Could we sing quietly together?'
    return 'I need a firm end time and no amplifier. Can you promise both?'

  def sample_choice(self, prompt, responses, **kwargs):
    del prompt, kwargs
    index = list(responses).index('ACCEPT') if 'ACCEPT' in responses else 0
    return index, responses[index], {}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--port', type=int, default=8820)
  parser.add_argument('--editor-port', type=int)
  parser.add_argument('--allowed-host', action='append', default=[])
  parser.add_argument('--root-path', default='')
  parser.add_argument('--model', default='qwen3:8b')
  parser.add_argument(
      '--think',
      action=argparse.BooleanOptionalAction,
      default=None,
      help=(
          'Enable/disable thinking on supported Ollama models; omitted uses'
          ' provider default.'
      ),
  )
  parser.add_argument(
      '--fixture',
      action='store_true',
      help='Scripted test responses; never presented as AI.',
  )
  parser.add_argument(
      '--output',
      type=pathlib.Path,
      help='New run directory (default: a timestamped folder under runs/).',
  )
  args = parser.parse_args()
  args.root_path = args.root_path.rstrip('/')
  if args.root_path and (
      not args.root_path.startswith('/')
      or '?' in args.root_path
      or '#' in args.root_path
  ):
    parser.error('--root-path must be a URL path such as /one-more-song.')
  if not 1 <= args.port <= 65535:
    parser.error('--port must be between 1 and 65535')
  if args.editor_port is not None and not 1 <= args.editor_port <= 65535:
    parser.error('--editor-port must be between 1 and 65535')
  if args.editor_port == args.port:
    parser.error('Player and editor ports must differ')
  if args.output is None:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y%m%dT%H%M%S%fZ'
    )
    args.output = pathlib.Path('runs') / f'one-more-song-{stamp}'
  if args.output.exists() and (
      not args.output.is_dir()
      or any(
          (args.output / name).exists()
          for name in (
              'public.json',
              'simulation.json',
              'timing.json',
              'log.html',
              'checkpoints',
          )
      )
  ):
    parser.error(
        '--output already contains a run. Choose a new directory to'
        ' preserve it.'
    )
  session = game.PlayerSession('fixture' if args.fixture else args.model)
  args.output.mkdir(parents=True, exist_ok=True)
  try:
    with (args.output / 'public.json').open('x', encoding='utf-8') as saved:
      json.dump(session.snapshot(), saved, ensure_ascii=False)
  except FileExistsError:
    parser.error(
        '--output already contains a run. Choose a new directory to'
        ' preserve it.'
    )
  model = (
      FixtureModel()
      if args.fixture
      else ollama_model.OllamaLanguageModel(
          model_name=args.model,
          think=args.think,
          system_message=(
              'Simulate the speaker specified in the supplied context. Follow'
              " that character's identity, goal and observations. Output only"
              " the requested response in that character's voice, not a copy"
              " of another speaker's words or the question."
          ),
      )
  )
  config, simulation = game.build(model, session)
  editor = None
  if args.editor_port:
    editor = simulation_server.SimulationServer(port=args.editor_port)
    editor.set_simulation(simulation)
    editor.set_html_content(
        visual_interface.visualize_config_to_html(
            config,
            title='One More Song — private designer',
            checkpoint_data=simulation.make_checkpoint_data(),
        )
    )
    editor.start()
    editor.broadcast_entity_info(simulation.make_checkpoint_data())
    editor.step_controller.play()
  app = human_web.create_app(
      session,
      runner=lambda: game.play(simulation, session, args.output, editor=editor),
      static_path=pathlib.Path(__file__).with_name('static'),
      journal_title=(
          'ONE MORE SONG\nSCRIPTED UI PREVIEW — not AI-generated.'
          if args.fixture
          else f'ONE MORE SONG\nAI-generated characters: {args.model}'
      ),
      journal_filename='one-more-song.txt',
      allowed_hosts=['127.0.0.1', 'localhost', *args.allowed_host],
      root_path=args.root_path,
  )
  print(
      f'Play One More Song: http://127.0.0.1:{args.port}{args.root_path}/',
      flush=True,
  )
  print(f'Run saved to: {args.output.resolve()}', flush=True)
  print(
      'Reload reconnects. Stop this server and relaunch for a fresh game.',
      flush=True,
  )
  try:
    uvicorn.run(app, host='127.0.0.1', port=args.port)
  finally:
    if editor:
      editor.step_controller.stop()
      editor.stop()


if __name__ == '__main__':
  main()
