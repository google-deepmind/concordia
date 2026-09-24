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
  parser.add_argument('--model', default='llama3.2:3b')
  parser.add_argument(
      '--fixture',
      action='store_true',
      help='Scripted test responses; never presented as AI.',
  )
  parser.add_argument(
      '--output', type=pathlib.Path, default=pathlib.Path('runs/one-more-song')
  )
  args = parser.parse_args()
  if not 1 <= args.port <= 65535:
    parser.error('--port must be between 1 and 65535')
  if args.editor_port is not None and not 1 <= args.editor_port <= 65535:
    parser.error('--editor-port must be between 1 and 65535')
  if args.editor_port == args.port:
    parser.error('Player and editor ports must differ')
  session = game.PlayerSession('fixture' if args.fixture else args.model)
  model = (
      FixtureModel()
      if args.fixture
      else ollama_model.OllamaLanguageModel(model_name=args.model)
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
      journal_title='ONE MORE SONG',
      journal_filename='one-more-song.txt',
  )
  try:
    uvicorn.run(app, host='127.0.0.1', port=args.port)
  finally:
    if editor:
      editor.step_controller.stop()
      editor.stop()


if __name__ == '__main__':
  main()
