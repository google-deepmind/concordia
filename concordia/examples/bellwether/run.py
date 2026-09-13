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

"""Serve Bellwether through the standard shared player/editor/CLI service."""

import argparse
import json
import pathlib
import time

from concordia.contrib.language_models.ollama import ollama_model
from concordia.examples.bellwether import game as rules
from concordia.examples.bellwether import game_service
from concordia.examples.bellwether import local_embeddings
from concordia.examples.bellwether import multiplayer
from concordia.examples.bellwether import researcher
from concordia.examples.bellwether import service
from concordia.language_model import call_limit_wrapper
from concordia.language_model import profiled_language_model
from concordia.utils import profiler
from concordia.utils import simulation_server
from concordia.utils import visual_interface


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--editor-port', type=int, default=8784)
  parser.add_argument('--player-port', type=int, default=8785)
  parser.add_argument(
      '--output',
      type=pathlib.Path,
      default=pathlib.Path('runs/bellwether-fixture'),
  )
  parser.add_argument(
      '--mode',
      choices=('slice', 'fixture', 'live'),
      default='slice',
      help=(
          'slice: original one-turn demo; fixture: scripted full night; live:'
          ' local model residents'
      ),
  )
  parser.add_argument(
      '--resident-prefab', choices=('minimal', 'basic'), default='minimal'
  )
  parser.add_argument('--model', default='llama3.2:3b')
  parser.add_argument(
      '--embedding-model',
      help=(
          'Already-installed local Ollama embedding model; default is a'
          ' constant placeholder.'
      ),
  )
  parser.add_argument(
      '--recipe',
      choices=researcher.RECIPES,
      default='bellwether',
      help=(
          'Trusted initial teaching case; selection never replaces an active'
          ' run.'
      ),
  )
  parser.add_argument(
      '--multiplayer',
      action='store_true',
      help=(
          'Host-approved Coordinator and Nell browser sessions; three AI'
          ' residents.'
      ),
  )
  parser.add_argument(
      '--public-origin',
      help=(
          'Exact HTTPS origin when proxying the player listener; never proxy'
          ' the editor.'
      ),
  )
  parser.add_argument(
      '--cookie-path',
      default='/',
      help='Player mount path, including leading and trailing slash.',
  )
  parser.add_argument(
      '--dispute-file',
      type=pathlib.Path,
      help='Optional JSON with text and recipients for High Tide.',
  )
  args = parser.parse_args(argv)
  if args.multiplayer and args.mode == 'slice':
    parser.error('--multiplayer requires --mode fixture or live')
  if args.recipe != 'bellwether' and args.mode == 'slice':
    parser.error('--recipe requires --mode fixture or live')
  if args.recipe != 'bellwether' and args.dispute_file:
    parser.error('--dispute-file requires --recipe bellwether')
  if args.embedding_model and args.mode == 'slice':
    parser.error('--embedding-model requires --mode fixture or live')
  dispute = (
      json.loads(args.dispute_file.read_text(encoding='utf-8'))
      if args.dispute_file
      else None
  )
  profile = profiler.ProfilerContext()
  profile.enable()
  embedder = (
      local_embeddings.OllamaEmbedder(args.embedding_model, profiler=profile)
      if args.embedding_model
      else None
  )
  model = None
  action_model = None
  if args.mode == 'live':

    def bounded_model(max_calls, response_format=None):
      return call_limit_wrapper.CallLimitLanguageModel(
          profiled_language_model.ProfiledLanguageModel(
              ollama_model.OllamaLanguageModel(
                  args.model,
                  request_timeout=90,
                  max_output_tokens=256,
                  response_format=response_format,
              ),
              model_name=args.model,
              profiler_instance=profile,
          ),
          max_calls=max_calls,
      )

    # Normal prose for basic perception; schema only for final resident acts.
    # Two existing counters reserve a combined maximum of 256 local calls.
    model = bounded_model(192)
    action_model = bounded_model(64, rules.resident_response_schema())
  game = (
      service.Bellwether(args.output, port=args.editor_port)
      if args.mode == 'slice'
      else (multiplayer.SharedGame if args.multiplayer else game_service.Game)(
          args.output,
          port=args.editor_port,
          model=model,
          action_model=action_model,
          embedder=embedder,
          actor_logic=args.resident_prefab,
          recipe=args.recipe,
          dispute=dispute,
          profiler=profile,
          **(
              {
                  'secure': bool(args.public_origin),
                  'cookie_path': args.cookie_path,
              }
              if args.multiplayer
              else {}
          ),
      )
  )
  if embedder is not None:
    game.embedding.update(kind='local_ollama', model=args.embedding_model)
  player = simulation_server.SimulationServer(
      port=args.player_port,
      html_content=pathlib.Path(__file__)
      .with_name('player.html')
      .read_text(encoding='utf-8'),
      operation_service=game.operations,
      audience='player',
      browser_sessions=(
          game.sessions if isinstance(game, multiplayer.SharedGame) else None
      ),
      public_origin=args.public_origin,
  )
  game.server.set_html_content(
      visual_interface.visualize_operations_to_html(
          game.config, title='Bellwether · ' + args.mode + ' editor'
      )
  )
  try:
    game.server.start()
    player.start()
    print(f'Editor: http://127.0.0.1:{game.server.bound_port}', flush=True)
    print(f'Player: http://127.0.0.1:{player.bound_port}', flush=True)
    print(
        'Mode: '
        + args.mode
        + '. Explicit Begin / run.start starts this session. No automatic'
        ' replay.',
        flush=True,
    )
    while True:
      time.sleep(0.5)
  except KeyboardInterrupt:
    pass
  finally:
    game.close()
    player.stop()


if __name__ == '__main__':
  main()
