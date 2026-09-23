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

"""Terminal transport for the same adventure and HumanActComponent API."""

import argparse
import pathlib

from concordia.contrib.language_models.ollama import ollama_model
from concordia.examples.astral_canticle import adventure


class TerminalInput:
  """A simple synchronous reader demonstrating a non-web transport."""

  def __call__(self, request):
    print(request.context)
    print(f'\n{request.entity_name}: {request.action_spec.call_to_action}')
    if request.action_spec.options:
      print('Choose exactly: ' + ' | '.join(request.action_spec.options))
    if request.error:
      print(request.error)
    return input('> ')

  def progress(self, step, actor):
    print(f'\nTurn {step}: {actor} acted.')

  def add_observation(self, observation):
    print('\n' + observation)

  def finish(self, message):
    print('\n' + message)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--role', choices=('player', 'gm'), default='player')
  parser.add_argument(
      '--player-prefab',
      choices=('minimal', 'basic'),
      default='minimal',
      help='Standard prefab for Ilyra; basic retains its LLM perceptions.',
  )
  parser.add_argument('--model', default='llama3.2:3b')
  parser.add_argument('--max-steps', type=int, default=30)
  parser.add_argument(
      '--output', type=pathlib.Path, default=pathlib.Path('runs/terminal')
  )
  args = parser.parse_args()
  if not 1 <= args.max_steps <= 300:
    parser.error('--max-steps must be between 1 and 300')
  try:
    adventure.play(
        ollama_model.OllamaLanguageModel(args.model),
        TerminalInput(),
        args.output,
        role=args.role,
        max_steps=args.max_steps,
        player_prefab=args.player_prefab,
    )
  except (EOFError, KeyboardInterrupt):
    print('\nStopped. Completed turns are saved in ' + str(args.output))


if __name__ == '__main__':
  main()
