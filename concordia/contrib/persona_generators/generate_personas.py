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

"""Standalone CLI utility to generate diverse personas.

Generates persona characteristics and formative memories using the formalized
registry and protocol. Outputs the results as a JSON file.

See: Persona Generators: Generating Diverse Synthetic Personas at Scale
     https://arxiv.org/abs/2602.03545
"""

import argparse
import importlib
import json
from absl import logging
from concordia.contrib.persona_generators import persona_generator_five
from concordia.contrib.persona_generators import persona_generator_four
from concordia.contrib.persona_generators import persona_generator_one
from concordia.contrib.persona_generators import persona_generator_three
from concordia.contrib.persona_generators import persona_generator_two
from concordia.contrib.persona_generators import two_stage_persona_generator
from concordia.language_model import language_model as concordia_lm

GENERATORS = {
    'base': two_stage_persona_generator.TwoStagePersonaGenerator,
    'alphaevolve_1': persona_generator_one.TwoStagePersonaGenerator,
    'alphaevolve_2': persona_generator_two.TwoStagePersonaGenerator,
    'alphaevolve_3': persona_generator_three.TwoStagePersonaGenerator,
    'alphaevolve_4': persona_generator_four.TwoStagePersonaGenerator,
    'alphaevolve_5': persona_generator_five.TwoStagePersonaGenerator,
}


def _create_language_model(
    api_type: str, model_name: str
) -> concordia_lm.LanguageModel:
  """Creates a language model using concordia.contrib.language_models.

  Uses importlib to avoid a static dependency on the language_models
  package, which lives in a different build target.

  Args:
    api_type: Key in the language_models registry (e.g. 'gemini', 'ollama').
    model_name: Model name passed to the constructor.

  Returns:
    An instantiated LanguageModel.
  """
  try:
    setup_module = importlib.import_module(
        'concordia.contrib.language_models'
    )
  except ImportError as e:
    raise ImportError(
        'Could not import concordia.contrib.language_models. '
        'Install with: pip install gdm-concordia[google]'
    ) from e
  return setup_module.language_model_setup(
      api_type=api_type,
      model_name=model_name,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description='Generate diverse personas.')
  parser.add_argument(
      '--output_path',
      required=True,
      help='Path to save the generated personas (JSON).',
  )
  parser.add_argument(
      '--num_personas',
      type=int,
      default=5,
      help='The number of personas to generate.',
  )
  parser.add_argument(
      '--generator', default='base', help='Which persona generator to use.'
  )
  parser.add_argument(
      '--initial_context', default='', help='Shared context for all personas.'
  )
  parser.add_argument(
      '--diversity_axes',
      default='',
      help='Comma-separated list of axes to encourage diversity.',
  )
  parser.add_argument(
      '--shared_memories',
      default='',
      help='Comma-separated list of shared memories.',
  )
  parser.add_argument(
      '--api_type', default='google_aistudio', help='The type of API to use.'
  )
  parser.add_argument(
      '--model_name', default='gemini-2.5-pro', help='Model name to use.'
  )

  args = parser.parse_args()

  logging.info('Creating language model...')
  try:
    model = _create_language_model(args.api_type, args.model_name)
  except ValueError as e:
    logging.fatal('Failed to create language model: %s', e)
    return

  logging.info('Loading generator: %s', args.generator)
  generator_cls = GENERATORS.get(args.generator)
  if not generator_cls:
    logging.fatal('Unknown generator: %s', args.generator)
    return

  generator = generator_cls(model)

  diversity_axes = args.diversity_axes.split(',') if args.diversity_axes else []
  shared_memories = (
      args.shared_memories.split(',') if args.shared_memories else []
  )

  logging.info(
      'Stage 1: Generating characteristics for %d personas...',
      args.num_personas,
  )
  characteristics = generator.generate_diverse_persona_characteristics(
      initial_context=args.initial_context,
      diversity_axes=diversity_axes,
      num_personas=args.num_personas,
  )

  logging.info('Generated %d characteristics.', len(characteristics))

  personas_output = []
  for i, char in enumerate(characteristics):
    name = char.get('name', f'Persona_{i}')
    logging.info('Stage 2: Generating memories for %s...', name)
    memories = generator.generate_single_persona_memories(char)

    personas_output.append({
        'name': name,
        'characteristics': char,
        'memories': memories,
        'shared_memories': shared_memories,
    })

  logging.info('Saving output to %s', args.output_path)
  with open(args.output_path, 'w') as f:
    json.dump(personas_output, f, indent=2)

  logging.info('Done!')


if __name__ == '__main__':
  main()
