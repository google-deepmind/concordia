# Copyright 2023 DeepMind Technologies Limited.
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

r"""Launch a simulation, replicating colab workflows in a regular python script.

Usage:
cd {concordia_root}/
python examples/modular/launch.py \
  --agent=AGENT_NAME \
  --environment=ENVIRONMENT_NAME \
  --api_type=API_TYPE \
  --model=MODEL_NAME \
  --embedder=EMBEDDER_NAME

Where AGENT_NAME indicates a file under concordia/factory/agent,
ENVIRONMENT_NAME indicates a file under examples/modular/environment,
API_TYPE is either 'openai' or 'mistral',
MODEL_NAME is a model listed at https://platform.openai.com/docs/models or
https://docs.mistral.ai/getting-started/models/,
and EMBEDDER_NAME specifies a sentence transformers embedding model listed at
https://huggingface.co/sentence-transformers.

This script will download the embedder from huggingface and cache it locally.

To debug without spending money on API calls, pass the the option:
  --disable_language_model
It replaces the language model with a null model that always returns an empty
string when asked for a free response and alwats selects the first option when
asked for a multiple choice.
"""

import argparse
import datetime
import importlib
import pathlib
import sys

from concordia.language_model import amazon_bedrock_model
from concordia.language_model import google_aistudio_model
from concordia.language_model import gpt_model
from concordia.language_model import langchain_ollama_model
from concordia.language_model import mistral_model
from concordia.language_model import no_language_model
from concordia.language_model import ollama_model
from concordia.language_model import pytorch_gemma_model
from concordia.utils import measurements as measurements_lib
import sentence_transformers


def language_model_setup(args):
  """Get the wrapped language model."""
  if args.disable_language_model:
    return no_language_model.NoLanguageModel()
  elif args.api_type == 'amazon_bedrock':
    return amazon_bedrock_model.AmazonBedrockLanguageModel(args.model_name)
  elif args.api_type == 'google_aistudio_model':
    return google_aistudio_model.GoogleAIStudioLanguageModel(args.model_name)
  elif args.api_type == 'langchain_ollama':
    return langchain_ollama_model.LangchainOllamaLanguageModel(args.model_name)
  elif args.api_type == 'mistral':
    return mistral_model.MistralLanguageModel(args.model_name)
  elif args.api_type == 'ollama':
    return ollama_model.OllamaLanguageModel(args.model_name)
  elif args.api_type == 'openai':
    return gpt_model.GptLanguageModel(args.model_name)
  elif args.api_type == 'pytorch_gemma':
    return pytorch_gemma_model.PyTorchGemmaLanguageModel(args.model_name)
  else:
    raise ValueError(f'Unrecognized api type: {args.api_type}')


# Setup for command line arguments
parser = argparse.ArgumentParser(description='Run a GDM-Concordia simulation.')
parser.add_argument('--agent',
                    action='store',
                    default='basic_entity_agent__main_role',
                    dest='agent_name')
parser.add_argument('--environment',
                    action='store',
                    default='reality_show',
                    dest='environment_name')
parser.add_argument('--api_type',
                    action='store',
                    default='openai',
                    dest='api_type')
parser.add_argument('--model',
                    action='store',
                    default='gpt-4o',
                    dest='model_name')
parser.add_argument('--embedder',
                    action='store',
                    default='all-mpnet-base-v2',
                    dest='embedder_name')
parser.add_argument('--disable_language_model',
                    action='store_true',
                    help=('replace the language model with a null model. This '
                          'makes it possible to debug without spending money '
                          'on api calls.'),
                    default=False,
                    dest='disable_language_model')
# Parse command line arguments
command_line_args = parser.parse_args()

# Load the agent config with importlib
IMPORT_AGENT_BASE_DIR = 'concordia.factory.agent'
agent_module = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.{command_line_args.agent_name}')
# Load the environment config with importlib
concordia_root_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(f'{concordia_root_dir}')
IMPORT_ENV_BASE_DIR = 'environment'
simulation = importlib.import_module(
    f'{IMPORT_ENV_BASE_DIR}.{command_line_args.environment_name}')

# Language Model setup
model = language_model_setup(command_line_args)
# Setup sentence encoder
st_model = sentence_transformers.SentenceTransformer(
    f'sentence-transformers/{command_line_args.embedder_name}')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

# Initialize the simulation
measurements = measurements_lib.Measurements()
runnable_simulation = simulation.Simulation(
    model=model,
    embedder=embedder,
    measurements=measurements,
    agent_module=agent_module,
)
# Run the simulation
results_log = runnable_simulation()

# Write the results log as an HTML file in the current working directory.
filename = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.html'
file_handle = open(filename, 'a')
file_handle.write(results_log)
file_handle.close()
