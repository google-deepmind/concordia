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

r"""Launch resident & visitor agents filling N-1 slots and 1 slot respectively.

Usage:
cd {concordia_root}/
PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_resident_visitor.py \
  --agent_resident=RESIDENT_AGENT_NAME \
  --agent_visitor=VISITOR_AGENT_NAME \
  --environment=ENVIRONMENT_NAME \
  --api_type=API_TYPE \
  --model=MODEL_NAME \
  --embedder=EMBEDDER_NAME

Where AGENT_NAME indicates a file under concordia/factory/agent,
ENVIRONMENT_NAME indicates a file under examples/modular/environment,
API_TYPE is one of the options named in concordia/language_model/utils.py,
e.g. 'google_aistudio_model', 'openai', 'mistral', 'ollama', 'amazon_bedrock'.
MODEL_NAME is a specific model under the chosen API_TYPE. See the corresponding
wrapper in concordia/language_model/ for the link to the website where the
model names are listed for each type of API.
and EMBEDDER_NAME specifies a sentence transformers embedding model listed at
https://huggingface.co/sentence-transformers.

This script will download the embedder from huggingface and cache it locally.

To debug without spending money on API calls, pass the the option:
  --disable_language_model
It replaces the language model with a null model that always returns an empty
string when asked for a free response and always selects the first option when
asked for a multiple choice.
"""

import argparse
import datetime
import importlib

from concordia.language_model import call_limit_wrapper
from concordia.language_model import utils
from concordia.utils.deprecated import measurements as measurements_lib
import numpy as np
import sentence_transformers


# Setup for command line arguments
parser = argparse.ArgumentParser(description='Run a GDM-Concordia simulation.')
parser.add_argument('--agent_resident',
                    action='store',
                    default='basic_agent',
                    dest='resident_agent_name')
parser.add_argument('--agent_visitor',
                    action='store',
                    default='rational_agent',
                    dest='visitor_agent_name')
parser.add_argument('--environment',
                    action='store',
                    default='reality_show',
                    dest='environment_name')
parser.add_argument('--api_type',
                    action='store',
                    default='mistral',
                    dest='api_type')
parser.add_argument('--model',
                    action='store',
                    default='codestral-latest',
                    dest='model_name')
parser.add_argument('--embedder',
                    action='store',
                    default='all-mpnet-base-v2',
                    dest='embedder_name')
parser.add_argument('--api_key',
                    action='store',
                    default=None,
                    dest='api_key')
parser.add_argument(
    '--device',
    action='store',
    default=None,
    dest='device',
    help=(
        'Device to use for model inference (e.g., "cuda:0" for '
        'GPU, "cpu" for CPU). Only applies to local models.'
    ),
)
parser.add_argument('--disable_language_model',
                    action='store_true',
                    help=('replace the language model with a null model. This '
                          'makes it possible to debug without spending money '
                          'on api calls.'),
                    default=False,
                    dest='disable_language_model')
# Parse command line arguments
args = parser.parse_args()

# Load the agent config with importlib
IMPORT_AGENT_BASE_DIR = 'concordia.factory.agent'
resident_agent_module = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.{args.resident_agent_name}')
visitor_agent_module = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.{args.visitor_agent_name}')
# Load the environment config with importlib
IMPORT_ENV_BASE_DIR = 'examples.deprecated.modular.environment'
simulation = importlib.import_module(
    f'{IMPORT_ENV_BASE_DIR}.{args.environment_name}')

# Language Model setup
model = utils.language_model_setup(
    api_type=args.api_type,
    model_name=args.model_name,
    api_key=args.api_key,
    device=args.device,
    disable_language_model=args.disable_language_model,
)

# Setup sentence encoder
if not args.disable_language_model:
  st_model = sentence_transformers.SentenceTransformer(
      f'sentence-transformers/{args.embedder_name}'
  )
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)
else:
  embedder = lambda x: np.ones(5)

# Initialize the simulation
measurements = measurements_lib.Measurements()
runnable_simulation = simulation.Simulation(
    model=model,
    embedder=embedder,
    measurements=measurements,
    resident_visitor_modules=(resident_agent_module, visitor_agent_module),
    override_agent_model=call_limit_wrapper.CallLimitLanguageModel(model),
)
# Run the simulation
_, results_log = runnable_simulation()

# Write the results log as an HTML file in the current working directory.
filename = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.html'
file_handle = open(filename, 'a')
file_handle.write(results_log)
file_handle.close()
