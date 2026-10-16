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

r"""CLI for the golden sets (see golden.py).

python -m examples.crucible.golden_cli --mode=mock
python -m examples.crucible.golden_cli --mode=ollama --model_name=qwen3:8b
"""

from absl import app
from absl import flags

from examples.crucible import golden
from examples.crucible import golden_passiton
from examples.crucible import model_setup

_TOY = flags.DEFINE_enum('toy', 'orders', ['orders', 'passiton'], 'Whose set.')
_MODE = flags.DEFINE_enum('mode', 'mock', ['mock', 'ollama', 'together'], '')
_MODEL_NAME = flags.DEFINE_string('model_name', None, 'Provider model name.')
_MAX_CALLS = flags.DEFINE_integer(
    'max_calls', 0, 'Live call cap (default: 3 calls per case per run).'
)
_REPEATS = flags.DEFINE_integer(
    'repeats', 1, 'Runs of the whole set; every run must pass.'
)
_CASES = flags.DEFINE_list(
    'cases', [], 'Case ids to run (default: all), to run a slow model in parts.'
)


def main(argv: list[str]) -> None:
  del argv
  module = golden_passiton if _TOY.value == 'passiton' else golden
  count = len(_CASES.value) or len(module.cases())
  calls = _MAX_CALLS.value or 3 * count * _REPEATS.value
  model = model_setup.build_model(_MODE.value, _MODEL_NAME.value, calls)
  print(module.main_json(model, _REPEATS.value, _CASES.value))


if __name__ == '__main__':
  app.run(main)
