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
from examples.crucible import model_setup

_MODE = flags.DEFINE_enum('mode', 'mock', ['mock', 'ollama', 'together'], '')
_MODEL_NAME = flags.DEFINE_string('model_name', None, 'Provider model name.')
_MAX_CALLS = flags.DEFINE_integer('max_calls', 100, 'Live call cap.')


def main(argv: list[str]) -> None:
  del argv
  model = model_setup.build_model(
      _MODE.value, _MODEL_NAME.value, _MAX_CALLS.value
  )
  print(golden.main_json(model))


if __name__ == '__main__':
  app.run(main)
