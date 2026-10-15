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

r"""Play audit: does skill show? (GUIDE.md, Phase 1 and Phase 4.)

Plays a toy's seat with the skilled, random and passive bots at many seeds
against the mock, and reports each bot's win rate and the skill gaps, plus
how many moves a turn offers that measurably change the board.

  python -m examples.crucible.play_audit --toy=claims --seeds=16
"""

import json

from absl import app
from absl import flags

from examples.crucible import audits

_TOY = flags.DEFINE_string('toy', 'claims', 'Which toy.')
_SCENARIO = flags.DEFINE_string('scenario', None, 'Toy scenario.')
_SEEDS = flags.DEFINE_integer('seeds', 16, 'Games per bot.')


def main(argv: list[str]) -> None:
  del argv
  print(json.dumps(audits.audit(_TOY.value, _SCENARIO.value, _SEEDS.value)))


if __name__ == '__main__':
  app.run(main)
