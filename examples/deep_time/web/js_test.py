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

"""The page's script modules: inlined whole, and tested.

The modules' own tests run under Node's test runner (`node --test`), with
payloads that real games published, so every instrument widget draws what
the game really sends. Without Node the JS tests are skipped.
"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import unittest

from absl.testing import absltest
from examples.deep_time import model_setup
from examples.deep_time import simulation
from examples.deep_time.web import server as server_lib

NODE = shutil.which('node')
DECLARED = re.compile(
    r'^export (?:async )?(?:function|const|let|class) ([A-Za-z_$][\w$]*)', re.M
)


def _payloads() -> dict[str, list[object]]:
  """The last payload of each topic, from a few short games."""
  found: dict[str, list[object]] = {}
  for scenario in ('sandbox', 'light_delay', 'daisyworld', 'herds'):
    sim, context = simulation.build_simulation(
        scenario, model_setup.make_models('mock'), max_ticks=3
    )
    with contextlib.redirect_stdout(io.StringIO()):
      sim.play(premise='', max_steps=3)
    for topic in context.bus.topics():
      history = context.bus.history(topic)
      if not history:
        continue
      try:
        data = json.loads(history[-1].payload)
      except (TypeError, ValueError):
        continue
      if isinstance(data, dict):
        found.setdefault(topic, []).append(data)
  return found


class PageScriptTest(absltest.TestCase):

  def test_every_module_is_inlined_whole(self):
    html = server_lib.page().decode('utf-8')
    self.assertNotIn(server_lib.MODULES_MARK, html)
    self.assertNotRegex(html, r'(?m)^(?:import|export) ')
    for name in server_lib.MODULES:
      with open(os.path.join(server_lib.SCRIPTS, name), encoding='utf-8') as f:
        source = f.read()
      # Imports take one line each, so the server can drop them.
      for line in source.splitlines():
        if line.startswith('import '):
          self.assertTrue(line.endswith(';'), f'{name}: {line}')
      for declared in DECLARED.findall(source):
        self.assertRegex(
            html,
            rf'(?m)^(?:async )?(?:function|const|let|class) {declared}\b',
            f'{name}: {declared}',
        )

  def test_every_module_on_disk_is_served(self):
    on_disk = sorted(
        f for f in os.listdir(server_lib.SCRIPTS) if f.endswith('.js')
    )
    self.assertEqual(sorted(server_lib.MODULES), on_disk)

  @unittest.skipIf(NODE is None, 'Node is not installed')
  def test_the_modules_pass_their_tests(self):
    path = os.path.join(self.create_tempdir().full_path, 'payloads.json')
    payloads = _payloads()
    self.assertContainsSubset(
        {'climate.global', 'bio.census', 'cycles.carbon', 'colony.status'},
        payloads,
    )
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(payloads, f)
    tests = sorted(
        os.path.join(server_lib.SCRIPTS, f)
        for f in os.listdir(server_lib.SCRIPTS)
        if f.endswith('.test.mjs')
    )
    self.assertNotEmpty(tests)
    assert NODE is not None
    result = subprocess.run(
        [NODE, '--test', *tests],
        env={**os.environ, 'DEEP_TIME_PAYLOADS': path},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
  absltest.main()
