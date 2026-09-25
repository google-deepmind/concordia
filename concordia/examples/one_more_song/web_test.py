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

"""Launch safety for hosts replaying the example; no simulations are run."""

import contextlib
import io
import pathlib
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from concordia.examples.one_more_song import web


class LaunchTest(absltest.TestCase):

  def test_unplayed_run_reserves_output_before_first_action(self):
    with tempfile.TemporaryDirectory() as directory:
      output = pathlib.Path(directory) / 'game'
      with mock.patch.object(
          sys, 'argv', ['web', '--fixture', '--output', str(output)]
      ):
        with mock.patch.object(
            web.game, 'build', return_value=(None, None)
        ) as build:
          with mock.patch.object(web.human_web, 'create_app'):
            with mock.patch.object(web.uvicorn, 'run'):
              with contextlib.redirect_stdout(io.StringIO()):
                web.main()
                with contextlib.redirect_stderr(io.StringIO()):
                  with self.assertRaises(SystemExit) as stopped:
                    web.main()
      self.assertEqual(stopped.exception.code, 2)
      self.assertEqual(build.call_count, 1)

  def test_existing_run_is_rejected_before_build_and_kept_intact(self):
    with tempfile.TemporaryDirectory() as directory:
      output = pathlib.Path(directory)
      saved = output / 'public.json'
      saved.write_text('previous conversation')
      with mock.patch.object(
          sys, 'argv', ['web', '--fixture', '--output', directory]
      ):
        with mock.patch.object(web.game, 'build') as build:
          with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as stopped:
              web.main()
          self.assertEqual(stopped.exception.code, 2)
          self.assertIn('Choose a new directory', error.getvalue())
          build.assert_not_called()
      self.assertEqual(saved.read_text(), 'previous conversation')

  def test_default_replays_use_separate_output_directories(self):
    outputs = []
    with tempfile.TemporaryDirectory() as directory:
      with contextlib.chdir(directory):
        with mock.patch.object(sys, 'argv', ['web', '--fixture']):
          with mock.patch.object(web.game, 'build', return_value=(None, None)):
            with mock.patch.object(web.game, 'play') as play:
              with mock.patch.object(web.human_web, 'create_app') as create:
                with mock.patch.object(web.uvicorn, 'run'):
                  with contextlib.redirect_stdout(io.StringIO()) as text:
                    for _ in range(2):
                      web.main()
                      create.call_args.kwargs['runner']()
                      outputs.append(play.call_args.args[2])
    self.assertNotEqual(outputs[0], outputs[1])
    self.assertTrue(all(p.parent == pathlib.Path('runs') for p in outputs))
    self.assertIn('http://127.0.0.1:8820/', text.getvalue())


if __name__ == '__main__':
  absltest.main()
