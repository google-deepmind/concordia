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

"""Restored duplicate suppression must not depend on the writer's hash seed."""

import json
import os
import pathlib
import subprocess
import sys

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory as memory
from concordia.components.agent import memory as memory_component
import numpy as np

# Run the real serialized state through independent interpreters. No provider
# request or embedding model is used.
_WORKER = """
import json
import sys
import numpy as np
from concordia.associative_memory import basic_associative_memory as memory
request = json.load(sys.stdin)
calls = []
def embed(text):
    calls.append(text)
    return np.array([len(text), 1.], dtype=float)
bank = memory.AssociativeMemoryBank(embed, allow_duplicates=request.get('allow_duplicates', False))
if request.get('state') is not None:
    bank.set_state(request['state'])
bank.extend(request.get('texts', []))
print(json.dumps({'state': bank.get_state(), 'texts': bank.get_all_memories_as_text(), 'calls': calls}))
"""


def run_worker(seed, request):
  env = dict(os.environ, PYTHONHASHSEED=str(seed))
  source_root = str(pathlib.Path(memory.__file__).resolve().parents[2])
  env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
  result = subprocess.run(
      [sys.executable, '-c', _WORKER],
      input=json.dumps(request),
      text=True,
      capture_output=True,
      env=env,
      check=True,
      timeout=30,
  )
  return json.loads(result.stdout)


class MemoryRestoreHashTest(parameterized.TestCase):

  @parameterized.parameters((11, 22), (22, 11), (0, 33))
  def test_cross_process_restore_rejects_existing_text_without_reembedding(
      self, writer, reader
  ):
    saved = run_worker(
        writer, {'texts': ['a saved observation', 'line one\nline two']}
    )
    restored = run_worker(
        reader,
        {
            'state': saved['state'],
            'texts': [
                'a saved observation',
                'line one line two',
                'a new observation',
            ],
        },
    )
    self.assertEqual(
        restored['texts'],
        ['a saved observation', 'line one line two', 'a new observation'],
    )
    self.assertEqual(restored['calls'], ['a new observation'])
    self.assertEqual(set(restored['state']), set(saved['state']))

  def test_repeated_roundtrips_across_seeds_keep_one_copy(self):
    state = None
    for seed in [41, 42, 43]:
      result = run_worker(
          seed, {'state': state, 'texts': ['one original memory']}
      )
      self.assertEqual(result['texts'], ['one original memory'])
      state = result['state']

  def test_allow_duplicates_still_preserves_repeated_events(self):
    saved = run_worker(
        11, {'texts': ['an event', 'an event'], 'allow_duplicates': True}
    )
    result = run_worker(
        22,
        {
            'state': saved['state'],
            'texts': ['an event'],
            'allow_duplicates': True,
        },
    )
    self.assertEqual(result['texts'], ['an event'] * 3)
    self.assertEqual(result['calls'], ['an event'])

  def test_loading_duplicate_allowed_history_does_not_delete_rows(self):
    saved = run_worker(
        11, {'texts': ['an event', 'an event'], 'allow_duplicates': True}
    )
    result = run_worker(22, {'state': saved['state'], 'texts': ['an event']})
    self.assertEqual(result['texts'], ['an event', 'an event'])
    self.assertEqual(result['calls'], [])

  def test_component_restore_uses_the_restored_bank_deduplication(self):
    saved = run_worker(11, {'texts': ['a saved observation']})
    bank = memory.AssociativeMemoryBank(lambda _: np.ones(2))
    bank.add('discard pending contents')
    component = memory_component.AssociativeMemory(bank)
    component.set_state({
        'memory_bank': saved['state'],
        'buffer': ['a saved observation', 'new event'],
    })
    component.update()
    self.assertEqual(
        bank.get_all_memories_as_text(), ['a saved observation', 'new event']
    )

  def test_same_process_restore_retains_embeddings_and_state_input(self):
    bank = memory.AssociativeMemoryBank(lambda text: np.array([len(text), 1.0]))
    bank.extend(['first event', 'second event'])
    state = json.loads(json.dumps(bank.get_state()))
    before = json.dumps(state, sort_keys=True)
    restored = memory.AssociativeMemoryBank(
        lambda _: self.fail('must not embed duplicates')
    )
    restored.set_state(state)
    restored.add('first event')
    self.assertEqual(
        restored.get_all_memories_as_text(), ['first event', 'second event']
    )
    np.testing.assert_array_equal(
        np.stack(restored.get_data_frame()['embedding'].tolist()),
        np.stack(bank.get_data_frame()['embedding'].tolist()),
    )
    self.assertEqual(json.dumps(state, sort_keys=True), before)

  def test_empty_checkpoint_can_still_accept_new_memories(self):
    saved = run_worker(11, {'texts': []})
    restored = run_worker(
        22, {'state': saved['state'], 'texts': ['first event', 'first event']}
    )
    self.assertEqual(restored['texts'], ['first event'])
    self.assertEqual(restored['calls'], ['first event'])


if __name__ == '__main__':
  absltest.main()
