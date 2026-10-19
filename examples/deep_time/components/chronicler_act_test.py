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

import contextlib
import io

from absl.testing import absltest
from concordia.agents import entity_agent
from examples.deep_time import model_setup
from examples.deep_time import simulation
from examples.deep_time.components import chronicler_act


class ChroniclerActTest(absltest.TestCase):

  def test_the_chronicler_hands_in_each_chapter_once_by_its_action(self):
    sim, context = simulation.build_simulation(
        'daisyworld', model_setup.make_models('mock'), seed=1
    )
    (chronicler,) = [e for e in sim.get_entities() if e.name == 'Chronicler']
    assert isinstance(chronicler, entity_agent.EntityAgent)
    self.assertIsInstance(
        chronicler.get_act_component(), chronicler_act.ChroniclerAct
    )
    with contextlib.redirect_stdout(io.StringIO()):  # The engine's log.
      sim.play(premise='', max_steps=20)
    chapters = [
        m for m in context.bus.history('chronicle') if m.source == 'Chronicler'
    ]
    self.assertNotEmpty(chapters)
    # Each chapter its mind finished, handed in once by its action.
    texts = [m.payload for m in chapters]
    self.assertLen(set(texts), len(texts))


if __name__ == '__main__':
  absltest.main()
