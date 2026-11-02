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

"""Tests for the asynchronous social media game master prefabs."""

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import next_game_master
from concordia.contrib.prefabs.game_master import async_social_media_with_moderation
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import async_social_media
from concordia.typing import entity as entity_lib
import numpy as np


class AsyncSocialMediaGameMasterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('async_social_media', async_social_media.GameMaster),
      (
          'async_social_media_with_moderation',
          async_social_media_with_moderation.GameMaster,
      ),
  )
  def test_names_itself_as_the_next_game_master(self, prefab_class):
    prefab = prefab_class(params={'name': 'forum_rules'})
    game_master = prefab.build(
        model=no_language_model.NoLanguageModel(),
        memory_bank=basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=lambda _: np.ones(3),
        ),
    )

    chosen = game_master.act(
        entity_lib.ActionSpec(
            call_to_action=next_game_master.DEFAULT_CALL_TO_NEXT_GAME_MASTER,
            output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=('initial setup', 'forum_rules'),
        )
    )

    # The engine looks this name up among its game masters, so it must be the
    # bare name, not a labeled context block.
    self.assertEqual(chosen, 'forum_rules')


if __name__ == '__main__':
  absltest.main()
