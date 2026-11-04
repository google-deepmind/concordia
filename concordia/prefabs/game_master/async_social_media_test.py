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

import json
import threading

from absl.testing import absltest
from absl.testing import parameterized
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import event_resolution
from concordia.components.game_master import next_game_master
from concordia.contrib.components.game_master import forum
from concordia.contrib.prefabs.game_master import async_social_media_with_moderation
from concordia.language_model import no_language_model
from concordia.prefabs.entity import minimal as minimal_entity
from concordia.prefabs.game_master import async_social_media
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import numpy as np


class AsyncSocialMediaGameMasterTest(parameterized.TestCase):

  def _build_forum_game_master(self, prefab_class, names):
    model = no_language_model.NoLanguageModel()

    def memory_bank():
      return basic_associative_memory.AssociativeMemoryBank(
          sentence_embedder=lambda _: np.ones(3)
      )

    players = [
        minimal_entity.Entity(params={'name': name}).build(
            model=model, memory_bank=memory_bank()
        )
        for name in names
    ]
    return prefab_class(params={'name': 'forum_rules'}, entities=players).build(
        model=model, memory_bank=memory_bank()
    )

  def _resolve_post(
      self,
      game_master,
      name,
      title,
      separator=': ',
      content='Fixture',
      memory_prefix='',
      observe=True,
  ):
    thread_id = threading.get_ident()
    game_master.set_capture_key_for_thread(thread_id, name)
    try:
      if observe:
        payload = json.dumps({
            'action': 'post',
            'author': name,
            'title': title,
            'content': content,
        })
        game_master.observe(
            f'{memory_prefix}{event_resolution.PUTATIVE_EVENT_TAG} '
            f'{name}{separator}{payload}'
        )
      return game_master.act(
          entity_lib.ActionSpec(
              call_to_action='Resolve',
              output_type=entity_lib.OutputType.RESOLVE,
          )
      )
    finally:
      game_master.clear_capture_key_for_thread(thread_id)

  def _posts(self, game_master):
    state = game_master.get_component(
        forum.DEFAULT_FORUM_COMPONENT_KEY, type_=forum.ForumState
    )
    return [
        (post.author, post.title) for post in reversed(state.get_recent_posts())
    ]

  @parameterized.product(
      prefab_class=(
          async_social_media.GameMaster,
          async_social_media_with_moderation.GameMaster,
      ),
      names=(
          ('Alice', 'Alice2'),
          ('Alice', 'Alice Smith'),
          ('Alice', 'Alice:Admin'),
      ),
      longer_first=(False, True),
      separator=(': ', ' -- ', ' '),
  )
  def test_resolves_each_prefix_named_players_own_action(
      self, prefab_class, names, longer_first, separator
  ):
    game_master = self._build_forum_game_master(prefab_class, names)
    actors = list(reversed(names)) if longer_first else list(names)
    expected = []
    for i, name in enumerate(actors):
      title = f'Action {i} by {name}'
      self._resolve_post(game_master, name, title, separator=separator)
      expected.append((name, title))
    self.assertEqual(self._posts(game_master), expected)

  @parameterized.product(
      prefab_class=(
          async_social_media.GameMaster,
          async_social_media_with_moderation.GameMaster,
      ),
  )
  def test_payload_mentions_of_another_tag_are_not_that_players_actions(
      self, prefab_class
  ):
    game_master = self._build_forum_game_master(prefab_class, ('Alice', 'Bob'))
    self._resolve_post(
        game_master,
        'Alice',
        'A quotation',
        content=(
            f'A harmless quotation: {event_resolution.PUTATIVE_EVENT_TAG} Bob:'
            ' text'
        ),
    )
    self._resolve_post(game_master, 'Bob', 'Bobs actual action')
    self.assertEqual(
        self._posts(game_master),
        [
            ('Alice', 'A quotation'),
            ('Bob', 'Bobs actual action'),
        ],
    )

  @parameterized.product(
      prefab_class=(
          async_social_media.GameMaster,
          async_social_media_with_moderation.GameMaster,
      ),
  )
  def test_capture_key_does_not_consume_a_prefix_named_players_pending_action(
      self, prefab_class
  ):
    game_master = self._build_forum_game_master(
        prefab_class, ('Alice', 'Alice2')
    )
    payload = json.dumps({
        'action': 'post',
        'author': 'Alice2',
        'title': 'Pending',
        'content': 'Fixture',
    })
    game_master.observe(
        f'{event_resolution.PUTATIVE_EVENT_TAG} Alice2: {payload}'
    )
    self._resolve_post(game_master, 'Alice', '', observe=False)
    self.assertEmpty(self._posts(game_master))
    self._resolve_post(game_master, 'Alice2', '', observe=False)
    self.assertEqual(self._posts(game_master), [('Alice2', 'Pending')])

  @parameterized.product(
      prefab_class=(
          async_social_media.GameMaster,
          async_social_media_with_moderation.GameMaster,
      ),
  )
  def test_restored_per_entity_counters_keep_timestamped_actions_separate(
      self, prefab_class
  ):
    game_master = self._build_forum_game_master(
        prefab_class, ('Alice', 'Alice2')
    )
    self._resolve_post(
        game_master, 'Alice2', 'First', memory_prefix='[timestamp] '
    )
    resolution = game_master.get_component(
        event_resolution.DEFAULT_RESOLUTION_COMPONENT_KEY,
        type_=entity_component.ContextComponent,
    )
    resolution.set_state(json.loads(json.dumps(resolution.get_state())))
    self._resolve_post(
        game_master, 'Alice', 'Second', memory_prefix='[timestamp] '
    )
    self._resolve_post(
        game_master, 'Alice2', 'Third', memory_prefix='[timestamp] '
    )
    self.assertEqual(
        self._posts(game_master),
        [
            ('Alice2', 'First'),
            ('Alice', 'Second'),
            ('Alice2', 'Third'),
        ],
    )

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
