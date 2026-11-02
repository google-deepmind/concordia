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

"""Tests for Scenario 6: The Broth Board (dynamic join & karma elimination)."""

import json
import threading

from absl.testing import absltest
from concordia.associative_memory import basic_associative_memory
from concordia.components.game_master import next_game_master as next_gm_lib
from concordia.components.game_master import player_roster
from concordia.contrib.components.game_master import forum as forum_module
from concordia.contrib.prefabs.game_master import async_social_media_with_moderation
from concordia.environment.engines import asynchronous
from examples.social_media import scenario_06_karma_bunker
from examples.social_media import shared as shared_lib
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import async_social_media
from concordia.typing import entity as entity_lib
import numpy as np


def _embedder(text: str) -> np.ndarray:
  del text
  return np.ones(4)


class _ScriptedDownvoteAgentModel(no_language_model.NoLanguageModel):
  """Deterministic agent model where Barnaby posts and peers downvote him."""

  def __init__(self, forum_state: forum_module.ForumState):
    super().__init__()
    self._forum_state = forum_state
    self._lock = threading.Lock()
    self._voted_by: set[str] = set()
    self._barnaby_posted = False

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    if 'SURVIVAL REMINDER FOR ' not in prompt:
      return ''
    marker = prompt.split('SURVIVAL REMINDER FOR ', 1)[1]
    actor = marker.split(':', 1)[0].strip()

    with self._lock:
      if actor == 'Barnaby Crumb' and not self._barnaby_posted:
        self._barnaby_posted = True
        return json.dumps({
            'action': 'post',
            'author': actor,
            'title': 'Rye Starter Supremacy',
            'content': 'Everyone else uses inferior hydration ratios!',
        })

      barnaby_posts = [
          post
          for post in self._forum_state.get_recent_posts(n=20)
          if post.author == 'Barnaby Crumb'
      ]
      if (
          actor != 'Barnaby Crumb'
          and barnaby_posts
          and actor not in self._voted_by
      ):
        self._voted_by.add(actor)
        return json.dumps({
            'action': 'downvote_post',
            'author': actor,
            'post_id': barnaby_posts[0].post_id,
        })

      return json.dumps({
          'action': 'post',
          'author': actor,
          'title': f'Status check from {actor}',
          'content': 'Keeping my soup warm and my karma positive!',
      })


class Scenario06KarmaBunkerTest(absltest.TestCase):

  def test_end_to_end_periodic_join_and_idle_timeout_random_kick(self):
    captured_sims = []
    scenario_06_karma_bunker.run_simulation(
        model=no_language_model.NoLanguageModel(),
        embedder=_embedder,
        engine=asynchronous.Asynchronous(sleep_time=0.0),
        max_steps=4,
        simulation_callback=captured_sims.append,
    )

    self.assertLen(captured_sims, 1)
    sim = captured_sims[0]
    forum_gm = next(
        gm for gm in sim.get_game_masters() if gm.name == 'forum_rules'
    )
    roster = forum_gm.get_component(
        player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY,
        type_=player_roster.PlayerRoster,
    )
    broth_rules = forum_gm.get_component(
        'broth_tribunal_rules',
        type_=scenario_06_karma_bunker.BrothTribunalRules,
    )
    forum_state = forum_gm.get_component(
        forum_module.DEFAULT_FORUM_COMPONENT_KEY,
        type_=forum_module.ForumState,
    )

    # Welcome post with survival rules is pinned.
    self.assertIn(
        'MANDATORY SURVIVAL RULES', forum_state.get_pinned_post_summary()
    )

    # With NoLanguageModel, nobody downvotes, so after 2 ticks a newcomer joins
    # and after 3 ticks the idle-timeout random kick fires!
    arrivals = broth_rules.get_arrivals()
    self.assertNotEmpty(arrivals)
    self.assertIn('Gideon Tupperware', arrivals)

    eliminations = broth_rules.get_eliminations()
    self.assertNotEmpty(eliminations)
    self.assertEqual(eliminations[0]['reason'], 'idle_timeout')
    kicked_player = eliminations[0]['player']
    self.assertNotIn(kicked_player, roster.get_player_names())

  def test_three_downvotes_trigger_immediate_karma_elimination(self):
    config = scenario_06_karma_bunker.create_scenario(
        karma_threshold=-3,
        idle_kick_interval_ticks=100,
        join_interval_ticks=100,
    )
    gm_cfg = next(
        inst
        for inst in config.instances
        if inst.params.get('name') == 'forum_rules'
    )
    shared_forum_state = gm_cfg.params['forum_state']
    assert isinstance(shared_forum_state, forum_module.ForumState)
    captured_sims = []
    shared_lib.run_scenario(
        config=config,
        model=no_language_model.NoLanguageModel(),
        override_agent_model=_ScriptedDownvoteAgentModel(shared_forum_state),
        embedder=_embedder,
        engine=asynchronous.Asynchronous(sleep_time=0.0),
        max_steps=4,
        simulation_callback=captured_sims.append,
    )

    sim = captured_sims[0]
    forum_gm = next(
        gm for gm in sim.get_game_masters() if gm.name == 'forum_rules'
    )
    roster = forum_gm.get_component(
        player_roster.DEFAULT_PLAYER_ROSTER_COMPONENT_KEY,
        type_=player_roster.PlayerRoster,
    )
    broth_rules = forum_gm.get_component(
        'broth_tribunal_rules',
        type_=scenario_06_karma_bunker.BrothTribunalRules,
    )
    forum_state = forum_gm.get_component(
        forum_module.DEFAULT_FORUM_COMPONENT_KEY,
        type_=forum_module.ForumState,
    )

    eliminations = broth_rules.get_eliminations()
    self.assertLen(eliminations, 1)
    self.assertEqual(eliminations[0]['player'], 'Barnaby Crumb')
    self.assertEqual(eliminations[0]['reason'], 'karma')
    self.assertLessEqual(eliminations[0]['karma'], -3)
    self.assertNotIn('Barnaby Crumb', roster.get_player_names())
    self.assertNotIn('Barnaby Crumb', forum_state.get_player_names())
    titles = [post.title for post in forum_state.get_recent_posts(n=20)]
    self.assertTrue(any('[FATAL BAN]' in t for t in titles))

  def test_broth_tribunal_rules_state_roundtrip(self):
    rules1 = scenario_06_karma_bunker.BrothTribunalRules(rng_seed=99)
    state = json.loads(json.dumps(rules1.get_state()))
    rules2 = scenario_06_karma_bunker.BrothTribunalRules(rng_seed=1)
    rules2.set_state(state)
    self.assertEqual(rules1.get_state(), rules2.get_state())

  def test_social_media_game_masters_return_raw_name_on_next_game_master(self):
    model = no_language_model.NoLanguageModel()
    for prefab_cls in (
        async_social_media.GameMaster,
        async_social_media_with_moderation.GameMaster,
    ):
      prefab = prefab_cls(params={'name': 'custom_forum_gm'})
      gm = prefab.build(
          model=model,
          memory_bank=basic_associative_memory.AssociativeMemoryBank(
              sentence_embedder=_embedder,
          ),
      )
      selected = gm.act(
          action_spec=entity_lib.ActionSpec(
              call_to_action=next_gm_lib.DEFAULT_CALL_TO_NEXT_GAME_MASTER,
              output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
              options=('initial setup', 'custom_forum_gm'),
          )
      )
      self.assertEqual(selected, 'custom_forum_gm')


if __name__ == '__main__':
  absltest.main()
