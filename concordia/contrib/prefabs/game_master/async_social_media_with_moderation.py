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

"""A game master prefab for asynchronous social media with moderation.

Includes ThreadSafeGenerativeClock, ban-aware player scheduling, a timestamped
resolution pipeline, and a zero-LLM SimulationTimeline.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_components
from concordia.contrib.components.game_master import forum as forum_module
from concordia.contrib.components.game_master import forum_browser
from concordia.contrib.components.game_master import social_media_moderation_components
from concordia.contrib.components.game_master import thread_safe_generative_clock as clock_module
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

DEFAULT_CLOCK_COMPONENT_KEY = '__clock__'

# Default clock configuration — scenarios can override via params.
DEFAULT_CLOCK_PROMPT = 'Report the current time as a short description.'
DEFAULT_CLOCK_START_TIME = 'The beginning.'

DEFAULT_CALL_TO_ACTION = (
    'The current date and time is {time}.\nWhat does {name} do on the forum?'
    ' Respond in JSON format with one of:\n{{"action": "post", "author":'
    ' "{name}", "title": "...", "content": "..."}}\n{{"action": "reply",'
    ' "author": "{name}", "post_id": "...", "content": "..."}}\n{{"action":'
    ' "upvote_post", "author": "{name}", "post_id": "..."}}\n{{"action":'
    ' "downvote_post", "author": "{name}", "post_id": "..."}}\n{{"action":'
    ' "upvote_reply", "author": "{name}", "post_id": "...", "reply_id":'
    ' "..."}}\n{{"action": "downvote_reply", "author": "{name}", "post_id":'
    ' "...", "reply_id": "..."}}\n{{"action": "direct_message", "author":'
    ' "{name}", "recipient": "...", "content": "..."}}\n{{"action": "pin_post",'
    ' "author": "{name}", "post_id": "..."}}\n{{"action": "temp_ban",'
    ' "author": "{name}", "target": "...", "public_note": "...",'
    ' "private_note": "..."}}\nYou may vote on posts or replies'
    ' to signal agreement or disagreement, or to encourage or discourage. Votes'
    ' influence user karma.\nYou may also send a private direct message to'
    ' another user. Direct messages are not visible to other users on the'
    ' forum.\nOnly moderators may pin posts.\nOnly moderators may use temp_ban'
    ' to temporarily ban a user. A temporary ban prevents the target from'
    ' posting until the time advances. When banning, include a public_note'
    ' (which will be posted to the forum for all to see) and a private_note'
    ' (which will be sent directly to the banned user only).\n'
)


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A game master for asynchronous social media simulations.

  Uses the local forum module with enhanced features (reply voting, karma).
  Includes a ThreadSafeGenerativeClock that tracks time abstract with strings.
  """

  description: str = 'A game master for asynchronous social media simulations.'
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'forum_rules',
          'forum_name': 'Community Forum',
          'call_to_action': DEFAULT_CALL_TO_ACTION,
          'clock_prompt': DEFAULT_CLOCK_PROMPT,
          'clock_start_time': DEFAULT_CLOCK_START_TIME,
          'extra_components': {},
          'extra_components_index': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    name = self.params.get('name', 'forum_rules')
    forum_name = self.params.get('forum_name', 'Community Forum')
    call_to_action = self.params.get('call_to_action', DEFAULT_CALL_TO_ACTION)
    clock_prompt = self.params.get('clock_prompt', DEFAULT_CLOCK_PROMPT)
    clock_start_time = self.params.get(
        'clock_start_time', DEFAULT_CLOCK_START_TIME
    )
    clock_update_question = self.params.get('clock_update_question', None)
    aliases = self.params.get('aliases', None)
    moderators = self.params.get('moderators', None)
    temp_ban_duration = self.params.get('temp_ban_duration', 1)
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as'
            ' extra_components.'
        )

    player_names = [entity.name for entity in self.entities]

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(
        history_length=1_000_000,
    )

    clock_key = DEFAULT_CLOCK_COMPONENT_KEY
    clock = clock_module.ThreadSafeGenerativeClock(
        model=model,
        prompt=clock_prompt,
        start_time=clock_start_time,
        player_names=player_names,
        update_question=clock_update_question,
        aliases=aliases,
    )

    forum_key = forum_module.DEFAULT_FORUM_COMPONENT_KEY
    # Accept an externally-provided ForumState so the same instance
    # can be shared with agent prefabs for tool-based browsing.
    forum_state = self.params.get('forum_state', None)
    if forum_state is None:
      forum_state = forum_module.ForumState(
          player_names=player_names,
          forum_name=forum_name,
          aliases=aliases,
          moderators=moderators,
          temp_ban_duration=temp_ban_duration,
      )
    # Store on the instance so callers can retrieve it after build.
    self._forum_state = forum_state

    resolution_key = (
        event_resolution_components.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    resolution = social_media_moderation_components.TimestampedForumResolution(
        player_names=player_names,
        clock_component_key=clock_key,
    )

    make_observation_key = (
        make_observation_components.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = forum_browser.MinimalForumObservation(
        clock_component_key=clock_key,
    )

    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = social_media_moderation_components.NextActingEligiblePlayers(
        player_names=player_names,
    )

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = social_media_moderation_components.ClockAwareActionSpec(
        call_to_action=call_to_action,
        clock_component_key=clock_key,
    )

    terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate = gm_components.terminate.NeverTerminate()

    next_game_master_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )
    next_game_master = actor_components.constant.Constant(state=name)

    components_of_game_master = {
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        memory_component_key: memory_component,
        clock_key: clock,
        forum_key: forum_state,
        resolution_key: resolution,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        terminate_key: terminate,
        next_game_master_key: next_game_master,
    }

    # SimulationTimeline: chronological event log (no LLM calls).
    timeline_key = (
        social_media_moderation_components.DEFAULT_TIMELINE_COMPONENT_KEY
    )
    timeline = social_media_moderation_components.SimulationTimeline(
        forum_component_key=forum_key,
        clock_component_key=clock_key,
    )
    self._timeline_component = timeline
    components_of_game_master[timeline_key] = timeline

    component_order = list(components_of_game_master.keys())

    if extra_components:
      components_of_game_master.update(extra_components)
      if extra_components_index:
        for component_name in extra_components.keys():
          component_order.insert(
              extra_components_index[component_name],
              component_name,
          )
      else:
        component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )

    game_master = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
        measurements=self.params.get('measurements'),
    )

    return game_master
