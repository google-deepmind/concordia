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

"""A prefab game master for a simulation of an X-like social platform.

Models an X-like (formerly Twitter-like) social media and dating platform.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import threading
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.utils import async_measurements

from examples.concordia_island.sim import fixed_clock
from examples.concordia_island.sim import internet_forum
from examples.concordia_island.sim import social_scheduler as social_scheduler_lib
from examples.concordia_island.sim import step_count_next_gm

DATING_CALL_TO_ACTION = (
    'What does {name} do on X? Respond in JSON format with one'
    ' of:\n{{"action": "create_profile", "author": "{name}",'
    ' "profile": {{"bio": "...", "interests": "...", "looking_for":'
    ' "..."}}}}\n{{"action": "select_partner", "author": "{name}", "target":'
    ' "Agent Name"}}\n{{"action": "swipe", "author": "{name}", "decisions":'
    ' {{"Agent Name": "Yes", "Other Agent": "No"}}}}\n{{"action":'
    ' "direct_message", "author": "{name}", "recipient": "...", "content":'
    ' "..."}}\n'
)

SOCIAL_CALL_TO_ACTION = (
    'What does {name} do on X? Respond in JSON format with one'
    ' of:\n{{"action": "post", "author": "{name}", "title": "...", "content":'
    ' "..."}}\n{{"action": "reply", "author": "{name}", "post_id": "...",'
    ' "content": "..."}}\n{{"action": "upvote_post", "author": "{name}",'
    ' "post_id": "..."}}\n{{"action": "downvote_post", "author": "{name}",'
    ' "post_id": "..."}}\n{{"action": "direct_message", "author": "{name}",'
    ' "recipient": "...", "content": "..."}}\n'
)

COMBINED_CALL_TO_ACTION = (
    'What does {name} do on X? Respond in JSON format with one'
    ' of:\n{{"action": "post", "author": "{name}", "title": "...", "content":'
    ' "..."}}\n{{"action": "reply", "author": "{name}", "post_id": "...",'
    ' "content": "..."}}\n{{"action": "upvote_post", "author": "{name}",'
    ' "post_id": "..."}}\n{{"action": "downvote_post", "author": "{name}",'
    ' "post_id": "..."}}\n{{"action": "create_profile", "author": "{name}",'
    ' "profile": {{"bio": "...", "interests": "...", "looking_for":'
    ' "..."}}}}\n{{"action": "select_partner", "author": "{name}", "target":'
    ' "Agent Name"}}\n{{"action": "swipe", "author": "{name}", "decisions":'
    ' {{"Agent Name": "Yes", "Other Agent": "No"}}}}\n{{"action":'
    ' "direct_message", "author": "{name}", "recipient": "...", "content":'
    ' "..."}}\n'
)

DEFAULT_CALL_TO_ACTION = COMBINED_CALL_TO_ACTION

_MODE_TO_CALL_TO_ACTION = {
    'dating': DATING_CALL_TO_ACTION,
    'social': SOCIAL_CALL_TO_ACTION,
    'combined': COMBINED_CALL_TO_ACTION,
}


class _NextActingEligiblePlayers(
    entity_component.ContextComponent,
):
  """A next_acting component that supports both async and sequential engines."""

  def __init__(
      self,
      player_names: Sequence[str] = (),
      pre_act_label: str = (
          gm_components.next_acting.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL
      ),
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._pre_act_label = pre_act_label
    self._rr_idx = 0
    self._lock = threading.Lock()

  def remove_player(self, player_name: str) -> None:
    with self._lock:
      if player_name in self._player_names:
        self._player_names.remove(player_name)

  def add_player(self, player_name: str) -> None:
    with self._lock:
      if player_name not in self._player_names:
        self._player_names.append(player_name)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      with self._lock:
        if not self._player_names:
          return ''
        # In the asynchronous engine, options contains only the single thread's
        # entity name (len == 1). In the sequential engine, options contains all
        # entities and NEXT_ACTING must return a single valid entity name.
        if action_spec.options and len(action_spec.options) == 1:
          candidate = action_spec.options[0]
          return candidate if candidate in self._player_names else ''
        eligible = [
            p for p in self._player_names
            if not action_spec.options or p in action_spec.options
        ]
        if not eligible:
          return ''
        chosen = eligible[self._rr_idx % len(eligible)]
        self._rr_idx += 1
        try:
          gm = self.get_entity()
          thread_id = threading.current_thread().ident
          if hasattr(gm, 'set_capture_key_for_thread'):
            gm.set_capture_key_for_thread(thread_id, chosen)
        except Exception:  # pylint: disable=broad-except
          pass
        return chosen
    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'player_names': list(self._player_names), 'rr_idx': self._rr_idx}

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._player_names = list(state['player_names'])
      self._rr_idx = int(state.get('rr_idx', 0))


@dataclasses.dataclass
class XLikeGameMaster(prefab_lib.Prefab):
  """A prefab game master for X-like social media and dating simulations."""

  description: str = 'A game master for X-like social media simulations.'
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'x_rules',
          'forum_name': 'X',
          'call_to_action': DEFAULT_CALL_TO_ACTION,
          'mode': 'combined',
          'island_gm_name': 'island rules',
          'next_gm_name': None,
          'mark_night_complete_on_clock': True,
          'max_steps': 2,
          'clock_key': 'clock',
          'social_scheduler_key': 'social_scheduler',
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
    name = self.params.get('name', 'x_rules')
    forum_name = self.params.get('forum_name', 'X')
    island_gm_name = self.params.get('island_gm_name', 'island rules')
    next_gm_name = self.params.get('next_gm_name') or island_gm_name
    mark_night_complete_on_clock = self.params.get(
        'mark_night_complete_on_clock',
        next_gm_name == island_gm_name,
    )
    mode = self.params.get('mode', 'combined')
    call_to_action = self.params.get(
        'call_to_action',
        _MODE_TO_CALL_TO_ACTION.get(mode, DEFAULT_CALL_TO_ACTION),
    )
    if (
        call_to_action == DEFAULT_CALL_TO_ACTION
        and mode in _MODE_TO_CALL_TO_ACTION
    ):
      call_to_action = _MODE_TO_CALL_TO_ACTION[mode]

    max_steps = self.params.get('max_steps', 2)
    clock_key = self.params.get('clock_key', 'clock')
    social_scheduler_key = self.params.get(
        'social_scheduler_key', 'social_scheduler'
    )
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

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

    forum_key = internet_forum.DEFAULT_FORUM_COMPONENT_KEY
    forum_state = internet_forum.InternetForumState(
        player_names=player_names,
        forum_name=forum_name,
    )

    resolution_key = (
        event_resolution_components.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    resolution = internet_forum.InternetForumResolution(
        player_names=player_names,
    )

    make_observation_key = (
        make_observation_components.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = internet_forum.InternetForumObservation()

    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = _NextActingEligiblePlayers(player_names=player_names)

    next_action_spec_key = (
        gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = gm_components.next_acting.FixedActionSpec(
        action_spec=entity_lib.free_action_spec(
            call_to_action=call_to_action,
        ),
    )

    terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate = gm_components.terminate.NeverTerminate()

    # Multi-GM handoff: switch to next_gm_name (either 'marketplace_rules' when
    # running X -> Marketplace -> Island consecutively, or 'island rules').
    next_game_master_key = (
        gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
    )
    next_game_master = step_count_next_gm.StepCountNextGM(
        player_names=player_names,
        island_gm_name=next_gm_name,
        instagram_gm_name=name,
        max_steps=max_steps,
        clock_key=clock_key,
        social_scheduler_key=social_scheduler_key,
        mark_night_complete_on_clock=mark_night_complete_on_clock,
    )

    # Social Scheduler (shared state via events list)
    social_events = self.params.get('social_events', [])
    scheduler = social_scheduler_lib.SocialScheduler(
        model=model,
        player_names=player_names,
        events=social_events,
        clock_key=clock_key,
    )

    components_of_game_master = {
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        memory_component_key: memory_component,
        forum_key: forum_state,
        resolution_key: resolution,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        terminate_key: terminate,
        next_game_master_key: next_game_master,
        social_scheduler_key: scheduler,
    }
    clock = self.params.get('clock')
    if clock is not None:
      if isinstance(clock, fixed_clock.FixedIntervalClock):
        components_of_game_master[clock_key] = fixed_clock.ClockProxy(clock)
      else:
        components_of_game_master[clock_key] = clock

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
        measurements=async_measurements.ReactiveMeasurements(),
    )

    return game_master
