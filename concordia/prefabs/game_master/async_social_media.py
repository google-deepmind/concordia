# Copyright 2025 DeepMind Technologies Limited.
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

"""A prefab game master for asynchronous social media simulations.

This game master is designed for simulations where entities interact through
digital platforms (forums, social media, etc.) using the asynchronous engine.
It includes a thread-safe digital technology component (e.g. a Forum) and
returns all players from next_acting, treating it as a participation filter
rather than a turn-ordering mechanism.
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
from concordia.contrib.components.game_master import forum as forum_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


DEFAULT_CALL_TO_ACTION = (
    'What does {name} do on the forum? Respond in JSON format with one of:\n'
    '{{"action": "post", "author": "{name}", "title": "...", '
    '"content": "..."}}\n'
    '{{"action": "reply", "author": "{name}", "post_id": "...", '
    '"content": "..."}}\n'
    '{{"action": "upvote", "author": "{name}", "post_id": "..."}}\n'
    '{{"action": "downvote", "author": "{name}", "post_id": "..."}}\n'
)


class _NextActingEligiblePlayers(
    entity_component.ContextComponent,
):
  """A next_acting component that always returns eligible players.

  In async social media simulations, all players are always eligible
  to act unless banned. This component acts as a participation filter —
  returning all players by default. Players can be removed (e.g. banned)
  or re-added dynamically.
  """

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
    self._lock = threading.Lock()

  def remove_player(self, player_name: str) -> None:
    """Remove a player from the active set (e.g. ban from platform)."""
    with self._lock:
      if player_name in self._player_names:
        self._player_names.remove(player_name)

  def add_player(self, player_name: str) -> None:
    """Add a player back to the active set (e.g. unban)."""
    with self._lock:
      if player_name not in self._player_names:
        self._player_names.append(player_name)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
      with self._lock:
        return ', '.join(self._player_names)
    return ''

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {'player_names': list(self._player_names)}

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._player_names = list(state['player_names'])


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab game master for asynchronous social media simulations.

  This game master is designed for simulations where entities interact through
  digital platforms (forums, social media, etc.). It includes:

  - A thread-safe social media component (e.g. a Forum) for managing platform
    state (posts, replies, votes).
  - A next_acting component that returns all players, treating it as a
    participation filter (e.g. for banning) rather than turn ordering.
  - A JSON-formatted action spec for structured platform interactions.
  - NeverTerminate — relies on max_steps for simulation termination.
  """

  description: str = 'A game master for asynchronous social media simulations.'
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'forum_rules',
          'forum_name': 'Community Forum',
          'call_to_action': DEFAULT_CALL_TO_ACTION,
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
    """Build a game master for social media simulations.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', 'forum_rules')
    forum_name = self.params.get('forum_name', 'Community Forum')
    call_to_action = self.params.get('call_to_action', DEFAULT_CALL_TO_ACTION)
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

    forum_key = forum_module.DEFAULT_FORUM_COMPONENT_KEY
    forum_state = forum_module.ForumState(
        player_names=player_names,
        forum_name=forum_name,
    )

    resolution_key = (
        event_resolution_components.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    resolution = forum_module.ForumResolution(
        player_names=player_names,
    )

    make_observation_key = (
        make_observation_components.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = forum_module.ForumObservation()

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
    }

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
    )

    return game_master
