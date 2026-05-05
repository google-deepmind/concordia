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

"""A prefab game master for interrupt-driven simulations.

This game master orchestrates entities interacting in a shared space using
interrupt masks and timers. Entities control their attention by specifying
which event tags they respond to (via prefix-based interrupt masks) and when
they next expect to become fully attentive (via a mandatory timer). The
simulation advances in simulated time, skipping dead air when all entities
are focused on tasks.

Key properties:
- Each entity always has exactly one pending timer.
- Polling an entity (for any reason) clears its mask AND timer.
- Every entity response must provide a new mask and a new timer.
- Timer events are private (only poll the owning entity).
- Non-timer events poll all entities whose mask matches.

Designed for use with the Simultaneous engine.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import interrupt_make_observation as interrupt_make_observation_component
from concordia.components.game_master import interrupt_next_acting as interrupt_next_acting_component
from concordia.components.game_master import interrupt_resolution as interrupt_resolution_component
from concordia.components.game_master import interrupt_scheduling
from concordia.components.game_master import interrupt_time_model
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.components.game_master import terminate as terminate_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib


PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

# Default call to action for entities. Instructs them to respond with action
# text followed by a JSON block specifying attention mask, timer, and tags.
DEFAULT_CALL_TO_ACTION = (
    'What does {name} do next? Describe their action, then end your'
    ' response with a JSON object specifying attention and schedule.\n\n'
    'JSON schema:\n'
    '{"tags": [...], "mask": [...], "timer": {"time": "...", "reason": "..."}}'
    '\n\n'
    'Fields:\n'
    '  mask   - list of event-tag prefixes to monitor.\n'
    '           [""]  = pay attention to everything\n'
    '           []    = pay attention to nothing until timer fires\n'
    '           ["chat.", "alarm."]  = only matching prefixes\n'
    '  timer  - when to next check in.\n'
    '           time: duration like "0m", "5m", "30m", "1h", "1h30m"\n'
    '           until: absolute time like "8:00", "14:30" (24-hour clock)\n'
    '           reason: human-readable explanation\n'
    '           Use "time" for a delay from now, or "until" for a specific\n'
    '           clock time. If both are given, "until" takes priority.\n'
    '  tags   - (optional) extra event tags others can match on.\n\n'
    'Examples:\n'
    'I wave hello to everyone.\n'
    '{"mask": ["chat."], "timer": {"time": "30m", "reason": "check messages"}}'
    '\n\n'
    'I focus on my work in silence.\n'
    '{"mask": [], "timer": {"time": "2h", "reason": "deep focus"},'
    ' "tags": ["status.busy"]}\n\n'
    'I study until the library closes.\n'
    '{"mask": ["study."], "timer": {"until": "21:00",'
    ' "reason": "library closes"}}\n'
)


# ──────────────────────────────────────────────────────────────────────────────
# Game Master Prefab
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """A prefab game master for interrupt-driven simulations.

  Entities interact in a shared space, controlling their attention via
  prefix-based interrupt masks and mandatory timers. Time advances in
  simulated jumps, skipping dead air when all entities are focused.

  This game master is designed for use with the Simultaneous engine.

  Params:
    name: Name of the game master entity.
    start_time: Initial simulated time as an ISO format string.
    event_tag_for_actions: Default tag prefix for entity action events.
    max_ticks: Optional maximum number of event processing ticks.
    call_to_action: The prompt given to entities when they act.
    initial_events: Optional list of dicts with keys 'timestamp' (ISO),
      'tag', 'source', 'description' to inject at startup.
    initial_entity_states: Optional dict mapping entity names to their
      starting configuration.  Each value is a dict with optional keys:
        - 'mask': A list of prefix strings (e.g. [''] for match-all,
          ['emergency.'] for one prefix, [] for match-none).  Default:
          [''] (match all).
        - 'timer_duration': A duration string (e.g. ``'30m'``, ``'1h'``)
          for the entity's initial timer.  Default: fires immediately
          at start_time.
        - 'timer_description': Human-readable label.  Default:
          'initial timer'.
      When provided, these replace the default initial timer and mask
      for the named entities, letting scenarios start in a known state
      without a bootstrap step.
    extra_components: Additional components to include.
    extra_components_index: Indices for extra components in the order.
  """

  description: str = 'An interrupt-driven game master with selective attention.'
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'interrupt_driven_rules',
          'start_time': '2026-01-01T09:00:00',
          'event_tag_for_actions': 'action',
          'max_ticks': None,
          'call_to_action': DEFAULT_CALL_TO_ACTION,
          'initial_events': [],
          'initial_entity_states': {},
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
    """Build an interrupt-driven game master.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use.

    Returns:
      A game master entity.
    """
    name = self.params.get('name', 'interrupt_driven_rules')
    start_time_str = self.params.get('start_time', '2026-01-01T09:00:00')
    event_tag_for_actions = self.params.get('event_tag_for_actions', 'action')
    max_ticks = self.params.get('max_ticks', None)
    call_to_action = self.params.get('call_to_action', DEFAULT_CALL_TO_ACTION)
    initial_events = self.params.get('initial_events', [])
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})

    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as'
            ' extra_components.'
        )

    # Resolve the time model.
    time_model = self.params.get('time_model', None)
    if time_model is None:
      start_time = datetime.datetime.fromisoformat(start_time_str)
      time_model = interrupt_time_model.DatetimeTimeModel(start_time)

    player_names = [entity.name for entity in self.entities]

    # ── Standard components ──────────────────────────────────────────────

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

    # ── Interrupt-driven components ──────────────────────────────────────

    scheduler_key = interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY
    scheduler = interrupt_scheduling.EntityScheduler(
        player_names=player_names,
        time_model=time_model,
        max_ticks=max_ticks,
    )

    # Inject initial events.
    for event_dict in initial_events:
      event = interrupt_scheduling.Event(
          timestamp=time_model.deserialize_time(event_dict['timestamp']),
          tag=event_dict['tag'],
          source=event_dict.get('source', '__external__'),
          description=event_dict['description'],
      )
      scheduler.inject_event(event)

    # Pre-configure entity states (masks and timers).
    initial_entity_states = self.params.get('initial_entity_states', {})
    for entity_name, state in initial_entity_states.items():
      mask_prefixes = state.get('mask', [''])
      scheduler.set_mask(
          entity_name, interrupt_scheduling.mask_from_prefixes(mask_prefixes)
      )
      duration_str = state.get('timer_duration', None)
      description = state.get('timer_description', 'initial timer')
      if duration_str is not None:
        expiry = time_model.add_duration(
            time_model.initial_time(), duration_str
        )
      else:
        expiry = time_model.initial_time()
      scheduler.set_timer(
          entity_name,
          interrupt_scheduling.Timer(
              expiry=expiry,
              entity_name=entity_name,
              description=description,
          ),
      )

    next_acting_key = next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_acting = interrupt_next_acting_component.InterruptNextActing(
        scheduler_component_key=scheduler_key,
    )

    next_action_spec_key = (
        next_acting_components.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    )
    next_action_spec = next_acting_components.FixedActionSpec(
        action_spec=entity_lib.free_action_spec(
            call_to_action=call_to_action,
        ),
    )

    # Register under the standard key so SwitchAct finds it.
    make_observation_key = '__make_observation__'
    make_observation = (
        interrupt_make_observation_component.InterruptMakeObservation(
            scheduler_component_key=scheduler_key,
        )
    )

    resolution_key = (
        event_resolution_components.DEFAULT_RESOLUTION_COMPONENT_KEY
    )
    resolution = interrupt_resolution_component.InterruptResolution(
        player_names=player_names,
        event_tag_for_actions=event_tag_for_actions,
        scheduler_component_key=scheduler_key,
    )

    terminate_key = terminate_components.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate = terminate_components.NeverTerminate()

    # ── Assemble ──────────────────────────────────────────────────────────

    components_of_game_master = {
        observation_component_key: observation,
        observation_to_memory_key: observation_to_memory,
        memory_component_key: memory_component,
        scheduler_key: scheduler,
        resolution_key: resolution,
        make_observation_key: make_observation,
        next_acting_key: next_acting,
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

    act_component = switch_act_component.SwitchAct(
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
