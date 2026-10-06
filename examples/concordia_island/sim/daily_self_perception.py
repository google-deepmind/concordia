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

"""Daily-cached SelfPerception component.

Runs the 'What kind of person is {agent_name}?' LLM call only once per
simulated day (detected via observation timestamps), returning the cached
value on subsequent ticks within the same day. When it does run, it includes
ImportantMemories (formative + journal + recent) as context to ground the
self-perception in the agent's full identity.

This saves ~7/8 of SelfPerception LLM calls with 8 ticks per simulated day,
since self-identity is relatively stable within a single day.
"""

from collections.abc import Sequence
import datetime
import re
import threading

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state

SELF_PERCEPTION_QUESTION = (
    'What kind of person is {agent_name}? Respond using 1-5 sentences.'
)

# Same observation timestamp pattern as ScheduleAwareness.
_OBS_TIME_PATTERN = re.compile(
    r'\[(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+'
    r'(\w+)\s+(\d+)(?:st|nd|rd|th),\s+'
    r'(\d{1,2}):(\d{2})\s*(AM|PM)\]'
)

_MONTH_MAP = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12,
}


class DailySelfPerception(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """SelfPerception that only makes an LLM call once per simulated day.

  On the first tick of each new day, runs a full self-perception query
  with ImportantMemories context (formative, journal, and recent memories)
  plus any additional component dependencies. On subsequent ticks within
  the same day, returns the cached value without an LLM call.

  Day boundaries are detected by parsing timestamps from GM observations
  (the same mechanism used by ScheduleAwareness).
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      num_memories_to_retrieve: int = 10,
      components: Sequence[str] = (),
      observation_component_key: str = '__observation__',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      pre_act_label: str | None = None,
  ):
    """Initializes DailySelfPerception.

    Args:
      model: The language model to use.
      num_memories_to_retrieve: Number of recent memories for the LLM prompt.
      components: Keys of other components to include as context (e.g.
        LifeAlteringEvents). These are queried via
        get_named_component_pre_act_value.
      observation_component_key: Key for the ImportantMemories / observation
        component. Its output is included in the LLM prompt for identity
        grounding when the daily call fires.
      memory_component_key: Key for the memory component.
      pre_act_label: Label prefix for pre_act output.
    """
    if pre_act_label is None:
      pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    super().__init__(pre_act_label)
    self._model = model
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._components = tuple(components)
    self._observation_component_key = observation_component_key
    self._memory_component_key = memory_component_key

    # Deliberately an RLock, replacing the plain Lock the base class installs.
    # ActionSpecIgnored.get_pre_act_value holds self._lock while it calls
    # _make_pre_act_value, and our override of _make_pre_act_value re-acquires
    # self._lock. With a non-reentrant Lock that self-deadlocks every time the
    # component is asked for its pre-act value. The base's type annotation is
    # what is wrong here, not this line.
    self._lock = threading.RLock()  # pyrefly: ignore[bad-assignment]
    self._cached_value: str | None = None
    self._cached_date: datetime.date | None = None
    self._current_dt: datetime.datetime | None = None

  def pre_observe(self, observation: str) -> str:
    """Parse timestamp from GM observation to track the current date."""
    matches = list(_OBS_TIME_PATTERN.finditer(observation))
    if matches:
      match = matches[-1]
      month_name, day_str, hour_str, minute_str, ampm = match.groups()
      month = _MONTH_MAP.get(month_name, 1)
      day = int(day_str)
      hour = int(hour_str)
      minute = int(minute_str)
      if ampm == 'PM' and hour != 12:
        hour += 12
      elif ampm == 'AM' and hour == 12:
        hour = 0
      try:
        dt = datetime.datetime(2026, month, day, hour, minute)
        with self._lock:
          self._current_dt = dt
      except ValueError:
        pass
    return ''

  def _is_new_day(self) -> bool:
    """Check if the current simulated date differs from the cached date."""
    with self._lock:
      if self._cached_date is None:
        return True
      if self._current_dt is None:
        return True
      return self._current_dt.date() != self._cached_date

  def _get_component_pre_act_label(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    return (
        f'  {self._get_component_pre_act_label(key)}: '
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def _run_llm_query(self) -> str:
    """Run the full SelfPerception LLM query with rich context."""
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    prompt = interactive_document.InteractiveDocument(self._model)

    # Include ImportantMemories context (formative + journal + recent).
    observation_context = self.get_named_component_pre_act_value(
        self._observation_component_key
    )
    if observation_context:
      obs_label = self._get_component_pre_act_label(
          self._observation_component_key
      )
      prompt.statement(f'  {obs_label}: {observation_context}')

    # Include additional component dependencies (e.g. LifeAlteringEvents).
    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    if component_states:
      prompt.statement(component_states)

    # Include recent memories.
    if self._num_memories_to_retrieve > 0:
      mems = '\n'.join([
          mem
          for mem in memory.retrieve_recent(
              limit=self._num_memories_to_retrieve
          )
      ])
      prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    question = SELF_PERCEPTION_QUESTION.format(agent_name=agent_name)
    answer_prefix = f'{agent_name} is '
    result = prompt.open_question(
        question,
        answer_prefix=answer_prefix,
        max_tokens=1000,
        terminators=('\n',),
    )
    result = answer_prefix + result

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
        'cached': False,
    }
    self._logging_channel(log)

    return result

  def _make_pre_act_value(self) -> str:
    if self._is_new_day():
      result = self._run_llm_query()
      with self._lock:
        self._cached_value = result
        if self._current_dt is not None:
          self._cached_date = self._current_dt.date()
        else:
          # No timestamp yet — cache with a sentinel date so we don't
          # re-run on every tick until the first observation arrives.
          self._cached_date = datetime.date(2026, 1, 1)
      logging.info(
          'DailySelfPerception: refreshed for date %s', self._cached_date
      )
      return result
    else:
      with self._lock:
        cached = self._cached_value or ''
      self._logging_channel({
          'Key': self.get_pre_act_label(),
          'Summary': 'Returning cached daily self-perception',
          'State': cached,
          'cached': True,
      })
      return cached

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'cached_value': self._cached_value,
          'cached_date': (
              self._cached_date.isoformat() if self._cached_date else None
          ),
          'current_dt': (
              self._current_dt.isoformat() if self._current_dt else None
          ),
          'num_memories_to_retrieve': self._num_memories_to_retrieve,
          'components': list(self._components),
          'observation_component_key': self._observation_component_key,
          'pre_act_label': self.get_pre_act_label(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      if 'cached_value' in state:
        self._cached_value = component_state.as_optional_str(
            state, 'cached_value'
        )
      if 'cached_date' in state:
        date_str = component_state.as_optional_str(state, 'cached_date')
        self._cached_date = (
            datetime.date.fromisoformat(date_str) if date_str else None
        )
      if 'current_dt' in state:
        dt_str = component_state.as_optional_str(state, 'current_dt')
        self._current_dt = (
            datetime.datetime.fromisoformat(dt_str) if dt_str else None
        )
      if 'num_memories_to_retrieve' in state:
        self._num_memories_to_retrieve = component_state.as_int(
            state, 'num_memories_to_retrieve', self._num_memories_to_retrieve
        )
      if 'components' in state:
        self._components = tuple(
            component_state.as_str_list(state, 'components')
        )
      if 'observation_component_key' in state:
        self._observation_component_key = component_state.as_str(
            state, 'observation_component_key'
        )
