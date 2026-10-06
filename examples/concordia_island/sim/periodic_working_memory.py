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

"""Periodically-updated WorkingMemory wrapper.

Wraps the WorkingMemory component to only run the expensive LLM narrative
update every N ticks (default 4 = twice per simulated day with 8 ticks/day),
returning the cached narrative on intermediate ticks.

The tick counter increments each time _make_pre_act_value is called, which
corresponds to each agent act cycle. This is simpler and more reliable than
timestamp parsing for tick-based scheduling.
"""

from collections.abc import Sequence
import threading

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component

from examples.concordia_island.sim import component_state

DEFAULT_PRE_ACT_LABEL = 'WorkingMemory'


class PeriodicWorkingMemory(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """WorkingMemory that only updates its LLM narrative every N ticks.

  Maintains the same stateful narrative as WorkingMemory, but avoids the
  expensive LLM call on most ticks. On non-update ticks, returns the
  previously cached narrative.

  The component still tracks post_act actions on every tick to maintain
  awareness of what the agent just did, so the next LLM update has full
  context.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      components: Sequence[str] = (),
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
      num_memories_to_retrieve: int = 25,
      update_interval_ticks: int = 4,
  ):
    """Initialize the periodic working memory component.

    Args:
      model: A language model for generating/updating the narrative.
      memory_component_key: Key for the memory component to retrieve memories.
      components: Keys of other components to condition the answer on.
      pre_act_label: Prefix for the component's output.
      num_memories_to_retrieve: Number of recent memories for updates.
      update_interval_ticks: How often to run the LLM update (default 4 = twice
        per simulated day with 8 ticks/day). Set to 1 to update every tick
        (original behavior).
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = components
    self._num_memories = num_memories_to_retrieve
    self._update_interval = update_interval_ticks

    # Deliberately an RLock, replacing the plain Lock the base class installs.
    # Reentrancy is needed on two counts: ActionSpecIgnored.get_pre_act_value
    # holds self._lock while calling _make_pre_act_value, and
    # _make_pre_act_value then re-acquires it again itself. A non-reentrant
    # Lock self-deadlocks here. The base's annotation is what is wrong, not
    # this line.
    self._lock = threading.RLock()  # pyrefly: ignore[bad-assignment]
    self._working_memory_narrative: str = ''
    self._tick_counter: int = 0

  def _should_update(self) -> bool:
    """Returns True if this tick should trigger an LLM update."""
    with self._lock:
      # Always update on the first call (tick 0) so we have a baseline.
      if not self._working_memory_narrative:
        return True
      return self._tick_counter % self._update_interval == 0

  def _run_llm_update(self) -> str:
    """Run the full working memory LLM narrative update."""
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    mems = '\n'.join(memory.retrieve_recent(limit=self._num_memories))

    prompt = interactive_document.InteractiveDocument(self._model)

    # Add context from dependent components.
    for key in self._components:
      component = self.get_entity().get_component(
          key, type_=action_spec_ignored.ActionSpecIgnored
      )
      value = self.get_named_component_pre_act_value(key)
      prompt.statement(f'{component.get_pre_act_label()}: {value}')

    prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    # Include previous working memory for continuity.
    with self._lock:
      prev = self._working_memory_narrative
    if prev:
      prompt.statement(f'Previous working memory of {agent_name}:\n{prev}')

    question = (
        f'You are {agent_name}. Update your working memory - a running'
        ' narrative that helps you understand your situation and make'
        ' decisions.\n\n'
        '1. **The Story So Far** (2-3 paragraphs): What has been happening?'
        ' Key events, people involved, narrative arc.\n'
        '2. **Theories About Others**: What do you believe about other'
        " people's goals, intentions, and mental states?\n"
        '3. **Your Goals and Desires**: What do you want? Current priorities'
        ' and motivations.\n'
        "4. **Key Uncertainties**: What don't you know that you'd like to"
        ' find out?\n\n'
        f'Write in first person as {agent_name}. Be specific and grounded'
        ' in memories. Keep it concise (400-600 words).'
    )

    result = prompt.open_question(
        question,
        answer_prefix=f"{agent_name}'s working memory: ",
        max_tokens=1500,
        terminators=(),
    )
    result = f"{agent_name}'s working memory: " + result

    with self._lock:
      self._working_memory_narrative = result

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': 'Working memory update (LLM refresh)',
        'State': result,
        'tick': self._tick_counter,
        'cached': False,
    })

    return result

  def _make_pre_act_value(self) -> str:
    """Return working memory, updating via LLM only every N ticks."""
    with self._lock:
      tick = self._tick_counter
      self._tick_counter += 1

    if self._should_update():
      logging.info(
          'PeriodicWorkingMemory: LLM update at tick %d (interval=%d)',
          tick,
          self._update_interval,
      )
      return self._run_llm_update()
    else:
      with self._lock:
        cached = self._working_memory_narrative
      self._logging_channel({
          'Key': self.get_pre_act_label(),
          'Summary': 'Working memory (cached)',
          'State': cached,
          'tick': tick,
          'cached': True,
      })
      return cached

  def get_narrative(self) -> str:
    """Return the current working memory narrative."""
    with self._lock:
      return self._working_memory_narrative

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          'working_memory_narrative': self._working_memory_narrative,
          'tick_counter': self._tick_counter,
          'update_interval': self._update_interval,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._working_memory_narrative = component_state.as_str(
          state, 'working_memory_narrative'
      )
      self._tick_counter = component_state.as_int(state, 'tick_counter')
      if 'update_interval' in state:
        self._update_interval = component_state.as_int(
            state, 'update_interval', self._update_interval
        )
