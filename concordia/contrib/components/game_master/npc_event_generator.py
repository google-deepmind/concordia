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

"""A generative component to create random NPC events for the Game Master."""

import random
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.language_model import language_model


class NpcEventGenerator(action_spec_ignored.ActionSpecIgnored):
  """A generative component that creates random NPC events using an LLM."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: Any,  # concordia.typing.GameClock
      scenario_context: str,
      event_probability: float = 0.15,
      pre_act_label: str = 'Ambient Event',
      verbose: bool = False,
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._clock = clock
    self._scenario_context = scenario_context
    self._event_probability = event_probability
    self._verbose = verbose

  def _make_pre_act_value(self) -> str:
    """Potentially returns a random event string."""
    if random.random() > self._event_probability:
      return ''  # Return empty string if no event

    current_time = self._clock._game_clock.now()  # pylint: disable=protected-access
    time_str = current_time.strftime('%A, %B %d, %Y at %I:%M %p')
    # Format for memory timestamp (e.g., "March 3, 2026, 8:45 AM")
    memory_timestamp = current_time.strftime('%B %d, %Y, %I:%M %p')

    prompt = (
        f'It is {time_str} in scenario: {self._scenario_context}.\nGenerate a'
        ' single sentence describing a brief, realistic, minor ambient'
        ' interruption or event suitable for this setting (e.g., a customer'
        ' question, phone call, environmental noise, unimportant NPC arrival).'
        ' This event should not be dramatic or plot-changing. Example:'
        " 'A phone rings behind the counter.' or 'A customer"
        " asks where the restroom is.'\nJust write the event description,"
        " do not include any timestamp or prefix."
    )
    try:
      event = (
          self._model.sample_text(prompt, max_tokens=100)
          .splitlines()[0]
          .strip()
      )
    except IndexError:
      event = ''  # Handle case where LLM returns empty string

    if not event:
      return ''

    # Clean up any accidental prefixes the LLM might have added
    if event.startswith('[EVENT]'):
      event = event[7:].strip()
    if not event.endswith('.'):
      event = f'{event}.'

    # Format with timestamp so agents can situate this in time
    formatted_event = f'[{memory_timestamp}] [EVENT] {event}'

    if self._verbose:
      print(f'NpcEventGenerator: triggered event: {formatted_event}')
    return formatted_event
