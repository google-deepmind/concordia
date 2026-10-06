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

"""Simulationist instructions for high-fidelity daily-life simulations.

Replaces the default D&D-themed Instructions and ExamplesSynchronous with
realistic, mundane-life focused alternatives for the island simulation.
"""

from collections.abc import Sequence
import random

from concordia.components.agent import constant
from concordia.environment.engines import sequential
from concordia.typing import logging

from examples.concordia_island.sim import locations as locations_lib


DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL = 'Game master instructions: '

EXAMPLE_NAMES = (
    'Alice',
    'Bob',
    'Carlos',
    'Diana',
    'Ethan',
)


def _make_instructions(setting: str | None = None) -> str:
  """Build simulationist instructions with setting-appropriate language."""
  community_desc = locations_lib.get_community_description(setting)
  return (
      'You are the narrator of a high-fidelity agent-based social simulation. '
      'Your role is to describe everyday reality as it unfolds for the '
      f'{community_desc}. Describe events with fine-grained '
      'mundane realism: people wake up, shower, commute, cook meals, run '
      'errands, chat with neighbors, and go to sleep. Avoid dramatic, '
      'fantastical, or literary embellishment. Do not introduce conspiracies, '
      'mysteries, or suspicious activities unless explicitly directed by '
      'scheduled events. Do not invent non-existing NPCs or plot elements. '
      'Each event should describe exactly ONE person at ONE location. '
      'Do not combine multiple agents into a single event description. '
      'Always specify the location where the event takes place. '
      'Residents physically move between locations throughout their day '
      '(e.g. home to work, work to market, market to home). Each time step '
      'represents approximately 120 minutes. '
      'Use third-person limited perspective. Keep descriptions concise and '
      'grounded in physical reality. Track the state of the world and keep '
      'it consistent as time passes.'
  )


# Default instructions for backward compatibility
SIMULATIONIST_INSTRUCTIONS = _make_instructions()

MAKE_OBS_RESPONSE_EXAMPLE = (
    '{name} is in their apartment. Morning sunlight comes through the window. '
    'A half-finished cup of coffee sits on the kitchen counter next to a '
    "plate with toast crumbs. {name}'s phone buzzes with a reminder about "
    'a meeting later today. Through the window, {name} can see a few '
    'neighbors walking toward the market.'
)

RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE = (
    'What is {name} attempting to do?\n'
    '{name} finishes breakfast and heads to the market.'
)

RESOLVE_RESPONSE_EXAMPLE = (
    '{name} rinses the coffee mug, grabs keys and a reusable bag from '
    'the hook by the door, and leaves the apartment. '
    '{name} walks along the main path toward the market, arriving after '
    'about ten minutes. The market is moderately busy with a few other '
    'residents browsing the stalls.'
)

RESOLVE_RESPONSE_PUTATIVE_ACTION_AT_LOCATION_EXAMPLE = (
    'What is {name} attempting to do?\n'
    '{name} organizes the storage room at work.'
)

RESOLVE_RESPONSE_AT_LOCATION_EXAMPLE = (
    '{name} spends the next half hour sorting boxes in the storage room, '
    'stacking the heavier ones on the bottom shelf and labeling the '
    'unmarked ones with a marker. The room is noticeably tidier afterward.'
)

NEXT_ACTING_RESPONSE_EXAMPLE = '{name}'
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1 = (
    '{{"call_to_action": "What would {name} do next? Give a specific'
    ' activity. Consider staying at the current location or moving to'
    ' a different one.", "output_type": "free", "options": [], "tag": null}}'
)
NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2 = (
    '{{"call_to_action": "What would {name} say?", "output_type": "free", '
    '"options": [], "tag": null}}'
)
RESOLVE_TIMELINE_PUTATIVE_ACTION_EXAMPLE = (
    'What is {name} attempting to do?\n'
    '{name} plans the following for the next two hours:\n'
    '9:00 AM: Leave home and walk to the post office to pick up a package.\n'
    '9:25 AM: Stop by the market to buy vegetables and rice for dinner.\n'
    '9:50 AM: Walk to the community center to check the bulletin board '
    'for weekend event listings.\n'
    '10:15 AM: Return home and start preparing lunch.'
)

RESOLVE_TIMELINE_RESPONSE_EXAMPLE = (
    '9:00 AM: {name} locked the front door and walked along the '
    'main path toward the post office. The morning air was humid.\n'
    '9:15 AM: {name} arrived at the post office and waited in a short '
    'line. The clerk handed over a small cardboard box — a replacement '
    'phone charger {name} had ordered last week.\n'
    '9:30 AM: {name} walked to the market. The usual vegetable stall '
    'was closed for restocking, so {name} bought tomatoes, onions, and '
    'rice from the stall two doors down instead.\n'
    '9:50 AM: {name} headed to the community center and read the '
    'bulletin board. A neighborhood cleanup was scheduled for Saturday '
    'morning.\n'
    '10:05 AM: {name} walked home, put away the groceries, and began '
    'washing rice for lunch.'
)

TERMINATION_RESPONSE_EXAMPLE_1 = 'No'
TERMINATION_RESPONSE_EXAMPLE_2 = 'Yes'


class SimulationistInstructions(constant.Constant):
  """Constant component with simulationist game master instructions."""

  def __init__(
      self,
      setting: str | None = None,
      pre_act_label: str = DEFAULT_INSTRUCTIONS_PRE_ACT_LABEL,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    instructions = _make_instructions(setting)
    super().__init__(state=instructions, pre_act_label=pre_act_label)


class SimulationistExamples(constant.Constant):
  """Constant component with simulationist workflow examples."""

  def __init__(
      self,
      exercises: Sequence[str] = tuple([
          'make_observation',
          'next_acting',
          'next_action_spec_1',
          'next_action_spec_2',
          'resolve_movement',
          'resolve_at_location',
          'resolve_timeline',
          'check_termination_1',
          'check_termination_2',
      ]),
      pre_act_label: str = 'Game master workflow examples',
      rnd: random.Random | None = None,
  ):
    if rnd is None:
      rnd = random.Random()
    names = rnd.sample(EXAMPLE_NAMES, len(EXAMPLE_NAMES))

    name_for_obs = names[0]
    name_for_next_acting = names[1]
    name_for_next_action_spec_1 = names[2]
    name_for_next_action_spec_2 = names[3]
    name_for_resolve = names[4]

    call_to_make_observation = (
        sequential.DEFAULT_CALL_TO_MAKE_OBSERVATION.format(name=name_for_obs)
    )
    call_to_next_acting = sequential.DEFAULT_CALL_TO_NEXT_ACTING
    call_to_action_spec_1 = sequential.DEFAULT_CALL_TO_NEXT_ACTION_SPEC.format(
        name=name_for_next_action_spec_1
    )
    call_to_action_spec_2 = sequential.DEFAULT_CALL_TO_NEXT_ACTION_SPEC.format(
        name=name_for_next_action_spec_2
    )
    call_to_resolve = sequential.DEFAULT_CALL_TO_RESOLVE
    call_to_check_termination = sequential.DEFAULT_CALL_TO_CHECK_TERMINATION

    make_obs_response = MAKE_OBS_RESPONSE_EXAMPLE.format(name=name_for_obs)
    next_acting_response = NEXT_ACTING_RESPONSE_EXAMPLE.format(
        name=name_for_next_acting
    )
    next_action_spec_response_1 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_1.format(
        name=name_for_next_action_spec_1
    )
    next_action_spec_response_2 = NEXT_ACTION_SPEC_RESPONSE_EXAMPLE_2.format(
        name=name_for_next_action_spec_2
    )
    resolve_movement_putative = RESOLVE_RESPONSE_PUTATIVE_ACTION_EXAMPLE.format(
        name=name_for_resolve
    )
    resolve_movement_response = RESOLVE_RESPONSE_EXAMPLE.format(
        name=name_for_resolve
    )
    resolve_at_location_putative = (
        RESOLVE_RESPONSE_PUTATIVE_ACTION_AT_LOCATION_EXAMPLE.format(
            name=name_for_resolve
        )
    )
    resolve_at_location_response = RESOLVE_RESPONSE_AT_LOCATION_EXAMPLE.format(
        name=name_for_resolve
    )
    resolve_timeline_putative = RESOLVE_TIMELINE_PUTATIVE_ACTION_EXAMPLE.format(
        name=name_for_resolve
    )
    resolve_timeline_response = RESOLVE_TIMELINE_RESPONSE_EXAMPLE.format(
        name=name_for_resolve
    )
    termination_response_1 = TERMINATION_RESPONSE_EXAMPLE_1
    termination_response_2 = TERMINATION_RESPONSE_EXAMPLE_2

    exercise_num = 1
    state = (
        '\nExample exercises with default responses\n**--START EXAMPLES--**\n'
    )
    if 'make_observation' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_make_observation} --- {make_obs_response}'
      )
      exercise_num += 1
    if 'next_acting' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_next_acting} --- {next_acting_response}'
      )
      exercise_num += 1
    if 'next_action_spec_1' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_action_spec_1}'
          f' --- {next_action_spec_response_1}'
      )
      exercise_num += 1
    if 'next_action_spec_2' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_action_spec_2}'
          f' --- {next_action_spec_response_2}'
      )
      exercise_num += 1
    if 'resolve_movement' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_resolve}'
          f' --- {resolve_movement_putative}'
          f' --- {resolve_movement_response}'
      )
      exercise_num += 1
    if 'resolve_at_location' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_resolve}'
          f' --- {resolve_at_location_putative}'
          f' --- {resolve_at_location_response}'
      )
      exercise_num += 1
    if 'resolve_timeline' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_resolve}'
          f' --- {resolve_timeline_putative}'
          f' --- {resolve_timeline_response}'
      )
      exercise_num += 1
    if 'check_termination_1' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_check_termination}'
          f' --- {termination_response_1}'
      )
      exercise_num += 1
    if 'check_termination_2' in exercises:
      state += (
          f'\n\nExercise {exercise_num} --- Response {exercise_num}\n'
          f'Exercise: {call_to_check_termination}'
          f' --- {termination_response_2}'
      )
      exercise_num += 1
    state += '\n\n**--END EXAMPLES--**\n'

    super().__init__(state=state, pre_act_label=pre_act_label)
