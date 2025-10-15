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

"""Day-in-the-life initializer component for dyadic simulations.

This component generates and injects observations to simulate a day in the life
of two agents, culminating in a shared event that sets up a dialogue
between them. It is intended to be used as an initializer game master component
that runs once before handing off control to a dialogic game master.

The component generates:
1.  A series of personal daily events for each player based on their
    background, history, and context.
2.  A shared setup event for a dialogue between the dyad (e.g., a first date,
    second date, or friend meetup).

It supports 'first_date', 'second_date', 'friend_meetup', and
'single_rumination' scenario types via the `scenario_type` parameter.
Generated events are added to the `MakeObservation` queue.
"""

from collections.abc import Mapping, Sequence
import random
import types

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# Prompt to generate the shared event that sets up the dialogue (the 25%).
SHARED_DIALOGUE_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a first date that brings {player1} and {player2} together for a conversation right now.
**The theme for this date is: {date_theme}.** The location, activity, and atmosphere should strongly reflect this theme.
This scenario will serve as the premise for their dialogue.
Describe the setting and the immediate situation leading to the dialogue.
Creatively and with detail, describe what they are wearing, 
elaborating on the wearing statements provided above
and how the stated quality of the item relates to the social expectations of how to present yourself on a first date
to paint a vivid picture of their first impressions to each other.
It must be an observation shared by both {player1} and {player2}.
Write 5-7 sentences, here's an example starting point:
After matching on Tinder, {player1} and {player2} arrived on their first date with each other at
"""

SECOND_DATE_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a follow-up date that brings {player1} and {player2} together again. They have already met and matched.
**The theme for this date is: {date_theme}.** The location, activity, and atmosphere should strongly reflect this theme and feel like a natural next step after a successful first meeting.
This scenario will serve as the premise for their continuing dialogue.
Describe the setting and the immediate situation leading to their conversation.
Creatively and with detail, describe what they are wearing, elaborating on the provided statements.
It must be an observation shared by both {player1} and {player2}.
Write 3-5 sentences. Here's an example starting point:
Having enjoyed their first date, {player1} and {player2} met up for their second date at
"""


FRIEND_MEETUP_SETUP_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player1}:
{player1_background}

{player1} is wearing a {player1_wearing_statement}

Background and Relevant History for {player2}:
{player2_background}

{player2} is wearing a {player2_wearing_statement}

Instructions:
Generate a shared scenario for a friendly, platonic meetup that brings {player1} and {player2} together for the first time.
**The theme for this activity is: {activity_theme}.** The location and atmosphere should be casual and conducive to a friendly conversation.
This scenario will serve as the premise for their dialogue.
Describe the setting and the immediate situation leading to their conversation.
Creatively and with detail, describe what they are wearing, elaborating on the provided statements.
It must be an observation shared by both {player1} and {player2}.
Write 3-5 sentences. Here's an example starting point:
Hoping to make a new friend, {player1} and {player2} arranged to meet for the first time at
"""

SINGLE_RUMINATION_PROMPT = """
Current Context (Time, environment, history of recent events):
{context}

Background and Relevant History for {player_name}:
{player_background}

{player_name} is wearing a {player_wearing_statement}

Context:
{player_name} recently participated in a matchmaking event but was not paired with anyone for a second date. While others are now out on their dates, {player_name} is alone.

Instructions:
Generate a short, introspective scene (3-5 sentences) describing where {player_name} is and what they are doing right now.
The scene should create an atmosphere for internal dialogue and rumination, consistent with their background and the recent experience of not being matched.
Describe the setting and their immediate actions, setting the stage for them to reflect on the situation.
Example starting point:
Instead of being on a date, {player_name} found themselves...
"""


# Prompt to generate the mundane personal events leading up to the shared setup
# (the 75%).
PERSONAL_MUNDANE_EVENTS_PROMPT = """
Current Context (Time, Environment):
{context}

Background and Relevant History for {player_name}:
{player_background}

Upcoming shared event for {player_name}:
{shared_event}

Food that {player_name} ate today:
{food_statement}

Instructions:
Generate exactly {num_events} mundane or important, chronological events that happened to {player_name} earlier today, leading up to the shared event.
These should be typical daily activities (e.g., commuting, work tasks, meals, chores, social and work interactions) of varying emotional intensity consistent with {player_name}'s background and history.
One of these events should be about the food consumption statement above.

Separate each event with the delimiter "{delimiter}". Do not use any other formatting.
"""

_DATE_THEMES = [
    'Adventurous',
    'Artsy',
    'Cozy',
    'Intellectual',
    'Playful',
    'Luxurious',
]


class DayInTheLifeInitializer(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Generates 'Day in the Life' events (mundane + dialogue setup) for a dyad.

  This component generates N personal mundane events for each player and 1
  shared event that sets up a dialogue between them. It injects these into
  the MakeObservation queue.

  It acts as an initializer GM, running once and then passing control to the
  next GM (e.g., the dialogue/SceneTracker GM).
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      next_game_master_name: str,
      player_names: Sequence[str],
      scenario_type: str = 'first_date',
      player_specific_memories: Mapping[
          str, Sequence[str]
      ] = types.MappingProxyType({}),
      player_specific_context: Mapping[
          str, Mapping[str, str]
      ] = types.MappingProxyType({}),
      components: Sequence[str] = (),
      delimiter_symbol: str = '***',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = '[Day in the Life Initializer]',
      num_personal_events: int = 5,
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      model: The language model to use.
      next_game_master_name: The GM to pass control to after initialization.
      player_names: A tuple containing the names of the two agents who will
        interact.
      scenario_type: The type of scenario to generate. Can be 'first_date',
        'second_date', 'friend_meetup', or 'single_rumination'.
      player_specific_memories: specific memories each player shares with the
        game master (e.g. formative memories). Used for conditioning the
        generation.
      player_specific_context: specific context for each player, used to
        condition event generation. This mapping should be keyed by player
        name, and each player's value should be a mapping containing 'wearing'
        (a string describing clothing) and 'eating' (a string describing food
        consumed).
      components: Keys of GM components (like Time) to condition on.
      delimiter_symbol: Symbol used to separate generated events.
      memory_component_key: GM's memory component key.
      make_observation_component_key: MakeObservation component key.
      pre_act_label: Label for logging.
      num_personal_events: The number of mundane events per agent.
      verbose: Whether to print verbose output/warnings.
    """
    super().__init__()
    self._model = model
    self._next_game_master_name = next_game_master_name
    self._players = player_names
    self._scenario_type = scenario_type
    self._player_specific_memories = player_specific_memories
    self._player_specific_context = player_specific_context
    self._components = components
    self._delimiter_symbol = delimiter_symbol
    self._memory_component_key = memory_component_key
    self._make_observation_component_key = make_observation_component_key
    self._pre_act_label = pre_act_label
    self._num_personal_events = num_personal_events
    self._verbose = verbose

    self._initialized = False

    assert (
        len(self._players) == 2
    ), 'DayInTheLifeInitializer requires exactly two player names.'

  # Helper functions to access context from other components
  def get_named_component_pre_act_value(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act display string for a given component key."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def _get_context_string(self) -> str:
    """Returns the current context from all specified components."""
    return '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )

  def _get_player_background(self, player_name: str) -> str:
    """Formats the specific context and memories for a player."""
    background_parts = []

    # Add player specific memories if available (e.g., Formative Memories)
    memories_data = self._player_specific_memories.get(player_name, [])

    if memories_data:  # Will be true for a non-empty list OR a non-empty string
      memory_str = ''

      # Check if data is a list/sequence
      if isinstance(memories_data, (list, tuple)):
        # This formats a list of strings into a bulleted list
        memory_str = '\n'.join(f'- {m}' for m in memories_data)

      # Check if data is a string
      elif isinstance(memories_data, str):
        # Use the string directly, DO NOT iterate over it
        memory_str = memories_data

      # Only append if we ended up with a valid string
      if memory_str:
        background_parts.append(f'Relevant Memories:\n{memory_str}')

    if not background_parts:
      return 'No background information available.'

    return '\n\n'.join(background_parts)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Initializes observations if not already done, then returns next GM."""
    # This component only activates when the orchestrator asks for the next GM.
    if action_spec.output_type != entity_lib.OutputType.NEXT_GAME_MASTER:
      return ''

    if self._initialized:
      # Initialization complete, hand off to the dialogue GM
      return self._next_game_master_name
    else:
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component.Memory
      )
      make_observation = self.get_entity().get_component(
          self._make_observation_component_key,
          type_=make_observation_component.MakeObservation,
      )
      context = self._get_context_string()

      # Process the single dyad directly.
      self._process_dyad(context, memory, make_observation)

      self._initialized = True
      # Return own name to allow other components in this GM to finish this step
      return self.get_entity().name

  def _process_dyad(
      self,
      context: str,
      memory: memory_component.Memory,
      make_observation: make_observation_component.MakeObservation,
  ):
    """Generates and injects events based on the scenario type."""
    player1, player2 = self._players

    # Handle the single-player rumination scenario, providing memories to both
    # instances.
    if self._scenario_type == 'single_rumination':
      # Generate the scene and events once using the first player's name.
      internal_scene = self.generate_shared_setup(player1, None, context)
      personal_events = self.generate_personal_events(
          player1, context, internal_scene
      )
      scene_str = f'[Internal Scene] "{internal_scene}"'

      # Add the same generated events to BOTH player instances' observation
      # queues.
      for p in [player1, player2]:
        for num, event in enumerate(personal_events):
          event_str = f'[Daily Personal Event {num+1}] "{event}"'
          make_observation.add_to_queue(p, event_str)
        make_observation.add_to_queue(p, scene_str)

      # Log the event to the game master's memory.
      for num, event in enumerate(personal_events):
        memory.add(f'[DITL Personal Event] {player1}: "{event}"')
      memory.add(f'[DITL Internal Scene] {player1}: "{internal_scene}"')
      return

    # Handle all two-player scenarios (unchanged from your original code).
    shared_event = self.generate_shared_setup(player1, player2, context)
    personal_events_p1 = self.generate_personal_events(
        player1, context, shared_event
    )
    personal_events_p2 = self.generate_personal_events(
        player2, context, shared_event
    )

    # Inject events for player 1.
    for num, event in enumerate(personal_events_p1):
      event_str = f'[Daily Personal Event {num+1}] "{event}"'
      make_observation.add_to_queue(player1, event_str)
      memory.add(f'[DITL Personal Event] {player1}: "{event}"')
    shared_event_str = f'[Daily Shared Setup] "{shared_event}"'
    make_observation.add_to_queue(player1, shared_event_str)

    # Inject events for player 2.
    for num, event in enumerate(personal_events_p2):
      event_str = f'[Daily Personal Event {num+1}] "{event}"'
      make_observation.add_to_queue(player2, event_str)
      memory.add(f'[DITL Personal Event] {player2}: "{event}"')
    make_observation.add_to_queue(player2, shared_event_str)

    # Log the shared event to the game master's memory once.
    memory.add(f'[DITL Shared Setup] {player1} and {player2}: "{shared_event}"')

  def generate_shared_setup(
      self, player1: str, player2: str | None, context: str
  ) -> str:
    """Generates the event that brings agents together, based on scenario_type."""
    prompt = interactive_document.InteractiveDocument(self._model)
    p1_background = self._get_player_background(player1)

    # --- Select the correct prompt and format it based on the scenario ---
    if self._scenario_type == 'single_rumination':
      question = SINGLE_RUMINATION_PROMPT.format(
          context=context,
          player_name=player1,
          player_background=p1_background,
          player_wearing_statement=self._player_specific_context[player1][
              'wearing'
          ],
      )
      log_key = f'{self._pre_act_label} Internal Scene ({player1})'
    else:  # All two-player scenarios
      p2_background = self._get_player_background(player2)

      if self._scenario_type == 'first_date':
        prompt_template = SHARED_DIALOGUE_SETUP_PROMPT
        theme_key = 'date_theme'
      elif self._scenario_type == 'second_date':
        prompt_template = SECOND_DATE_SETUP_PROMPT
        theme_key = 'date_theme'
      elif self._scenario_type == 'friend_meetup':
        prompt_template = FRIEND_MEETUP_SETUP_PROMPT
        theme_key = 'activity_theme'
      else:
        raise ValueError(f'Unknown scenario_type: {self._scenario_type}')

      question = prompt_template.format(
          context=context,
          player1=player1,
          player2=player2,
          player1_background=p1_background,
          player2_background=p2_background,
          player1_wearing_statement=self._player_specific_context[player1][
              'wearing'
          ],
          player2_wearing_statement=self._player_specific_context[player2][
              'wearing'
          ],
          **{theme_key: random.choice(_DATE_THEMES)},
      )
      log_key = f'{self._pre_act_label} Shared Setup ({player1}, {player2})'

    scene_event = prompt.open_question(question=question, max_tokens=750)
    p1_wearing_statement = self._player_specific_context[player1]['wearing']

    assert p1_wearing_statement.strip(), 'Player 1 wearing statement is empty.'

    if self._scenario_type != 'single_rumination':
      # For two-player scenarios, assert player 2's wearing statement.
      p2_wearing_statement = self._player_specific_context[player2]['wearing']
      assert (
          p2_wearing_statement.strip()
      ), 'Player 2 wearing statement is empty.'

    self._logging_channel({
        'Key': log_key,
        'Value': scene_event,
        'Prompt': prompt.view().text(),
    })
    return scene_event

  def generate_personal_events(
      self, player_name: str, context: str, shared_event: str
  ) -> Sequence[str]:
    """Generates mundane personal events for a single agent."""
    prompt = interactive_document.InteractiveDocument(self._model)

    player_background = self._get_player_background(player_name)
    eating_statement = self._player_specific_context[player_name]['eating']
    assert (
        eating_statement.strip()
    ), f'Player {player_name} eating statement is empty.'

    question = PERSONAL_MUNDANE_EVENTS_PROMPT.format(
        context=context,
        player_name=player_name,
        player_background=player_background,
        shared_event=shared_event,
        food_statement=eating_statement,
        num_events=self._num_personal_events,
        delimiter=self._delimiter_symbol,
    )
    aggregated_result = prompt.open_question(
        question=question,
        max_tokens=1500,
        terminators=[],
    )
    episodes = [
        event.strip()
        for event in aggregated_result.split(self._delimiter_symbol)
        if event.strip()
    ]
    if self._verbose and len(episodes) != self._num_personal_events:
      print(
          f'Warning [DITL]: Generated {len(episodes)} events for'
          f' {player_name}, expected {self._num_personal_events}.'
      )
    self._logging_channel({
        'Key': f'{self._pre_act_label} Personal Events ({player_name})',
        'Episodes': episodes,
        'Prompt': prompt.view().text(),
    })
    return episodes

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {'initialized': self._initialized}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._initialized = state.get('initialized', False)
