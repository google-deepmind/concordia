# Copyright 2023 DeepMind Technologies Limited.
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

"""Component that helps a game master decide whose turn is next."""

from collections.abc import Callable, Sequence
import random

from concordia.agents import entity_agent
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
import termcolor


DEFAULT_RESOLUTION_COMPONENT_KEY = '__resolution__'
DEFAULT_RESOLUTION_PRE_ACT_LABEL = 'Event'
DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL = ''

PUTATIVE_EVENT_TAG = '[putative_event]'
EVENT_TAG = '[event]'


class EventResolution(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that resolves events."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      event_resolution_steps: (
          Sequence[
              Callable[
                  [interactive_document.InteractiveDocument, str, str], str
              ]
          ]
          | None
      ) = None,
      components: Sequence[str] = (),
      notify_observers: bool = False,
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      next_acting_component_key: str = (
          next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
      pre_act_label: str = DEFAULT_RESOLUTION_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      event_resolution_steps: thinking steps for the event resolution component
        to use whenever it converts putative events like action suggestions into
        real events in the simulation.
      components: Keys of components to condition the event resolution on.
      notify_observers: Whether to explicitly notify observers of the event.
      make_observation_component_key: The name of the MakeObservation component
        to use to notify observers of the event.
      memory_component_key: The name of the Memory component to use to retrieve
        memories.
      next_acting_component_key: The name of the NextActing component to use to
        get the name of the player whose turn it is.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._event_resolution_steps = event_resolution_steps
    self._components = tuple(components)
    self._notify_observers = notify_observers
    self._pre_act_label = pre_act_label

    self._make_observation_component_key = make_observation_component_key
    self._memory_component_key = memory_component_key
    self._next_acting_component_key = next_acting_component_key

    self._active_entity_name = None
    self._putative_action = None

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    prompt_to_log = ''
    observers_prompt_to_log = ''
    self._active_entity_name = None
    self._putative_action = None
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      prompt = interactive_document.InteractiveDocument(self._model)
      component_states = '\n'.join(
          [self._component_pre_act_display(key) for key in self._components]
      )
      prompt.statement(f'{component_states}\n')
      self._active_entity_name = (
          self.get_entity()
          .get_component(
              self._next_acting_component_key,
              type_=next_acting_components.NextActing,
          )
          .get_currently_active_player()
      )
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=memory_component.Memory
      )
      suggestions = memory.scan(selector_fn=lambda x: PUTATIVE_EVENT_TAG in x)
      if not suggestions:
        raise RuntimeError('No suggested events to resolve.')
      if self._active_entity_name is None:
        raise RuntimeError('No active entity suggesting an event to resolve.')

      # Filter suggestions to only those from the active player.
      # This is necessary because memory.scan() returns all matching items, and
      # without filtering we might pick up another player's action.
      #
      # NOTE: memory.scan() returns items in insertion order (recency), so after
      # filtering by player, [-1] gives the most recent action from that player.
      # If memory.scan() behavior ever changes to not preserve insertion order,
      # this code would need to be updated to use timestamps or sequence
      # numbers.
      active_player_prefix = f' {self._active_entity_name}'
      filtered_suggestions = [
          s
          for s in suggestions
          if (
              s[
                  s.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG) :
              ].startswith(active_player_prefix)
          )
      ]
      if filtered_suggestions:
        # Take the most recent (last) suggestion from this player.
        selected_suggestion = filtered_suggestions[-1]
      else:
        # Fallback: if no player-specific match found, use the last suggestion.
        selected_suggestion = suggestions[-1]

      putative_action = selected_suggestion[
          selected_suggestion.find(PUTATIVE_EVENT_TAG)
          + len(PUTATIVE_EVENT_TAG) :
      ]

      # Check if the action starts with the active entity name,
      # remove it along with any common separator (: or --) if present.
      prefix_to_remove = f' {self._active_entity_name}'
      if putative_action.startswith(prefix_to_remove):
        remainder = putative_action[len(prefix_to_remove) :]
        # Strip leading colon or dashes separator if present
        if remainder.startswith(':'):
          remainder = remainder[1:]
        elif remainder.startswith(' --'):
          remainder = remainder[3:]
        self._putative_action = remainder.strip()
      else:
        self._putative_action = putative_action

      putative_action = f'Putative event to resolve: {putative_action}'
      prompt.statement(putative_action)
      prompt, event_statement = run_chain_of_thought(
          thoughts=self._event_resolution_steps,
          premise=putative_action,
          document=prompt,
          active_player_name=self._active_entity_name,
      )

      observers_prompt_to_log = ''
      if self._notify_observers:
        make_observation = self.get_entity().get_component(
            self._make_observation_component_key,
            type_=make_observation_component.MakeObservation,
        )
        observers_prompt = prompt.copy()
        observers_prompt.statement(f'Event that occurred: {event_statement}')
        observer_names_str = observers_prompt.open_question(
            question=(
                'Which entities are aware of the event? Answer with a '
                'comma-separated list of entity names.'
            )
        )
        observer_names = observer_names_str.split(',')
        for name in observer_names:
          make_observation.add_to_queue(name.strip(' .,'), event_statement)

        observers_prompt_to_log = observers_prompt.view().text()

      result = f'{self._pre_act_label}: {event_statement}\n'
      prompt_to_log = prompt.view().text()

    self._log(
        key=self._pre_act_label,
        value=result,
        prompt=prompt_to_log,
        observers_prompt=observers_prompt_to_log,
    )
    return result

  def get_active_entity_name(self) -> str | None:
    """Returns the name of the entity that just took an action."""
    return self._active_entity_name

  def get_putative_action(self) -> str | None:
    """Returns the putative action from the entity that just took an action."""
    return self._putative_action

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        '_active_entity_name': self._active_entity_name,
        '_putative_action': self._putative_action,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._active_entity_name = state['_active_entity_name']
    self._putative_action = state['_putative_action']

  def _log(self, key: str, value: str, prompt: str, observers_prompt: str):
    self._logging_channel({
        'Key': key,
        'Summary': value,
        'Value': value,
        'Prompt': prompt,
        'Details': {
            'Observers prompt': observers_prompt,
        },
    })


class DisplayEvents(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A component that displays recent events in `pre_act` loaded from memory."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      num_events_to_retrieve: int = 100,
      pre_act_label: str = 'Recent events',
      event_filter_fn: Callable[[str], bool] | None = None,
  ):
    """Initializes the component.

    Args:
      model: The language model to use for the component.
      memory_component_key: The name of the memory component in which to write
        records of events.
      num_events_to_retrieve: The number of events to retrieve.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.
      event_filter_fn: Optional predicate (event_string) -> bool. When provided,
        only events for which this function returns True will be included.
        Subclasses can use this for entity-aware filtering (e.g. by location) by
        providing a closure that captures the relevant state.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._num_events_to_retrieve = num_events_to_retrieve
    self._event_filter_fn = event_filter_fn

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    events = memory.scan(selector_fn=lambda x: EVENT_TAG in x)

    if self._event_filter_fn:
      events = [e for e in events if self._event_filter_fn(e)]

    limit = self._num_events_to_retrieve
    if limit > len(events):
      limit = len(events)
    events = events[-limit:]
    events = [
        f'{i}). {event.split(EVENT_TAG)[-1]}' for i, event in enumerate(events)
    ]

    events_str = '\n'.join(events)

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': events_str,
        'Value': events_str,
    })

    return events_str

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    return {
        'memory_component_key': self._memory_component_key,
        'num_events_to_retrieve': self._num_events_to_retrieve,
        'pre_act_label': self.get_pre_act_label(),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    if 'memory_component_key' in state:
      self._memory_component_key = state['memory_component_key']
    if 'num_events_to_retrieve' in state:
      self._num_events_to_retrieve = state['num_events_to_retrieve']


class SendEventToRelevantPlayers(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """A component that passes events to relevant players."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      make_observation_component_key: str,
      components: Sequence[str] = (),
      player_filter: Callable[[], Sequence[str]] | None = None,
      pre_act_label: str = DEFAULT_SEND_PRE_ACT_VALUES_TO_PLAYERS_PRE_ACT_LABEL,
  ):
    """Initializes a component that sends component pre-act values to players.

    This component sends the events during the post-act phase to the players
    that are aware of the event, by queueing them in the MakeObservation
    component.

    Args:
      model: The language model to use for the component.
      player_names: Names of players (used if no player_filter is supplied).
      make_observation_component_key: the key for a MakeObservation component to
        add the pre-act values to the queue of events to observe.
      components: Keys of components to condition whether the event is relevant.
        If empty, all events are relevant.
      player_filter: Optional callback that returns the list of player names
        that should receive observations. When provided, this function is called
        during post_act to determine which players get the event. This can be
        used to filter by scene participants or any other criteria.
      pre_act_label: Prefix to add to the output of the component when called in
        `pre_act`.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    super().__init__()
    self._model = model
    self._player_names = player_names
    self._pre_act_label = pre_act_label
    self._components = tuple(components)
    self._player_filter = player_filter
    self._queue = {}
    self._last_action_spec = None
    self._optional_make_observation_component_key = (
        make_observation_component_key
    )

    self._map_names_to_previous_observations = {
        player_name: '' for player_name in player_names
    }

  def get_named_component_pre_act_value(self, component_name: str) -> str:
    """Returns the pre-act value of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_value()
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'{self.get_component_pre_act_label(key)}:\n'
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    self._last_action_spec = action_spec
    return ''

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
    result = ''
    prompts_to_log = {}
    if (
        self._last_action_spec
        and self._last_action_spec.output_type == entity_lib.OutputType.RESOLVE
    ):
      prompt = interactive_document.InteractiveDocument(self._model)
      proceed = True

      # Use player_filter if provided, otherwise all players
      relevant_players = self._player_names
      if self._player_filter:
        relevant_players = self._player_filter()

      for active_entity_name in relevant_players:
        if self._components:
          component_states = '\n'.join(
              [self._component_pre_act_display(key) for key in self._components]
          )
          prompt.statement(f'{component_states}\n')
          prompt.statement(f'{action_attempt}\n')
          proceed = prompt.yes_no_question(
              question=(
                  f'Is {active_entity_name} aware of the latest event above?'
              )
          )

        prompts_to_log[active_entity_name] = prompt.view().text()

        if proceed:
          # Remove their previous observation since they have already seen it.
          result = action_attempt.replace(
              self._map_names_to_previous_observations[active_entity_name], ''
          )
          if self._optional_make_observation_component_key:
            make_observation = self.get_entity().get_component(
                self._optional_make_observation_component_key,
                type_=make_observation_component.MakeObservation,
            )
            make_observation.add_to_queue(active_entity_name, result)

        self._map_names_to_previous_observations[active_entity_name] += result

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result,
        'Value': result,
        'Prompts': prompts_to_log,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Returns the state of the component."""
    action_spec_dict = (
        self._last_action_spec.to_dict() if self._last_action_spec else None
    )
    return {
        '_queue': self._queue,
        '_last_action_spec': action_spec_dict,
        '_map_names_to_previous_observations': (
            self._map_names_to_previous_observations
        ),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the state of the component."""
    self._queue = state['_queue']
    action_spec_dict = state['_last_action_spec']
    if action_spec_dict and isinstance(action_spec_dict, dict):
      self._last_action_spec = entity_lib.action_spec_from_dict(
          action_spec_dict
      )
    else:
      self._last_action_spec = None
    self._map_names_to_previous_observations = state[
        '_map_names_to_previous_observations'
    ]


def identity(
    chain_of_thought: interactive_document.InteractiveDocument,
    premise: str,
    active_player_name: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    premise: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del chain_of_thought, active_player_name
  return premise


def extract_direct_quote(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Outputs the premise. Use this to create a pass-through chain of thought.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the action attempt
  """
  inner_chain_of_thought = chain_of_thought.new()
  inner_chain_of_thought.statement(f'{action_attempt}')
  proceed = inner_chain_of_thought.yes_no_question(
      question=f'Did {active_player_name} explicitly say or write anything?')
  if proceed:
    proceed_with_exact = inner_chain_of_thought.yes_no_question(
        question=(
            f'Does the text state exactly what {active_player_name} said or '
            'wrote?'))
    if proceed_with_exact:
      direct_quote = inner_chain_of_thought.open_question(
          question=f'What exactly did {active_player_name} say or write?',
          max_tokens=2500,
          terminators=(),
      )
      chain_of_thought.statement(f'[direct quote] {direct_quote}')

  return action_attempt


def determine_success_and_why(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del active_player_name
  success = chain_of_thought.yes_no_question(
      'Does the attempted action succeed? If the attempted action '
      + 'is easy to accomplish then the attempt should be successful '
      + 'unless there is a specific reason for it to fail.'
  )
  why_failed = 'this failed'  # will be overwritten if needed.
  if success:
    chain_of_thought.statement('The attempt succeeded.')
  else:
    chain_of_thought.statement('The attempt failed.')
    why_failed = chain_of_thought.open_question(
        'Why did the attempt fail?',
    )

  if action_attempt[-1] == '.':
    action_attempt = action_attempt[:-1] + ','
  success_or_not = 'successful' if success else 'not successful'
  result = f'{action_attempt} and was {success_or_not}.'
  if not success:
    result = f'{result}. However, {why_failed}'

  chain_of_thought.statement(result)
  return result


def result_to_causal_statement(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines the causal outcome of the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  del active_player_name
  effect = chain_of_thought.open_question(
      'Because of that, what happens as a result?',
      max_tokens=1200,
  )
  raw_causal_statement = f'{event} Because of that, {effect}'
  causal_statement = chain_of_thought.open_question(
      question=(
          'Rewrite the following statements to be one sentence and to better '
          'highlight cause and effect. Do not express uncertainty (e.g. say '
          '"Francis released the demon" not "Francis could release the demon" '
          'and not "The demon may have been released")\n'
          f'Statements: {raw_causal_statement}\n'
      ),
      max_tokens=1500,
      terminators=(),
  )
  return causal_statement


def attempt_to_result(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  del active_player_name
  result = chain_of_thought.open_question(
      'What happens as a result of the attempted action?'
      ' Take into account the location and status of each player.',
      max_tokens=1200,
      terminators=(),
  )
  raw_causal_statement = f'{action_attempt} Because of that, {result}'
  return raw_causal_statement


def attempt_to_most_likely_outcome(
    chain_of_thought: interactive_document.InteractiveDocument,
    action_attempt: str,
    active_player_name: str,
):
  """Determine success of action_attempt and reason for success/failure.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    action_attempt: the attempted action
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  _ = chain_of_thought.open_question(
      f'Where is {active_player_name}?',
      max_tokens=1200,
      terminators=(),
  )
  _ = chain_of_thought.open_question(
      f'What is {active_player_name} trying to do?',
      max_tokens=1200,
      terminators=(),
  )
  _ = chain_of_thought.open_question(
      f"List at least 3 possible direct consequences of {active_player_name}'s "
      'action. Never assume any other person will take a voluntary action. '
      'Be specific and concrete. Never beg the question. For instance, it is '
      'wrong to say "Alex finds something". Instead specify exactly '
      'what Alex finds. For example "Alex finds a teddy bear".',
      max_tokens=3000,
      terminators=(),
  )
  result = chain_of_thought.open_question(
      'Which outcome is the most likely?',
      max_tokens=1200,
      terminators=(),
  )
  raw_causal_statement = f'{action_attempt} Because of that, {result}'
  return raw_causal_statement


def result_to_who_what_where(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines who have done what where, given the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  del active_player_name
  chain_of_thought.statement(event)
  causal_statement = chain_of_thought.open_question(
      'Rewrite the statements above to better highlight'
      ' the main person the event is about, where and what they did, and what'
      ' happened as a result. Do not express uncertainty (e.g. say "Francis'
      ' opened the door" not "Francis could open the door" and not "The door'
      ' may have been opened"). If anyone spoke then make sure to include '
      ' exaxtly what they said verbatim.\n',
      max_tokens=1500,
      terminators=(),
  )
  return causal_statement


def result_to_effect_caused_by_active_player(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Determines who have done what where, given the event.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the event to determine the causal outcome of
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  chain_of_thought.statement(event)
  causal_statement = chain_of_thought.open_question(
      'Rewrite the statements above to be one sentence and to better highlight '
      f'what {active_player_name} did, and what happened as a result. '
      'Do not express uncertainty (e.g. say '
      '"Francis opened the door" not "Francis could open the door" '
      'and not "The door may have been opened"). Remember that the role of '
      'the game master in a tabletop role-playing game is akin to the author '
      'for all parts of the story not written by the player characters. '
      'Therefore, it is critical always to take a stance on what is happening '
      'and invent when necessary. For instance, if Francis opens a door to a '
      'room no one visited before then the game master should invent what is '
      'in the room using common sense and knowledge of the game world.',
      max_tokens=1500,
      terminators=(),
  )
  return causal_statement


def restore_direct_quote(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    active_player_name: str,
):
  """Restore details from action attempt lost in subsequent processing.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the candidate event
    active_player_name: name of player whose turn it currently is

  Returns:
    string describing the outcome
  """
  chain_of_thought.statement(
      '\n\nCandidate event statement which may have lost direct '
      f'quotes: {event}')
  event_with_quote = chain_of_thought.open_question(
      question=(
          'Incorporate the exact text of anything said or written '
          + f'by {active_player_name} into the candidate event statement. '
          + 'Note that all direct quotes should have been tagged in the '
          + f'text above with [direct quote]. If {active_player_name} '
          + 'said or wrote anything then their direct quote must be part of '
          + 'the answer. It is also important to maintain as much detail as '
          + 'possible from the latest candidate event statement.'
      ),
      max_tokens=3500,
      terminators=(
          '\nCandidate event statement',
          '\nQuestion',
          '\nAnswer',
          '\n\n',
      ),
  )
  return event_with_quote


class AccountForAgencyOfOthers:
  """Prevents players from taking voluntary actions they do not agree to take.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[entity_agent.EntityAgent],
      verbose: bool = False,
  ):
    self._model = model
    self._players = players
    self._player_names = [player.name for player in players]
    self._verbose = verbose
    self._player_by_name = {player.name: player for player in players}

  def __call__(
      self,
      chain_of_thought: interactive_document.InteractiveDocument,
      candidate_event: str,
      active_player_name: str,
  ):
    tmp_chain_of_thought = interactive_document.InteractiveDocument(
        model=self._model)
    tmp_chain_of_thought.statement(f'Event: {candidate_event}')
    _ = tmp_chain_of_thought.open_question(
        'Describe all voluntary actions taken by any individual in the '
        + 'event above.',
        max_tokens=1500,
    )

    voluntary_act_of_inactive_player = tmp_chain_of_thought.yes_no_question(
        ('Does the event indicate or imply that anyone besides ' +
         f'{active_player_name} took a voluntary action?'),
    )

    possible_outcomes = []
    players_who_would_not = []
    if voluntary_act_of_inactive_player:
      inactive_players_who_acted_str = tmp_chain_of_thought.open_question(
          question=(
              f'Aside from {active_player_name}, which individuals took a '
              + 'voluntary action?\n'
              + 'Respond with a comma-separated list, for example: \n'
              + 'Jacob,Alfred,Patricia'
          )
      )
      inactive_players_who_acted = inactive_players_who_acted_str.split(',')
      random.shuffle(inactive_players_who_acted)
      for player in inactive_players_who_acted:
        player_ = player.strip(' ')
        if player_ in self._player_names:
          tmp_chain_of_thought_per_player = tmp_chain_of_thought.copy()
          what_did_they_do = tmp_chain_of_thought_per_player.open_question(
              f'In one sentence, what did {player_} do?',
              answer_prefix=f'{player_} ',
          )
          what_did_they_do = f'{player_} {what_did_they_do}'
          call_to_action = ('Is the following possible action something that ' +
                            '{name} would do in this situation?\n' +
                            f'Possible action: {what_did_they_do}\n')
          action_spec = entity_lib.choice_action_spec(
              call_to_action=call_to_action,
              options=['Yes', 'No'],
              tag='action',
          )
          would_they_do_it = self._player_by_name[player_].act(action_spec)
          action_spec.validate(would_they_do_it)
          if self._verbose:
            print(termcolor.colored(
                tmp_chain_of_thought_per_player.view().text(), 'yellow'))
            print(termcolor.colored(f'Would they do it? {would_they_do_it}',
                                    'yellow'))
          if would_they_do_it == 'No':
            players_who_would_not.append(player_)
            no_chain_of_thought = interactive_document.InteractiveDocument(
                model=self._model)
            no_chain_of_thought.statement(
                f'Event that did not occur: {candidate_event}')
            no_chain_of_thought.statement(
                'A reason the above event did not occur is the fact '
                + f'that {player_} would not have acted that way.'
            )
            outcome = no_chain_of_thought.open_question(
                'Given the above, what happened instead? The answer should '
                + f'be what would have happened but for {player_}. Answer in '
                + 'the form of a simple statement of cause and effect.'
            )
            possible_outcomes.append(outcome)
            if self._verbose:
              print(termcolor.colored(
                  no_chain_of_thought.view().text(), 'yellow'))

    if players_who_would_not:
      players_who_would_not_str = ', '.join(players_who_would_not)
      chain_of_thought.statement(
          'The aforementioned event could not have occurred because the ' +
          'following individuals would not have acted that way: ' +
          f'{players_who_would_not_str}.')
      for outcome in possible_outcomes:
        chain_of_thought.statement(
            'Therefore a likely effect of ' +
            f"{active_player_name}'s attempted action is: " + outcome)
      candidate_event = chain_of_thought.open_question(
          f"What happened as a direct result of {active_player_name}'s "
          + 'attempted action? Take into account the reactions of '
          + f'{players_who_would_not_str}. Highlight how '
          + f"{active_player_name}'s action caused its actual effect.",
          max_tokens=1500,
      )
      if self._verbose:
        print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
    return candidate_event


class Conversation:
  """Resolve conversations into events.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      players: Sequence[entity_agent.EntityAgent],
      verbose: bool = False,
  ):
    self._model = model
    self._players = players
    self._player_names = [player.name for player in players]
    self._verbose = verbose
    self._player_by_name = {player.name: player for player in players}

  def __call__(
      self,
      chain_of_thought: interactive_document.InteractiveDocument,
      candidate_event: str,
      active_player_name: str,
  ) -> str:
    tmp_chain_of_thought = interactive_document.InteractiveDocument(
        model=self._model)
    tmp_chain_of_thought.statement(f'Event: {candidate_event}')
    conversation_occurred = tmp_chain_of_thought.yes_no_question(
        'Does the event suggest a conversation?',
    )
    if not conversation_occurred:
      return candidate_event

    participants_str = tmp_chain_of_thought.open_question(
        question=('Who participated in the conversation? Respond in a '
                  'comma-separated list e.g. "Jacob,Sasha,Ishita". Use full '
                  'names if known.'),
        terminators=('.',),
    )
    participant_names = participants_str.split(',')
    if not participant_names:
      return candidate_event
    contributions = {}
    for participant_name in participant_names:
      name = participant_name.strip()
      if name in self._player_names:
        action_spec = entity_lib.free_action_spec(
            call_to_action=(
                f'A conversation is suggested by {candidate_event}. '
                'In this event, what would {name} say?'),
            tag='action',
        )
        contribution = self._player_by_name[name].act(action_spec)
      else:
        contribution = tmp_chain_of_thought.open_question(
            question=(f'What would {name} say?')
        )
      contributions[name] = f'{name}: {contribution}'

    contributions_str = '\n'.join(contributions.values())
    chain_of_thought.statement(
        f'Contributions to the conversation:\n{contributions_str}\n'
    )
    conversation = chain_of_thought.open_question(
        question=('Generate a conversation consistent with all the relevant '
                  'information above, especially the explicit contributions '
                  'of each participant and their underlying intentions. Only '
                  'characters with explicit contributions listed above should '
                  'speak in the conversation. No one else.'),
        max_tokens=2200,
        terminators=(),
        question_label='Exercise',
    )

    for participant_name in contributions:
      name = participant_name.strip()
      if name in self._player_names:
        self._player_by_name[name].observe(conversation)

    if self._verbose:
      print(termcolor.colored(
          (f'Contributions:\n{contributions_str}\n\n'
           f'Conversation:\n{conversation}'),
          'yellow'))
    return conversation


class RemoveSpecificText:
  """Remove specific text from a string.
  """

  def __init__(
      self,
      substring_to_remove: str,
  ):
    self._substring_to_remove = substring_to_remove

  def __call__(
      self,
      chain_of_thought: interactive_document.InteractiveDocument,
      candidate_event: str,
      active_player_name: str,
  ) -> str:
    result = candidate_event.replace(self._substring_to_remove, '')
    return result


def get_action_category_and_player_capability(
    chain_of_thought: interactive_document.InteractiveDocument,
    putative_event: str,
    active_player_name: str,
):
  """Determines the category of the attempted action.

  Inspired by the tabletop role-playing game rule book: "Girl by moonlight" by
  Andrew Gillis. 2023. Evil Hat Productions LLC.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    putative_event: the candidate event
    active_player_name: name of player whose turn it currently is

  Returns:
    unmodified string describing the putative event.
  """
  chain_of_thought.statement(
      'The following types of voluntary action are possible:\n1. Confess --'
      ' When a character confesses, they expose the inner world of their'
      ' thoughts and feelings to another. They might reveal to someone that it'
      ' was them who left anonymous gifts at their doorstop. They might admit'
      ' to a friend that they were once affiliated with a rival'
      ' organization.\n2. Forgive -- When a character forgives, they show that'
      ' they care for someone despite a mistake they have made. They might'
      ' offer a chance for reconciliation to a friend that wronged them. They'
      ' might reach out to a troubled individual with a difficult past, in the'
      ' hopes that they might find redemption.\n3. Perceive -- When a character'
      ' perceives, they see the world as it presents itself, without judgement.'
      ' They might observe someone and note their daily routine. They might see'
      ' beauty present in even the darkest and most unusual corners of the'
      ' world.\n4. Express --  When a character expresses, they communicate'
      ' purposefully to achieve a specific outcome. They might share an'
      ' important life lesson about friendship with a younger colleague. They'
      ' might negotiate with a long-standing competitor to form a temporary'
      ' alliance for mutual benefit.\n5. Defy -- When a character defies, they'
      ' muster their courage and face opposition head on. They might stand up'
      ' to a bully. They might confront a dangerous adversary with'
      ' determination. They might speak truth to power in a tense meeting'
      ' situation.\n6. Empathize -- When a character empathizes, they'
      ' understand a person intuitively, and feel their emotions as if they'
      " were their own. They might listen to someone's story, and gain an"
      ' understanding of their perspective. They might connect with a person'
      ' others find frightening, and sense the vulnerability beneath their'
      ' intimidating exterior.\n7. Conceal -- When a character conceals, they'
      ' hide their true intentions and feelings. They might pass unnoticed, as'
      ' just another face in the crowd. They might suppress their true feelings'
      ' and lie to someone or present a composed façade during a difficult'
      ' conversation.\n8. Flow -- When a character flows, they move with'
      ' adaptability and respond intuitively to changing circumstances. They'
      ' might navigate a physical obstacle with practiced agility. They might'
      ' seamlessly adjust their behavior to match social expectations in'
      ' different environments.\n9. Analyze -- When a character analyzes, they'
      ' search beyond the surface presentation of the world, and discover'
      ' secrets. They might study an old document for historical information.'
      " They might discern a weakness in an opponent's strategy.\n"
  )
  categories = (
      'Confess',
      'Forgive',
      'Perceive',
      'Express',
      'Defy',
      'Empathize',
      'Conceal',
      'Flow',
      'Analyze',
  )
  chain_of_thought.statement(
      f'The player responsible for {active_player_name} suggests: '
      f'{putative_event}')
  category_idx = chain_of_thought.multiple_choice_question(
      question=(f'What category does {active_player_name}\'s latest voluntary '
                'action fall into?'),
      answers=categories,
  )
  category = categories[category_idx]
  _ = chain_of_thought.open_question(
      question=(
          f'Is {active_player_name} proficient in actions of type {category} '
          'and currently able to perform them? Why or why not? Never refer '
          'directly to action categories, rather provide an appropriate answer '
          'to the question in terms that make sense within the story itself.'
      ),
      max_tokens=1500,
      terminators=(),
  )
  return putative_event


def maybe_inject_narrative_push(
    chain_of_thought: interactive_document.InteractiveDocument,
    putative_event: str,
    active_player_name: str,
):
  """Maybe inject some new event to push the narrative forward.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    putative_event: putative event to (maybe) add a narrrative push to
    active_player_name: name of player whose turn it currently is

  Returns:
  """
  del active_player_name
  if not chain_of_thought.yes_no_question(
      question='Is the story traced out by the above list of events repetitive?'
  ):
    return putative_event

  tmp_chain_of_thought = chain_of_thought.copy()
  plausible_events = tmp_chain_of_thought.open_question(
      question=('Suggest five plausible events or complications to the '
                'putative event being resolved that may now occur to move '
                'the narrative forward. Put each one on a separate line. '
                'Do not include titles, headings, or numbers. Just '
                'include the in-narrative description of each of the plausible '
                'events or complications. Only suggest the sort of thing that '
                'the game master of a typical tabletop role-playing game would '
                'be able to decide on their own without reducing '
                'the players\' feelings of agency or making the world feel '
                'arbitrary and unfair. '
                'In particular, never assume a player character takes '
                'a voluntary action. Non-player characters, on the other hand, '
                'can always be used to push the narrative forward since they '
                'are controlled by the game master.'
                'Eg:\nfirst plausible event\nsecond plausible event\n'
                'third plausible event\nfourth plausible event\n'
                'fifth plausible event\n'),
      max_tokens=1500,
      terminators=(),
  )
  plausible_events = plausible_events.split('\n')

  additional_plausible_event = random.choice(plausible_events)
  combined_plausible_event = (f'1. {putative_event}\n'
                              f'2. {additional_plausible_event}')

  composite_event = chain_of_thought.open_question(
      question=(f'Given:\n{combined_plausible_event}\n\nHow does 2 modify, '
                'complicate, extend, or otherwise change the meaning of 1? '
                'Answer in the form of an in-narrative compound event that '
                'incorporates both 1 and 2 into a single composite event.'),
      max_tokens=1500,
      terminators=(),
  )

  return composite_event


def maybe_cut_to_next_scene(
    chain_of_thought: interactive_document.InteractiveDocument,
    event: str,
    unused_active_player_name: str,
):
  """Determine if the story demands a cut in time.

  Args:
    chain_of_thought: the document to condition on and record the thoughts
    event: the latest event
    unused_active_player_name: name of player whose turn it currently is

  Returns:
  """
  _ = chain_of_thought.open_question(
      question=('What is the current scene about? What is the narrative '
                'function of the latest event to occur in the scene?'),
      max_tokens=1500,
      terminators=(),
  )
  chain_of_thought.statement(
      'Stories often transition between scenes in order to highlight the '
      'most crucial or dramatic moments in an overall narrative, and to skip '
      'uninteresting details. Time gaps between scenes can span any duration.'
  )
  cut_to_next_scene = chain_of_thought.yes_no_question(
      question=(
          'Considering the current story, does the latest event constitute '
          'a good place in the overall story for the current scene to end? '
          'Note that the only valid reasons to end a scene are:\n'
          '*Resolution of Immediate Conflict: A central conflict or tension '
          'within the scene might be resolved, even if larger conflicts '
          'remain.\n'
          '*Pacing Considerations: If a scene has served its purpose and '
          'risks becoming drawn out or repetitive, it\'s time to end it '
          'and move on to advance the story.\n'
          '*Revelation of Key Information: Once crucial information has been '
          'revealed, a scene might end to allow the audience (or other '
          'characters) to process it.\n'
          '*Catalyst for Future Action: The scene might end just after an '
          'event that will clearly trigger the events of the next scene.\n'
          '*Establishment of Setting or Character: If the primary goal of '
          'the scene was to establish a new location or reveal a significant '
          'aspect of a character, it can end once that\'s achieved.\n'
          '*Emphasis of a Theme: The scene might end after a moment that '
          'strongly reinforces a central theme of the story.\n'
          '*Natural Break Points: Sometimes the physical environment or the '
          'flow of events within the narrative suggests a natural place to '
          'pause. For instance, when all characters go to sleep it\'s a good '
          'time to end the scene.\n'
          '**Remember: If none of the above criteria apply, or if a transition '
          'would feel too abrupt, then continue the current scene '
          '(respond "No" to the question). **\n')
  )
  if cut_to_next_scene:
    next_scene_and_time_till_it_starts = chain_of_thought.open_question(
        question=(
            'What duration should we declare to have passed '
            'before presenting the next truly compelling scene? What will '
            'be the underlying premise or driving force of that scene? Be '
            'creative in your suggestions here. Include just the time duration '
            'on its own line (just one line) at the end of your answer, and '
            'include no other text on that line aside from the duration. '
            'Describe the duration with a phrase such as "a few minutes '
            'later", "three days later", "next year", "many years later", '
            'or "in the far future".'
        ),
        max_tokens=1200,
        terminators=(),
    ).strip()
    splits = next_scene_and_time_till_it_starts.split('\n')
    if len(splits) >= 2:
      duration = splits[-1].strip()
      event += f'\n\n[CUT TO NEXT SCENE]\n\nSetting: {duration}\n'

  return event


def run_chain_of_thought(
    thoughts: Sequence[
        Callable[[interactive_document.InteractiveDocument, str, str], str]
    ],
    premise: str,
    document: interactive_document.InteractiveDocument,
    active_player_name: str,
):
  """Run a chain of thoughts in the document.

  Args:
    thoughts: sequence of 'thought' functions each of which process a document
      and candidate event string.
    premise:  the starting premise of the chain
    document: the working document
    active_player_name: name of player whose turn it currently is

  Returns:
    document: the final version of the document that recorded the chain
    conclusion: the result of the last thought
  """
  conclusion = premise

  for f in thoughts:
    conclusion = f(document, premise, active_player_name)
    premise = conclusion
  return document, conclusion

