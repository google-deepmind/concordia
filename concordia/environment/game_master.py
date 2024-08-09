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

"""A Generic Game Master."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import random
from typing import Any

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import agent as agent_lib
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.typing import game_master as simulacrum_game_master
from concordia.utils import concurrency
from concordia.utils import helper_functions
import termcolor


DEFAULT_THOUGHTS = (
    thought_chains.attempt_to_result,
    thought_chains.result_to_who_what_where,
)


DEFAULT_GAME_MASTER_INSTRUCTIONS = (
    'This is a social science experiment. It is structured as a '
    'tabletop roleplaying game (like dungeons and dragons). You are the '
    'game master. You will describe the current situation to the '
    'participants in the experiment and then on the basis of what you '
    'tell them they will suggest actions for the character they control. '
    'Aside from you, each other participant controls just one character. '
    'You are the game master so you may control any non-player '
    'character. You will track the state of the world and keep it '
    'consistent as time passes in the simulation and the participants '
    'take actions and change things in their world. Remember that this '
    'is a serious social science experiment. It is not just a game. It '
    'need not be fun for the participants. Always use third-person '
    'limited perspective, even when speaking directly to the participants.'
)


@dataclasses.dataclass
class LogEntry:
  """A log entry to be inserted into the game master's log at a given time.

  Attributes:
    date: the time associated with this log entry (in-game time)
    event_statement: a statement of the event that occurred
    summary: information about the event
  """

  date: datetime.datetime
  event_statement: str
  summary: str


class GameMaster(simulacrum_game_master.GameMaster):
  """A generic game master."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      clock: game_clock.GameClock,
      players: Sequence[
          basic_agent.BasicAgent
          | entity_agent_with_logging.EntityAgentWithLogging
      ],
      name: str = 'Game Master',
      update_thought_chain: (
          Sequence[
              Callable[
                  [interactive_document.InteractiveDocument, str, str], str
              ]
          ]
      ) = DEFAULT_THOUGHTS,
      components: Sequence[component.Component] = (),
      action_spec: agent_lib.ActionSpec | Mapping[str, agent_lib.ActionSpec] = (
          agent_lib.DEFAULT_ACTION_SPEC
      ),
      randomise_initiative: bool = False,
      player_observes_event: bool = True,
      players_act_simultaneously: bool = True,
      verbose: bool = False,
      concurrent_externalities: bool = True,
      concurrent_action: bool = False,
      use_default_instructions: bool = True,
      log_color: str = 'red',
  ):
    """Game master constructor.

    Args:
      model: a language model
      memory: an associative memory
      clock: a clock
      players: a sequence of generative agent simulacra which is assumed to
        contain only information that players also can access.
      name: name of the game master.
      update_thought_chain: chain of thoughts for update from player
      components: components to condition on
      action_spec: action_specs to pass to agents
      randomise_initiative: whether to randomise initiative (who goes first )
        order
      player_observes_event: send outcome of the players action back as
        observation. Helpful to turn off if using direct_effect externality to
        avoid duplicate memories.
      players_act_simultaneously: advance time after all players have acted, if
        false then advance time after each player acts.
      verbose: whether to print debugging information or not.
      concurrent_externalities: if true, runs externalities in separate threads
      concurrent_action: if true, runs player actions and events in separate
        threads
      use_default_instructions: set to False if you want to skip the standard
        instructions used for the game master, e.g. do this if you plan to pass
        custom instructions as a constant component instead.
      log_color: color in which to print logs
    """
    self._name = name
    self._model = model
    self._memory = memory
    self._clock = clock
    self._log_color = log_color
    self._randomise_initiative = randomise_initiative
    self._player_observes_event = player_observes_event
    self._players_act_simultaneously = players_act_simultaneously
    if isinstance(action_spec, agent_lib.ActionSpec):
      self._action_spec = {player.name: action_spec for player in players}
    else:
      self._action_spec = dict(action_spec)
    self._concurrent_action = concurrent_action

    components = list(components)
    if use_default_instructions:
      instructions_component = generic_components.constant.ConstantComponent(
          state=DEFAULT_GAME_MASTER_INSTRUCTIONS, name='Instructions'
      )
      components.insert(0, instructions_component)

    self._components = {}
    for comp in components:
      if comp.name() in self._components:
        raise ValueError(f'Duplicate component name: {comp.name()}')
      else:
        self._components[comp.name()] = comp

    self._verbose = verbose

    self._update_from_player_thoughts = update_thought_chain

    self._players_by_name = {player.name: player for player in players}
    if len(self._players_by_name) != len(players):
      raise ValueError('Duplicate player names')

    self._concurrent_externalities = concurrent_externalities
    self._log = []

    self.reset()

  @property
  def name(self) -> str:
    return self._name

  def get_history(self) -> Sequence[Mapping[str, Any]]:
    return self._log.copy()

  def insert_history(self, log_entry: LogEntry):
    """Insert a log entry into the game master's log, often used with scenes."""
    update_log = {
        'date': log_entry.date,
        'Event statement': log_entry.event_statement,
        'Summary': log_entry.summary,
    }
    self._log.append(update_log)

  def extend_history(self, new_history: Sequence[Mapping[str, Any]]):
    self._log.extend(new_history)

  def get_memory(self) -> associative_memory.AssociativeMemory:
    return self._memory

  def get_data_frame(self):
    return self._memory.get_data_frame()

  def _print(self, entry, color=None):
    print(termcolor.colored(entry, color or self._log_color))

  def reset(self):
    self._last_chain = None
    self._num_players = len(self._players_by_name.keys())

  def get_player_names(self):
    return list(self._players_by_name.keys())

  def update_from_player(self, player_name: str, action_attempt: str):
    prompt = interactive_document.InteractiveDocument(self._model)

    concurrency.map_parallel(
        lambda construct: construct.update_before_event(
            f'{player_name}: {action_attempt}'
        ),
        self._components.values(),
    )

    for comp in self._components.values():
      state_of_component = comp.state()
      if state_of_component:
        prompt.statement(comp.name() + ': ' + state_of_component + '\n')

    prompt.statement(f"\n{player_name}'s attempted action: {action_attempt}")

    # Produce the event that has happened as the result of the action attempt
    prompt, event_statement = thought_chains.run_chain_of_thought(
        self._update_from_player_thoughts,
        action_attempt,
        prompt,
        player_name,
    )

    self._memory.add(event_statement)

    # This gives duplicates if direct_effect-like component is used
    if self._player_observes_event:
      self._players_by_name[player_name].observe(event_statement)

    if self._verbose:
      self._print(
          '\nGM context of action and chain of thought:\n'
          + prompt.view().text()
      )

    if self._verbose:
      self._print(event_statement, 'white')

    update_log = {
        'date': self._clock.now(),
        'Event statement': event_statement,
        'Summary': f'{player_name} -- {event_statement}',
        'Chain of thought': {
            'Summary': "Game Master's chain of thought",
            'Chain': prompt.view().text().splitlines(),
        },
        'Active player': {
            'Name': player_name,
            'Action attempt': action_attempt,
            'Context for action selection and components': (
                self._players_by_name[player_name].get_last_log()
            ),
        },
    }

    # Consequences
    def get_externality(externality):
      return externality.update_after_event(event_statement)

    if self._concurrent_externalities:
      concurrency.map_parallel(get_externality, self._components.values())
    else:
      for externality in self._components.values():
        externality.update_after_event(event_statement)

    self._last_chain = prompt

    for externality in self._components.values():
      last_log = externality.get_last_log()
      if last_log:
        if 'date' in last_log.keys():
          last_log.pop('date')
        if 'Event statement' in last_log.keys():
          last_log.pop('Event statement')

        update_log[externality.name()] = last_log

    self._log.append(update_log)

    return event_statement

  def view_for_player(self, player_name):
    """Send observations to a player."""
    for comp in self._components.values():
      state_of_component = comp.partial_state(player_name)
      if state_of_component:
        for observation in state_of_component.splitlines():
          if observation:
            self._players_by_name[player_name].observe(observation)

    return

  def update_components(self) -> None:
    # MULTI THREAD!
    def _get_recursive_update_func(
        comp: component.Component,
    ) -> Callable[[], None]:
      return lambda: helper_functions.apply_recursively(
          comp, function_name='update'
      )

    with concurrency.executor() as pool:
      for comp in self._components.values():
        pool.submit(_get_recursive_update_func(comp))

  def _step_player(
      self,
      player: basic_agent.BasicAgent,
      action_spec: agent_lib.ActionSpec | None = None,
  ):
    self.update_components()
    self.view_for_player(player_name=player.name)

    if action_spec is None:
      action_spec_this_time = self._action_spec[player.name]
    else:
      action_spec_this_time = action_spec

    action = player.act(action_spec_this_time)
    action_spec_this_time.validate(action)

    self.update_from_player(action_attempt=action, player_name=player.name)

  def step(
      self,
      *,
      active_players: Sequence[basic_agent.BasicAgent] | None = None,
      action_spec: (
          agent_lib.ActionSpec | Mapping[str, agent_lib.ActionSpec] | None
      ) = None,
  ):
    """Steps the game.

    At each step players all take a turn 'quasisimultaneously' with regard to
    the main game clock, but still in a specific order within the timestep.
    This is the same principle as initiative order in dungeons and dragons.

    Args:
      active_players: Optionally specify players to take turns in this round.
      action_spec: Optionally specify what kind of actions to ask the agents to
        generate.
    """
    if active_players:
      players = list(active_players)
    else:
      players = list(self._players_by_name.values())

    if action_spec is None:
      step_player_fn = lambda player: self._step_player(player=player)
    elif isinstance(action_spec, Mapping):
      step_player_fn = lambda player: self._step_player(
          player=player, action_spec=action_spec[player.name]
      )
    elif isinstance(action_spec, agent_lib.ActionSpec):
      step_player_fn = lambda player: self._step_player(
          player=player, action_spec=action_spec
      )
    else:
      raise TypeError('Invalid action_spec parameter type')

    if self._randomise_initiative:
      random.shuffle(players)

    if self._concurrent_action:
      concurrency.map_parallel(step_player_fn, players)
    else:
      for player in players:
        step_player_fn(player)
        if not self._players_act_simultaneously:
          self._clock.advance()
    if self._players_act_simultaneously:
      self._clock.advance()

  def run_episode(self, max_steps: int = 20) -> Sequence[str]:
    for _ in range(max_steps):
      self.step()
      for comp in self._components.values():
        if comp.terminate_episode():
          return self._memory.retrieve_recent(k=1000, add_time=True)
    return self._memory.retrieve_recent(k=1000, add_time=True)

  def add_component(self, comp: component.Component) -> None:
    """Add a component to the game master."""
    self._components[comp.name()] = comp

  def remove_component(self, component_name: str) -> None:
    """Remove a component from the game master by name."""
    del self._components[component_name]

  def terminate_episode(self) -> bool:
    """Check if the episode should be terminated."""
    for comp in self._components.values():
      if comp.terminate_episode():
        return True
    return False
