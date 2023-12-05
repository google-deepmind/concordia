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

from collections.abc import Callable, Sequence
import concurrent.futures
import random

from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import agent as simulacrum_agent
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.typing import game_master as simulacrum_game_master
import termcolor


DEFAULT_THOUGHTS = [
    thought_chains.attempt_to_result,
    thought_chains.result_to_who_what_where,
]


class GameMaster(simulacrum_game_master.GameMaster):
  """A generic game master."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      clock: game_clock.GameClock,
      players: Sequence[basic_agent.BasicAgent],
      name: str = 'Game Master',
      update_thought_chain: (
          Sequence[
              Callable[[interactive_document.InteractiveDocument, str], str]
          ]
          | None
      ) = None,
      components: Sequence[component.Component] | None = None,
      action_spec: simulacrum_agent.ActionSpec | None = None,
      randomise_initiative: bool = False,
      player_observes_event: bool = True,
      players_act_simultaneously: bool = True,
      verbose: bool = False,
      concurrent_externalities: bool = True,
      concurrent_action: bool = False,
      log_colour: str = 'red',
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
      action_spec: specific action_spec to pass to agents, default is used if
        None
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
      log_colour: colour in which to print logs
    """
    self._name = name
    self._model = model
    self._memory = memory
    self._clock = clock
    self._players = players
    self._log_colour = log_colour
    self._randomise_initiative = randomise_initiative
    self._player_observes_event = player_observes_event
    self._players_act_simultaneously = players_act_simultaneously
    self._action_spec = action_spec or simulacrum_agent.DEFAULT_ACTION_SPEC
    self._concurrent_action = concurrent_action

    self._components = {}
    for comp in components:
      if comp.name() in self._components:
        raise ValueError(f'Duplicate component name: {comp.name()}')
      else:
        self._components[comp.name()] = comp

    self._verbose = verbose

    self._update_from_player_thoughts = update_thought_chain or DEFAULT_THOUGHTS

    self._players_by_name = {player.name: player for player in self._players}

    self._concurrent_externalities = concurrent_externalities
    self._log = []

    self.reset()

  def name(self):
    return self._name

  def get_history(self):
    return self._log.copy()

  def get_data_frame(self):
    return self._memory.get_data_frame()

  def _print(self, entry, colour=None):
    print(termcolor.colored(entry, colour or self._log_colour))

  def reset(self):
    self._last_chain = None
    self._num_players = len(self._players)

  def get_player_names(self):
    return [player.name for player in self._players]

  def update_from_player(self, player_name: str, action_attempt: str):
    prompt = interactive_document.InteractiveDocument(self._model)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      executor.map(
          lambda construct: construct.update_before_event(
              f'{player_name}: {action_attempt}'
          ),
          self._components,
      )

    for comp in self._components.values():
      state_of_component = comp.state()
      if state_of_component:
        prompt.statement(comp.name() + ': ' + comp.state() + '\n')

    prompt.statement(f"\n{player_name}'s attempted action: {action_attempt}")

    # Produce the event that has happened as the result of the action attempt
    prompt, event_statement = thought_chains.run_chain_of_thought(
        self._update_from_player_thoughts, action_attempt, prompt
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
        'Summary': event_statement,
        'Chain of thought': {
            'Summary': "Game Master's chain of thought",
            'Chain': prompt.view().text().splitlines(),
        },
        'Active player': {
            'Name': player_name,
            'Action attempt': action_attempt,
            'Chain of thought': self._players_by_name[
                player_name
            ].get_last_log(),
        },
    }

    # Consequences
    def get_externality(externality):
      return externality.update_after_event(event_statement)

    if self._concurrent_externalities:
      with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(get_externality, self._components.values())
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
        self._players_by_name[player_name].observe(
            comp.name() + ': ' + state_of_component
        )

    return

  def update_components(self) -> None:
    # MULTI THREAD!
    with concurrent.futures.ThreadPoolExecutor() as executor:
      executor.map(
          lambda construct: construct.update(), list(self._components.values()))

  def _step_player(self, player: basic_agent.BasicAgent):
    self.update_components()
    self.view_for_player(player_name=player.name)
    action = player.act(self._action_spec)

    self.update_from_player(action_attempt=action, player_name=player.name)

  def step(self):
    """Steps the game.

    At each step players all take a turn 'quasisimultaneously' with regard to
    the main game clock, but still in a specific order within the timestep.
    This is the same principle as initiative order in dungeons and dragons.
    """
    players = list(self._players)

    if self._randomise_initiative:
      random.shuffle(players)

    if self._concurrent_action:
      with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(self._step_player, players)
    else:
      for player in players:
        self._step_player(player)
        if not self._players_act_simultaneously:
          self._clock.advance()
    if self._players_act_simultaneously:
      self._clock.advance()

  def run_episode(self, max_steps: int = 20) -> list[str]:
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
