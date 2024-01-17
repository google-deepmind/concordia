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


"""Externality for the Game Master, which generates conversations."""

from collections.abc import Sequence
import datetime

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.clocks import game_clock
from concordia.components import agent as sim_components
from concordia.document import interactive_document
from concordia.environment.scenes import conversation as conversation_scene
from concordia.language_model import language_model
from concordia.typing import clock as clock_lib
from concordia.typing import component
from concordia.utils import helper_functions
import termcolor


class Conversation(component.Component):
  """Conversation generator."""

  def __init__(
      self,
      players: Sequence[basic_agent.BasicAgent],
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      clock: game_clock.MultiIntervalClock,
      burner_memory_factory: blank_memories.MemoryFactory,
      cap_nonplayer_characters: int = 3,
      game_master_instructions: str = '',
      shared_context: str = '',
      components: Sequence[component.Component] | None = None,
      allow_self_talk: bool = False,
      verbose: bool = False,
      print_colour: str = 'magenta',
  ):
    """Initializes the generator of conversations.

    Args:
      players: A list of players to generate conversations for.
      model: A language model to use for generating utterances.
      memory: GM memory, used to add the summary of the conversation
      clock: multi interval game clock. If conversation happens, the clock will
        advance in higher gear during the conversation scene.
      burner_memory_factory: a memory factory to create temporary memory for
        npcs and conversation gm
      cap_nonplayer_characters: The maximum number of non-player characters
        allowed in the conversation.
      game_master_instructions: A string to use as the game master instructions.
      shared_context: A string to use as the generic context for the NPCs.
      components: components that contextualise the conversation
      allow_self_talk: allow players to have a conversation with themselves
      verbose: Whether to print debug messages or not.
      print_colour: colour in which to print logs
    """
    self._players = players
    self._model = model
    self._cap_nonplayer_characters = cap_nonplayer_characters
    self._game_master_instructions = game_master_instructions
    self._shared_context = shared_context
    self._history = []
    self._verbose = verbose
    self._print_colour = print_colour
    self._components = components or []
    self._clock = clock
    self._burner_memory_factory = burner_memory_factory
    self._memory = memory
    self._allow_self_talk = allow_self_talk
    self._all_player_names = [player.name for player in self._players]
    self._min_speakers = 1 if self._allow_self_talk else 2

  def name(self) -> str:
    return 'Conversations'

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_player_names(self):
    return [player.name for player in self._players]

  def _log(self, entry):
    print(termcolor.colored(entry, self._print_colour))

  def _make_npc(
      self, name: str, scene_clock: clock_lib.GameClock
  ) -> basic_agent.BasicAgent:
    context = (
        f'{name} is a non-player character. Everyone knows the'
        f' following:\n{self._shared_context}'
    )

    mem = self._burner_memory_factory.make_blank_memory()

    npc = basic_agent.BasicAgent(
        model=self._model,
        memory=mem,
        agent_name=name,
        clock=scene_clock,
        components=[
            generic_components.constant.ConstantComponent(
                name='Instructions:', state=self._game_master_instructions
            ),
            generic_components.constant.ConstantComponent(
                name='General knowledge:', state=context
            ),
            sim_components.observation.Observation(
                agent_name=name,
                memory=mem,
                clock_now=scene_clock.now,
                timeframe=datetime.timedelta(days=1),
            ),
        ],
        verbose=True,
    )
    return npc

  def _get_nonplayer_characters(
      self,
      prompt: interactive_document.InteractiveDocument,
      scene_clock: clock_lib.GameClock,
  ) -> list[basic_agent.BasicAgent]:
    prompt = prompt.copy()
    nonplayer_characters = []
    npcs_exist = prompt.yes_no_question(
        'Are there any non-player characters in the conversation?'
    )

    if npcs_exist:
      npcs = prompt.open_question(
          'Provide the list of non-player characters in the conversation '
          + 'as a comma-separated list. For example: "bartender, merchant" '
          + 'or "accountant, pharmacist, fishmonger". Non-player '
          + 'characters should be named only by generic characteristics '
          + 'such as their profession or role (e.g. shopkeeper).'
      )
      npc_names = helper_functions.extract_from_generated_comma_separated_list(
          npcs
      )
      if len(npc_names) > self._cap_nonplayer_characters:
        npc_names = npc_names[: self._cap_nonplayer_characters]

      nonplayer_characters = [
          self._make_npc(name, scene_clock) for name in npc_names
      ]

    return nonplayer_characters

  def _generate_convo_summary(self, convo: list[str]):
    summary = self._model.sample_text(
        '\n'.join(
            convo + ['Summaries the conversation above in one sentence.'],
        ),
        max_characters=2000,
        max_tokens=2000,
        terminators=(),
    )
    return summary

  def _who_talked(
      self,
      player_names_in_conversation: list[str],
      nonplayers_in_conversation: list[basic_agent.BasicAgent],
  ):
    who_talked = (
        'Summary of a conversation between '
        + ', '.join(player_names_in_conversation)
        + '. '
    )
    if nonplayers_in_conversation:
      who_talked = (
          who_talked
          + 'Also present: '
          + ', '.join([
              npc_conversant.name
              for npc_conversant in nonplayers_in_conversation
          ])
          + '.'
      )
    return who_talked

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:
    """Potentially creates the conversation from an event statement.

    Args:
      event_statement: A string describing the event.

    Returns:
      A list of strings describing the conversation.
    """
    document = interactive_document.InteractiveDocument(self._model)
    player_names = self.get_player_names()

    for construct in self._components:
      document.statement(construct.name() + ': ' + construct.state() + '\n')

    document.statement(f'Event: {event_statement}\n')
    conversation_occurred = document.yes_no_question(
        'Does the event suggest anyone said anything or is about to speak?'
    )
    if self._verbose:
      self._log('\n Checking if conversation occurred.')

    conversation_log = {
        'date': self._clock.now(),
        'Event statement': event_statement,
        'Summary': 'No conversation occurred.',
    }

    # if yes, then propagate the event
    if conversation_occurred:
      player_names_in_conversation = []
      if self._verbose:
        self._log('\n Conversation occurred. ')
      document.statement('Conversation occurred.')
      for player_name in player_names:
        in_conversation = helper_functions.filter_copy_as_statement(
            document
        ).yes_no_question(
            'Does the event description explicitly state that'
            f' {player_name} took part in the conversation?'
        )
        if in_conversation:
          player_names_in_conversation.append(player_name)
      if self._verbose:
        self._log(
            '\n Players in conversation:'
            + ', '.join(player_names_in_conversation)
            + '.\n'
        )
      if self._verbose:
        self._log(document.view().text())

      if player_names_in_conversation:
        players_in_conversation = [
            player
            for player in self._players
            if player.name in player_names_in_conversation
        ]

        nonplayers_in_conversation = self._get_nonplayer_characters(
            document, self._clock
        )

        # this ensures that npcs can't duplicate players due to LLM mistake
        nonplayers_in_conversation = [
            player
            for player in nonplayers_in_conversation
            if player.name not in self._all_player_names
        ]
        total_speakers = len(nonplayers_in_conversation) + len(
            players_in_conversation
        )

        if total_speakers < self._min_speakers:
          self._history.append(conversation_log)
          return

        convo_scene = conversation_scene.make_conversation_game_master(
            players_in_conversation + nonplayers_in_conversation,
            clock=self._clock,
            model=self._model,
            memory_factory=self._burner_memory_factory,
            name='Conversation scene',
            premise=event_statement,
        )
        with self._clock.higher_gear():
          scene_output = convo_scene.run_episode()
        conversation_summary = self._generate_convo_summary(scene_output)

        for player in players_in_conversation:
          player.observe(conversation_summary)

        who_talked = self._who_talked(
            player_names_in_conversation, nonplayers_in_conversation
        )

        conversation_log = {
            'date': self._clock.now(),
            'Who talked?': who_talked,
            'Event statement': event_statement,
            'Summary': conversation_summary,
            'Full conversation': scene_output,
            'Chain of thought': {
                'Summary': 'Conversation chain of thought',
                'Chain': document.view().text().splitlines(),
            },
            'Scene log': convo_scene.get_history(),
        }

        conversation_summary = who_talked + ' ' + conversation_summary
        self._memory.add(conversation_summary)

        if self._verbose:
          self._log(scene_output)
          self._log(conversation_summary)

    self._history.append(conversation_log)
