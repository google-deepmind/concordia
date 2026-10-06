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

"""Async conversation components for multi-turn dialogues.

Designed for the asynchronous engine.

IMPORTANT: These components are designed specifically for use with the
`asynchronous.Asynchronous` engine. They rely on the async engine's per-agent
threading model for turn coordination - agents not holding the "turn" will
simply loop back naturally rather than blocking.

This module provides three GM components following the forum.py pattern:

  - AsyncConversationState: Thread-safe shared state tracking active
    conversations, participants, turns, and utterances. Registered under
    __async_conversation__ and accessed by other components via
    get_entity().get_component().

  - AsyncConversationObservation: Returns dialogue context for the current
    speaker (registered as __make_observation__). Agents see conversation
    history and whether it's their turn.

  - AsyncConversationResolution: Parses speech actions and updates conversation
    state (registered as __resolution__). Enforces turn-taking by rejecting
    out-of-turn speech attempts.

Usage:
  1. Add AsyncConversationState to your GM's components dict
  2. Add AsyncConversationObservation and AsyncConversationResolution
  3. Use the asynchronous.Asynchronous engine
  4. Call create_conversation() to start a dialogue between two agents
"""

from collections.abc import Sequence
import dataclasses
import datetime
import logging
import re
import threading
import time
from typing import Any

from concordia.components.game_master import event_resolution
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from examples.concordia_island.sim import component_state

PUTATIVE_EVENT_TAG = event_resolution.PUTATIVE_EVENT_TAG

_EXIT_PATTERNS = [
    r'\bleav(e|es|ing)\b',
    r'\bdepart(s|ed|ing)?\b',
    r'\bgoodnight\b',
    r'\bgood\s*night\b',
    r'\bgoodbye\b',
    r'\bbye\b',
    r'\bhead(s|ing)?\s*(out|home|back)\b',
    r'\bconclud(e|es|ed|ing)\b',
    r'\bterminat(e|es|ed|ing)\b',
    r'\bstep(s|ping)?\s*(out|away)\b',
    r'\bwalk(s|ing)?\s*(away|out|off)\b',
    r'\bgot\s*to\s*(go|run|head)\b',
    r'\bsee\s*you\s*(later|tomorrow|soon)\b',
    r'\btake\s*care\b',
]
_EXIT_REGEX = re.compile('|'.join(_EXIT_PATTERNS), re.IGNORECASE)


DEFAULT_ASYNC_CONVERSATION_KEY = '__async_conversation__'
DEFAULT_CONVERSATION_PRE_ACT_LABEL = '\nConversation'
DEFAULT_MAX_TURNS = 8
DEFAULT_TERMINATE_CHECK_MIN_TURN = 4

DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    'What is the current situation faced by {name}? What do they now observe?'
    ' Only include information of which they are aware.'
)


@dataclasses.dataclass
class Conversation:
  """A single conversation between co-located agents."""

  conv_id: int
  participants: tuple[str, ...]
  location: str
  utterances: list[tuple[str, str]] = dataclasses.field(default_factory=list)
  turn: int = 0
  active: bool = True
  started_at: str = ''
  max_turns: int = DEFAULT_MAX_TURNS
  terminate_check_min_turn: int = DEFAULT_TERMINATE_CHECK_MIN_TURN
  wall_start_time: float = dataclasses.field(default_factory=time.monotonic)
  context: str = ''


class AsyncConversationState(entity_component.ContextComponent):
  """Thread-safe async conversation state managing dialogues between agents.

  Registered under __async_conversation__ and accessed by
  ConversationObservation
  and ConversationResolution via get_entity().get_component(). This follows
  the same pattern as ForumState wrapping forum data.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      max_turns: int = DEFAULT_MAX_TURNS,
      clock_key: str | None = None,
      conversation_timeout: float = 180.0,
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._max_turns = max_turns
    self._clock_key = clock_key
    self._conversation_timeout = conversation_timeout

    self._lock = threading.RLock()
    self._conversations: dict[int, Conversation] = {}
    self._next_conv_id = 0
    self._player_to_conv: dict[str, int] = {}
    self._completed_pairs: dict[frozenset[str], int] = {}
    self._last_seen_index: dict[str, int] = {}

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    return ''

  def create_conversation(
      self,
      participants: tuple[str, ...],
      location: str,
      max_turns: int | None = None,
      terminate_check_min_turn: int | None = None,
      context: str = '',
  ) -> int:
    """Start a new conversation between co-located agents.

    Args:
      participants: Tuple of participant names in speaking order.
      location: Where the conversation is taking place.
      max_turns: Override default max turns for this conversation.
      terminate_check_min_turn: Minimum turns before boring-check kicks in.
        Defaults to DEFAULT_TERMINATE_CHECK_MIN_TURN (4). Set higher (e.g. 30)
        for scheduled dates to let them develop before the GM can cut them.
      context: Optional context string to pass to the conversation.

    Returns:
      The conversation ID.
    """
    with self._lock:
      for p in participants:
        if p in self._player_to_conv:
          existing_conv_id = self._player_to_conv[p]
          existing_conv = self._conversations.get(existing_conv_id)
          if existing_conv is not None and existing_conv.active:
            # Player is in an active conversation — don't create a new one.
            if context and context != '__GENERATING__':
              existing_conv.context = context
            return existing_conv_id
          else:
            # Stale mapping to an inactive/missing conversation — clean up.
            logging.warning(
                'Cleaned up stale conversation mapping for %s (conv %s)',
                p,
                existing_conv_id,
            )
            del self._player_to_conv[p]

      conv_id = self._next_conv_id
      self._next_conv_id += 1
      self._conversations[conv_id] = Conversation(
          conv_id=conv_id,
          participants=participants,
          location=location,
          utterances=[],
          turn=0,
          active=True,
          started_at=datetime.datetime.now().isoformat(),
          max_turns=max_turns if max_turns is not None else self._max_turns,
          terminate_check_min_turn=(
              terminate_check_min_turn
              if terminate_check_min_turn is not None
              else DEFAULT_TERMINATE_CHECK_MIN_TURN
          ),
          context=context,
      )
      for p in participants:
        self._player_to_conv[p] = conv_id
      return conv_id

  def update_context(self, conv_id: int, context: str) -> None:
    """Update context for an existing active conversation."""
    with self._lock:
      conv = self._conversations.get(conv_id)
      if conv is not None:
        conv.context = context

  def is_in_conversation(self, player_name: str) -> bool:
    """Check if player is currently in an active conversation.

    Also performs safety checks:
    - Cleans up stale mappings to inactive/missing conversations.
    - Detects orphaned participants (not all participants registered).
    - Force-ends conversations that exceed the wall-time timeout.

    Args:
      player_name: The name of the player to check.

    Returns:
      True if the player is in an active conversation, False otherwise.
    """
    with self._lock:
      conv_id = self._player_to_conv.get(player_name)
      if conv_id is None:
        return False
      conv = self._conversations.get(conv_id)
      if conv is None or not conv.active:
        # Stale mapping — clean up.
        logging.warning(
            'Cleaned up stale conversation mapping for %s (conv %s)',
            player_name,
            conv_id,
        )
        del self._player_to_conv[player_name]
        return False
      # Orphan detection: ensure all participants are still registered.
      for p in conv.participants:
        if self._player_to_conv.get(p) != conv_id:
          logging.warning(
              'Conversation %d has orphaned participant %s '
              '(partner %s missing from registry) — force-ending.',
              conv_id,
              player_name,
              p,
          )
          self._end_conversation_internal(conv_id)
          return False
      # Wall-time timeout.
      elapsed = time.monotonic() - conv.wall_start_time
      if elapsed > self._conversation_timeout:
        logging.warning(
            'Conversation %d [%s] timed out after %.0fs — force-ending.',
            conv_id,
            ', '.join(conv.participants),
            elapsed,
        )
        self._end_conversation_internal(conv_id)
        return False
      return True

  def is_my_turn(self, player_name: str) -> bool:
    """Check if it's this player's turn to speak.

    Args:
      player_name: The name of the player to check.

    Returns:
      True if player is in an active conversation and it's their turn.
    """
    with self._lock:
      conv_id = self._player_to_conv.get(player_name)
      if conv_id is None:
        return False
      conv = self._conversations.get(conv_id)
      if conv is None or not conv.active:
        return False
      return conv.participants[conv.turn] == player_name

  def add_utterance(self, player_name: str, text: str) -> bool:
    """Add an utterance if it's the player's turn.

    Args:
      player_name: Who is speaking.
      text: What they said.

    Returns:
      True if utterance was added (it was their turn), False otherwise.
    """
    with self._lock:
      conv_id = self._player_to_conv.get(player_name)
      if conv_id is None:
        return False
      conv = self._conversations.get(conv_id)
      if conv is None or not conv.active:
        return False
      if conv.participants[conv.turn] != player_name:
        return False
      if not text or not text.strip():
        return False

      conv.utterances.append((player_name, text))
      conv.turn = (conv.turn + 1) % len(conv.participants)

      if len(conv.utterances) >= conv.max_turns:
        conv.active = False
        self._end_conversation_internal(conv_id)

      return True

  def _end_conversation_internal(
      self, conv_id: int, *, for_tick: int | None = None
  ) -> None:
    """Internal method to end a conversation. Caller must hold lock."""
    conv = self._conversations.get(conv_id)
    if conv is None:
      return
    conv.active = False
    wall_duration = time.monotonic() - conv.wall_start_time
    logging.info(
        'Conversation %d [%s] at %s: %d turns, %.1fs wall time',
        conv_id,
        ', '.join(conv.participants),
        conv.location,
        len(conv.utterances),
        wall_duration,
    )
    group = frozenset(conv.participants)
    current_tick = self._get_current_tick()
    self._completed_pairs[group] = (
        for_tick if for_tick is not None else current_tick
    )
    for name in conv.participants:
      if self._player_to_conv.get(name) == conv_id:
        del self._player_to_conv[name]
    if for_tick is not None and current_tick != for_tick:
      logging.info(
          'AsyncConversationState: Suppressing clock notification for ended'
          ' conv %d: originating tick %d != current tick %d',
          conv_id,
          for_tick,
          current_tick,
      )
      return
    self._notify_clock_agents_acted(conv.participants)

  def _get_current_tick(self) -> int:
    if self._clock_key is None:
      return 0
    try:
      # `current_tick` is supplied by the concrete clock component, not by the
      # `BaseComponent` that `get_component` is declared to return.
      clock: Any = self.get_entity().get_component(self._clock_key)
      return clock.current_tick
    except (AttributeError, KeyError):
      return 0

  def _notify_clock_agents_acted(self, participants: tuple[str, ...]) -> None:
    if self._clock_key is None:
      return
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      for name in participants:
        clock.mark_agent_acted(name)
    except (AttributeError, KeyError):
      pass

  def is_pair_in_cooldown(
      self, agent_a: str, agent_b: str, cooldown_ticks: int = 2
  ) -> bool:
    with self._lock:
      pair = frozenset((agent_a, agent_b))
      if pair not in self._completed_pairs:
        return False
      completed_tick = self._completed_pairs[pair]
      current_tick = self._get_current_tick()
      return (current_tick - completed_tick) < cooldown_ticks

  def is_group_in_cooldown(
      self, agents: tuple[str, ...], cooldown_ticks: int = 2
  ) -> bool:
    with self._lock:
      current_tick = self._get_current_tick()
      agent_set = frozenset(agents)
      for completed_group, completed_tick in self._completed_pairs.items():
        if completed_group.issubset(agent_set):
          if (current_tick - completed_tick) < cooldown_ticks:
            return True
      return False

  def end_conversation(
      self, conv_id: int, *, for_tick: int | None = None
  ) -> None:
    with self._lock:
      self._end_conversation_internal(conv_id, for_tick=for_tick)

  def end_all_conversations(self, reason: str = '') -> None:
    """Force-ends all currently active conversations."""
    with self._lock:
      active_ids = [
          cid for cid, conv in self._conversations.items() if conv.active
      ]
      for cid in active_ids:
        if reason:
          logging.info(
              'AsyncConversationState: Force-ending conv %d: %s', cid, reason
          )
        self._end_conversation_internal(cid)

  def get_conversation_for(self, player_name: str) -> Conversation | None:
    """Get the conversation a player is in, if any."""
    with self._lock:
      conv_id = self._player_to_conv.get(player_name)
      if conv_id is None:
        return None
      return self._conversations.get(conv_id)

  def get_dialogue_context(self, player_name: str) -> str:
    """Get conversation context for player's observation.

    On first call for a conversation, returns the full preamble.
    On subsequent calls, returns exactly ONE new utterance per call so
    that each observation cycle delivers a single speech act.  The
    seen-index advances by one each time the method is called, ensuring
    a proper turn-by-turn cadence in the agent's observation stream.

    Args:
      player_name: The player requesting context.

    Returns:
      String describing the conversation state, or empty if not in one.
    """
    with self._lock:
      conv_id = self._player_to_conv.get(player_name)
      if conv_id is None:
        return ''
      conv = self._conversations.get(conv_id)
      if conv is None:
        return ''

      others = [p for p in conv.participants if p != player_name]
      others_str = ', '.join(others)

      if not conv.active:
        return 'The conversation has ended.'

      seen_key = f'{player_name}_{conv_id}'

      if seen_key not in self._last_seen_index:
        self._last_seen_index[seen_key] = -1
        lines = []
        if conv.context:
          lines.append(conv.context)
        else:
          lines.append(
              f'You are having a conversation with {others_str}'
              f' at {conv.location}.'
          )
        return '\n'.join(lines)

      # Deliver exactly one unseen utterance per call.
      last_seen = self._last_seen_index[seen_key]
      next_index = last_seen + 1
      lines = []
      if next_index < len(conv.utterances):
        speaker, text = conv.utterances[next_index]
        lines.append(f'{speaker}: "{text}"')
        self._last_seen_index[seen_key] = next_index
      # else: no new utterances — player is caught up.

      if not conv.active:
        lines.append('The conversation has ended.')

      return '\n'.join(lines)

  def get_transcript(self, conv_id: int) -> str:
    """Get full transcript of a conversation."""
    with self._lock:
      conv = self._conversations.get(conv_id)
      if conv is None:
        return ''
      participant_str = ', '.join(conv.participants)
      lines = [f'Conversation at {conv.location} between {participant_str}:']
      for speaker, text in conv.utterances:
        lines.append(f'  {speaker}: "{text}"')
      return '\n'.join(lines)

  def get_active_conversations(self) -> list[Conversation]:
    """Return all currently active conversations."""
    with self._lock:
      return [c for c in self._conversations.values() if c.active]

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      convs_state = {}
      for cid, conv in self._conversations.items():
        convs_state[str(cid)] = dataclasses.asdict(conv)
      return {
          'conversations': convs_state,
          'next_conv_id': self._next_conv_id,
          'player_to_conv': dict(self._player_to_conv),
          'players': list(self._player_names),
          'last_seen_index': dict(self._last_seen_index),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._conversations = {}
      convs_data = state.get('conversations', {})
      if isinstance(convs_data, dict):
        for cid_str, conv_data in convs_data.items():
          if not isinstance(conv_data, dict):
            continue
          fields: dict[str, Any] = {str(k): v for k, v in conv_data.items()}
          missing = [
              name
              for name in ('conv_id', 'participants', 'location')
              if name not in fields
          ]
          if missing:
            logging.warning(
                'AsyncConversationState.set_state: skipping conversation %s, '
                'checkpoint record is missing %s',
                cid_str,
                ', '.join(missing),
            )
            continue
          fields['participants'] = tuple(fields['participants'])
          fields['utterances'] = [
              tuple(u) for u in fields.get('utterances', [])
          ]
          self._conversations[int(str(cid_str))] = Conversation(**fields)
      next_id = state.get('next_conv_id', 0)
      self._next_conv_id = (
          int(next_id) if isinstance(next_id, (int, float)) else 0
      )
      player_to_conv_raw = state.get('player_to_conv', {})
      if isinstance(player_to_conv_raw, dict):
        self._player_to_conv = {
            str(k): int(v)
            for k, v in player_to_conv_raw.items()
            if isinstance(v, (int, float))
        }
      else:
        self._player_to_conv = {}
      # Restore dialogue delivery tracking so scene-setup is never
      # re-delivered after a checkpoint round-trip.
      last_seen_raw = state.get('last_seen_index', {})
      if isinstance(last_seen_raw, dict):
        self._last_seen_index = {
            str(k): int(v)
            for k, v in last_seen_raw.items()
            if isinstance(v, (int, float))
        }
      else:
        self._last_seen_index = {}


class AsyncConversationObservation(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Returns dialogue context to players in conversations.

  Registered under __make_observation__ key so SwitchAct uses it for
  MAKE_OBSERVATION. Accesses ConversationState via get_entity().get_component().
  """

  def __init__(
      self,
      async_conversation_key: str = DEFAULT_ASYNC_CONVERSATION_KEY,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      pre_act_label: str = '\nConversation Context',
  ):
    super().__init__()
    self._async_conversation_key = async_conversation_key
    self._call_to_make_observation = call_to_make_observation
    self._pre_act_label = pre_act_label

  def _get_conversation_state(self) -> AsyncConversationState:
    return self.get_entity().get_component(
        self._async_conversation_key, type_=AsyncConversationState
    )

  def _extract_player_name(self, call_to_action: str) -> str:
    """Extract player name from the call to action string."""
    prefix, suffix = self._call_to_make_observation.split('{name}')
    if not call_to_action.startswith(prefix):
      return ''
    if not call_to_action.endswith(suffix):
      return ''
    return call_to_action.removeprefix(prefix).removesuffix(suffix)

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
      player_name = self._extract_player_name(action_spec.call_to_action)
      if player_name:
        conv_state = self._get_conversation_state()
        result = conv_state.get_dialogue_context(player_name)

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result[:100] if result else '',
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


# Minimum turn count before the GM starts checking for ending conversations.
_TERMINATE_CHECK_MIN_TURN = 4

_TERMINATE_NOTE = (
    'Any conversation that becomes repetitive always ends immediately. '
    'When participants have said goodbye, finished their exchange, or '
    'stated an intention to leave, the conversation is over.'
)


class AsyncConversationResolution(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Resolves speech actions in conversations.

  Registered under __resolution__ key so SwitchAct uses it for RESOLVE.
  Parses the putative action to extract speaker and utterance, then adds
  to the conversation if it's the speaker's turn.

  Asks the GM (via an LLM call or exit intent check) whether the
  conversation has ended or become repetitive, terminating it early if so.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      model: language_model.LanguageModel | None = None,
      terminate_boring: bool = True,
      async_conversation_key: str = DEFAULT_ASYNC_CONVERSATION_KEY,
      pre_act_label: str = DEFAULT_CONVERSATION_PRE_ACT_LABEL,
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._model = model
    self._terminate_boring = terminate_boring and model is not None
    self._async_conversation_key = async_conversation_key
    self._pre_act_label = pre_act_label
    self._pending_events: dict[str, str] = {}

  def _get_conversation_state(self) -> AsyncConversationState:
    return self.get_entity().get_component(
        self._async_conversation_key, type_=AsyncConversationState
    )

  def _extract_speaker_and_utterance(
      self, action_text: str
  ) -> tuple[str | None, str | None]:
    """Extract speaker name and utterance from action text."""
    text = action_text.strip()
    for name in self._player_names:
      if text.startswith(f'{name}:'):
        return name, text[len(f'{name}:') :].strip()
      if text.startswith(f'{name} '):
        remainder = text[len(f'{name} ') :]
        if remainder.startswith('--'):
          remainder = remainder[2:]
        elif remainder.startswith('said:'):
          remainder = remainder[5:]
        elif remainder.startswith('says:'):
          remainder = remainder[5:]
        return name, remainder.strip()
    return None, text

  @classmethod
  def _is_marketplace_json(cls, text: str) -> bool:
    """Return True if text is a structured marketplace JSON action spec."""
    if not text:
      return False
    stripped = text.strip().strip('"').strip("'").strip()
    if stripped.startswith('{') and stripped.endswith('}'):
      return any(
          f'"action": "{act}"' in stripped or f'"action":"{act}"' in stripped
          for act in ('bid', 'save', 'withdraw', 'pass')
      )
    return False

  def _has_exit_intent(self, text: str) -> bool:
    return bool(_EXIT_REGEX.search(text))

  def _should_terminate_conversation(self, conv: Conversation) -> bool:
    """Ask the GM whether the conversation has ended or become repetitive.

    Follows the Dialogic Game Master pattern from third_party Concordia:
    Checks if participants have said goodbye / departed, or asks the GM LLM
    whether the conversation is finished.

    Args:
      conv: The conversation to check.

    Returns:
      True if the conversation should be terminated, False otherwise.
    """
    if not self._terminate_boring:
      return False
    turn_count = len(conv.utterances)
    if turn_count < 2:
      return False

    _, last_text = conv.utterances[-1]
    has_exit = self._has_exit_intent(last_text)

    # If recent utterances have exit intent, check on every turn.
    # Otherwise check every other turn once turn_count >= min_turns.
    min_turn = getattr(
        conv, 'terminate_check_min_turn', _TERMINATE_CHECK_MIN_TURN
    )
    if not has_exit and (turn_count < min_turn or turn_count % 2 != 0):
      return False

    # Check if both participants in a 2-person conversation expressed exit
    # intent.
    if len(conv.participants) == 2 and len(conv.utterances) >= 2:
      p1_recent = any(
          self._has_exit_intent(txt)
          for spk, txt in conv.utterances[-4:]
          if spk == conv.participants[0]
      )
      p2_recent = any(
          self._has_exit_intent(txt)
          for spk, txt in conv.utterances[-4:]
          if spk == conv.participants[1]
      )
      if p1_recent and p2_recent:
        logging.info(
            'Both participants [%s] expressed exit intent; GM ending conv %d at'
            ' turn %d',
            ', '.join(conv.participants),
            conv.conv_id,
            turn_count,
        )
        return True

    if self._model is None:
      return False

    # Build a short transcript of the last few utterances
    recent = conv.utterances[-6:]  # last 6 for context
    transcript = '\n'.join(f'{speaker}: "{text}"' for speaker, text in recent)

    doc = interactive_document.InteractiveDocument(self._model)
    doc.statement(f'Note: {_TERMINATE_NOTE}')
    doc.statement(
        'The following is a conversation between'
        f' {", ".join(conv.participants)} at {conv.location}:'
    )
    doc.statement(transcript)
    terminate_options = [
        'Yes, the conversation is over',
        'No, the conversation will continue',
    ]
    choice = doc.multiple_choice_question(
        question='Is the conversation finished?',
        answers=terminate_options,
        randomize_choices=False,
    )

    should_end = choice == 0  # 'Yes, the conversation is over'
    if should_end:
      logging.info(
          'GM terminated conversation %d [%s] at turn %d (dialogic GM check:'
          ' finished)',
          conv.conv_id,
          ', '.join(conv.participants),
          turn_count,
      )
    return should_end

  def pre_observe(self, observation: str) -> str:
    if PUTATIVE_EVENT_TAG in observation:
      tag_end = observation.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
      raw = observation[tag_end:].strip()
      for name in self._player_names:
        if raw.startswith(name):
          self._pending_events[name] = observation
          return ''
      self._pending_events['__unknown__'] = observation
    return ''

  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    result = ''
    if action_spec.output_type == entity_lib.OutputType.RESOLVE:
      action_text = None
      if self._pending_events:
        gm = self.get_entity()
        active_entity_name = getattr(gm, '_active_capture_key', None)

        selected = None
        if active_entity_name:
          selected = self._pending_events.pop(active_entity_name, None)

        if not selected and '__unknown__' in self._pending_events:
          selected = self._pending_events.pop('__unknown__')

        if selected:
          tag_pos = selected.find(PUTATIVE_EVENT_TAG) + len(PUTATIVE_EVENT_TAG)
          action_text = selected[tag_pos:]

      if action_text:
        speaker, utterance = self._extract_speaker_and_utterance(action_text)

        if speaker is not None and utterance:
          # Reject marketplace JSON bids from being stored as conversation
          # utterances — same guard as IslandEventResolution._resolve_speech.
          if self._is_marketplace_json(utterance):
            logging.info(
                'Rejected marketplace JSON as speech for %s: %s',
                speaker,
                utterance[:80],
            )
          else:
            conv_state = self._get_conversation_state()
            if conv_state.is_in_conversation(speaker):
              if conv_state.is_my_turn(speaker):
                conv_state.add_utterance(speaker, utterance)
                result = f'{speaker} said: "{utterance}"'

                # Check if GM wants to end the conversation
                conv = conv_state.get_conversation_for(speaker)
                if conv is not None and conv.active:
                  if self._should_terminate_conversation(conv):
                    conv_state.end_conversation(conv.conv_id)
              else:
                result = f'{speaker} is waiting for their turn to speak.'
            else:
              result = f'{speaker}: {utterance}'

    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result[:100] if result else '',
        'Value': result,
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass


class AsyncConversationTrigger(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Triggers conversations between co-located agents.

  This component checks for agents at the same location and calls
  create_conversation() on AsyncConversationState to start dialogues.

  The trigger runs in post_act() after each GM act step. It checks
  co-location and creates conversations for pairs of free agents.
  Agents can have multiple conversations throughout the day — pairs
  are only blocked while their conversation is active.

  Usage:
    Add this component to your GM after AsyncConversationState and Locations.
    It will automatically trigger conversations when agents are co-located.
  """

  def __init__(
      self,
      player_names: Sequence[str],
      async_conversation_key: str = DEFAULT_ASYNC_CONVERSATION_KEY,
      locations_key: str = 'locations',
      clock_key: str = 'clock',
      min_agents: int = 2,
      max_conversation_size: int = 0,
      cooldown_ticks: int = 2,
      pre_act_label: str = '\nConversation Trigger',
  ):
    super().__init__()
    self._player_names = list(player_names)
    self._async_conversation_key = async_conversation_key
    self._locations_key = locations_key
    self._clock_key = clock_key
    self._min_agents = min_agents
    self._max_conversation_size = max_conversation_size
    self._cooldown_ticks = cooldown_ticks
    self._pre_act_label = pre_act_label
    self._step_counter = 0
    self._last_event = ''

  def _get_conversation_state(self) -> AsyncConversationState | None:
    try:
      return self.get_entity().get_component(
          self._async_conversation_key, type_=AsyncConversationState
      )
    except (AttributeError, KeyError):
      return None

  def _get_locations(self) -> dict[str, str]:
    try:
      locations = self.get_entity().get_component(self._locations_key)
      if locations is not None:
        state = locations.get_state()
        entity_locations = state.get('entity_locations')
        if isinstance(entity_locations, dict):
          return {str(k): str(v) for k, v in entity_locations.items()}
    except (AttributeError, KeyError):
      pass
    return {}

  def _check_and_trigger(self) -> str:
    conv_state = self._get_conversation_state()
    if conv_state is None:
      return ''

    # Don't trigger new conversations on the very first tick of a new day.
    # The island GM is about to hand off to the marketplace GM after the
    # overnight skip.  Any conversations started here would collide with
    # the marketplace action specs and leak JSON bids into speech.
    try:
      clock: Any = self.get_entity().get_component(self._clock_key)
      current_dt = clock._current_dt  # pylint: disable=protected-access
      if (
          hasattr(clock, '_waking_start')
          and current_dt.hour == getattr(clock, '_waking_start')
      ):
        if not hasattr(clock, 'is_nighttime_completed') or not (
            clock.is_nighttime_completed(current_dt.day)
            or clock.is_nighttime_completed(current_dt.day - 1)
        ):
          logging.info(
              'ConvTrigger: skipping at %s (marketplace not yet complete)',
              current_dt,
          )
          return ''
    except (AttributeError, KeyError):
      pass

    entity_locations = self._get_locations()
    if not entity_locations:
      return ''

    location_to_agents: dict[str, list[str]] = {}
    for name, loc in entity_locations.items():
      if name in self._player_names and loc:
        location_to_agents.setdefault(loc, []).append(name)

    triggered = []
    for loc, agents_at_loc in location_to_agents.items():
      if len(agents_at_loc) >= self._min_agents:
        free_agents = [
            a for a in agents_at_loc if not conv_state.is_in_conversation(a)
        ]
        if len(free_agents) < self._min_agents:
          continue

        if self._max_conversation_size > 0:
          group = tuple(free_agents[: self._max_conversation_size])
        else:
          group = tuple(free_agents)

        if not conv_state.is_group_in_cooldown(group, self._cooldown_ticks):
          conv_state.create_conversation(group, loc)
          names_str = ', '.join(group)
          triggered.append(f'{names_str} at {loc}')
          logging.info(
              'ConvTrigger: started conversation with %s at %s',
              names_str,
              loc,
          )

    self._step_counter += 1

    if triggered:
      return f'Started {len(triggered)} conversations: {triggered}'
    return ''

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    return ''

  def post_act(self, event: str) -> str:
    self._last_event = event
    return ''

  def update(self) -> None:
    result = self._check_and_trigger()
    self._logging_channel({
        'Key': self._pre_act_label,
        'Summary': result[:100] if result else 'No new conversations',
        'Value': result,
    })

  def get_state(self) -> entity_component.ComponentState:
    return {
        'step_counter': self._step_counter,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._step_counter = component_state.as_int(state, 'step_counter', 0)
