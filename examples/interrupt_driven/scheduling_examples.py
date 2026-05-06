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

r"""Interrupt-driven game master: worked examples.

This file is a runnable tutorial demonstrating how the interrupt-driven
game master orchestrates entities inside the Concordia Simultaneous
engine.  Run it directly to watch the simulation unfold::

    python3 -m concordia.examples.interrupt_driven.scheduling_examples

The output shows every step of every scenario: which event fired, who
got polled, what each entity said, and what mask and timer they set.
If you change a mask, add an event, or swap a scripted response, the
output will change accordingly --- try it and see!

How to read the output
----------------------
On the first step of each scenario, the trace lists any initial events
and entity states before showing the step itself.  After that, each
step shows the time, the event that fired, who was polled, what each
polled entity said, and the resolution.  A typical first step looks
like this::

    ─── Step 1 ──────────────────────────────────────────
      ── Initial event queue ──
        [09:00] announcement.special (from barista)
               "The barista announces the daily
               special: iced lavender latte."

      ── Initial entity states ──
        Alice:  mask=[emergency.]  timer=09:30 (journal ...)
        Bob:  mask=[""]  (matches all events)  timer=09:15 (...)

      Time:   09:00
      Event:  announcement.special  (from barista)
              "The barista announces the daily special:
              iced lavender latte."
      Polled: Bob
      Bob responded:
        action: I notice the barista making an
                announcement about the daily special.
        mask:   [""]  (matches all events)
        timer:  15m  (glance around again)
      Resolved: Bob

   When an entity uses an absolute time for its timer (via the
   ``"until"`` key), the trace shows it like this::

         timer:  until 15:00  (exam study session ends)

The ``mask:`` field shows the entity's interrupt-mask prefix list.
``[""]`` is the match-all mask (the empty-string prefix matches every
tag), and ``[]`` matches nothing --- only timer expiries can
interrupt.  A list like ``[emergency.]`` matches only event tags
starting with that prefix.  If an entity emits extra event tags, they
appear in a ``tags:`` line.  The parenthetical annotations (e.g.
``(matches all events)``) are printed automatically whenever the
mask is ``[""]`` or ``[]``.

The trace is generated from the simulation state, not from static
strings.  If Alice's mask were ``[""]`` instead of ``["emergency."]``,
you would see her name appear in ``Polled:`` too.

Architecture recap
------------------
1. Entities are built with ``ConcatActComponent`` configured with
   ``prefix_entity_name=False``, so responses do not include the entity
   name prefix.

2. The game master is created via
   ``interrupt_driven.GameMaster(entities=players).build(...)``.
   Internally it holds an ``EntityScheduler`` plus three inline
   components (``_InterruptNextActing``, ``_InterruptMakeObservation``,
   ``_InterruptResolution``) implementing the engine protocol.

3. The Simultaneous engine loop calls, in order:
   - ``gm.act(NEXT_ACTING)`` → pick next event/timer, return names
   - ``gm.act(MAKE_OBSERVATION)`` → build observation for each entity
   - each entity acts (``sample_text``)
   - ``gm.act(RESOLVE)`` → parse JSON, update scheduler

Since we want deterministic runs, ``ScriptedModel`` maps entity names
to queued responses (no LLM required).
"""

from collections.abc import Collection, Mapping, Sequence

from typing import Any, override

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.environment.engines import simultaneous
from examples.interrupt_driven import log_printing
from concordia.language_model import language_model
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import interrupt_driven
import numpy as np


# ─────────────────────────────────────────────────────────────
# Embedder stub (required by AssociativeMemoryBank)
# ─────────────────────────────────────────────────────────────


def _embedder(text: str):
  """Trivial embedder that returns a random 3-d vector."""
  del text
  return np.random.rand(3)


# ─────────────────────────────────────────────────────────────
# ScriptedModel — deterministic responses for demos
# ─────────────────────────────────────────────────────────────


class ScriptedModel(no_language_model.NoLanguageModel):
  r"""Language model that returns pre-programmed strings.

  Each entity name maps to a FIFO queue of responses.  On every call
  to ``sample_text``, the model inspects the prompt to figure out
  *which* entity is currently acting.  It looks for the entity name
  in the call-to-action line, which ``ConcatActComponent`` renders as
  ``"Exercise: What does <EntityName> do next? ..."``

  If the queue is empty --- or the entity name cannot be determined
  --- a safe default is returned that keeps the simulation ticking:
  empty mask (match nothing) + 1 h idle timer.  The empty mask
  ensures the entity stops being polled for subsequent events,
  breaking the action-event cascade and allowing time to advance.

  Usage::

      model = ScriptedModel({
          'Alice': [
              'Alice waves.\n'
              '{"mask": [""], "timer": {"time": "30m", "reason": "x"}}',
          ],
          'Bob': [
              'Bob nods.\n'
              '{"mask": [""], "timer": {"time": "15m", "reason": "y"}}',
          ],
      })
  """

  _DEFAULT_RESPONSE = '{"mask": [], "timer": {"time": "1h", "reason": "idle"}}'

  def __init__(
      self,
      responses: dict[str, list[str]] | None = None,
  ):
    super().__init__()
    # Deep-copy the lists so the caller's original is not mutated.
    self._queues: dict[str, list[str]] = {
        name: list(queue) for name, queue in (responses or {}).items()
    }

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = (language_model.DEFAULT_TERMINATORS),
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    entity_name = self._extract_entity_name(prompt)
    if entity_name and entity_name in self._queues:
      queue = self._queues[entity_name]
      if queue:
        return queue.pop(0)
    return self._DEFAULT_RESPONSE

  # ── private helpers ──────────────────────────────────────

  def _extract_entity_name(self, prompt: str) -> str | None:
    """Finds the entity name in the ConcatActComponent prompt.

    The call-to-action ``DEFAULT_CALL_TO_ACTION`` uses the phrase
    ``"What does {name} do next?"`` which ``ConcatActComponent``
    renders with the real entity name.  We look for that pattern.

    As a fallback (e.g. when ``prefix_entity_name=True``), we also
    check for ``"Answer: <name> "`` at the end of the prompt.

    Args:
      prompt: The full prompt text.

    Returns:
      The entity name, or None if not found.
    """
    # Primary: look for "What does <Name> do next?" in the call-to-action.
    cta_marker = 'What does '
    cta_idx = prompt.rfind(cta_marker)
    if cta_idx != -1:
      after_marker = prompt[cta_idx + len(cta_marker) :]
      # The name is everything up to " do next?"
      do_idx = after_marker.find(' do next?')
      if do_idx != -1:
        candidate = after_marker[:do_idx].strip()
        if candidate and candidate in self._queues:
          return candidate

    # Fallback: look for "Answer: <name>" (works with prefix_entity_name=True).
    marker = 'Answer: '
    idx = prompt.rfind(marker)
    if idx != -1:
      after = prompt[idx + len(marker) :]
      name = after.split()[0] if after.strip() else None
      if name and name in self._queues:
        return name

    return None


# ─────────────────────────────────────────────────────────────
# Scenario runner helper
# ─────────────────────────────────────────────────────────────


def _run_scenario(
    player_names: Sequence[str],
    responses: dict[str, list[str]],
    start_time: str = '2026-01-01T09:00:00',
    initial_events: Sequence[dict[str, str]] | None = None,
    initial_entity_states: (
        dict[str, dict[str, str]] | None
    ) = None,
    max_steps: int = 3,
) -> list[Mapping[str, Any]]:
  """Set up and run an interrupt-driven scenario.

  Returns the structured engine log so the caller can inspect it
  or print it with ``_print_log()``.

  Args:
    player_names: Names of the participant entities.
    responses: Per-entity scripted model responses.
    start_time: ISO-format start time for the simulation.
    initial_events: Optional list of dicts with keys ``timestamp`` (ISO),
      ``tag``, ``source``, ``description`` to inject into the event queue before
      the simulation starts.
    initial_entity_states: Optional dict mapping entity names to their starting
      configuration.  Each value is a dict with optional keys:  - ``mask``: A
      list of prefix strings (same semantics as the ``mask`` field in entity
      JSON responses, e.g. ``['']`` for match-all, ``['emergency.']`` for one
      prefix, ``[]`` for match-none).  Default: ``['']``. - ``timer_duration``:
      A duration string (e.g. ``'30m'``, ``'1h'``) for the entity's initial
      timer.  Default: unset (timer fires at start).
      timer'``.  When provided, these replace the default initial timer (which
      fires at ``start_time`` and would consume a scripted response just to
      bootstrap).  This lets scenarios start entities in a known state without a
      bootstrap step.
    max_steps: Number of engine ticks to run.

  Returns:
    The engine log (a list of per-step dicts).
  """
  # ── 1. Language model ────────────────────────────────────
  model = ScriptedModel(responses)

  # ── 2. Player entities ───────────────────────────────────
  players = []
  for name in player_names:
    act = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        prefix_entity_name=False,
    )
    player = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act,
        context_components={},
    )
    players.append(player)

  # ── 3. Game master ───────────────────────────────────────
  params = {
      'name': 'interrupt_driven_rules',
      'start_time': start_time,
      'initial_events': list(initial_events or []),
      'initial_entity_states': initial_entity_states or {},
  }
  gm_config = interrupt_driven.GameMaster(
      entities=players,
      params=params,
  )
  memory_bank = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=_embedder,
  )
  gm = gm_config.build(model=model, memory_bank=memory_bank)

  # ── 4. Run the engine, collecting a log ──────────────────
  engine = simultaneous.Simultaneous()
  log: list[Mapping[str, Any]] = []
  engine.run_loop(
      game_masters=[gm],
      entities=players,
      max_steps=max_steps,
      log=log,
  )

  return log


# ═════════════════════════════════════════════════════════════
# Example 1 — Coffee Shop
# ═════════════════════════════════════════════════════════════


def coffee_shop_example() -> list[Mapping[str, Any]]:
  """Two friends at a coffee shop with different attention.

  Alice is writing in her journal.  She's in deep focus and only
  responds to emergencies (mask: ``["emergency."]``).

  Bob is people-watching (mask: ``[""]`` — everything).

  Both start pre-configured with these masks and timers set 30 min
  into the future (via ``initial_entity_states``).  A barista
  ``announcement.special`` event is injected at 09:00 --- the very
  start of the simulation.  Since ``announcement.special`` does not
  start with ``emergency.``, Alice's mask won't match it, but Bob's
  ``[""]`` (match-all) will.

  Step 1 shows **only Bob** being polled for the barista event.
  Later, when Alice's timer expires, she reappears with her own
  response.

  **Things to try:**

  - Change Alice's initial mask from ``["emergency."]`` to ``[""]``
    and re-run.  You should see Alice appear in ``Polled:`` for
    step 1.
  - Remove the initial event entirely.  The first step should then
    be Bob's timer expiry at 09:15.
  - Add a second event with tag ``emergency.fire`` and see that
    *both* Alice and Bob get polled for it.

  Returns:
    The engine log.
  """
  initial_entity_states = {
      'Alice': {
          'mask': ['emergency.'],
          'timer_duration': '30m',
          'timer_description': 'journal entry halfway done',
      },
      'Bob': {
          'mask': [''],
          'timer_duration': '15m',
          'timer_description': 'glance around again',
      },
  }
  # Scripted responses for each entity.
  alice_responses = [
      (
          # Alice's timer expires, but she chooses to continue writing with
          # the same intensity as before. She will be done in 30 minutes and
          # will then decide what to do next.
          'Alice continues writing in the journal.\n'
          '{"mask": ["emergency."], "timer":'
          ' {"time": "30m", "reason": "finish journal entry"}}'
      ),
      (
          # This is the exciting cliffhanger to our simulated simulation.
          'Alice beams back up to the Enterprise, disappearing in a veil of '
          'shimmering light.\n'
          '{"mask": ["klingon.", "wesley."], "timer":'
          ' {"time": "180m", "reason": "party ends in Ten Forward"}}'
      ),
  ]
  bob_responses = [
      (
          # Bob notices the barista and sets a 15-min timer. Before this he
          # is apparently doing constant, un-interruptable people-watching.
          # In 15 minutes, barring interruptions, his timer will expire and he
          # will need to decide what to do next.
          'Bob notices the barista making an announcement'
          ' about the daily special.\n'
          '{"mask": [""], "timer":'
          ' {"time": "15m", "reason": "glance around again"}}'
      ),
      (
          # Bob's timer expires; he glances around the shop. Note that the
          # timer he now sets will never fire, since Alice's "I continue
          # writing" action will interrupt Bob. Bob will have to react and
          # set a new timer.
          'Bob looks around the coffee shop and sips the drink.\n'
          '{"mask": [""], "timer":'
          ' {"time": "20m", "reason": "keep people-watching (never fires!)"}}'
      ),
      (
          # Here is Bob's reaction to seeing Alice journalling. He notices it
          # and then decides to keep watching people. With his timer choice,
          # he will decide in 20 minutes what to do next unless he is
          # interrupted by an event before that.
          'Bob sees Alice journalling and continues people-watching.\n'
          '{"mask": [""], "timer"'
          ' {"time": "20m", "reason": "people-o-rama"}}'
      )
  ]

  return _run_scenario(
      player_names=['Alice', 'Bob'],
      initial_entity_states=initial_entity_states,
      responses={'Alice': alice_responses, 'Bob': bob_responses},
      start_time='2026-06-15T09:00:00',
      initial_events=[{
          'timestamp': '2026-06-15T09:00:00',
          'tag': 'announcement.special',
          'source': 'barista',
          'description': (
              'The barista announces the daily special: iced lavender latte.'
          ),
      }],
      max_steps=5,
  )


# ═════════════════════════════════════════════════════════════
# Example 2 — Kitchen Timer
# ═════════════════════════════════════════════════════════════


def kitchen_timer_example() -> list[Mapping[str, Any]]:
  """Cross-entity signalling via ``tags`` and timer discarding.

  Chef is cooking pasta (mask: ``["kitchen."]``, timer: 30 min).
  Reader is reading a novel (mask: ``["food."]``, timer: 15 min for
  first hunger pangs).

  The scenario demonstrates two key features of the event system:

  1. **Action tagging** — Chef tags the dinner announcement with
     ``"tags": ["food.ready"]``.  This creates an extra event whose
     tag Reader's ``["food."]`` mask matches, even though the
     default ``action.Chef`` tag would not.

  2. **Timer discarding on interrupt** — When Reader is polled by
     the ``food.ready`` event, ``poll_entity`` clears Reader's
     pending timer ("give up and go to a falafel shop").  That timer
     never fires.  This is visible in the trace: you will never see
     a timer expiry for "falafel shop".

  Step-by-step:

  1. *18:15* — Reader's 15 min timer fires (first hunger pang).
     Reader stretches, keeps reading.  Sets mask ``["food."]`` and
     a 20 min timer for "give up and go to falafel shop".
  2. *18:30* — Chef's 30 min timer fires (pasta done).  Chef
     announces dinner with ``"tags": ["food.ready"]``.  Sets 40 min
     timer for cleanup.
  3. *18:30* — The ``food.ready`` event fires (same timestamp).
     Reader's mask ``["food."]`` matches.  Reader is polled — the
     pending "falafel shop" timer is **discarded**.  Reader
     starts eating, sets 30 min timer.
  4. *19:00* — Reader's 30 min timer fires (meal done).  Reader
     dozes in the easy chair.  Sets 1 h timer.
  5. *19:10* — Chef's 40 min timer fires (cleanup time).  Chef
     cleans up, sets 20 min timer.

  **Things to try:**

  - Remove the ``"tags"`` entry from Chef's response.  Reader
    should no longer be polled by Chef's announcement — the
    "falafel shop" timer fires instead.
  - Change Reader's initial mask to ``[""]``.  Reader should be
    polled by Chef's ``action.Chef`` event even without tags.

  Returns:
    The engine log.
  """
  initial_entity_states = {
      'Chef': {
          'mask': ['kitchen.'],
          'timer_duration': '30m',
          'timer_description': 'pasta is ready',
      },
      'Reader': {
          'mask': ['food.'],
          'timer_duration': '15m',
          'timer_description': 'feeling first hunger pang',
      },
  }
  chef_responses = [
      (  # Chef's 30-min timer fires → announce dinner.
          'The pasta is done! Chef calls out: Dinner is ready!\n'
          '{"mask": ["kitchen."], "tags": ["food.ready"],'
          ' "timer": {"time": "40m", "reason": "cleanup after dinner"}}'
      ),
      (  # Chef's 40-min timer fires → cleanup.
          'Time to clean up the kitchen.\n'
          '{"mask": [""], "timer":'
          ' {"time": "20m", "reason": "finish cleaning"}}'
      ),
  ]
  reader_responses = [
      (  # Reader's 15-min timer fires → first hunger pang.
          'Reader stretches and yawns. Getting hungry,'
          " but Reader will keep reading for now.\n"
          '{"mask": ["food."], "timer":'
          ' {"time": "20m", "reason": "give up and go to falafel shop"}}'
      ),
      (  # food.ready event interrupts → start eating.
          'Reader hears that dinner is ready!'
          ' Reader puts down the novel and heads to the kitchen.\n'
          '{"mask": [], "timer":'
          ' {"time": "30m", "reason": "finish eating"}}'
      ),
      (  # Reader's 30-min timer fires → meal done, doze.
          'That was delicious.'
          ' Reader returns to the easy chair and dozes off.\n'
          '{"mask": [], "timer":'
          ' {"time": "1h", "reason": "wake up from nap"}}'
      ),
  ]

  return _run_scenario(
      player_names=['Chef', 'Reader'],
      initial_entity_states=initial_entity_states,
      responses={
          'Chef': chef_responses,
          'Reader': reader_responses,
      },
      start_time='2026-03-14T18:00:00',
      max_steps=5,
  )


# ═════════════════════════════════════════════════════════════
# Example 3 — Morning Routine
# ═════════════════════════════════════════════════════════════


def morning_routine_example() -> list[Mapping[str, Any]]:
  """Three housemates with staggered timers and mask gaps.

  All three start pre-configured (via ``initial_entity_states``):

  - Alex: showering, ignores everything (mask: ``[]`` → MATCH_NONE),
    timer 30 min (finishes at 07:30).
  - Bailey: making coffee, listens for ``chat.`` events (mask:
    ``["chat."]``), timer 15 min (coffee brews at 07:15).
  - Casey: asleep, ignores everything (mask: ``[]`` → MATCH_NONE),
    timer 1 hour (wakes up at 08:00).

  A ``chat.group`` event is injected at 07:00.  Only Bailey's mask
  matches, so step 1 shows just Bailey being polled.  Then timers
  fire in order: Bailey (07:15) → Alex (07:30) → Casey (08:00).

  **Things to try:**

  - Give Casey a ``["chat."]`` mask.  Casey should then be polled
    alongside Bailey when the chat.group event fires.
  - Remove Bailey's ``["chat."]`` mask (set to ``[]``).  The first
    step should become a timer expiry for Bailey (the earliest
    timer) instead of a chat event match.

  Returns:
    The engine log.
  """
  initial_entity_states = {
      'Alex': {
          'mask': [],
          'timer_duration': '30m',
          'timer_description': 'finish shower',
      },
      'Bailey': {
          'mask': ['chat.'],
          'timer_duration': '15m',
          'timer_description': 'coffee is brewed',
      },
      'Casey': {
          'mask': [],
          'timer_duration': '1h',
          'timer_description': 'wake up',
      },
  }
  alex_responses = [
      (  # Alex's 30-min timer: finishes shower.
          'Alex gets out of the shower and gets dressed.\n'
          '{"mask": ["chat."], "timer":'
          ' {"time": "30m", "reason": "head to work"}}'
      ),
  ]
  bailey_responses = [
      (  # Bailey sees the chat event at 07:00.
          'Bailey reads the group chat and sips coffee.\n'
          '{"mask": ["chat."], "timer":'
          ' {"time": "15m", "reason": "coffee is brewed"}}'
      ),
      (  # Bailey's 15-min timer: coffee is done.
          'Bailey pours a fresh cup of coffee.\n'
          '{"mask": ["chat."], "timer":'
          ' {"time": "30m", "reason": "second cup"}}'
      ),
  ]
  casey_responses = [
      (  # Casey's 1-hour timer: wakes up.
          'Casey wakes up groggily and checks the phone.\n'
          '{"mask": [""], "timer":'
          ' {"time": "30m", "reason": "get ready"}}'
      ),
  ]

  return _run_scenario(
      player_names=['Alex', 'Bailey', 'Casey'],
      initial_entity_states=initial_entity_states,
      responses={
          'Alex': alex_responses,
          'Bailey': bailey_responses,
          'Casey': casey_responses,
      },
      start_time='2026-07-04T07:00:00',
      initial_events=[{
          'timestamp': '2026-07-04T07:00:00',
          'tag': 'chat.group',
          'source': 'Group Chat Weatherbot',
          'description': 'Good morning! Sunny day today, high of 23C.',
      }],
      max_steps=6,
  )


# ═════════════════════════════════════════════════════════════
# Example 4 — Study Hall
# ═════════════════════════════════════════════════════════════


def study_hall_example() -> list[Mapping[str, Any]]:
  """Absolute-time timers with ``"until"``.

  Maya is studying for an exam.  She sets her timer with
  ``"until": "15:00"`` — an absolute clock time rather than a
  relative duration.  Leo is working on a problem set with ordinary
  duration-based timers.  The scenario shows both styles coexisting.

  Step-by-step:

  1. *13:45* — Leo's 45-min timer fires.  He takes a break and
     announces it with ``"tags": ["study.break"]``.  He sets a
     relative 10-min timer for his break.
  2. *13:45* — The ``study.break`` event matches Maya's
     ``["study."]`` mask.  She acknowledges, but resumes studying
     and re-sets her timer with ``"until": "15:00"``.  Note that
     the scheduler correctly computes the remaining time — it does
     not restart a full 2-hour wait.
  3. *13:55* — Leo's 10-min break timer fires.  He returns to work
     and sets an absolute timer with ``"until": "14:30"``.
  4. *14:30* — Leo's absolute timer fires.  He finishes his problem
     set and sets a 30-min relative timer.
  5. *15:00* — Maya's absolute timer fires.  She switches from
     studying to reviewing notes.

  Key features demonstrated:

  - **Absolute time timer** (``"until": "15:00"``) as the primary
    scheduling mechanism.
  - **Re-setting an absolute timer** after an interruption.
  - **Mixed usage**: both ``"time"`` (relative) and ``"until"``
    (absolute) timers in the same simulation.
  - **Multiple entities** using absolute timers.

  **Things to try:**

  - Change Maya's ``"until": "15:00"`` to ``"until": "8:00"``
    (before the current 13:00 start).  The day-rollover logic
    should push the timer to 08:00 the next day.
  - Give Leo an ``"until"`` time equal to the current simulated
    time (e.g. ``"until": "13:55"`` when it's already 13:55).
    The timer should also roll to the next day.
  - Change both entities to use only ``"time"`` durations to
    compare the trace output.

  Returns:
    The engine log.
  """
  initial_entity_states = {
      'Maya': {
          'mask': ['study.'],
          'timer_duration': '2h',
          'timer_description': 'exam study session ends at 15:00',
      },
      'Leo': {
          'mask': ['study.'],
          'timer_duration': '45m',
          'timer_description': 'finish first problem',
      },
  }
  maya_responses = [
      (  # Maya is interrupted by Leo's study.break event at 13:45.
          # She acknowledges the break but keeps studying, re-setting
          # her absolute timer to the same target time.
          'Maya glances up and waves at Leo, then goes back to studying.\n'
          '{"mask": ["study."], "timer":'
          ' {"until": "15:00", "reason": "exam study session ends"}}'
      ),
      (  # Maya's absolute timer fires at 15:00.
          'Time to switch to reviewing the notes.\n'
          '{"mask": [""], "timer":'
          ' {"time": "1h", "reason": "finish reviewing notes"}}'
      ),
  ]
  leo_responses = [
      (  # Leo's 45-min timer fires at 13:45. He takes a break.
          'Leo puts the pen down and stretches. Break time!\n'
          '{"mask": ["study."], "tags": ["study.break"],'
          ' "timer": {"time": "10m", "reason": "break is over"}}'
      ),
      (  # Leo's 10-min break timer fires at 13:55.
          # He returns to work and sets an absolute timer.
          'Break is over, back to the problem set.\n'
          '{"mask": ["study."], "timer":'
          ' {"until": "14:30", "reason": "finish problem set"}}'
      ),
      (  # Leo's absolute timer fires at 14:30.
          'Done with the problem set! Leo packs up the notes.\n'
          '{"mask": [""], "timer":'
          ' {"time": "30m", "reason": "head home"}}'
      ),
  ]

  return _run_scenario(
      player_names=['Maya', 'Leo'],
      initial_entity_states=initial_entity_states,
      responses={
          'Maya': maya_responses,
          'Leo': leo_responses,
      },
      start_time='2026-09-10T13:00:00',
      max_steps=5,
  )


# ═════════════════════════════════════════════════════════════
# Main entry point — run all examples
# ═════════════════════════════════════════════════════════════


def _run_all() -> None:
  """Run every scenario and print a step-by-step trace."""
  scenarios = [
      ('Coffee Shop', coffee_shop_example),
      ('Kitchen Timer', kitchen_timer_example),
      ('Morning Routine', morning_routine_example),
      ('Study Hall', study_hall_example),
  ]

  for title, scenario_fn in scenarios:
    print()
    print(f'{"═" * 56}')
    print(f' {title}')
    print(f'{"═" * 56}')
    log = scenario_fn()
    log_printing.print_log(log)
    print()


# ─────────────────────────────────────────────────────────────
# Test wrappers (for py_configurable_test BUILD target)
# ─────────────────────────────────────────────────────────────
# Each test runs a scenario and checks that a non-empty log was
# produced --- meaning the engine actually ran.


class CoffeeShopExampleTest(absltest.TestCase):
  """Wraps coffee_shop_example as a test case."""

  def test_coffee_shop_runs(self):
    log = coffee_shop_example()
    self.assertNotEmpty(log)


class KitchenTimerExampleTest(absltest.TestCase):
  """Wraps kitchen_timer_example as a test case."""

  def test_kitchen_timer_runs(self):
    log = kitchen_timer_example()
    self.assertNotEmpty(log)


class MorningRoutineExampleTest(absltest.TestCase):
  """Wraps morning_routine_example as a test case."""

  def test_morning_routine_runs(self):
    log = morning_routine_example()
    self.assertNotEmpty(log)


class StudyHallExampleTest(absltest.TestCase):
  """Wraps study_hall_example as a test case."""

  def test_study_hall_runs(self):
    log = study_hall_example()
    self.assertNotEmpty(log)


if __name__ == '__main__':
  # When run directly, print full traces for all scenarios.
  # When run via the test runner, the TestCase classes above are
  # discovered automatically.
  _run_all()
  absltest.main()
