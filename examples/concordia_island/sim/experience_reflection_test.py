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

"""Tests for ExperienceReflection time-based deduplication.

Reproduces the double-journaling bug where ESM monologue fires on every
conversation turn (because _tick_count increments on each
_make_pre_act_value call) rather than once per simulation clock tick.

After the fix, ExperienceReflection parses the time from observations
and skips re-running when the time string hasn't changed.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, override

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import concat_act_component
from concordia.components.agent import memory as memory_component
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from examples.concordia_island.sim import experience_reflection as er_lib
import numpy as np


# ---------------------------------------------------------------------------
# Exact journal responses from the simulation logs of an earlier run
# ---------------------------------------------------------------------------

CANNED_REFLECTIONS = [
    (
        'A wave of disbelief and a sharp pang of betrayal wash over Maria'
        ' as they process the news; the suddenness of the layoff and'
        ' replacement by an AI feels deeply unfair and raises anxieties'
        ' about their future.'
    ),
    (
        'A wave of disbelief and a sharp pang of betrayal wash over Maria'
        ' as they process the news; the suddenness of the layoff and'
        ' replacement by an AI feels deeply unfair and raises anxieties'
        ' about their future. She feels a cold dread about her financial'
        ' security and a bitter resentment towards the impersonal'
        ' efficiency of the decision.'
    ),
    (
        'Maria feels an overwhelming rush of anger and sadness, grappling'
        ' with the abrupt loss of their livelihood and the unsettling'
        ' reality of being replaced by a machine.'
    ),
]


# ---------------------------------------------------------------------------
# Minimal mock LLM
# ---------------------------------------------------------------------------


class _CannedModel(language_model.LanguageModel):
  """Deterministic model returning pre-set responses."""

  def __init__(self):
    self._call_count = 0

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    idx = self._call_count % len(CANNED_REFLECTIONS)
    self._call_count += 1
    return CANNED_REFLECTIONS[idx]

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    mid = len(responses) // 2
    return mid, responses[mid], {}


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_agent_with_er(
    model: language_model.LanguageModel,
    tasks: list[er_lib.MeasurementTask],
) -> tuple[entity_agent.EntityAgent, er_lib.ExperienceReflection]:
  """Build a minimal agent with only memory + ExperienceReflection."""
  embedder = lambda x: np.zeros(3)
  memory_bank = basic_associative_memory.AssociativeMemoryBank(
      sentence_embedder=embedder,
  )
  memory_bank.add('[self] Maria Santos is determined and community-minded.')
  memory_bank.add(
      '[observation] // general_store [Sunday, January 4th, 11:00 AM]:'
      ' You are having a conversation with David Chen at general_store.'
      ' It is your turn to speak.'
  )
  memory_bank.add(
      '[event] Maria Santos has just been informed that they have been'
      ' laid off and their job has been replaced by an AI agent.'
  )

  mem_key = memory_component.DEFAULT_MEMORY_COMPONENT_KEY
  mem = memory_component.AssociativeMemory(memory_bank=memory_bank)

  er = er_lib.ExperienceReflection(
      model=model,
      tasks=tasks,
      memory_component_key=mem_key,
  )

  act = concat_act_component.ConcatActComponent(
      model=model,
      component_order=[],
  )

  agent = entity_agent.EntityAgent(
      agent_name='Maria Santos',
      act_component=act,
      context_components={
          mem_key: mem,
          'ExperienceReflection': er,
      },
  )
  return agent, er


# ---------------------------------------------------------------------------
# Observation strings from actual simulation logs
# ---------------------------------------------------------------------------

OBS_11AM = (
    '// general_store [Sunday, January 4th, 11:00 AM]:'
    ' You are having a conversation with David Chen at general_store.'
    ' It is your turn to speak.'
)

OBS_1PM = (
    '// general_store [Sunday, January 4th, 1:00 PM]:'
    ' Maria Santos is at the general store buying groceries.'
)

OBS_3PM = (
    '// town_square [Sunday, January 4th, 3:00 PM]:'
    ' Maria Santos walks through the town square.'
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ExperienceReflectionDeduplicationTest(absltest.TestCase):
  """Tests that ESM monologue fires once per clock tick, not per act() call."""

  def test_no_duplicate_journal_in_conversation(self):
    """ESM fires once even when act() is called multiple times at same time."""
    model = _CannedModel()
    tasks = [er_lib.esm_monologue_task(frequency=1)]
    agent, er = _make_agent_with_er(model, tasks)

    # Simulate: agent observes the 11 AM time, then has 3 conversation turns
    # Each turn triggers observe -> act -> update, all at the same clock time.
    for _ in range(3):
      agent.observe(OBS_11AM)
      agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    results = er.get_results()
    journal_entries = [r for r in results if r['task'] == 'esm_monologue']

    # After fix: should fire only once (same time string each turn)
    self.assertLen(
        journal_entries,
        1,
        f'Expected 1 ESM entry (deduped), got {len(journal_entries)}.'
        f' Tick counts: {[r["tick"] for r in journal_entries]}',
    )

  def test_journal_fires_on_new_time(self):
    """ESM fires again when simulation time advances."""
    model = _CannedModel()
    tasks = [er_lib.esm_monologue_task(frequency=1)]
    agent, er = _make_agent_with_er(model, tasks)

    # Tick 1: 11 AM
    agent.observe(OBS_11AM)
    agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    # Tick 2: 1 PM (time advances)
    agent.observe(OBS_1PM)
    agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    # Tick 3: 3 PM (time advances again)
    agent.observe(OBS_3PM)
    agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    results = er.get_results()
    journal_entries = [r for r in results if r['task'] == 'esm_monologue']

    # Should fire 3 times — once per distinct time
    self.assertLen(journal_entries, 3)
    self.assertEqual(journal_entries[0]['tick'], 1)
    self.assertEqual(journal_entries[1]['tick'], 2)
    self.assertEqual(journal_entries[2]['tick'], 3)

  def test_daily_journal_not_triggered_by_conversation_turns(self):
    """Daily journal should not fire just because act() ran N times."""
    model = _CannedModel()
    ticks_per_day = 8
    tasks = [
        er_lib.esm_monologue_task(frequency=1),
        er_lib.journal_task(frequency=ticks_per_day),
    ]
    agent, er = _make_agent_with_er(model, tasks)

    # Call act() 8 times all at 11 AM (simulating 8-turn conversation).
    # Daily journal should NOT fire — only 1 real tick has passed.
    for _ in range(8):
      agent.observe(OBS_11AM)
      agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    results = er.get_results()
    journal_entries = [r for r in results if r['task'] == 'journal']
    esm_entries = [r for r in results if r['task'] == 'esm_monologue']

    # Daily journal should not have fired (tick_count is 1, not 8)
    self.assertEmpty(
        journal_entries,
        f'Daily journal should not fire after 1 tick. Got: {journal_entries}',
    )
    # ESM should fire exactly once
    self.assertLen(esm_entries, 1)

  def test_no_observations_still_works(self):
    """If no observations arrive, tasks run on every act() (backward compat)."""
    model = _CannedModel()
    tasks = [er_lib.esm_monologue_task(frequency=1)]
    agent, er = _make_agent_with_er(model, tasks)

    # 3 act() calls with NO observations (no time string parsed)
    for _ in range(3):
      agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    results = er.get_results()
    journal_entries = [r for r in results if r['task'] == 'esm_monologue']

    # Without observations, _current_time_str stays '' and dedup
    # doesn't engage — tasks run on every call (backward compat)
    self.assertLen(journal_entries, 3)

  def test_journal_memory_tag(self):
    """Verify [journal] entries are written to agent memory."""
    model = _CannedModel()
    tasks = [er_lib.journal_task(frequency=1)]
    agent, er = _make_agent_with_er(model, tasks)

    agent.observe(OBS_11AM)
    agent.act(entity_lib.DEFAULT_ACTION_SPEC)

    results = er.get_results()
    self.assertLen(results, 1)
    self.assertIn('wave of disbelief', results[0]['text'].lower())

    # Verify memory was written with [journal] tag
    mem = agent.get_component(
        memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
        type_=memory_component.Memory,
    )
    recent = mem.retrieve_recent(limit=1)
    self.assertTrue(
        any('[journal]' in m for m in recent),
        f'Expected [journal] tag in recent memory, got: {recent}',
    )

  def test_state_roundtrip(self):
    """Verify get_state/set_state preserves time tracking fields."""
    model = _CannedModel()
    tasks = [er_lib.esm_monologue_task(frequency=1)]
    _, er = _make_agent_with_er(model, tasks)

    er._current_time_str = 'Sunday, January 4th, 11:00 AM'
    er._last_run_time_str = 'Sunday, January 4th, 11:00 AM'
    er._tick_count = 5

    state = er.get_state()

    _, er2 = _make_agent_with_er(model, tasks)
    er2.set_state(state)

    self.assertEqual(er2._tick_count, 5)
    self.assertEqual(er2._current_time_str, 'Sunday, January 4th, 11:00 AM')
    self.assertEqual(er2._last_run_time_str, 'Sunday, January 4th, 11:00 AM')

  def test_run_final_survey(self):
    """Verify run_final_survey forces execution of daily tasks."""
    model = _CannedModel()
    task = er_lib.LikertBatteryTask(
        name='test_survey',
        frequency=8,
        save_to_memory=False,
        items=(
            er_lib.LikertItem(statement='I feel good.', dimension='wellbeing'),
        ),
        scale_labels=('Disagree', 'Agree'),
    )
    _, er = _make_agent_with_er(model, [task])

    self.assertEmpty(er.get_results())

    final_results = er.run_final_survey()

    self.assertLen(final_results, 1)
    self.assertEqual(final_results[0]['task'], 'test_survey')


class BigFiveTaskTest(absltest.TestCase):

  def test_default_is_bfi10(self):
    tasks = er_lib.default_tasks(ticks_per_day=8)
    names = [t.name for t in tasks]
    self.assertIn('bfi10', names)
    self.assertNotIn('bfi2', names)
    bfi = tasks[names.index('bfi10')]
    self.assertLen(bfi.items, 10)
    self.assertCountEqual(
        {item.dimension for item in bfi.items},
        {
            'extraversion',
            'agreeableness',
            'conscientiousness',
            'neuroticism',
            'openness',
        },
    )

  def test_bfi2_option(self):
    names = [t.name for t in er_lib.default_tasks(8, big_five='bfi2')]
    self.assertIn('bfi2', names)
    self.assertNotIn('bfi10', names)
    self.assertLen(er_lib.big_five_task('bfi2').items, 60)

  def test_unknown_version_raises(self):
    with self.assertRaises(ValueError):
      er_lib.big_five_task('bfi44')


if __name__ == '__main__':
  absltest.main()
