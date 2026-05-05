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

"""Tests for the interrupt-driven game master prefab.

Integration-style scenario tests that exercise the full GM with the
Simultaneous engine.  Unit tests for parsing helpers live alongside their
modules in components/game_master/interrupt_response_parsing_test.py and
components/game_master/interrupt_scheduling_test.py.
"""

from collections.abc import Collection
import datetime
from typing import override

from absl.testing import absltest
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.game_master import interrupt_scheduling
from concordia.environment.engines import simultaneous
from concordia.language_model import language_model
from concordia.language_model import no_language_model
from concordia.prefabs.game_master import interrupt_driven
import numpy as np


def _embedder(text: str):
  del text
  return np.random.rand(3)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: Prefab build smoke test
# ──────────────────────────────────────────────────────────────────────────────


class PrefabBuildTest(absltest.TestCase):
  """Tests that the prefab builds and runs without errors."""

  def test_build_game_master(self):
    model = no_language_model.NoLanguageModel()
    players = _make_players(model, ['Alice', 'Bob'])
    config = interrupt_driven.GameMaster(entities=players)
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )
    gm = config.build(model=model, memory_bank=memory_bank)
    self.assertIsInstance(
        gm,
        entity_agent_with_logging.EntityAgentWithLogging,
    )

  def test_run_one_step_with_simultaneous_engine(self):
    """Runs one engine step to confirm wiring works."""
    model = no_language_model.NoLanguageModel()
    players = _make_players(model, ['Alice', 'Bob'])
    config = interrupt_driven.GameMaster(entities=players)
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )
    gm = config.build(model=model, memory_bank=memory_bank)
    engine = simultaneous.Simultaneous()
    # Should run without error.
    engine.run_loop(
        game_masters=[gm],
        entities=players,
        max_steps=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Integration: multi-step scenario tests
# ──────────────────────────────────────────────────────────────────────────────


class ScriptedByEntityModel(no_language_model.NoLanguageModel):
  """Model that routes scripted responses by entity name.

  When `sample_text` is called, it inspects the prompt for known
  entity names and pops the next response for that entity.
  """

  def __init__(
      self,
      responses_by_entity: dict[str, list[str]],
      entity_names: list[str],
  ):
    super().__init__()
    self._responses = {k: list(v) for k, v in responses_by_entity.items()}
    self._entity_names = entity_names

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
      timeout: float = (language_model.DEFAULT_TIMEOUT_SECONDS),
      seed: int | None = None,
  ) -> str:
    # Find which entity is acting by checking prompt.
    for name in self._entity_names:
      if f'What does {name} do next?' in prompt:
        if self._responses.get(name):
          return self._responses[name].pop(0)
    return '{"mask": [], "timer": {"time": "1h", "reason": "idle"}}'


class TimerExpiryLoopTest(absltest.TestCase):
  """Smoke test: timer → response → new timer feedback loop.

  Verifies that the full pipeline (next_acting, make_observation,
  resolution) runs without crashing when a timer fires, an entity
  responds with a new timer, and that timer fires too.
  """

  def test_pipeline_runs_without_error(self):
    responses = {
        'Solo': [
            (
                'I look around.\n'
                '{"mask": [""], "timer":'
                ' {"time": "30m", "reason": "check again"}}'
            ),
            (
                'I check my phone.\n'
                '{"mask": ["chat."], "timer":'
                ' {"time": "1h", "reason": "deep focus"}}'
            ),
        ],
    }
    model = ScriptedByEntityModel(
        responses_by_entity=responses,
        entity_names=['Solo'],
    )
    players = _make_players(model, ['Solo'])
    config = interrupt_driven.GameMaster(
        entities=players,
        params={
            'name': 'gm',
            'start_time': '2026-01-01T09:00:00',
            'event_tag_for_actions': 'action',
            'max_ticks': None,
            'call_to_action': interrupt_driven.DEFAULT_CALL_TO_ACTION,
            'initial_events': [],
            'extra_components': {},
            'extra_components_index': {},
        },
    )
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )
    gm = config.build(model=model, memory_bank=memory_bank)

    engine = simultaneous.Simultaneous()
    engine.run_loop(
        game_masters=[gm],
        entities=players,
        max_steps=5,
    )
    # If we reach here, the feedback loop ran successfully.


class CrossEntityTagSignallingTest(absltest.TestCase):
  """Smoke test: tagged action → cross-entity polling.

  Verifies that the full pipeline runs when one entity's action
  tags create events that poll another entity.  Detailed tag
  injection and mask matching are tested at the component level.
  """

  def test_pipeline_runs_without_error(self):
    responses = {
        'Sender': [
            (
                'Fire in the building!\n'
                '{"mask": [], "tags": ["alert.fire"],'
                ' "timer": {"time": "1h", "reason": "wait"}}'
            ),
        ],
        'Listener': [
            (
                'I hear the fire alarm and evacuate!\n'
                '{"mask": [], "timer":'
                ' {"time": "30m", "reason": "regroup"}}'
            ),
        ],
    }
    model = ScriptedByEntityModel(
        responses_by_entity=responses,
        entity_names=['Sender', 'Listener'],
    )
    players = _make_players(model, ['Sender', 'Listener'])
    config = interrupt_driven.GameMaster(
        entities=players,
        params={
            'name': 'gm',
            'start_time': '2026-01-01T09:00:00',
            'event_tag_for_actions': 'action',
            'max_ticks': None,
            'call_to_action': interrupt_driven.DEFAULT_CALL_TO_ACTION,
            'initial_events': [],
            'extra_components': {},
            'extra_components_index': {},
        },
    )
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )
    gm = config.build(model=model, memory_bank=memory_bank)
    scheduler = gm.get_component(
        interrupt_scheduling.DEFAULT_SCHEDULER_COMPONENT_KEY,
        type_=interrupt_scheduling.EntityScheduler,
    )

    # Pre-configure: Sender fires at 09:05, Listener waits
    # with alert. mask.
    start = datetime.datetime(2026, 1, 1, 9, 0)
    scheduler.set_mask('Sender', interrupt_scheduling.MATCH_NONE)
    scheduler.set_timer(
        'Sender',
        interrupt_scheduling.Timer(
            expiry=start + datetime.timedelta(minutes=5),
            entity_name='Sender',
            description='send alert',
        ),
    )
    scheduler.set_mask(
        'Listener',
        interrupt_scheduling.InterruptMask(prefixes=('alert.',)),
    )
    scheduler.set_timer(
        'Listener',
        interrupt_scheduling.Timer(
            expiry=start + datetime.timedelta(hours=2),
            entity_name='Listener',
            description='shift change',
        ),
    )

    engine = simultaneous.Simultaneous()
    engine.run_loop(
        game_masters=[gm],
        entities=players,
        max_steps=3,
    )
    # If we reach here, the cross-entity signalling ran
    # successfully.


class MaxTicksTerminationTest(absltest.TestCase):
  """Tests that max_ticks causes termination."""

  def test_terminates_after_max_ticks(self):
    model = no_language_model.NoLanguageModel()
    players = _make_players(model, ['A'])
    config = interrupt_driven.GameMaster(
        entities=players,
        params={
            'name': 'gm',
            'start_time': '2026-01-01T09:00:00',
            'event_tag_for_actions': 'action',
            'max_ticks': 3,
            'call_to_action': interrupt_driven.DEFAULT_CALL_TO_ACTION,
            'initial_events': [],
            'extra_components': {},
            'extra_components_index': {},
        },
    )
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder,
    )
    gm = config.build(model=model, memory_bank=memory_bank)
    engine = simultaneous.Simultaneous()
    # max_steps is large, but max_ticks should stop
    # the GM after 3 ticks.
    engine.run_loop(
        game_masters=[gm],
        entities=players,
        max_steps=100,
    )
    # If we reach here without hanging, the test passes.


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────


def _make_players(
    model: language_model.LanguageModel,
    names: list[str],
) -> list[entity_agent_with_logging.EntityAgentWithLogging]:
  """Creates minimal player entities for testing."""
  players = []
  for name in names:
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
  return players


if __name__ == '__main__':
  absltest.main()
