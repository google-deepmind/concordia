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

from collections.abc import Collection, Mapping, Sequence
from typing import Any, override

from absl.testing import absltest
from examples.adventure import adventure_prefab
from examples.adventure import playtester_bot
from examples.adventure import story_state
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class _CountingModel(language_model.LanguageModel):
  """Returns a fixed action and counts how often the bot asked for one."""

  def __init__(self):
    super().__init__()
    self.calls = 0

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
    del prompt, max_tokens, terminators, temperature, top_p, top_k
    del timeout, seed
    self.calls += 1
    return 'I look around the room carefully and ask who is in charge.'

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del prompt, seed
    return 0, responses[0], {}


class _IdleAct(entity_component.ActingComponent):
  """Placeholder player policy; the tests drive the bot directly."""

  @override
  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del context, action_spec
    return 'wait'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


_ACTION_SPEC = entity_lib.DEFAULT_ACTION_SPEC


def _make_sim(model):
  bible = story_state.load_shipped_bible('bermondsey')
  return adventure_prefab.AdventureSimulation(
      model=model,
      player_act_component=_IdleAct(),
      enable_async_npc_plan=False,
      bible=bible,
  )


def _make_bot(sim, model, **kwargs):
  bot = playtester_bot.PlaytesterActComponent(
      model=model, ledger=sim.ledger, total_steps=12, **kwargs
  )
  bot.set_entity(sim.player_entity)
  bot.set_logging_channel(lambda datum: None)
  return bot


class CouncilFollowTest(absltest.TestCase):

  def test_council_follower_sends_faculty_prefixed_option_without_llm(self):
    model = _CountingModel()
    sim = _make_sim(model)
    bot = _make_bot(
        sim,
        model,
        archetype=playtester_bot.PlaytesterArchetype.COUNCIL_FOLLOWER,
    )
    options = sim.ledger.story.active_voice_options
    self.assertNotEmpty(options, 'bible with voices should seed a council')
    calls_before = model.calls

    action = bot.get_action_attempt({}, _ACTION_SPEC)

    self.assertTrue(action.startswith('['), action)
    faculty = action[1 : action.index(']')]
    self.assertIn(faculty, sim.ledger.story.player_voices)
    lowest = min(o.affinity for o in options)
    chosen = next(o for o in options if o.voice == faculty)
    self.assertEqual(chosen.affinity, lowest)
    self.assertEqual(model.calls, calls_before)
    self.assertEqual(bot.council_follows, 1)

    sim.ledger.record_player_action(action)
    self.assertEqual(sim.ledger.story.voice_choice_log[-1]['voice'], faculty)
    self.assertEqual(sim.ledger.story.voice_affinity[faculty], 2)

  def test_zero_rate_never_follows_and_one_rate_always_follows(self):
    model = _CountingModel()
    sim = _make_sim(model)
    never = _make_bot(sim, model, council_follow_rate=0.0)
    always = _make_bot(sim, model, council_follow_rate=1.0)
    for _ in range(4):
      never.get_action_attempt({}, _ACTION_SPEC)
      always.get_action_attempt({}, _ACTION_SPEC)
    self.assertEqual(never.council_follows, 0)
    # Every third turn is a movement turn and never follows the council.
    self.assertEqual(always.council_follows, 3)

  def test_follow_coin_is_seeded(self):
    model = _CountingModel()
    sim_a = _make_sim(model)
    sim_b = _make_sim(model)
    bot_a = _make_bot(sim_a, model, council_follow_rate=0.5, seed=7)
    bot_b = _make_bot(sim_b, model, council_follow_rate=0.5, seed=7)
    actions_a = [bot_a.get_action_attempt({}, _ACTION_SPEC) for _ in range(6)]
    actions_b = [bot_b.get_action_attempt({}, _ACTION_SPEC) for _ in range(6)]
    self.assertEqual(
        [a.startswith('[') for a in actions_a],
        [b.startswith('[') for b in actions_b],
    )

  def test_backtrack_probe_runs_once_then_resumes_exploration(self):
    model = _CountingModel()
    sim = _make_sim(model)
    bot = playtester_bot.PlaytesterActComponent(
        model=model,
        ledger=sim.ledger,
        archetype=playtester_bot.PlaytesterArchetype.DETECTIVE_INVESTIGATOR,
        total_steps=10,
        council_follow_rate=0.0,
    )
    bot.set_entity(sim.player_entity)
    directives: list[str] = []
    bot.set_logging_channel(
        lambda datum: directives.append(str(datum.get('Directive', '')))
    )

    origin_id = sim.ledger.player_location_id
    exits = list(sim.ledger.get_location(origin_id).exits.values())
    room_b = exits[0]
    room_c = exits[1] if len(exits) > 1 else exits[0]

    # Steps 1-3: visit 3 rooms so len(visited_set) >= 3.
    bot.get_action_attempt({}, _ACTION_SPEC)
    sim.ledger.move_entity(sim.ledger.player_name, room_b)
    bot.get_action_attempt({}, _ACTION_SPEC)
    sim.ledger.move_entity(sim.ledger.player_name, room_c)
    bot.get_action_attempt({}, _ACTION_SPEC)

    # Fast-forward to step 7 (backtrack_threshold = int(10 * 0.70) = 7).
    bot.set_state(
        {'step': 6, 'council_follows': 0, 'rehydration_probed': False}
    )
    bot.get_action_attempt({}, _ACTION_SPEC)
    self.assertStartsWith(directives[-1], 'BACKTRACK PROBE:')

    # Arrive back at origin_id on step 8: runs REHYDRATION CONTINUITY PROBE.
    sim.ledger.move_entity(sim.ledger.player_name, origin_id)
    bot.get_action_attempt({}, _ACTION_SPEC)
    self.assertStartsWith(directives[-1], 'REHYDRATION CONTINUITY PROBE:')

    # Step 9 (9 % 3 == 0): resumes FRONTIER TRAVERSAL, not rehydration.
    bot.get_action_attempt({}, _ACTION_SPEC)
    self.assertStartsWith(directives[-1], 'FRONTIER TRAVERSAL:')


if __name__ == '__main__':
  absltest.main()
