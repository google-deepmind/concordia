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

"""Tests for the roster -> persona pipeline (generate_population/personas)."""

from collections.abc import Collection, Mapping, Sequence
import json
import os
from typing import Any, override

from absl.testing import absltest
from concordia.language_model import language_model
from examples.concordia_island.personas import generate_personas
from examples.concordia_island.personas import generator as generator_lib
from examples.concordia_island.personas.populations import generate_population


class _ScriptedModel(language_model.LanguageModel):
  """Returns queued replies for open questions (test double, not a fallback).

  If the queue is empty, it answers each batch prompt with one valid
  personality/backstory object per character, and any other prompt with
  `default_text`.
  """

  def __init__(self, replies: Sequence[str] = (), default_text: str = ''):
    self._replies = list(replies)
    self._default_text = default_text
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
    self.calls += 1
    if self._replies:
      return self._replies.pop(0)
    if 'demographic slots to fill' in prompt:
      n = prompt.count('Character ')
      return json.dumps([
          {'personality': f'Trait line {i}', 'backstory': f'Story {i}.'}
          for i in range(n)
      ])
    return self._default_text

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    return 0, responses[0], {}


class GeneratePopulationTest(absltest.TestCase):

  def test_dry_run_brecksville_roster(self):
    agents = generate_population.generate_population(
        model=None, num_agents=40, seed=3
    )
    roster = generate_population.build_roster(
        agents, setting='brecksville', seed=3, dry_run=True
    )
    meta = roster['metadata']
    self.assertEqual(meta['location_name'], 'Brecksville, Ohio')
    self.assertEqual(meta['setting'], 'ohio_suburb')
    self.assertEqual(meta['setting_preset'], 'brecksville_1000')
    self.assertTrue(meta['dry_run'])
    self.assertLen(roster['agents'], 40)
    names = [a['name'] for a in agents]
    self.assertLen(set(names), 40)
    # Couples share a home, so check the prefix set rather than per tier.
    self.assertContainsSubset(
        {a['home_place'].rsplit('_unit_', 1)[0] for a in agents},
        {'millbrook_apts', 'brecksville_commons', 'chippewa_ridge',
         'riverview_estates', 'timber_creek'},
    )
    self.assertFalse(any('personality' in a for a in agents))

  def test_island_setting_uses_concordia_island_name(self):
    self.assertEqual(
        generate_population.community_name_for('island'), 'Concordia Island'
    )
    self.assertEqual(generate_population.sim_setting_preset('island'), 'island')

  def test_unknown_setting_raises(self):
    with self.assertRaises(ValueError):
      generate_population.canonical_setting('atlantis')

  def test_llm_details_retry_then_succeed(self):
    model = _ScriptedModel(replies=['not json'])
    agents = generate_population.generate_population(
        model=model, num_agents=5, seed=1, batch_size=5, max_retries=2
    )
    self.assertEqual(model.calls, 2)
    self.assertTrue(all(a['personality'] and a['backstory'] for a in agents))

  def test_llm_details_fail_loudly(self):
    model = _ScriptedModel(replies=['not json', '[]', '{"a": 1}'])
    with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
      generate_population.generate_population(
          model=model, num_agents=5, seed=1, batch_size=5, max_retries=3
      )


class GeneratePersonasTest(absltest.TestCase):

  def test_dry_run_roster_rejected(self):
    agents = generate_population.generate_population(
        model=None, num_agents=3, seed=2
    )
    with self.assertRaisesRegex(ValueError, 'dry_run'):
      generate_personas.roster_to_agent_configs(agents)

  def test_roster_to_saved_personas(self):
    roster_model = _ScriptedModel()
    agents = generate_population.generate_population(
        model=roster_model, num_agents=4, seed=5, batch_size=4
    )
    roster = generate_population.build_roster(
        agents, setting='brecksville', seed=5
    )
    out = self.create_tempdir().full_path
    persona_model = _ScriptedModel(
        default_text=(
            'At age 10, they learned to ride a bike.\n'
            'At age 18, they moved out of their parents\' house.'
        )
    )
    saved = generate_personas.generate_personas(
        persona_model, roster, output_dir=out, date_label='test_set', workers=2
    )
    with open(os.path.join(saved, 'metadata.json')) as f:
      meta = json.load(f)
    self.assertEqual(meta['setting_preset'], 'brecksville_1000')
    self.assertEqual(meta['num_agents'], 4)
    loaded = generator_lib.load_personas(cns_path=out, date_label='test_set')
    self.assertCountEqual(loaded, [a['name'] for a in agents])
    with self.assertRaises(FileExistsError):
      generate_personas.generate_personas(
          persona_model, roster, output_dir=out, date_label='test_set'
      )


if __name__ == '__main__':
  absltest.main()
