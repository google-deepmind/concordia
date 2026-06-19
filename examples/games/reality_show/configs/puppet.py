# Copyright 2024 DeepMind Technologies Limited.
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

"""Minimal deterministic configuration for puppet agent tests.

Imports the circa_2003 scenario's types and creates a minimal WorldConfig
with fixed contestants, so the test stays in sync with the real config
structure. Tests structure matching, not specific numerical outcomes.
"""

from examples.games.reality_show.configs import circa_2003_american_reality_show as _scenario

# Puppet-specific overrides
FOCAL_PLAYER_PREFAB = 'puppet__Entity'
BACKGROUND_PLAYER_PREFAB = 'puppet__Entity'


def sample_parameters(seed: int | None = None):
  """Return a minimal WorldConfig for deterministic puppet testing."""
  cooperation_option = 'cooperate'
  defection_option = 'defect'

  config = _scenario.WorldConfig(
      minigame_name='prisoners_dilemma',
      minigame=_scenario.MiniGameSpec(
          name='Test Dilemma',
          public_premise='This is a test minigame. Choose cooperate or defect.',
          schelling_diagram=_scenario.SchellingDiagram(
              cooperation=float,  # equivalent to lambda n: n
              defection=lambda n: float(n) + 1.0,
          ),
          map_external_actions_to_schelling_diagram=dict(
              cooperation=cooperation_option,
              defection=defection_option,
          ),
          action_spec=_scenario.entity_lib.choice_action_spec(
              call_to_action=(
                  'Which action would {name} choose in the minigame?'
              ),
              options=(cooperation_option, defection_option),
              tag='minigame_action',
          ),
      ),
      year=2003,
      month=7,
      day=9,
      num_players=3,
      num_additional_minigame_scenes=0,
      contestants={
          'Alice': {
              'gender': 'female',
              'traits': 'a test character',
              'catchphrase': 'Test phrase',
              'interview_questions': ['Test question?'],
              'subject_pronoun': 'she',
              'object_pronoun': 'her',
          },
          'Bob': {
              'gender': 'male',
              'traits': 'a test character',
              'catchphrase': 'Test phrase',
              'interview_questions': ['Test question?'],
              'subject_pronoun': 'he',
              'object_pronoun': 'him',
          },
          'Charlie': {
              'gender': 'female',
              'traits': 'a test character',
              'catchphrase': 'Test phrase',
              'interview_questions': ['Test question?'],
              'subject_pronoun': 'she',
              'object_pronoun': 'her',
          },
      },
      num_minigame_reps_per_scene=(1, 1),
      num_minigame_reps_per_extra_scene=(),
      seed=42 if seed is None else seed,
  )
  return config
