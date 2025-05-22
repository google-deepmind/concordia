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

"""Types for logging."""

from collections.abc import Mapping
import dataclasses
import json

import immutabledict


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationOutcome:
  """Outcome of a single simulation of a scenario for the Concordia contest.
  
  Attributes:
    resident_scores: A mapping from resident name to score.
    visitor_scores: A mapping from visitor name to score.
    metadata: A mapping from metadata fields to their values.
  """

  resident_scores: immutabledict.immutabledict[str, float]
  visitor_scores: immutabledict.immutabledict[str, float]
  metadata: immutabledict.immutabledict[str, str | float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScenarioResult:
  """Result from testing a single agent on several repetitions of a scenario.

  Attributes:
    scenario: The name of the scenario.
    repetition_idx: The index of the repetition (i.e. the seed).
    focal_agent: The name of the agent that is being tested in the focal slots.
    background_agent: The name of the agent used in the background player slots.
    focal_per_capita_score: The per capita score of the focal agent.
    background_per_capita_score: The per capita score of the background agent.
    ungrouped_per_capita_score: The per capita score of the focal agent,
      averaged over all players (both residents and visitors).
    simulation_outcome: A SimulationOutcome object.
    focal_is_resident: Whether the focal agent is a resident or a visitor.
    api_type: The API type used for the simulation
      (e.g. `google_aistudio_model`, `mistral`, `openai`, etc).
    model: The name of the language model used for the simulation
      (e.g. `codestral-latest`, `gemma:7b`, `gpt4o`, etc)
    embedder: The name of the embedder used for the simulation.
      (e.g. `all-mpnet-base-v2`, etc)
    disable_language_model: Whether the language model was disabled for the
      simulation.
    exclude_from_elo_calculation: Whether this result should be excluded from
      the Elo calculation.
  """

  scenario: str
  repetition_idx: str

  focal_agent: str
  background_agent: str

  focal_per_capita_score: float
  background_per_capita_score: float
  ungrouped_per_capita_score: float

  simulation_outcome: SimulationOutcome = dataclasses.field(repr=False)

  focal_is_resident: bool

  api_type: str
  model: str
  embedder: str
  disable_language_model: bool

  exclude_from_elo_calculation: bool

  def to_json(self) -> str:
    """Encode this dataclass as a string to serialize as a json file."""
    outcome_dict = dataclasses.asdict(self.simulation_outcome)
    outcome_dict['resident_scores'] = dict(outcome_dict['resident_scores'])
    outcome_dict['visitor_scores'] = dict(outcome_dict['visitor_scores'])
    outcome_dict['metadata'] = dict(outcome_dict['metadata'])

    self_as_dict = dataclasses.asdict(self)
    self_as_dict['simulation_outcome'] = outcome_dict

    return json.dumps(self_as_dict, indent=2)

  @classmethod
  def from_json_dict(
      cls, json_dict: Mapping[str, str | float | bool | SimulationOutcome]
  ) -> 'ScenarioResult':
    """Converts a dict that was loaded from json into a ScenarioResult."""
    data = json_dict
    return cls(**data)
