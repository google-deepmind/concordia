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

"""Persona generation package for Concordia Island simulations.

This package provides:
- PersonaGenerator: 4-stage persona generation pipeline
- PopulationConfig: Population-level configuration
- PersonaData: Data container for generated personas
- load_personas / save_personas: local storage persistence

To build a population end to end (roster -> personas -> run.py), see
personas/README.md: populations/generate_population.py, then
generate_personas.py.
"""

from examples.concordia_island.personas.generator import load_personas
from examples.concordia_island.personas.generator import PersonaData
from examples.concordia_island.personas.generator import PersonaGenerator
from examples.concordia_island.personas.generator import PopulationConfig
from examples.concordia_island.personas.generator import save_personas
from examples.concordia_island.personas.generator import save_personas_local
