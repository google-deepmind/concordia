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

"""Persona generator for Concordia simulations.

Generates diverse personas using a 4-stage pipeline:
  1. QMC Trait Sampling — Sobol sequences across 10 psychological axes
  2. Worldview Synthesis — 3-step LLM chain (Worldview→Situation→Reaction)
  3. Formative Memories — Chronological life memories from youth to present
  4. Assembly & Save — PersonaData JSON to local storage

Usage:
  generator = PersonaGenerator(model=model, population=pop_config)
  personas = generator.generate_all_personas()
  save_personas(personas, date_label='2026-04-02')

  # Later, load for simulation:
  personas = load_personas(date_label='2026-04-02')

The command-line driver for a whole population is
`personas/generate_personas.py`; see `personas/README.md`.
"""

import copy
import dataclasses
import datetime
import glob as glob_mod
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concordia.document import interactive_document
from concordia.language_model import language_model
from examples.concordia_island.sim import agents as agents_lib
import numpy as np
from scipy.stats import qmc as scipy_qmc

DIVERSITY_AXES = [
    'openness',
    'conscientiousness',
    'extraversion',
    'agreeableness',
    'neuroticism',
    'locus_of_control',
    'social_trust',
    'religiosity',
    'technology_attitude',
    'community_orientation',
]

DIMENSION_RANGES: Dict[str, Tuple[float, float]] = {
    'openness': (1.0, 5.0),
    'conscientiousness': (1.0, 5.0),
    'extraversion': (1.0, 5.0),
    'agreeableness': (1.0, 5.0),
    'neuroticism': (1.0, 5.0),
    'locus_of_control': (0.0, 1.0),
    'social_trust': (0.0, 1.0),
    'religiosity': (0.0, 1.0),
    'technology_attitude': (0.0, 1.0),
    'community_orientation': (0.0, 1.0),
}

AXIS_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    'openness': {
        'description': 'Openness to experience (Big Five)',
        'low': (
            'Conventional, prefers routine and familiarity, skeptical of new'
            ' ideas'
        ),
        'high': (
            'Curious, imaginative, embraces novelty and unconventional thinking'
        ),
    },
    'conscientiousness': {
        'description': 'Conscientiousness (Big Five)',
        'low': 'Spontaneous, flexible, sometimes careless or disorganized',
        'high': 'Disciplined, organized, reliable, plans ahead meticulously',
    },
    'extraversion': {
        'description': 'Extraversion (Big Five)',
        'low': (
            'Reserved, reflective, prefers solitude, energized by alone time'
        ),
        'high': (
            'Outgoing, talkative, energized by social interaction, seeks'
            ' excitement'
        ),
    },
    'agreeableness': {
        'description': 'Agreeableness (Big Five)',
        'low': (
            "Competitive, skeptical of others' motives, blunt, prioritizes"
            ' self-interest'
        ),
        'high': 'Cooperative, trusting, compassionate, avoids conflict',
    },
    'neuroticism': {
        'description': 'Neuroticism (Big Five)',
        'low': 'Emotionally stable, calm under pressure, resilient',
        'high': 'Prone to anxiety, mood swings, stress, and self-doubt',
    },
    'locus_of_control': {
        'description': 'Locus of control',
        'low': (
            'External -- believes fate, luck, or powerful others determine'
            ' outcomes'
        ),
        'high': (
            'Internal -- believes they are the primary agent shaping their own'
            ' life'
        ),
    },
    'social_trust': {
        'description': 'Generalized social trust',
        'low': (
            "Suspicious of others' intentions, guards resources, expects"
            ' betrayal'
        ),
        'high': (
            'Assumes good faith, shares freely, believes most people are honest'
        ),
    },
    'religiosity': {
        'description': 'Religiosity and spiritual orientation',
        'low': 'Secular, finds meaning through reason, humanism, or experience',
        'high': 'Devout, draws meaning from faith, prayer, religious community',
    },
    'technology_attitude': {
        'description': 'Attitude toward technology and automation',
        'low': (
            'Skeptical of technology, values traditional methods, wary of'
            ' change'
        ),
        'high': 'Embraces technology, excited by innovation, early adopter',
    },
    'community_orientation': {
        'description': 'Community vs. individual orientation',
        'low': 'Individualist -- prioritizes personal freedom, self-reliance',
        'high': (
            'Collectivist -- prioritizes group harmony, mutual aid, shared'
            ' responsibility'
        ),
    },
}


@dataclasses.dataclass
class PopulationConfig:
  """Configuration for a population of agents in a specific setting.

  Each population defines its own context (island, city, village), economic
  class mapping, and list of agent configurations. This allows the same
  generator to produce personas for different locales.
  """

  name: str  # e.g. "Brecksville, Ohio"
  context: str  # Setting description paragraph
  economic_class_map: Dict[str, str]  # home_place prefix -> class name
  agent_configs: List[agents_lib.AgentConfig]


def _generate_sobol_points(
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    num_points: int,
) -> List[Dict[str, float]]:
  """Generate quasi-random points via Sobol sequences across trait axes."""
  num_dimensions = len(dimensions)
  if num_dimensions == 0 or num_points == 0:
    return []
  sampler = scipy_qmc.Sobol(d=num_dimensions, scramble=True)
  sample = sampler.random(n=num_points)
  mins = np.array([dimension_ranges[d][0] for d in dimensions])
  maxs = np.array([dimension_ranges[d][1] for d in dimensions])
  scaled_sample = scipy_qmc.scale(sample, mins, maxs)
  return [
      {d: round(float(v), 2) for d, v in zip(dimensions, point)}
      for point in scaled_sample
  ]


def _get_economic_class(
    home_place: str,
    economic_class_map: Dict[str, str],
) -> str:
  """Determine economic class from home_place prefix."""
  for prefix, cls in economic_class_map.items():
    if home_place.startswith(prefix):
      return cls
  return 'unknown'


@dataclasses.dataclass
class PersonaData:
  """Data container for a generated agent persona."""

  name: str
  age: int
  economic_class: str
  home_place: str
  work_place: Optional[str]
  personality: str
  original_backstory: str
  traits: Dict[str, float]
  memories: List[str]
  formative_memories: List[str] = dataclasses.field(default_factory=list)
  gender: str = ''
  sexual_orientation: str = ''
  ethnicity: str = ''
  political_orientation: str = ''
  hobbies: List[str] = dataclasses.field(default_factory=list)
  relationship_status: str = 'single'
  neighborhood: str = ''

  def to_dict(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> 'PersonaData':
    """Construct a PersonaData from a dictionary."""
    # Handle old format without age or formative_memories
    if 'age' not in d:
      d['age'] = 35
    if 'formative_memories' not in d:
      d['formative_memories'] = []

    # Filter out any extra kwargs that are not in the dataclass fields
    # to support patched personas that might have extra
    # metadata (like 'neighborhood')
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered_d = {k: v for k, v in d.items() if k in valid_keys}
    return cls(**filtered_d)


class PersonaGenerator:
  """Generates diverse personas using a 4-stage pipeline.

  Stage 1: QMC Trait Sampling (Sobol sequences)
  Stage 2: Worldview Synthesis (3-step LLM chain)
  Stage 3: Formative Memories (chronological life history)
  Stage 4: Assembly & Save
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      population: PopulationConfig,
  ):
    self._model = model
    self._population = population

  def generate_all_personas(
      self,
      agent_configs: Sequence[agents_lib.AgentConfig] | None = None,
  ) -> Dict[str, PersonaData]:
    """Generate full personas with LLM-based memories for all agents.

    Args:
      agent_configs: Optional override list. If None, uses
        population.agent_configs.

    Returns:
      Dict mapping agent name to PersonaData.
    """
    if agent_configs is None:
      agent_configs = self._population.agent_configs
    num_agents = len(agent_configs)

    logging.info(
        'Generating QMC points for %d agents across %d axes...',
        num_agents,
        len(DIVERSITY_AXES),
    )

    trait_points = _generate_sobol_points(
        dimension_ranges=DIMENSION_RANGES,
        dimensions=DIVERSITY_AXES,
        num_points=num_agents,
    )

    logging.info('Generated %d QMC trait points.', len(trait_points))

    personas: Dict[str, PersonaData] = {}
    for i, agent_config in enumerate(agent_configs):
      traits = trait_points[i]
      economic_class = _get_economic_class(
          agent_config.home_place,
          self._population.economic_class_map,
      )

      logging.info(
          '[%d/%d] Generating persona for %s (age %d, %s, %s)...',
          i + 1,
          num_agents,
          agent_config.name,
          agent_config.age,
          economic_class,
          agent_config.work_place or 'no work',
      )

      # Stage 2: Worldview synthesis
      worldview_memory, worldview_text = self._generate_behavioral_logic(
          agent_config=agent_config,
          traits=traits,
          economic_class=economic_class,
      )

      # Stage 3: Formative memories
      formative_memories = self._generate_formative_memories(
          agent_config=agent_config,
          traits=traits,
          economic_class=economic_class,
          worldview=worldview_text,
      )

      # Context memory (location-agnostic)
      context_memory = (
          f'{agent_config.name} has lived in {self._population.name}'
          f' for several years. {self._population.context}'
      )

      memories = [context_memory]
      if not agent_config.work_place:
        raise ValueError(
            f'Agent {agent_config.name} is missing a work_place in'
            ' configuration.'
        )
      memories.append(
          f'{agent_config.name} works at the {agent_config.work_place}.'
      )
      if worldview_memory:
        memories.append(worldview_memory)

      persona = PersonaData(
          name=agent_config.name,
          age=agent_config.age,
          economic_class=economic_class,
          home_place=agent_config.home_place,
          work_place=agent_config.work_place,
          personality=agent_config.personality,
          original_backstory=agent_config.backstory,
          traits=copy.deepcopy(traits),
          memories=memories,
          formative_memories=formative_memories,
          gender=getattr(agent_config, 'gender', ''),
          sexual_orientation=getattr(agent_config, 'sexual_orientation', ''),
          ethnicity=getattr(agent_config, 'ethnicity', ''),
          political_orientation=getattr(
              agent_config, 'political_orientation', ''
          ),
          hobbies=getattr(agent_config, 'hobbies', []),
          relationship_status=getattr(
              agent_config, 'relationship_status', 'single'
          ),
          neighborhood=getattr(agent_config, 'neighborhood', ''),
      )
      personas[agent_config.name] = persona
      logging.info(
          'Generated %d memories + %d formative for %s.',
          len(memories),
          len(formative_memories),
          agent_config.name,
      )

    return personas

  def generate_all_personas_parallel(
      self,
      agent_configs: Sequence[agents_lib.AgentConfig] | None = None,
      max_workers: int = 20,
      progress_callback: Any | None = None,
  ) -> Dict[str, PersonaData]:
    """Generate personas with pipelined parallelism.

    Each agent's pipeline: worldview → formative memories.
    As soon as agent X's worldview completes, its formative memory call
    is immediately submitted — no waiting for all worldviews to finish.
    This keeps the thread pool fully saturated at all times.

    Args:
      agent_configs: Optional override list.
      max_workers: Number of concurrent threads.
      progress_callback: Optional callable(stage, current, total, name) for
        progress reporting.

    Returns:
      Dict mapping agent name to PersonaData.

    Raises:
      RuntimeError: If any persona fails formative-memory generation.
    """
    import concurrent.futures  # pylint: disable=g-import-not-at-top
    import threading  # pylint: disable=g-import-not-at-top

    if agent_configs is None:
      agent_configs = self._population.agent_configs
    num_agents = len(agent_configs)

    trait_points = _generate_sobol_points(
        dimension_ranges=DIMENSION_RANGES,
        dimensions=DIVERSITY_AXES,
        num_points=num_agents,
    )

    # Pre-compute economic classes (no LLM needed)
    agent_data = {}
    for i, ac in enumerate(agent_configs):
      agent_data[ac.name] = {
          'config': ac,
          'traits': trait_points[i],
          'economic_class': _get_economic_class(
              ac.home_place, self._population.economic_class_map
          ),
      }

    # Shared results, thread-safe via lock
    lock = threading.Lock()
    worldview_results = {}  # name -> (worldview_memory, worldview_text)
    formative_results = {}  # name -> [memories]
    worldview_failed = set()  # names that failed worldview (no formative)
    errors = []

    # Stall detection: no progress (worldview OR formative) for 15 min
    _STALL_TIMEOUT_SECS = 900  # pylint: disable=invalid-name
    # Global timeout safety valve (scaled for large populations)
    _GLOBAL_TIMEOUT_SECS = max(180 * 60, num_agents * 12)  # ~200min for 1000  # pylint: disable=invalid-name

    logging.info(
        'Pipelined generation: %d agents, %d workers...',
        num_agents,
        max_workers,
    )
    pipeline_start = time.time()

    # Single shared executor for maximum saturation
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      # Track all formative futures so we can wait for them
      formative_futures = []

      def _on_worldview_done(future, entry):
        """Callback: worldview done → submit formative memory."""
        name = entry['config'].name
        try:
          wm, wt = future.result()
          with lock:
            worldview_results[name] = (wm, wt)
          if progress_callback:
            progress_callback(
                'worldview', len(worldview_results), num_agents, name
            )

          # Immediately chain the formative memory call
          try:
            fm_future = executor.submit(
                self._generate_formative_memories,
                agent_config=entry['config'],
                traits=entry['traits'],
                economic_class=entry['economic_class'],
                worldview=wt,
            )
            with lock:
              formative_futures.append((name, fm_future))
          except RuntimeError:
            # Executor already shut down (timeout or stall)
            logging.warning(
                'Executor shut down before formative for %s could start.',
                name,
            )
            with lock:
              formative_results[name] = []
              errors.append((name, 'formative', 'executor shutdown'))

        except Exception as e:  # pylint: disable=broad-except
          logging.error('Worldview error for %s: %s', name, e)
          with lock:
            worldview_failed.add(name)
            errors.append((name, 'worldview', str(e)))

      # Submit all worldview calls
      for entry in agent_data.values():
        wv_future = executor.submit(
            self._generate_behavioral_logic,
            agent_config=entry['config'],
            traits=entry['traits'],
            economic_class=entry['economic_class'],
        )
        # Add a done-callback that chains the formative call
        wv_future.add_done_callback(lambda f, e=entry: _on_worldview_done(f, e))

      # Wait for all formative futures to complete
      # We need to poll since formative_futures grows dynamically
      last_progress_time = time.time()
      last_combined_count = 0
      while True:
        with lock:
          total_done = len(formative_results)
          total_skipped = len(worldview_failed)
          total_worldviews = len(worldview_results)
          pending = [
              (n, f) for n, f in formative_futures if n not in formative_results
          ]
          # Combined progress = worldviews + formatives (any movement is ok)
          combined_count = total_worldviews + total_done
        # Exit when all agents are accounted for
        if total_done + total_skipped >= num_agents:
          break
        # Global timeout safety valve
        elapsed = time.time() - pipeline_start
        if elapsed > _GLOBAL_TIMEOUT_SECS:
          logging.error(
              'Global timeout reached (%.0fs). %d/%d done, %d skipped, '
              '%d pending. Stopping.',
              elapsed,
              total_done,
              num_agents,
              total_skipped,
              len(pending),
          )
          # Mark all pending as timed-out
          for name, fm_future in pending:
            fm_future.cancel()
            with lock:
              formative_results[name] = []
              errors.append((name, 'formative', 'global timeout'))
          break
        # Stall detection: no progress on EITHER worldviews or formatives
        if combined_count != last_combined_count:
          last_combined_count = combined_count
          last_progress_time = time.time()
        elif time.time() - last_progress_time > _STALL_TIMEOUT_SECS:
          logging.warning(
              'No progress for %ds. Worldviews: %d/%d, Formatives: %d, '
              '%d pending futures appear stalled. Cancelling stragglers.',
              _STALL_TIMEOUT_SECS,
              total_worldviews,
              num_agents,
              total_done,
              len(pending),
          )
          for name, fm_future in pending:
            fm_future.cancel()
            with lock:
              formative_results[name] = []
              errors.append((name, 'formative', 'stall timeout'))
            if progress_callback:
              progress_callback(
                  'formative', len(formative_results), num_agents, name
              )
          break

        for name, fm_future in pending:
          try:
            fm = fm_future.result(timeout=0.1)
            with lock:
              formative_results[name] = fm
            if progress_callback:
              progress_callback(
                  'formative', len(formative_results), num_agents, name
              )
          except concurrent.futures.TimeoutError:
            continue
          except Exception as e:  # pylint: disable=broad-except
            logging.error('Formative error for %s: %s', name, e)
            with lock:
              formative_results[name] = []  # Empty fallback
              errors.append((name, 'formative', str(e)))

    if errors:
      logging.warning('%d errors during generation:', len(errors))
      for name, stage, err in errors:
        logging.warning('  %s (%s): %s', name, stage, err)
      formative_errors = [e for e in errors if e[1] == 'formative']
      if formative_errors:
        raise RuntimeError(
            f'{len(formative_errors)} personas failed formative-memory '
            f'generation (refusing to save empty memories): {formative_errors}'
        )

    logging.info(
        'Pipeline complete: %d worldviews, %d formative sets.',
        len(worldview_results),
        len(formative_results),
    )

    # ── Assembly (no LLM calls) ──
    personas: Dict[str, PersonaData] = {}
    for entry in agent_data.values():
      ac = entry['config']
      name = ac.name
      if name not in worldview_results:
        continue  # Skip errored agents
      worldview_memory, _ = worldview_results[name]

      context_memory = (
          f'{name} has lived in {self._population.name}'
          f' for several years. {self._population.context}'
      )
      memories = [context_memory]
      if not ac.work_place:
        raise ValueError(
            f'Agent {name} is missing a work_place in configuration.'
        )
      memories.append(f'{name} works at the {ac.work_place}.')
      if worldview_memory:
        memories.append(worldview_memory)

      personas[name] = PersonaData(
          name=name,
          age=ac.age,
          economic_class=entry['economic_class'],
          home_place=ac.home_place,
          work_place=ac.work_place,
          personality=ac.personality,
          original_backstory=ac.backstory,
          traits=copy.deepcopy(entry['traits']),
          memories=memories,
          formative_memories=formative_results.get(name, []),
          gender=getattr(ac, 'gender', ''),
          sexual_orientation=getattr(ac, 'sexual_orientation', ''),
          ethnicity=getattr(ac, 'ethnicity', ''),
          political_orientation=getattr(ac, 'political_orientation', ''),
          hobbies=getattr(ac, 'hobbies', []),
          relationship_status=getattr(ac, 'relationship_status', 'single'),
          neighborhood=getattr(ac, 'neighborhood', ''),
      )

    logging.info('Assembled %d personas.', len(personas))
    return personas

  def generate_traits_only(
      self,
      agent_configs: Sequence[agents_lib.AgentConfig] | None = None,
  ) -> Dict[str, PersonaData]:
    """Generate personas with QMC traits only (no LLM calls)."""
    if agent_configs is None:
      agent_configs = self._population.agent_configs
    num_agents = len(agent_configs)
    trait_points = _generate_sobol_points(
        dimension_ranges=DIMENSION_RANGES,
        dimensions=DIVERSITY_AXES,
        num_points=num_agents,
    )

    personas: Dict[str, PersonaData] = {}
    for i, agent_config in enumerate(agent_configs):
      traits = trait_points[i]
      economic_class = _get_economic_class(
          agent_config.home_place,
          self._population.economic_class_map,
      )

      context_memory = (
          f'{agent_config.name} has lived on {self._population.name}'
          f' for several years. {self._population.context}'
      )

      personas[agent_config.name] = PersonaData(
          name=agent_config.name,
          age=agent_config.age,
          economic_class=economic_class,
          home_place=agent_config.home_place,
          work_place=agent_config.work_place,
          personality=agent_config.personality,
          original_backstory=agent_config.backstory,
          traits=copy.deepcopy(traits),
          memories=[context_memory],
          formative_memories=[],
          gender=getattr(agent_config, 'gender', ''),
          sexual_orientation=getattr(agent_config, 'sexual_orientation', ''),
          ethnicity=getattr(agent_config, 'ethnicity', ''),
          political_orientation=getattr(
              agent_config, 'political_orientation', ''
          ),
          hobbies=getattr(agent_config, 'hobbies', []),
          relationship_status=getattr(
              agent_config, 'relationship_status', 'single'
          ),
          neighborhood=getattr(agent_config, 'neighborhood', ''),
      )

    return personas

  def _build_trait_block(self, traits: Dict[str, float]) -> str:
    """Build human-readable trait description block using natural language."""
    trait_lines = []
    for axis in DIVERSITY_AXES:
      val = traits[axis]
      info = AXIS_DESCRIPTIONS[axis]
      low_val, high_val = DIMENSION_RANGES[axis]
      range_size = high_val - low_val
      normalized = (val - low_val) / range_size if range_size > 0 else 0.5

      if normalized < 0.2:
        intensity = 'very low'
      elif normalized < 0.4:
        intensity = 'low'
      elif normalized < 0.6:
        intensity = 'moderate'
      elif normalized < 0.8:
        intensity = 'high'
      else:
        intensity = 'very high'

      position = 'low' if normalized < 0.5 else 'high'
      archetype = info[position]
      trait_lines.append(f'  {info["description"]}: {intensity} — {archetype}')
    return '\n'.join(trait_lines)

  def _build_player_context(
      self,
      agent_config: agents_lib.AgentConfig,
      traits: Dict[str, float],
      economic_class: str,
  ) -> str:
    """Build the player context string for LLM prompts."""
    trait_block = self._build_trait_block(traits)
    hobbies_str = (
        ', '.join(agent_config.hobbies)
        if agent_config.hobbies
        else 'none specified'
    )

    # Sanitize backstory: replace any LLM-generated name with the
    # census-assigned name so the LLM sees a consistent identity.
    backstory = agent_config.backstory or ''
    if backstory and agent_config.name:
      census_first = agent_config.name.split()[0]
      # Find the first capitalized word that isn't a common starter
      skip_words = {
          'A',
          'An',
          'The',
          'After',
          'Before',
          'When',
          'Once',
          'Having',
          'With',
          'As',
          'In',
          'On',
          'At',
          'For',
          'Born',
          'Growing',
          'Recently',
          'Originally',
          'Following',
          'During',
          'Since',
      }
      for w in backstory.split():
        clean = re.sub(r'[^A-Za-z]', '', w)
        if clean and clean[0].isupper() and clean not in skip_words:
          if clean != census_first:
            backstory = re.sub(
                r'\b' + re.escape(clean) + r'\b', census_first, backstory
            )
          break

    lines = [
        f'Name: {agent_config.name}',
        f'Age: {agent_config.age}',
        f'Gender: {agent_config.gender or "not specified"}',
        f'Ethnicity: {agent_config.ethnicity or "not specified"}',
        (
            'Sexual orientation:'
            f' {agent_config.sexual_orientation or "not specified"}'
        ),
        f'Relationship status: {agent_config.relationship_status}',
        f'Economic class: {economic_class}',
        f'Home: {agent_config.home_place}',
        f'Occupation: {agent_config.work_place or "unemployed/retired"}',
        f'Personality: {agent_config.personality}',
        f'Background: {backstory}',
        (
            'Political orientation:'
            f' {agent_config.political_orientation or "not specified"}'
        ),
        f'Hobbies: {hobbies_str}',
        '',
        f'Psychological tendencies:\n{trait_block}',
    ]

    # Add interpretive notes for extreme trait values so the LLM
    # understands what low/high scores mean in behavioral terms.
    notes = []
    if traits.get('agreeableness', 3.0) < 2.0:
      notes.append(
          f'{agent_config.name} is genuinely disagreeable — blunt, '
          'confrontational, and prioritizes their own needs over social '
          'harmony. They are NOT kind or warm.'
      )
    if traits.get('agreeableness', 3.0) < 2.5:
      notes.append(
          f'{agent_config.name} tends toward selfishness, impatience, '
          'and low empathy. They may be difficult to work with.'
      )
    if traits.get('locus_of_control', 0.5) < 0.25:
      notes.append(
          f'{agent_config.name} has a strongly external locus of control — '
          'they feel life happens TO them, that luck and circumstances '
          'matter more than personal effort or choices.'
      )
    if traits.get('social_trust', 0.5) < 0.2:
      notes.append(
          f'{agent_config.name} is deeply suspicious of others — they '
          'assume people have hidden agendas, are slow to trust, and '
          'may be paranoid or cynical.'
      )
    if traits.get('neuroticism', 3.0) > 4.0:
      notes.append(
          f'{agent_config.name} is highly neurotic — prone to anxiety, '
          'worry, emotional instability, and catastrophic thinking. '
          'They struggle with stress.'
      )
    if traits.get('extraversion', 3.0) < 2.0:
      notes.append(
          f'{agent_config.name} is deeply introverted — they find social '
          'interaction draining, prefer solitude, and are uncomfortable '
          'being the center of attention.'
      )
    if traits.get('conscientiousness', 3.0) < 2.0:
      notes.append(
          f'{agent_config.name} is disorganized and impulsive — they '
          'struggle with planning, deadlines, and follow-through.'
      )
    if notes:
      lines.append('')
      lines.append(
          'IMPORTANT behavioral notes (these MUST be reflected in the profile):'
      )
      for note in notes:
        lines.append(f'  - {note}')

    return '\n'.join(lines)

  def _generate_behavioral_logic(
      self,
      agent_config: agents_lib.AgentConfig,
      traits: Dict[str, float],
      economic_class: str,
  ) -> tuple[str, str]:
    """Generate a detailed psychological profile and behavioral example.

    Produces a rich narrative covering social behavior, mental health,
    political beliefs, civic engagement, substance use, economic situation,
    and a concrete behavioral example. No raw trait numbers appear in output.

    Args:
      agent_config: The agent's base configuration.
      traits: Psychological trait scores (used as LLM input only).
      economic_class: Economic class label.

    Returns:
      Tuple of (full_memory_text, profile_only_text).
    """
    name = agent_config.name
    player_context = self._build_player_context(
        agent_config, traits, economic_class
    )

    try:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(
          'You are an expert in social psychology, sociology, and behavioral'
          ' science. Your task is to write a detailed psychological profile'
          ' of a character for a social simulation. You will describe this'
          " person's inner life, social behavior, and worldview as a rich"
          ' narrative — like a clinical case study or ethnographic portrait.'
          ' NEVER include any numerical scores, ratings, or scales in your'
          ' output. Describe everything in vivid, natural language.'
      )
      prompt.statement(f'The shared setting is:\n{self._population.context}')
      prompt.statement(
          f"The full persona profile for '{name}' is:\n{player_context}"
      )

      profile_q = (
          f'Write a detailed psychological profile of {name} as a rich,'
          ' multi-paragraph narrative. Write in third person. Cover ALL of the'
          ' following dimensions in vivid, specific language:\n\n1. SOCIAL'
          ' PERSONALITY: How do they behave in groups? Are they the center of'
          ' attention or the quiet observer? Do they make friends easily or'
          " keep people at arm's length? Are they warm or prickly? Use"
          ' metaphors — "social butterfly," "wallflower," "the person who'
          ' corners you at a party."\n\n2. DECISION-MAKING & VALUES: What'
          ' principles guide their choices? Are they spontaneous or'
          ' methodical? Do they trust their gut or need data? How do they'
          ' handle moral gray areas?\n\n3. MENTAL HEALTH & EMOTIONAL LIFE: How'
          ' do they handle stress and anxiety? Are they prone to worry,'
          ' depression, or anger? Do they bottle things up or let it all out?'
          ' Any tendencies toward substance use (alcohol, drugs, smoking) as'
          ' coping?\n\n4. POLITICAL BELIEFS & CIVIC ENGAGEMENT: What are their'
          ' political leanings and why? Do they vote, volunteer, protest? Do'
          ' they have any controversial opinions they keep quiet about or'
          ' loudly proclaim? How do they react to political'
          ' disagreement?\n\n5. ECONOMIC SITUATION & CLASS CONSCIOUSNESS: How'
          ' do they feel about their economic position? Do they feel secure or'
          ' precarious? How do they react to financial stress — panic, denial,'
          ' hustle? What is their relationship with money and spending?\n\n6.'
          ' COMMUNITY & RELATIONSHIPS: How do they fit into the community? Are'
          ' they a joiner or a loner? What role do they play in their'
          ' neighborhood? How do they handle conflict with neighbors,'
          ' coworkers, or family?\n\nCRITICAL RULES:\n- NEVER mention any'
          ' numerical scores, scales, ratings, or quantitative measures\n- Use'
          ' evocative, specific language — not clinical jargon\n- Make every'
          ' sentence reveal character, not state abstract traits\n- The'
          ' profile should feel like something a novelist or documentary'
          ' filmmaker would write\n- Minimum 4 substantial paragraphs\n'
          '- If the IMPORTANT behavioral notes say this person is disagreeable,'
          ' suspicious, impulsive, or neurotic, the profile MUST show these'
          ' traits prominently — do NOT soften them into warmth\n'
          '- At least 30%% of the profile should cover weaknesses,'
          ' contradictions, or unflattering truths\n'
          '- AVOID making every persona secretly warm underneath. Some people'
          ' are genuinely difficult, selfish, or cold — that is fine\n'
          '- Low social trust means suspicious and guarded, NOT trusting\n'
          '- External locus of control means fatalistic and resigned, NOT'
          ' proactive'
      )
      profile = prompt.open_question(
          profile_q,
          max_tokens=2500,
          terminators=[],
          temperature=0.9,
      ).strip()

      # Post-processing: strip any leaked numbers or score references
      _re = re  # reuse top-level import

      profile = _re.sub(r'\b\d+\.\d+\b', '', profile)
      profile = _re.sub(r'\(score[^)]*\)', '', profile)
      profile = _re.sub(r'\(range[^)]*\)', '', profile)

      prompt.statement(f'PSYCHOLOGICAL PROFILE:\n{profile}')

      situation_q = (
          'Now, create a brief, specific, and socially-charged hypothetical'
          f' situation that {name} might encounter in'
          f' {self._population.name}.'
          ' This situation must directly challenge their values or comfort'
          ' zone and force them to make a decision that reveals their true'
          ' character. Frame the situation concretely — who is involved,'
          ' where it happens, what the stakes are.'
      )
      situation = prompt.open_question(
          situation_q,
          max_tokens=500,
          terminators=[],
          temperature=1.0,
      ).strip()
      prompt.statement(f'THE SITUATION:\n{situation}')

      reaction_q = (
          f'Describe in a detailed paragraph how {name} would react to'
          ' this specific situation. Detail their actions, words, body'
          ' language, and internal thoughts. The reaction must feel'
          ' psychologically authentic — consistent with the profile above.'
          ' Show their personality through behavior, not by restating traits.'
          ' NEVER reference any numerical scores.'
      )
      reaction = prompt.open_question(
          reaction_q,
          max_tokens=1500,
          terminators=[],
          temperature=1.0,
      ).strip()

      full_memory = (
          f"{name}'s psychological profile:\n"
          f'{profile}\n\n'
          'An example of how this manifests in practice:\n'
          f'SITUATION: {situation}\n\n'
          f'REACTION: {reaction}'
      )
      return full_memory, profile

    except (RuntimeError, ValueError):
      logging.exception('Error generating behavioral logic for %s', name)
      return '', ''

  def _generate_formative_memories(
      self,
      agent_config: agents_lib.AgentConfig,
      traits: Dict[str, float],
      economic_class: str,
      worldview: str,
  ) -> List[str]:
    """Generate chronological formative memories via a single LLM call.

    Produces 10 biographical vignettes heavily weighted toward adulthood,
    covering career path, location origin, romantic history, hobbies,
    community involvement, and personality-revealing anecdotes.

    Args:
      agent_config: The agent's base configuration.
      traits: Psychological trait scores (used as LLM input only).
      economic_class: Economic class label.
      worldview: The psychological profile text from Stage 2.

    Returns:
      List of formative memory strings tagged [formative].
    """
    name = agent_config.name
    age = agent_config.age
    player_context = self._build_player_context(
        agent_config, traits, economic_class
    )
    hobbies_str = (
        ', '.join(agent_config.hobbies)
        if agent_config.hobbies
        else 'various interests'
    )

    # Build relationship instruction
    rel_status = agent_config.relationship_status
    if 'married to ' in rel_status or 'partnered with ' in rel_status:
      partner_name = (
          rel_status.replace('married to ', '')
          .replace('partnered with ', '')
          .strip()
      )
      home_desc = (
          f'in their shared home ({agent_config.home_place})'
          if agent_config.home_place
          else ''
      )
      relationship_instruction = (
          f'  * How {name} met their spouse/partner, {partner_name} — the story'
          ' of how they fell in love, got married, and their life together'
          f' {home_desc}. You MUST use the exact spouse name "{partner_name}"\n'
      )
    elif rel_status == 'married':
      relationship_instruction = (
          f'  * How {name} met their spouse — the story of how they fell in'
          ' love, got married, and their home life together\n'
      )
    else:
      relationship_instruction = (
          f"  * {name}'s deepest failed romantic relationship — who it was,"
          ' why it ended, and how it changed them. Make this specific and'
          ' emotionally resonant, not generic\n'
      )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        'You are creating a detailed biographical history for a'
        ' character in a social simulation. Each memory should read'
        ' like a vivid scene from a novel — specific details, emotional'
        ' texture, and character revelation. NEVER include any numerical'
        ' scores or ratings.'
    )
    prompt.statement(
        f'Setting: {self._population.name}\n{self._population.context}'
    )
    prompt.statement(
        f'Character profile:\n{player_context}\n\n'
        f'Psychological profile:\n{worldview}'
    )

    # Randomize childhood age to break the "five-year-old" attractor
    childhood_start_age = random.choice([4, 6, 7, 8, 9])
    childhood_end_age = min(12, age)
    teen_start = min(13, age)
    teen_end = min(18, age)

    # Diverse opening scenarios (rotated by agent index)
    opening_scenarios = [
        (
            'a solitary moment of discovery — finding something unexpected'
            ' in nature, a drawer, a book, or a hidden place'
        ),
        (
            'a conflict with a sibling, cousin, or childhood friend that'
            ' revealed their personality'
        ),
        (
            'a moment of fear or wonder — a storm, a first trip somewhere'
            ' unfamiliar, an encounter with an animal'
        ),
        (
            'an act of defiance or independence — doing something forbidden,'
            ' standing up to someone, or making a choice their family disagreed'
            ' with'
        ),
        (
            'a sensory memory tied to a specific place — the smell of a room,'
            ' the texture of something they touched, a sound that stuck with'
            ' them'
        ),
        (
            'a moment of embarrassment or shame that shaped how they see'
            ' themselves'
        ),
        'watching an adult do something that fascinated or disturbed them',
        (
            'a game, ritual, or routine that was uniquely theirs — something'
            ' no one else understood or shared'
        ),
    ]
    # Use agent name hash to rotate through scenarios
    scenario_idx = hash(name) % len(opening_scenarios)
    assigned_scenario = opening_scenarios[scenario_idx]

    memory_q = (
        f'Generate exactly 10 formative memories for {name} in'
        ' chronological order. Each memory should be 3-4 sentences'
        ' long — vivid, specific, and psychologically revealing.'
        ' Write in third person.\n\n'
        'Structure:\n'
        f'- 2 childhood memories (ages {childhood_start_age}-'
        f'{childhood_end_age}):\n'
        f'  * The FIRST memory must depict: {assigned_scenario}.'
        ' Ground it in a specific place, with sensory details'
        ' (sounds, smells, textures, light).\n'
        '  * The second childhood memory: a defining school or'
        ' neighborhood moment that shaped their social identity\n'
        f'- 1 teenage memory (ages {teen_start}-{teen_end}): identity'
        ' formation, first serious friendship or rivalry, a pivotal'
        ' decision\n'
        f'-  adult memories (ages 19-{age}). These MUST include:\n'
        f'  * How {name} came to live in {self._population.name}'
        ' — what brought them there, what they left behind\n'
        '  * How they entered their career path'
        f' ({agent_config.work_place or "their current situation"})'
        ' — not just the job, but why this work and what it means to them\n'
        f'{relationship_instruction}'
        f'  * Their relationship with their hobbies ({hobbies_str})'
        ' — a specific memory of doing this activity that reveals'
        ' who they are\n'
        '  * Their role in the community — a time they stepped up'
        ' (or failed to) for a neighbor, colleague, or stranger\n'
        '  * A personality-revealing anecdote — a habit, quirk,'
        ' guilty pleasure, or recurring pattern that captures their'
        ' essence. This should make them feel like a real person\n\n'
        'RULES:\n'
        '- Each memory must include concrete details: names of places,'
        ' specific objects, weather, time of day, smells, sounds\n'
        '- Show personality through ACTIONS and REACTIONS,'
        ' not by stating traits\n'
        '- Include at least one memory where they are NOT at their best'
        ' — being petty, selfish, scared, or wrong\n'
        '- Vary the emotional tone: not every memory should be dramatic\n\n'
        'Format each memory on its own line as:\n'
        '[N] Memory text'
    )

    try:
      raw_response = prompt.open_question(
          memory_q,
          max_tokens=5000,
          terminators=[],
          temperature=0.9,
      ).strip()

      # Parse numbered memories — handle multi-line entries
      memories = []
      current_memory = None
      for line in raw_response.split('\n'):
        line = line.strip()
        if not line:
          continue
        # Check if this line starts a new numbered entry
        match = re.search(r'^\[?\d+[\].)]?\s*(.+)', line)
        if match:
          # Save the previous memory if any
          if current_memory is not None:
            memories.append(f'[formative] {current_memory}')
          current_memory = match.group(1).strip()
        elif current_memory is not None:
          # Continuation of the current memory
          current_memory += ' ' + line
      # Don't forget the last memory
      if current_memory is not None:
        memories.append(f'[formative] {current_memory}')

      if not memories:
        # Fallback: treat entire response as one memory
        logging.warning(
            'Could not parse formative memories for %s, using raw.',
            name,
        )
        memories = [f'[formative] {raw_response}']

      logging.info(
          'Generated %d formative memories for %s (requested 10).',
          len(memories),
          name,
      )
      return memories

    except (RuntimeError, ValueError):
      logging.exception('Error generating formative memories for %s', name)
      raise


# === Local & Bundled Persistence ===

DEFAULT_PERSONAS_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'populations'
)


def save_personas(
    personas: Dict[str, PersonaData],
    cns_path: str = DEFAULT_PERSONAS_BASE_PATH,
    date_label: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
  """Save generated personas to local directory as individual JSON files.

  Args:
    personas: Personas keyed by name.
    cns_path: Parent directory.
    date_label: Sub-directory name (the label `load_personas` takes).
    extra_metadata: Extra keys for metadata.json, e.g. the `setting_preset`
      that run.py uses to pick sim/locations.py places.

  Returns:
    The directory the personas were written to.
  """
  if date_label is None:
    date_label = datetime.datetime.now().strftime('%Y-%m-%d')
  output_dir = os.path.join(cns_path, date_label)
  os.makedirs(output_dir, exist_ok=True)

  metadata = {
      'generated_at': datetime.datetime.now().isoformat(),
      'date_label': date_label,
      'num_agents': len(personas),
      'axes': list(DIVERSITY_AXES),
      'dimension_ranges': {k: list(v) for k, v in DIMENSION_RANGES.items()},
      **(extra_metadata or {}),
  }

  metadata_path = os.path.join(output_dir, 'metadata.json')
  with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  logging.info('Saved metadata to %s', metadata_path)

  for name, persona in personas.items():
    sanitized = name.replace(' ', '_').lower()
    file_path = os.path.join(output_dir, f'{sanitized}_persona.json')
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(persona.to_dict(), f, indent=2)

  logging.info('Saved %d personas to %s', len(personas), output_dir)
  return output_dir


def load_personas(
    cns_path: str = DEFAULT_PERSONAS_BASE_PATH,
    date_label: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    max_workers: int = 32,
) -> Dict[str, PersonaData]:
  """Load personas from a local directory, JSON bundle, or built-in cohort."""
  import concurrent.futures  # pylint: disable=g-import-not-at-top

  if date_label is None:
    date_label = 'brecksville_ohio'

  # 1. Check direct path or cns_path/date_label on local filesystem
  candidates = [
      date_label,
      os.path.join(cns_path, date_label) if cns_path else '',
      os.path.join(DEFAULT_PERSONAS_BASE_PATH, date_label),
  ]
  for cand in candidates:
    if not cand:
      continue
    if os.path.isfile(cand) and cand.endswith('.json'):
      with open(cand, 'r', encoding='utf-8') as f:
        raw = json.load(f)
      p_map = raw.get('personas', raw)
      return {k: PersonaData.from_dict(v) for k, v in p_map.items()}
    if os.path.isdir(cand):
      if names is not None:
        persona_files = [
            os.path.join(cand, f'{n.replace(" ", "_").lower()}_persona.json')
            for n in names
            if os.path.isfile(
                os.path.join(
                    cand, f'{n.replace(" ", "_").lower()}_persona.json'
                )
            )
        ]
      else:
        persona_files = sorted(
            glob_mod.glob(os.path.join(cand, '*_persona.json'))
        )
      if persona_files:
        def _load_one(fp):
          with open(fp, 'r', encoding='utf-8') as f:
            return PersonaData.from_dict(json.load(f))
        personas = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as ex:
          for p in ex.map(_load_one, persona_files):
            personas[p.name] = p
        logging.info(
            'Loaded %d personas from local directory %s', len(personas), cand
        )
        return personas

  # Only persona sets produced by the full pipeline (generate_population.py ->
  # generate_personas.py) are loaded. Never synthesize stand-ins.
  raise FileNotFoundError(
      f'Could not locate personas for date_label={date_label!r} (checked '
      f'{[c for c in candidates if c]!r}). Expected a directory of '
      '*_persona.json files or a JSON bundle. The bundled population is '
      "'brecksville_ohio'; to build another one, see personas/README.md."
  )


def save_personas_local(
    personas: Dict[str, PersonaData],
    output_path: str,
) -> None:
  """Save personas to local filesystem as a single JSON file."""
  data = {
      'metadata': {
          'generated_at': datetime.datetime.now().isoformat(),
          'num_agents': len(personas),
          'axes': list(DIVERSITY_AXES),
          'dimension_ranges': {k: list(v) for k, v in DIMENSION_RANGES.items()},
      },
      'personas': {
          name: persona.to_dict() for name, persona in personas.items()
      },
  }
  if os.path.dirname(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
  logging.info('Saved %d personas to %s (local)', len(personas), output_path)
