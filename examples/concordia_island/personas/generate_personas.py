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

r"""Step 2 of persona generation: turn a roster into full personas.

See `personas/README.md` for the whole pipeline. Reads a roster written by
`personas/populations/generate_population.py`, runs `PersonaGenerator` on every
agent (Sobol trait sampling, worldview chain, formative memories) and writes
one `<name>_persona.json` per agent plus `metadata.json` to
`<output_dir>/<date_label>/`. `run.py --personas_date=<date_label>` then loads
the set, and picks the matching `sim/locations.py` preset from the
`setting_preset` recorded in `metadata.json`.

The script stops with an error if any agent fails, rather than saving a
partial or padded population.

Usage:
  python -m examples.concordia_island.personas.generate_personas \
    --input_roster=/tmp/brecksville_roster.json \
    --date_label=my_brecksville \
    --api_type=google_aistudio \
    --model_name=gemini-2.5-flash
"""

from collections.abc import Mapping, Sequence
import json
import logging
import os
import sys
import threading
from typing import Any

from absl import app
from absl import flags
from concordia.language_model import language_model
from examples.concordia_island.personas import generator as generator_lib
from examples.concordia_island.personas.populations import generate_population
from examples.concordia_island.sim import agents as agents_lib

FLAGS = flags.FLAGS


def _define_flags() -> None:
  """Defines the command-line flags (only when run as a script).

  Keeping them out of import time lets tests import this module alongside
  run.py, which defines some of the same flag names.
  """
  flags.DEFINE_string(
      'input_roster',
      None,
      'Roster JSON written by personas/populations/generate_population.py.',
      required=True,
  )
  flags.DEFINE_string(
      'date_label',
      None,
      'Name of the output persona set (pass it to run.py --personas_date).',
      required=True,
  )
  flags.DEFINE_string(
      'output_dir',
      generator_lib.DEFAULT_PERSONAS_BASE_PATH,
      'Parent directory for the persona set. Defaults to'
      ' personas/populations/, where run.py looks first.',
  )
  flags.DEFINE_string(
      'api_type',
      'google_aistudio',
      'Concordia language_model_setup api_type (e.g. google_aistudio,'
      ' openai, vllm, ollama, huggingface).',
  )
  flags.DEFINE_string(
      'model_name',
      'gemini-2.5-flash',
      'Model name for language_model_setup.',
  )
  flags.DEFINE_string(
      'api_key',
      None,
      'API key (falls back to GOOGLE_API_KEY, GEMINI_API_KEY or'
      ' OPENAI_API_KEY).',
  )
  flags.DEFINE_integer(
      'workers', 16, 'Concurrent LLM calls during generation.'
  )


def roster_to_agent_configs(
    roster_agents: Sequence[Mapping[str, Any]],
) -> list[agents_lib.AgentConfig]:
  """Converts roster entries to AgentConfigs.

  Args:
    roster_agents: The `agents` list of a roster JSON.

  Returns:
    One AgentConfig per roster entry.

  Raises:
    ValueError: If an entry lacks a field the persona pipeline needs (dry-run
      rosters have no personality or backstory and are rejected here).
  """
  required = ('name', 'home_place', 'work_place', 'age', 'personality',
              'backstory')
  configs = []
  for i, a in enumerate(roster_agents):
    missing = [k for k in required if not a.get(k)]
    if missing:
      raise ValueError(
          f'Roster agent {i} ({a.get("name", "?")}) is missing {missing}. Was'
          ' the roster built with --dry_run?'
      )
    configs.append(
        agents_lib.AgentConfig(
            name=a['name'],
            home_place=a['home_place'],
            work_place=a['work_place'],
            personality=a['personality'],
            backstory=a['backstory'],
            age=a['age'],
            gender=a.get('gender', ''),
            sexual_orientation=a.get('sexual_orientation', ''),
            ethnicity=a.get('ethnicity', ''),
            political_orientation=a.get('political_orientation', ''),
            hobbies=list(a.get('hobbies', [])),
            relationship_status=a.get('relationship_status', 'single'),
            neighborhood=a.get('neighborhood', ''),
        )
    )
  return configs


def population_config_for(
    roster: Mapping[str, Any],
) -> generator_lib.PopulationConfig:
  """Builds the PopulationConfig (community name, context, class map)."""
  meta = roster['metadata']
  setting = meta['setting']
  setting_config = generate_population.get_setting_config(setting)
  tiers = setting_config.get(
      'economic_tiers', generate_population.ECONOMIC_TIERS
  )
  return generator_lib.PopulationConfig(
      name=meta['location_name'],
      context=meta['context'],
      economic_class_map={cfg['prefix']: t for t, cfg in tiers.items()},
      agent_configs=roster_to_agent_configs(roster['agents']),
  )


def generate_personas(
    model: language_model.LanguageModel,
    roster: Mapping[str, Any],
    output_dir: str,
    date_label: str,
    workers: int = 16,
) -> str:
  """Generates and saves personas for every agent in a roster.

  Args:
    model: Language model for the worldview and formative-memory stages.
    roster: Parsed roster JSON (`{"metadata": ..., "agents": [...]}`).
    output_dir: Parent directory for the persona set.
    date_label: Name of the persona set directory.
    workers: Concurrent LLM calls.

  Returns:
    The directory the personas were written to.

  Raises:
    FileExistsError: If the output directory already has files.
    RuntimeError: If any agent did not get a persona.
  """
  target = os.path.join(output_dir, date_label)
  if os.path.isdir(target) and os.listdir(target):
    raise FileExistsError(
        f'{target} already exists and is not empty; pick another'
        ' --date_label or remove it.'
    )
  population = population_config_for(roster)
  gen = generator_lib.PersonaGenerator(model=model, population=population)

  lock = threading.Lock()
  done = {'worldview': 0, 'formative': 0}
  total = len(population.agent_configs)

  def _progress(stage, unused_current, unused_total, name):
    with lock:
      done[stage] = done.get(stage, 0) + 1
      sys.stderr.write(
          f'\r  worldview {done["worldview"]}/{total}  formative'
          f' {done["formative"]}/{total}  {name[:30]:<30}'
      )
      sys.stderr.flush()

  personas = gen.generate_all_personas_parallel(
      agent_configs=population.agent_configs,
      max_workers=workers,
      progress_callback=_progress,
  )
  sys.stderr.write('\n')
  missing = sorted(
      ac.name for ac in population.agent_configs if ac.name not in personas
  )
  if missing:
    raise RuntimeError(
        f'{len(missing)} of {total} agents have no persona (worldview step'
        f' failed); nothing was saved. First few: {missing[:10]}'
    )
  meta = roster['metadata']
  return generator_lib.save_personas(
      personas,
      cns_path=output_dir,
      date_label=date_label,
      extra_metadata={
          'setting': meta['location_name'],
          'setting_preset': meta['setting_preset'],
          'roster_seed': meta.get('seed'),
      },
  )


def main(argv: Sequence[str]) -> None:
  del argv
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(message)s',
      stream=sys.stderr,
  )
  with open(FLAGS.input_roster, 'r', encoding='utf-8') as f:
    roster = json.load(f)
  if roster['metadata'].get('dry_run'):
    raise ValueError(
        f'{FLAGS.input_roster} is a dry-run roster (no personalities or'
        ' backstories). Rebuild it without --dry_run.'
    )

  from concordia.contrib.language_models import language_model_setup  # pylint: disable=g-import-not-at-top

  api_key = (
      FLAGS.api_key
      or os.environ.get('GOOGLE_API_KEY', '')
      or os.environ.get('GEMINI_API_KEY', '')
      or os.environ.get('OPENAI_API_KEY', '')
      or None
  )
  model = language_model_setup(
      api_type=FLAGS.api_type, model_name=FLAGS.model_name, api_key=api_key
  )
  saved = generate_personas(
      model,
      roster,
      output_dir=FLAGS.output_dir,
      date_label=FLAGS.date_label,
      workers=FLAGS.workers,
  )
  logging.info(
      'Saved %d personas to %s. Run them with: python -m'
      ' examples.concordia_island.run --personas_date=%s%s',
      len(roster['agents']),
      saved,
      FLAGS.date_label,
      ''
      if FLAGS.output_dir == generator_lib.DEFAULT_PERSONAS_BASE_PATH
      else f' --personas_dir={FLAGS.output_dir}',
  )


if __name__ == '__main__':
  _define_flags()
  app.run(main)
