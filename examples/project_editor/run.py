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

"""What should the roommates play? A private editor with opt-in language models."""

import argparse
import dataclasses
import getpass
import math
import os
from pathlib import Path
import sys
import time
from typing import ClassVar
from urllib.parse import urlsplit
import uuid
import warnings

from concordia.contrib import language_models
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_config
from concordia.utils import simulation_server
import numpy as np

from examples.project_editor import template


@dataclasses.dataclass(frozen=True)
class ModelSelection:
  """Process-local model settings, never part of an authored project."""

  _runtime_api_key: ClassVar[str | None] = None
  _credential_source: ClassVar[str] = 'not initialized'
  backend: str = 'none'
  model_name: str | None = None
  # InitVar is intentionally excluded from repr, fields and dataclasses.asdict.
  api_key: dataclasses.InitVar[str | None] = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self, api_key: str | None) -> None:
    if api_key is not None and self.backend != 'together_ai':
      raise ValueError('A prompted key requires the together_ai backend.')
    object.__setattr__(self, '_runtime_api_key', api_key)
    if not self.backend.strip():
      raise ValueError('Model backend must not be empty.')
    if self.backend == 'none':
      if self.model_name is not None:
        raise ValueError('--model-name requires a backend other than none.')
    elif not self.model_name or not self.model_name.strip():
      raise ValueError('A live backend requires --model-name.')

  @property
  def label(self) -> str:
    if self.backend == 'none':
      return 'Free mock · NoLanguageModel · output is a development stub'
    label = f'Live model configured · {self.backend} · {self.model_name}'
    if self.backend == 'together_ai':
      label += (
          ' · Credential precedence: prompted key, TOGETHER_API_KEY,'
          ' TOGETHER_AI_API_KEY (names only)'
      )
    return label

  def create_model(self) -> language_model.LanguageModel:
    """Called only at explicit execution; provider errors never fall back."""
    prompted = self._runtime_api_key
    key = None
    source = 'not applicable'
    if self.backend == 'together_ai':
      key = (
          prompted
          if prompted is not None
          else os.getenv('TOGETHER_API_KEY') or None
      )
      source = (
          'prompted key'
          if prompted is not None
          else 'TOGETHER_API_KEY'
          if key
          else 'TOGETHER_AI_API_KEY (adapter fallback)'
      )
    object.__setattr__(self, '_credential_source', source)
    try:
      return language_models.language_model_setup(
          api_type=self.backend,
          model_name=self.model_name or '',
          api_key=key,
          disable_language_model=self.backend == 'none',
      )
    except Exception:
      if self.backend == 'together_ai':
        raise ValueError(
            'Could not initialize Together; credential source: '
            + source
            + '. Check credential availability and provider configuration.'
        ) from None
      raise


def prompt_api_key() -> str:
  """Read only masked terminal input; never allow getpass's echo fallback."""
  if not sys.stdin.isatty():
    raise ValueError('--prompt-api-key requires a local interactive terminal.')
  try:
    with warnings.catch_warnings():
      warnings.simplefilter('error', getpass.GetPassWarning)
      value = getpass.getpass('Together API key (hidden, local process only): ')
  except (Exception, KeyboardInterrupt):
    raise ValueError('Masked API key entry cancelled or unavailable.') from None
  if not value or not value.strip():
    raise ValueError('API key entry must not be empty.')
  return value


def build(
    config: prefab_lib.Config,
    *,
    model: language_model.LanguageModel | None = None,
) -> generic.Simulation:
  """Build standard prefabs without executing them or making model calls."""
  return generic.Simulation(
      config=config,
      model=model if model is not None else no_language_model.NoLanguageModel(),
      embedder=lambda _: np.ones(8),
      engine=sequential.Sequential(),
  )


def save_result(registry, document, log, output: Path) -> Path:
  """Retain each run separately; the initial definition is never runtime state."""
  destination = output / uuid.uuid4().hex
  destination.mkdir(parents=True)
  (destination / 'initial-project.json').write_text(
      registry.dumps(document), encoding='utf-8'
  )
  (destination / 'log.json').write_text(log.to_json(), encoding='utf-8')
  (destination / 'log.html').write_text(log.to_html(), encoding='utf-8')
  return destination


def create_editor(
    document: dict | None = None,
    *,
    port: int = 8080,
    output: Path = Path('project-run'),
    step_delay: float = 1.0,
    public_origin: str | None = None,
    model_selection: ModelSelection = ModelSelection(),
) -> simulation_server.SimulationServer:
  """Configure, but do not start or run, the existing SimulationServer.

  Preview always uses NoLanguageModel. The selected provider is initialized
  only by Run. The optional delay paces completed steps in either mode.
  """
  if not math.isfinite(step_delay) or not 0 <= step_delay <= 5:
    raise ValueError('Step delay must be between 0 and 5 seconds.')
  registry = template.registry()
  initial = registry.normalize(
      document
      if document is not None
      else registry.default_document(template.TEMPLATE_KEY)
  )
  server = simulation_server.SimulationServer(
      port=port, public_origin=public_origin
  )

  def run(config: prefab_lib.Config, requested_steps: int) -> None:
    saved_definition = server.get_project()['document']
    sim = build(config, model=model_selection.create_model())
    controller = server.step_controller
    server.set_simulation(sim)
    server.broadcast_entity_info(sim.make_checkpoint_data())

    def completed_step(step):
      server.broadcast_entity_info(sim.make_checkpoint_data())
      server.broadcast_step(step)
      deadline = time.monotonic() + step_delay
      while not controller.should_stop() and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    try:
      log = sim.play(
          max_steps=requested_steps,
          step_controller=controller,
          step_callback=completed_step,
      )
    except language_model.InvalidResponseError as error:
      if model_selection.backend == 'together_ai':
        raise language_model.InvalidResponseError(
            str(error)
            + ' Credential source: '
            + model_selection._credential_source
            + '. Precedence: prompted key > TOGETHER_API_KEY >'
            ' TOGETHER_AI_API_KEY. Restart from your own credential-bearing'
            ' shell after correcting configuration.'
        ) from None
      raise
    server.set_project_log(log)
    destination = save_result(registry, saved_definition, log, output)
    print(f'{model_selection.label} run artifacts: {destination}', flush=True)
    steps = server.get_status()['current_step']
    if controller.should_stop():
      reason = 'Stop requested through the run controls.'
    elif steps >= requested_steps:
      reason = f'Requested step limit reached ({requested_steps}).'
    else:
      reason = (
          'Game master ended the run before the step limit; with the default'
          ' scene-aware prefab this happens when its scene sequence is'
          ' exhausted.'
      )
    server.broadcast_completion(reason)

  server.configure_project(
      registry,
      initial,
      run_with_steps=run,
      integrated=True,
      title=(
          f'Roommate music lab · {model_selection.label} · '
          f'{step_delay:g}s pacing'
      ),
      preview=lambda config: build(config).make_checkpoint_data(),
  )
  return server


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--project', type=Path, help='Saved registered project JSON'
  )
  parser.add_argument('--port', type=int, default=8080)
  parser.add_argument(
      '--model-backend',
      '--api-type',
      default='none',
      help='Standard Concordia backend (e.g. together_ai); default: none',
  )
  parser.add_argument('--model-name', help='Explicit provider model identifier')
  parser.add_argument(
      '--prompt-api-key',
      action='store_true',
      help=(
          'Read a masked Together key from your local terminal; never'
          ' stdin/echo fallback'
      ),
  )
  parser.add_argument('--output', type=Path, default=Path('project-run'))
  parser.add_argument(
      '--step-delay',
      type=float,
      default=1.0,
      help='Editor delay after each step (0–5 seconds)',
  )
  parser.add_argument(
      '--public-origin',
      help=(
          'Exact private HTTPS origin browsers must use; '
          'omit for direct local HTTP access'
      ),
  )
  parser.add_argument(
      '--headless',
      action='store_true',
      help=(
          'Explicitly execute with the selected model instead of opening the'
          ' editor'
      ),
  )
  args = parser.parse_args()
  try:
    selection = ModelSelection(args.model_backend, args.model_name)
    if args.prompt_api_key and selection.backend != 'together_ai':
      raise ValueError('--prompt-api-key requires --model-backend together_ai.')
    if not 0 <= args.port <= 65535:
      raise ValueError('--port must be between 0 and 65535.')
    if not math.isfinite(args.step_delay) or not 0 <= args.step_delay <= 5:
      raise ValueError('--step-delay must be between 0 and 5 seconds.')
    if args.public_origin is not None:
      origin = urlsplit(args.public_origin)
      if (
          origin.scheme != 'https'
          or not origin.netloc
          or origin.path
          or origin.query
          or origin.fragment
          or origin.username
      ):
        raise ValueError(
            '--public-origin must be an exact HTTPS origin without a path.'
        )
  except ValueError as error:
    parser.error(str(error))
  registry = template.registry()
  document = (
      registry.loads(args.project.read_text(encoding='utf-8'))
      if args.project
      else registry.default_document(template.TEMPLATE_KEY)
  )
  if args.prompt_api_key:
    try:
      selection = ModelSelection(
          args.model_backend, args.model_name, api_key=prompt_api_key()
      )
    except ValueError as error:
      parser.error(str(error))
  if args.headless:
    print(f'{selection.label} — explicit headless run', flush=True)
    config = registry.to_config(document)
    log = build(config, model=selection.create_model()).play(
        max_steps=min(10, config.default_max_steps)
    )
    print(save_result(registry, document, log, args.output), flush=True)
    return
  server = create_editor(
      document,
      port=args.port,
      output=args.output,
      step_delay=args.step_delay,
      public_origin=args.public_origin,
      model_selection=selection,
  )
  server.start()
  editor_origin = args.public_origin or f'http://127.0.0.1:{server.bound_port}'
  print(
      f'{"Mock" if selection.backend == "none" else "Live"} editor: '
      f'{editor_origin}/ — {selection.label}. Run is explicit; '
      'preview uses NoLanguageModel. Private access only.',
      flush=True,
  )
  try:
    while True:
      time.sleep(0.2)
  except KeyboardInterrupt:
    server.step_controller.stop()
  finally:
    server.stop()


if __name__ == '__main__':
  main()
