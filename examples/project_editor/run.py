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

"""What should the roommates play? A private, free mock editor demonstration."""

import argparse
import math
from pathlib import Path
import time
import uuid

from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import prefab as prefab_lib
from concordia.utils import project_config
from concordia.utils import simulation_server
import numpy as np

from examples.project_editor import template


def build(config: prefab_lib.Config) -> generic.Simulation:
  """Build standard prefabs without executing them or making model calls."""
  return generic.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
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
) -> simulation_server.SimulationServer:
  """Configure, but do not start or run, the existing SimulationServer.

  The optional delay is mock demonstration pacing in the step callback. The
  engine, model and component behavior remain standard. All model output is a
  stub, not evidence of realistic social behavior.
  """
  if not math.isfinite(step_delay) or not 0 <= step_delay <= 5:
    raise ValueError('Mock step delay must be between 0 and 5 seconds.')
  registry = template.registry()
  initial = registry.normalize(
      document
      if document is not None
      else registry.default_document(template.TEMPLATE_KEY)
  )
  server = simulation_server.SimulationServer(
      port=port, public_origin=public_origin
  )

  def run(config: prefab_lib.Config) -> None:
    saved_definition = server.get_project()['document']
    sim = build(config)
    controller = server.step_controller
    server.set_simulation(sim)
    server.broadcast_entity_info(sim.make_checkpoint_data())

    def completed_step(step):
      server.broadcast_entity_info(sim.make_checkpoint_data())
      server.broadcast_step(step)
      deadline = time.monotonic() + step_delay
      while not controller.should_stop() and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    log = sim.play(step_controller=controller, step_callback=completed_step)
    destination = save_result(registry, saved_definition, log, output)
    print(f'Mock run artifacts: {destination}', flush=True)
    server.broadcast_completion()

  server.configure_project(
      registry,
      initial,
      run,
      integrated=True,
      title=f'Roommate music lab · Free mock · {step_delay:g}s pacing',
      preview=lambda config: build(config).make_checkpoint_data(),
  )
  return server


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--project', type=Path, help='Saved registered project JSON'
  )
  parser.add_argument('--port', type=int, default=8080)
  parser.add_argument('--output', type=Path, default=Path('project-run'))
  parser.add_argument(
      '--step-delay',
      type=float,
      default=1.0,
      help='Mock-only delay after each step (0–5 seconds)',
  )
  parser.add_argument(
      '--public-origin', help='Exact private HTTPS proxy origin'
  )
  parser.add_argument(
      '--headless',
      action='store_true',
      help='Explicitly execute a mock run instead of opening the editor',
  )
  args = parser.parse_args()
  registry = template.registry()
  document = (
      registry.loads(args.project.read_text(encoding='utf-8'))
      if args.project
      else registry.default_document(template.TEMPLATE_KEY)
  )
  if args.headless:
    log = build(registry.to_config(document)).play()
    print(save_result(registry, document, log, args.output), flush=True)
    return
  server = create_editor(
      document,
      port=args.port,
      output=args.output,
      step_delay=args.step_delay,
      public_origin=args.public_origin,
  )
  server.start()
  print(
      f'Mock editor: http://127.0.0.1:{server.bound_port}/ — Run is explicit; '
      'output is a development stub. Private access only.',
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
