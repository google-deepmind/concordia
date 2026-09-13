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

"""Open or run the same restricted initial project; no model runs on Open."""

import argparse
from pathlib import Path
import time

from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic
from concordia.typing import prefab as prefab_lib
from concordia.utils import simulation_server
from concordia.utils import visual_interface
import numpy as np

from examples.project_editor import template


def build(config: prefab_lib.Config) -> generic.Simulation:
  """Use standard prefabs/engine with the explicitly mock development model.

  Real-model applications may pass their existing model/embedder here. The
  bundled stub is for workflow verification, not social or scientific evidence.
  """
  return generic.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=lambda _: np.ones(8),
      engine=sequential.Sequential(),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--project', type=Path, help='Saved conversation-v1 JSON')
  parser.add_argument('--port', type=int, default=8080)
  parser.add_argument('--output', type=Path, default=Path('project-run'))
  parser.add_argument(
      '--headless',
      action='store_true',
      help='Explicitly run instead of opening the editor',
  )
  args = parser.parse_args()
  registry = template.registry()
  document = (
      registry.loads(args.project.read_text(encoding='utf-8'))
      if args.project
      else registry.normalize(registry.default_document(template.TEMPLATE_KEY))
  )

  def save_result(log):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'initial-project.json').write_text(
        registry.dumps(document), encoding='utf-8'
    )
    (args.output / 'log.json').write_text(log.to_json(), encoding='utf-8')
    (args.output / 'log.html').write_text(log.to_html(), encoding='utf-8')

  if args.headless:
    save_result(build(registry.to_config(document)).play())
    return

  server = simulation_server.SimulationServer(port=args.port)

  def run(config: prefab_lib.Config) -> None:
    nonlocal document
    # Record the normalized reopened draft supplied by the server.
    document = server.get_project()['document']
    sim = build(config)
    server.set_simulation(sim)
    server.set_runtime_html_content(
        visual_interface.visualize_config_to_html(
            config,
            title='Runtime state — return to / for initial project',
            checkpoint_data=sim.make_checkpoint_data(),
        )
    )
    server.broadcast_entity_info(sim.make_checkpoint_data())
    log = sim.play(
        step_controller=server.step_controller,
        step_callback=server.broadcast_step,
    )
    save_result(log)
    server.broadcast_completion()

  server.configure_project(registry, document, run)
  server.start()
  print(
      f'Initial project editor: http://127.0.0.1:{server.bound_port}/ (mock'
      ' model; Run is explicit)',
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
