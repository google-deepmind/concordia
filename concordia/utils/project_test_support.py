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

"""Core-owned registered project fixtures; build only, never execute."""

import copy
import dataclasses
import functools
import io
import json
import time
from typing import Any, cast, override
from unittest import mock
import urllib.parse
import uuid

from concordia.agents import entity_agent
from concordia.agents import entity_agent_with_logging
from concordia.environment.engines import sequential
from concordia.language_model import no_language_model
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import minimal
from concordia.prefabs.game_master import dialogic
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.simulation import generic
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import async_measurements as async_measurements_lib
from concordia.utils import project_components
from concordia.utils import project_config
from concordia.utils import simulation_server
from concordia.utils import structured_logging
import numpy as np

TEMPLATE_KEY = 'conversation-v2'
LEGACY_TEMPLATE_KEY = 'conversation-v1'


def _legacy_config() -> prefab_lib.Config:
  """Preserve saved two-minimal-player projects without changing their meaning."""
  return prefab_lib.Config(
      prefabs={'minimal': minimal.Entity(), 'dialogic': dialogic.GameMaster()},
      instances=[
          prefab_lib.InstanceConfig(
              prefab='minimal',
              role=prefab_lib.Role.ENTITY,
              params={
                  'name': name,
                  'custom_instructions': instructions,
                  'goal': '',
                  'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
                  # InstanceConfig's str annotation predates this prefab's bool.
              },
          )
          for name, instructions in [
              (
                  'Alice',
                  (
                      'You are Alice, a university roommate. Describe your'
                      ' current musical interests and listen to your roommate.'
                  ),
              ),
              (
                  'Bob',
                  (
                      'You are Bob, a university roommate. Discuss music'
                      ' without assuming that either person changes their'
                      ' tastes.'
                  ),
              ),
          ]
      ]
      + [
          prefab_lib.InstanceConfig(
              prefab='dialogic',
              role=prefab_lib.Role.GAME_MASTER,
              params={
                  'name': 'Conversation',
                  'next_game_master_name': 'Conversation',
                  'acting_order': 'fixed',
                  'can_terminate_simulation': (
                      False  # pyrefly: ignore[bad-assignment]
                  ),
              },
          )
      ],
      default_premise=(
          'Two university roommates are discussing what music to listen to in'
          ' their shared kitchen.'
      ),
      default_max_steps=4,
  )


def make_config() -> prefab_lib.Config:
  """Contrast Alice's minimal context with Bob's three-question reasoning."""
  legacy = _legacy_config()
  bob = prefab_lib.InstanceConfig(
      prefab='basic',
      role=prefab_lib.Role.ENTITY,
      params={
          'name': 'Bob',
          'goal': (
              'As a university roommate, find music that both you and Alice'
              ' enjoy without assuming either person changes their tastes.'
          ),
          'randomize_choices': False,  # pyrefly: ignore[bad-assignment]
      },
  )
  return dataclasses.replace(
      legacy,
      prefabs={**legacy.prefabs, 'basic': basic.Entity()},
      instances=[legacy.instances[0], bob, legacy.instances[2]],
  )


def validate(document: dict[str, Any]) -> None:
  """Constrain the actual dialogic prefab enum, not its generated behavior."""
  gm = next(
      item for item in document['instances'] if item['id'] == 'conversation'
  )
  if gm['params']['acting_order'] not in (
      'fixed',
      'random',
      'game_master_choice',
  ):
    raise project_config.ValidationError(
        '$.instances[conversation].params.acting_order',
        'expected fixed, random or game_master_choice',
    )


def registry() -> project_config.Registry:
  """Caller-owned fixed template registry; imported JSON chooses no code."""
  return project_config.Registry({
      key: project_config.Template(
          factory=factory,
          instance_ids=('alice', 'bob', 'conversation'),
          references={
              (
                  'conversation',
                  'next_game_master_name',
              ): prefab_lib.Role.GAME_MASTER
          },
          validate=validate,
      )
      for key, factory in (
          (TEMPLATE_KEY, make_config),
          (LEGACY_TEMPLATE_KEY, _legacy_config),
      )
  })


def build(config: prefab_lib.Config) -> generic.Simulation:
  """Construct standard components for inspection without playing a simulation."""
  return generic.Simulation(
      config=config,
      model=no_language_model.NoLanguageModel(),
      embedder=lambda _: np.ones(8),
      engine=sequential.Sequential(),
  )


def builder_registry() -> project_config.Registry:
  """Opt-in reusable prototypes for structural authoring tests."""

  def validate_all(document):
    for item in document['instances']:
      if item['role'] == 'game_master' and item['params'][
          'acting_order'
      ] not in ('fixed', 'random', 'game_master_choice'):
        raise project_config.ValidationError(
            '$.instances[' + item['id'] + '].params.acting_order',
            'unsupported acting order',
        )

  return project_config.Registry({
      'builder-v1': project_config.Template(
          factory=make_config,
          instance_ids=('alice', 'bob', 'conversation'),
          references={
              (
                  'conversation',
                  'next_game_master_name',
              ): prefab_lib.Role.GAME_MASTER
          },
          editable_instances=True,
          validate=validate_all,
      ),
  })


def scene_config():
  """Build a dramaturgic scene configuration for testing."""
  base = make_config()
  gm_params: dict[str, Any] = {
      'name': 'Conversation',
      'allow_llm_fallback': False,
      'scenes': [
          scene_lib.SceneSpec(
              scene_type=scene_lib.SceneTypeSpec(
                  name='Kitchen discussion',
                  game_master_name='Conversation',
                  default_premise={
                      'Alice': ['Listen to one another.'],
                      'Bob': ['Listen to one another.'],
                  },
              ),
              participants=['Alice', 'Bob'],
              num_rounds=2,
          )
      ],
  }
  return dataclasses.replace(
      base,
      prefabs={
          **base.prefabs,
          'dramaturgic': dialogic_and_dramaturgic.GameMaster(),
      },
      instances=[
          *base.instances[:2],
          prefab_lib.InstanceConfig(
              prefab='dramaturgic',
              role=prefab_lib.Role.GAME_MASTER,
              params=gm_params,
          ),
      ],
  )


def scene_registry():
  return project_config.Registry({
      'scenes-v1': project_config.Template(
          factory=scene_config,
          fixed_parameters={'conversation': ('scenes',)},
          instance_ids=('alice', 'bob', 'conversation'),
          editable_instances=True,
          component_types=project_components.standard_types(
              ('alice', 'bob'), ('alice', 'bob', 'conversation')
          ),
      )
  })


def as_agent(entity) -> entity_agent.EntityAgent:
  """Check the concrete entity produced by the registered test prefabs."""
  assert isinstance(entity, entity_agent.EntityAgent)
  return entity


def component_order(entity) -> list[str]:
  """Read and validate an acting component's serialized order in tests."""
  value = as_agent(entity).get_act_component().get_state()['component_order']
  assert isinstance(value, list)
  assert all(isinstance(item, str) for item in value)
  return cast(list[str], value)


def request(
    editor_adapter: Any,
    operation: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Build a developer operation request for an editor service."""
  return dict(
      operation=operation,
      arguments=arguments or {},
      revision=editor_adapter.service.revision,
      references=copy.deepcopy(editor_adapter.service.references),
      retry_key=str(uuid.uuid4()),
  )


def dispatch(
    editor_adapter: Any,
    operation: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Dispatch a developer operation against an editor service."""
  return editor_adapter.service.dispatch(
      'developer', request(editor_adapter, operation, arguments)
  )


def eventually(predicate: Any) -> None:
  """Poll a predicate until it succeeds or times out."""
  deadline = time.monotonic() + 3
  while not predicate():
    if time.monotonic() >= deadline:
      raise AssertionError('Boundary was not reached')
    time.sleep(0.005)


def editor(maximum: int = 40, runner: Any = None) -> Any:
  """Configure an integrated project server for testing without starting HTTP."""
  reg = registry()
  document = reg.default_document(TEMPLATE_KEY)
  document['max_steps'] = maximum
  server = simulation_server.SimulationServer(port=0)
  server.configure_project(
      reg, document, run_with_steps=runner or mock.Mock(), integrated=True
  )
  return server


def joined(server: simulation_server.SimulationServer) -> None:
  """Wait for a server's background project thread to finish."""
  assert server._project_thread is not None  # pylint: disable=protected-access
  server._project_thread.join(3)  # pylint: disable=protected-access
  assert not server._project_thread.is_alive()  # pylint: disable=protected-access


def fake_http(server: simulation_server.SimulationServer):
  """Create an in-process urlopen stub backed by a SimulationServer."""

  def _request(req, **_kwargs):
    path = urllib.parse.urlsplit(req.full_url).path
    service = server.operation_service
    assert service is not None
    if path == '/api/state':
      result = service.snapshot('developer')
    elif path == '/api/operations':
      result = service.discover('developer')
    else:
      result = service.dispatch('developer', json.loads(req.data))
    return io.BytesIO(json.dumps(result).encode())

  return _request


_DEFAULT_ACTION_SPEC_JSON = json.dumps({
    'call_to_action': 'What do you do?',
    'output_type': 'free',
    'options': [],
    'tag': None,
})


class MockEntity(entity_agent_with_logging.EntityAgentWithLogging):
  """Minimal logged entity stub for asynchronous engine integration tests."""

  def __init__(self, name: str) -> None:
    self._name = name
    self._observations = []
    self._act_count = 0
    self._component_logging = async_measurements_lib.ReactiveMeasurements()

  @functools.cached_property
  @override
  def name(self) -> str:
    return self._name

  @override
  def observe(self, observation: str) -> None:
    self._observations.append(observation)

  def get_last_log(self) -> dict[str, Any]:
    return {'LastNObservations': {'Summary': str(self._observations[-1:])}}

  @override
  def act(
      self,
      action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC,
  ) -> str:
    self._act_count += 1
    if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
      return _DEFAULT_ACTION_SPEC_JSON
    elif action_spec.output_type == entity_lib.OutputType.TERMINATE:
      return entity_lib.BINARY_OPTIONS['negative']
    elif action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      return 'entity_0'
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      return action_spec.options[0]
    else:
      raise ValueError(f'Unsupported output type: {action_spec.output_type}')


def create_sample_log() -> structured_logging.SimulationLog:
  """Create a sample SimulationLog fixture for CLI/session log tests."""
  log = structured_logging.SimulationLog()
  log.add_entry(
      step=1,
      timestamp='2024-01-01T10:00:00',
      entity_name='Alice',
      component_name='ActComponent',
      entry_type='entity',
      summary='Alice said hello to Bob',
      raw_data={
          'key': 'Entity [Alice]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Alice said "Hello Bob, nice to meet you."',
                  'Prompt': 'What does Alice do next?',
              },
              '__observation__': {
                  'Key': 'Observation',
                  'Value': ['Bob is standing nearby.'],
              },
              'Instructions': {
                  'Key': 'Instructions',
                  'Value': 'You are Alice, a friendly baker.',
              },
          },
      },
  )
  log.add_entry(
      step=1,
      timestamp='2024-01-01T10:00:00',
      entity_name='Bob',
      component_name='ActComponent',
      entry_type='entity',
      summary='Bob waved at Alice',
      raw_data={
          'key': 'Entity [Bob]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Bob waved back warmly.',
                  'Prompt': 'What does Bob do next?',
              },
          },
      },
  )
  log.add_entry(
      step=2,
      timestamp='2024-01-01T10:01:00',
      entity_name='Alice',
      component_name='ActComponent',
      entry_type='entity',
      summary='Alice offered coffee',
      raw_data={
          'key': 'Entity [Alice]',
          'value': {
              '__act__': {
                  'Key': 'action',
                  'Value': 'Alice offered to buy coffee for Bob.',
                  'Prompt': 'What does Alice do next?',
              },
          },
      },
  )
  log.add_entry(
      step=2,
      timestamp='2024-01-01T10:01:00',
      entity_name='default rules',
      component_name='game_master',
      entry_type='step',
      summary='Step 2: Alice and Bob are having coffee',
      raw_data={
          'key': 'default rules',
          'value': {
              'resolve': {
                  '__resolution__': {
                      'Value': 'Alice and Bob sat down for coffee.',
                  },
                  'tension_tracker': {
                      '__act__': {
                          'Value': '0.2',
                      },
                  },
              },
          },
      },
  )
  log.attach_memories(
      entity_memories={
          'Alice': [
              'Alice loves hiking in the mountains.',
              'Alice works as a baker.',
              'Alice met Bob at the coffee shop.',
          ],
          'Bob': [
              'Bob is a journalist.',
              'Bob met Alice today.',
          ],
      },
      game_master_memories=['The simulation started at 10:00 AM.'],
  )
  return log
