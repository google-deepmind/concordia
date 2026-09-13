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

"""Full-night specialization of the shared Bellwether service."""

import copy
import json
import time

from concordia.examples.bellwether import game
from concordia.examples.bellwether import game_prefab
from concordia.examples.bellwether import public_account
from concordia.examples.bellwether import researcher
from concordia.examples.bellwether import scenario
from concordia.examples.bellwether import service
from concordia.utils import operation_service as ops


class Game(service.Bellwether):
  """Twelve human choices, bounded resident turns, one authoritative service."""

  initial_status = 'Ready to begin the night'
  run_description = (
      'Run the three-watch night with twelve human choices and bounded resident'
      ' turns.'
  )
  response_description = (
      'Submit a clarified human attempt for standard Sequential resolution.'
  )

  def __init__(
      self,
      output,
      *,
      model=None,
      action_model=None,
      embedder=None,
      actor_logic='minimal',
      recipe='bellwether',
      dispute=None,
      session_id=None,
      port=0,
      profiler=None,
      human_readers=None
  ):
    self.fixture = model is None
    self.backend = 'fixture' if self.fixture else 'live'

    def configuration(reader):
      self.case = researcher.prepare_case(
          recipe,
          reader,
          actor_logic=actor_logic,
          action_model=action_model,
          human_readers=human_readers,
          dispute=dispute,
      )
      self.world = self.case.world
      return self.case.config

    self.profiler = profiler
    self.step_times = []
    self._action_started = None
    super().__init__(
        output,
        session_id=session_id,
        port=port,
        config_factory=configuration,
        model=model or game_prefab.FixtureModel(),
        embedder=embedder,
        max_steps=64,
    )
    self.world.lock = self.operations.lock
    self.operations.references['project_id'] = 'bellwether-night-v1'
    self.inbox.add_observation(self.case.opening)

  def _seed(self):
    """The case factory already seeded this world before building entities."""

  def _register(self):
    super()._register()
    self.operations.register(
        ops.Operation(
            'game.public_account',
            'Download public events and accounting only, not a checkpoint.',
            {
                'format': ops.Parameter(
                    'string', 'json, html or svg', max_length=4
                )
            },
            lambda args: public_account.export(
                self.world,
                fixture=self.fixture,
                phase=self.phase,
                format_name=args['format'],
                manifest=self.case.manifest,
            ),
            audiences=(*self.player_audiences, 'role:spectator', 'developer'),
        )
    )
    self.operations.register(
        ops.Operation(
            'game.begin',
            'Begin the night. One run only; reload never restarts.',
            {},
            self._start,
            audiences=self.player_audiences,
            mutation=True,
        )
    )
    self.operations.register(
        ops.Operation(
            'game.preview',
            'Clarify an ordinary-language attempt without effects.',
            {
                'text': ops.Parameter(
                    'string', 'Proposed action, without a turn cost'
                )
            },
            lambda args: game.parse_action(args['text']),
            audiences=(*self.player_audiences, 'developer'),
        )
    )
    self.operations.register(
        ops.Operation(
            'run.resume',
            'Resume the existing worker, never replay or replace it.',
            {},
            self._resume,
            mutation=True,
        )
    )

  def _resume(self, args):
    del args
    if self._worker is None or not self._worker.is_alive():
      raise ops.OperationError(
          'no_active_run', 'There is no paused active run.'
      )
    self.server.step_controller.play()
    self.operations.publish({'kind': 'run.resumed'})
    return {'phase': self.phase}

  def player_view(self):
    public = copy.deepcopy(scenario.PUBLIC)
    public['opening'] = self.case.opening
    public['watch'] = self.world.view()['watch']
    return {
        'scenario': public,
        'night': self.world.view(),
        'human': self.inbox.snapshot(),
        'phase': self.phase,
        'fixture': self.fixture,
        'recipe': copy.deepcopy(self.case.manifest),
    }

  def developer_view(self):
    view = super().developer_view()
    view.update({
        'initial': {
            'mechanics': copy.deepcopy(scenario.INITIAL),
            'recipe': copy.deepcopy(self.case.manifest),
            'dispute': copy.deepcopy(self.world.dispute),
            'institutions': copy.deepcopy(self.world.institutions),
        },
        'night': self.world.get_state(),
        'inventory': self.world.inventory_state(),
        'trace': copy.deepcopy(self.simulation.get_raw_log()),
        'backend': 'fixture' if self.fixture else 'live',
        'step_times': list(self.step_times),
        'model_profile': self.profiler.get_stats() if self.profiler else None,
        'capability_gaps': [
            'checkpoint/restore',
            'branches/experiments',
            'in-flight edits/cancellation',
            'initial-project authoring',
            'multi-field transactions',
            'undo/redo',
        ],
    })
    return view

  def _respond(self, args):
    # Validate before waking the human component or spending a choice.
    game.parse_action(args['response'])
    result = super()._respond(args)
    if result['accepted']:
      self._action_started = time.monotonic()
    return result

  def _step(self, data):
    super()._step(data)
    with self.operations.lock:
      self.step_times.append({
          'step': data.step,
          'actor': data.acting_entity,
          'seconds_since_human_submission': (
              time.monotonic() - self._action_started
              if self._action_started is not None
              else None
          ),
      })
      self.operations.publish({
          'kind': 'game.resolved',
          'actor': data.acting_entity,
          'watch': self.world.view()['watch'],
      })

  def _finish(self):
    if not self.world.finished and not self._failure:
      self.phase = 'failed'
      self._failure = 'IncompleteNight'
    self.inbox.finish(
        'Dawn has come.'
        if not self._failure
        else 'The night paused unexpectedly; your journal is retained.'
    )
    self.output.mkdir(parents=True, exist_ok=True)
    (self.output / 'outcome.json').write_text(
        json.dumps(
            {
                'player': self.player_view(),
                'backend': 'fixture' if self.fixture else 'live',
                'step_times': self.step_times,
                'model_profile': (
                    self.profiler.get_stats() if self.profiler else None
                ),
                'model_cost': None,
                'embedding': copy.deepcopy(self.embedding),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
