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

"""One More Song: a short conversation, followed by two explicit AI votes."""

import json
import pathlib
import threading
import time
from typing import Any
import uuid

from concordia.components.agent import concat_act_component
from concordia.components.agent import human_act_component
from concordia.components.game_master import event_resolution
from concordia.components.game_master import make_observation
from concordia.components.game_master import next_acting
from concordia.components.game_master import switch_act
from concordia.components.game_master import terminate
from concordia.environment import engine
from concordia.environment.engines import sequential
from concordia.examples.astral_canticle import human_io
from concordia.prefabs.entity import minimal
from concordia.prefabs.simulation import generic
from concordia.typing import entity
from concordia.typing import entity_component
from concordia.typing import prefab
from concordia.utils import structured_logging
import numpy as np

PLAYER = 'You'
CAST = ('You', 'Maya', 'Leon')
PREMISE = (
    'A neighbourhood benefit concert is at closing time. The organiser '
    '(whose speaker label is "You") wants an encore plan that both Maya '
    'and Leon will accept. Maya is the singer; Leon lives next door and '
    'needs quiet. The organiser has three speaking turns: ask what matters, '
    'negotiate, then make a final offer. Maya and Leon each speak twice, '
    'then independently vote ACCEPT or DECLINE on that final offer. '
    'Everyone hears what is said. Words are proposals and promises, not proof '
    'that anyone has already performed an action. No one can speak for another.'
)
ACCEPT = 'ACCEPT'
DECLINE = 'DECLINE'
SPEECH = entity.free_action_spec(
    call_to_action=(
        'What does {name} say to the others? Respond to their actual words; '
        'speak directly in first person in at most 45 words. Address the '
        'current proposal before adding a concern or alternative. Output only '
        'your spoken words, without a speaker label. Do not narrate other '
        'people, invent earlier conversations or agreement, or use analysis.'
    )
)
BALLOT = entity.choice_action_spec(
    call_to_action=(
        '{name}, independently decide whether to accept the organiser’s final '
        'proposal, given your goal and the conversation. A promise is not '
        'guaranteed performance. Vote ACCEPT or DECLINE; either is allowed.'
    ),
    options=(ACCEPT, DECLINE),
)


class BallotPhase(entity_component.ContextComponent):
  """Scenario rule: seven speaking turns, then Maya and Leon vote.

  Turn order and execution remain the standard Concordia implementations.
  This component only selects the scenario's speech/ballot ActionSpec.
  """

  def __init__(self):
    super().__init__()
    self.completed = 0

  def pre_act(self, action_spec):
    if action_spec.output_type == entity.OutputType.NEXT_ACTION_SPEC:
      return engine.action_spec_to_string(
          BALLOT if self.completed >= 7 else SPEECH
      )
    return ''

  def get_state(self):
    return {'completed': self.completed}

  def set_state(self, state):
    completed = state['completed']
    if type(completed) is not int or not 0 <= completed <= 9:
      raise ValueError('completed must be an integer from 0 to 9')
    self.completed = completed


class PlayerSession(human_io.HumanSession):
  """Reuse the reconnectable inbox; add only this game's public scoreboard."""

  def __init__(self, mode):
    super().__init__(initial_status='Take your first turn when you are ready.')
    self._public_lock = threading.Lock()
    self._public: dict[str, Any] = {
        'mode': mode,
        'session_id': uuid.uuid4().hex,
        'turn': 1,
        'votes': {},
        'ending': None,
        'events': [],
        'latencies_seconds': [],
    }

  def record(self, step, elapsed):
    with self._public_lock:
      self._public['events'].append({
          'step': step.step,
          'actor': step.acting_entity,
          'text': step.action,
      })
      self._public['latencies_seconds'].append(round(elapsed, 3))
      self._public['turn'] = min(3, step.step // 3 + 1)
      if step.step >= 8:
        vote = step.action.removeprefix(step.acting_entity + ':').strip()
        if vote not in (ACCEPT, DECLINE):
          raise ValueError('Ballot must be a validated choice, not narrative.')
        self._public['votes'][step.acting_entity] = vote

  def conclude(self):
    with self._public_lock:
      votes = self._public['votes']
      success = len(votes) == 2 and all(v == ACCEPT for v in votes.values())
      self._public['ending'] = (
          'Agreement reached — both accepted your final proposal.'
          if success
          else (
              'No shared agreement — not everyone accepted the final proposal.'
          )
      )
      message = self._public['ending']
    self.finish(message)

  def snapshot(self):
    snapshot = super().snapshot()
    with self._public_lock:
      snapshot['game'] = json.loads(json.dumps(self._public))
      # The shared journal must use the same complete public transcript as the
      # player page. No human input follows the ballot, so inbox observations
      # alone would omit the final votes (and can duplicate human utterances).
      snapshot['entries'] = [
          {'kind': 'story', 'text': event['text']}
          for event in self._public['events']
      ]
      if self._public['ending']:
        snapshot['entries'].append(
            {'kind': 'story', 'text': self._public['ending']}
        )
    # Never expose model prompts, private goals, other actors' contexts or logs.
    if snapshot['pending']:
      snapshot['pending'].pop('context', None)
    return snapshot


def configuration(reader):
  """Compose existing human/minimal prefabs and standard GM components."""

  class Human(prefab.Prefab):
    description = 'Human organiser, with standard minimal prefab context.'

    def build(self, model, memory_bank):
      return minimal.Entity(params=self.params).build(
          model,
          memory_bank,
          act_component_factory=lambda order: human_act_component.HumanActComponent(
              reader, component_order=order
          ),
      )

  class Speaker(prefab.Prefab):
    description = (
        'Minimal agent speaking directly, without a second name prefix.'
    )

    def build(self, model, memory_bank):
      return minimal.Entity(params=self.params).build(
          model,
          memory_bank,
          act_component_factory=lambda order: concat_act_component.ConcatActComponent(
              model=model,
              component_order=order,
              prefix_entity_name=False,
              randomize_choices=False,
          ),
      )

  class Conversation(prefab.Prefab):
    description = 'Public conversation with a fixed order and explicit ballot.'

    def build(self, model, memory_bank):
      extra = {
          switch_act.DEFAULT_NEXT_ACTING_COMPONENT_KEY: (
              next_acting.NextActingInFixedOrder(sequence=CAST)
          ),
          switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY: BallotPhase(),
          switch_act.DEFAULT_TERMINATE_COMPONENT_KEY: (
              terminate.NeverTerminate()
          ),
          switch_act.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY: (
              make_observation.MakeObservation(
                  model=model, player_names=CAST, allow_llm_fallback=False
              )
          ),
          switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY: (
              event_resolution.EventResolution(
                  model=model,
                  event_resolution_steps=(
                      event_resolution.RemoveSpecificText(
                          substring_to_remove='Putative event to resolve:  '
                      ),
                  ),
                  notify_observers=False,
              )
          ),
      }
      params: dict[str, Any] = {**self.params, 'extra_components': extra}
      return minimal.Entity(params=params).build(
          model,
          memory_bank,
          act_component=switch_act.SwitchAct(model=model, entity_names=CAST),
      )

  instances = [
      prefab.InstanceConfig(
          prefab='human',
          role=prefab.Role.ENTITY,
          params={
              'name': PLAYER,
              'custom_instructions': PREMISE + ' You are the organiser.',
          },
      )
  ]
  for name, identity, goal in (
      (
          'Maya',
          'An exuberant singer who loves including shy audience members.',
          (
              'Give the audience a warm final memory. You dislike being'
              ' abruptly silenced but can accept a quiet, short, unamplified'
              ' song if treated respectfully. You decide for yourself; do not'
              ' agree automatically.'
          ),
      ),
      (
          'Leon',
          'A tired, practical neighbour who still wants the event to succeed.',
          (
              'Protect a sleeping child from amplified music and leave'
              ' promptly. You might accept a brief unamplified encore with a'
              ' clear end. A vague assurance is not enough. You decide for'
              ' yourself.'
          ),
      ),
  ):
    params: dict[str, Any] = {
        'name': name,
        'goal': goal,
        'randomize_choices': False,
        'custom_instructions': (
            f'{PREMISE} You are {name}. {identity} Speak only as {name}.'
        ),
    }
    instances.append(
        prefab.InstanceConfig(
            prefab='speaker',
            role=prefab.Role.ENTITY,
            params=params,
        )
    )
  instances.append(
      prefab.InstanceConfig(
          prefab='conversation',
          role=prefab.Role.GAME_MASTER,
          params={'name': 'Conversation', 'custom_instructions': PREMISE},
      )
  )
  return prefab.Config(
      prefabs={
          'speaker': Speaker(),
          'human': Human(),
          'conversation': Conversation(),
      },
      instances=instances,
      default_premise=PREMISE,
      default_max_steps=9,
  )


def build(model, reader):
  config = configuration(reader)
  simulation = generic.Simulation(
      config=config,
      model=model,
      embedder=lambda _: np.ones(8),
      engine=sequential.Sequential(),
  )
  for actor in simulation.get_entities():
    actor.observe(PREMISE)
  return config, simulation


def play(simulation, session, output: pathlib.Path, *, editor=None):
  """Run nine standard engine steps and save standard logs plus public metrics."""
  output.mkdir(parents=True, exist_ok=True)
  gm = simulation.get_game_masters()[0]
  observation = gm.get_component(
      switch_act.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
  )
  phase = gm.get_component(switch_act.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY)
  start = last = time.monotonic()

  def save():
    log = structured_logging.SimulationLog.from_raw_log(
        simulation.get_raw_log()
    )
    values = {
        'simulation.json': log.to_json(),
        'log.html': log.to_html(title='One More Song — designer log'),
        'public.json': json.dumps(session.snapshot(), indent=2),
    }
    for name, data in values.items():
      temporary = output / (name + '.tmp')
      temporary.write_text(data, encoding='utf-8')
      temporary.replace(output / name)

  def step_done(step):
    nonlocal last
    now = time.monotonic()
    phase.completed = step.step
    # Ballots are public results, not further dialogue. Each character should
    # judge the final proposal without first observing the other's vote.
    if step.step < 8:
      observation.add_to_queue('all', step.action)
    session.record(step, now - last)
    session.progress(step.step, step.acting_entity)
    simulation.save_checkpoint(step.step, str(output / 'checkpoints'))
    last = now
    if editor:
      editor.broadcast_step(step)
      editor.broadcast_entity_info(simulation.make_checkpoint_data())
    save()

  try:
    simulation.play(
        max_steps=9,
        step_callback=step_done,
        step_controller=editor.step_controller if editor else None,
    )
    if phase.completed == 9:
      session.conclude()
    else:
      session.finish('The host stopped this run before the ballot finished.')
  except Exception:
    # Mark the public artifact terminal before saving it. The browser runner
    # also handles exceptions, but only after this finally block has run.
    # Keep provider diagnostics private and preserve the original exception.
    session.finish(
        'The run stopped before completion. Your recorded conversation is'
        ' saved.'
    )
    raise
  finally:
    save()
    (output / 'timing.json').write_text(
        json.dumps(
            {
                'elapsed_seconds_including_human_wait': round(
                    time.monotonic() - start, 3
                ),
                'completed_steps': phase.completed,
            },
            indent=2,
        )
    )
