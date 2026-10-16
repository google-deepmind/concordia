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

"""The colonists: a C5 entity on the contrib async-deliberation prefab.

They read their own colony's status fresh (it is published where they are)
and their own sheet, deliberate in the background on the slow model, and
act each generation on the fast model.
"""

from concordia.components.agent import report_function
from examples.deep_time import actors as actors_lib
from examples.deep_time import game as game_lib
from examples.deep_time import perception
from examples.deep_time.prefabs import mission as mission_prefab
from examples.deep_time.prefabs import polity as polity_prefab
from examples.deep_time.world import publishing
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

SUBSCRIPTIONS = {'Colony status': 'colony.status'}
DEFAULT_GOAL = (
    'Keep every colonist fed and breathing, and warm this world until it is'
    ' habitable.'
)


def colony_config(
    context: game_lib.GameContext,
    slow_model: language_model.LanguageModel,
) -> prefab_lib.InstanceConfig:
  """Registers the colonists' actor and returns their entity config."""
  colony = context.planet.colony
  if colony is None:
    raise ValueError('Colonists need a colony in the scenario.')
  rules = context.scenario['colony']
  name = f'Colonists of {colony.name}'
  context.add_actor(
      actors_lib.ColonyActor(
          name,
          context.planet,
          share=float(rules.get('focus_share', 0.15)),
          plan_share=float(rules.get('plan_share', 0.4)),
      ),
      # The colony's news and their own decrees, not its status payload
      # (that is their 'Colony status' reading, rendered).
      hears=('world.tick', 'colony.events', 'colony.decrees'),
  )
  return prefab_lib.InstanceConfig(
      prefab=polity_prefab.POLITY_PREFAB,
      role=prefab_lib.Role.ENTITY,
      params={
          'name': name,
          'goal': rules.get('goal', DEFAULT_GOAL),
          'bus': context.bus,
          'subscriptions': dict(SUBSCRIPTIONS),
          'renderers': {
              topic: publishing.renderer_for(topic)
              for topic in SUBSCRIPTIONS.values()
          },
          'clock_topic': 'world.tick',
          'plan_topic': 'colony.plan',
          'sim_scheduler': context.sim,
          'deliberation_latency_ticks': int(rules.get('plan_latency_ticks', 2)),
          'deliberation_model': slow_model,
          'randomize_choices': False,
          'measurements': context.measurements,
          **context.act_params(name),
          'extra_components': {
              'Our colony': report_function.ReportFunction(
                  function=colony.sheet,
                  pre_act_label='\nOur colony',
              ),
              # Home's transmissions, heard years after they were sent.
              'Messages from home': perception.instrument(
                  context,
                  'mission.uplink',
                  owner=name,
                  label='Messages from home',
                  initial='No message from home has reached us yet.',
                  transported=True,
                  channel=lambda: mission_prefab.light_delay(context),
              ),
          },
      },
  )
