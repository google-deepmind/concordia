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

"""An entity that deliberates in the background and acts on its beliefs.

The three cognition tiers of `concordia.components.agent.pubsub`, assembled
from stock components:

* **Beliefs**: one `pubsub.Subscriber` per subscribed `TopicBus` topic. With a
  clock topic, each belief shows how old its latest reading is
  (`operators.with_age`), so the entity can tell fresh news from stale.
* **Slow mind**: the stock `plan.Plan`, wrapped in `pubsub.Publisher` so that
  planning runs in the background (on threads, or a fixed number of ticks
  later on a contrib `SimTimeScheduler`). Its plans are published on a topic,
  where other entities may subscribe to them too.
* **Fast act**: a `ConcatActComponent` that reads the instructions, goal,
  beliefs, the latest *finished* plan (with its age) and recent observations,
  so acting never waits for planning. Like the standard `basic` and `minimal`
  prefabs, `build` accepts a runtime acting policy instead (for example
  `HumanActComponent`, so a person plays the entity on the same context).

Example:

```python
bus = topics.TopicBus(clock=clock.ticks)
config = prefab_lib.InstanceConfig(
    prefab='basic_with_async_deliberation__Entity',
    role=prefab_lib.Role.ENTITY,
    params={
        'name': 'Ada',
        'goal': 'Keep the harbour supplied.',
        'bus': bus,
        'subscriptions': {'Market prices': 'econ.prices.harbour'},
        'renderers': {'econ.prices.harbour': render_prices},
        'clock_topic': 'world.tick',
        'sim_scheduler': sim_scheduler,  # Or None for real threads.
        'deliberation_latency_ticks': 3,
    },
)
```
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import pubsub
from concordia.contrib.pubsub import operators
from concordia.contrib.pubsub import sim_time
from concordia.contrib.pubsub import topics
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from reactivex import operators as ops

DELIBERATION_COMPONENT_KEY = 'Deliberation'
PLAN_BELIEF_COMPONENT_KEY = 'Current plan'
AGE_TEMPLATE = '{text} [{age}]'


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """An entity with background deliberation and topic-bound beliefs."""

  supports_extra_components = True

  description: str = (  # pyrefly: ignore[bad-override]
      'An entity whose planning runs in the background (a pubsub Publisher'
      ' around the stock Plan component) while it acts every step on its'
      ' latest beliefs: TopicBus subscriptions and the last finished plan,'
      ' each shown with its age.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Alice',
          'goal': '',
          # A contrib `TopicBus`. Without it the entity has no subscriptions
          # and its plans are not published on a topic.
          'bus': None,
          # Component key -> topic name, one belief per entry.
          'subscriptions': {},
          # Topic name -> function from a `topics.Message` to the text the
          # entity reads (default: the raw payload).
          'renderers': {},
          # Topic whose message times are "now" (e.g. 'world.tick'); beliefs
          # then show their age. Empty: no ages.
          'clock_topic': '',
          # Topic the slow mind publishes plans to ('<name>.plan' if empty).
          'plan_topic': '',
          # A contrib `SimTimeScheduler`: plans land exactly
          # `deliberation_latency_ticks` after they were started. None: real
          # background threads.
          'sim_scheduler': None,
          'deliberation_latency_ticks': 2,
          # Start deliberating as soon as an observation arrives (an alarm),
          # not only when the entity acts.
          'trigger_on_observe': False,
          # Ignore triggers while a deliberation is in flight.
          'skip_if_busy': True,
          # Model for the background plan (e.g. a bigger, slower one).
          # None: the same model the entity acts with.
          'deliberation_model': None,
          'observation_history': 20,
          'randomize_choices': True,
          'extra_components': {},
          'extra_components_index': {},
          # Keys of `extra_components` the background plan should also read
          # (e.g. a belief built with a custom transport).
          'plan_components': (),
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      *,
      act_component: entity_component.ActingComponent | None = None,
      act_component_factory: (
          Callable[[Sequence[str]], entity_component.ActingComponent] | None
      ) = None,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the entity.

    Args:
      model: The language model for acting (and for planning, unless the
        `deliberation_model` param is set).
      memory_bank: The entity's memory bank.
      act_component: Optional runtime acting policy (e.g. `HumanActComponent`)
        instead of the default `ConcatActComponent`. Context components,
        including background deliberation, are unchanged.
      act_component_factory: Optional factory receiving the exact context order
        the default policy would use. Cannot be combined with `act_component`.

    Returns:
      The entity.

    Raises:
      ValueError: If both `act_component` and `act_component_factory` are
        given.
    """
    if act_component is not None and act_component_factory is not None:
      raise ValueError(
          'Provide act_component or act_component_factory, not both.'
      )
    self.check_extra_components()
    raw_extra = self.params.get('extra_components', {})
    extra_indices = self.params.get('extra_components_index', {})
    if extra_indices and raw_extra:
      if extra_indices.keys() != raw_extra.keys():
        raise ValueError(
            'extra_components_index must have the same keys as'
            ' extra_components.'
        )
    name = self.params.get('name', 'Alice')
    goal = self.params.get('goal', '')
    bus: topics.TopicBus | None = self.params.get('bus')
    subscriptions: Mapping[str, str] = self.params.get('subscriptions', {})
    if subscriptions and bus is None:
      raise ValueError('`subscriptions` need a `bus`.')
    clock_topic = self.params.get('clock_topic', '')
    plan_topic = self.params.get('plan_topic', '') or f'{name}.plan'
    sim_scheduler: sim_time.SimTimeScheduler | None = self.params.get(
        'sim_scheduler'
    )

    renderers: Mapping[str, Callable[[topics.Message], str]] = self.params.get(
        'renderers', {}
    )

    def belief_transform(topic_bus: topics.TopicBus, topic: str):
      render = renderers.get(topic, topics.payload)
      if not clock_topic:
        return ops.map(render)
      now = topic_bus.observable(clock_topic).pipe(ops.map(lambda m: m.time))
      return operators.with_age(now, render=render, template=AGE_TEMPLATE)

    components: dict[str, entity_component.ContextComponent] = {}
    if bus is not None:
      components[topics.DEFAULT_TOPIC_BUS_COMPONENT_KEY] = (
          topics.TopicBusComponent(bus)
      )
    instructions_key = 'Instructions'
    components[instructions_key] = agent_components.instructions.Instructions(
        agent_name=name, pre_act_label='\nInstructions'
    )
    goal_key = None
    if goal:
      goal_key = 'Goal'
      components[goal_key] = agent_components.constant.Constant(
          state=goal, pre_act_label='\nGoal'
      )

    belief_keys = []
    for key, topic in subscriptions.items():
      assert bus is not None
      components[key] = bus.make_subscriber(
          topic,
          subscriber_name=name,
          initial_value='Nothing yet.',
          pre_act_label=f'\n{key}',
          transform=belief_transform(bus, topic),
      )
      belief_keys.append(key)

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    components['ObservationToMemory'] = (
        agent_components.observation.ObservationToMemory()
    )
    components[observation_key] = (
        agent_components.observation.LastNObservations(
            history_length=int(self.params.get('observation_history', 20)),
            pre_act_label='\nRecent observations',
        )
    )
    components[agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY] = (
        agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
    )

    plan_extras = list(self.params.get('plan_components', ()))
    missing = set(plan_extras) - set(self.params.get('extra_components', {}))
    if missing:
      raise ValueError(
          f'plan_components {sorted(missing)} are not in extra_components'
      )
    plan = agent_components.plan.Plan(
        model=self.params.get('deliberation_model') or model,
        components=[*belief_keys, *plan_extras, observation_key],
        goal_component_key=goal_key,
        pre_act_label='\nPlan',
    )
    publisher_kwargs = {
        'skip_if_busy': bool(self.params.get('skip_if_busy', True)),
        'trigger_on_observe': bool(
            self.params.get('trigger_on_observe', False)
        ),
        'scheduler': sim_time.publisher_scheduler(
            sim_scheduler,
            int(self.params.get('deliberation_latency_ticks', 2)),
            key=name,
        ),
        'pre_act_label': DELIBERATION_COMPONENT_KEY,
    }
    initial_plan = 'No plan yet; acting on instinct.'
    if bus is not None:
      deliberation = bus.make_publisher(
          plan_topic, component=plan, source=name, **publisher_kwargs
      )
      plan_belief = bus.make_subscriber(
          plan_topic,
          subscriber_name=name,
          initial_value=initial_plan,
          pre_act_label=f'\n{PLAN_BELIEF_COMPONENT_KEY}',
          transform=belief_transform(bus, plan_topic),
      )
    else:
      deliberation = sim_time.SimTimePublisher(
          component=plan, **publisher_kwargs
      )
      plan_belief = pubsub.Subscriber(
          publisher=deliberation,
          initial_value=initial_plan,
          pre_act_label=f'\n{PLAN_BELIEF_COMPONENT_KEY}',
      )
    # The publisher must be a component so that acting triggers it; it adds
    # nothing to the prompt itself.
    components[DELIBERATION_COMPONENT_KEY] = deliberation
    components[PLAN_BELIEF_COMPONENT_KEY] = plan_belief

    builtin_keys = set(components)
    self.validate_extra_components(components)
    extra = {
        k: copy.deepcopy(c) if getattr(c, '_entity', None) is not None else c
        for k, c in raw_extra.items()
    }
    components.update(extra)
    order = [instructions_key]
    if goal_key:
      order.append(goal_key)
    order += [*belief_keys, PLAN_BELIEF_COMPONENT_KEY, observation_key]
    if extra:
      if extra_indices:
        for key in extra:
          if key in order:
            order.remove(key)
          order.insert(extra_indices[key], key)
      else:
        obs_pos = order.index(observation_key)
        for key in extra:
          if key not in order and key not in builtin_keys:
            order.insert(obs_pos, key)
            obs_pos += 1

    if act_component_factory is not None:
      act_component = act_component_factory(tuple(order))
    elif act_component is None:
      act_component = agent_components.concat_act_component.ConcatActComponent(
          model=model,
          component_order=order,
          randomize_choices=bool(self.params.get('randomize_choices', True)),
      )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
        measurements=self.params.get('measurements'),
    )
