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

"""Tests for adding and removing an entity agent's components."""

from collections.abc import Callable, Sequence
import json
import threading
from typing import override

from absl.testing import absltest
from concordia.agents import entity_agent
from concordia.agents import entity_agent_with_logging
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class _Note(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Says its text before acting, counts the acts it took part in."""

  def __init__(self, text: str):
    super().__init__()
    self.text = text
    self.acts = 0
    self.calls: list[str] = []

  @override
  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    self.calls.append('pre_act')
    self._logging_channel({'Key': 'note', 'Value': self.text})
    return self.text

  @override
  def post_act(self, action_attempt: str) -> str:
    del action_attempt
    self.calls.append('post_act')
    self.acts += 1
    return ''

  @override
  def pre_observe(self, observation: str) -> str:
    del observation
    self.calls.append('pre_observe')
    return ''

  @override
  def update(self) -> None:
    self.calls.append('update')

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'acts': self.acts}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self.acts = int(state['acts'])  # pyrefly: ignore[bad-argument-type]


class _Hook(entity_component.ContextComponent):
  """Runs a function of its entity when the entity observes."""

  def __init__(self, on_observe: Callable[[entity_agent.EntityAgent], None]):
    super().__init__()
    self._on_observe = on_observe

  @override
  def pre_observe(self, observation: str) -> str:
    del observation
    self._on_observe(self.get_entity())  # pyrefly: ignore[bad-argument-type]
    return ''

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


class _JoinContexts(entity_component.ActingComponent):
  """Acts by joining its contexts, in an optional explicit order."""

  def __init__(self, order: Sequence[str] | None = None):
    super().__init__()
    self._order = order

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del action_spec
    return ' '.join(
        contexts[name] for name in sorted(contexts) if contexts[name]
    )

  def get_context_concat_order(self) -> Sequence[str] | None:
    return self._order

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    del state


def _note_factory(
    name: str, spec: entity_component.ComponentState
) -> entity_component.ContextComponent:
  del name
  return _Note(str(spec['text']))


def _agent(**components) -> entity_agent.EntityAgent:
  return entity_agent.EntityAgent(
      'Ada',
      act_component=_JoinContexts(),
      context_components=components,
      component_factory=_note_factory,
  )


class AddAndRemoveComponentsTest(absltest.TestCase):

  def test_a_component_added_while_ready_joins_at_once(self):
    agent = _agent(a=_Note('one'))
    two = _Note('two')

    agent.add_component('b', two)

    self.assertIs(agent.get_component('b'), two)
    self.assertIs(two.get_entity(), agent)
    self.assertEqual(agent.act(), 'one two')

  def test_a_component_removed_while_ready_leaves_at_once(self):
    agent = _agent(a=_Note('one'), b=_Note('two'))

    agent.remove_component('b')

    self.assertNotIn('b', agent.get_all_context_components())
    self.assertEqual(agent.act(), 'one')

  def test_a_component_added_during_observe_joins_when_it_ends(self):
    late = _Note('late')

    def adopt(entity):
      entity.add_component('late', late)
      self.assertNotIn('late', entity.get_all_context_components())

    agent = _agent(hook=_Hook(adopt))

    agent.observe('a custom is offered')

    self.assertIs(agent.get_component('late'), late)
    self.assertEqual(late.calls, [])  # It took no part in that observe.
    self.assertEqual(agent.act(), 'late')

  def test_a_component_removed_during_observe_leaves_when_it_ends(self):
    old = _Note('old')
    agent = _agent(old=old, hook=_Hook(lambda e: e.remove_component('old')))

    agent.observe('a custom fades')

    self.assertEqual(old.calls, ['pre_observe', 'update'])  # The whole call.
    self.assertNotIn('old', agent.get_all_context_components())

  def test_a_change_requested_by_another_thread_waits_for_the_call(self):
    started, release = threading.Event(), threading.Event()

    class _Slow(_Note):

      @override
      def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        started.set()
        release.wait(timeout=10)
        return super().pre_act(action_spec)

    agent = _agent(slow=_Slow('slow'))
    result = []
    acting = threading.Thread(target=lambda: result.append(agent.act()))
    acting.start()
    self.assertTrue(started.wait(timeout=10))

    agent.add_component('late', _Note('late'))
    self.assertNotIn('late', agent.get_all_context_components())
    release.set()
    acting.join(timeout=10)

    self.assertEqual(result, ['slow'])
    self.assertIn('late', agent.get_all_context_components())

  def test_names_are_unique_and_removals_known(self):
    agent = _agent(a=_Note('one'))
    with self.assertRaises(ValueError):
      agent.add_component('a', _Note('again'))
    with self.assertRaises(KeyError):
      agent.remove_component('b')

  def test_a_component_of_another_entity_is_refused(self):
    shared = _Note('shared')
    _agent(a=shared)
    with self.assertRaises(ValueError):
      _agent().add_component('a', shared)

  def test_a_component_the_act_component_orders_cannot_be_removed(self):
    agent = entity_agent.EntityAgent(
        'Ada',
        act_component=_JoinContexts(order=['a']),
        context_components={'a': _Note('one')},
    )
    with self.assertRaises(ValueError):
      agent.remove_component('a')

  def test_an_added_component_logs_on_a_channel_of_its_name(self):
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        'Ada', act_component=_JoinContexts()
    )
    agent.add_component('custom', _Note('share the catch'))

    agent.act()

    self.assertEqual(
        agent.get_last_log()['custom'],
        {'Key': 'note', 'Value': 'share the catch'},
    )


class StateTest(absltest.TestCase):

  def test_added_and_removed_components_are_restored(self):
    agent = _agent(a=_Note('one'), b=_Note('two'))
    agent.add_component(
        'c', _note_factory('c', {'text': 'three'}), {'text': 'three'}
    )
    agent.remove_component('b')
    agent.act()
    state = json.loads(json.dumps(agent.get_state()))

    restored = _agent(a=_Note('one'), b=_Note('two'))
    restored.set_state(state)

    self.assertEqual(sorted(restored.get_all_context_components()), ['a', 'c'])
    self.assertEqual(restored.get_component('c', type_=_Note).acts, 1)
    self.assertEqual(restored.act(), 'one three')
    self.assertEqual(
        restored.get_state(),
        {**state, 'context_components': {'a': {'acts': 2}, 'c': {'acts': 2}}},
    )

  def test_a_state_without_changes_restores_the_built_in_components(self):
    agent = _agent(a=_Note('one'), b=_Note('two'))
    state = agent.get_state()
    agent.remove_component('b')
    agent.add_component('c', _Note('three'), {'text': 'three'})

    agent.set_state(state)

    self.assertEqual(sorted(agent.get_all_context_components()), ['a', 'b'])
    self.assertNotIn('dynamic_components', agent.get_state())

  def test_a_component_added_without_a_spec_is_not_restored(self):
    agent = _agent(a=_Note('one'))
    agent.add_component('c', _Note('three'))
    state = agent.get_state()

    restored = _agent(a=_Note('one'))
    restored.set_state(state)

    self.assertEqual(sorted(restored.get_all_context_components()), ['a'])

  def test_changes_in_one_call_apply_in_order_with_their_own_specs(self):
    def churn(entity):
      entity.add_component('c', _Note('three'), {'text': 'three'})
      entity.remove_component('c')
      entity.add_component('c', _Note('again'))

    agent = _agent(hook=_Hook(churn))
    agent.observe('a custom comes and goes')

    self.assertEqual(agent.get_component('c', type_=_Note).text, 'again')
    self.assertNotIn('dynamic_components', agent.get_state())

  def test_added_components_need_a_factory_to_be_restored(self):
    agent = _agent()
    agent.add_component('c', _Note('three'), {'text': 'three'})
    state = agent.get_state()

    without_factory = entity_agent.EntityAgent('Ada', _JoinContexts())
    with self.assertRaises(ValueError):
      without_factory.set_state(state)


if __name__ == '__main__':
  absltest.main()
