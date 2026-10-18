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

"""The Chronicler's act: a rule that hands in what its slow mind wrote.

The Chronicler acts every tick without a language model: acting is what runs
its slow mind (a `pubsub.Publisher`) in the background. When the mind has
finished a chapter the Chronicler has not handed in, its act hands it in,
the chapter's words in the action (`actors.HAND_IN_CHAPTER`), and the World
GM publishes it on the chronicle; otherwise it keeps watch.
"""

from collections.abc import Mapping
from typing import override

from concordia.components.agent import pubsub
from examples.deep_time import actors as actors_lib
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ChroniclerAct(entity_component.ActingComponent):
  """Hands in the newest chapter its slow mind wrote, else keeps watch."""

  def __init__(self, draft_component_key: str, *, blank: str = ''):
    """Initializes the component.

    Args:
      draft_component_key: Key of the `pubsub.Subscriber` to the slow mind's
        chapters, on the same entity.
      blank: Its value before the mind has written anything.
    """
    super().__init__()
    self._draft_component_key = draft_component_key
    self._handed = blank

  @override
  def get_action_attempt(
      self,
      contexts: Mapping[str, str],
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    del contexts, action_spec
    draft = (
        self.get_entity()
        .get_component(self._draft_component_key, type_=pubsub.Subscriber)
        .get_current_value()
        .strip()
    )
    if not draft or draft == self._handed:
      return actors_lib.CHRONICLER_RESPONSE
    self._handed = draft
    return f'{actors_lib.HAND_IN_CHAPTER}: {draft}'

  @override
  def get_state(self) -> entity_component.ComponentState:
    return {'handed': self._handed}

  @override
  def set_state(self, state: entity_component.ComponentState) -> None:
    self._handed = str(state.get('handed', self._handed))
