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

"""Transport-neutral requests for human-controlled entities.

A reader may use a terminal, a GUI, or a blocking queue serviced by a web
adapter. No transport, session, or network dependency belongs in this module.
"""

from collections.abc import Mapping
import dataclasses
from typing import Protocol

from concordia.typing import entity as entity_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class HumanInputRequest:
  """One action request, including any feedback from an invalid response.

  Attributes:
    request_id: Opaque identifier, stable across validation retries and unique
      for each action. Adapters should reject responses to stale requests.
    entity_name: Entity being controlled. Adapters may route by this name or use
      a separate reader instance per entity; no global input queue is required.
    action_spec: The actual specification, with {name} replaced in its prompt.
      NEXT_ACTION_SPEC requires a JSON action specification for a player.
    contexts: Read-only snapshot of this entity's pre-act contexts. These may
      contain private information: only show them to its human controller.
    context: This entity's complete pre-act context assembled by the acting
      component in its configured order, with labels and line breaks intact.
      Transports should display this string, not reassemble the contexts map.
    error: Human-readable validation feedback, or None on the first attempt.
    previous_response: The invalid response, retained so a UI can allow editing.
  """

  request_id: str
  entity_name: str
  action_spec: entity_lib.ActionSpec
  contexts: Mapping[str, str]
  context: str = ''
  error: str | None = None
  previous_response: str | None = None


class HumanInput(Protocol):
  """Blocking input boundary used by HumanActComponent.

  Return a string once the human submits an answer. The component retries
  invalid responses using the same request_id and updated feedback. Do not
  automatically resubmit an invalid answer. EOF, cancellation, and transport
  failures should raise; they are deliberately not converted into game actions.

  Transport adapters own timeouts, disconnect/reconnect behavior, and
  cancellation. A mobile disconnect need not cancel a pending request. If an
  exception aborts an EntityAgent action, callers should end that run rather
  than assume the agent lifecycle has reset.
  """

  def __call__(self, request: HumanInputRequest) -> str:
    """Wait for a response from the controller of the requested entity."""
    ...
