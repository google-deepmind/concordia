# Copyright 2023 DeepMind Technologies Limited.
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


"""Metrics for simulations."""

import abc
from typing import Any

from concordia.document import interactive_document


class Metric(metaclass=abc.ABCMeta):
  """A class to hold logic for tracking state variables of a simulation."""

  @abc.abstractmethod
  def name(
      self,
  ) -> str:
    """Returns the name of the measurement."""
    raise NotImplementedError

  @abc.abstractmethod
  def update(
      self,
      observation: str,
      active_player_name: str,
      document: interactive_document.InteractiveDocument,
  ) -> None:
    """Process the observation then compute metric and store it."""
    raise NotImplementedError

  @abc.abstractmethod
  def state(self) -> list[dict[str, Any]] | None:
    """Return the current state of all the tracked variables."""
    raise NotImplementedError
