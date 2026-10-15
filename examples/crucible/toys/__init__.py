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

"""The registry of crucible toys: name -> how to build it, mock it, bot it."""

from collections.abc import Callable, Mapping
import dataclasses
import os
import random
from typing import Any

from concordia.language_model import language_model
import yaml

from examples.crucible.toys import base

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

Policy = Callable[[Any, str, random.Random], str]


def load_data(name: str) -> dict[str, Any]:
  with open(os.path.join(DATA_DIR, f'{name}.yaml')) as f:
    return yaml.safe_load(f)


@dataclasses.dataclass(frozen=True)
class ToyKind:
  """Everything the harness needs to run one toy.

  Attributes:
    name: The toy's key.
    build: (model, data, scenario) -> a new toy.
    mock: data -> the toy's mock answerer (`choice`, `text`).
    policies: Bot policies by name ('skilled', 'random', 'passive').
    seat: The actor a person or a bot plays.
  """

  name: str
  build: Callable[
      [language_model.LanguageModel, Mapping[str, Any], str], base.Toy
  ]
  mock: Callable[[Mapping[str, Any]], Any]
  policies: Mapping[str, Policy]
  seat: str


def registry() -> dict[str, ToyKind]:
  # pylint: disable=g-import-not-at-top
  from examples.crucible.toys.claims import bots as claims_bots
  from examples.crucible.toys.claims import mock as claims_mock
  from examples.crucible.toys.claims import toy as claims_toy
  from examples.crucible.toys.relay import bots as relay_bots
  from examples.crucible.toys.relay import mock as relay_mock
  from examples.crucible.toys.relay import toy as relay_toy
  # pylint: enable=g-import-not-at-top
  return {
      'claims': ToyKind(
          name='claims',
          build=claims_toy.ClaimsToy,
          mock=claims_mock.ClaimsMock,
          policies=claims_bots.POLICIES,
          seat='Player',
      ),
      'relay': ToyKind(
          name='relay',
          build=relay_toy.RelayToy,
          mock=relay_mock.RelayMock,
          policies=relay_bots.POLICIES,
          seat='Relay',
      ),
  }
