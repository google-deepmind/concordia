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

"""Scenarios: the Sandbox, and presets of it (settings, not code)."""

from collections.abc import Mapping
import copy
from typing import Any

from examples.lifeways.world import world as world_lib

SANDBOX: Mapping[str, Any] = {
    'scenario': 'sandbox',
    'seed': 0,
    'rows': 16,
    'cols': 24,
    'richness': 0.45,
    'founders': 12,
    'mode': 'sexual',
    'mutation': 0.05,  # Copying errors at birth.
    # The channels customs pass by among the living, and their reach and
    # fidelity; leave one out and it does not exist.
    'channels': copy.deepcopy(dict(world_lib.DEFAULT_CHANNELS)),
    'child_cost': 2.0,
    'max_years': None,  # The Sandbox runs indefinitely.
    'history_kept': 200,
    'founding_customs': [
        {
            'kind': 'norm',
            'domain': 'kinship',
            'text': 'We raise a child only when the stores can feed it.',
        },
        {
            'kind': 'norm',
            'domain': 'sharing',
            'text': 'We share our food with those near us who go hungry.',
            'held_by': 0.5,
        },
        {
            'kind': 'skill',
            'domain': 'food',
            'text': 'Smoking fish over green alder.',
            'params': {'level': 2},
        },
        {
            'kind': 'belief',
            'domain': 'sacred',
            'text': 'Storms come when the shrine is neglected.',
        },
        {
            'kind': 'value',
            'domain': 'identity',
            'text': 'We are the River people; we do not forget a debt.',
        },
        {
            'kind': 'habit',
            'domain': 'food',
            'text': 'In lean years, gather before anything else.',
        },
        {
            'kind': 'bias',
            'domain': 'learning',
            'text': 'Learn from those whose stores are full.',
            'held_by': 0.5,
        },
    ],
}

PRESETS: Mapping[str, Mapping[str, Any]] = {
    'sandbox': {},
}


def settings(name: str = 'sandbox', **overrides: Any) -> dict[str, Any]:
  """A scenario's settings: the Sandbox's, a preset's changes, overrides."""
  merged = copy.deepcopy(dict(SANDBOX))
  merged.update(copy.deepcopy(dict(PRESETS[name])))
  merged['scenario'] = name
  merged.update(overrides)
  return merged
