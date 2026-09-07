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

"""Tests for the language model setup registry."""

import importlib.util

from absl.testing import absltest
from concordia.contrib import language_models


class LanguageModelsRegistryTest(absltest.TestCase):
  """Tests that every registry entry resolves to an importable module."""

  def test_all_registry_entries_resolve_to_modules(self):
    for api_type, model_path in language_models._REGISTRY.items():
      module_path, _ = model_path.rsplit('.', 1)
      full_module = f'concordia.contrib.language_models.{module_path}'
      self.assertIsNotNone(
          importlib.util.find_spec(full_module),
          msg=(
              f'api_type {api_type!r} references module {full_module!r} which'
              ' does not exist.'
          ),
      )

  def test_google_aistudio_is_a_registered_api_type(self):
    # `google_aistudio` is the default api_type used by the example run
    # scripts and the persona generator CLI.
    self.assertIn('google_aistudio', language_models._REGISTRY)


if __name__ == '__main__':
  absltest.main()
