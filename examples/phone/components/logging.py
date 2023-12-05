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

"""Logger."""

import termcolor


class Logger:
  """Utility for logs messages depending on verbosity."""

  def __init__(self, color: str = 'magenta', verbose=False, semi_verbose=True):
    self._color = color
    self._verbose = verbose
    self._semi_verbose = semi_verbose

  def verbose(self, entry: str):
    if self._verbose:
      self._log(entry)

  def semi_verbose(self, entry: str):
    if self._semi_verbose:
      self._log(entry)

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._color))
