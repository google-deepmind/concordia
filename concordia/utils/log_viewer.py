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

"""Utilities for building portable structured log viewers."""

import importlib.resources
import json

from concordia.utils import structured_logging


_LOG_DATA_MARKER = 'let LOG_DATA = null;'
_SCRIPT_END_MARKER = '</script>'


def build_self_contained_viewer(
    log: structured_logging.SimulationLog,
    *,
    log_name: str = 'structured log',
) -> str:
  """Embeds a structured log in the standard viewer HTML.

  Args:
    log: The structured simulation log to embed.
    log_name: A display name for the preloaded log.

  Returns:
    A self-contained HTML document.

  Raises:
    ValueError: If the packaged viewer template is missing expected markers.
  """
  viewer = (
      importlib.resources.files('concordia.utils')
      .joinpath('log_viewer.html')
      .read_text(encoding='utf-8')
  )
  if _LOG_DATA_MARKER not in viewer:
    raise ValueError('Log viewer template is missing the log data marker.')

  # Escaping every start-tag slash prevents embedded log text from terminating
  # the surrounding script element. ensure_ascii also makes JavaScript line
  # separator characters safe inside the generated source.
  embedded_log = json.dumps(log.to_dict(), ensure_ascii=True).replace(
      '</', '<\\/'
  )
  viewer = viewer.replace(
      _LOG_DATA_MARKER, f'let LOG_DATA = {embedded_log};', 1
  )

  script_index = viewer.rfind(_SCRIPT_END_MARKER)
  if script_index < 0:
    raise ValueError('Log viewer template is missing its closing script tag.')
  display_name = json.dumps(log_name, ensure_ascii=True)
  bootstrap = f"""
window.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('fileInfo').textContent =
      'Preloaded structured log: ' + {display_name};
  renderViewer();
}});
"""
  return viewer[:script_index] + bootstrap + viewer[script_index:]
