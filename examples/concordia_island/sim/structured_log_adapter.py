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

"""structured_log logging adapter for Concordia island simulation.

Converts Concordia's raw_log format to structured_log entries.
"""

from collections.abc import Mapping, Sequence
import json
from typing import Any

from absl import logging


def raw_log_to_structured_log_entries(
    raw_log: Sequence[Mapping[str, Any]],
    experiment_id: str | None = None,
) -> list[dict[str, Any]]:
  """Convert Concordia raw_log to structured_log-compatible entries.

  Args:
    raw_log: List of log entries from Simulation.play()
    experiment_id: Optional experiment/run identifier

  Returns:
    List of structured_log entry dictionaries
  """
  structured_log_entries = []

  for i, entry in enumerate(raw_log):
    step = entry.get("Step", i)

    entity_entries = {k: v for k, v in entry.items() if k.startswith("Entity")}
    gm_entries = {
        k: v
        for k, v in entry.items()
        if not k.startswith("Entity") and k not in ("Step", "Summary")
    }

    base_entry = {
        "step": step,
        "experiment_id": experiment_id,
    }

    for entity_key, entity_log in entity_entries.items():
      entity_name = entity_key.replace("Entity [", "").replace("]", "")
      structured_log_entries.append({
          **base_entry,
          "type": "entity_action",
          "entity_name": entity_name,
          "data": sanitize_for_json(entity_log),
      })

    for gm_key, gm_log in gm_entries.items():
      if isinstance(gm_log, dict):
        make_obs = gm_log.get("make_observation", {})
        if make_obs:
          structured_log_entries.append({
              **base_entry,
              "type": "observation",
              "game_master": gm_key,
              "data": sanitize_for_json(make_obs),
          })

        resolve = gm_log.get("resolve", {})
        if resolve:
          structured_log_entries.append({
              **base_entry,
              "type": "event_resolution",
              "game_master": gm_key,
              "data": sanitize_for_json(resolve),
          })

  return structured_log_entries


def sanitize_for_json(obj: Any) -> Any:
  """Sanitize object for JSON serialization."""
  if isinstance(obj, dict):
    return {k: sanitize_for_json(v) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [sanitize_for_json(item) for item in obj]
  elif isinstance(obj, (str, int, float, bool, type(None))):
    return obj
  else:
    return str(obj)


def log_to_structured_log(
    structured_log_entries: Sequence[dict[str, Any]],
    writer,
) -> None:
  """Write entries to structured_log using provided writer.

  Args:
    structured_log_entries: Entries from raw_log_to_structured_log_entries()
    writer: structured_log writer instance
  """
  for entry in structured_log_entries:
    try:
      writer.write(entry)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to write structured_log entry: %s", e)


def save_html_log(
    simulation_log,
    output_path: str,
) -> None:
  """Save simulation log as HTML file.

  Args:
    simulation_log: SimulationLog from sim.play()
    output_path: Path to write HTML file
  """
  html_content = simulation_log.to_html()
  with open(output_path, "w") as f:
    f.write(html_content)
  logging.info("Saved HTML log to %s", output_path)


def save_raw_log_json(
    raw_log: Sequence[Mapping[str, Any]],
    output_path: str,
) -> None:
  """Save raw log as JSON file.

  Args:
    raw_log: Raw log from simulation
    output_path: Path to write JSON file
  """
  sanitized = [sanitize_for_json(dict(entry)) for entry in raw_log]
  with open(output_path, "w") as f:
    json.dump(sanitized, f, indent=2)
  logging.info("Saved raw log JSON to %s", output_path)
