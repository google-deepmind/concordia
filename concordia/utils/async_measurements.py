# Copyright 2025 DeepMind Technologies Limited.
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

"""Reactive measurements for async engines using reactivex."""

import contextlib
import threading
from typing import Any

from concordia.utils import measurements as measurements_lib
from reactivex import subject as reactivex_subject


class ReactiveMeasurements(measurements_lib.Measurements):
  """Measurements subclass that emits data via reactivex Subjects.

  In addition to storing data in channels (like the base Measurements class),
  this class emits each published datum through a reactivex Subject. This
  enables reactive subscribers to capture log data atomically as it is produced.

  Usage:
    measurements = ReactiveMeasurements()

    # Components set up logging channels as usual:
    component.set_logging_channel(
        measurements.get_channel('my_component').append
    )

    # To capture logs atomically during an act() call:
    with measurements.capture() as captured:
        result = entity.act(action_spec)
    # captured now contains {channel_name: datum} for all data emitted
    # during the act() call.
  """

  def __init__(self):
    super().__init__()
    self._subject = reactivex_subject.Subject()
    self._active_captures: dict[int, tuple[str, dict[str, Any]]] = {}
    self._capture_lock = threading.Lock()

  def publish_datum(
      self, channel: str, datum: Any, capture_key: str | None = None
  ) -> None:
    super().publish_datum(channel, datum, capture_key=capture_key)
    if capture_key is not None:
      with self._capture_lock:
        for _, (key, captured) in self._active_captures.items():
          if key == capture_key:
            captured[channel] = datum
    self._subject.on_next((channel, datum))

  @contextlib.contextmanager
  def capture(self, key: str):
    """Context manager that captures data published with a matching key.

    Only data emitted via publish_datum with a capture_key matching this
    capture's key will be stored. This prevents cross-contamination when
    multiple entity threads share the same ReactiveMeasurements instance.

    This captures data from ALL threads (not just the caller), which is
    necessary because EntityAgent._parallel_call_ runs components in worker
    threads.

    Args:
      key: The capture key, typically an entity or game master name.

    Yields:
      A dict that accumulates {channel_name: datum} entries.
    """
    captured: dict[str, Any] = {}
    capture_id = id(captured)
    with self._capture_lock:
      self._active_captures[capture_id] = (key, captured)
    try:
      yield captured
    finally:
      with self._capture_lock:
        self._active_captures.pop(capture_id, None)

  def subscribe(self, callback):
    """Subscribe to all datum emissions.

    Args:
      callback: Called with (channel_name, datum) for each emission.

    Returns:
      A disposable subscription.
    """
    return self._subject.subscribe(on_next=callback)

  def dispose(self):
    """Complete the subject and release resources."""
    self._subject.on_completed()
    self._subject.dispose()
