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

"""A module that acts like a registry of measurements for experimenter use."""

import threading
from typing import Any, Dict, Set
from reactivex import subject


class Measurements:
  """A registry of measurements for experimenter use."""

  def __init__(self):
    """Initializes the Measurements object."""
    self._channels: Dict[str, subject.Subject] = {}
    self._channels_lock: threading.Lock = threading.Lock()

  def _get_channel_or_create(self, channel: str) -> subject.Subject:
    """Create a channel if one doesn't already exist.

    Assumes the channels lock has been acquired. Raises RuntimeError if not.

    Args:
      channel: The channel name to create.

    Returns:
      The channel with the given name.

    Raises:
      RuntimeError: if the channels lock is not acquired.
    """
    if not self._channels_lock.locked():
      raise RuntimeError('Channels lock is not acquired.')
    if channel not in self._channels:
      # TODO(b/313610238): Maybe limit the number of new channels
      self._channels[channel] = subject.ReplaySubject()
    return self._channels[channel]

  def publish_datum(self, channel: str, datum: Any) -> None:
    """Publishes a datum to the channel.

    Args:
      channel: The channel name to push the datum into. If the channel doesn't
        exist yet, it will be created.
      datum: The payload to push into the channel.
    """
    with self._channels_lock:
      self._get_channel_or_create(channel).on_next(datum)

  def available_channels(self) -> Set[str]:
    """Returns the names of all available channels."""
    with self._channels_lock:
      keys: set[str] = set(self._channels.keys())
      return keys

  def get_channel(self, channel: str) -> subject.Subject:
    """Returns the channel for the given name.

    Args:
      channel: The channel name to get. If the channel doesn't exist yet, it
        will be created.
    """
    with self._channels_lock:
      return self._get_channel_or_create(channel)

  def close_channel(self, channel: str) -> None:
    """Closes the channel for the given name.

    Args:
      channel: The channel to close. If the channel doesn't exist yet, it will
        be created.
    """
    with self._channels_lock:
      self._get_channel_or_create(channel).on_completed()
      del self._channels[channel]

  def close(self) -> None:
    """Closes all channels."""
    with self._channels_lock:
      for channel in self._channels.values():
        channel.on_completed()
      self._channels.clear()
