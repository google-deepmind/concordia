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

"""Optional local embeddings for the standard memory-bank callable interface.

No custom retrieval or implicit model downloads. Caller-selected models must
already be installed in the local Ollama service and support embeddings.
"""

import contextlib
import threading

from concordia.utils import profiler as profiler_lib
import numpy as np
import ollama


class OllamaEmbedder:
  """Finite normalized vectors with bounded local HTTP and private metrics."""

  def __init__(
      self,
      model: str,
      *,
      timeout: float = 30,
      profiler: profiler_lib.ProfilerContext | None = None
  ):
    if not isinstance(model, str) or not model.strip():
      raise ValueError('Choose an installed local embedding model.')
    if not np.isfinite(timeout) or timeout <= 0:
      raise ValueError('Embedding timeout must be finite and positive.')
    self.model = model
    self._profiler = profiler
    self._lock = threading.Lock()
    self._dimensions = None
    # Do not inherit proxy/remote-host settings for private memory text.
    self._client = ollama.Client(
        host='http://127.0.0.1:11434', timeout=timeout, trust_env=False
    )
    information = self._client.show(model)
    if 'embedding' not in (information.get('capabilities') or []):
      raise ValueError('Selected local model does not support embeddings.')

  def __call__(self, text: str) -> np.ndarray:
    if not isinstance(text, str):
      raise ValueError('Memory input must be text.')
    profile = self._profiler
    if profile:
      profile.increment_counter('embedding.requests')
    timing = profile.track('embedding') if profile else contextlib.nullcontext()
    try:
      with timing:
        response = self._client.embed(
            model=self.model, input=text, truncate=False, keep_alive='5m'
        )
        rows = response.get('embeddings', [])
        if len(rows) != 1:
          raise ValueError(
              'Embedding response must contain exactly one vector.'
          )
        vector = np.asarray(rows[0], dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
          raise ValueError('Embedding must be a finite nonempty vector.')
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
          raise ValueError('Embedding must have a finite nonzero norm.')
        with self._lock:
          if self._dimensions is not None and vector.size != self._dimensions:
            raise ValueError('Embedding dimensions changed within one run.')
          self._dimensions = vector.size
        vector = vector / norm
        if profile:
          profile.record_value('embedding.dimensions', float(vector.size))
        return vector
    except Exception:
      if profile:
        profile.increment_counter('embedding.failures')
      raise  # Never silently replace failed semantic vectors with placeholders.
