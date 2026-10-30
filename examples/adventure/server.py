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
"""HTTP and Server-Sent Events (SSE) server for interactive Adventure play.

Connects a browser UI (`http://127.0.0.1:<PORT>`) to `AdventureSimulation`
via a non-blocking `BrowserInputBridge` (implementing
`human_input.HumanInput`) and a real-time `TopicBus` SSE subscriber.
"""

import base64
from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import hashlib
import html
import http
from http import server as http_server
import importlib
import inspect
import json
import os
import queue
import re
import threading
import time
from typing import Any

from concordia.components.agent import human_act_component
from concordia.contrib.pubsub import topics
from examples.adventure import adventure_prefab
from examples.adventure import author_world
from examples.adventure import story_state
from examples.adventure import world_ledger
from concordia.typing import human_input

_MAX_ACTION_LENGTH = 500
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_ALLOWED_STATIC_FILES = frozenset({'index.html', 'styles.css'})
_ALLOWED_IMAGE_MIMES = frozenset({
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/svg+xml; charset=utf-8',
})
_IMAGE_KEY_RE = re.compile(r'^[a-z0-9_]{1,80}$')
_INLINE_SCRIPT_PATTERN = re.compile(
    rb'<script id="adventure-client-script">(.*?)</script>',
    re.DOTALL,
)
DEFAULT_IMAGE_MODEL = 'gemini-3.1-flash-image'
DEFAULT_FALLBACK_IMAGE_MODEL = 'gemini-3.1-flash-lite-image'
DEFAULT_DIRECTOR_TEXT_MODEL = 'gemini-flash-latest'
_PHOTOGRAPHY_BEAT_KINDS = frozenset({
    'intro',
    'scene_resolution',
    'npc_resolution',
    'arrival',
})


def _load_genai_modules() -> tuple[Any | None, Any | None]:
  """Lazily imports `google.genai` and `google.genai.types` when available."""
  try:
    genai = importlib.import_module('google.genai')
    genai_types = importlib.import_module('google.genai.types')
    return genai, genai_types
  except ImportError:
    return None, None


def _extract_genai_client(model: Any) -> Any | None:
  """Unwraps a Concordia LanguageModel to find or create a `genai.Client`."""
  cur = model
  has_custom_client = False
  for _ in range(4):
    if cur is None:
      break
    client = getattr(cur, '_client', None)
    if client is not None:
      if hasattr(getattr(client, 'models', None), 'generate_content'):
        return client
      if hasattr(client, 'generate_content'):
        has_custom_client = True
    cur = getattr(cur, '_model', None)
  api_key = os.environ.get('GOOGLE_API_KEY', '') or os.environ.get(
      'GEMINI_API_KEY', ''
  )
  if api_key and (has_custom_client or model is not None):
    genai, _ = _load_genai_modules()
    if genai is not None:
      try:
        return genai.Client(api_key=api_key)
      except Exception:  # pylint: disable=broad-exception-caught
        return None
  return None


def _extract_director_text_model(model: Any) -> Any | None:
  """Returns `model` if it wraps a live text generation client."""
  cur = model
  for _ in range(4):
    if cur is None:
      return None
    client = getattr(cur, '_client', None)
    if client is not None and (
        hasattr(client, 'generate_content')
        or hasattr(getattr(client, 'models', None), 'generate_content')
        or hasattr(client, 'chat')
    ):
      return model
    cur = getattr(cur, '_model', None)
  return None


def _detect_safe_image_mime(image_bytes: bytes) -> str | None:
  """Inspects magic bytes to confirm binary image format."""
  if not image_bytes or len(image_bytes) > _MAX_IMAGE_BYTES:
    return None
  if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
    return 'image/png'
  if image_bytes.startswith(b'\xff\xd8\xff'):
    return 'image/jpeg'
  if (
      len(image_bytes) >= 12
      and image_bytes[:4] == b'RIFF'
      and image_bytes[8:12] == b'WEBP'
  ):
    return 'image/webp'
  return None


def _build_fallback_svg(
    *, kind: str, title: str, subtitle: str
) -> tuple[str, bytes]:
  """Builds a deterministic atmospheric SVG illustration when offline."""
  safe_title = html.escape((title or 'Scene').strip()[:44])
  safe_sub = html.escape((subtitle or '').strip()[:76])
  digest = hashlib.sha256(f'{kind}:{title}'.encode('utf-8')).digest()
  hue_a = 195 + (digest[0] % 50)
  hue_b = 25 + (digest[1] % 35)
  if kind == 'npc':
    words = [w for w in re.findall(r'[A-Za-z0-9]+', title or '') if w]
    initials = ''.join(w[0].upper() for w in words[:2]) or 'NP'
    safe_init = html.escape(initials)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="hsl({hue_a},38%,18%)"/>'
        f'<stop offset="100%" stop-color="hsl({hue_b},45%,12%)"/>'
        '</linearGradient></defs>'
        '<rect width="160" height="160" rx="18" fill="url(#g)"/>'
        '<circle cx="80" cy="60" r="26" fill="rgba(255,255,255,0.14)"/>'
        '<path d="M34 142 C38 104 122 104 126 142 Z" '
        'fill="rgba(255,255,255,0.12)"/>'
        '<text x="80" y="69" text-anchor="middle" fill="#f3e9d2" '
        'font-family="Georgia,serif" font-size="22" font-weight="bold">'
        f'{safe_init}</text>'
        '<text x="80" y="148" text-anchor="middle" fill="#cbd5e1" '
        f'font-family="sans-serif" font-size="10">{safe_title[:18]}</text>'
        '</svg>'
    )
  else:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 320">'
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="hsl({hue_a},42%,14%)"/>'
        f'<stop offset="65%" stop-color="hsl({hue_a},32%,22%)"/>'
        f'<stop offset="100%" stop-color="hsl({hue_b},38%,12%)"/>'
        '</linearGradient></defs>'
        '<rect width="640" height="320" fill="url(#bg)"/>'
        '<circle cx="510" cy="78" r="34" fill="rgba(245,208,125,0.18)"/>'
        '<path d="M0 245 L95 190 L170 218 L275 162 L380 215 L490 172 '
        'L640 225 L640 320 L0 320 Z" fill="rgba(10,15,24,0.68)"/>'
        '<path d="M0 272 L150 242 L310 264 L470 236 L640 262 L640 320 '
        'L0 320 Z" fill="rgba(6,9,15,0.88)"/>'
        '<text x="28" y="274" fill="#f8fafc" font-family="Georgia,serif" '
        f'font-size="22" font-weight="bold">{safe_title}</text>'
        '<text x="28" y="298" fill="#94a3b8" font-family="sans-serif" '
        f'font-size="12">{safe_sub}</text>'
        '</svg>'
    )
  return 'image/svg+xml; charset=utf-8', svg.encode('utf-8')


class SceneImageGenerator:
  """Photography Director and cache for scene panels and NPC portraits."""

  def __init__(
      self,
      *,
      client: Any | None = None,
      text_model: Any | None = None,
      image_model: str = DEFAULT_IMAGE_MODEL,
      fallback_image_model: str = DEFAULT_FALLBACK_IMAGE_MODEL,
  ):
    """Initializes the Photography Director and background worker pool."""
    self._client = client
    self._text_model = text_model
    self._image_model = image_model
    self._fallback_image_model = fallback_image_model
    self._lock = threading.Lock()
    self._cache: dict[str, tuple[str, bytes]] = {}
    self._captions: dict[str, str] = {}
    self._in_flight: set[str] = set()
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix='adventure-img-gen'
    )

  def clear_cache(self) -> None:
    """Clears scene panel cache on reset while keeping NPC portraits cached."""
    with self._lock:
      loc_keys = [k for k in self._cache if k.startswith('loc_')]
      for k in loc_keys:
        del self._cache[k]
        self._captions.pop(k, None)
      self._in_flight.clear()

  def get_image(self, image_key: str) -> tuple[str, bytes] | None:
    """Returns `(mime_type, bytes)` for `image_key`, or None if absent."""
    if not _IMAGE_KEY_RE.fullmatch(image_key):
      return None
    with self._lock:
      return self._cache.get(image_key)

  def _generate_via_client(
      self, *, prompt: str, aspect_ratio: str
  ) -> tuple[str, bytes] | None:
    """Calls Gemini Flash Image with fallback to Imagen Fast."""
    if self._client is None:
      return None
    _, genai_types = _load_genai_modules()
    if genai_types is None:
      return None
    for attempt in range(2):
      try:
        response = self._client.models.generate_content(
            model=self._image_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=genai_types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )
        for part in getattr(response, 'parts', None) or []:
          inline = getattr(part, 'inline_data', None)
          raw_bytes = getattr(inline, 'data', None) if inline else None
          if isinstance(raw_bytes, bytes):
            mime = _detect_safe_image_mime(raw_bytes)
            if mime is not None:
              return mime, raw_bytes
      except Exception:  # pylint: disable=broad-exception-caught
        if attempt == 0:
          time.sleep(1.5)

    if self._fallback_image_model:
      try:
        fallback_resp = self._client.models.generate_content(
            model=self._fallback_image_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=genai_types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )
        for part in getattr(fallback_resp, 'parts', None) or []:
          inline = getattr(part, 'inline_data', None)
          raw_bytes = getattr(inline, 'data', None) if inline else None
          if isinstance(raw_bytes, bytes):
            mime = _detect_safe_image_mime(raw_bytes)
            if mime is not None:
              return mime, raw_bytes
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return None

  def _direct_visual_prompt(
      self, *, director_instruction: str, fallback_prompt: str
  ) -> str:
    """Uses Gemini Flash to rewrite game state into a literal visual prompt."""
    if self._text_model is not None:
      try:
        rewritten = (
            self._text_model.sample_text(
                director_instruction,
                max_tokens=320,
                terminators=(),
                temperature=0.2,
            )
            or ''
        ).strip()
        if len(rewritten) >= 40:
          return rewritten
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    if self._client is None:
      return fallback_prompt
    _, genai_types = _load_genai_modules()
    if genai_types is None:
      return fallback_prompt
    try:
      resp = self._client.models.generate_content(
          model=DEFAULT_DIRECTOR_TEXT_MODEL,
          contents=director_instruction,
          config=genai_types.GenerateContentConfig(
              max_output_tokens=320,
              temperature=0.2,
              thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
          ),
      )
      rewritten = (getattr(resp, 'text', None) or '').strip()
      if len(rewritten) >= 40:
        return rewritten
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return fallback_prompt

  def _direct_photography_shot(
      self,
      *,
      director_instruction: str,
      fallback_prompt: str,
      fallback_caption: str,
  ) -> tuple[str, str]:
    """Runs the Photography Director LLM to select camera framing and prompt."""
    if self._text_model is None and self._client is None:
      return fallback_caption, fallback_prompt
    try:
      raw = ''
      if self._text_model is not None:
        raw = (
            self._text_model.sample_text(
                director_instruction,
                max_tokens=360,
                terminators=(),
                temperature=0.25,
            )
            or ''
        ).strip()
      if not raw and self._client is not None:
        _, genai_types = _load_genai_modules()
        if genai_types is not None:
          resp = self._client.models.generate_content(
              model=DEFAULT_DIRECTOR_TEXT_MODEL,
              contents=director_instruction,
              config=genai_types.GenerateContentConfig(
                  max_output_tokens=360,
                  temperature=0.25,
                  thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
              ),
          )
          raw = (getattr(resp, 'text', None) or '').strip()
      if raw:
        caption = fallback_caption
        prompt_lines: list[str] = []
        for line in raw.splitlines():
          stripped = line.strip()
          if not stripped:
            continue
          if stripped.upper().startswith('CAPTION:'):
            cap_val = stripped.split(':', 1)[1].strip().strip('"\'')
            if cap_val:
              caption = cap_val[:140]
          elif stripped.upper().startswith('PROMPT:'):
            p_val = stripped.split(':', 1)[1].strip()
            if p_val:
              prompt_lines.append(p_val)
          else:
            prompt_lines.append(stripped)
        final_prompt = ' '.join(prompt_lines).strip()
        if len(final_prompt) >= 40:
          return caption, final_prompt
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return fallback_caption, fallback_prompt

  def _complete_location_image(
      self,
      *,
      ledger: world_ledger.WorldLedger,
      location_id: str,
      image_key: str,
      title: str,
      description: str,
      prompt: str,
      director_instruction: str = '',
      entry_id: int | None = None,
      fallback_caption: str = '',
  ) -> None:
    """Worker task that populates a Photography Director scene panel."""
    try:
      shot_caption = fallback_caption
      if director_instruction:
        shot_caption, final_prompt = self._direct_photography_shot(
            director_instruction=director_instruction,
            fallback_prompt=prompt,
            fallback_caption=fallback_caption,
        )
      else:
        final_prompt = prompt
      result = self._generate_via_client(
          prompt=final_prompt, aspect_ratio='16:9'
      )
      if result is None:
        result = _build_fallback_svg(
            kind='location', title=title, subtitle=shot_caption or description
        )
      with self._lock:
        self._cache[image_key] = result
        if shot_caption:
          self._captions[image_key] = shot_caption
      url = f'/api/image/{image_key}'
      if entry_id is not None:
        ledger.set_narrative_entry_image(
            entry_id,
            url,
            shot_caption=shot_caption,
            location_id=location_id,
        )
      else:
        ledger.set_location_image_url(location_id, url)
    finally:
      with self._lock:
        self._in_flight.discard(image_key)

  def _complete_npc_portrait(
      self,
      *,
      ledger: world_ledger.WorldLedger,
      npc_name: str,
      image_key: str,
      role_summary: str,
      prompt: str,
      director_instruction: str = '',
  ) -> None:
    """Worker task that populates an NPC's portrait."""
    try:
      if director_instruction:
        final_prompt = self._direct_visual_prompt(
            director_instruction=director_instruction,
            fallback_prompt=prompt,
        )
      else:
        final_prompt = prompt
      result = self._generate_via_client(
          prompt=final_prompt, aspect_ratio='1:1'
      )
      if result is None:
        result = _build_fallback_svg(
            kind='npc', title=npc_name, subtitle=role_summary
        )
      with self._lock:
        self._cache[image_key] = result
      ledger.set_npc_portrait_url(npc_name, f'/api/image/{image_key}')
    finally:
      with self._lock:
        self._in_flight.discard(image_key)

  def ensure_scene_images(self, ledger: world_ledger.WorldLedger) -> None:
    """Schedules Photography Director panels for beats and cached NPC avatars."""
    setting_style = (
        ledger.story.tone.setting_style
        or ledger.story.tone.name
        or 'gritty atmospheric narrative setting'
    )
    world_tag = hashlib.sha256(ledger.premise.encode('utf-8')).hexdigest()[:6]
    loc_id = ledger.player_location_id
    current_loc = ledger.get_location(loc_id)
    co_located = ledger.npcs_at_location(loc_id)

    # 1. Ensure co-located NPC portraits (cached per character across the run).
    for npc in co_located:
      npc_slug = world_ledger.normalize_location_id(npc.name)[:48]
      npc_key = f'npc_{world_tag}_{npc_slug}'
      if not _IMAGE_KEY_RE.fullmatch(npc_key) or npc.portrait_url:
        continue
      should_schedule = False
      with self._lock:
        cached = self._cache.get(npc_key)
        if cached is not None:
          ledger.set_npc_portrait_url(npc.name, f'/api/image/{npc_key}')
        elif npc_key not in self._in_flight:
          self._in_flight.add(npc_key)
          should_schedule = True
      if should_schedule:
        npc_prompt = (
            'Single-frame graphic novel character portrait, head and '
            'shoulders, crisp detailed ink linework with dramatic lighting '
            'and color, no panel borders, no speech bubbles, no text. '
            f'World aesthetic: {setting_style}. '
            f'Character to depict literally: {npc.name} — '
            f'{npc.role_summary[:240]}. '
            f'Visual demeanor and traits: {npc.traits[:180]}.'
        )
        npc_director_instruction = (
            'You are a Visual Art Director writing a 50-75 word image prompt '
            'for a single-frame 1:1 graphic novel character portrait (head and '
            'shoulders, bold ink linework and atmospheric color, no panel '
            'borders, no speech bubbles, no text). Depict ONLY this single '
            'character in their exact described clothing, facial expression, '
            'and era.\n\n'
            f'WORLD SETTING: {setting_style}\n'
            f'CHARACTER: {npc.name}\n'
            f'APPEARANCE & ROLE: {npc.role_summary}\n'
            f'TRAITS & DEMEANOR: {npc.traits}\n'
            f'CURRENT LOCATION: {current_loc.name if current_loc else ""}\n\n'
            'Output ONLY the final 50-75 word image prompt paragraph.'
        )
        if self._client is None:
          self._complete_npc_portrait(
              ledger=ledger,
              npc_name=npc.name,
              image_key=npc_key,
              role_summary=npc.role_summary,
              prompt=npc_prompt,
          )
        else:
          self._executor.submit(
              self._complete_npc_portrait,
              ledger=ledger,
              npc_name=npc.name,
              image_key=npc_key,
              role_summary=npc.role_summary,
              prompt=npc_prompt,
              director_instruction=npc_director_instruction,
          )

    # 2. Photography Director: shoot a 16:9 graphic-novel panel for each new
    # visual beat ('intro', 'scene_resolution', 'npc_resolution', 'arrival').
    narrative_log = ledger.get_narrative_log()
    visual_entries = [
        e
        for e in narrative_log[-6:]
        if e.get('kind') in _PHOTOGRAPHY_BEAT_KINDS and e.get('id') is not None
    ]

    last_player_action = ''
    for e in reversed(narrative_log):
      if e.get('kind') == 'player_action' and e.get('text'):
        last_player_action = str(e['text'])[:200]
        break

    if visual_entries and current_loc is not None:
      for entry in visual_entries[-2:]:
        if entry.get('image_url'):
          continue
        entry_id = int(entry['id'])
        entry_loc_id = str(entry.get('location_id') or loc_id)
        entry_loc = ledger.get_location(entry_loc_id) or current_loc
        loc_npcs = ledger.npcs_at_location(entry_loc_id)
        loc_slug = world_ledger.normalize_location_id(entry_loc_id)[:36]
        loc_key = f'loc_{world_tag}_e{entry_id}_{loc_slug}'
        if not _IMAGE_KEY_RE.fullmatch(loc_key):
          continue

        should_schedule = False
        with self._lock:
          cached = self._cache.get(loc_key)
          if cached is not None:
            ledger.set_narrative_entry_image(
                entry_id,
                f'/api/image/{loc_key}',
                shot_caption=self._captions.get(loc_key, ''),
                location_id=entry_loc_id,
            )
          elif loc_key not in self._in_flight:
            self._in_flight.add(loc_key)
            should_schedule = True

        if not should_schedule:
          continue

        beat_kind = str(entry.get('kind') or 'scene_resolution')
        beat_actor = str(entry.get('actor') or 'Scene')
        beat_text = str(entry.get('text') or '').strip()
        props_text = ', '.join(entry_loc.affordances[:5])
        if loc_npcs:
          npc_lines = '; '.join(
              f'{n.name} ({n.role_summary[:180]})' for n in loc_npcs
          )
          cast_clause = (
              f'EXACTLY {len(loc_npcs)} OTHER PERSON(S) IN THE ROOM: '
              f'{npc_lines}. ZERO background patrons, ZERO crowd, ZERO extras.'
          )
        else:
          cast_clause = (
              'EXACTLY 0 OTHER PEOPLE IN THE ROOM: empty interior, no '
              'background figures anywhere.'
          )

        if beat_kind in ('intro', 'arrival'):
          shot_guidance = (
              'Choose a wide establishing first-person POV camera shot '
              'introducing the room, its key interactive props, and the exact '
              'present character(s).'
          )
          default_caption = f'Establishing Shot — {entry_loc.name}'
        elif beat_kind == 'npc_resolution':
          shot_guidance = (
              'Choose a dramatic medium or close-up comic-panel camera angle '
              f'focused on {beat_actor} performing the exact physical action '
              'and emotional reaction described in CURRENT HAPPENING.'
          )
          default_caption = f'{beat_actor} — {entry_loc.name}'
        else:
          shot_guidance = (
              'Choose a dynamic graphic-novel camera angle (close-up on hands '
              'interacting with a prop, over-the-shoulder two-shot, or tense '
              'medium confrontation shot) capturing the exact physical '
              'moment described in CURRENT HAPPENING.'
          )
          default_caption = f'Scene Panel — {entry_loc.name}'

        loc_prompt = (
            'Single-frame graphic novel illustration, crisp detailed ink '
            'linework with atmospheric lighting and color. Full-bleed single '
            'scene: no panel borders, no split screens, no speech bubbles, '
            f'no text or signage. {cast_clause} '
            f'Location: {entry_loc.name} — {entry_loc.description[:240]} '
            f'Current happening: {beat_text[:340]} '
            f'Visible props: {props_text}.'
        )
        loc_director_instruction = (
            'You are the Photography Director for an interactive graphic-novel '
            'adventure game. Your job is to direct a single 16:9 comic panel '
            'depicting the exact CURRENT HAPPENING below.\n'
            'STRICT CINEMATOGRAPHY RULES:\n'
            f'1. CAMERA & ACTION: {shot_guidance} Depict the specific physical '
            'action, object interaction, or confrontation in CURRENT HAPPENING '
            'rather than a generic static room.\n'
            '2. LITERAL CAMERA VIEW ONLY: Convert any auditory or off-screen '
            'metaphors (such as "rain hammers the roof whenever the overland '
            'train rumbles overhead") into purely visible interior surfaces '
            '(such as "solid enclosed damp brick arch ceiling and corrugated '
            'iron roof panels overhead"). NEVER mention unseen external '
            'objects like "train", "railway tracks", or "sky" when indoors, '
            'or the image model will cut a hole in the ceiling to draw them.\n'
            '3. STRICT HEADCOUNT & ERA: State the exact number of visible '
            'people at both the start and end of PROMPT. Depict ONLY the '
            'listed present character(s) in their exact described modern / '
            'setting-accurate clothing (never 1920s period suits unless the '
            'setting is 1920s). Explicitly forbid any crowd, patrons, or '
            'background extras.\n'
            '4. STYLE: Single-frame graphic novel illustration, bold ink '
            'linework and atmospheric color, no panel borders, no speech '
            'bubbles, no text.\n\n'
            f'WORLD SETTING: {setting_style}\n'
            f'LOCATION: {entry_loc.name} — {entry_loc.description[:280]}\n'
            f'INTERACTIVE PROPS IN ROOM: {props_text}\n'
            f'PRESENT CHARACTERS: {cast_clause}\n'
            f'RECENT PLAYER ACTION: {last_player_action}\n'
            f'CURRENT HAPPENING ({beat_kind} by {beat_actor}): '
            f'{beat_text[:420]}\n\n'
            'Respond in EXACTLY two lines:\n'
            'CAPTION: <concise 6-14 word camera angle & action caption>\n'
            'PROMPT: <70-100 word single-frame graphic novel image prompt>'
        )
        if self._client is None:
          self._complete_location_image(
              ledger=ledger,
              location_id=entry_loc_id,
              image_key=loc_key,
              title=entry_loc.name,
              description=entry_loc.description,
              prompt=loc_prompt,
              entry_id=entry_id,
              fallback_caption=default_caption,
          )
        else:
          self._executor.submit(
              self._complete_location_image,
              ledger=ledger,
              location_id=entry_loc_id,
              image_key=loc_key,
              title=entry_loc.name,
              description=entry_loc.description,
              prompt=loc_prompt,
              director_instruction=loc_director_instruction,
              entry_id=entry_id,
              fallback_caption=default_caption,
          )
    elif current_loc is not None and not current_loc.image_url:
      loc_slug = world_ledger.normalize_location_id(loc_id)[:48]
      loc_key = f'loc_{world_tag}_{loc_slug}'
      if _IMAGE_KEY_RE.fullmatch(loc_key):
        self._complete_location_image(
            ledger=ledger,
            location_id=loc_id,
            image_key=loc_key,
            title=current_loc.name,
            description=current_loc.description,
            prompt=f'{current_loc.name}: {current_loc.description[:280]}',
            fallback_caption=f'Establishing Shot — {current_loc.name}',
        )

  def shutdown(self) -> None:
    """Shuts down the background image worker pool."""
    self._executor.shutdown(wait=False, cancel_futures=True)


def _resolve_static_dir() -> str:
  """Resolves the static assets directory in both open-source and monorepo."""
  return os.path.abspath(story_state.resolve_asset_path('static'))


def _load_inline_app_script() -> tuple[bytes, str]:
  """Extracts the embedded client script and its CSP SHA-256 digest."""
  index_path = os.path.join(_resolve_static_dir(), 'index.html')
  if not os.path.isfile(index_path):
    return b'', ''
  with open(index_path, 'rb') as f:
    html_bytes = f.read()
  match = _INLINE_SCRIPT_PATTERN.search(html_bytes)
  if match is None:
    return b'', ''
  script_bytes = match.group(1)
  digest = base64.b64encode(hashlib.sha256(script_bytes).digest()).decode(
      'ascii'
  )
  return script_bytes, digest


class BrowserInputBridge(human_input.HumanInput):
  """Thread-safe queue bridging HTTP POST actions to `HumanActComponent`."""

  def __init__(self, timeout_seconds: float = 300.0):
    """Initializes the instance."""
    self._queue: queue.Queue[str] = queue.Queue()
    self._timeout_seconds = timeout_seconds
    self._pending_count = 0
    self._lock = threading.Lock()

  def submit_action(self, action_text: str) -> None:
    """Enqueues a validated player action from the web UI."""
    cleaned = action_text.strip()
    if not cleaned:
      raise ValueError('Action text must not be empty.')
    if len(cleaned) > _MAX_ACTION_LENGTH:
      raise ValueError(
          f'Action text exceeds maximum length of {_MAX_ACTION_LENGTH} chars.'
      )
    with self._lock:
      self._pending_count += 1
    self._queue.put(cleaned)

  def clear(self, wake_blocked_reader: bool = False) -> None:
    """Drains any queued actions and resets the pending counter."""
    with self._lock:
      old_queue = self._queue
      while not old_queue.empty():
        try:
          old_queue.get_nowait()
        except queue.Empty:
          break
      self._pending_count = 0
      if wake_blocked_reader:
        for _ in range(4):
          old_queue.put('')
        self._queue = queue.Queue()

  def bind_new_session(
      self,
  ) -> tuple[human_input.HumanInput, Callable[[], bool]]:
    """Retires the previous queue and returns isolated `(reader, ready_fn)`."""
    with self._lock:
      old_queue = self._queue
      while not old_queue.empty():
        try:
          old_queue.get_nowait()
        except queue.Empty:
          break
      for _ in range(4):
        old_queue.put('')
      session_queue: queue.Queue[str] = queue.Queue()
      self._queue = session_queue
      self._pending_count = 0

    def _session_ready() -> bool:
      with self._lock:
        if self._queue is not session_queue:
          return False
        return self._pending_count > 0

    bridge_ref = self

    class _SessionReader(human_input.HumanInput):
      """Reads human player actions from the active session queue."""

      def __call__(self, request: human_input.HumanInputRequest) -> str:
        del request
        with bridge_ref._lock:
          if bridge_ref._queue is not session_queue:
            return ''
        try:
          action = session_queue.get(timeout=bridge_ref._timeout_seconds)
          with bridge_ref._lock:
            if bridge_ref._queue is not session_queue:
              return ''
            bridge_ref._pending_count = max(0, bridge_ref._pending_count - 1)
          return action
        except queue.Empty:
          with bridge_ref._lock:
            if bridge_ref._queue is not session_queue:
              return ''
          return 'look around carefully'

    return _SessionReader(), _session_ready

  def has_pending_action(self) -> bool:
    """Returns True when at least one player action is waiting in the queue."""
    with self._lock:
      return self._pending_count > 0

  def __call__(self, request: human_input.HumanInputRequest) -> str:
    """Pops and returns the next queued player action for `HumanActComponent`."""
    del request
    with self._lock:
      active_queue = self._queue
    try:
      action = active_queue.get(timeout=self._timeout_seconds)
      with self._lock:
        if self._queue is not active_queue:
          return ''
        self._pending_count = max(0, self._pending_count - 1)
      return action
    except queue.Empty:
      return 'look around carefully'


class AdventureServer:
  """Threaded HTTP + SSE server exposing the live Adventure simulation."""

  def __init__(
      self,
      sim: adventure_prefab.AdventureSimulation,
      input_bridge: BrowserInputBridge,
      *,
      host: str = '127.0.0.1',
      port: int = 8765,
      reset_fn: Callable[..., adventure_prefab.AdventureSimulation] | None = (
          None
      ),
  ):
    """Initializes the instance.

    Args:
      sim: Running or ready AdventureSimulation instance.
      input_bridge: BrowserInputBridge wired to the player's HumanActComponent.
      host: Bind address ('127.0.0.1' for local tests, '0.0.0.0' for Cloudtop).
      port: TCP port to listen on (0 selects an ephemeral free port).
      reset_fn: Optional callback that stops the old simulation, starts a fresh
        one, and returns the new AdventureSimulation instance.
    """
    self._sim = sim
    self._bridge = input_bridge
    self._host = host
    self._reset_fn = reset_fn
    self._prose_mode = 'concise'
    self._sim.ledger.set_prose_mode(self._prose_mode)
    self._sse_clients_lock = threading.Lock()
    self._sse_queues: list[queue.Queue[str]] = []
    sim_model = getattr(self._sim, '_model', None)
    self._image_generator = SceneImageGenerator(
        client=_extract_genai_client(sim_model),
        text_model=_extract_director_text_model(sim_model),
    )
    self._image_generator.ensure_scene_images(self._sim.ledger)

    self._bus_subscription = self._sim.ledger.bus.matching('').subscribe(
        self._on_bus_message
    )
    if hasattr(self._sim.scene_resolver, 'set_on_turn_state_change'):
      self._sim.scene_resolver.set_on_turn_state_change(
          self._on_turn_state_change
      )
    handler_cls = self._make_handler_class()

    class _ReusableThreadingHTTPServer(http_server.ThreadingHTTPServer):
      allow_reuse_address = True

    self._httpd = _ReusableThreadingHTTPServer((host, port), handler_cls)
    self._httpd.daemon_threads = True
    self._server_thread: threading.Thread | None = None

  @property
  def server_port(self) -> int:
    """Returns the bound TCP port."""
    return int(self._httpd.server_address[1])

  def reset_simulation(
      self, new_bible: story_state.WorldBible | None = None
  ) -> dict[str, Any]:
    """Resets the live simulation to Tick 0 (optionally with `new_bible`)."""
    if self._reset_fn is None and new_bible is None:
      raise RuntimeError('Simulation reset is not configured on this server.')
    self._bus_subscription.dispose()
    self._sim.stop()
    self._bridge.clear(wake_blocked_reader=True)
    if self._reset_fn is not None:
      sig = inspect.signature(self._reset_fn)
      if sig.parameters:
        self._sim = self._reset_fn(new_bible)
      else:
        self._sim = self._reset_fn()
    else:
      session_reader, session_ready = self._bridge.bind_new_session()
      self._sim = adventure_prefab.AdventureSimulation(
          model=self._sim._model,  # pylint: disable=protected-access
          player_act_component=human_act_component.HumanActComponent(
              input_reader=session_reader
          ),
          player_ready_fn=session_ready,
          enable_async_npc_plan=False,
          bible=new_bible,
      )
    self._sim.ledger.set_prose_mode(self._prose_mode)
    self._image_generator.clear_cache()
    self._bus_subscription = self._sim.ledger.bus.matching('').subscribe(
        self._on_bus_message
    )
    if hasattr(self._sim.scene_resolver, 'set_on_turn_state_change'):
      self._sim.scene_resolver.set_on_turn_state_change(
          self._on_turn_state_change
      )
    state = self.build_state_payload()
    init_pkt = json.dumps({'type': 'init', 'state': state})
    with self._sse_clients_lock:
      active_queues = list(self._sse_queues)
    for client_q in active_queues:
      try:
        client_q.put_nowait(init_pkt)
      except queue.Full:
        pass
    return state

  def generate_world(
      self,
      *,
      preset: str = '',
      title: str = '',
      premise: str = '',
      tone: str = '',
      use_llm: bool = False,
  ) -> dict[str, Any]:
    """Loads a preset seed or generates a new WorldBible from a prompt."""
    clean_preset = preset.strip().lower()
    if clean_preset in ('bermondsey', 'glassharbor'):
      bible = story_state.load_shipped_bible(clean_preset)
    elif clean_preset in ('vampire_romance', 'crimson_court', 'st_judes'):
      bible = author_world.draft_world_bible(
          title=title.strip() or "The Night Shift at St. Jude's",
          premise=premise.strip(),
          tone_name='vampire_romance',
          model=(
              self._sim._model if use_llm else None  # pylint: disable=protected-access
          ),
      )
    elif clean_preset in (
        'dystopian_scifi',
        'sector_9',
        'neon_hab',
        'outpost_7',
    ):
      bible = author_world.draft_world_bible(
          title=title.strip() or 'The Sabotage at Outpost 7',
          premise=premise.strip(),
          tone_name='dystopian_scifi',
          model=(
              self._sim._model if use_llm else None  # pylint: disable=protected-access
          ),
      )
    else:
      effective_premise = premise.strip()
      if not effective_premise:
        raise ValueError('Provide a world "preset" or a non-empty "premise".')
      effective_title = title.strip() or 'The Unwritten Chronicle'
      effective_tone = tone.strip() or 'noir'
      bible = author_world.draft_world_bible(
          title=effective_title,
          premise=effective_premise,
          tone_name=effective_tone,
          model=(
              self._sim._model if use_llm else None  # pylint: disable=protected-access
          ),
      )
    return self.reset_simulation(new_bible=bible)

  def get_cached_image(self, image_key: str) -> tuple[str, bytes] | None:
    """Returns `(mime_type, bytes)` for `image_key` from the image cache."""
    return self._image_generator.get_image(image_key)

  def build_state_payload(self) -> dict[str, Any]:
    """Builds the full JSON state snapshot for `/api/state` and SSE updates."""
    self._image_generator.ensure_scene_images(self._sim.ledger)
    ledger_dict = self._sim.ledger.to_dict()
    latency_dict = self._sim.latency_tracker.summary()
    memories = self._sim.roster.collect_all_entity_memories(
        self._sim.player_entity
    )
    pending_player = self._bridge.has_pending_action()
    engine_busy = self._sim.is_turn_busy()
    next_acting = self._sim.next_acting_component
    npc_pending = next_acting.has_pending_npc_turn()
    player_in_flight = (
        next_acting.is_player_turn_in_flight()
        if hasattr(next_acting, 'is_player_turn_in_flight')
        else False
    )
    resolving_player = bool(pending_player or player_in_flight)
    if resolving_player:
      responding_actors = []
    elif hasattr(next_acting, 'responding_npc_names'):
      responding_actors = next_acting.responding_npc_names()
    else:
      responding_actors = []
    return {
        'ledger': ledger_dict,
        'latency': latency_dict,
        'memories': memories,
        'pending_player_actions': pending_player,
        'player_steps_completed': self._sim.player_steps_completed,
        'resolving_player': resolving_player,
        'responding_actors': responding_actors,
        'turn_busy': bool(pending_player or engine_busy or npc_pending),
    }

  def _on_turn_state_change(self) -> None:
    """Notifies connected SSE clients when turn/actor response status changes."""
    with self._sse_clients_lock:
      active_queues = list(self._sse_queues)
    if not active_queues:
      return
    pkt = json.dumps({'type': 'turn_state'})
    for client_q in active_queues:
      try:
        client_q.put_nowait(pkt)
      except queue.Full:
        pass

  def _on_bus_message(self, message: topics.Message) -> None:
    """Broadcasts TopicBus events to all connected SSE clients."""
    with self._sse_clients_lock:
      active_queues = list(self._sse_queues)
    if not active_queues:
      return
    event_packet = json.dumps({
        'type': 'bus_event',
        'message': message.to_dict(),
    })
    for client_q in active_queues:
      try:
        client_q.put_nowait(event_packet)
      except queue.Full:
        pass

  def start_background(self) -> int:
    """Starts the HTTP server on a background daemon thread and returns port."""
    self._server_thread = threading.Thread(
        target=self._httpd.serve_forever,
        name='adventure-http-server',
        daemon=True,
    )
    self._server_thread.start()
    return self.server_port

  def shutdown(self) -> None:
    """Stops the HTTP server and disposes the TopicBus subscription."""
    self._bus_subscription.dispose()
    self._image_generator.shutdown()
    if self._server_thread is not None:
      self._httpd.shutdown()
      self._server_thread.join(timeout=2.0)
    self._httpd.server_close()

  def build_logs_payload(self) -> dict[str, Any]:
    """Builds the structured simulation log dictionary for `/api/logs`."""
    return self._sim.build_structured_log().to_dict()

  def submit_player_action(self, action_text: str) -> None:
    """Submits a player action to the input bridge and ledger."""
    self._bridge.submit_action(action_text)
    self._sim.ledger.record_player_action(action_text)

  def set_prose_mode(self, mode: str) -> dict[str, Any]:
    """Sets the GM prose verbosity mode ('concise' or 'literary') and broadcasts."""
    applied = self._sim.ledger.set_prose_mode(mode)
    self._prose_mode = applied
    return self._broadcast_state_update()

  def _broadcast_state_update(self) -> dict[str, Any]:
    """Builds and broadcasts the current state to all SSE clients."""
    state = self.build_state_payload()
    pkt = json.dumps({'type': 'init', 'state': state})
    with self._sse_clients_lock:
      active_queues = list(self._sse_queues)
    for client_q in active_queues:
      try:
        client_q.put_nowait(pkt)
      except queue.Full:
        pass
    return state

  def init_player_voices(
      self,
      *,
      active_ranks: Mapping[str, int] | None = None,
      custom_voices: Sequence[
          Mapping[str, Any] | story_state.SubMindState
      ] = (),
  ) -> dict[str, Any]:
    """Configures the player's Parliament of Sub-Minds and broadcasts state."""
    self._sim.configure_player_parliament(
        active_ranks=active_ranks,
        custom_voices=custom_voices,
        use_parallel_llm=False,
    )
    return self._broadcast_state_update()

  def mint_player_voice(
      self,
      *,
      concept: str,
      name_hint: str = '',
      domain_hint: str = '',
      rank: int = 2,
  ) -> tuple[story_state.SubMindState, dict[str, Any]]:
    """Procedurally mints a custom Sub-Mind, refreshes options, and broadcasts."""
    sm = self._sim.mint_custom_sub_mind(
        concept=concept,
        name_hint=name_hint,
        domain_hint=domain_hint,
        rank=rank,
        origin='custom',
    )
    with self._sim.ledger.lock:
      self._sim.ledger.story.active_voice_options.clear()
      if self._sim.ledger.tick == 0:
        self._sim.ledger.story.voice_option_log.clear()
    self._sim.submind_council.convene_council(
        location_id=self._sim.ledger.player_location_id,
        use_parallel_llm=False,
    )
    return sm, self._broadcast_state_update()

  def register_sse_client(self) -> queue.Queue[str]:
    """Registers a new SSE client queue and returns it."""
    client_q: queue.Queue[str] = queue.Queue(maxsize=64)
    with self._sse_clients_lock:
      self._sse_queues.append(client_q)
    return client_q

  def unregister_sse_client(self, client_q: queue.Queue[str]) -> None:
    """Removes a disconnected SSE client queue."""
    with self._sse_clients_lock:
      if client_q in self._sse_queues:
        self._sse_queues.remove(client_q)

  def _make_handler_class(
      self,
  ) -> Callable[..., http_server.BaseHTTPRequestHandler]:
    """Creates a request handler class bound to this server instance."""
    outer = self

    class _Handler(http_server.BaseHTTPRequestHandler):
      """Request handler enforcing strict CSP, nosniff, and method allowlist."""

      def log_message(self, format_str: str, *args: Any) -> None:
        del format_str, args

      def _send_security_headers(self, content_type: str) -> None:
        """Writes standard security and Content-Security-Policy headers."""
        _, script_sha256 = _load_inline_app_script()
        script_src = (
            f"'self' 'sha256-{script_sha256}'" if script_sha256 else "'self'"
        )
        self.send_header('Content-Type', content_type)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header(
            'Permissions-Policy', 'camera=(), microphone=(), geolocation=()'
        )
        self.send_header('Cache-Control', 'no-store')
        self.send_header(
            'Content-Security-Policy',
            f"default-src 'self'; script-src {script_src}; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'; object-src 'none'; base-uri 'self'",
        )

      def _send_json(
          self, payload: dict[str, Any], status: int = http.HTTPStatus.OK
      ) -> None:
        """Serializes and writes a JSON response with security headers."""
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self._send_security_headers('application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def _serve_static(self, filename: str, content_type: str) -> None:
        """Serves an allowlisted static file with security headers."""
        safe_name = os.path.basename(filename)
        if safe_name not in _ALLOWED_STATIC_FILES:
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        static_root = _resolve_static_dir()
        full_path = os.path.normpath(os.path.join(static_root, safe_name))
        if not full_path.startswith(static_root + os.sep) or not os.path.isfile(
            full_path
        ):
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        with open(full_path, 'rb') as f:
          content = f.read()
        self.send_response(http.HTTPStatus.OK)
        self._send_security_headers(content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

      def _serve_app_bundle(self) -> None:
        """Serves the embedded client script from `index.html` as `/app.js`."""
        content, _ = _load_inline_app_script()
        if not content:
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        self.send_response(http.HTTPStatus.OK)
        self._send_security_headers('application/javascript; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

      def _serve_generated_image(self, image_key: str) -> None:
        """Serves a validated location or NPC portrait image from the cache."""
        safe_key = image_key.strip()
        if not _IMAGE_KEY_RE.fullmatch(safe_key):
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        cached = outer.get_cached_image(safe_key)
        if cached is None:
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        mime_type, content = cached
        if (
            mime_type not in _ALLOWED_IMAGE_MIMES
            or not content
            or len(content) > _MAX_IMAGE_BYTES
        ):
          self._send_json(
              {'error': 'Not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        self.send_response(http.HTTPStatus.OK)
        self._send_security_headers(mime_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

      def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handles GET routes for static assets, state, logs, and SSE."""
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
          self._serve_static('index.html', 'text/html; charset=utf-8')
          return
        if path == '/styles.css':
          self._serve_static('styles.css', 'text/css; charset=utf-8')
          return
        if path == '/app.js':
          self._serve_app_bundle()
          return
        if path.startswith('/api/image/'):
          self._serve_generated_image(path[len('/api/image/') :])
          return
        if path == '/api/state':
          self._send_json(outer.build_state_payload())
          return
        if path == '/api/logs':
          self._send_json(outer.build_logs_payload())
          return
        if path == '/api/events':
          self.send_response(http.HTTPStatus.OK)
          self._send_security_headers('text/event-stream; charset=utf-8')
          self.send_header('Connection', 'keep-alive')
          self.end_headers()
          client_q = outer.register_sse_client()
          try:
            init_pkt = json.dumps({
                'type': 'init',
                'state': outer.build_state_payload(),
            })
            self.wfile.write(f'data: {init_pkt}\n\n'.encode('utf-8'))
            self.wfile.flush()
            while True:
              try:
                pkt = client_q.get(timeout=15.0)
                parsed_pkt = json.loads(pkt)
                if isinstance(parsed_pkt, dict) and 'state' not in parsed_pkt:
                  parsed_pkt['state'] = outer.build_state_payload()
                  pkt = json.dumps(parsed_pkt)
                self.wfile.write(f'data: {pkt}\n\n'.encode('utf-8'))
                self.wfile.flush()
              except queue.Empty:
                self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
          except (BrokenPipeError, ConnectionResetError, OSError):
            pass
          finally:
            outer.unregister_sse_client(client_q)
          return

        self._send_json(
            {'error': 'Route not found'}, status=http.HTTPStatus.NOT_FOUND
        )

      def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handles POST routes for player actions, voice init, and resets."""
        path = self.path.split('?', 1)[0]
        if path == '/api/reset':
          try:
            new_state = outer.reset_simulation()
          except RuntimeError as err:
            self._send_json(
                {'error': str(err)}, status=http.HTTPStatus.BAD_REQUEST
            )
            return
          self._send_json({'status': 'reset', 'state': new_state})
          return
        if path not in (
            '/api/action',
            '/api/init_voices',
            '/api/mint_voice',
            '/api/generate_world',
            '/api/prose_mode',
        ):
          self._send_json(
              {'error': 'Route not found'}, status=http.HTTPStatus.NOT_FOUND
          )
          return
        try:
          length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
          length = 0
        if length <= 0 or length > 16384:
          self._send_json(
              {'error': 'Invalid request body size.'},
              status=http.HTTPStatus.BAD_REQUEST,
          )
          return
        raw_body = self.rfile.read(length).decode('utf-8', errors='replace')
        try:
          data = json.loads(raw_body)
        except json.JSONDecodeError:
          self._send_json(
              {'error': 'Malformed JSON payload.'},
              status=http.HTTPStatus.BAD_REQUEST,
          )
          return
        if not isinstance(data, dict):
          self._send_json(
              {'error': 'Payload must be a JSON object.'},
              status=http.HTTPStatus.BAD_REQUEST,
          )
          return

        if path == '/api/prose_mode':
          mode = str(
              data.get('prose_mode') or data.get('mode') or 'concise'
          ).strip()
          new_state = outer.set_prose_mode(mode)
          self._send_json({
              'status': 'updated',
              'prose_mode': (
                  new_state.get('ledger', {}).get('prose_mode', 'concise')
              ),
              'state': new_state,
          })
          return

        if path == '/api/generate_world':
          preset = str(data.get('preset') or '').strip()
          title = str(data.get('title') or '').strip()
          premise = str(data.get('premise') or data.get('prompt') or '').strip()
          tone = str(data.get('tone') or '').strip()
          use_llm = bool(data.get('use_llm') or data.get('useLlm') or False)
          try:
            new_state = outer.generate_world(
                preset=preset,
                title=title,
                premise=premise,
                tone=tone,
                use_llm=use_llm,
            )
          except ValueError as err:
            self._send_json(
                {'error': str(err)}, status=http.HTTPStatus.BAD_REQUEST
            )
            return
          self._send_json({'status': 'world_generated', 'state': new_state})
          return

        if path == '/api/init_voices':
          raw_ranks = (
              data.get('active_ranks')
              if 'active_ranks' in data
              else data.get('activeRanks')
          )
          active_ranks: dict[str, int] | None = None
          if isinstance(raw_ranks, Mapping):
            active_ranks = {}
            for k, v in raw_ranks.items():
              try:
                active_ranks[str(k)] = int(v)
              except (TypeError, ValueError):
                continue
          raw_custom = (
              data.get('custom_voices')
              if 'custom_voices' in data
              else data.get('customVoices')
          )
          custom_voices: list[Mapping[str, Any]] = []
          if isinstance(raw_custom, Sequence) and not isinstance(
              raw_custom, str
          ):
            for item in raw_custom:
              if isinstance(item, Mapping):
                custom_voices.append(item)
          new_state = outer.init_player_voices(
              active_ranks=active_ranks,
              custom_voices=custom_voices,
          )
          self._send_json({'status': 'initialized', 'state': new_state})
          return

        if path == '/api/mint_voice':
          concept = str(data.get('concept') or '').strip()
          if not concept:
            self._send_json(
                {'error': 'Field "concept" must be a non-empty string.'},
                status=http.HTTPStatus.BAD_REQUEST,
            )
            return
          name_hint = str(
              data.get('name_hint') or data.get('nameHint') or ''
          ).strip()
          domain_hint = str(
              data.get('domain_hint') or data.get('domainHint') or ''
          ).strip()
          try:
            rank = int(data.get('rank', 2))
          except (TypeError, ValueError):
            rank = 2
          minted, new_state = outer.mint_player_voice(
              concept=concept,
              name_hint=name_hint,
              domain_hint=domain_hint,
              rank=rank,
          )
          self._send_json({
              'status': 'minted',
              'voice': minted.to_dict(),
              'state': new_state,
          })
          return

        if not isinstance(data.get('action'), str):
          self._send_json(
              {'error': 'Field "action" must be a string.'},
              status=http.HTTPStatus.BAD_REQUEST,
          )
          return
        try:
          outer.submit_player_action(data['action'])
        except ValueError as err:
          self._send_json(
              {'error': str(err)}, status=http.HTTPStatus.BAD_REQUEST
          )
          return
        self._send_json(
            {'status': 'queued', 'state': outer.build_state_payload()}
        )

      def do_PUT(self) -> None:  # pylint: disable=invalid-name
        self._send_json(
            {'error': 'Method not allowed'},
            status=http.HTTPStatus.METHOD_NOT_ALLOWED,
        )

      def do_DELETE(self) -> None:  # pylint: disable=invalid-name
        self._send_json(
            {'error': 'Method not allowed'},
            status=http.HTTPStatus.METHOD_NOT_ALLOWED,
        )

    return _Handler
