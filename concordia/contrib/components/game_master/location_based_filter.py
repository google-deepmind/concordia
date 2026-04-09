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

"""Component for enforcing location-based partial observability."""

import re
from typing import cast

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component


class LocationBasedFilter(entity_component.ContextComponent):
  """Enforces information barriers based on agent locations.

  This component extracts location information from narratives and provides
  filtered views that only include information accessible to agents based on
  their current location. This creates structural partial observability for
  Theory of Mind testing.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      entity_names: list[str],
      pre_act_label: str = 'Location-Based Observability Rules',
  ):
    self._model = model
    self._entity_names = entity_names
    self._pre_act_label = pre_act_label
    self._entity_locations: dict[str, str] = {}
    self._location_to_entities: dict[str, set[str]] = {}

  def get_pre_act_value(self) -> str:
    if not self._entity_locations:
      return 'No location information available yet.'
    lines = []
    lines.append('**ENFORCED PARTIAL OBSERVABILITY RULES:**')
    lines.append('Agents can ONLY observe events at their current location.')
    lines.append('')
    lines.append('Current locations:')
    for entity, location in sorted(self._entity_locations.items()):
      location_mates = self._location_to_entities.get(location, set())
      other_agents = sorted(location_mates - {entity})
      if other_agents:
        lines.append(
            f'  - {entity} is at "{location}" (with: {", ".join(other_agents)})'
        )
      else:
        lines.append(f'  - {entity} is at "{location}" (alone)')
    return '\n'.join(lines)

  def get_pre_act_label(self) -> str:
    """Returns the label for pre-act display."""
    return self._pre_act_label

  def extract_locations_from_narrative(self, narrative: str) -> dict[str, str]:
    entity_locations = {}
    scene_pattern = r'---\s*SCENE:\s*([^(]+?)\s*\([^)]+\)\s*---\s*\*\*Present:\*\*\s*([^\n]+)'
    matches = re.finditer(scene_pattern, narrative, re.IGNORECASE)
    for match in matches:
      location = match.group(1).strip()
      present_str = match.group(2).strip()
      present_names = re.split(r',|\sand\s', present_str)
      for name in present_names:
        name = name.strip()
        for entity_name in self._entity_names:
          if (
              entity_name.lower() in name.lower()
              or name.lower() in entity_name.lower()
          ):
            entity_locations[entity_name] = location
            break
    return entity_locations

  def update_locations(self, entity_locations: dict[str, str]) -> None:
    self._entity_locations = entity_locations.copy()
    self._location_to_entities.clear()
    for entity, location in entity_locations.items():
      if location not in self._location_to_entities:
        self._location_to_entities[location] = set()
      self._location_to_entities[location].add(entity)

  def get_location(self, entity_name: str) -> str | None:
    return self._entity_locations.get(entity_name)

  def get_entities_at_location(self, location: str) -> set[str]:
    return self._location_to_entities.get(location, set()).copy()

  def can_observe(self, observer: str, target: str) -> bool:
    observer_loc = self._entity_locations.get(observer)
    target_loc = self._entity_locations.get(target)
    if observer_loc is None or target_loc is None:
      return False
    return observer_loc == target_loc

  def filter_narrative_for_entity(
      self, narrative: str, entity_name: str
  ) -> str:
    regex_filtered = self._regex_filter_narrative(narrative, entity_name)
    if len(regex_filtered) > 100:
      return regex_filtered
    return self._llm_filter_narrative(narrative, entity_name)

  def _regex_filter_narrative(self, narrative: str, entity_name: str) -> str:
    entity_location = self._entity_locations.get(entity_name)
    if not entity_location:
      return f'[No location information available for {entity_name}]'
    scene_splits = re.split(r'(---\s*SCENE:[^-]*---)', narrative)
    filtered_parts = []
    i = 0
    while i < len(scene_splits):
      if i == 0 and scene_splits[i].strip():
        filtered_parts.append(scene_splits[i])
        i += 1
        continue
      if '--- SCENE:' in scene_splits[i]:
        scene_header = scene_splits[i]
        scene_content = scene_splits[i + 1] if i + 1 < len(scene_splits) else ''
        present_match = re.search(
            r'\*\*Present:\*\*\s*([^\n]+)', scene_content, re.IGNORECASE
        )
        if present_match:
          present_str = present_match.group(1).strip()
          is_present = False
          present_names = re.split(r',|\sand\s', present_str)
          for name in present_names:
            name = name.strip()
            if (
                entity_name.lower() in name.lower()
                or name.lower() in entity_name.lower()
            ):
              is_present = True
              break
          if is_present:
            filtered_parts.append(scene_header)
            filtered_parts.append(scene_content)
        i += 2
      else:
        i += 1
    if len(filtered_parts) <= 1:
      return f'[{entity_name} did not witness any scenes during this period]'
    return ''.join(filtered_parts)

  def _llm_filter_narrative(self, narrative: str, entity_name: str) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Full Narrative:\n{narrative}')
    filtering_instructions = f"""You are a strict information filter enforcing partial observability.

Your task: Extract ONLY the parts of the narrative that {entity_name} could have DIRECTLY observed.

**CRITICAL RULES:**
1. {entity_name} can ONLY observe events at their physical location
2. If the narrative mentions {entity_name} is at Location A, they CANNOT see events at Location B
3. If {entity_name} is not mentioned in a scene, they were NOT there - exclude that scene entirely
4. Do NOT infer or assume {entity_name}'s presence - they must be explicitly mentioned
5. Keep the original narrative style, just remove unobservable content

**What to include:**
- Scenes explicitly stating {entity_name} is present
- Events {entity_name} directly participated in
- Things {entity_name} could see, hear, or experience at their location

**What to exclude:**
- Scenes where {entity_name} is not mentioned
- Events at different locations from {entity_name}
- Other characters' private thoughts or actions {entity_name} couldn't witness

**Output format:**
Return the filtered narrative preserving any scene headings where {entity_name} was present.
If {entity_name} was not present anywhere, return exactly: "[{entity_name} was not present in any scenes during this period]"

Begin filtering now. Return ONLY the filtered narrative.
"""
    estimated_output_tokens = min(4000, max(1000, len(narrative) * 2))
    filtered = prompt.open_question(
        filtering_instructions,
        max_tokens=estimated_output_tokens,
        terminators=(),
    )
    filtered = re.sub(
        r'<thinking>.*?</thinking>', '', filtered, flags=re.DOTALL
    ).strip()
    return filtered

  def post_act(self, action_attempt: str) -> str:
    if not action_attempt or not action_attempt.strip():
      return ''
    extracted_locations = self.extract_locations_from_narrative(action_attempt)
    if extracted_locations:
      self.update_locations(extracted_locations)
    return ''

  def get_state(self) -> entity_component.ComponentState:
    return {
        'entity_locations': self._entity_locations,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._entity_locations = cast(
        dict[str, str], state.get('entity_locations', {})
    )
    self._location_to_entities.clear()
    for entity, location in self._entity_locations.items():
      if location not in self._location_to_entities:
        self._location_to_entities[location] = set()
      self._location_to_entities[location].add(entity)
