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

"""Game Master component for controlled entity profile presentation in experiments."""

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from typing import Any, cast

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


DEFAULT_PRE_ACT_LABEL = 'Controlled Profile Presentations'


class ControlledProfilePresentation(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging,
):
  """Game Master component for controlling profile presentation in experiments.

  Decouples internal entity identities (e.g., 'Author-A', 'Candidate-1') from
  the profile attributes and display names (aliases) observed by participants.
  This enables controlled experimental interventions (such as blinded peer
  review, resume evaluation, status/prestige manipulation, or counterbalancing)
  without altering underlying entity implementations or leaking canonical
  identities.
  """

  def __init__(
      self,
      profiles: Mapping[str, Mapping[str, Any]],
      aliases: Mapping[str, str] | None = None,
      default_visible_fields: Sequence[str] = (),
      pre_act_label: str = DEFAULT_PRE_ACT_LABEL,
  ):
    """Initializes the component.

    Args:
      profiles: Canonical profile attributes for each entity.
      aliases: Optional mapping from internal entity names to display aliases
        (e.g., {'Author-A': 'Contributor 1'}). If None, entity names are used.
      default_visible_fields: Sequence of fields shown by default when no
        observer-specific condition is configured.
      pre_act_label: Label displayed in pre_act context.

    Raises:
      ValueError: If profiles are empty, aliases are invalid, or visible fields
        do not exist on profiles.
    """
    super().__init__()
    if not profiles:
      raise ValueError('Profiles mapping must not be empty.')

    self._profiles = copy.deepcopy(dict(profiles))

    if aliases is not None:
      if set(aliases) != set(self._profiles):
        raise ValueError(
            'Aliases must cover exactly the configured profile subjects.'
        )
      if any(not isinstance(a, str) or not a.strip() for a in aliases.values()):
        raise ValueError('Aliases must be non-empty strings.')
      if len(set(aliases.values())) != len(aliases):
        raise ValueError('Aliases must be unique across all subjects.')
      self._aliases = dict(aliases)
    else:
      self._aliases = {name: name for name in self._profiles}

    default_fields_tuple = tuple(default_visible_fields)
    if len(set(default_fields_tuple)) != len(default_fields_tuple):
      raise ValueError('default_visible_fields must not contain duplicates.')
    for subject, prof in self._profiles.items():
      missing = set(default_fields_tuple) - set(prof)
      if missing:
        raise ValueError(
            f'Field(s) {missing} missing in profile for {subject}.'
        )

    self._default_visible_fields = default_fields_tuple
    self._pre_act_label = pre_act_label
    self._observer_conditions: dict[str, dict[str, Any]] = {}
    self._audit_trail: list[dict[str, Any]] = []

  def set_condition_for_observer(
      self,
      observer: str,
      visible_fields: Sequence[str] | None = None,
      overrides: Mapping[str, Mapping[str, Any]] | None = None,
      swaps: Mapping[str, str] | None = None,
      use_aliases: bool = True,
  ) -> None:
    """Configures the experimental presentation condition for a specific observer.

    Args:
      observer: Name of the observing entity.
      visible_fields: Specific fields visible to this observer. If None,
        uses default_visible_fields. Empty sequence means fully blinded.
      overrides: Explicit field overrides per subject, e.g.
        {'Author-A': {'title': 'Junior Engineer'}}.
      swaps: Pairwise or cyclical profile swaps between subjects for
        counterbalancing, e.g. {'Author-A': 'Author-B', 'Author-B': 'Author-A'}.
        Takes precedence over overrides if both are specified.
      use_aliases: Whether to display participant aliases instead of internal
        entity names.

    Raises:
      ValueError: If overrides, swaps, or fields reference unknown subjects or
        fields not in the canonical profile.
    """
    fields = (
        self._default_visible_fields
        if visible_fields is None
        else tuple(visible_fields)
    )
    if len(set(fields)) != len(fields):
      raise ValueError('visible_fields must not contain duplicates.')

    for subject, prof in self._profiles.items():
      missing = set(fields) - set(prof)
      if missing:
        raise ValueError(
            f'Field(s) {missing} missing in canonical profile for {subject}.'
        )

    validated_overrides: dict[str, dict[str, Any]] = {}
    if overrides:
      for subj, changes in overrides.items():
        if subj not in self._profiles:
          raise ValueError(f'Override refers to unknown subject {subj}.')
        invalid_fields = set(changes) - set(self._profiles[subj])
        if invalid_fields:
          raise ValueError(
              f'Override field(s) {invalid_fields} not in canonical profile for'
              f' {subj}.'
          )
        hidden_overrides = set(changes) - set(fields)
        if hidden_overrides:
          raise ValueError(
              f'Cannot override hidden field(s) {hidden_overrides} for {subj}.'
          )
        validated_overrides[subj] = copy.deepcopy(dict(changes))

    validated_swaps: dict[str, str] = {}
    if swaps:
      for source_subj, target_subj in swaps.items():
        if source_subj not in self._profiles:
          raise ValueError(
              f'Swap source refers to unknown subject {source_subj}.'
          )
        if target_subj not in self._profiles:
          raise ValueError(
              f'Swap target refers to unknown subject {target_subj}.'
          )
        validated_swaps[source_subj] = target_subj

    self._observer_conditions[observer] = {
        'visible_fields': fields,
        'overrides': validated_overrides,
        'swaps': validated_swaps,
        'use_aliases': use_aliases,
    }

  def _resolve_subject_profile(
      self,
      observer: str,
      subject: str,
  ) -> tuple[str, dict[str, Any]]:
    """Resolves displayed name and visible attributes for a subject."""
    cond = self._observer_conditions.get(observer, {
        'visible_fields': self._default_visible_fields,
        'overrides': {},
        'swaps': {},
        'use_aliases': True,
    })

    display_name = self._aliases[subject] if cond['use_aliases'] else subject
    visible_fields = cond['visible_fields']
    swaps = cond.get('swaps', {})
    overrides = cond.get('overrides', {})

    if subject in swaps:
      source_subject = swaps[subject]
      base_profile = self._profiles[source_subject]
    else:
      base_profile = self._profiles[subject]

    subj_overrides = overrides.get(subject, {})
    selected_attrs = {}
    for f in visible_fields:
      val = subj_overrides.get(f, base_profile.get(f))
      selected_attrs[f] = copy.deepcopy(val)

    return display_name, selected_attrs

  def render_for(
      self,
      observer: str,
      subjects: Sequence[str] | None = None,
      format_style: str = 'natural',
  ) -> str:
    """Renders the safe profile presentation for the given observer.

    Args:
      observer: The observing entity.
      subjects: Optional subset and ordering of subjects to present.
        Defaults to all configured subjects in profile order.
      format_style: 'natural' (human-readable cards) or 'json' (structured
        JSON).

    Returns:
      Formatted profile presentation string safe for observer.

    Raises:
      ValueError: If subjects contain unknown entity names or format_style is
        unknown.
    """
    subjects_to_render = (
        tuple(self._profiles.keys()) if subjects is None else tuple(subjects)
    )
    for subj in subjects_to_render:
      if subj not in self._profiles:
        raise ValueError(f'Unknown subject {subj}.')

    resolved = []
    for subj in subjects_to_render:
      name, attrs = self._resolve_subject_profile(observer, subj)
      resolved.append((name, attrs))

    if format_style == 'natural':
      lines = ['Participant profiles:']
      for name, attrs in resolved:
        if attrs:
          attr_str = '; '.join(f'{k}: {v}' for k, v in attrs.items())
          lines.append(f'- {name}: {attr_str}')
        else:
          lines.append(f'- {name}')
      rendered_text = '\n'.join(lines)
    elif format_style == 'json':
      data = [
          {'participant': name, 'profile': attrs} for name, attrs in resolved
      ]
      rendered_text = json.dumps(data, sort_keys=True)
    else:
      raise ValueError(f'Unknown format_style: {format_style}')

    cond = self._observer_conditions.get(observer, {
        'visible_fields': self._default_visible_fields,
        'overrides': {},
        'swaps': {},
        'use_aliases': True,
    })

    audit_entry = {
        'observer': observer,
        'subjects': list(subjects_to_render),
        'displayed_names': [name for name, _ in resolved],
        'visible_fields': list(cond['visible_fields']),
        'overrides': copy.deepcopy(cond.get('overrides', {})),
        'swaps': copy.deepcopy(cond.get('swaps', {})),
        'format_style': format_style,
        'rendered_text': rendered_text,
        'sha256': hashlib.sha256(rendered_text.encode('utf-8')).hexdigest(),
    }
    self._audit_trail.append(audit_entry)
    self._logging_channel(audit_entry)

    return rendered_text

  def queue_presentation(
      self,
      observer: str,
      observation_component: Any,
      subjects: Sequence[str] | None = None,
      prefix: str = '',
      suffix: str = '',
      format_style: str = 'natural',
  ) -> str:
    """Renders and adds the safe profile presentation directly to an observation queue.

    Args:
      observer: The observing entity.
      observation_component: Component supporting `add_to_queue(entity, event)`
        (e.g., MakeObservation) or `add(entity, event, player_names)`.
      subjects: Optional subset/order of subjects to render.
      prefix: Optional text preceding the profile block.
      suffix: Optional text following the profile block.
      format_style: 'natural' or 'json'.

    Returns:
      The complete queued message string.
    """
    presentation = self.render_for(
        observer=observer,
        subjects=subjects,
        format_style=format_style,
    )
    parts = [p for p in (prefix.strip(), presentation, suffix.strip()) if p]
    event = '\n\n'.join(parts) if len(parts) > 1 else parts[0]

    if hasattr(observation_component, 'add_to_queue'):
      observation_component.add_to_queue(observer, event)
    elif hasattr(observation_component, 'add'):
      observation_component.add(
          observer, event, list(self._profiles.keys())
      )
    else:
      raise TypeError(
          'observation_component must implement `add_to_queue` or `add`.'
      )

    return event

  def get_audit_trail(self) -> Sequence[Mapping[str, Any]]:
    """Returns a copy of the immutable researcher audit trail."""
    return copy.deepcopy(self._audit_trail)

  def verify_field_not_disclosed(
      self,
      observer: str,
      forbidden_field: str,
  ) -> bool:
    """Verifies that a forbidden profile field was never presented to an observer.

    Args:
      observer: Name of the observer.
      forbidden_field: Name of the attribute that must have remained hidden.

    Returns:
      True if the field was never visible in any presentation rendered for the
      observer.
    """
    for entry in self._audit_trail:
      if entry['observer'] == observer:
        if forbidden_field in entry['visible_fields']:
          return False
    return True

  def get_pre_act_value(self) -> str:
    lines = ['**CONTROLLED PROFILE PRESENTATION STATUS:**']
    lines.append(f'Total subjects: {len(self._profiles)}')
    lines.append(f'Configured observers: {len(self._observer_conditions)}')
    lines.append(f'Presentations generated: {len(self._audit_trail)}')
    return '\n'.join(lines)

  def get_pre_act_label(self) -> str:
    return self._pre_act_label

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    del action_spec
    return self.get_pre_act_value()

  def get_state(self) -> entity_component.ComponentState:
    return {
        'observer_conditions': copy.deepcopy(self._observer_conditions),
        'audit_trail': copy.deepcopy(self._audit_trail),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._observer_conditions = copy.deepcopy(
        cast(dict[str, dict[str, Any]], state.get('observer_conditions', {}))
    )
    self._audit_trail = copy.deepcopy(
        cast(list[dict[str, Any]], state.get('audit_trail', []))
    )
