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

"""Visualization tool for Concordia simulation configurations.

This module generates HTML/SVG visualizations of simulation configs,
showing entities as rectangles containing their component names.
Features a layout with left, center, and right panels.
"""

import json
from typing import Any

from concordia.typing import prefab as prefab_lib


# Color schemes for different roles
_COLORS = {
    prefab_lib.Role.ENTITY: {
        "header": "#3B82F6",  # Blue
        "body": "#DBEAFE",
        "border": "#1D4ED8",
        "text": "#1E3A8A",
    },
    prefab_lib.Role.GAME_MASTER: {
        "header": "#10B981",  # Green
        "body": "#D1FAE5",
        "border": "#047857",
        "text": "#064E3B",
    },
    prefab_lib.Role.INITIALIZER: {
        "header": "#F59E0B",  # Orange
        "body": "#FEF3C7",
        "border": "#B45309",
        "text": "#78350F",
    },
}

_ENTITY_WIDTH = 340
_ENTITY_MIN_HEIGHT = 120
_COMPONENT_HEIGHT = 24
_COMPONENT_MARGIN = 4
_HEADER_HEIGHT = 50
_PADDING = 20
_GRID_GAP = 30
_SECTION_GAP = 50
_SECTION_HEADER_HEIGHT = 40

_TRUNCATE_VALUE_CHARS = 40
_TRUNCATE_DISPLAY_CHARS = 35


def _get_entity_name(instance: prefab_lib.InstanceConfig) -> str:
  """Extract entity name from instance config."""
  name = instance.params.get("name", "Unnamed")
  return str(name) if name else "Unnamed"


def _format_param_value(value) -> str:
  """Format a parameter value as a short string for display."""
  try:
    value_str = str(value)
    # Truncate long values
    if len(value_str) > _TRUNCATE_VALUE_CHARS:
      value_str = value_str[: _TRUNCATE_VALUE_CHARS - 3] + "..."
    return value_str
  except (TypeError, ValueError):
    return "<complex>"


def _format_param_value_full(value) -> str:
  """Format a parameter value as a full string for the detail panel."""
  try:
    return str(value)
  except (TypeError, ValueError):
    return "<complex>"


def _get_component_items(
    instance: prefab_lib.InstanceConfig,
) -> list[tuple[str, str]]:
  """Extract component (param key, value) pairs from instance config.

  Args:
    instance: The instance configuration.

  Returns:
    List of (key, formatted_value) tuples, excluding 'name' param.
  """
  items = []
  for k, v in instance.params.items():
    if k != "name":
      items.append((k, _format_param_value(v)))
  return items


def _calculate_entity_height(num_components: int) -> int:
  """Calculate the height of an entity box based on number of components."""
  content_height = num_components * (_COMPONENT_HEIGHT + _COMPONENT_MARGIN)
  return max(_ENTITY_MIN_HEIGHT, _HEADER_HEIGHT + content_height + _PADDING)


def _render_entity_svg(
    instance: prefab_lib.InstanceConfig,
    x: int,
    y: int,
    entity_id: str,
) -> tuple[str, int]:
  """Render a single entity as SVG elements.

  Args:
    instance: The instance configuration to render.
    x: X coordinate for the entity box.
    y: Y coordinate for the entity box.
    entity_id: Unique identifier for click handling.

  Returns:
    Tuple of (svg_string, height of this entity).
  """
  colors = _COLORS[instance.role]
  name = _get_entity_name(instance)
  prefab_name = instance.prefab
  component_items = _get_component_items(instance)

  # Calculate height including action area
  content_height = len(component_items) * (
      _COMPONENT_HEIGHT + _COMPONENT_MARGIN
  )
  action_area_height = 230
  height = max(
      _ENTITY_MIN_HEIGHT,
      _HEADER_HEIGHT + content_height + _PADDING + action_area_height,
  )

  svg_parts = []

  # Clickable group wrapper
  svg_parts.append(
      f'<g class="entity-card" data-entity-id="{entity_id}" '
      f'data-entity-name="{_escape_svg(name)}" '
      'style="cursor: pointer;">'
  )

  # Main rectangle with rounded corners
  svg_parts.append(
      f'<rect x="{x}" y="{y}" width="{_ENTITY_WIDTH}" height="{height}" '
      f'rx="8" ry="8" fill="{colors["body"]}" stroke="{colors["border"]}" '
      'stroke-width="2" class="entity-bg"/>'
  )

  # Header background
  svg_parts.append(
      f'<rect x="{x}" y="{y}" width="{_ENTITY_WIDTH}" height="{_HEADER_HEIGHT}"'
      f' rx="8" ry="8" fill="{colors["header"]}"/>'
  )
  # Square off bottom corners of header
  svg_parts.append(
      f'<rect x="{x}" y="{y + _HEADER_HEIGHT - 8}" width="{_ENTITY_WIDTH}" '
      f'height="8" fill="{colors["header"]}"/>'
  )

  # Entity name
  svg_parts.append(
      f'<text x="{x + _ENTITY_WIDTH // 2}" y="{y + 22}" text-anchor="middle"'
      ' font-family="Arial, sans-serif" font-size="14" font-weight="bold"'
      f' fill="white">{_escape_svg(name)}</text>'
  )

  # Prefab type (smaller, below name)
  svg_parts.append(
      f'<text x="{x + _ENTITY_WIDTH // 2}" y="{y + 40}" text-anchor="middle"'
      ' font-family="Arial, sans-serif" font-size="10"'
      f' fill="rgba(255,255,255,0.8)">{_escape_svg(prefab_name)}</text>'
  )

  # Components (param name: value pairs)
  comp_y = y + _HEADER_HEIGHT + _COMPONENT_MARGIN
  for param_name, param_value in component_items:
    # Component rectangle
    svg_parts.append(
        f'<rect x="{x + 10}" y="{comp_y}" '
        f'width="{_ENTITY_WIDTH - 20}" height="{_COMPONENT_HEIGHT}" '
        f'rx="4" ry="4" fill="white" stroke="{colors["border"]}" '
        'stroke-width="1" opacity="0.7"/>'
    )
    # Param name (bold) and value (regular) - using tspan for different styles
    display_name = param_name
    display_value = param_value
    # Truncate if too long
    total_len = len(display_name) + 2 + len(display_value)
    if total_len > _TRUNCATE_DISPLAY_CHARS:
      available = _TRUNCATE_DISPLAY_CHARS - len(display_name) - 5
      if available > 0:
        display_value = display_value[:available] + "..."
      else:
        display_value = "..."

    svg_parts.append(
        f'<text x="{x + 20}" y="{comp_y + 16}" font-family="Arial, sans-serif"'
        f' font-size="11" fill="{colors["text"]}"><tspan'
        f' font-weight="bold">{_escape_svg(display_name)}</tspan>:'
        f" {_escape_svg(display_value)}</text>"
    )
    comp_y += _COMPONENT_HEIGHT + _COMPONENT_MARGIN

  # Action display section (updated dynamically via SSE)
  action_id = name.replace(" ", "_").replace("'", "_")
  action_id = "".join(c for c in action_id if c.isalnum() or c == "_")
  action_y = comp_y + 10

  svg_parts.append(
      f'<text x="{x + 10}" y="{action_y + 12}" font-family="Arial, sans-serif"'
      ' font-size="10" font-weight="bold" fill="#888">Latest Action:</text>'
  )
  svg_parts.append(
      f'<foreignObject x="{x + 10}" y="{action_y + 18}" '
      f'width="{_ENTITY_WIDTH - 20}" height="200">'
      f'<div xmlns="http://www.w3.org/1999/xhtml" id="action_{action_id}" '
      'style="font-family: Arial, sans-serif; font-size: 11px; color: #ccc; '
      "font-style: italic; padding: 6px; background: rgba(0,0,0,0.4); "
      "border-radius: 6px; word-wrap: break-word; max-height: 190px; "
      'overflow-y: auto;">(waiting...)</div>'
      "</foreignObject>"
  )

  badge_id = f"gm_badge_{action_id}"
  svg_parts.append(
      f'<g id="{badge_id}" style="display: none;" class="active-gm-badge">'
      f'<rect x="{x + _ENTITY_WIDTH - 64}" y="{y + 4}" width="56" '
      'height="18" rx="9" ry="9" fill="#4CAF50" opacity="0.9"/>'
      f'<text x="{x + _ENTITY_WIDTH - 36}" y="{y + 16}" '
      'text-anchor="middle" font-family="Arial, sans-serif" '
      'font-size="9" font-weight="bold" fill="white">ACTIVE</text>'
      "</g>"
  )

  svg_parts.append("</g>")  # Close clickable group

  return "\n".join(svg_parts), height


def _escape_svg(text: str) -> str:
  """Escape text for SVG."""
  return (
      str(text)
      .replace("&", "&amp;")
      .replace("<", "&lt;")
      .replace(">", "&gt;")
      .replace('"', "&quot;")
  )


def _render_section_header(title: str, x: int, y: int, width: int) -> str:
  """Render a section header."""
  return f"""
<rect x="{x}" y="{y}" width="{width}" height="{_SECTION_HEADER_HEIGHT}"
  rx="4" ry="4" fill="#F3F4F6" stroke="#D1D5DB" stroke-width="1"/>
<text x="{x + 15}" y="{y + 26}" font-family="Arial, sans-serif"
  font-size="16" font-weight="bold" fill="#374151">{_escape_svg(title)}</text>
"""


def _build_entity_data(
    config: prefab_lib.Config,
    checkpoint_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
  """Build a dictionary of entity data for JavaScript access.

  Args:
    config: The simulation config.
    checkpoint_data: Optional checkpoint data from simulation with
      component_info.

  Returns:
    Dict mapping entity IDs to their data for the Inspector panel.
  """
  entity_data = {}
  entity_id = 0

  # Build lookup for checkpoint data by entity name
  checkpoint_entities = {}
  checkpoint_gms = {}
  if checkpoint_data:
    checkpoint_entities = checkpoint_data.get("entities", {})
    checkpoint_gms = checkpoint_data.get("game_masters", {})

  for instance in config.instances:
    eid = f"entity_{entity_id}"
    entity_name = _get_entity_name(instance)

    # Get component_info from checkpoint if available
    component_info = None
    if instance.role == prefab_lib.Role.ENTITY:
      if entity_name in checkpoint_entities:
        component_info = checkpoint_entities[entity_name].get("component_info")
    else:
      if entity_name in checkpoint_gms:
        component_info = checkpoint_gms[entity_name].get("component_info")

    entity_data[eid] = {
        "name": entity_name,
        "prefab": instance.prefab,
        "role": instance.role.value,
        "params": [
            {"key": k, "value": _format_param_value_full(v)}
            for k, v in instance.params.items()
        ],
        "component_info": component_info,
    }
    entity_id += 1

  return entity_data


def visualize_config(
    config: prefab_lib.Config,
    checkpoint_data: dict[str, Any] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
  """Generate an SVG visualization of the simulation config.

  Args:
    config: The simulation configuration to visualize.
    checkpoint_data: Optional checkpoint data from simulation to enrich entity
      info with component class names.

  Returns:
    Tuple of (SVG string, entity data dict for JavaScript).
  """
  # Group instances by role
  entities = [i for i in config.instances if i.role == prefab_lib.Role.ENTITY]
  game_masters = [
      i for i in config.instances if i.role == prefab_lib.Role.GAME_MASTER
  ]
  initializers = [
      i for i in config.instances if i.role == prefab_lib.Role.INITIALIZER
  ]

  svg_parts = []
  current_y = _PADDING
  entity_id_counter = [0]  # Use list for nonlocal mutation

  # Calculate grid width
  max_cols = 4
  grid_width = max_cols * (_ENTITY_WIDTH + _GRID_GAP) - _GRID_GAP + 2 * _PADDING

  def render_group(
      instances: list[prefab_lib.InstanceConfig],
      title: str,
      start_y: int,
  ) -> int:
    """Render a group of instances and return the new y position."""
    if not instances:
      return start_y

    nonlocal svg_parts

    # Section header
    svg_parts.append(
        _render_section_header(
            f"{title} ({len(instances)})",
            _PADDING,
            start_y,
            grid_width - 2 * _PADDING,
        )
    )
    start_y += _SECTION_HEADER_HEIGHT + 15

    # Layout instances in a grid
    col = 0
    row_y = start_y
    row_max_height = 0

    for instance in instances:
      x = _PADDING + col * (_ENTITY_WIDTH + _GRID_GAP)
      entity_id = f"entity_{entity_id_counter[0]}"
      entity_id_counter[0] += 1
      entity_svg, height = _render_entity_svg(instance, x, row_y, entity_id)
      svg_parts.append(entity_svg)
      row_max_height = max(row_max_height, height)

      col += 1
      if col >= max_cols:
        col = 0
        row_y += row_max_height + _GRID_GAP
        row_max_height = 0

    # Move to next section
    if col > 0:  # Incomplete row
      row_y += row_max_height

    return row_y + _SECTION_GAP

  # Render each group
  current_y = render_group(entities, "ENTITIES (Agents)", current_y)
  current_y = render_group(game_masters, "GAME MASTERS", current_y)
  current_y = render_group(initializers, "INITIALIZERS", current_y)

  # Build entity data for JavaScript
  entity_data = _build_entity_data(config, checkpoint_data)

  # Combine into final SVG
  total_height = current_y
  svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 {grid_width} {total_height}"
  width="100%" id="config-svg"
  preserveAspectRatio="xMidYMin meet">
  <rect width="100%" height="100%" fill="#2D2D2D"/>
  {"".join(svg_parts)}
</svg>"""

  return svg, entity_data


def visualize_config_to_html(
    config: prefab_lib.Config,
    title: str = "Simulation Configuration",
    checkpoint_data: dict[str, Any] | None = None,
) -> str:
  """Generate a complete HTML page with the SVG visualization.

  Features Unity-style layout with left, center, right, and bottom panels.
  Clicking on entities shows their full details in the right sidebar.

  Args:
    config: The simulation configuration to visualize.
    title: Title for the HTML page.
    checkpoint_data: Optional checkpoint data from simulation to show component
      class names in the Inspector.

  Returns:
    Complete HTML page as a string.
  """
  svg, entity_data = visualize_config(config, checkpoint_data)
  entity_data_json = json.dumps(entity_data)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape_svg(title)}</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1E1E1E;
      color: #E0E0E0;
      margin: 0;
      padding: 0;
      height: 100vh;
      overflow: hidden;
    }}

    /* Unity-style layout */
    .layout {{
      display: grid;
      grid-template-columns: 250px 1fr 350px;
      grid-template-rows: auto 1fr 180px;
      height: 100vh;
      gap: 2px;
    }}

    .header {{
      grid-column: 1 / -1;
      background: #323232;
      padding: 10px 20px;
      border-bottom: 1px solid #3C3C3C;
      display: flex;
      align-items: center;
      gap: 20px;
    }}

    .header h1 {{
      margin: 0;
      font-size: 16px;
      color: #E0E0E0;
    }}

    /* Left sidebar - reserved for future use */
    .left-sidebar {{
      background: #2D2D2D;
      border-right: 1px solid #3C3C3C;
      padding: 10px;
      overflow-y: auto;
    }}

    .sidebar-title {{
      font-size: 12px;
      font-weight: bold;
      color: #888;
      text-transform: uppercase;
      margin-bottom: 10px;
      padding-bottom: 5px;
      border-bottom: 1px solid #3C3C3C;
    }}

    /* Center panel - main SVG view */
    .center-panel {{
      background: #2D2D2D;
      overflow: auto;
      padding: 10px;
    }}

    .svg-container {{
      background: #252525;
      border-radius: 4px;
      min-height: 100%;
    }}

    /* Right sidebar - Inspector panel */
    .right-sidebar {{
      background: #2D2D2D;
      border-left: 1px solid #3C3C3C;
      padding: 10px;
      overflow-y: auto;
    }}

    .inspector-header {{
      background: #3C3C3C;
      margin: -10px -10px 10px -10px;
      padding: 10px;
      border-bottom: 1px solid #4C4C4C;
    }}

    .inspector-title {{
      font-size: 14px;
      font-weight: bold;
      color: #E0E0E0;
    }}

    .inspector-subtitle {{
      font-size: 11px;
      color: #888;
      margin-top: 2px;
    }}

    .inspector-empty {{
      color: #666;
      font-style: italic;
      padding: 20px;
      text-align: center;
    }}

    .component-section {{
      background: #383838;
      border-radius: 4px;
      margin-bottom: 8px;
      overflow: hidden;
    }}

    .component-header {{
      background: #404040;
      padding: 8px 10px;
      font-size: 12px;
      font-weight: bold;
      color: #E0E0E0;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .component-header:hover {{
      background: #4A4A4A;
    }}

    .component-body {{
      padding: 8px 10px;
    }}

    .param-row {{
      display: flex;
      padding: 4px 0;
      border-bottom: 1px solid #2D2D2D;
      font-size: 11px;
    }}

    .param-row:last-child {{
      border-bottom: none;
    }}

    .param-name {{
      font-weight: bold;
      color: #7CB7FF;
      min-width: 100px;
      flex-shrink: 0;
    }}

    .param-value {{
      color: #C0C0C0;
      word-break: break-word;
      max-height: 100px;
      overflow-y: auto;
    }}

    /* Expandable component state styles */
    .component-item {{
      margin-bottom: 6px;
      background: #2D2D2D;
      border-radius: 3px;
    }}

    .component-item-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 8px;
      cursor: pointer;
      font-size: 11px;
    }}

    .component-item-header:hover {{
      background: #404040;
    }}

    .component-item-name {{
      color: #7CB7FF;
      font-weight: bold;
    }}

    .component-item-class {{
      color: #888;
      font-size: 10px;
    }}

    .component-item-toggle {{
      color: #666;
      font-size: 10px;
    }}

    .component-state {{
      display: none;
      padding: 6px 8px;
      background: #252525;
      border-top: 1px solid #3C3C3C;
      font-size: 10px;
    }}

    .component-state.expanded {{
      display: block;
    }}

    .state-row {{
      display: flex;
      padding: 3px 0;
      border-bottom: 1px solid #333;
    }}

    .state-row:last-child {{
      border-bottom: none;
    }}

    .state-key {{
      color: #9CDCFE;
      min-width: 80px;
      flex-shrink: 0;
    }}

    .state-value {{
      color: #CE9178;
      word-break: break-all;
      max-height: 60px;
      overflow-y: auto;
    }}

    .state-empty {{
      color: #666;
      font-style: italic;
    }}

    /* Dynamic state editing styles */
    .dynamic-badge {{
      display: inline-block;
      background: #2d6a3e;
      color: #4EC9B0;
      font-size: 8px;
      padding: 1px 4px;
      border-radius: 3px;
      margin-left: 4px;
      vertical-align: middle;
      font-weight: bold;
      letter-spacing: 0.5px;
    }}

    .dynamic-input {{
      background: #1E1E1E;
      color: #CE9178;
      border: 1px solid #4EC9B0;
      border-radius: 2px;
      padding: 2px 4px;
      font-size: 10px;
      font-family: 'Consolas', 'Courier New', monospace;
      width: 100%;
      box-sizing: border-box;
      margin-top: 2px;
    }}

    .dynamic-input:focus {{
      outline: none;
      border-color: #569CD6;
      box-shadow: 0 0 3px rgba(86, 156, 214, 0.3);
    }}

    .dynamic-save-btn {{
      background: #2d6a3e;
      color: #4EC9B0;
      border: none;
      border-radius: 2px;
      padding: 2px 6px;
      font-size: 9px;
      cursor: pointer;
      margin-left: 4px;
      margin-top: 2px;
    }}

    .dynamic-save-btn:hover {{
      background: #3d8a5e;
    }}

    .dynamic-save-btn:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}

    .dynamic-row {{
      display: flex;
      flex-direction: column;
      padding: 4px 0;
      border-bottom: 1px solid #333;
    }}

    .dynamic-row-header {{
      display: flex;
      align-items: center;
    }}

    .dynamic-row-editor {{
      display: flex;
      align-items: center;
      margin-top: 2px;
    }}

    /* Bottom panel - console/logs */
    .bottom-panel {{
      grid-column: 1 / -1;
      background: #1E1E1E;
      border-top: 1px solid #3C3C3C;
      padding: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .console-header {{
      background: #2D2D2D;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: bold;
      color: #A0A0A0;
      border-bottom: 1px solid #3C3C3C;
      flex-shrink: 0;
    }}

    .console-output {{
      flex: 1;
      overflow-y: auto;
      padding: 8px 10px;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      font-size: 12px;
      line-height: 1.4;
      color: #D4D4D4;
    }}

    .console-line {{
      margin: 2px 0;
    }}

    .console-line .timestamp {{
      color: #6A9955;
    }}

    .console-line.info {{
      color: #3B82F6;
    }}

    .console-line.success {{
      color: #4EC9B0;
    }}

    .console-line.warning {{
      color: #DCDCAA;
    }}

    .console-line.error {{
      color: #F14C4C;
    }}

    /* Premise display */
    .premise-box {{
      background: #3D3520;
      border: 1px solid #5C5030;
      border-radius: 4px;
      padding: 10px;
      margin-bottom: 10px;
    }}

    .premise-label {{
      font-size: 11px;
      font-weight: bold;
      color: #C0A040;
      margin-bottom: 4px;
    }}

    .premise-text {{
      font-size: 12px;
      color: #E0D0A0;
    }}

    /* Legend */
    .legend {{
      display: flex;
      gap: 15px;
      font-size: 12px;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}

    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
    }}

    /* Entity hover effect */
    .entity-card:hover .entity-bg {{
      filter: brightness(1.1);
    }}

    .entity-card.selected .entity-bg {{
      stroke-width: 4;
      filter: brightness(1.15);
    }}

    /* Placeholder content */
    .placeholder {{
      color: #555;
      font-size: 12px;
      font-style: italic;
      padding: 10px;
    }}

    /* Simulation Controls */
    .sim-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: auto;
    }}

    .sim-controls button {{
      background: #4A4A4A;
      border: 1px solid #5A5A5A;
      color: #E0E0E0;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: background 0.2s;
    }}

    .sim-controls button:hover {{
      background: #5A5A5A;
    }}

    .sim-controls button:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}

    .sim-controls button.active {{
      background: #3B82F6;
      border-color: #5BA3F6;
    }}

    .step-counter {{
      background: #3C3C3C;
      padding: 6px 12px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 13px;
    }}

    .status-indicator {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #666;
    }}

    .status-indicator.running {{
      background: #10B981;
      animation: pulse 1s infinite;
    }}

    .status-indicator.paused {{
      background: #F59E0B;
    }}

    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.5; }}
    }}

    /* Entity action overlay */
    .entity-action {{
      position: absolute;
      bottom: -20px;
      left: 0;
      right: 0;
      font-size: 10px;
      color: #A0A0A0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <!-- Header -->
    <div class="header">
      <h1>{_escape_svg(title)}</h1>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-color" style="background: #3B82F6;"></div>
          <span>Entity</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background: #10B981;"></div>
          <span>Game Master</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background: #F59E0B;"></div>
          <span>Initializer</span>
        </div>
      </div>
      <div class="sim-controls">
        <div class="status-indicator" id="status-indicator"></div>
        <button id="btn-play" onclick="simPlay()" title="Play">▶</button>
        <button id="btn-pause" onclick="simPause()" title="Pause">⏸</button>
        <button id="btn-step" onclick="simStep()" title="Step one timestep">1▶</button>
        <div class="step-counter">Step: <span id="step-counter">0</span></div>
      </div>
    </div>

    <!-- Left Sidebar (reserved for hierarchy/scene view) -->
    <div class="left-sidebar">
      <div class="sidebar-title">Hierarchy</div>
      <div class="premise-box">
        <div class="premise-label">Default Premise</div>
        <div class="premise-text">{_escape_svg(config.default_premise or 'No premise specified')}</div>
      </div>
      <div class="placeholder">More content will appear here in the future</div>
    </div>

    <!-- Center Panel (main view) -->
    <div class="center-panel">
      <div class="svg-container">
        {svg}
      </div>
    </div>

    <!-- Right Sidebar (Inspector) -->
    <div class="right-sidebar" id="inspector">
      <div class="inspector-header">
        <div class="inspector-title" id="inspector-title">Inspector</div>
        <div class="inspector-subtitle" id="inspector-subtitle">Select an entity</div>
      </div>
      <div class="inspector-empty" id="inspector-empty">
        Click on an entity to view its details
      </div>
      <div id="inspector-content" style="display: none;"></div>
    </div>

    <!-- Bottom Panel - Console -->
    <div class="bottom-panel">
      <div class="console-header">Console</div>
      <div class="console-output" id="console-output">
        <div class="console-line info"><span class="timestamp">[--:--:--]</span> Ready. Select an entity to inspect its components.</div>
      </div>
    </div>
  </div>

  <script>
    // Entity data from Python
    const entityData = {entity_data_json};

    let selectedEntity = null;

    // Add click handlers to all entity cards
    document.querySelectorAll('.entity-card').forEach(card => {{
      card.addEventListener('click', function(e) {{
        const entityId = this.getAttribute('data-entity-id');
        selectEntity(entityId);
      }});
    }});

    function selectEntity(entityId) {{
      logConsole('Selected entity: ' + entityId, 'info');

      // Remove previous selection
      if (selectedEntity) {{
        const prev = document.querySelector(`[data-entity-id="${{selectedEntity}}"]`);
        if (prev) prev.classList.remove('selected');
      }}

      // Add new selection
      selectedEntity = entityId;
      const card = document.querySelector(`[data-entity-id="${{entityId}}"]`);
      if (card) card.classList.add('selected');

      // Update inspector
      updateInspector(entityId);
    }}

    function updateInspector(entityId) {{
      const data = entityData[entityId];
      if (!data) return;

      document.getElementById('inspector-title').textContent = data.name;
      document.getElementById('inspector-subtitle').textContent = data.prefab;
      document.getElementById('inspector-empty').style.display = 'none';

      const content = document.getElementById('inspector-content');
      content.style.display = 'block';

      // Build component sections
      let html = '';

      // Component Info section (if available from checkpoint)
      if (data.component_info) {{
        html += '<div class="component-section">';
        html += '<div class="component-header">▼ Components</div>';
        html += '<div class="component-body">';

        // Act component
        if (data.component_info.act_component) {{
          html += '<div class="param-row">';
          html += '<span class="param-name">Act Component</span>';
          html += `<span class="param-value">${{escapeHtml(data.component_info.act_component.class_name)}}</span>`;
          html += '</div>';
        }}

        // Context processor
        if (data.component_info.context_processor) {{
          html += '<div class="param-row">';
          html += '<span class="param-name">Context Processor</span>';
          html += `<span class="param-value">${{escapeHtml(data.component_info.context_processor.class_name)}}</span>`;
          html += '</div>';
        }}

        html += '</div></div>';

        // Context components section with expandable state
        if (data.component_info.context_components && Object.keys(data.component_info.context_components).length > 0) {{
          html += '<div class="component-section">';
          html += '<div class="component-header">▼ Context Components (click to expand)</div>';
          html += '<div class="component-body">';

          for (const [key, comp] of Object.entries(data.component_info.context_components)) {{
            const compId = `comp_${{key.replace(/[^a-zA-Z0-9]/g, '_')}}`;
            html += `<div class="component-item">`;
            html += `<div class="component-item-header" onclick="toggleComponentState('${{compId}}')">`;
            html += `<span><span class="component-item-name">${{escapeHtml(key)}}</span>`;
            html += `<span class="component-item-class"> (${{escapeHtml(comp.class_name)}})</span></span>`;
            html += `<span class="component-item-toggle" id="toggle_${{compId}}">▶</span>`;
            html += '</div>';

            // State section (initially hidden)
            html += `<div class="component-state" id="${{compId}}">`;
            if (comp.state && Object.keys(comp.state).length > 0) {{
              const dynamicKeys = comp.dynamic_state ? Object.keys(comp.dynamic_state) : [];
              for (const [stateKey, stateVal] of Object.entries(comp.state)) {{
                const isDynamic = dynamicKeys.includes(stateKey);
                if (isDynamic) {{
                  // Editable dynamic field
                  const inputId = `dyn_${{key.replace(/[^a-zA-Z0-9]/g, '_')}}_${{stateKey}}`;
                  html += '<div class="dynamic-row">';
                  html += '<div class="dynamic-row-header">';
                  html += `<span class="state-key">${{escapeHtml(stateKey)}}</span>`;
                  html += '<span class="dynamic-badge">DYNAMIC</span>';
                  html += '</div>';
                  html += '<div class="dynamic-row-editor">';
                  const valStr = typeof stateVal === 'object' ? JSON.stringify(stateVal) : String(stateVal);
                  html += `<input class="dynamic-input" id="${{inputId}}" type="text" value="${{escapeHtml(valStr)}}" />`;
                  html += `<button class="dynamic-save-btn" data-entity="${{escapeHtml(data.name)}}" data-component="${{escapeHtml(key)}}" data-state-key="${{escapeHtml(stateKey)}}" data-input-id="${{inputId}}">Save</button>`;
                  html += '</div>';
                  html += '</div>';
                }} else {{
                  // Read-only static field
                  html += '<div class="state-row">';
                  html += `<span class="state-key">${{escapeHtml(stateKey)}}</span>`;
                  const valStr = typeof stateVal === 'object' ? JSON.stringify(stateVal, null, 1) : String(stateVal);
                  html += `<span class="state-value">${{escapeHtml(valStr)}}</span>`;
                  html += '</div>';
                }}
              }}
            }} else {{
              html += '<div class="state-empty">No state data</div>';
            }}
            html += '</div></div>';
          }}

          html += '</div></div>';
        }}
      }}

      // Prefab Parameters section
      html += '<div class="component-section">';
      html += '<div class="component-header">▼ Prefab Parameters</div>';
      html += '<div class="component-body">';

      for (const param of data.params) {{
        html += '<div class="param-row">';
        html += `<span class="param-name">${{escapeHtml(param.key)}}</span>`;
        html += `<span class="param-value">${{escapeHtml(param.value)}}</span>`;
        html += '</div>';
      }}

      html += '</div></div>';

      // Metadata section
      html += '<div class="component-section">';
      html += '<div class="component-header">▼ Metadata</div>';
      html += '<div class="component-body">';
      html += '<div class="param-row">';
      html += '<span class="param-name">Role</span>';
      html += `<span class="param-value">${{escapeHtml(data.role)}}</span>`;
      html += '</div>';
      html += '<div class="param-row">';
      html += '<span class="param-name">Prefab</span>';
      html += `<span class="param-value">${{escapeHtml(data.prefab)}}</span>`;
      html += '</div>';
      html += '</div></div>';

      content.innerHTML = html;

      content.querySelectorAll('.dynamic-save-btn').forEach(btn => {{
        btn.addEventListener('click', function() {{
          saveComponentState(
            this.getAttribute('data-entity'),
            this.getAttribute('data-component'),
            this.getAttribute('data-state-key'),
            this.getAttribute('data-input-id')
          );
        }});
      }});
    }}

    function toggleComponentState(compId) {{
      const stateEl = document.getElementById(compId);
      const toggleEl = document.getElementById('toggle_' + compId);
      if (stateEl && toggleEl) {{
        if (stateEl.classList.contains('expanded')) {{
          stateEl.classList.remove('expanded');
          toggleEl.textContent = '▶';
        }} else {{
          stateEl.classList.add('expanded');
          toggleEl.textContent = '▼';
        }}
      }}
    }}

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    function saveComponentState(entityName, componentName, stateKey, inputId) {{
      const input = document.getElementById(inputId);
      if (!input) return;
      const value = input.value;
      logConsole(`Saving ${{componentName}}.${{stateKey}}...`, 'info');

      const payload = {{
        entity_name: entityName,
        component_name: componentName,
        key: stateKey,
        value: value,
      }};

      sendPostCommand('/cmd/set_component_state', payload, function(response) {{
        try {{
          const result = JSON.parse(response);
          if (result.status === 'ok') {{
            logConsole('Dynamic state updated: ' + result.message, 'success');
          }} else {{
            logConsole('Error updating state: ' + result.message, 'error');
          }}
        }} catch (e) {{
          logConsole('Parse error: ' + e.message, 'error');
        }}
      }});
    }}

    function sendPostCommand(endpoint, data, successCallback) {{

      var xhr = new XMLHttpRequest();
      xhr.open('POST', endpoint, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.onreadystatechange = function() {{
        if (xhr.readyState === 4) {{
          if (xhr.status === 200) {{
            successCallback(xhr.responseText);
          }} else {{
            logConsole('Request failed (status ' + xhr.status + ')', 'error');
          }}
        }}
      }};
      xhr.onerror = function() {{
        logConsole('Network error — is the server running?', 'error');
      }};
      xhr.send(JSON.stringify(data));
    }}

    // Simulation control functions
    let isRunning = false;
    let eventSource = null;

    function simPlay() {{
      sendCommand('/cmd/play', function(response) {{
        logConsole('▶ Simulation playing', 'success');
        isRunning = true;
        updateControlState();
      }});
    }}

    function simPause() {{
      sendCommand('/cmd/pause', function(response) {{
        logConsole('⏸ Simulation paused', 'success');
        isRunning = false;
        updateControlState();
      }});
    }}

    function simStep() {{
      sendCommand('/cmd/step', function(response) {{
        logConsole('⏭ Step executed', 'success');
        isRunning = false;
        updateControlState();
      }});
    }}

    function sendCommand(endpoint, successCallback) {{
      var xhr = new XMLHttpRequest();
      xhr.open('GET', endpoint, true);
      xhr.onreadystatechange = function() {{
        if (xhr.readyState === 4) {{
          if (xhr.status === 200) {{
            successCallback(xhr.responseText);
          }} else {{
            logConsole('Command failed (status ' + xhr.status + ')', 'error');
          }}
        }}
      }};
      xhr.onerror = function() {{
        logConsole('Network error — is the server running?', 'error');
      }};
      xhr.send();
    }}

    function logConsole(message, type) {{
      type = type || 'info';
      var output = document.getElementById('console-output');
      var line = document.createElement('div');
      line.className = 'console-line ' + type;
      var now = new Date();
      var timestamp = now.toTimeString().split(' ')[0];
      line.innerHTML = '<span class="timestamp">[' + timestamp + ']</span> ' + message;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    }}

    function updateControlState() {{
      const indicator = document.getElementById('status-indicator');
      const btnPlay = document.getElementById('btn-play');
      const btnPause = document.getElementById('btn-pause');
      const btnStep = document.getElementById('btn-step');

      if (isRunning) {{
        indicator.className = 'status-indicator running';
        btnPlay.classList.add('active');
        btnPause.classList.remove('active');
        btnStep.disabled = true;
      }} else {{
        indicator.className = 'status-indicator paused';
        btnPlay.classList.remove('active');
        btnPause.classList.add('active');
        btnStep.disabled = false;
      }}
    }}

    function updateStepCounter(step) {{
      document.getElementById('step-counter').textContent = step;
    }}

    function updateEntityAction(entityName, action) {{
      const actionEl = document.getElementById('action_' + entityName.replace(/[^a-zA-Z0-9]/g, '_'));
      if (actionEl) {{
        actionEl.textContent = action;
      }}
    }}

    function setActiveGM(gmName) {{
      document.querySelectorAll('.active-gm-badge').forEach(badge => {{
        badge.style.display = 'none';
      }});
      const badgeId = 'gm_badge_' + gmName.replace(/[^a-zA-Z0-9]/g, '_');
      const badge = document.getElementById(badgeId);
      if (badge) {{
        badge.style.display = 'block';
      }}
    }}

    function connectSSE() {{
      eventSource = new EventSource('/events');

      eventSource.onmessage = function(event) {{
        const data = JSON.parse(event.data);

        // Handle simulation completion
        if (data.completion) {{
          logConsole('✓ Simulation completed', 'success');
          document.querySelector('.step-counter').textContent = 'COMPLETE';
          document.querySelector('.step-counter').style.color = '#00ff00';
          isPlaying = false;
          return;
        }}

        // Handle entity info update (component data from simulation)
        if (data.entity_info) {{

          // Update entityData with component info
          for (const [entityName, entityInfo] of Object.entries(data.entities)) {{
            // Find the entityData entry by name
            for (const [eid, edata] of Object.entries(entityData)) {{
              if (edata.name === entityName && entityInfo.component_info) {{
                edata.component_info = entityInfo.component_info;
              }}
            }}
          }}
          for (const [gmName, gmInfo] of Object.entries(data.game_masters)) {{
            for (const [eid, edata] of Object.entries(entityData)) {{
              if (edata.name === gmName && gmInfo.component_info) {{
                edata.component_info = gmInfo.component_info;
              }}
            }}
          }}
          // Refresh inspector if an entity is selected
          if (selectedEntity) {{
            updateInspector(selectedEntity);
          }}
          return;
        }}

        updateStepCounter(data.step);
        logConsole(`Step ${{data.step}}`, 'info');

        // Update entity actions
        if (data.entity_actions) {{
          for (const [entity, action] of Object.entries(data.entity_actions)) {{
            logConsole(`🎭 ${{entity}}: ${{action.substring(0, 120)}}`, 'info');
            updateEntityAction(entity, action);
          }}
        }}

        // Show active GM badge
        if (data.game_master) {{
          setActiveGM(data.game_master);
          logConsole(`⚖️ ${{data.game_master}} resolved action`, 'info');
        }}

        // Update GM card with resolve result
        if (data.acting_entity && data.action) {{
          updateEntityAction(data.acting_entity, data.action);
        }}

        // If currently selected entity, update inspector with pre_act values
        if (selectedEntity && data.entity_logs && data.entity_logs[selectedEntity]) {{
          updateInspectorWithLiveData(data.entity_logs[selectedEntity]);
        }}
      }};

      eventSource.onerror = function(err) {{
        console.log('SSE connection error, reconnecting...');
      }};
    }}

    function updateInspectorWithLiveData(logs) {{
      // Update the inspector panel with live component values
      const content = document.querySelector('.inspector-content');
      if (!content || !logs) return;

      for (const [key, value] of Object.entries(logs)) {{
        const valueEl = document.getElementById('live_' + key.replace(/[^a-zA-Z0-9]/g, '_'));
        if (valueEl && typeof value === 'object' && value.Value) {{
          valueEl.textContent = String(value.Value).substring(0, 100);
        }}
      }}
    }}

    // Initialize
    updateControlState();

    // Try to connect to SSE (will work when served from simulation server)
    if (window.location.protocol !== 'file:') {{
      // Test server connectivity first
      fetch('/status')
        .then(r => {{
          return r.json();
        }})
        .then(data => {{
          logConsole('Connected to simulation server', 'success');
        }})
        .catch(err => {{
          logConsole('Server connection failed — is the server running?', 'error');
        }});

      connectSSE();
    }} else {{
      logConsole('Running from file:// - server features disabled', 'warning');
    }}
  </script>
</body>
</html>"""

  return html
