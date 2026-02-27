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

"""HTML rendering for structured simulation logs.

This module provides the render_dynamic_html function which converts a
SimulationLog into an interactive HTML page with JavaScript-based content
composition and deduplication.
"""

from __future__ import annotations

import html
import json
from typing import Any

from concordia.utils import structured_logging

# pylint: disable=g-inconsistent-quotes
# pylint: disable=invalid-name


def render_dynamic_html(
    simulation_log: structured_logging.SimulationLog,
    entity_memories: dict[str, list[str]] | None = None,
    game_master_memories: list[str] | None = None,
    player_scores: dict[str, Any] | None = None,
    title: str = 'Simulation Log',
) -> str:
  """Render the log to HTML with JavaScript-based content composition.

  Args:
    simulation_log: The log to render.
    entity_memories: Dict mapping entity names to lists of memory strings.
    game_master_memories: List of game master memory strings.
    player_scores: Optional dict of player scores to display.
    title: Title for the HTML page.

  Returns:
    Complete HTML string with embedded data and dynamic rendering.
  """

  # Build the content store data for JavaScript
  content_store_data = simulation_log.content_store.to_dict()

  # Build entries data
  entries_data = []
  for entry in simulation_log.entries:
    entries_data.append({
        'step': entry.step,
        'timestamp': entry.timestamp,
        'entity_name': entry.entity_name,
        'component_name': entry.component_name,
        'entry_type': entry.entry_type,
        'summary': entry.summary,
        'deduplicated_data': dict(entry.deduplicated_data),
    })

  # Build entity memories data
  entity_memories_data = entity_memories or {}
  gm_memories_data = game_master_memories or []

  # Get entity names for tabs
  entity_names = simulation_log.get_entity_names()
  # Filter to only entities that have memories
  entity_tabs = [name for name in entity_names if name in entity_memories_data]

  # Build the HTML
  html_parts = [
      """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>"""
      + html.escape(title)
      + """</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
.container { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.tab { overflow: hidden; border-bottom: 2px solid #ddd; margin-bottom: 15px; }
.tab button { background-color: #f1f1f1; border: none; padding: 12px 24px; cursor: pointer; font-size: 14px; transition: all 0.3s; border-radius: 4px 4px 0 0; margin-right: 2px; }
.tab button:hover { background-color: #ddd; }
.tab button.active { background-color: #667eea; color: white; }
.tabcontent { display: none; padding: 15px; animation: fadeIn 0.3s; }
.tabcontent.active { display: block; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.step { margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #667eea; border-radius: 4px; }
.step-header { font-weight: bold; color: #333; margin-bottom: 8px; }
.entry { margin: 5px 0; padding: 8px; background: white; border-radius: 4px; }
.entry-entity { color: #667eea; font-weight: bold; }
.entry-component { color: #888; font-size: 12px; }
.entry-summary { margin: 5px 0; }
.content-block { margin: 5px 0; padding: 8px; background: #f0f0f0; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto; }
.content-label { font-weight: bold; color: #555; font-size: 11px; text-transform: uppercase; }
.memory-item { margin: 5px 0; padding: 8px; background: #f9f9f9; border-radius: 4px; }
details { margin: 5px 0; }
details summary { cursor: pointer; font-weight: bold; padding: 5px; background: #f0f0f0; border-radius: 4px; }
details[open] summary { background: #e0e0e0; }
/* Step-level details get special styling with a colored left border */
details.step-details { margin: 10px 0; border-left: 3px solid #667eea; padding-left: 10px; }
details.step-details > summary { background: #e8e8ff; font-size: 14px; }
details.step-details[open] > summary { background: #d8d8ff; }
/* Content inside steps is indented */
details.step-details > details { margin-left: 15px; }
.summary { padding: 10px; background: #e8f4f8; border-radius: 4px; margin-bottom: 15px; }
h1 { color: #333; margin-bottom: 5px; }
.subtitle { color: #666; margin-bottom: 20px; }
</style>
</head>
<body>
<div class="container">
<h1>"""
      + html.escape(title)
      + """</h1>
<p class="subtitle">Click on the tabs to view different sections.</p>
"""
  ]

  if player_scores:
    html_parts.append(
        '<div class="summary">'
        f'Player Scores: {html.escape(str(player_scores))}</div>'
    )

  # Tab buttons
  html_parts.append('<div class="tab">\n')
  html_parts.append(
      '<button class="tablinks active" onclick="openTab(event, \'gm_log\')">Game Master log</button>\n'  # pylint: disable=line-too-long
  )
  for name in entity_tabs:
    safe_id = html.escape(name.replace(' ', '_'))
    html_parts.append(
        f'<button class="tablinks" onclick="openTab(event, \'{safe_id}\')">{html.escape(name)}</button>\n'  # pylint: disable=line-too-long
    )
  if gm_memories_data:
    html_parts.append(
        '<button class="tablinks" onclick="openTab(event, \'gm_memories\')">Game Master Memories</button>\n'  # pylint: disable=line-too-long
    )
  html_parts.append('</div>\n')

  # GM Log tab - uses JavaScript to render
  html_parts.append('<div id="gm_log" class="tabcontent active"></div>\n')

  # Entity tabs
  for name in entity_tabs:
    safe_id = html.escape(name.replace(' ', '_'))
    html_parts.append(f'<div id="{safe_id}" class="tabcontent"></div>\n')

  # GM Memories tab
  if gm_memories_data:
    html_parts.append('<div id="gm_memories" class="tabcontent"></div>\n')

  # Embed data as JSON
  html_parts.append('<script>\n')
  html_parts.append('const CONTENT_STORE = ')
  html_parts.append(json.dumps(content_store_data))
  html_parts.append(';\n')
  html_parts.append('const ENTRIES = ')
  html_parts.append(json.dumps(entries_data))
  html_parts.append(';\n')
  html_parts.append('const ENTITY_MEMORIES = ')
  html_parts.append(json.dumps(entity_memories_data))
  html_parts.append(';\n')
  html_parts.append('const GM_MEMORIES = ')
  html_parts.append(json.dumps(gm_memories_data))
  html_parts.append(';\n')

  # Add JavaScript for rendering — regex patterns use raw strings to avoid
  # Python backslash warnings.
  _CONTENT_REF_RE = r"!\[([^\]]*)\](content_ref:([a-f0-9]+)\)"
  _IMAGE_MD_RE = r"!\[([^\]]*)\]((data:image\/[^)]+)\)"
  html_parts.append("""
function getContent(id) {
  const raw = CONTENT_STORE[id] || '';
  return resolveContentRefs(raw);
}

function resolveContentRefs(text) {
  if (typeof text !== 'string') return text;
  var re = new RegExp('""" + _CONTENT_REF_RE + """', 'g');
  return text.replace(re, function(m, alt, refId) {
    const data = CONTENT_STORE[refId];
    return data ? '![' + alt + '](' + data + ')' : m;
  });
}

function renderImageMarkdown(text) {
  var re = new RegExp('""" + _IMAGE_MD_RE + """', 'g');
  return text.replace(re, function(m, alt, src) {
    return '<img src="' + src + '" alt="' + alt + '" style="max-width:400px;max-height:400px;display:block;margin:8px 0;border-radius:4px;" loading=\"lazy\">';
  });
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML.replace(/\\\\n/g, '<br />');
}

function renderObject(obj) {
  if (obj === null || obj === undefined) {
    return '';
  }

  if (typeof obj === 'string') {
    if (obj.includes('data:image/')) {
      return renderImageMarkdown(escapeHtml(obj));
    }
    return escapeHtml(obj);
  }

  if (typeof obj === 'number' || typeof obj === 'boolean') {
    return String(obj);
  }

  if (Array.isArray(obj)) {
    let html = '';
    obj.forEach(item => {
      html += renderObject(item) + '<br />';
    });
    return html;
  }

  if (typeof obj === 'object') {
    if (obj._ref && Object.keys(obj).length === 1) {
      const content = getContent(obj._ref);
      if (content !== undefined) {
        if (content.includes('data:image/')) {
          return renderImageMarkdown(escapeHtml(content));
        }
        return escapeHtml(content);
      }
      return '[ref:' + obj._ref + ']';
    }

    // Determine summary from special keys (like PythonObjectToHTMLConverter)
    let summary = '';
    if (obj.date) {
      summary = escapeHtml(obj.date);
      if (obj.Summary) {
        summary += '  ' + escapeHtml(obj.Summary);
      }
    } else if (obj.Summary) {
      summary = escapeHtml(obj.Summary);
    } else if (obj.Name) {
      summary = escapeHtml(obj.Name);
    } else if (obj.Key) {
      summary = escapeHtml(obj.Key);
    } else if (obj.Value !== undefined || obj.value !== undefined) {
      // Entity data with a "Value" key - use "Details" as summary
      summary = 'Details';
    } else {
      // For all other objects, use "Details" as a generic summary
      summary = 'Details';
    }

    let html = '<details>';
    html += '<summary>' + summary + '</summary>';

    for (const [key, value] of Object.entries(obj)) {
      if (key !== 'date' && key !== 'Summary') {
        html += '<b><ul>' + escapeHtml(key) + '</b>';
        html += '<li>' + renderObject(value) + '</li></ul>';
      }
    }

    html += '</details>';
    return html;
  }

  return String(obj);
}

// Render object children directly without wrapping in outer <details>
// Used when we already have an outer details wrapper for the entity
function renderObjectChildren(obj) {
  if (obj === null || obj === undefined) {
    return '';
  }

  if (typeof obj !== 'object' || Array.isArray(obj)) {
    return renderObject(obj);
  }

  let html = '';
  for (const [key, value] of Object.entries(obj)) {
    if (key !== 'date' && key !== 'Summary') {
      html += '<b><ul>' + escapeHtml(key) + '</b>';
      html += '<li>' + renderObject(value) + '</li></ul>';
    }
  }
  return html;
}

function renderGMLog() {
  const container = document.getElementById('gm_log');
  let html = '';

  // Group entries by step
  const stepMap = {};
  ENTRIES.forEach(entry => {
    if (!stepMap[entry.step]) stepMap[entry.step] = [];
    stepMap[entry.step].push(entry);
  });

  const steps = Object.keys(stepMap).map(Number).sort((a, b) => a - b);

  steps.forEach(step => {
    const entries = stepMap[step];
    \n\
    // Build step summary from entries
    let stepSummary = 'Step ' + step;
    if (entries.length > 0 && entries[0].summary) {
      stepSummary += ' --- ' + entries[0].summary;
    }
    \n\
    html += '<details class="step-details" open>';
    html += '<summary><b>' + escapeHtml(stepSummary) + '</b></summary>';

    entries.forEach(entry => {
      // Create a label for this entry (like "Entity [name]" or component name)
      let entryLabel = entry.entity_name;
      if (entry.entry_type === 'entity') {
        entryLabel = 'Entity [' + entry.entity_name + ']';
      }
      \n\
      // If entry has deduplicated_data, render it as collapsible content
      if (entry.deduplicated_data && Object.keys(entry.deduplicated_data).length > 0) {
        html += '<details>';
        html += '<summary>' + escapeHtml(entryLabel) + '</summary>';
        // Render all the data in deduplicated_data recursively
        for (const [key, value] of Object.entries(entry.deduplicated_data)) {
          html += '<b><ul>' + escapeHtml(key) + '</b>';
          html += '<li>' + renderObject(value) + '</li></ul>';
        }
        html += '</details>';
      } else {
        // Simple entry with no data
        html += '<div class="entry">';
        html += '<span class="entry-entity">' + escapeHtml(entryLabel) + '</span>';
        html += ' <span class="entry-component">(' + escapeHtml(entry.component_name) + ')</span>';
        if (entry.summary) {
          html += '<div class="entry-summary">' + escapeHtml(entry.summary) + '</div>';
        }
        html += '</div>';
      }
    });

    html += '</details>';
  });

  container.innerHTML = html || '<p>No log entries.</p>';
}

function renderEntityMemories(entityName, containerId) {
  const container = document.getElementById(containerId);
  const memories = ENTITY_MEMORIES[entityName] || [];
  let html = '';

  if (memories.length === 0) {
    html = '<p>No memories for ' + escapeHtml(entityName) + '.</p>';
  } else {
    memories.forEach(mem => {
      html += '<div class="memory-item">' + escapeHtml(mem) + '</div>';
    });
  }

  container.innerHTML = html;
}

function renderGMMemories() {
  const container = document.getElementById('gm_memories');
  if (!container) return;

  let html = '';
  if (GM_MEMORIES.length === 0) {
    html = '<p>No game master memories.</p>';
  } else {
    GM_MEMORIES.forEach(mem => {
      html += '<div class="memory-item">' + escapeHtml(mem) + '</div>';
    });
  }

  container.innerHTML = html;
}

function openTab(evt, tabId) {
  // Hide all tab content
  document.querySelectorAll('.tabcontent').forEach(tc => {
    tc.classList.remove('active');
  });

  // Remove active class from all buttons
  document.querySelectorAll('.tablinks').forEach(btn => {
    btn.classList.remove('active');
  });

  // Show the selected tab
  document.getElementById(tabId).classList.add('active');
  evt.currentTarget.classList.add('active');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  renderGMLog();

  // Render entity memories
  Object.keys(ENTITY_MEMORIES).forEach(name => {
    const containerId = name.replace(/ /g, '_');
    if (document.getElementById(containerId)) {
      renderEntityMemories(name, containerId);
    }
  });

  renderGMMemories();
});
</script>
""")

  html_parts.append('</div>\n</body>\n</html>')

  return ''.join(html_parts)
