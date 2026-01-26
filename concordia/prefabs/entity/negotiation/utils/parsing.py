"""Utilities for parsing LLM responses in negotiation components."""

from typing import Any, Dict, List, Optional

from concordia.prefabs.entity.negotiation.config import ParsingConfig


def parse_structured_response(
    response: str,
    sections: Optional[List[str]] = None,
    default_confidence: Optional[float] = None,
) -> Dict[str, Any]:
  """Parse an LLM response with labeled sections.

  Handles responses in format:
      ANALYSIS: Some analysis text
      RECOMMENDATIONS: First recommendation
      Second recommendation
      CONFIDENCE: 0.8
      KEY_FACTORS: Factor 1
      Factor 2

  Args:
      response: Raw LLM response text
      sections: List of section names to look for. Defaults to common sections.
      default_confidence: Default confidence value if parsing fails.

  Returns:
      Dict mapping section names to their content.
      List sections (RECOMMENDATIONS, KEY_FACTORS, RISKS, OPPORTUNITIES)
      return lists. Other sections return strings.
  """
  if default_confidence is None:
    default_confidence = ParsingConfig.DEFAULT_CONFIDENCE

  if sections is None:
    sections = ParsingConfig.DEFAULT_SECTIONS

  list_sections = {
      'RECOMMENDATIONS', 'KEY_FACTORS', 'RISKS',
      'OPPORTUNITIES', 'SUGGESTIONS'
  }

  parsed: Dict[str, Any] = {
      'analysis': '',
      'recommendations': [],
      'confidence': default_confidence,
      'key_factors': [],
      'risks': [],
      'opportunities': []
  }

  current_section: Optional[str] = None

  for line in response.strip().split('\n'):
    line = line.strip()
    if not line:
      continue

    # Check if line starts a new section
    section_found = False
    for section in sections:
      prefix = f"{section}:"
      if line.upper().startswith(prefix.upper()):
        current_section = section.lower()
        content = line[len(prefix):].strip()

        if section.upper() == 'CONFIDENCE':
          try:
            parsed['confidence'] = float(content)
          except (ValueError, IndexError):
            parsed['confidence'] = default_confidence
        elif section.upper() in list_sections:
          parsed[current_section] = [content] if content else []
        else:
          parsed[current_section] = content
        section_found = True
        break

    # If not a new section, append to current section
    if not section_found and current_section:
      if current_section == 'confidence':
        continue  # Skip additional lines for confidence
      elif isinstance(parsed.get(current_section), list):
        if line:  # Don't add empty lines
          parsed[current_section].append(line)
      else:
        parsed[current_section] = parsed.get(current_section, '') + ' ' + line

  return parsed


def parse_confidence(response: str, default: float = 0.7) -> float:
  """Extract confidence score from response.

  Args:
      response: LLM response text
      default: Default value if parsing fails

  Returns:
      Confidence value between 0 and 1
  """
  parsed = parse_structured_response(response, ['CONFIDENCE'])
  try:
    confidence = float(parsed.get('confidence', default))
    return max(0.0, min(1.0, confidence))
  except (ValueError, TypeError):
    return default


def parse_score(response: str, default: float = 0.5) -> float:
  """Extract numeric score from response.

  Args:
      response: LLM response text
      default: Default value if parsing fails

  Returns:
      Score value between 0 and 1
  """
  parsed = parse_structured_response(response, ['SCORE'])
  try:
    score = float(parsed.get('score', default))
    return max(0.0, min(1.0, score))
  except (ValueError, TypeError):
    return default
