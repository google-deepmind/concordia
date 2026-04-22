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

r"""Two-stage persona generator for creating diverse personas (AlphaEvolve Solution 1).

Part of the Persona Generators framework described in:
  Persona Generators: Generating Diverse Synthetic Personas at Scale
  https://arxiv.org/abs/2602.03545

This module provides a TwoStagePersonaGenerator class for generating diverse
personas and their individual memories in two stages.

Example Usage:

from concordia.language_model import no_language_model

# Use NoLanguageModel for testing. For real use, see language_model_setup.
model = no_language_model.NoLanguageModel()

generator = TwoStagePersonaGenerator(model)

initial_context = "A group of scientists on a remote arctic expedition."
diversity_axes = ["introversion/extroversion", "optimism/pessimism"]
num_personas = 5

characteristics = generator.generate_diverse_persona_characteristics(
    initial_context=initial_context,
    diversity_axes=diversity_axes,
    num_personas=num_personas,
)
for persona in characteristics:
  memories = generator.generate_single_persona_memories(persona)
  persona["memories"] = memories
"""

from collections.abc import Sequence
import json
import logging
from typing import Any, Dict, List, Optional

from concordia.contrib.data.questionnaires import base_questionnaire
from concordia.contrib.persona_generators import quasi_monte_carlo as qmc
from concordia.document import interactive_document
from concordia.language_model import language_model


class TwoStagePersonaGenerator:
  """Generates diverse personas and their formative memories in two stages."""

  def __init__(self, model: language_model.LanguageModel):
    """Initializes the TwoStagePersonaGenerator.

    Args:
      model: The language model to use for generation.
    """
    self._model = model

  def generate_diverse_persona_characteristics(
      self,
      initial_context: str,
      diversity_axes: Sequence[str],
      num_personas: int,
      questionnaire: Optional[base_questionnaire.QuestionnaireBase] = None,
  ) -> List[Dict[str, Any]]:
    """Generates a list of diverse persona characteristics using an LLM.

    This is done in two steps:
    1. Sample ideal values for persona traits using Quasi-Monte Carlo (QMC).
    2. Generate detailed persona descriptions based on these sampled values.

    Args:
      initial_context: Shared context for all personas.
      diversity_axes: Axes along which to encourage diversity (e.g.,
        personality).
      num_personas: The number of personas to generate.
      questionnaire: Optional questionnaire object to source dimension ranges
        from.

    Returns:
      A list of dictionaries, each representing a persona.
    """
    # 1. Generate diverse points in the characteristic space using QMC.
    if questionnaire:
      all_dimension_ranges = questionnaire.get_dimension_ranges()
      dimension_ranges = {
          axis: all_dimension_ranges[axis] for axis in diversity_axes
      }
    else:
      dimension_ranges = {axis: (0.0, 1.0) for axis in diversity_axes}

    ideal_characteristics = qmc.generate_sobol_points(
        dimension_ranges=dimension_ranges,
        dimensions=list(diversity_axes),
        num_points=num_personas,
    )
    logging.info(
        'Generated %d QMC points for axes %s: %s',
        len(ideal_characteristics),
        list(diversity_axes),
        ideal_characteristics,
    )

    # 2. Generate persona descriptions based on the QMC points.
    base_prompt = interactive_document.InteractiveDocument(self._model)

    base_prompt.statement(
        f'The shared context for the personas is: {initial_context}'
    )
    diversity_axes_str = ', '.join(diversity_axes)
    base_prompt.statement(
        'I want to create several diverse personas based on the following'
        f' axes: {diversity_axes_str}.'
    )

    dimensions = sorted(list(diversity_axes))
    # Generate rich explanations and archetypes for each diversity axis to
    # ground the main generation prompt.
    axis_explanations_doc = interactive_document.InteractiveDocument(
        self._model
    )
    axis_explanations_doc.statement(f'The shared context is: {initial_context}')

    axis_json_schema = {}
    for axis in dimensions:
      min_val, max_val = dimension_ranges[axis]
      axis_json_schema[axis] = {
          'explanation': (
              'A brief explanation of this axis in the given context.'
          ),
          'low_archetype': (
              'A brief character sketch for a persona at the low end of the'
              f' scale (value ≈ {min_val}).'
          ),
          'high_archetype': (
              'A brief character sketch for a persona at the high end of the'
              f' scale (value ≈ {max_val}).'
          ),
      }

    axis_explanation_question = (
        'I am creating diverse personas along the following axes:'
        f' {diversity_axes_str}. To help ground the generation process, for'
        ' each axis, please provide a brief explanation and describe two'
        ' archetypal characters representing the extreme ends of the'
        ' scale.\n\nProvide the output as a single JSON object using the'
        f' following schema:\n{json.dumps(axis_json_schema, indent=2)}'
    )

    try:
      explanation_json_str = axis_explanations_doc.open_question(
          axis_explanation_question, max_tokens=3000, temperature=0.5
      )
      explanation_data = json.loads(explanation_json_str)

      explanations_str = (
          'To better understand these axes, here are some explanations and'
          ' archetypes for their extremes:\n\n'
      )
      for axis, data in explanation_data.items():
        if axis in dimensions:  # Verify the LLM returned a requested axis.
          min_val, max_val = dimension_ranges[axis]
          explanations_str += f'Axis: "{axis}"\n'
          explanations_str += f'Explanation: {data.get("explanation", "N/A")}\n'
          explanations_str += (
              f'Low Archetype (value ≈ {min_val}):'
              f' {data.get("low_archetype", "N/A")}\n'
          )
          explanations_str += (
              f'High Archetype (value ≈ {max_val}):'
              f' {data.get("high_archetype", "N/A")}\n\n'
          )
      base_prompt.statement(explanations_str)
      logging.info('Successfully generated and added axis explanations.')

    except (json.JSONDecodeError, KeyError) as e:
      logging.warning('Failed to generate or parse axis explanations: %s', e)
      # Fallback to stating the numerical scales if the explanation fails.
      trait_scales_parts = []
      for dim in dimensions:
        min_val, max_val = dimension_ranges[dim]
        trait_scales_parts.append(
            f'"{dim}" is on a scale from {min_val} to {max_val}'
        )
      trait_scales_str = '\n'.join(trait_scales_parts)
      base_prompt.statement(
          f'The traits are scaled as follows:\n{trait_scales_str}'
      )

    points_str = '\n'.join([
        f'Persona {i+1} traits: {json.dumps(char)}'
        for i, char in enumerate(ideal_characteristics)
    ])
    base_prompt.statement(
        'I have pre-determined the following trait values for each persona to'
        f' ensure diversity:\n{points_str}'
    )

    question_intro = (
        f'Now, please generate {num_personas} distinct personas, one for each'
        ' set of trait values provided.'
    )

    example_str = '...'
    if ideal_characteristics:
      p1_json = json.dumps(ideal_characteristics[0])
      example_p1 = (
          '  "persona_1": {\n'
          '    "name": "[Persona Name 1]",\n'
          f'    "axis_position": {p1_json},\n'
          '    "description": "[Detailed description 1 consistent with'
          ' traits]"\n'
          '  }'
      )
      example_str = example_p1
      if len(ideal_characteristics) > 1:
        p2_json = json.dumps(ideal_characteristics[1])
        example_p2 = (
            ',\n  "persona_2": {\n'
            '    "name": "[Persona Name 2]",\n'
            f'    "axis_position": {p2_json},\n'
            '    "description": "[Detailed description 2 consistent with'
            ' traits]"\n'
            '  }'
        )
        example_str += example_p2
      if len(ideal_characteristics) > 2:
        example_str += ',\n  ...'

    question = (
        f'{question_intro}\nFor each persona, use the provided trait values as'
        " the 'axis_position' and generate a corresponding name and"
        " description. The 'axis_position' must be a dictionary that"
        ' exactly matches the pre-determined values for that persona (e.g.'
        " Persona 1 should use Persona 1's trait values)."
        '\nThe set of personas should be diverse across the'
        f' axes: {diversity_axes_str}.\nProvide the output as a single JSON'
        " object, where keys are strings like 'persona_1', 'persona_2', etc.,"
        f" up to 'persona_{num_personas}'. Each value in the object should be"
        ' a dictionary representing a persona. Each persona object should'
        " have at least a 'name', 'axis_position', and a 'description' field."
        " The 'axis_position' MUST MATCH the trait values provided for that"
        " persona. The description should detail the persona's"
        ' characteristics and background, ensuring they are highly consistent'
        " with the 'axis_position' values, and explain how they relate to the"
        ' diversity'
        f' axes.\nExample format for the output:\n{{\n{example_str}\n}}\n'
        'Make sure the entire output is just the JSON object and nothing'
        ' else.'
    )

    for _ in range(3):  # Allow for retries if JSON parsing fails
      prompt = base_prompt.copy()
      try:
        generated_json = prompt.open_question(
            question,
            max_tokens=5000 * num_personas,
            terminators=(),
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        # Attempt to clean the JSON output
        clean_json = generated_json.strip()
        if clean_json.startswith('```json'):
          clean_json = clean_json[7:]
        if clean_json.endswith('```'):
          clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        personas_dict = json.loads(clean_json)
        if isinstance(personas_dict, dict) and all(
            isinstance(p, dict) for p in personas_dict.values()
        ):
          personas_list = list(personas_dict.values())
          if len(personas_list) < num_personas:
            prompt.statement(
                'The previous output was valid JSON but contained too few'
                f' personas ({len(personas_list)} instead of'
                f' {num_personas}). Please try again, ensuring you'
                f' generate exactly {num_personas} personas.'
            )
            continue

          if len(personas_list) > num_personas:
            personas_list = personas_list[:num_personas]

          # Add initial context to each persona for stage 2
          for p in personas_list:
            p['initial_context'] = initial_context
          logging.info(
              'Successfully generated characteristics for %d personas.',
              len(personas_list),
          )
          return personas_list
        else:
          prompt.statement(
              'The previous output was not a valid JSON object containing'
              ' persona dictionaries. Please try again, ensuring the output'
              ' is a single JSON object with the specified format.'
          )
          continue
      except json.JSONDecodeError as e:
        prompt.statement(
            f'The previous output failed to parse as JSON: {e}. Please try'
            ' again, ensuring the output is valid JSON.'
        )
    else:  # If retry loop finishes without break
      return []  # Return empty list if generation failed.

  def generate_single_persona_memories(
      self,
      persona_details: Dict[str, Any],
  ) -> List[str]:
    """Generates formative memories for a single persona.

    Args:
      persona_details: Dictionary containing the persona's characteristics.

    Returns:
      A list of strings representing the persona's formative memories.
    """
    player_name = persona_details.get('name', 'Unknown')
    logging.info(
        "Generating agent's Logic of Appropriateness for persona: %s",
        player_name,
    )
    description = persona_details.get('description', '')
    axis_position = persona_details.get('axis_position', {})

    # Combine description and axis positions to guide generation.
    axis_str = json.dumps(axis_position)
    player_specific_context = (
        f"{description}\n\nThis persona's quantitative positions on the"
        f' diversity axes are: {axis_str}'
    )
    shared_memories = [persona_details.get('initial_context', '')]

    try:
      memories = self._generate_backstory_episodes(
          player_name=player_name,
          shared_memories=shared_memories,
          player_specific_context=player_specific_context,
      )
      logging.info(
          'Generated Logic of Appropriateness for %s: %s', player_name, memories
      )
      return list(memories)
    except (RuntimeError, ValueError):
      # Catching specific exceptions that might arise from LLM interaction
      # or processing of its output within _generate_backstory_episodes.
      logging.exception(
          'Error generating Logic of Appropriateness for %s',
          player_name,
      )
      return []  # Return empty list on error to be robust

  def _generate_backstory_episodes(
      self,
      player_name: str,
      shared_memories: Sequence[str] = (),
      player_specific_context: str = '',
  ) -> Sequence[str]:
    """Generates backstory episodes for a character.

    Args:
      player_name: The name of the player.
      shared_memories: Memories shared among all players.
      player_specific_context: Context specific to this player.

    Returns:
      A sequence containing the generated backstory.
    """
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement("----- Agent's Logic of Appropriateness Generation -----")
    prompt.statement(
        'You are an expert in computational social science and behavioral'
        " psychology. Your task is to define an agent's core decision-making"
        " logic based on their identity, following the 'Logic of"
        " Appropriateness' framework."
    )
    shared_memories_str = '\n'.join(shared_memories)
    prompt.statement(
        f'The shared context for the simulation is:\n{shared_memories_str}'
    )
    prompt.statement(
        f"The specific persona profile for '{player_name}'"
        f' is:\n{player_specific_context}'
    )

    question = (
        'Based on the shared context and the specific profile of'
        f' {player_name}, write a single, rich paragraph that explains how this'
        ' person interprets situations and decides on appropriate actions.'
        " This paragraph should encapsulate their personal 'Logic of"
        " Appropriateness'.\n\nThe paragraph must explicitly reference their"
        ' identity, background, and core traits from their profile (including'
        ' their quantitative axis scores). It should answer the question:'
        " 'What would a person like me do in a situation like this?' from"
        ' their perspective. Focus on their internal reasoning process. For'
        ' example, how do they weigh different factors? What values or fears'
        ' guide their choices? How do their past experiences shape their view'
        " of what is 'appropriate' behavior?\n\nThis is not a list of"
        ' memories, but a description of their cognitive and ethical framework'
        ' for navigating the world.'
    )

    logic_of_appropriateness = prompt.open_question(
        question,
        max_tokens=1500,
        terminators=[],
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return [logic_of_appropriateness.strip()]
