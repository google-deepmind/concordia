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

r"""Two-stage persona generator for creating diverse personas (AlphaEvolve Solution 3).

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
          '    "core_motivation": "[A concise sentence capturing the essence of'
          ' this persona\'s driving force that makes them unique.]",\n'
          '    "description": "[Detailed description 1 consistent with'
          ' traits and core_motivation, including specific opinions'
          ' reflecting their axis positions.]"\n'
          '  }'
      )
      example_str = example_p1
      if len(ideal_characteristics) > 1:
        p2_json = json.dumps(ideal_characteristics[1])
        example_p2 = (
            ',\n  "persona_2": {\n'
            '    "name": "[Persona Name 2]",\n'
            f'    "axis_position": {p2_json},\n'
            '    "core_motivation": "[A different concise motivation for this'
            ' persona, ensuring they are distinct from Persona 1.]",\n'
            '    "description": "[Detailed description 2 consistent with'
            ' traits and core_motivation, including different opinions from'
            ' Persona 1 that reflect their unique axis positions.]"\n'
            '  }'
        )
        example_str += example_p2
      if len(ideal_characteristics) > 2:
        example_str += ',\n  ...'

    question = (
        f'{question_intro}\nFor each persona, use the provided trait values as'
        " the 'axis_position' and generate a corresponding name,"
        " 'core_motivation', and 'description'.\n\n**Crucially, your goal is to"
        ' maximize the diversity of the generated personas.** Each persona'
        ' should be a unique individual, clearly distinct from the others.'
        ' Create characters that are foils to one another. Explore unusual and'
        ' even counter-intuitive combinations of the axis values to cover the'
        ' full spectrum of human personalities.\n\n'
        "The 'axis_position' MUST be a dictionary that exactly matches the"
        ' pre-determined values for that persona.\n'
        "The 'core_motivation' MUST be a single, concise sentence that"
        " encapsulates the persona's unique driving force, worldview, or"
        ' primary goal. This motivation should be a direct consequence of'
        ' their specific axis positions and should clearly differentiate them'
        ' from all other personas.\n'
        "The 'description' should detail the persona's background and"
        ' characteristics, consistent with their axis positions and flowing'
        " logically from their 'core_motivation'. **Crucially, it must also"
        ' include specific opinions, beliefs, or preferences that directly'
        ' reflect their quantitative position on each of the diversity axes.**'
        ' This makes the persona concrete and measurable.\n\n'
        'Provide the output as a single JSON object, where keys are strings'
        f" like 'persona_1', 'persona_2', etc., up to 'persona_{num_personas}'."
        ' Each value should be a dictionary for a persona.\n'
        f'Example format for the output:\n{{\n{example_str}\n}}\n'
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
    core_motivation = persona_details.get('core_motivation', '')

    # Combine all profile elements to guide generation.
    axis_parts = []
    if axis_position:
      for axis, value in sorted(axis_position.items()):
        axis_parts.append(f'- {axis}: {value:.3f}')
    axis_str = '\n'.join(axis_parts)

    player_specific_context = (
        f'Description: {description}\n\n'
        f'Core Motivation: {core_motivation}\n\n'
        f'Trait Scores:\n{axis_str}'
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
        'Based on the shared context and the detailed profile of'
        f' {player_name}, write a single, rich paragraph that synthesizes their'
        " personal 'Logic of Appropriateness'. This paragraph will serve as"
        " the core of the agent's identity and guide all their actions in a"
        ' simulation.\n\nThe paragraph MUST:\n1. Serve as a direct expansion'
        " of their 'Core Motivation'.\n2. Integrate their background from the"
        " 'description' with their quantitative 'Trait Scores'.\n3. Explicitly"
        ' reference and explain the meaning of their specific trait scores'
        " (e.g., 'a high score in `risk_tolerance` means they tend"
        " to...').\n4. Describe their internal cognitive and ethical"
        ' framework. How do they interpret the world? What principles, values,'
        ' or fears guide their choices?\n5. Explain their decision-making'
        ' process: How do they weigh different factors when faced with a'
        " choice? What makes an action 'appropriate' for someone like"
        ' them?\n\nThis is not a story or a list of memories, but a dense,'
        " analytical description of the persona's worldview and psychological"
        ' makeup.'
    )

    logic_of_appropriateness = prompt.open_question(
        question,
        max_tokens=1500,
        terminators=[],
        temperature=0.9,
        top_p=0.95,
        top_k=64,
    )
    return [logic_of_appropriateness.strip()]
