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

r"""Two-stage persona generator for creating diverse personas (AlphaEvolve Solution 5).

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
import re
from typing import Any, Dict, List

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
      max_batch_size: int = 10,
  ) -> List[Dict[str, Any]]:
    """Generates a list of diverse persona characteristics using an LLM.

    Args:
      initial_context: Shared context for all personas.
      diversity_axes: Axes along which to encourage diversity (e.g.,
        personality).
      num_personas: The number of personas to generate.
      max_batch_size: The maximum number of personas to generate in each batch.
        This is used to avoid generating too many personas at once, which can
        cause the LLM to hallucinate or run out of tokens.

    Returns:
      A list of dictionaries, each representing a persona.
    """
    base_prompt = interactive_document.InteractiveDocument(self._model)

    base_prompt.statement(
        f"The shared context for the personas is: {initial_context}"
    )
    diversity_axes_str = ", ".join(diversity_axes)
    base_prompt.statement(
        "I want to create several diverse personas based on the following"
        f" axes: {diversity_axes_str}."
    )
    base_prompt.statement(
        "For this task, a persona is a description of a simulated individual—"
        "including their background, personality, and key life experiences—"
        "that produces believable human-like behavior in a social setting."
    )

    explanation = base_prompt.open_question(
        "First, please explain what each of these diversity axes means and how "
        "characters can differ along them. This will help guide the generation "
        "process.",
        max_tokens=2000,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    print(f"LLM Explanation of Diversity Axes:\n{explanation}")
    base_prompt.statement(
        f"Here is the explanation of the diversity axes:\n{explanation}"
    )

    all_personas = []
    batch_idx = 0
    while len(all_personas) < num_personas:
      batch_idx += 1
      prompt = base_prompt.copy()
      batch_size = min(max_batch_size, num_personas - len(all_personas))
      if not all_personas:
        question_intro = f"Now, please generate {batch_size} distinct personas."
      else:
        existing_personas_str = json.dumps(all_personas, indent=2)
        prompt.statement(
            "The following personas have already been generated:\n"
            f"{existing_personas_str}"
        )
        question_intro = (
            f"Now, please generate {batch_size} *additional* distinct personas."
            " These new personas should be qualitatively different from the"
            " ones already generated, filling in gaps in the diversity space."
            " Analyze the 'axis_position' of the existing personas and create"
            " new ones that occupy combinations of axis values that are"
            " currently missing or underrepresented. Aim to cover the extremes"
            " and create contrasts."
        )

      if not all_personas:
        strategy_prompt = (
            "Your first task is to seed this simulation with maximum diversity."
            " Generate a set of foundational 'archetypes' that are as different"
            " from one another as possible. Create characters that represent"
            " the absolute extremes of the axes, and then create one or two"
            " with highly unusual, counter-intuitive combinations of traits to"
            " act as 'wildcards'."
        )
      else:
        strategy_prompt = (
            "Analyze the existing personas to identify the most densely "
            "populated region in the diversity space (i.e., where characters "
            "are most similar). Your primary goal is to counteract this "
            "clustering. Generate new personas that are 'anti-personas'—the "
            "polar opposites of the characters in that cluster. Also, identify "
            "the largest empty region in the diversity space and create "
            "characters to inhabit that 'void'. Your goal is to actively push "
            "the boundaries of the population and ensure no two characters are "
            "alike."
        )

      question = (
          question_intro
          + "\n"
          + strategy_prompt
          + (
              " Each persona should be unique. The set of personas should be"
              f" maximally diverse across the axes: {diversity_axes_str}.\n\nTo"
              " ensure maximum diversity, also generate personas with:\n-"
              " **Internal Contradictions**: Personas with complex, even"
              " contradictory, traits. Their background or core beliefs might"
              " conflict with their outward behavior or their position on"
              " another axis (e.g., an optimist with a tragic past, a"
              " pragmatist who takes a leap of faith). The description should"
              " explain the source of this dissonance.\n\nFor the"
              " 'axis_position' field, use descriptive labels like 'Very Low',"
              " 'Low', 'Medium', 'High', 'Very High'.\n\nProvide the output as"
              " a single JSON object, where keys are strings like 'persona_1',"
              f" 'persona_2', etc., up to f'persona_{batch_size}'. Each value"
              " in the object should be a dictionary representing a persona."
              " Each persona object should have a 'name', 'axis_position',"
              " 'core_motivation', 'defining_experience',"
              " 'specific_attitudes', and 'description' field.\n-"
              " 'core_motivation': A concise (1-2 sentence) explanation for"
              " *why* the character holds their axis positions.\n-"
              " 'defining_experience': A brief (1-2 sentence) description of a"
              " pivotal life event that shaped the persona's core motivation"
              " and attitudes.\n- 'specific_attitudes': A dictionary where"
              f" keys are the diversity axes ({diversity_axes_str}) and values"
              " are concrete, measurable opinions or preferences reflecting"
              " their position (e.g., for 'technology_adoption', an attitude"
              " could be 'Views smartphones as essential for family, but"
              " dismisses social media as trivial.').\n- 'description':"
              " Elaborates on motivation and attitudes, detailing the"
              " persona's background and characteristics.\nExample format for"
              ' the output:\n{{\n  "persona_1": {{\n    "name":'
              ' "[Persona Name 1]",\n    "axis_position": {{ "axis1":'
              ' "Very High", "axis2": "Low" }},\n   '
              ' "core_motivation": "[Concise reason for axis'
              ' positions.]",\n    "defining_experience": "[Brief'
              ' description of a formative event.]",\n   '
              ' "specific_attitudes": {{ "axis1": "[Specific positive'
              ' opinion]", "axis2": "[Specific negative preference]"'
              ' }},\n    "description": "[Detailed description'
              ' elaborating on motivation and attitudes.]"\n  }},\n '
              " ...\n}}\nMake sure the entire output is just the JSON object"
              " and nothing else."
          )
      )

      # Add robust JSON parsing with retries to improve reliability.
      new_personas = []
      for attempt in range(3):  # Retry up to 3 times
        raw_response = ""  # Initialize raw_response
        try:
          raw_response = prompt.open_question(question)
          # The LLM often wraps the JSON in markdown. Find the JSON block.
          json_match = re.search(r"\{.*\}", raw_response, re.S)
          if not json_match:
            raise ValueError("No JSON object found in the response.")
          json_str = json_match.group(0)
          new_personas_dict = json.loads(json_str)
          new_personas = list(new_personas_dict.values())
          if not new_personas:
            raise ValueError(
                "Generated JSON is valid but contains no personas."
            )
          break  # Success
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
          error_message = (
              f"Batch {batch_idx}, Attempt {attempt + 1} failed: {e}"
          )
          print(error_message)
          if attempt < 2:
            faulty_response_snippet = raw_response[:200]
            prompt.statement(
                "Your previous response could not be parsed. Error: "
                f"{e.__class__.__name__}: {e}. "
                f"Snippet: '{faulty_response_snippet}...'. "
                "Please try again, ensuring the output is a single, valid JSON "
                "object and nothing else."
            )
          else:
            print(f"Batch {batch_idx} failed after 3 attempts. Skipping.")
            new_personas = []  # Ensure we add an empty list

      all_personas.extend(new_personas)

      for i in range(3):  # Allow for retries if JSON parsing fails
        try:
          print(
              f"Batch {batch_idx}, Attempt {i+1}: Generating {batch_size}"
              f" personas. Total so far: {len(all_personas)}."
          )
          generated_json = prompt.open_question(
              question,
              max_tokens=5000 * batch_size,
              terminators=(),
              temperature=1.0,
              top_p=0.95,
              top_k=64,
          )
          print(f"Raw LLM Output for batch:\n{generated_json}")
          # Attempt to clean the JSON output
          clean_json = generated_json.strip()
          if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
          if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
          clean_json = clean_json.strip()

          personas_dict = json.loads(clean_json)
          if isinstance(personas_dict, dict) and all(
              isinstance(p, dict) for p in personas_dict.values()
          ):
            personas_in_batch = list(personas_dict.values())
            if len(personas_in_batch) < batch_size:
              print(
                  f"LLM generated {len(personas_in_batch)} personas,"
                  f" but {batch_size} were requested. Retrying."
              )
              prompt.statement(
                  "The previous output was valid JSON but contained too few"
                  f" personas ({len(personas_in_batch)} instead of"
                  f" {batch_size}). Please try again, ensuring you"
                  f" generate exactly {batch_size} personas."
              )
              continue  # Force retry by skipping the break and starting next
              # iter.

            if len(personas_in_batch) > batch_size:
              print(
                  f"Warning: LLM generated {len(personas_in_batch)} personas,"
                  f" but {batch_size} were requested in this batch."
              )
            # Add initial context to each persona for stage 2
            for p in personas_in_batch:
              p["initial_context"] = initial_context
            all_personas.extend(personas_in_batch)
            break  # Exit retry loop on success
          else:
            print("Generated output is not a dictionary of dictionaries.")
            prompt.statement(
                "The previous output was not a valid JSON object containing"
                " persona dictionaries. Please try again, ensuring the output"
                " is a single JSON object with the specified format."
            )
        except json.JSONDecodeError as e:
          print(f"JSON Decode Error: {e}")
          prompt.statement(
              f"The previous output failed to parse as JSON: {e}. Please try"
              " again, ensuring the output is valid JSON."
          )
      else:  # If retry loop finishes without break
        print(
            "Failed to generate valid JSON for batch after multiple attempts."
        )
        return all_personas  # Return personas generated so far and stop.

    return all_personas

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
    player_name = persona_details.get("name", "Unknown")
    player_specific_context = persona_details.get("description", "")
    initial_context = persona_details.get("initial_context", "")

    try:
      # Step 1: Generate the 'logic of appropriateness' paragraph.
      logic_doc = interactive_document.InteractiveDocument(self._model)
      logic_doc.statement(
          "You are an expert in computational social science and character "
          "development for agent-based models."
      )
      logic_doc.statement(
          "Your task is to define a character's core decision-making logic "
          "based on their identity, following the Concordia framework."
      )
      logic_doc.statement(
          "A Concordia agent decides how to act by answering three questions: "
          "1. 'What kind of situation is this?', "
          "2. 'What kind of person am I?', "
          "3. 'What would a person like me do in a situation like this?'"
      )
      core_motivation = persona_details.get("core_motivation", "")
      defining_experience = persona_details.get("defining_experience", "")
      specific_attitudes = persona_details.get("specific_attitudes", {})

      logic_doc.statement("\n--- Character Profile ---")
      logic_doc.statement(f"Name: {player_name}")
      if initial_context:
        logic_doc.statement(f"Overall Setting: {initial_context}")
      logic_doc.statement(f"Identity and Background: {player_specific_context}")
      if core_motivation:
        logic_doc.statement(f"Core Motivation: {core_motivation}")
      if defining_experience:
        logic_doc.statement(f"Defining Experience: {defining_experience}")
      if specific_attitudes:
        attitudes_str = "\n".join([
            f"- {axis}: {attitude}"
            for axis, attitude in specific_attitudes.items()
        ])
        logic_doc.statement(f"Specific Attitudes:\n{attitudes_str}")
      logic_doc.statement("-------------------------\n")
      question_logic = (
          "Based on the character profile above, especially their 'Defining"
          " Experience', 'Core Motivation', and 'Specific Attitudes', write a"
          " single paragraph that explains their 'logic of appropriateness'."
          " This paragraph must provide a clear causal link from their defining"
          " experience to their core motivation, attitudes, and background, and"
          " then to their likely behaviors. It should describe how the"
          " character interprets situations and decides on appropriate actions"
          " by referencing their core identity (background, values, etc.),"
          " ensuring the logic is consistent with all provided details. This"
          " text will be the primary context that guides the agent's behavior"
          " in a simulation, helping it answer the question: 'What would a"
          " person like me do in a situation like this?'"
      )
      logic_of_appropriateness = logic_doc.open_question(
          question_logic,
          temperature=0.8,
          max_tokens=500,
          top_p=0.95,
          top_k=64,
      ).strip()

      # Step 2: Generate a Cognitive Heuristic.
      heuristic_doc = interactive_document.InteractiveDocument(self._model)
      heuristic_doc.statement(
          "You are a cognitive psychologist specializing in heuristics and "
          "biases. Your task is to distill a complex character profile into a "
          "core decision-making shortcut."
      )
      heuristic_doc.statement("\n--- Character Profile ---")
      heuristic_doc.statement(f"Name: {player_name}")
      heuristic_doc.statement(
          f"Identity and Background: {player_specific_context}"
      )
      if core_motivation:
        heuristic_doc.statement(f"Core Motivation: {core_motivation}")
      if defining_experience:
        heuristic_doc.statement(f"Defining Experience: {defining_experience}")
      heuristic_doc.statement(
          f"Core Decision-Making Logic (Deliberate): {logic_of_appropriateness}"
      )
      heuristic_doc.statement("-------------------------\n")
      heuristic_question = (
          "Based on this profile, what is the single, core cognitive "
          "heuristic or 'rule of thumb' this person uses to make *quick* "
          "decisions under stress, ambiguity, or uncertainty? This should "
          "reflect their most ingrained, gut-level instinct. Frame it as a "
          "concise, memorable principle or motto. Do not explain it. "
          "Examples: 'When in doubt, defer to the expert.' or 'Always "
          "question the official story.' or 'Protect the group at all costs.'"
      )
      cognitive_heuristic = heuristic_doc.open_question(
          heuristic_question,
          temperature=0.9,
          max_tokens=100,
          top_p=0.95,
          top_k=64,
      ).strip()

      # Step 3: Generate a Behavioral Signature.
      signature_doc = interactive_document.InteractiveDocument(self._model)
      signature_doc.statement(
          "You are a behavioral scientist and expert observer."
      )
      signature_doc.statement("\n--- Character Profile ---")
      signature_doc.statement(f"Name: {player_name}")
      signature_doc.statement(
          f"Identity and Background: {player_specific_context}"
      )
      signature_doc.statement(
          f"Core Decision-Making Logic (Deliberate): {logic_of_appropriateness}"
      )
      signature_doc.statement(
          f"Cognitive Heuristic (Gut-level rule): {cognitive_heuristic}"
      )
      signature_doc.statement("-------------------------\n")
      signature_question = (
          "Based on the character's profile and their core heuristic, "
          "describe a 'behavioral signature'—a concrete, observable action or "
          "pattern of behavior that exemplifies this rule in practice. Write "
          "a single, concise paragraph."
      )
      behavioral_signature = signature_doc.open_question(
          signature_question,
          temperature=0.8,
          max_tokens=300,
          top_p=0.95,
          top_k=64,
      ).strip()

      # Step 4: Combine the context elements with clear labels.
      return [
          f"LOGIC OF APPROPRIATENESS:\n{logic_of_appropriateness}",
          (
              "COGNITIVE HEURISTIC (Decision-making shortcut under "
              f"pressure):\n{cognitive_heuristic}"
          ),
          (
              "BEHAVIORAL SIGNATURE (Observable consequence of "
              f"heuristic):\n{behavioral_signature}"
          ),
      ]

    except (RuntimeError, ValueError) as e:
      # Catching specific exceptions that might arise from LLM interaction
      # or processing of its output.
      print(
          "Error generating comprehensive context for"
          f" {player_name}: {type(e).__name__}: {e}"
      )
      return []  # Return an empty list to indicate failure for this agent.
