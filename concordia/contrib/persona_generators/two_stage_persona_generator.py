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

r"""Two-stage persona generator for creating diverse personas.

Part of the Persona Generators framework described in:
  Persona Generators: Generating Diverse Synthetic Personas at Scale
  https://arxiv.org/abs/2602.03545

This module provides a TwoStagePersonaGenerator class for generating diverse
personas and their formative memories in two stages.

Example Usage (can be run in a main script):

from concordia.language_model import no_language_model
model = no_language_model.NoLanguageModel()

generator = TwoStagePersonaGenerator(model)

initial_context = "A group of scientists on a remote arctic expedition."
diversity_axes = ["introversion/extroversion", "optimism/pessimism"]
num_personas = 5

results = generator.generate(initial_context, diversity_axes, num_personas)

print("\n--- Combined Results ---")
print(json.dumps(results, indent=2))
"""

from collections.abc import Sequence
import json
import re
from typing import Any, Dict, List, cast

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.utils import concurrency


class TwoStagePersonaGenerator:
  """Generates diverse personas and their formative memories in two stages."""

  def __init__(self, model: language_model.LanguageModel):
    """Initializes the TwoStagePersonaGenerator.

    Args:
      model: The language model to use for generation.
    """
    self._model = model

  def _generate_backstory_episodes(
      self,
      player_name: str,
      shared_memories: Sequence[str] = (),
      player_specific_context: str = "",
      sentences_per_episode: int = 5,
      delimiter_symbol: str = "***",
      num_memories: int = 4,
  ) -> Sequence[str]:
    """Generates backstory episodes for a character.

    Args:
      player_name: The name of the player.
      shared_memories: Memories shared among all players.
      player_specific_context: Context specific to this player.
      sentences_per_episode: The maximum number of sentences per episode.
      delimiter_symbol: The delimiter to use between episodes.
      num_memories: The number of memories to generate.

    Returns:
      A sequence of generated backstory episodes.
    """
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement("----- Role Playing Master Class -----\n")
    prompt.statement("Question: What is the protagonist's name?")
    prompt.statement(f"Answer: {player_name}\n")
    prompt.statement("Question: Describe the setting or background.")
    shared_memories_str = "\n".join(shared_memories)
    prompt.statement(f"Answer: {shared_memories_str}\n")

    if player_specific_context:
      prompt.statement(
          "Question: Describe the personal context of the protagonist."
      )
      prompt.statement(f"Answer: {player_specific_context}\n")

    gender = prompt.open_question("What is the protagonist's gender?")
    date_of_birth = prompt.open_question(
        "What year was protagonist born? Respond with just the year as a "
        'number, e.g. "1990".',
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    question = (
        f"Write a life story for a {gender} character "
        f"named {player_name} who was born in {date_of_birth}."
    )
    question += (
        f"Begin the story when {player_name} is very young and end it when they"
        f" are of their current age. Generate {num_memories} memories as"
        f" {num_memories} paragraphs in total. The story may include details"
        " such as (but not limited to) any of the following: what their job is"
        " or was, what their typical day was or is like, what their goals,"
        " desires, hopes, dreams, and aspirations are, and have been, as well"
        " as their drives, duties, responsibilities, and obligations. It"
        " should clarify what gives them joy and what are they afraid of. It"
        " may include their friends and family, as well as antagonists. It"
        " should be a complete life story for a complete person but it should"
        " not specify how their life ends. The reader should be left with a"
        f" profound understanding of {player_name}."
    )
    backstory = prompt.open_question(
        question,
        max_tokens=1125 * num_memories,
        terminators=["\nQuestion", "-----"],
    )
    backstory = re.sub(r"\.\s", ".\n", backstory)

    inner_prompt = interactive_document.InteractiveDocument(self._model)
    inner_prompt.statement("Creative Writing Master Class\n")
    inner_prompt.statement("Character background story:\n\n" + backstory)
    question = (
        f"Given the life story above, invent {num_memories} formative episodes"
        " from "
        f"the life of {player_name}. "
        f"They should be memorable events for {player_name} and "
        "important for establishing who they are as a person. They should "
        f"be consistent with {player_name}'s personality and "
        f"circumstances. Describe each episode from {player_name}'s "
        "perspective and use third-person limited point of view. Each"
        " episode "
        "must mention their age at the time the event occurred using"
        " language "
        f'such as "When {player_name} was 5 years old, they '
        'experienced..." . Use past tense. Write no more than'
        f" {sentences_per_episode} sentences "
        "per episode. Separate episodes from one another by the delimiter "
        f'"{delimiter_symbol}". Do not apply any other '
        "special formatting besides these delimiters."
    )
    aggregated_result = inner_prompt.open_question(
        question=question,
        max_tokens=6000,
        terminators=[],
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    episodes = list(aggregated_result.split(delimiter_symbol))
    return episodes

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
            " These new personas should be different from the ones already"
            " generated."
        )

      question = question_intro + (
          " Each persona should be unique and the set of personas should be"
          f" diverse across the axes: {diversity_axes_str}. Consider the"
          " explanations above when creating the personas.\nProvide the output"
          " as a single JSON object, where keys are strings like"
          " 'persona_1', 'persona_2', etc., up to"
          f" f'persona_{batch_size}'. Each value in the object should be a"
          " dictionary representing a persona. Each persona object should have"
          " at least a 'name', 'axis_position', and a 'description'"
          " field. The description should detail the persona's"
          " characteristics, background, and how they relate to the diversity"
          ' axes.\nExample format for the output:\n{{\n  "persona_1": {{\n'
          '    "name": "[Persona Name 1]",\n    "axis_position": {{ "axis1":'
          ' "value1", "axis2": "value2" }},\n    "description": "[Detailed'
          ' description 1]"\n  }},\n  "persona_2": {{\n    "name":'
          ' "[Persona Name 2]",\n     "axis_position": {{ "axis1": "value3",'
          ' "axis2": "value4" }},\n    "description": "[Detailed description'
          ' 2]"\n  }},\n  ...\n}}\nMake sure the entire output is just the'
          " JSON object and nothing else."
      )

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
      num_memories: int = 4,
  ) -> List[str]:
    """Generates formative memories for a single persona.

    Args:
      persona_details: Dictionary containing the persona's characteristics.
      num_memories: The number of memories to generate.

    Returns:
      A list of strings representing the persona's formative memories.
    """
    player_name = persona_details.get("name", "Unknown")
    player_specific_context = persona_details.get("description", "")
    shared_memories = [persona_details.get("initial_context", "")]

    try:
      memories = self._generate_backstory_episodes(
          player_name=player_name,
          shared_memories=shared_memories,
          player_specific_context=player_specific_context,
          num_memories=num_memories,
      )
      return list(memories)
    except (RuntimeError, ValueError) as e:
      # Catching specific exceptions that might arise from LLM interaction
      # or processing of its output within _generate_backstory_episodes.
      print(
          f"Error generating memories for {player_name}:"
          f" {type(e).__name__}: {e}"
      )
      return []  # Return empty list on error to be robust

  def stage_2_generate_memories_concurrently(
      self,
      persona_details_list: List[Dict[str, Any]],
      max_workers: int = 10,
  ) -> Dict[str, List[str]]:
    """Generates formative memories for multiple personas concurrently.

    Args:
      persona_details_list: A list of dictionaries, each representing a persona.
      max_workers: Maximum number of concurrent threads.

    Returns:
      A dictionary mapping persona names to their list of formative memories.
    """
    tasks = {}
    for persona_details in persona_details_list:
      player_name = persona_details.get("name")
      if player_name:
        tasks[player_name] = (
            lambda pd=persona_details: self.generate_single_persona_memories(pd)
        )

    results, errors = concurrency.run_tasks_in_background(
        tasks, max_workers=max_workers
    )

    if errors:
      for name, error in errors.items():
        print(f"Error processing persona {name}: {error}")

    # The concurrency utility returns dict[str, Any], but we know from the tasks
    # that the values are List[str]. We use cast to inform the type checker.
    final_results: Dict[str, List[str]] = cast(Dict[str, List[str]], results)
    return final_results

  def generate(
      self,
      initial_context: str,
      diversity_axes: Sequence[str],
      num_personas: int,
      max_workers: int = 10,
  ) -> Dict[str, Dict[str, Any]]:
    """Generates diverse personas and their formative memories.

    Args:
      initial_context: Shared context for all personas.
      diversity_axes: Axes along which to encourage diversity.
      num_personas: The number of personas to generate.
      max_workers: Maximum number of concurrent threads for memory generation.

    Returns:
      A dictionary mapping persona names to their details and memories.
    """
    # Stage 1
    persona_characteristics = self.generate_diverse_persona_characteristics(
        initial_context, diversity_axes, num_personas
    )

    print("--- Stage 1 Output ---")
    print(json.dumps(persona_characteristics, indent=2))

    if not persona_characteristics:
      print("No persona characteristics generated, skipping stage 2.")
      return {}

    # Stage 2
    persona_memories = self.stage_2_generate_memories_concurrently(
        persona_characteristics, max_workers
    )
    print("\n--- Stage 2 Output ---")
    for name, memories in persona_memories.items():
      print(f"\n{name}:")
      for mem in memories:
        print(f"- {mem}")

    # Combine results
    output = {}
    for char in persona_characteristics:
      name = char.get("name")
      if name is None or not isinstance(name, str):
        print(
            "Warning: Skipping persona due to missing or invalid 'name':"
            f" {char}"
        )
        continue

      persona_data: Dict[str, Any] = {
          "characteristics": char,
          "memories": persona_memories.get(name, []),
      }
      output[name] = persona_data
    return output
