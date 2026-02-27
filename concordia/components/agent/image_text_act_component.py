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

"""An acting component that generates both text and image outputs in JSON."""

from collections.abc import Sequence
import json
import re
from typing import override

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

_IMAGE_MODES = frozenset({'image_first', 'text_first', 'choice'})

_MAX_IMAGE_ATTEMPTS = 2

_FAILED_IMAGE_PLACEHOLDER = 'FAILED TO MAKE AN IMAGE'

_IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(data:(image/[a-zA-Z0-9.\-]+);base64,([^)]+)\)'
)

DEFAULT_IMAGE_PROMPT_QUESTION = (
    'Given the context above, write a short prompt (one or two sentences) '
    'describing an image that would accompany this activity. The prompt '
    'should describe a visual scene, photograph, or illustration '
    'relevant to the situation.'
)

DEFAULT_TEXT_WITH_IMAGE_QUESTION = (
    'Given the context above and the following image description:\n'
    '{image_description}\n\n'
)

DEFAULT_IMAGE_FROM_TEXT_QUESTION = (
    'Given the following text that was just written:\n'
    '{text}\n\n'
    'Write a short prompt (one or two sentences) describing an image that '
    'would accompany this text. The prompt should describe a visual '
    'scene, photograph, or illustration relevant to the text.'
)

DEFAULT_ORDERING_QUESTION = (
    'Given the context above, should this response start by generating an '
    'image to inspire the text, or should the text be written first and '
    'then an image created to accompany it?'
)

_IMAGE_GENERATION_PROMPT = (
    'Generate an image based on the following description. '
    'Do not include any non-embedded text in your response, only output '
    'the image. The image may contain embedded text, such as in a meme.\n\n'
)


def _extract_images(text: str) -> list[dict[str, str]]:
  images = []
  for match in _IMAGE_PATTERN.finditer(text):
    images.append({
        'alt': match.group(1),
        'mime_type': match.group(2),
        'data': match.group(3),
    })
  return images


class ImageTextActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """A component that generates both text and image outputs in JSON format.

  This component extends the basic concat-act pattern to produce structured
  JSON output containing both a text response and an image. The ordering
  of generation (image-first or text-first) is controlled by the `image_mode`
  parameter:
    - 'image_first': always generate image first, then text conditioned on it
    - 'text_first': always generate text first, then image from text
    - 'choice': let the LLM decide which ordering to use

  When image generation fails after retries, the image field contains a
  placeholder string. When no image model is provided, the image field is null.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      image_model: language_model.LanguageModel | None = None,
      image_mode: str = 'choice',
      component_order: Sequence[str] | None = None,
      prefix_entity_name: bool = True,
      randomize_choices: bool = True,
      image_prompt_question: str = DEFAULT_IMAGE_PROMPT_QUESTION,
      text_with_image_question: str = DEFAULT_TEXT_WITH_IMAGE_QUESTION,
      image_from_text_question: str = DEFAULT_IMAGE_FROM_TEXT_QUESTION,
      ordering_question: str = DEFAULT_ORDERING_QUESTION,
  ):
    super().__init__()
    if image_mode not in _IMAGE_MODES:
      raise ValueError(
          f'Invalid image_mode: {image_mode!r}. Must be one of {_IMAGE_MODES}.'
      )
    self._model = model
    self._image_model = image_model
    self._image_mode = image_mode
    self._prefix_entity_name = prefix_entity_name
    self._randomize_choices = randomize_choices
    self._image_prompt_question = image_prompt_question
    self._text_with_image_question = text_with_image_question
    self._image_from_text_question = image_from_text_question
    self._ordering_question = ordering_question
    if component_order is None:
      self._component_order = None
    else:
      self._component_order = tuple(component_order)
    if self._component_order is not None:
      if len(set(self._component_order)) != len(self._component_order):
        raise ValueError(
            'The component order contains duplicate components: '
            + ', '.join(self._component_order)
        )

  def _context_for_action(
      self,
      contexts: entity_component.ComponentContextMapping,
  ) -> str:
    if self._component_order is None:
      return '\n'.join(context for context in contexts.values() if context)
    else:
      order = self._component_order + tuple(
          sorted(set(contexts.keys()) - set(self._component_order))
      )
      return '\n'.join(contexts[name] for name in order if contexts[name])

  def _generate_image(self, image_prompt: str) -> str | None:
    if self._image_model is None:
      return None
    full_prompt = _IMAGE_GENERATION_PROMPT + image_prompt
    for _ in range(_MAX_IMAGE_ATTEMPTS):
      try:
        result = self._image_model.sample_text(
            prompt=full_prompt,
            max_tokens=2000,
        )
      except Exception:  # pylint: disable=broad-except
        continue
      images = _extract_images(result)
      if images:
        return f'![image](data:{images[0]["mime_type"]};base64,{images[0]["data"]})'
    return _FAILED_IMAGE_PLACEHOLDER

  def _determine_ordering(
      self, prompt: interactive_document.InteractiveDocument
  ) -> str:
    if self._image_mode in ('image_first', 'text_first'):
      return self._image_mode
    idx = prompt.multiple_choice_question(
        question=self._ordering_question,
        answers=['Generate image first', 'Write text first'],
    )
    return 'image_first' if idx == 0 else 'text_first'

  def _generate_image_first(
      self,
      prompt: interactive_document.InteractiveDocument,
      call_to_action: str,
  ) -> dict[str, str | None]:
    image_prompt = prompt.open_question(
        self._image_prompt_question,
        max_tokens=300,
        terminators=(),
    )

    image_result = self._generate_image(image_prompt)

    text_question = (
        self._text_with_image_question.format(image_description=image_prompt)
        + call_to_action
    )
    text_output = ''
    if self._prefix_entity_name:
      text_output = self.get_entity().name + ' '
    text_output += prompt.open_question(
        text_question,
        max_tokens=2200,
        answer_prefix=text_output,
        terminators=(),
        question_label='Exercise',
    )

    return {'text': text_output, 'image': image_result}

  def _generate_text_first(
      self,
      prompt: interactive_document.InteractiveDocument,
      call_to_action: str,
  ) -> dict[str, str | None]:
    text_output = ''
    if self._prefix_entity_name:
      text_output = self.get_entity().name + ' '
    text_output += prompt.open_question(
        call_to_action,
        max_tokens=2200,
        answer_prefix=text_output,
        terminators=(),
        question_label='Exercise',
    )

    image_prompt_question = self._image_from_text_question.format(
        text=text_output
    )
    image_prompt = prompt.open_question(
        image_prompt_question,
        max_tokens=300,
        terminators=(),
    )

    image_result = self._generate_image(image_prompt)

    return {'text': text_output, 'image': image_result}

  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + '\n')

    call_to_action = action_spec.call_to_action.replace(
        '{name}', self.get_entity().name
    )
    if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
      ordering = self._determine_ordering(prompt)

      if ordering == 'image_first':
        result = self._generate_image_first(prompt, call_to_action)
      else:
        result = self._generate_text_first(prompt, call_to_action)

      output = json.dumps(result)
      self._log(output, prompt)
      return output
    elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      idx = prompt.multiple_choice_question(
          question=call_to_action,
          answers=action_spec.options,
          randomize_choices=self._randomize_choices,
      )
      output = action_spec.options[idx]
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      if self._prefix_entity_name:
        prefix = self.get_entity().name + ' '
      else:
        prefix = ''
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt)
      try:
        return str(float(sampled_text))
      except ValueError:
        return 'nan'
    else:
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}. '
          'Supported output types are: FREE, CHOICE, and FLOAT.'
      )

  def _log(self, result: str, prompt: interactive_document.InteractiveDocument):
    self._logging_channel({
        'Summary': f'Action: {result}',
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
    })

  def get_context_concat_order(self) -> Sequence[str] | None:
    return self._component_order

  def get_state(self) -> entity_component.ComponentState:
    return {
        'component_order': (
            list(self._component_order) if self._component_order else None
        ),
        'prefix_entity_name': self._prefix_entity_name,
        'randomize_choices': self._randomize_choices,
        'image_mode': self._image_mode,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'component_order' in state:
      order = state['component_order']
      self._component_order = tuple(order) if order else None
    if 'prefix_entity_name' in state:
      self._prefix_entity_name = state['prefix_entity_name']
    if 'randomize_choices' in state:
      self._randomize_choices = state['randomize_choices']
    if 'image_mode' in state:
      self._image_mode = state['image_mode']
