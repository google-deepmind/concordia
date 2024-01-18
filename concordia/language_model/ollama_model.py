from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from collections.abc import Collection, Sequence
from string import Template
from typing_extensions import override

import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, filename='ollama.log')
logger = logging.getLogger('ollama')

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 3
_MAX_SAMPLE_TEXT_ATTEMPTS = 5

# PROMPT_TEMPLATE = Template("""<s>[INST] <<SYS>>\n$system_message\n<</SYS>>\n$message[/INST]""")
PROMPT_TEMPLATE = Template("$system_message\n\n$message")

# TEXT_SYSTEM_MESSAGE = """Your task is to follow instructions and answer questions correctly and concisely."""
TEXT_SYSTEM_MESSAGE = ""

MULTIPLE_CHOICE_SYSTEM_MESSAGE = """The following is a multiple choice question.
You must always respond with one of the possible choices.
You may not respond with anything else.
"""
# MULTIPLE_CHOICE_SYSTEM_MESSAGE = ""

class OllamaLanguageModel(language_model.LanguageModel):
    """Language Model that uses Ollama LLM models."""

    def __init__(
        self,
        model_name: str,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
        streaming: bool = False,
    ):
        """Initializes the instance.

        Args:
          model_name: The language model to use.
          measurements: The measurements object to log usage statistics to.
          channel: The channel to write the statistics to.
        """
        self._model_name = model_name
        self._measurements = measurements
        self._channel = channel
        if streaming:
            self._client = Ollama(
                model=model_name, 
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
        else:
            self._client = Ollama(model=model_name)

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
        system_message = TEXT_SYSTEM_MESSAGE,
    ) -> str:
        
        message = prompt
        prompt = PROMPT_TEMPLATE.substitute(system_message=system_message, message=message)

        logger.debug(f"Sending prompt to LLM: {prompt}")
        for retry in range(_MAX_SAMPLE_TEXT_ATTEMPTS): 
            try:
                response = self._client(
                    prompt,
                    stop=terminators,
                    temperature=temperature,
                )
            except ValueError as e:
                logger.error(f"Error while calling LLM with input: {prompt}. Attempt {retry+1} failed.")
                if retry == _MAX_SAMPLE_TEXT_ATTEMPTS - 1: 
                    logger.error(f"Max retries exceeded. Raising exception.")
                    raise language_model.InvalidResponseError(prompt)
            else:
                logger.debug(f"Succeeded after {retry+1} attempts.")
                logger.debug(f"Response from LLM: {response}")
                break

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {'raw_text_length': len(response)},
            )
        return response

    @staticmethod
    def extract_choices(text):
        match = re.search(r'\(?(\w)\)', text)
        if match:
            return match.group(1)
        else:
            return None

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        max_characters = len(max(responses, key=len))

        attempts = 1
        for _ in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            sample = self.sample_text(
                prompt,
                max_characters=max_characters,
                temperature=0.0,
                seed=seed,
                system_message=MULTIPLE_CHOICE_SYSTEM_MESSAGE,
            )
            answer = self.extract_choices(sample)
            try:
                idx = responses.index(answer)
            except ValueError:
                attempts += 1
                continue
            else:
                if self._measurements is not None:
                    self._measurements.publish_datum(
                        self._channel, {'choices_calls': attempts}
                    )
                debug = {}
                return idx, responses[idx], debug

        logger.error(f"Multiple choice failed after {_MAX_MULTIPLE_CHOICE_ATTEMPTS} attempts.\nLLM Input: {prompt}\nLLM Output: {sample}\nExtracted Answer: {answer}")

        raise language_model.InvalidResponseError(
             f'Too many multiple choice attempts.\nLLM Input: {prompt}\nLLM Output: {sample}'
        )