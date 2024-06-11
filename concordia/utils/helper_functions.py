# Copyright 2023 DeepMind Technologies Limited.
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

"""Helper functions."""

from collections.abc import Iterable, Sequence
import datetime

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
from concordia.utils import concurrency


def filter_copy_as_statement(
    doc: interactive_document.InteractiveDocument,
    include_tags: Iterable[str] = (),
    exclude_tags: Iterable[str] = (),
) -> interactive_document.InteractiveDocument:
  """Copy interactive document as an initial statement.

  Args:
    doc: document to copy
    include_tags: tags to include in the statement.
    exclude_tags: tags to filter out from the statement.
      interactive_document.DEBUG_TAG will always be added.

  Returns:
    an interactive document containing a filtered copy of the input document.
  """
  filtered_view = doc.view(
      include_tags=include_tags,
      exclude_tags={interactive_document.DEBUG_TAG, *exclude_tags},
  )
  result_doc = doc.new()
  result_doc.statement(filtered_view.text())
  return result_doc


def extract_from_generated_comma_separated_list(x: str) -> Sequence[str]:
  """Extract from a maybe badly formatted comma-separated list."""
  result = x.split(',')
  return [item.strip('" ') for item in result]


def is_count_noun(x: str, model: language_model.LanguageModel) -> bool:
  """Output True if the input is a count noun, not a mass noun.

  For a count noun you ask how *many* there are. For a mass noun you ask how
  *much* there is.

  Args:
    x: input string. It should be a noun.
    model: a language model

  Returns:
    True if x is a count noun and False if x is a mass noun.
  """
  examples = (
      'Question: is money a count noun? [yes/no]\n' + 'Answer: no\n'
      'Question: is coin a count noun? [yes/no]\n' + 'Answer: yes\n'
      'Question: is water a count noun? [yes/no]\n' + 'Answer: no\n'
      'Question: is apple a count noun? [yes/no]\n' + 'Answer: yes\n'
      'Question: is token a count noun? [yes/no]\n' + 'Answer: yes\n'
  )
  doc = interactive_document.InteractiveDocument(model=model)
  doc.statement(examples)
  answer = doc.yes_no_question(question=f'is {x} a count noun? [yes/no]')
  return answer


def timedelta_to_readable_str(td: datetime.timedelta):
  """Converts a datetime.timedelta object to a readable string."""
  hours = td.seconds // 3600
  minutes = (td.seconds % 3600) // 60
  seconds = td.seconds % 60

  readable_str = []
  if hours > 0:
    readable_str += [f'{hours} hour' if hours == 1 else f'{hours} hours']
  if minutes > 0:
    if hours > 0:
      readable_str += ' and '
    readable_str += [
        f'{minutes} minute' if minutes == 1 else f'{minutes} minutes'
    ]
  if seconds > 0:
    if hours > 0 or minutes > 0:
      readable_str += [' and ']
    readable_str += [
        f'{seconds} second' if seconds == 1 else f'{seconds} seconds'
    ]

  return ''.join(readable_str)


def apply_recursively(
    parent_component: component.Component,
    function_name: str,
    function_arg: str | None = None,
    concurrent_child_calls: bool = False,
) -> None:
  """Recursively applies a function to each component in a tree of components.

  Args:
    parent_component: the component to apply the function to.
    function_name: the name of the function to apply.
    function_arg: the argument to pass to the function.
    concurrent_child_calls: whether to call the function on child components
      concurrently.
  """

  if concurrent_child_calls:
    with concurrency.executor(
        max_workers=len(parent_component.get_components())) as executor:
      for child_component in parent_component.get_components():
        executor.submit(
            apply_recursively,
            child_component,
            function_name,
            function_arg=function_arg,
            concurrent_child_calls=concurrent_child_calls,
            executor=executor,
        )
  else:
    for child_component in parent_component.get_components():
      apply_recursively(
          child_component, function_name, function_arg=function_arg
      )

  if function_arg is None:
    getattr(parent_component, function_name)()
  else:
    getattr(parent_component, function_name)(function_arg)
