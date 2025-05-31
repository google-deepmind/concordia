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
import functools
import inspect
import re
import types
from typing import Any

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing.deprecated import component
from concordia.utils import concurrency


def extract_text_between_delimiters(text: str, delimiter: str) -> str | None:
  """Extracts text between the first two occurrences of a delimiter in a string.

  Args:
    text: The string to search through and extract from.
    delimiter: The delimiter string.

  Returns:
    The extracted text, or None if the delimiter does not appear twice.
  """
  first_delimiter_index = text.find(delimiter)
  if first_delimiter_index == -1:
    return None
  second_delimiter_index = text.find(delimiter,
                                     first_delimiter_index + len(delimiter))
  if second_delimiter_index == -1:
    return None

  return text[first_delimiter_index + len(delimiter):second_delimiter_index]


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
    concurrency.run_tasks({
        f'{child_component.name}.{function_name}': functools.partial(
            apply_recursively,
            parent_component=child_component,
            function_name=function_name,
            function_arg=function_arg,
            concurrent_child_calls=concurrent_child_calls,
        )
        for child_component in parent_component.get_components()
    })
  else:
    for child_component in parent_component.get_components():
      apply_recursively(
          parent_component=child_component,
          function_name=function_name,
          function_arg=function_arg,
      )

  if function_arg is None:
    getattr(parent_component, function_name)()
  else:
    getattr(parent_component, function_name)(function_arg)


def get_package_classes(module: types.ModuleType):
  """Load all classes defined in any file within a package."""
  package_name = module.__package__
  prefabs = {}
  submodule_names = [
      value for value in dir(module) if not value.startswith('__')]
  for submodule_name in submodule_names:
    submodule = getattr(module, submodule_name)
    all_var_names = dir(submodule)
    for var_name in all_var_names:
      var = getattr(submodule, var_name)
      if inspect.isclass(var) and var.__module__.startswith(package_name):
        key = f'{submodule_name}__{var_name}'
        prefabs[key] = var()
  return prefabs


def print_pretty_prefabs(data_dict):
  """Generates a Markdown string representation of a dictionary.

  Each object's representation (from its __repr__ method) is formatted
  to show the class name and its arguments on separate lines,
  indented, within a Python code block for syntax highlighting.
  Lines for 'entities=None' or 'entities=()' will be omitted.

  Args:
      data_dict (dict): The dictionary to format. Values are expected to be
        objects whose __repr__ produces a string like "ClassName(arg1=value1,
        arg2=value2, ...)".

  Returns:
      str: A string formatted as Markdown.
  """
  output_lines = []

  if not data_dict:
    return '(The dictionary is empty)'

  for key, value_obj in data_dict.items():
    output_lines.append('---')

    value_str = repr(value_obj)

    output_lines.append(f'**`{key}`**:')
    output_lines.append('```python')

    first_paren_idx = value_str.find('(')

    if first_paren_idx != -1 and value_str.endswith(')'):
      class_name = value_str[:first_paren_idx]
      output_lines.append(f'{class_name}(')

      last_paren_idx = value_str.rfind(')')

      if last_paren_idx > first_paren_idx:
        args_content = value_str[first_paren_idx + 1 : last_paren_idx]

        if args_content.strip():
          raw_split_args = re.split(
              r',\s*(?=[_a-zA-Z][_a-zA-Z0-9]*=)', args_content
          )

          # Filter arguments before printing
          args_to_print = []
          for arg_str in raw_split_args:
            stripped_arg_str = arg_str.strip()
            if not stripped_arg_str:  # Skip if argument is empty after strip
              continue

            # Check if the argument is 'entities=None' or 'entities=()'
            if (
                stripped_arg_str == 'entities=None'
                or stripped_arg_str == 'entities=()'
            ):
              continue  # Skip this argument

            args_to_print.append(stripped_arg_str)

          if args_to_print:
            for i, final_arg_str in enumerate(args_to_print):
              line_to_add = f'    {final_arg_str}'

              if i < len(args_to_print) - 1:  # If it's not the last argument
                line_to_add += ','

              output_lines.append(line_to_add)

        output_lines.append(')')
      else:
        output_lines.append(')')
    else:
      output_lines.append(value_str)

    output_lines.append('```')

  if data_dict:
    output_lines.append('---')

  return '\n'.join(output_lines)


def find_data_in_nested_structure(
    data: Sequence[Any] | dict[str, Any], key: str
) -> list[Any]:
  """Recursively finds all instances of a given key in nested dictionaries/lists."""
  results = []
  if isinstance(data, dict):
    for k, v in data.items():
      if k == key:
        results.append(v)
      results.extend(find_data_in_nested_structure(v, key))
  elif isinstance(data, list):
    for item in data:
      results.extend(find_data_in_nested_structure(item, key))
  return results
