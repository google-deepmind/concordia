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

"""File manipulation and data storage utilities."""

from collections.abc import Sequence
import csv
import json
from typing import Any


def append_to_csv(file_path: str, new_row: Sequence[str]):
  """Append a new row to a CSV file."""
  with open(file_path, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(new_row)


def read_csv(
    file_path: str, unpack_singleton_rows: bool = False
) -> Sequence[Sequence[Any]]:
  """Read a CSV file and return its contents as a list of lists.

  Create the Comma Separated Value (CSV) file if it does not already exist.

  Args:
    file_path: the path to the file to load.
    unpack_singleton_rows: if enabled, then return singleton rows without
      enclosing them in lists.

  Returns:
    The contents of the CSV file as a list of lists or a list of singletons if
      `unpack_singleton_rows` is enabled.
  """
  data = []
  with open(file_path, 'a+', encoding='utf-8') as file:
    file.seek(0)  # Move to the beginning of the file for reading
    reader = csv.reader(file, delimiter='\n')
    for row in reader:
      data.append(row)
  if unpack_singleton_rows:
    data = [row[0] for row in data if len(row) == 1]
  return data


def append_to_json(file_path: str, json_str: str):
  """Append to a json file."""
  # Read the existing data
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
  except FileNotFoundError:
    # If the file doesn't exist, start with an empty list
    data = []
  # Append new data
  data.append(json_str)
  # Write the updated data back to the same file
  with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2)


def load_from_json_file(file_path: str):
  """Load data from a json file."""
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
    return data
  except FileNotFoundError as err:
    raise FileNotFoundError(f"The file '{file_path}' was not found.") from err
  except json.JSONDecodeError as err:
    raise ValueError(
        f"The file '{file_path}' does not contain valid JSON."
    ) from err
  except Exception as err:
    raise RuntimeError(f'An unexpected error occurred: {str(err)}') from err
