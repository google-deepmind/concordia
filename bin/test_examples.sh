#!/bin/bash
#
# Copyright 2024 DeepMind Technologies Limited.
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
#
# Test the examples.
set -euxo pipefail
cd "$(dirname "$0")/.."
FAILURES=false

echo "pytest examples..."
pytest examples || FAILURES=true
echo
echo

echo "pytype examples..."
pytype examples || FAILURES=true
echo
echo

echo "pylint examples..."
pylint --errors-only examples || FAILURES=true
echo
echo

echo "convert notebooks..."
./bin/convert_notebooks.sh notebooks
echo
echo

echo "pytype notebooks..."
pytype --pythonpath=. notebooks || FAILURES=true
echo
echo

echo "pylint notebooks..."
pylint --errors-only notebooks || FAILURES=true
echo
echo

if "${FAILURES}"; then
  echo -e '\033[0;31mFAILURE\033[0m' && exit 1
else
  echo -e '\033[0;32mSUCCESS\033[0m'
fi
