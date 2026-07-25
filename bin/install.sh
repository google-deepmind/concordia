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
# Install concordia.
set -euxo pipefail
cd "$(dirname "$0")/.."

echo 'Installing requirements...'
REQUIREMENTS_FILE=requirements.txt
# pytype does not yet support Python 3.13+. Its C++ extension fails to build
# from source on environments without a C++20-capable compiler. Filter it out
# so the rest of the dependencies can install cleanly.
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)"; then
  echo 'Python >= 3.13 detected — filtering out pytype (unsupported)...'
  REQUIREMENTS_FILE=/tmp/requirements-filtered.txt
  python -c "
import sys, re
content = open('requirements.txt').read()
# Remove the pytype== block (package line + hashes + trailing comments).
content = re.sub(
    r'^pytype==.*?(?=^[a-zA-Z_])',
    '',
    content,
    flags=re.MULTILINE | re.DOTALL,
)
sys.stdout.write(content)
" > "${REQUIREMENTS_FILE}"
fi
pip install --no-deps --require-hashes --requirement "${REQUIREMENTS_FILE}"
echo
echo

echo 'Installing Concordia...'
pip install --no-deps --no-index --no-build-isolation --editable .
echo
echo

pip list
