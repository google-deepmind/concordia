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
# Sets up the virtual environment.
set -euo pipefail
cd "$(dirname "$0")/.."

readonly VENV_PATH="${1:-venv}"
if [[ -d "${VENV_PATH}" ]]; then
  read -p "Virtual environment "${VENV_PATH}" already exists. Overwrite? (Y/N) " confirm
  [[ "${confirm}" = [Yy]* ]] && rm -rf "${VENV_PATH}" || exit 1
fi

echo "Creating virtual environment at ${VENV_PATH}..."
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}"/bin/activate
python --version
pip --version
pip list
echo
echo

./bin/install.sh
