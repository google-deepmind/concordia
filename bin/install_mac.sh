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
# Install Concordia on macOS.
#
# This script uses pip-compile to generate macOS-specific requirements by:
# 1. Running pip-compile from a clean temp directory (bypasses pyproject.toml's
#    all_extras=true setting which would include Linux-only packages like vllm)
# 2. Compiling only the 'dev' extra from setup.py, which includes all dev tools
#    but excludes platform-specific GPU packages
# 3. Installing the generated requirements and Concordia in editable mode
#
# This approach was suggested by the maintainers as "Option A" for macOS support.
set -euxo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "Error: This script is intended for macOS only."
  exit 1
fi

if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Error: Concordia requires Python 3.12 or newer."
  exit 1
fi

echo 'Installing pip-tools...'
python -m pip install pip-tools

echo 'Generating macOS-specific requirements from setup.py[dev]...'
# Run from temp directory to bypass pyproject.toml's [tool.pip-tools] config
cd "$BUILD_DIR"
pip-compile \
    --extra dev \
    --strip-extras \
    --no-emit-index-url \
    --output-file=requirements_mac.txt \
    "${REPO_DIR}/setup.py"

echo 'Installing dependencies...'
python -m pip install --requirement "$BUILD_DIR/requirements_mac.txt"

echo 'Installing Concordia in editable mode...'
cd "$REPO_DIR"
python -m pip install --no-deps --editable .

echo
echo 'Installation complete!'
python -m pip list

