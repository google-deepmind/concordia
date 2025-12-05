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
# Install Concordia on macOS or Windows.
#
# This script installs from requirements_nolinux.txt, which contains all
# dependencies EXCEPT the vllm extra (Linux-only GPU packages).
#
# For Linux with GPU support, use bin/install.sh instead.
set -euxo pipefail
cd "$(dirname "$0")/.."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "Warning: On Linux, consider using bin/install.sh for GPU support."
  echo "Continuing with requirements_nolinux.txt (no vllm/CUDA)..."
fi

if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Error: Concordia requires Python 3.12 or newer."
  exit 1
fi

echo 'Installing requirements...'
pip install --no-deps --require-hashes --requirement requirements_nolinux.txt
echo
echo

echo 'Installing Concordia...'
pip install --no-deps --no-index --no-build-isolation --editable .
echo
echo

pip list
