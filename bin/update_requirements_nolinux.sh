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
# Update requirements_nolinux.txt for macOS and Windows.
#
# This script generates a platform-specific lock file that includes all extras
# EXCEPT 'vllm' (which requires Linux-only CUDA packages). It must be run on
# macOS or Windows to generate platform-appropriate dependencies.
#
# Run this script when:
# - setup.py dependencies change
# - examples/requirements.in changes
# - You need to update package versions for macOS/Windows
set -euxo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

# Run pip-compile from a temp directory to avoid pyproject.toml's all_extras=true
cd "$BUILD_DIR"
pip-compile \
    --generate-hashes \
    --allow-unsafe \
    --strip-extras \
    --no-emit-index-url \
    --extra amazon \
    --extra dev \
    --extra google \
    --extra huggingface \
    --extra langchain \
    --extra mistralai \
    --extra ollama \
    --extra openai \
    --extra together \
    --output-file="${REPO_DIR}/requirements_nolinux.txt" \
    "${REPO_DIR}/setup.py" "${REPO_DIR}/examples/requirements.in"

echo
echo "Updated: ${REPO_DIR}/requirements_nolinux.txt"
