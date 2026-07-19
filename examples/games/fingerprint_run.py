# Copyright 2026 DeepMind Technologies Limited.
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

"""Computes a tamper-evident fingerprint of one or more completed game runs.

`run.py` writes a `simulation_scores_{game}_{scenario}.json` file for each
run, containing the per-agent focal/background scores plus run metadata
(model name, timestamp, command). This script reads a set of those score
files and produces two hashes:

- `content_hash`: a SHA-256 Merkle root over every (scenario, role, agent,
  score) tuple across all supplied runs. Two runs with identical
  `content_hash` produced identical per-scenario per-agent outcomes, even if
  they differ in when/how/on-what-model they were run. This is what makes
  model drift detectable: matching aggregate scores can hide different
  per-scenario outcomes, but they can't produce the same `content_hash`.
- `bundle_hash`: a SHA-256 hash over the `content_hash` plus each run's
  identity (game, scenario, model name, timestamp, command). Two runs with
  the same scores but different provenance share a `content_hash` but not a
  `bundle_hash`.

Usage:
  python -m examples.games.fingerprint_run \
      --scores /tmp/simulation_scores_*.json
  python -m examples.games.fingerprint_run \
      --scores /tmp/simulation_scores_haggling_fruitville.json \
      --output /tmp/fingerprint.json
"""

import argparse
import glob
import hashlib
import json
import pathlib
import sys


def _leaf_hash(scenario: str, role: str, agent: str, score: float) -> str:
  """Hashes a single (scenario, role, agent, score) outcome."""
  canonical = f"{scenario}|{role}|{agent}|{score!r}"
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merkle_root(leaves: list[str]) -> str:
  """Computes a SHA-256 Merkle root over a list of leaf hashes.

  Args:
    leaves: Leaf hashes. The caller is responsible for sorting them into a
      canonical order first, so that the resulting root is independent of
      input ordering.

  Returns:
    The hex-encoded Merkle root. An empty input yields the hash of the
    empty string, so that "no data" is distinguishable from any real run.
  """
  if not leaves:
    return hashlib.sha256(b"").hexdigest()
  level = list(leaves)
  while len(level) > 1:
    next_level = []
    for i in range(0, len(level), 2):
      left = level[i]
      # Duplicate the last node when the level has an odd size.
      right = level[i + 1] if i + 1 < len(level) else left
      next_level.append(
          hashlib.sha256((left + right).encode("utf-8")).hexdigest()
      )
    level = next_level
  return level[0]


def collect_leaves(runs: list[dict]) -> list[str]:
  """Builds the sorted list of leaf hashes for a set of runs."""
  leaves = []
  for run in runs:
    scenario = run.get("scenario", "")
    for role, key in (
        ("focal", "focal_scores"),
        ("background", "background_scores"),
    ):
      for agent, score in run.get(key, {}).items():
        leaves.append(_leaf_hash(scenario, role, agent, float(score)))
  return sorted(leaves)


def build_content_hash(runs: list[dict]) -> str:
  """Computes the content_hash: a Merkle root over all scored outcomes."""
  return _merkle_root(collect_leaves(runs))


def build_bundle_hash(content_hash: str, runs: list[dict]) -> str:
  """Computes the bundle_hash: content_hash plus run identity/provenance."""
  identity = sorted(
      (
          {
              "game": run.get("game"),
              "scenario": run.get("scenario"),
              "model_name": run.get("model_name"),
              "api_type": run.get("api_type"),
              "timestamp": run.get("timestamp"),
              "command": run.get("command"),
          }
          for run in runs
      ),
      key=lambda entry: (entry["game"] or "", entry["scenario"] or ""),
  )
  payload = json.dumps(
      {"content_hash": content_hash, "runs": identity}, sort_keys=True
  )
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint(runs: list[dict]) -> dict:
  """Builds the full fingerprint report for a set of runs."""
  content_hash = build_content_hash(runs)
  bundle_hash = build_bundle_hash(content_hash, runs)
  return {
      "content_hash": content_hash,
      "bundle_hash": bundle_hash,
      "num_runs": len(runs),
      "scenarios": sorted({run.get("scenario", "") for run in runs}),
  }


def load_score_files(paths: list[str]) -> list[dict]:
  """Loads and parses each score JSON file in `paths`."""
  runs = []
  for path in paths:
    with open(path, "r", encoding="utf-8") as f:
      runs.append(json.load(f))
  return runs


def resolve_paths(patterns: list[str]) -> list[str]:
  """Expands a list of literal paths and/or glob patterns into file paths."""
  paths = []
  for pattern in patterns:
    matched = sorted(glob.glob(pattern))
    if not matched and pathlib.Path(pattern).is_file():
      matched = [pattern]
    if not matched:
      print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
    paths.extend(matched)
  return paths


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Compute a tamper-evident fingerprint over one or more completed "
          "game runs, from the score files written by run.py."
      )
  )
  parser.add_argument(
      "--scores",
      nargs="+",
      required=True,
      help=(
          "One or more paths and/or glob patterns to "
          "simulation_scores_{game}_{scenario}.json files."
      ),
  )
  parser.add_argument(
      "--output",
      type=str,
      default=None,
      help="Optional path to write the fingerprint report as JSON.",
  )
  args = parser.parse_args(argv)

  paths = resolve_paths(args.scores)
  if not paths:
    print("Error: no score files found.", file=sys.stderr)
    sys.exit(1)

  runs = load_score_files(paths)
  fingerprint = build_fingerprint(runs)

  print(f"Runs included: {fingerprint['num_runs']}")
  print(f"Scenarios: {', '.join(fingerprint['scenarios'])}")
  print(f"content_hash: {fingerprint['content_hash']}")
  print(f"bundle_hash:  {fingerprint['bundle_hash']}")

  if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
      json.dump(fingerprint, f, indent=2)
    print(f"Fingerprint report written to {args.output}")


if __name__ == "__main__":
  main()
