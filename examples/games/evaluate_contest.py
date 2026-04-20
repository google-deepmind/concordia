"""Evaluate the submitted agent on all contest scenarios."""

import argparse
import datetime
import importlib
import json
import os
import sys
from typing import Dict, List, Any

import numpy as np

from concordia.contrib import language_models
from concordia.examples.games import scenarios as scenarios_lib
import sentence_transformers

def get_game(scenario_name: str) -> str:
    if "haggling_multi_item" in scenario_name:
        return "haggling_multi_item"
    elif "haggling" in scenario_name:
        return "haggling"
    elif "pub_coordination" in scenario_name:
        return "pub_coordination"
    return ""

def main():
    parser = argparse.ArgumentParser(description="Run Concordia evaluation")
    parser.add_argument("--agent", type=str, default="rational__Entity")
    parser.add_argument("--api_type", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--disable_language_model", action="store_true")
    args = parser.parse_args()

    model = language_models.language_model_setup(
        api_type=args.api_type,
        model_name=args.model,
        disable_language_model=args.disable_language_model,
    )

    if not args.disable_language_model:
        st_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
        embedder = lambda x: st_model.encode(x, show_progress_bar=False)
    else:
        embedder = lambda x: np.ones(5)

    scenarios_results = []
    
    for scenario_name, scenario_config in scenarios_lib.SCENARIO_CONFIGS.items():
        if scenario_name.startswith("_"):
            continue # skip test scenarios
        game = get_game(scenario_name)
        if not game:
            print(f"Cannot identify game for {scenario_name}")
            continue

        print(f"Running {scenario_name} in {game}")

        # Load the game's simulation module
        simulation_path = f"concordia.examples.games.{game}.simulation"
        simulation = importlib.import_module(simulation_path)

        # Load the scenario config
        config_path = f"concordia.examples.games.{game}.configs.{scenario_config.config_module}"
        config_lib = importlib.import_module(config_path)

        # Test agent prefab override
        if hasattr(config_lib, "FOCAL_PLAYER_PREFAB"):
            setattr(config_lib, "FOCAL_PLAYER_PREFAB", args.agent)

        try:
            results = simulation.run_simulation(
                config=config_lib,
                model=model,
                embedder=embedder,
            )
            
            focal_scores = results.get("focal_scores", {})
            focal_mean = float(np.mean(list(focal_scores.values()))) if focal_scores else 0.0

            scenarios_results.append({
                "scenario": scenario_name,
                "focal_agent": args.agent,
                "focal_per_capita_score": focal_mean
            })
            print(f"Finished {scenario_name}. Focal Score: {focal_mean}")

        except Exception as e:
            print(f"Error in {scenario_name}: {e}")

    # Output to json
    results_dir = "evaluations"
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"{args.agent}_out.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scenarios_results, f, indent=2)
    print(f"Evaluation complete. Results saved to {out_file}.")

if __name__ == "__main__":
    main()
