"""Calculate Elo Ratings for the Contest submissions."""

import argparse
import glob
import json
import os
import sys

def update_elo(rating_a, rating_b, score_a, score_b, k=32):
    # Standard Elo formula
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 / (1.0 + 10 ** ((rating_a - rating_b) / 400.0))

    if score_a > score_b: # A wins
        actual_a, actual_b = 1.0, 0.0
    elif score_a < score_b: # B wins
        actual_a, actual_b = 0.0, 1.0
    else: # Draw
        actual_a, actual_b = 0.5, 0.5

    new_a = rating_a + k * (actual_a - expected_a)
    new_b = rating_b + k * (actual_b - expected_b)
    return new_a, new_b

def main():
    parser = argparse.ArgumentParser(description="Calculate Elo ratings")
    parser.add_argument("--eval_dir", type=str, default="evaluations")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.eval_dir, "*_out.json"))
    if not files:
        print(f"No evaluation results found in {args.eval_dir}.")
        return

    agent_scores: dict[str, dict[str, float]] = {}
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        for res in results:
            agent = res["focal_agent"]
            scenario = res["scenario"]
            score = res["focal_per_capita_score"]
            
            if agent not in agent_scores:
                agent_scores[agent] = {}
            agent_scores[agent][scenario] = score

    agents = list(agent_scores.keys())
    if not agents:
        print("No agent data found.")
        return

    # Initialize Elo ratings
    ratings = {agent: 1000.0 for agent in agents}
    
    # Find all common scenarios
    all_scenarios = set()
    for ag_sc in agent_scores.values():
        all_scenarios.update(ag_sc.keys())
        
    # Pairwise comparison between agents against the baseline (their relative scores)
    for scenario in all_scenarios:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent_a, agent_b = agents[i], agents[j]
                
                # Check if both have scores for this scenario
                if scenario in agent_scores[agent_a] and scenario in agent_scores[agent_b]:
                    score_a = agent_scores[agent_a][scenario]
                    score_b = agent_scores[agent_b][scenario]
                    
                    new_a, new_b = update_elo(ratings[agent_a], ratings[agent_b], score_a, score_b)
                    ratings[agent_a] = new_a
                    ratings[agent_b] = new_b

    print("Elo ratings for each agent:")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for agent, rating in sorted_ratings:
        print(f"{agent}: {rating:.2f}")

if __name__ == "__main__":
    main()
