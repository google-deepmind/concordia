# Copyright 2025 [SoyGema] - Modifications and additions with Claude Code
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

"""Utility for exporting evolutionary simulation results to text files with visualizations."""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from concordia.typing import evolutionary as evolutionary_types
from concordia.utils import measurements as measurements_lib


def create_payoff_matrix_figure(
    cooperative_scores: List[float],
    selfish_scores: List[float],
    save_path: Optional[Path] = None
) -> str:
    """Create and save a payoff matrix visualization.
    
    Args:
        cooperative_scores: List of scores from cooperative agents
        selfish_scores: List of scores from selfish agents
        save_path: Path to save the figure (optional)
        
    Returns:
        String representation of the figure or path to saved file
    """
    # Calculate average payoffs for matrix
    avg_coop = np.mean(cooperative_scores) if cooperative_scores else 0.0
    avg_selfish = np.mean(selfish_scores) if selfish_scores else 0.0
    
    # Create 2x2 payoff matrix visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Payoff matrix heatmap
    matrix = np.array([[avg_coop, avg_selfish], [avg_selfish, avg_coop]])
    im = ax1.imshow(matrix, cmap='RdYlBu', aspect='equal')
    
    # Add labels and values
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Cooperative', 'Selfish'])
    ax1.set_yticklabels(['Cooperative', 'Selfish'])
    ax1.set_title('Average Payoff Matrix')
    
    # Add value annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    # Score distribution comparison
    if cooperative_scores and selfish_scores:
        ax2.hist(cooperative_scores, alpha=0.7, label='Cooperative', color='blue', bins=10)
        ax2.hist(selfish_scores, alpha=0.7, label='Selfish', color='red', bins=10)
        ax2.set_xlabel('Scores')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution by Strategy')
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(save_path)
    else:
        # Return as string representation
        plt.close()
        return f"Payoff Matrix: Coop={avg_coop:.2f}, Selfish={avg_selfish:.2f}"


def extract_language_model_outputs(
    measurements: measurements_lib.Measurements,
    measurement_channels: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Extract language model generated content from measurements.
    
    Args:
        measurements: Measurements object containing simulation data
        measurement_channels: Dictionary mapping channel keys to names
        
    Returns:
        List of dictionaries containing language model outputs
    """
    lm_outputs = []
    
    # Extract individual agent decisions and reasoning
    individual_scores = measurements.get_channel(
        measurement_channels['individual_scores']
    )
    
    for entry in individual_scores:
        if 'reasoning' in entry or 'decision_context' in entry:
            lm_outputs.append({
                'type': 'agent_decision',
                'agent_name': entry.get('agent_name', 'unknown'),
                'strategy': entry.get('goal', 'unknown'),
                'score': entry.get('score', 0),
                'reasoning': entry.get('reasoning', 'No reasoning captured'),
                'timestamp': entry.get('timestamp', 'unknown')
            })
    
    return lm_outputs


def format_configuration_summary(config: evolutionary_types.EvolutionConfig) -> str:
    """Format configuration parameters as readable text.
    
    Args:
        config: Evolution configuration object
        
    Returns:
        Formatted string representation of configuration
    """
    lines = [
        "=== SIMULATION CONFIGURATION ===",
        f"Population Size: {config.pop_size}",
        f"Number of Generations: {config.num_generations}",
        f"Number of Rounds per Generation: {config.num_rounds}",
        f"Selection Method: {config.selection_method}",
        f"Top K Survivors: {config.top_k}",
        f"Mutation Rate: {config.mutation_rate:.3f}",
        "",
        "=== LANGUAGE MODEL CONFIGURATION ===",
        f"API Type: {config.api_type}",
        f"Model Name: {config.model_name}",
        f"Embedder: {config.embedder_name}",
        f"Device: {config.device}",
        f"Language Model Disabled: {config.disable_language_model}",
        ""
    ]
    
    if config.api_key:
        lines.append("API Key: [CONFIGURED]")
    else:
        lines.append("API Key: [NOT SET]")
    
    return "\n".join(lines)


def export_simulation_results(
    config: evolutionary_types.EvolutionConfig,
    measurements: measurements_lib.Measurements,
    measurement_channels: Dict[str, str],
    output_dir: Path,
    run_id: Optional[str] = None
) -> Tuple[Path, Path]:
    """Export complete simulation results to text file with visualizations.
    
    Args:
        config: Evolution configuration used for the simulation
        measurements: Measurements object containing all collected data
        measurement_channels: Dictionary mapping channel keys to names
        output_dir: Directory to save results
        run_id: Optional identifier for the run
        
    Returns:
        Tuple of (text_file_path, figure_path)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID if not provided
    if run_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"evolution_run_{timestamp}"
    
    text_file = output_dir / f"{run_id}_results.txt"
    figure_file = output_dir / f"{run_id}_payoff_matrix.png"
    
    # Collect all measurement data
    all_data = {}
    for channel_key, channel_name in measurement_channels.items():
        all_data[channel_key] = measurements.get_channel(channel_name)
    
    # Extract strategy-specific scores for payoff matrix
    cooperative_scores = []
    selfish_scores = []
    
    for entry in all_data.get('individual_scores', []):
        score = entry.get('score', 0)
        strategy = entry.get('goal', '')
        
        if 'COOPERATIVE' in strategy.upper():
            cooperative_scores.append(score)
        elif 'SELFISH' in strategy.upper():
            selfish_scores.append(score)
    
    # Create payoff matrix figure
    figure_path_str = create_payoff_matrix_figure(
        cooperative_scores, selfish_scores, figure_file
    )
    
    # Extract language model outputs
    lm_outputs = extract_language_model_outputs(measurements, measurement_channels)
    
    # Generate comprehensive text report
    with open(text_file, 'w') as f:
        f.write(f"EVOLUTIONARY SIMULATION RESULTS\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("=" * 60 + "\n\n")
        
        # Configuration section
        f.write(format_configuration_summary(config))
        f.write("\n")
        
        # Results summary
        final_gen_data = measurements.get_last_datum(
            measurement_channels['generation_summary']
        )
        convergence_data = measurements.get_last_datum(
            measurement_channels['convergence_metrics']
        )
        
        f.write("=== SIMULATION RESULTS SUMMARY ===\n")
        if final_gen_data:
            f.write(f"Final Cooperation Rate: {final_gen_data['cooperation_rate']:.3f}\n")
            f.write(f"Final Average Score: {final_gen_data['avg_score']:.2f}\n")
            f.write(f"Cooperative Agents: {final_gen_data['cooperative_agents']}\n")
            f.write(f"Selfish Agents: {final_gen_data['selfish_agents']}\n")
        
        if convergence_data:
            f.write(f"Converged to Cooperation: {convergence_data['converged_to_cooperation']}\n")
            f.write(f"Converged to Selfishness: {convergence_data['converged_to_selfishness']}\n")
        
        f.write(f"\nPayoff Analysis:\n")
        f.write(f"Average Cooperative Score: {np.mean(cooperative_scores):.2f}\n")
        f.write(f"Average Selfish Score: {np.mean(selfish_scores):.2f}\n")
        f.write(f"Cooperative Advantage: {np.mean(cooperative_scores) - np.mean(selfish_scores):.2f}\n")
        f.write(f"Payoff Matrix Figure: {figure_file.name}\n\n")
        
        # Detailed metrics by channel
        f.write("=== DETAILED MEASUREMENT DATA ===\n")
        for channel_key, channel_data in all_data.items():
            f.write(f"\n--- {channel_key.upper().replace('_', ' ')} ---\n")
            f.write(f"Total Entries: {len(channel_data)}\n")
            
            if channel_data:
                f.write("Latest Entry:\n")
                latest = channel_data[-1]
                for key, value in latest.items():
                    if isinstance(value, (list, dict)):
                        f.write(f"  {key}: {json.dumps(value, indent=2)}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        # Language model outputs section
        f.write("=== LANGUAGE MODEL OUTPUTS ===\n")
        if lm_outputs:
            f.write(f"Total LM Decision Entries: {len(lm_outputs)}\n\n")
            for i, output in enumerate(lm_outputs[:10]):  # Show first 10
                f.write(f"Decision {i+1}:\n")
                f.write(f"  Agent: {output['agent_name']}\n")
                f.write(f"  Strategy: {output['strategy']}\n")
                f.write(f"  Score: {output['score']}\n")
                f.write(f"  Reasoning: {output['reasoning']}\n")
                f.write(f"  Timestamp: {output['timestamp']}\n\n")
            
            if len(lm_outputs) > 10:
                f.write(f"... and {len(lm_outputs) - 10} more entries\n\n")
        else:
            f.write("No language model reasoning captured (likely using dummy model)\n\n")
        
        # Generation-by-generation evolution
        f.write("=== EVOLUTION TIMELINE ===\n")
        gen_summaries = all_data.get('generation_summary', [])
        for gen_data in gen_summaries:
            gen = gen_data.get('generation', 'unknown')
            coop_rate = gen_data.get('cooperation_rate', 0)
            avg_score = gen_data.get('avg_score', 0)
            f.write(f"Generation {gen}: Cooperation={coop_rate:.3f}, Avg Score={avg_score:.2f}\n")
        
        f.write(f"\n=== END OF REPORT ===\n")
    
    return text_file, figure_file


def save_simulation_results(
    config: evolutionary_types.EvolutionConfig,
    measurements: measurements_lib.Measurements,
    output_dir: str = "simulation_results",
    run_id: Optional[str] = None
) -> Tuple[Path, Path]:
    """Convenience function to save simulation results with default settings.
    
    Args:
        config: Evolution configuration
        measurements: Measurements from simulation
        output_dir: Directory name for results (default: "simulation_results")
        run_id: Optional run identifier
        
    Returns:
        Tuple of (text_file_path, figure_path)
    """
    from examples.evolutionary_simulation import MEASUREMENT_CHANNELS
    
    output_path = Path(output_dir)
    return export_simulation_results(
        config, measurements, MEASUREMENT_CHANNELS, output_path, run_id
    )


# Convenience function for direct usage
def quick_export(measurements: measurements_lib.Measurements, config: evolutionary_types.EvolutionConfig):
    """Quick export function for immediate use after running a simulation."""
    text_file, figure_file = save_simulation_results(config, measurements)
    print(f"Results saved to: {text_file}")
    print(f"Payoff matrix saved to: {figure_file}")
    return text_file, figure_file