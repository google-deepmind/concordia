#!/usr/bin/env python3
"""
Enhanced Results Exporter for Evolutionary Simulations
Creates structured results with model-specific folders and comprehensive analysis.
"""

import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedResultsExporter:
    """Enhanced results exporter with structured folders and detailed analysis."""
    
    def __init__(self, base_results_dir: str = "simulation_results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
    def create_results_folder(self, model_name: str, timestamp: Optional[str] = None) -> Path:
        """Create a structured results folder for this simulation run."""
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
        # Sanitize model name for folder name
        safe_model_name = model_name.replace("/", "-").replace(" ", "-")
        folder_name = f"{safe_model_name}_{timestamp}"
        
        results_folder = self.base_results_dir / folder_name
        results_folder.mkdir(exist_ok=True)
        
        logger.info(f"Created results folder: {results_folder}")
        return results_folder
        
    def save_payoff_matrix(self, results_folder: Path, payoff_data: Dict[str, Any]):
        """Save payoff matrix in structured text format."""
        payoff_file = results_folder / "payoff_matrix.txt"
        
        with open(payoff_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EVOLUTIONARY SIMULATION PAYOFF MATRIX\n")
            f.write("=" * 60 + "\n\n")
            
            # Extract generation data
            generations = payoff_data.get('generations', [])
            
            for gen_idx, gen_data in enumerate(generations, 1):
                f.write(f"GENERATION {gen_idx}:\n")
                f.write("-" * 40 + "\n")
                
                # Agent scores
                scores = gen_data.get('scores', {})
                f.write("Agent Scores:\n")
                for agent, score in scores.items():
                    f.write(f"  {agent}: {score:.2f}\n")
                
                # Strategy distribution
                cooperative = gen_data.get('cooperative_count', 0)
                selfish = gen_data.get('selfish_count', 0)
                f.write(f"\nStrategy Distribution:\n")
                f.write(f"  Cooperative: {cooperative}\n")
                f.write(f"  Selfish: {selfish}\n")
                f.write(f"  Cooperation Rate: {gen_data.get('cooperation_rate', 0):.2%}\n")
                
                # Average scores by strategy
                coop_avg = gen_data.get('avg_cooperative_score', 0)
                selfish_avg = gen_data.get('avg_selfish_score', 0)
                f.write(f"\nAverage Scores by Strategy:\n")
                f.write(f"  Cooperative Agents: {coop_avg:.2f}\n")
                f.write(f"  Selfish Agents: {selfish_avg:.2f}\n\n")
                
        logger.info(f"Saved payoff matrix to: {payoff_file}")
        
    def save_simulation_results(self, results_folder: Path, results_data: Dict[str, Any]):
        """Save basic simulation results in text format.""" 
        results_file = results_folder / "simulation_results.txt"
        
        with open(results_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EVOLUTIONARY SIMULATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic simulation info
            f.write("SIMULATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            config = results_data.get('config', {})
            f.write(f"Model: {config.get('model_name', 'Unknown')}\n")
            f.write(f"Population Size: {config.get('pop_size', 'Unknown')}\n")
            f.write(f"Generations: {config.get('num_generations', 'Unknown')}\n")
            f.write(f"Rounds per Generation: {config.get('num_rounds', 'Unknown')}\n")
            f.write(f"Selection Method: {config.get('selection_method', 'Unknown')}\n")
            f.write(f"Mutation Rate: {config.get('mutation_rate', 'Unknown')}\n\n")
            
            # Final results
            f.write("FINAL RESULTS:\n")
            f.write("-" * 30 + "\n")
            final_coop_rate = results_data.get('final_cooperation_rate', 0)
            f.write(f"Final Cooperation Rate: {final_coop_rate:.2%}\n")
            
            total_generations = len(results_data.get('generations', []))
            f.write(f"Total Generations Completed: {total_generations}\n")
            
            # Performance metrics
            perf = results_data.get('performance', {})
            f.write(f"Simulation Duration: {perf.get('duration_seconds', 'Unknown')}s\n")
            f.write(f"Average Generation Time: {perf.get('avg_generation_time', 'Unknown')}s\n")
            
        logger.info(f"Saved simulation results to: {results_file}")
        
    def save_detailed_analysis(self, results_folder: Path, analysis_data: Dict[str, Any]):
        """Save detailed simulation state analysis - the comprehensive breakdown."""
        analysis_file = results_folder / "detailed_analysis.txt"
        
        with open(analysis_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 DETAILED EVOLUTIONARY SIMULATION STATE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            # Model and setup info
            f.write("🚀 MODEL AND SETUP:\n")
            f.write("-" * 40 + "\n")
            config = analysis_data.get('config', {})
            f.write(f"Language Model: {config.get('model_name', 'Unknown')}\n")
            f.write(f"Model Type: {config.get('api_type', 'Unknown')}\n")
            f.write(f"Device: {config.get('device', 'Unknown')}\n")
            f.write(f"Embedder: {config.get('embedder_name', 'Unknown')}\n")
            f.write(f"Language Model Active: {not config.get('disable_language_model', True)}\n\n")
            
            # Generation-by-generation analysis
            generations = analysis_data.get('generations', [])
            
            for gen_idx, gen_data in enumerate(generations, 1):
                f.write(f"🧬 GENERATION {gen_idx} ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                
                # Strategy distribution
                coop_count = gen_data.get('cooperative_count', 0)
                selfish_count = gen_data.get('selfish_count', 0)
                coop_rate = gen_data.get('cooperation_rate', 0)
                
                f.write(f"Strategy Distribution: {coop_count} Cooperative, {selfish_count} Selfish\n")
                f.write(f"Cooperation Rate: {coop_rate:.1%}\n")
                
                # Behavioral insights
                if gen_idx == 1:
                    if coop_rate == 0.5:
                        f.write("🔍 Initial Exploration: Agents testing both cooperative and selfish strategies\n")
                    elif coop_rate > 0.5:
                        f.write("🔍 Cooperative Bias: Population shows initial preference for cooperation\n") 
                    else:
                        f.write("🔍 Selfish Bias: Population shows initial preference for self-interest\n")
                elif gen_idx > 1:
                    prev_rate = generations[gen_idx-2].get('cooperation_rate', 0)
                    if coop_rate > prev_rate:
                        f.write("🔍 Cooperation Increasing: Selection pressure favoring cooperative strategies\n")
                    elif coop_rate < prev_rate:
                        f.write("🔍 Cooperation Decreasing: Selection pressure favoring selfish strategies\n")
                    else:
                        f.write("🔍 Stable Equilibrium: Cooperation rate maintained from previous generation\n")
                
                # Payoff analysis
                scores = gen_data.get('scores', {})
                f.write(f"\nPayoff Distribution:\n")
                for agent, score in scores.items():
                    strategy = "Cooperative" if agent in gen_data.get('cooperative_agents', []) else "Selfish"
                    f.write(f"  {agent}: {score:.2f} ({strategy})\n")
                
                # Strategic insights
                coop_avg = gen_data.get('avg_cooperative_score', 0)
                selfish_avg = gen_data.get('avg_selfish_score', 0)
                
                if coop_avg > selfish_avg:
                    f.write(f"💡 Cooperation Advantage: Cooperative agents outperformed (avg {coop_avg:.2f} vs {selfish_avg:.2f})\n")
                elif selfish_avg > coop_avg:
                    f.write(f"💡 Selfish Advantage: Free-riding was beneficial (avg {selfish_avg:.2f} vs {coop_avg:.2f})\n")
                else:
                    f.write(f"💡 Balanced Payoffs: Equal rewards regardless of strategy (avg {coop_avg:.2f})\n")
                
                f.write("\n")
            
            # Overall evolutionary dynamics
            f.write("🎯 EVOLUTIONARY DYNAMICS SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            if len(generations) >= 3:
                initial_rate = generations[0].get('cooperation_rate', 0)
                final_rate = generations[-1].get('cooperation_rate', 0)
                
                if final_rate > initial_rate:
                    f.write("📈 Evolution Toward Cooperation: Population evolved to be more cooperative\n")
                elif final_rate < initial_rate:
                    f.write("📉 Evolution Toward Selfishness: Population evolved to be more selfish\n")
                else:
                    f.write("➡️ Stable Evolution: Population maintained consistent cooperation levels\n")
                
                f.write(f"Initial Cooperation: {initial_rate:.1%} → Final Cooperation: {final_rate:.1%}\n")
                
            # Game theory insights
            f.write(f"\n🎮 GAME THEORY INSIGHTS:\n")
            f.write("-" * 50 + "\n")
            f.write("Public Goods Game Dynamics:\n")
            f.write("- Individual Contribution: 1 unit to common pool\n") 
            f.write("- Pool Multiplier: 1.6x total contributions\n")
            f.write("- Distribution: Equal shares regardless of contribution\n")
            f.write("- Nash Equilibrium: All agents contribute nothing (selfish)\n")
            f.write("- Pareto Optimal: All agents contribute (cooperative)\n")
            
            final_rate = analysis_data.get('final_cooperation_rate', 0)
            if final_rate > 0.8:
                f.write("✅ Result: Population achieved near-Pareto optimal cooperation\n")
            elif final_rate > 0.5:
                f.write("🔄 Result: Population found mixed equilibrium favoring cooperation\n")
            elif final_rate > 0.2:
                f.write("⚖️ Result: Population settled on balanced mixed strategy\n")
            else:
                f.write("❌ Result: Population converged near Nash equilibrium (mostly selfish)\n")
                
        logger.info(f"Saved detailed analysis to: {analysis_file}")
        
    def save_generation_log(self, results_folder: Path, generation_log: List[Dict[str, Any]]):
        """Save complete log of all LLM-generated text during the simulation."""
        log_file = results_folder / "generation_log.txt"
        
        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🎯 COMPLETE SIMULATION GENERATION LOG\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This file contains every piece of text generated by the language model\n")
            f.write("during the evolutionary simulation, organized by generation and round.\n\n")
            
            current_generation = None
            current_round = None
            
            for entry in generation_log:
                generation = entry.get('generation', 'Unknown')
                round_num = entry.get('round', 'Unknown')
                agent = entry.get('agent', 'Unknown')
                timestamp = entry.get('timestamp', 'Unknown')
                prompt = entry.get('prompt', '')
                response = entry.get('response', '')
                reasoning = entry.get('reasoning', '')
                action = entry.get('action', '')
                
                # Generation header
                if generation != current_generation:
                    if current_generation is not None:
                        f.write("\n" + "="*60 + "\n")
                    f.write(f"🧬 GENERATION {generation}\n")
                    f.write("="*60 + "\n\n")
                    current_generation = generation
                    current_round = None
                
                # Round header  
                if round_num != current_round:
                    if current_round is not None:
                        f.write("\n" + "-"*40 + "\n")
                    f.write(f"⚡ Round {round_num}\n")
                    f.write("-"*40 + "\n\n")
                    current_round = round_num
                
                # Agent interaction
                f.write(f"🤖 {agent} [{timestamp}]\n")
                f.write("~" * 30 + "\n")
                
                if prompt:
                    f.write("💭 PROMPT RECEIVED:\n")
                    f.write(f"{prompt}\n\n")
                
                if reasoning:
                    f.write("🧠 REASONING PROCESS:\n")
                    f.write(f"{reasoning}\n\n")
                
                if response:
                    f.write("💬 FULL RESPONSE:\n")
                    f.write(f"{response}\n\n")
                
                if action:
                    f.write("⚡ FINAL ACTION:\n")
                    f.write(f"{action}\n\n")
                
                f.write("~" * 30 + "\n\n")
            
            # Summary statistics
            f.write("\n" + "="*80 + "\n")
            f.write("📊 GENERATION LOG STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            total_interactions = len(generation_log)
            f.write(f"Total LLM Interactions: {total_interactions}\n")
            
            # Count by generation
            gen_counts = {}
            agent_counts = {}
            for entry in generation_log:
                gen = entry.get('generation', 'Unknown')
                agent = entry.get('agent', 'Unknown')
                gen_counts[gen] = gen_counts.get(gen, 0) + 1
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            f.write("\nInteractions by Generation:\n")
            for gen, count in sorted(gen_counts.items()):
                f.write(f"  Generation {gen}: {count} interactions\n")
            
            f.write("\nInteractions by Agent:\n")
            for agent, count in sorted(agent_counts.items()):
                f.write(f"  {agent}: {count} interactions\n")
            
            # Calculate total text generated
            total_chars = sum(len(entry.get('response', '')) for entry in generation_log)
            total_words = sum(len(entry.get('response', '').split()) for entry in generation_log)
            
            f.write(f"\nText Generation Statistics:\n")
            f.write(f"  Total Characters Generated: {total_chars:,}\n")
            f.write(f"  Total Words Generated: {total_words:,}\n")
            f.write(f"  Average Words per Interaction: {total_words/max(total_interactions, 1):.1f}\n")
                
        logger.info(f"Saved generation log to: {log_file}")
        
    def save_metadata(self, results_folder: Path, metadata: Dict[str, Any]):
        """Save simulation metadata as JSON."""
        metadata_file = results_folder / "metadata.json"
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.datetime.now().isoformat()
            
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Saved metadata to: {metadata_file}")
        
    def export_complete_results(self, model_name: str, results_data: Dict[str, Any], 
                              analysis_data: Dict[str, Any], metadata: Dict[str, Any],
                              generation_log: Optional[List[Dict[str, Any]]] = None) -> Path:
        """Export all results in the new structured format."""
        
        # Create results folder
        results_folder = self.create_results_folder(model_name)
        
        # Save all components
        self.save_payoff_matrix(results_folder, results_data)
        self.save_simulation_results(results_folder, results_data)
        self.save_detailed_analysis(results_folder, analysis_data)
        self.save_metadata(results_folder, metadata)
        
        # Save generation log if available
        if generation_log:
            self.save_generation_log(results_folder, generation_log)
        else:
            logger.warning("No generation log provided - skipping generation_log.txt")
        
        logger.info(f"✅ Complete results exported to: {results_folder}")
        return results_folder