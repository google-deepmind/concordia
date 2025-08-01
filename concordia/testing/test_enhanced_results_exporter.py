#!/usr/bin/env python3
"""
Test script for the enhanced results exporter.
"""

import datetime
import unittest
import tempfile
from pathlib import Path

from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter

def test_enhanced_results():
    """Test the enhanced results exporter with sample data."""
    print("🧪 Testing Enhanced Results Exporter...")

    exporter = EnhancedResultsExporter()

    # Sample data similar to what the simulation would produce
    sample_results_data = {
        'config': {
            'model_name': 'google/gemma-7b-it',
            'api_type': 'pytorch_gemma',
            'pop_size': 4,
            'num_generations': 3,
            'num_rounds': 6,
            'selection_method': 'topk',
            'mutation_rate': 0.1,
            'device': 'mps',
            'disable_language_model': False,
        },
        'generations': [
            {
                'generation': 1,
                'scores': {'Agent_1': 1.8, 'Agent_2': 1.8, 'Agent_3': 0.8, 'Agent_4': 0.8},
                'cooperative_count': 2,
                'selfish_count': 2,
                'cooperation_rate': 0.5,
                'avg_cooperative_score': 1.8,
                'avg_selfish_score': 0.8,
                'cooperative_agents': ['Agent_1', 'Agent_2'],
                'selfish_agents': ['Agent_3', 'Agent_4'],
            },
            {
                'generation': 2,
                'scores': {'Agent_1': 0.4, 'Agent_2': 1.4, 'Agent_3': 1.4, 'Agent_4': 1.4},
                'cooperative_count': 4,
                'selfish_count': 0,
                'cooperation_rate': 1.0,
                'avg_cooperative_score': 1.15,
                'avg_selfish_score': 0.0,
                'cooperative_agents': ['Agent_1', 'Agent_2', 'Agent_3', 'Agent_4'],
                'selfish_agents': [],
            },
            {
                'generation': 3,
                'scores': {'Agent_1': 1.6, 'Agent_2': 1.6, 'Agent_3': 1.6, 'Agent_4': 1.6},
                'cooperative_count': 3,
                'selfish_count': 1,
                'cooperation_rate': 0.75,
                'avg_cooperative_score': 1.6,
                'avg_selfish_score': 1.6,
                'cooperative_agents': ['Agent_1', 'Agent_2', 'Agent_3'],
                'selfish_agents': ['Agent_4'],
            }
        ],
        'final_cooperation_rate': 0.75,
        'performance': {
            'duration_seconds': 180.5,
            'avg_generation_time': 60.17,
        }
    }

    sample_analysis_data = {
        'config': sample_results_data['config'],
        'generations': sample_results_data['generations'],
        'final_cooperation_rate': 0.75,
        'evolutionary_insights': {
            'trend': 'stable_cooperation',
            'initial_cooperation': 0.5,
            'final_cooperation': 0.75,
            'cooperation_change': 0.25
        }
    }

    sample_metadata = {
        'simulation_info': {
            'model_name': 'google/gemma-7b-it',
            'api_type': 'pytorch_gemma',
            'device': 'mps',
            'language_model_active': True,
        },
        'performance': {
            'duration_seconds': 180.5,
            'avg_generation_time': 60.17,
        },
        'system_info': {
            'platform': 'Darwin',
            'python_version': '3.11.6',
            'torch_version': '2.7.1',
            'mps_available': True,
        },
        'experiment_parameters': {
            'population_size': 4,
            'generations': 3,
            'rounds_per_generation': 6,
            'selection_method': 'topk',
            'mutation_rate': 0.1,
        }
    }

    # Create sample generation log
    sample_generation_log = [
        {
            'timestamp': '10:23:45.123',
            'generation': 1,
            'round': 1,
            'agent': 'Agent_1',
            'prompt': 'You are participating in a public goods game. Will you contribute?',
            'response': 'I need to consider the group benefit versus my individual gain. Given that the pool is multiplied by 1.6 and shared equally, contributing would benefit everyone. However, if others don\'t contribute, I lose my investment. I choose to contribute to encourage cooperation.',
            'reasoning': 'Analyzing cooperation vs defection trade-offs',
            'action': 'Yes'
        },
        {
            'timestamp': '10:23:46.456',
            'generation': 1,
            'round': 1,
            'agent': 'Agent_2',
            'prompt': 'You are participating in a public goods game. Will you contribute?',
            'response': 'Looking at this game, I can see that if everyone contributes, we all benefit. But if I\'m the only one contributing, I lose money while others gain. I need to be strategic here.',
            'reasoning': 'Strategic thinking about free-rider problem',
            'action': 'No'
        },
        {
            'timestamp': '10:25:12.789',
            'generation': 2,
            'round': 1,
            'agent': 'Agent_1',
            'prompt': 'Your payoff last round was 1.80. Will you contribute this round?',
            'response': 'My cooperation paid off last round with a good payoff. This suggests others are also cooperating. I should continue this strategy to maintain group cooperation.',
            'reasoning': 'Positive reinforcement from previous cooperation',
            'action': 'Yes'
        }
    ]

    # Test the export
    try:
        results_folder = exporter.export_complete_results(
            model_name="test-gemma-7b-it",
            results_data=sample_results_data,
            analysis_data=sample_analysis_data,
            metadata=sample_metadata,
            generation_log=sample_generation_log
        )

        print(f"✅ Test successful! Results exported to: {results_folder}")
        print(f"📁 Folder structure created:")

        # List the created files
        for file_path in results_folder.iterdir():
            print(f"   - {file_path.name}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_results()
    if success:
        print("\n🎉 Enhanced results system is ready!")
    else:
        print("\n💥 Need to fix issues before deploying")
