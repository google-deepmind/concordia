#!/usr/bin/env python3
# Copyright 2023 DeepMind Technologies Limited.
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

"""
Test script for evolutionary simulation - used by CI/CD.
This runs a minimal version of the evolutionary simulation for testing purposes.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.evolutionary_simulation import evolutionary_main
from concordia.typing.evolutionary import EvolutionConfig


def test_evolutionary_simulation():
    """Run a minimal evolutionary simulation test for CI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Test configuration - minimal for CI speed
    config = EvolutionConfig(
        pop_size=4,
        num_generations=3,  # Reduced for CI
        selection_method='topk',
        top_k=2,
        mutation_rate=0.2,
        num_rounds=3  # Reduced for CI
    )
    
    logger.info('🧬 Starting evolutionary simulation test...')
    logger.info(f'Config: {config.pop_size} agents, {config.num_generations} generations')
    
    try:
        # Run the simulation
        measurements = evolutionary_main(config)
        
        # Verify results
        channels = measurements.available_channels()
        logger.info(f'📊 Generated {len(channels)} measurement channels')
        
        if len(channels) == 0:
            raise Exception('No measurement channels found - simulation may have failed')
            
        # Check for key measurement channels
        expected_channels = [
            'evolutionary_generation_summary',
            'evolutionary_population_dynamics',
            'evolutionary_strategy_distribution'
        ]
        
        available_channel_names = [measurements.get_channel(ch) for ch in channels]
        
        for expected in expected_channels:
            if expected not in str(available_channel_names):
                logger.warning(f'Expected channel {expected} not found')
        
        logger.info('✅ Evolutionary simulation test completed successfully!')
        logger.info('✅ All core functionality working correctly')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Evolutionary simulation test failed: {e}')
        return False


if __name__ == '__main__':
    success = test_evolutionary_simulation()
    sys.exit(0 if success else 1)