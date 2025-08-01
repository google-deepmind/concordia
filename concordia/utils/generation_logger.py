#!/usr/bin/env python3
"""
Generation Logger for Evolutionary Simulations
Captures all language model interactions during simulation runs.
"""

import datetime
import threading
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GenerationLogger:
    """Captures and stores all language model interactions during simulations."""
    
    def __init__(self):
        self.interactions: List[Dict[str, Any]] = []
        self.current_generation = 1
        self.current_round = 1
        self.lock = threading.Lock()  # Thread-safe logging
        
    def set_generation(self, generation: int):
        """Update the current generation number."""
        with self.lock:
            self.current_generation = generation
            logger.debug(f"Generation logger: Now in generation {generation}")
    
    def set_round(self, round_num: int):
        """Update the current round number."""
        with self.lock:
            self.current_round = round_num
            logger.debug(f"Generation logger: Now in round {round_num}")
            
    def log_interaction(self, agent: str, prompt: str = "", response: str = "", 
                       reasoning: str = "", action: str = "", extra_data: Optional[Dict] = None):
        """Log a single LLM interaction."""
        with self.lock:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            interaction = {
                'timestamp': timestamp,
                'generation': self.current_generation,
                'round': self.current_round,
                'agent': agent,
                'prompt': prompt.strip() if prompt else "",
                'response': response.strip() if response else "",
                'reasoning': reasoning.strip() if reasoning else "",
                'action': str(action).strip() if action else "",
            }
            
            # Add any extra data
            if extra_data:
                interaction.update(extra_data)
                
            self.interactions.append(interaction)
            
            logger.debug(f"Logged interaction for {agent} in Gen {self.current_generation}, Round {self.current_round}")
    
    def log_agent_decision(self, agent: str, prompt: str, full_response: str, 
                          final_action: str, reasoning: str = ""):
        """Log a complete agent decision-making process."""
        self.log_interaction(
            agent=agent,
            prompt=prompt,
            response=full_response,
            reasoning=reasoning,
            action=final_action,
            extra_data={'interaction_type': 'decision'}
        )
    
    def log_agent_observation(self, agent: str, observation: str):
        """Log what an agent observes."""
        self.log_interaction(
            agent=agent,
            prompt=observation,
            extra_data={'interaction_type': 'observation'}
        )
    
    def log_game_master_action(self, action_description: str, context: str = ""):
        """Log game master actions."""
        self.log_interaction(
            agent="GameMaster",
            prompt=context,
            response=action_description,
            extra_data={'interaction_type': 'game_master'}
        )
    
    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Get all logged interactions."""
        with self.lock:
            return self.interactions.copy()
    
    def get_generation_interactions(self, generation: int) -> List[Dict[str, Any]]:
        """Get interactions for a specific generation."""
        with self.lock:
            return [i for i in self.interactions if i.get('generation') == generation]
    
    def get_agent_interactions(self, agent: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific agent."""
        with self.lock:
            return [i for i in self.interactions if i.get('agent') == agent]
    
    def clear_log(self):
        """Clear all logged interactions."""
        with self.lock:
            self.interactions.clear()
            logger.info("Generation log cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.lock:
            if not self.interactions:
                return {'total_interactions': 0}
            
            total_interactions = len(self.interactions)
            
            # Count by generation
            gen_counts = {}
            agent_counts = {}
            interaction_types = {}
            
            total_words = 0
            total_chars = 0
            
            for interaction in self.interactions:
                # Generation counts
                gen = interaction.get('generation', 'Unknown')
                gen_counts[gen] = gen_counts.get(gen, 0) + 1
                
                # Agent counts
                agent = interaction.get('agent', 'Unknown')
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
                
                # Interaction type counts
                int_type = interaction.get('interaction_type', 'unknown')
                interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
                
                # Text statistics
                response = interaction.get('response', '')
                total_chars += len(response)
                total_words += len(response.split())
            
            return {
                'total_interactions': total_interactions,
                'generations': dict(sorted(gen_counts.items())),
                'agents': dict(sorted(agent_counts.items())),
                'interaction_types': interaction_types,
                'text_stats': {
                    'total_characters': total_chars,
                    'total_words': total_words,
                    'avg_words_per_interaction': total_words / max(total_interactions, 1)
                }
            }

# Global instance for easy access
_global_logger = None

def get_generation_logger() -> GenerationLogger:
    """Get the global generation logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = GenerationLogger()
    return _global_logger

def reset_generation_logger():
    """Reset the global generation logger."""
    global _global_logger
    _global_logger = GenerationLogger()
    
def log_llm_interaction(agent: str, prompt: str = "", response: str = "", 
                       reasoning: str = "", action: str = "", **kwargs):
    """Convenient function to log LLM interactions."""
    logger_instance = get_generation_logger()
    logger_instance.log_interaction(agent, prompt, response, reasoning, action, kwargs)