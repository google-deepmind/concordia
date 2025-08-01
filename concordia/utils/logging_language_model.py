#!/usr/bin/env python3
"""
Logging Language Model Wrapper
Intercepts and logs all language model interactions for analysis.
"""

from collections.abc import Collection, Sequence
from typing import Any
import re
import logging

from concordia.language_model import language_model
from concordia.utils.generation_logger import get_generation_logger

logger = logging.getLogger(__name__)

class LoggingLanguageModel(language_model.LanguageModel):
    """Language model wrapper that logs all interactions."""
    
    def __init__(self, base_model: language_model.LanguageModel, agent_name: str = "Unknown"):
        self.base_model = base_model
        self.agent_name = agent_name
        self.generation_logger = get_generation_logger()
    
    def set_agent_name(self, agent_name: str):
        """Update the agent name for logging."""
        self.agent_name = agent_name
        
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = 1800,  # 30 minutes for long simulations
        seed: int | None = None,
    ) -> str:
        """Sample text and log the interaction."""
        
        # Try to extract agent name from prompt context
        dynamic_agent_name = self._extract_agent_name_from_prompt(prompt)
        agent_name = dynamic_agent_name if dynamic_agent_name else self.agent_name
        
        # Call the underlying model
        response = self.base_model.sample_text(
            prompt,
            max_tokens=max_tokens,
            terminators=terminators,
            temperature=temperature,
            timeout=timeout,
            seed=seed
        )
        
        # Extract action from response (look for Yes/No patterns)
        action = self._extract_action(response)
        
        # Log the interaction
        self.generation_logger.log_interaction(
            agent=agent_name,
            prompt=prompt,
            response=response,
            action=action,
            extra_data={
                'max_tokens': max_tokens,
                'temperature': temperature,
                'response_length': len(response),
                'word_count': len(response.split())
            }
        )
        
        logger.debug(f"Logged LLM interaction for {agent_name}: {len(response)} chars")
        
        return response
    
    def _extract_agent_name_from_prompt(self, prompt: str) -> str:
        """Extract agent name from the prompt context."""
        import re
        
        # Look for common patterns like "Agent_1", "Entity Agent_2", etc.
        agent_patterns = [
            r'\bAgent_(\d+)\b',
            r'\bEntity (Agent_\d+)\b',
            r'(\w+) is next to act',
            r'Entity (\w+) observed',
            r'Entity (\w+) chose action',
        ]
        
        for pattern in agent_patterns:
            match = re.search(pattern, prompt)
            if match:
                # Return the captured group (agent name)
                captured = match.group(1)
                if captured.startswith('Agent_'):
                    return captured
                elif captured.isdigit():
                    return f"Agent_{captured}"
                else:
                    return captured
        
        return None
    
    def _extract_action(self, response: str) -> str:
        """Extract the final action/decision from the response."""
        # Look for common patterns in evolutionary simulation responses
        response_lower = response.lower().strip()
        
        # Check for explicit Yes/No
        if 'yes' in response_lower and 'no' not in response_lower:
            return "Yes"
        elif 'no' in response_lower and 'yes' not in response_lower:
            return "No"
        elif 'contribute' in response_lower:
            return "Contribute"
        elif 'defect' in response_lower or 'keep' in response_lower:
            return "Defect"
        else:
            # Extract last meaningful word
            words = response.strip().split()
            if words:
                return words[-1]
            return "Unknown"
    
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> str:
        """Sample a choice and log it."""
        if hasattr(self.base_model, 'sample_choice'):
            result = self.base_model.sample_choice(prompt, responses, seed=seed)
            
            # Log the choice interaction
            self.generation_logger.log_interaction(
                agent=self.agent_name,
                prompt=prompt,
                response=f"Chose: {result}",
                action=result,
                extra_data={
                    'interaction_type': 'choice',
                    'available_options': list(responses),
                    'selected_option': result
                }
            )
            
            return result
        else:
            # Fallback to text sampling
            options_text = ", ".join(f"'{r}'" for r in responses)
            full_prompt = f"{prompt}\n\nOptions: {options_text}\n\nChoice:"
            response = self.sample_text(full_prompt, max_tokens=10)
            
            # Find best match
            for option in responses:
                if option.lower() in response.lower():
                    return option
            return responses[0]  # Default to first option

def wrap_language_model_with_logging(base_model: language_model.LanguageModel, 
                                   agent_name: str = "Unknown") -> LoggingLanguageModel:
    """Wrap a language model with logging capabilities."""
    return LoggingLanguageModel(base_model, agent_name)

class AgentLanguageModelManager:
    """Manages language model wrappers for multiple agents."""
    
    def __init__(self, base_model: language_model.LanguageModel):
        self.base_model = base_model
        self.agent_models = {}
        
    def get_agent_model(self, agent_name: str) -> LoggingLanguageModel:
        """Get or create a logging wrapper for an agent."""
        if agent_name not in self.agent_models:
            self.agent_models[agent_name] = LoggingLanguageModel(self.base_model, agent_name)
        return self.agent_models[agent_name]
    
    def set_generation(self, generation: int):
        """Update generation for all agent models."""
        get_generation_logger().set_generation(generation)
        
    def set_round(self, round_num: int):
        """Update round for all agent models.""" 
        get_generation_logger().set_round(round_num)