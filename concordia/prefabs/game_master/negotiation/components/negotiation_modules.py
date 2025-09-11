# Copyright 2025 DeepMind Technologies Limited.
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

"""Base interface for modular negotiation game master components."""

import abc
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.typing import entity_component


@dataclasses.dataclass
class ModuleContext:
  """Context passed between GM modules."""
  negotiation_id: str
  participants: List[str]
  current_phase: str
  current_round: int
  active_modules: Dict[str, Set[str]]  # participant -> active modules
  shared_data: Dict[str, Any]


class NegotiationGMModule(entity_component.ContextComponent):
  """Base class for negotiation game master modules.

  Each module provides specialized functionality for managing
  specific aspects of advanced negotiations.
  """

  def __init__(
      self,
      name: str,
      priority: int = 50,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize the GM module.

    Args:
      name: Module name
      priority: Processing priority (higher = later)
      config: Module-specific configuration
    """
    self._module_name = name
    self._priority = priority
    self._config = config or {}
    self._enabled = True
    self._module_state: Dict[str, Any] = {}

  @abc.abstractmethod
  def get_supported_agent_modules(self) -> Set[str]:
    """Return set of agent modules this GM module supports."""
    pass

  @abc.abstractmethod
  def validate_action(
      self,
      actor: str,
      action: str,
      context: ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate an action based on module-specific rules.

    Returns:
      Tuple of (is_valid, error_message)
    """
    pass

  @abc.abstractmethod
  def update_state(
      self,
      event: str,
      actor: str,
      context: ModuleContext,
  ) -> None:
    """Update module state based on events."""
    pass

  @abc.abstractmethod
  def get_observation_context(
      self,
      observer: str,
      context: ModuleContext,
  ) -> str:
    """Get module-specific observation context for a participant."""
    pass

  @abc.abstractmethod
  def get_module_report(self) -> str:
    """Get a report of module activity and state."""
    pass

  def should_activate(self, context: ModuleContext) -> bool:
    """Determine if this module should be active given the context."""
    # Check if any participant has compatible modules
    for participant, modules in context.active_modules.items():
      if modules.intersection(self.get_supported_agent_modules()):
        return True
    return False

  def get_priority(self) -> int:
    """Get module processing priority."""
    return self._priority

  def is_enabled(self) -> bool:
    """Check if module is enabled."""
    return self._enabled

  def set_enabled(self, enabled: bool) -> None:
    """Enable or disable the module."""
    self._enabled = enabled

  def get_module_state(self, key: str) -> Any:
    """Get value from module state."""
    return self._module_state.get(key)

  def set_module_state(self, key: str, value: Any) -> None:
    """Set value in module state."""
    self._module_state[key] = value

  def pre_act(self, action_spec) -> str:
    """Provide module context before action."""
    if not self._enabled:
      return ""
    return self.get_module_report()

  def post_act(self, action_attempt: str) -> None:
    """Process module updates after action."""
    pass

  def pre_observe(self, observation: str) -> None:
    """Process observations."""
    pass

  def post_observe(self) -> None:
    """Post-observation processing."""
    pass

  def update(self) -> None:
    """Update internal state."""
    pass

  @property
  def name(self) -> str:
    """Component name."""
    return f'NegotiationGMModule_{self._module_name}'


class NegotiationGMModuleRegistry:
  """Registry for available GM modules."""

  _modules: Dict[str, type] = {}

  @classmethod
  def register(cls, name: str, module_class: type) -> None:
    """Register a GM module."""
    if not issubclass(module_class, NegotiationGMModule):
      raise ValueError(f"{module_class} must inherit from NegotiationGMModule")
    cls._modules[name] = module_class

  @classmethod
  def get_module(cls, name: str) -> Optional[type]:
    """Get a registered module class."""
    return cls._modules.get(name)

  @classmethod
  def list_modules(cls) -> List[str]:
    """List all registered modules."""
    return list(cls._modules.keys())

  @classmethod
  def create_module(
      cls,
      name: str,
      config: Optional[Dict[str, Any]] = None,
  ) -> Optional[NegotiationGMModule]:
    """Create an instance of a registered module."""
    module_class = cls.get_module(name)
    if module_class:
      return module_class(name=name, config=config)
    return None


def detect_agent_modules(agents: List[Any]) -> Dict[str, Set[str]]:
  """Detect which advanced modules each agent is using.

  Args:
    agents: List of negotiation agents

  Returns:
    Mapping from agent name to set of active module names
  """
  agent_modules = {}

  # Module component names that indicate advanced modules
  module_indicators = {
      'CulturalAdaptation': 'cultural_adaptation',
      'TemporalStrategy': 'temporal_strategy',
      'SwarmIntelligence': 'swarm_intelligence',
      'UncertaintyAware': 'uncertainty_aware',
      'StrategyEvolution': 'strategy_evolution',
      'TheoryOfMind': 'theory_of_mind',
  }

  for agent in agents:
    modules = set()

    # Check agent's components
    if hasattr(agent, '_context_components'):
      for comp_name in agent._context_components:
        if comp_name in module_indicators:
          modules.add(module_indicators[comp_name])

    agent_modules[agent.name] = modules

  return agent_modules


def suggest_gm_modules(agent_modules: Dict[str, Set[str]]) -> List[str]:
  """Suggest GM modules based on agent capabilities.

  Args:
    agent_modules: Mapping from agent name to active modules

  Returns:
    List of recommended GM module names
  """
  suggested = set()

  # Mapping from agent modules to corresponding GM modules
  module_mapping = {
      'cultural_adaptation': 'cultural_awareness',
      'temporal_strategy': 'temporal_dynamics',
      'swarm_intelligence': 'collective_intelligence',
      'uncertainty_aware': 'uncertainty_management',
      'strategy_evolution': 'strategy_evolution',
      'theory_of_mind': 'social_intelligence',
  }

  # Check all agent modules
  for agent, modules in agent_modules.items():
    for module in modules:
      if module in module_mapping:
        suggested.add(module_mapping[module])

  return list(suggested)
