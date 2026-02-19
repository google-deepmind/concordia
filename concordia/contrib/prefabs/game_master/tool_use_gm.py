# Copyright 2024 DeepMind Technologies Limited.
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

"""GameMaster prefab that enables tool use via MCP protocol."""

import sys
from collections.abc import Sequence

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.tools import mcp_client
from concordia.typing import prefab as prefab_lib


def build_tool_use_game_master(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemoryBank,
    mcp_server_command: str,
    mcp_server_args: Sequence[str],
    clock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    name: str = 'GameMaster with Tool Support',
    update_thought_chain: (
        Sequence[thought_chains_lib.ThoughtChain] | None
    ) = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a GameMaster that can execute external tools via MCP.
    
    This GameMaster monitors agent actions and autonomously executes
    appropriate MCP tools when agents request external information.
    
    Args:
        model: Language model for the GM
        memory: Memory bank for the GM
        mcp_server_command: Command to start MCP server (e.g., sys.executable)
        mcp_server_args: Arguments for MCP server (e.g., path to server script)
        clock: Simulation clock
        players: List of player agents in the simulation
        name: Name for this GameMaster
        update_thought_chain: Optional custom thought chain
        
    Returns:
        Configured GameMaster entity with MCP tool support
        
    Example:
```python
        gm = build_tool_use_game_master(
            model=language_model,
            memory=memory_bank,
            mcp_server_command=sys.executable,
            mcp_server_args=['concordia/tools/mcp_servers/file_reader_server.py'],
            clock=clock,
            players=[alice, bob],
        )
```
    """
    # Initialize MCP client
    client = mcp_client.MCPClient(
        server_command=mcp_server_command,
        server_args=list(mcp_server_args),
    )
    
    # Create client manager
    manager = mcp_client.MCPClientManager()
    manager.add_client('file-tools', client)
    
    # Connect to MCP server
    mcp_client.run_sync(manager.connect_all())
    
    # Get available tools
    all_tools = mcp_client.run_sync(manager.list_all_tools())
    tool_descriptions = []
    for client_name, tools in all_tools.items():
        for tool in tools:
            tool_descriptions.append(
                f"- {tool['name']}: {tool['description']}"
            )
    
    tools_text = '\n'.join(tool_descriptions)
    
    # Build GM components
    instructions = gm_components.instructions.Instructions(
        agent_name=name,
        logging_channel=None,
    )
    
    # Custom instructions that mention tool availability
    tool_instructions = gm_components.instructions.Instructions(
        agent_name=name,
        logging_channel=None,
        pre_act_key='Available external tools',
    )
    tool_instructions.set_pre_act_value(
        f"""The following external tools are available for agents to use:

{tools_text}

When an agent's action indicates they want to use a tool (e.g., "I read the file", 
"I check the document"), automatically execute the appropriate tool and inject 
the result into the agent's observations.

Tool execution format:
- Detect: Agent says "I read /tmp/file.txt"
- Execute: read_file(path="/tmp/file.txt")
- Result: Inject file contents into next observation

Always execute tools when agents request external information."""
    )
    
    # Observation component
    observation = gm_components.observation.Observation(
        clock_now=clock.now,
        memory=memory.get_data_frame(),
        timeframe_delta_from=None,
        timeframe_delta_until=None,
        component_name='Recent events',
    )
    
    # Tool execution component
    from concordia.components.game_master import mcp_tool_executor
    
    tool_executor = mcp_tool_executor.MCPToolExecutor(
        model=model,
        mcp_client_manager=manager,
        memory=memory,
        component_name='Tool Executor',
    )
    
    # Relevant memories
    player_names = [player.name for player in players]
    
    relevant_memories = (
        gm_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components={
                gm_components.all_similar_memories.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            },
            num_memories_to_retrieve=10,
            pre_act_key='Relevant memories',
        )
    )
    
    # Player status
    player_status = gm_components.player_status.PlayerStatus(
        clock_now=clock.now,
        model=model,
        memory=memory,
        player_names=player_names,
    )
    
    # Conversation scene
    convo_scene = gm_components.conversation.Conversation(
        players=players,
        model=model,
        memory=memory,
        clock=clock,
        burner_memory_factory=lambda: associative_memory.AssociativeMemory(
            sentence_embedder=memory.get_embedder(),
        ),
        components={},
        cap_nonplayer_characters=3,
        game_master_instructions=(
            'This is a social science experiment. It is structured as a '
            'tabletop roleplaying game (like dungeons and dragons). You are '
            'the game master and storyteller. With tool support enabled, '
            'agents can access real external information during the simulation.'
        ),
        verbose=False,
    )
    
    # Component order
    entity_components = [
        instructions,
        tool_instructions,
        observation,
        relevant_memories,
        player_status,
        tool_executor,  # Tool executor monitors and acts on events
        convo_scene,
    ]
    
    components_of_agent = {
        _get_component_name(component): component 
        for component in entity_components
    }
    
    # Build the GameMaster entity
    gm_entity = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=convo_scene,
        context_components=components_of_agent,
        component_logging=None,
    )
    
    return gm_entity


def _get_component_name(component) -> str:
    """Helper to get component name."""
    if hasattr(component, 'name'):
        if callable(component.name):
            return component.name()
        return component.name
    return component.__class__.__name__
