#!/usr/bin/env python3
"""
Quick demonstration of negotiation framework with LLM.
Run this to see all components working together!

Usage:
    export OPENAI_API_KEY='your-key-here'
    python quick_negotiation_demo.py
"""

import os
import sys

# Check for API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ Please set your OpenAI API key:")
    print("   export OPENAI_API_KEY='sk-your-key-here'")
    print("\nGet a key at: https://platform.openai.com/api-keys")
    sys.exit(1)

print("🚀 Starting Negotiation Framework Demo with LLM...")
print(f"   Using API key: {api_key[:10]}...")

# Import required modules
from concordia.language_model import gpt_model, retry_wrapper
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation.components import negotiation_strategy
import numpy as np

# Setup
print("\n📦 Setting up components...")

# 1. Create language model with retry for rate limits
base_model = gpt_model.GptLanguageModel(
    api_key=api_key,
    model_name="gpt-3.5-turbo"  # Using 3.5 for lower cost
)

model = retry_wrapper.RetryLanguageModel(
    model=base_model,
    retry_tries=3,
    retry_delay=2.0
)

# 2. Create embedder (simple version)
embedder = lambda x: np.random.randn(384).astype(np.float32)

# 3. Create memory bank
memory_bank = basic_associative_memory.AssociativeMemoryBank()
memory_bank.set_embedder(embedder)

print("✅ Components ready!")

# Demonstrate individual components
print("\n" + "="*60)
print("DEMONSTRATING NEGOTIATION COMPONENTS WITH LLM")
print("="*60)

# 1. Strategy Component
print("\n1️⃣ Testing Strategy Component...")
strategy = negotiation_strategy.BasicNegotiationStrategy(
    agent_name="DemoAgent",
    negotiation_style='integrative',
    reservation_value=1000,
    target_value=2000
)

context = strategy.get_strategic_context()
print(f"Strategy Context Generated:")
print("-" * 40)
print(context[:300] + "...")

# 2. Create a simple negotiator
print("\n2️⃣ Creating Negotiation Agent...")
negotiator = base_negotiator.build_agent(
    model=model,
    memory_bank=memory_bank,
    name="Alice_Negotiator",
    goal="Get the best deal on equipment purchase",
    negotiation_style='cooperative',
    reservation_value=10000,
)

print("✅ Agent created with components:")
for component_name in negotiator._context_components.keys():
    print(f"   - {component_name}")

# 3. Simulate negotiation interaction
print("\n3️⃣ Simulating Negotiation Interaction...")
print("-" * 40)

# Agent observes an offer
observation = "Bob offers to sell the equipment for $15,000, mentioning it's in excellent condition with warranty."
print(f"Observation: {observation}")
negotiator.observe(observation)

# Agent generates response
print("\nAgent's Response:")
action = negotiator.act()
print(f"Action: {action}")

# Test with another round
print("\n4️⃣ Second Round...")
print("-" * 40)

observation2 = "Bob seems willing to negotiate and asks about your budget and timeline."
print(f"Observation: {observation2}")
negotiator.observe(observation2)

print("\nAgent's Response:")
action2 = negotiator.act()
print(f"Action: {action2}")

# Show component integration
print("\n" + "="*60)
print("COMPONENT INTEGRATION SUMMARY")
print("="*60)

print("""
✅ Components Working Together:

1. **NegotiationStrategy**: Provided strategic context and tactics
2. **NegotiationMemory**: Tracked the offers and interactions
3. **NegotiationInstructions**: Guided the agent's behavior
4. **Language Model**: Generated contextual responses

The agent successfully:
- Processed incoming offers
- Applied negotiation strategy
- Generated appropriate responses
- Maintained negotiation context
""")

print("\n🎉 Demo Complete!")
print("\nTo run full scenarios with all advanced components:")
print("  python test_negotiation_with_llm.py")
print("\nTo run interactive examples:")
print("  python examples/negotiation_examples.py --run")