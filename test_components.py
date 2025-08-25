#!/usr/bin/env python3
"""
Manual testing of individual negotiation components.
"""

from concordia.prefabs.game_master.negotiation.components import negotiation_modules
from concordia.prefabs.game_master.negotiation.components import gm_social_intelligence
from concordia.prefabs.game_master.negotiation.components import gm_cultural_awareness


def test_social_intelligence():
    """Test the social intelligence GM module."""
    print("🧠 Testing Social Intelligence Module...")
    
    # Create module
    social_intel = gm_social_intelligence.SocialIntelligenceGM()
    
    # Test emotion detection
    emotion = social_intel.detect_emotion(
        "I am really frustrated with this whole process!", 
        "Alice", 
        3
    )
    
    if emotion:
        print(f"   ✅ Detected emotion: {emotion.primary_emotion}")
        print(f"      Intensity: {emotion.intensity:.1f}")
        print(f"      Valence: {emotion.valence:.1f}")
    
    # Test deception detection
    social_intel.check_consistency("Bob", "I can pay up to $500", 1)
    deception = social_intel.check_consistency("Bob", "I cannot pay more than $300", 2)
    
    if deception:
        print(f"   ✅ Detected deception: {deception.indicator_type}")
        print(f"      Description: {deception.description}")


def test_cultural_awareness():
    """Test the cultural awareness GM module.""" 
    print("\n🌍 Testing Cultural Awareness Module...")
    
    # Create module
    cultural = gm_cultural_awareness.CulturalAwarenessGM()
    
    # Set participant cultures
    cultural.set_participant_culture("Alice", "western_business")
    cultural.set_participant_culture("Bob", "east_asian")
    
    # Test cultural violation detection
    violation = cultural.detect_cultural_violation(
        "Alice",
        "You are wrong about this pricing", 
        "Bob"
    )
    
    if violation:
        print(f"   ✅ Detected cultural violation: {violation}")
    
    # Test context for observer
    context = negotiation_modules.ModuleContext(
        negotiation_id='test',
        participants=['Alice', 'Bob'],
        current_phase='bargaining',
        current_round=3,
        active_modules={'Alice': set(), 'Bob': set()},
        shared_data={},
    )
    
    cultural_context = cultural.get_observation_context("Alice", context)
    print(f"   ✅ Cultural context for Alice:")
    print(f"      {cultural_context.strip()}")


def test_module_registry():
    """Test the module registry system."""
    print("\n📋 Testing Module Registry...")
    
    # List registered modules
    registry = negotiation_modules.NegotiationGMModuleRegistry
    modules = registry.list_modules()
    
    print(f"   ✅ Registered modules ({len(modules)}):")
    for module_name in sorted(modules):
        print(f"      - {module_name}")
    
    # Test creating modules
    social_module = registry.create_module('social_intelligence')
    if social_module:
        print(f"   ✅ Successfully created: {social_module.name}")


if __name__ == "__main__":
    print("🧪 Testing Individual Negotiation Components")
    print("=" * 50)
    
    test_social_intelligence()
    test_cultural_awareness() 
    test_module_registry()
    
    print("\n🎉 Component testing completed!")