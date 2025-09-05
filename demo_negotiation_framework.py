#!/usr/bin/env python3
"""Demonstration and testing of the negotiation framework.

This script demonstrates how to test the negotiation framework components
that are actually implemented, and shows how to run complete negotiations.
"""

import os
import sys
import hashlib
import numpy as np
from typing import Dict, List, Optional

# Add examples to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

# Import negotiation examples
import negotiation_examples

print("="*70)
print("  CONCORDIA NEGOTIATION FRAMEWORK DEMONSTRATION")
print("="*70)
print("\nThis demo shows how to test and use the negotiation framework.\n")


def test_basic_configurations():
    """Test that basic negotiation configurations can be created."""
    print("\n" + "-"*60)
    print("  Testing Basic Configuration Creation")
    print("-"*60)
    
    configs_to_test = [
        ("Basic Price Negotiation", negotiation_examples.create_basic_price_negotiation_config),
        ("Cross-Cultural Business", negotiation_examples.create_cross_cultural_business_negotiation_config),
        ("Multi-Party Resource", negotiation_examples.create_multi_party_resource_allocation_config),
        ("Adaptive Learning", negotiation_examples.create_adaptive_learning_negotiation_config),
        ("Information Asymmetry", negotiation_examples.create_information_asymmetry_scenario_config),
        ("Complete Capabilities", negotiation_examples.demonstrate_complete_negotiation_capabilities_config),
    ]
    
    successful = 0
    failed = 0
    
    for name, config_func in configs_to_test:
        try:
            print(f"\n✓ Testing {name}...")
            config = config_func(api_key=None)
            
            # Verify config has required attributes
            assert hasattr(config, 'default_premise')
            assert hasattr(config, 'default_max_steps')
            assert hasattr(config, 'prefabs')
            assert hasattr(config, 'instances')
            assert len(config.instances) >= 2  # At least 2 entities and 1 GM
            
            print(f"  ✅ {name} configuration created successfully")
            print(f"     - Premise: {config.default_premise[:50]}...")
            print(f"     - Max steps: {config.default_max_steps}")
            print(f"     - Instances: {len(config.instances)}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ {name} configuration failed: {e}")
            failed += 1
    
    print(f"\n📊 Configuration Test Results: {successful} passed, {failed} failed")
    return successful, failed


def test_negotiation_components():
    """Test individual negotiation components that are implemented."""
    print("\n" + "-"*60)
    print("  Testing Negotiation Components")
    print("-"*60)
    
    # Test the negotiation_strategy component which we know is implemented
    from concordia.prefabs.entity.negotiation.components import negotiation_strategy
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Cooperative Strategy
    try:
        print("\n✓ Testing Cooperative Strategy...")
        strategy = negotiation_strategy.CooperativeStrategy()
        opening = strategy.get_opening_position(100, 200)
        assert 100 < opening < 200
        print(f"  ✅ Cooperative strategy works (opening position: {opening:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Cooperative strategy failed: {e}")
        tests_failed += 1
    
    # Test 2: Competitive Strategy
    try:
        print("\n✓ Testing Competitive Strategy...")
        strategy = negotiation_strategy.CompetitiveStrategy()
        opening = strategy.get_opening_position(100, 200)
        assert opening > 200  # Should anchor high
        print(f"  ✅ Competitive strategy works (opening position: {opening:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Competitive strategy failed: {e}")
        tests_failed += 1
    
    # Test 3: Integrative Strategy
    try:
        print("\n✓ Testing Integrative Strategy...")
        strategy = negotiation_strategy.IntegrativeStrategy()
        opening = strategy.get_opening_position(100, 200)
        assert 100 < opening < 200
        print(f"  ✅ Integrative strategy works (opening position: {opening:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Integrative strategy failed: {e}")
        tests_failed += 1
    
    # Test 4: BasicNegotiationStrategy Component
    try:
        print("\n✓ Testing BasicNegotiationStrategy Component...")
        component = negotiation_strategy.BasicNegotiationStrategy(
            agent_name="TestAgent",
            negotiation_style='cooperative',
            reservation_value=100.0,
            target_value=200.0
        )
        context = component.get_strategic_context()
        assert "COOPERATIVE" in context
        assert "100.00" in context
        print(f"  ✅ BasicNegotiationStrategy component works")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ BasicNegotiationStrategy component failed: {e}")
        tests_failed += 1
    
    print(f"\n📊 Component Test Results: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed


def test_mock_negotiation():
    """Test a complete negotiation with mock models."""
    print("\n" + "-"*60)
    print("  Testing Mock Negotiation Simulation")
    print("-"*60)
    
    try:
        # Setup mock model and embedder
        model, embedder = negotiation_examples.setup_language_model_and_embedder(
            use_gpt=False,  # Use NoLanguageModel
            api_key=None
        )
        
        print("\n✓ Creating basic price negotiation configuration...")
        config = negotiation_examples.create_basic_price_negotiation_config(api_key=None)
        
        print("✓ Running negotiation simulation (3 steps)...")
        html_log, raw_log = negotiation_examples.run_negotiation_simulation(
            config, model, embedder, max_steps=3
        )
        
        # Verify simulation ran
        assert html_log is not None
        assert len(raw_log) > 0
        
        print(f"  ✅ Mock negotiation completed successfully!")
        print(f"     - Simulation steps: {len(raw_log)}")
        print(f"     - HTML log generated: {len(html_log)} characters")
        
        # Save the log
        with open('test_negotiation_log.html', 'w') as f:
            f.write(html_log)
        print(f"     - Log saved to: test_negotiation_log.html")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Mock negotiation failed: {e}")
        return False


def run_interactive_demo():
    """Run an interactive demonstration."""
    print("\n" + "-"*60)
    print("  Interactive Negotiation Demo")
    print("-"*60)
    
    print("\nSelect a negotiation scenario to demonstrate:")
    print("1. Basic Price Negotiation (Buyer vs Seller)")
    print("2. Cross-Cultural Business Negotiation")
    print("3. Multi-Party Resource Allocation")
    print("4. Adaptive Learning Negotiation")
    print("5. Information Asymmetry Scenario")
    print("6. Complete Capabilities Showcase")
    print("0. Run all tests (no simulation)")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    if choice == "0":
        # Run all tests
        config_passed, config_failed = test_basic_configurations()
        comp_passed, comp_failed = test_negotiation_components()
        mock_success = test_mock_negotiation()
        
        print("\n" + "="*60)
        print("  OVERALL TEST SUMMARY")
        print("="*60)
        print(f"✅ Configuration Tests: {config_passed} passed, {config_failed} failed")
        print(f"✅ Component Tests: {comp_passed} passed, {comp_failed} failed")
        print(f"✅ Mock Negotiation: {'Passed' if mock_success else 'Failed'}")
        
        total_passed = config_passed + comp_passed + (1 if mock_success else 0)
        total_tests = config_passed + config_failed + comp_passed + comp_failed + 1
        print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")
        
    elif choice in ["1", "2", "3", "4", "5", "6"]:
        # Run specific example
        print(f"\n🎭 Running Example {choice}...")
        
        try:
            # Get API key if available
            api_key = os.getenv("OPENAI_API_KEY")
            
            # Run the example
            html_log, raw_log = negotiation_examples.run_example_demonstration(
                int(choice),
                api_key
            )
            
            if html_log:
                print(f"\n✅ Negotiation simulation completed!")
                print(f"   Check negotiation_example_{choice}.html for details")
            
        except Exception as e:
            print(f"\n❌ Error running example: {e}")
            print("💡 This may be because the NoLanguageModel doesn't support full simulations")
            print("   Set OPENAI_API_KEY environment variable to use a real model")
    
    else:
        print("Invalid choice. Running all tests...")
        run_interactive_demo()


def main():
    """Main entry point."""
    print("\n🔍 Starting Negotiation Framework Testing...\n")
    
    # Check if running in automated mode
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("Running automated tests...")
        config_passed, config_failed = test_basic_configurations()
        comp_passed, comp_failed = test_negotiation_components()
        mock_success = test_mock_negotiation()
        
        # Exit with appropriate code
        all_passed = (config_failed == 0 and comp_failed == 0 and mock_success)
        sys.exit(0 if all_passed else 1)
    else:
        # Interactive mode
        run_interactive_demo()


if __name__ == "__main__":
    main()