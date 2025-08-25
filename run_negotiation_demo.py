#!/usr/bin/env python3
"""
Simple demo script showing how to run a negotiation manually.
"""

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.clocks import game_clock
from concordia.language_model import no_language_model  # For testing without API

# Import negotiation prefabs
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


def run_simple_negotiation():
    """Run a simple bilateral negotiation between buyer and seller."""
    print("🤝 Setting up a simple price negotiation...")
    
    # Set up language model and clock
    model = no_language_model.NoLanguageModel()
    clock = game_clock.FixedIntervalClock()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()
    
    # Create buyer agent (wants lower price)
    print("👤 Creating buyer agent (competitive style)...")
    buyer = base_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="Alice_Buyer",
        goal="Purchase the vintage guitar at the lowest possible price",
        negotiation_style="competitive", 
        reservation_value=800.0,  # Won't pay more than $800
        ethical_constraints="Be firm but fair. Don't lie about budget.",
    )
    
    # Create seller agent (wants higher price) 
    print("👤 Creating seller agent (cooperative style)...")
    seller = base_negotiator.build_agent(
        model=model,
        memory_bank=memory_bank,
        name="Bob_Seller",
        goal="Sell the vintage guitar at a fair market price", 
        negotiation_style="cooperative",
        reservation_value=600.0,  # Won't sell for less than $600
        ethical_constraints="Be honest about the guitar's condition and value",
    )
    
    # Create game master
    print("🎯 Creating negotiation game master...")
    gm = negotiation.build_bilateral_negotiation(
        model=model,
        memory_bank=memory_bank,
        entities=[buyer, seller],
        name="Guitar Price Negotiation",
        max_rounds=8,
    )
    
    print(f"✅ Negotiation setup complete!")
    print(f"   Buyer: {buyer._agent_name} (reservation: $800)")
    print(f"   Seller: {seller._agent_name} (reservation: $600)")
    print(f"   Game Master: {gm._agent_name}")
    print(f"   ZOPA (Zone of Agreement): $600-$800")
    
    # Show the negotiation strategy contexts
    print("\n" + "="*60)
    print("STRATEGIC CONTEXTS:")
    print("="*60)
    
    # Get buyer's strategy context
    buyer_context = None
    seller_context = None
    
    # Look for negotiation strategy components
    for component_name, component in buyer._context_components.items():
        if hasattr(component, 'get_strategic_context'):
            print(f"\n🧠 {buyer._agent_name}'s Strategy:")
            print("-" * 40)
            buyer_context = component.get_strategic_context()
            print(buyer_context)
            break
    
    for component_name, component in seller._context_components.items():
        if hasattr(component, 'get_strategic_context'):
            print(f"\n🧠 {seller._agent_name}'s Strategy:")
            print("-" * 40) 
            seller_context = component.get_strategic_context()
            print(seller_context)
            break
    
    print("\n" + "="*60)
    print("SIMULATION READY!")
    print("="*60)
    print("""
To run a full simulation, you would need to:

1. Set up a proper language model (GPT, Claude, etc.)
2. Use Concordia's simulation environment to run rounds
3. Each round, agents would:
   - Observe the current situation
   - Use their negotiation strategy to decide action
   - Make offers, counteroffers, or accept/reject
   - Game master tracks state and enforces rules

The negotiation would continue until:
- Agreement reached (both accept an offer)
- Maximum rounds exceeded  
- One party walks away
""")

    return buyer, seller, gm


if __name__ == "__main__":
    try:
        buyer, seller, gm = run_simple_negotiation()
        print("\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()