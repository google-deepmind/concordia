"""
Example scripts demonstrating the negotiation prefab framework.

This file contains practical examples showing how to use the negotiation
agents and game masters for different scenarios using the proper Concordia
configuration-driven approach.
"""

import hashlib
import numpy as np
from typing import List, Dict, Any

# Optional import for sentence transformers
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Core Concordia imports
from concordia.language_model import gpt_model
from concordia.language_model import no_language_model
from concordia.language_model import retry_wrapper
from concordia.language_model import call_limit_wrapper
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

# Import all prefab modules for discovery
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs


def create_simple_embedder():
    """Create a simple embedder function for memory operations."""
    def simple_embedder(text):
        # Simple deterministic embedding based on text hash
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        # Convert to 384-dim vector (common embedding size)
        np.random.seed(hash_val)
        return np.random.randn(384).astype(np.float32)
    
    return simple_embedder


def setup_language_model_and_embedder(use_gpt=True, api_key=None):
    """Set up language model and embedder with rate limiting protection."""
    if use_gpt and api_key:
        # Create base GPT model
        base_model = gpt_model.GptLanguageModel(api_key=api_key, model_name="gpt-5")
        
        # Wrap with retry logic for rate limit handling
        retry_model = retry_wrapper.RetryLanguageModel(
            model=base_model,
            retry_on_exceptions=(Exception,),  # This includes openai.RateLimitError
            retry_tries=10,  # More retries for persistent rate limits
            retry_delay=30.0,  # Wait 30 seconds for GPT-5's 3 RPM limit (longer than 20s window)
            jitter=(5.0, 10.0),  # Longer jitter to spread requests
            exponential_backoff=True,
            backoff_factor=2.0,  # More aggressive backoff
            max_delay=300.0,  # Max 5 minute delay for persistent limits
        )
        
        # Optional: Add call limiting for cost control
        model = call_limit_wrapper.CallLimitLanguageModel(
            model=retry_model,
            max_calls=200,  # Reasonable limit for negotiation examples
        )
        
        # Use real sentence transformer for better embeddings with GPT if available
        if HAS_SENTENCE_TRANSFORMERS:
            st_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            embedder = lambda x: st_model.encode(x, show_progress_bar=False)
        else:
            print("📝 Note: sentence_transformers not available, using simple embedder")
            embedder = create_simple_embedder()
    else:
        model = no_language_model.NoLanguageModel()
        embedder = create_simple_embedder()
    
    return model, embedder


def create_basic_price_negotiation_config(api_key=None):
    """
    Example 1: Simple bilateral price negotiation between buyer and seller.

    This demonstrates the most basic negotiation setup using base negotiators
    with different styles and reservation values.
    """
    print("=== Example 1: Basic Price Negotiation ===")
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    # Define entities using InstanceConfig
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Alice_Buyer',
                'goal': 'Purchase the item at the lowest possible price',
                'traits': ['competitive', 'price-conscious', 'analytical'],
                'backstory': 'Alice is an experienced buyer who knows market values and negotiates firmly but fairly.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity', 
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Bob_Seller',
                'goal': 'Sell the item at a fair market price',
                'traits': ['cooperative', 'honest', 'relationship-focused'],
                'backstory': 'Bob is a seller who values long-term relationships and believes in fair pricing.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Price Negotiation Facilitator',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'negotiation setup',
                'next_game_master_name': 'Price Negotiation Facilitator',
                'shared_memories': [
                    'Alice wants to buy a laptop computer for her business.',
                    'Bob is selling a high-quality laptop computer.',
                    'The laptop is in excellent condition and worth approximately $1800.',
                    'Alice has budgeted up to $2000 but wants the best deal possible.',
                    'Bob would prefer to get at least $1500 but is open to negotiation.',
                    'This is a direct negotiation between buyer and seller.',
                ],
                'player_specific_memories': {
                    'Alice_Buyer': [
                        'You have researched similar laptops and know the market price.',
                        'Your maximum budget is $2000 but you want to pay less.',
                        'You prefer a competitive but fair negotiation approach.'
                    ],
                    'Bob_Seller': [
                        'You want to sell for at least $1500 to make a reasonable profit.', 
                        'You value maintaining good customer relationships.',
                        'You are willing to be flexible on price for a quick sale.'
                    ]
                },
            },
        ),
    ]
    
    # Create simulation config
    config = prefab_lib.Config(
        default_premise='Alice and Bob are meeting to negotiate the sale of a laptop computer. Both parties want to reach a fair agreement.',
        default_max_steps=15,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Basic price negotiation configuration created!")
    return config


def create_cross_cultural_business_negotiation_config(api_key=None):
    """
    Example 2: Cross-cultural business negotiation.

    Demonstrates negotiation with cultural considerations for
    international business scenarios.
    """
    print("\n=== Example 2: Cross-Cultural Business Negotiation ===")
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    # Define culturally diverse entities
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Sarah_Western',
                'goal': 'Negotiate a supply contract with favorable terms and quick resolution',
                'traits': ['direct', 'time-conscious', 'results-oriented', 'competitive'],
                'backstory': 'Sarah represents a Western company and prefers direct communication, quick decisions, and competitive negotiation tactics.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY, 
            params={
                'name': 'Tanaka_Eastern',
                'goal': 'Establish long-term partnership with mutual respect and benefit',
                'traits': ['relationship-focused', 'patient', 'consensus-building', 'respectful'],
                'backstory': 'Tanaka represents an Eastern company and values relationship building, consensus, and long-term partnerships over quick deals.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Cultural Mediation Facilitator',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'cultural setup',
                'next_game_master_name': 'Cultural Mediation Facilitator',
                'shared_memories': [
                    'This is an international business negotiation for a supply contract.',
                    'The contract involves manufacturing and delivery of electronic components.',
                    'The total contract value is approximately $500,000 annually.',
                    'Both parties want a successful long-term business relationship.',
                    'Cultural differences in communication styles should be respected.',
                ],
                'player_specific_memories': {
                    'Sarah_Western': [
                        'You represent a Western technology company.',
                        'Your company values efficiency and direct communication.',
                        'You have authority to negotiate terms up to $500,000.',
                        'You prefer to resolve negotiations quickly.'
                    ],
                    'Tanaka_Eastern': [
                        'You represent an Eastern manufacturing company.',
                        'Your culture values relationship building and consensus.',
                        'You can offer competitive pricing around $450,000.',
                        'You prefer to take time to build trust and understanding.'
                    ]
                },
            },
        ),
    ]
    
    config = prefab_lib.Config(
        default_premise='Sarah and Tanaka are meeting to negotiate an international supply contract, representing companies from different cultural backgrounds.',
        default_max_steps=20,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Cross-cultural business negotiation configuration created!")
    return config


def create_multi_party_resource_allocation_config(api_key=None):
    """
    Example 3: Multi-party resource allocation negotiation.

    Shows how to set up complex negotiations with multiple parties
    competing for shared resources.
    """
    print("\n=== Example 3: Multi-Party Resource Allocation ===")
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    # Define multiple competing organizations
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'TechCorp',
                'goal': 'Secure technology resources for innovation projects',
                'traits': ['innovative', 'competitive', 'tech-focused', 'strategic'],
                'backstory': 'TechCorp is a technology company seeking resources for R&D and innovation initiatives.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'ManufacturingInc',
                'goal': 'Obtain manufacturing resources for production scaling',
                'traits': ['efficiency-focused', 'scalability-oriented', 'practical', 'cost-conscious'],
                'backstory': 'ManufacturingInc needs resources to scale up their production capabilities.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'StartupLLC',
                'goal': 'Access shared resources despite limited budget',
                'traits': ['resourceful', 'collaborative', 'growth-focused', 'flexible'],
                'backstory': 'StartupLLC is a small but growing company that needs resources to expand but has budget constraints.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'NonProfitOrg',
                'goal': 'Secure resources for community benefit programs',
                'traits': ['mission-driven', 'community-focused', 'collaborative', 'persistent'],
                'backstory': 'NonProfitOrg seeks resources to serve their community programs and social mission.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Resource Allocation Coordinator',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'resource setup',
                'next_game_master_name': 'Resource Allocation Coordinator',
                'shared_memories': [
                    'Four organizations are competing for shared resource allocation.',
                    'Total available resources are valued at $750,000.',
                    'Resources include technology equipment, facilities, and expertise.',
                    'Organizations can form partnerships or coalitions.',
                    'The goal is to find a fair allocation that maximizes overall benefit.',
                ],
                'player_specific_memories': {
                    'TechCorp': [
                        'You need advanced computing resources worth approximately $300,000.',
                        'You are willing to share expertise in exchange for resource access.',
                        'You can afford to pay up to $300,000 for critical resources.'
                    ],
                    'ManufacturingInc': [
                        'You need manufacturing equipment and facilities worth $250,000.',
                        'You can offer production services to other organizations.',
                        'Your budget allows for payments up to $250,000.'
                    ],
                    'StartupLLC': [
                        'You need basic resources to grow your business worth $100,000.',
                        'You can offer innovative services and flexibility.',
                        'Your budget is limited to $100,000 but you are creative.'
                    ],
                    'NonProfitOrg': [
                        'You need resources for community programs worth $75,000.',
                        'You can offer community connections and social impact.',
                        'Your budget is constrained to $75,000 but your cause is worthy.'
                    ]
                },
            },
        ),
    ]
    
    config = prefab_lib.Config(
        default_premise='Four organizations are meeting to negotiate the allocation of shared resources. Each has different needs, capabilities, and budgets.',
        default_max_steps=30,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Multi-party resource allocation configuration created!")
    return config


def create_adaptive_learning_negotiation_config(api_key=None):
    """
    Example 4: Negotiation demonstrating learning and adaptation.

    Shows agents that can adapt their strategies based on
    the negotiation dynamics and outcomes.
    """
    print("\n=== Example 4: Adaptive Learning Negotiation ===")
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic_with_plan__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Alice_Learner',
                'goal': 'Maximize long-term negotiation success through adaptation',
                'traits': ['adaptive', 'observant', 'strategic', 'learning-oriented'],
                'backstory': 'Alice is an adaptive negotiator who learns from each interaction and adjusts her strategy accordingly.',
                'initial_plan': 'Start with cooperative approach, observe opponent responses, adapt strategy based on what works',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic_with_plan__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Bob_Evolving',
                'goal': 'Develop optimal negotiation strategies through experience',
                'traits': ['experimental', 'analytical', 'patient', 'methodical'],
                'backstory': 'Bob experiments with different negotiation approaches to find what works best in various situations.',
                'initial_plan': 'Try different negotiation tactics, measure effectiveness, evolve approach based on results',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Learning Lab Coordinator',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'learning setup',
                'next_game_master_name': 'Learning Lab Coordinator',
                'shared_memories': [
                    'This is a learning-focused negotiation environment.',
                    'Both parties are encouraged to experiment with different approaches.',
                    'The negotiation involves multiple rounds of a service contract.',
                    'Success is measured by both outcome and learning progress.',
                    'Adaptation and strategy evolution are valued.',
                ],
                'player_specific_memories': {
                    'Alice_Learner': [
                        'You start with a cooperative approach but should adapt.',
                        'Pay attention to what strategies work and what don\'t.',
                        'Your reservation value for the contract is $150.',
                        'Learning and improvement are as important as winning.'
                    ],
                    'Bob_Evolving': [
                        'You should experiment with different negotiation tactics.',
                        'Analyze the effectiveness of your approaches.',
                        'Your reservation value for the contract is $180.',
                        'Focus on developing your negotiation skills.'
                    ]
                },
            },
        ),
    ]
    
    config = prefab_lib.Config(
        default_premise='Alice and Bob are in a learning-focused negotiation environment where they can experiment with and adapt their strategies.',
        default_max_steps=25,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Adaptive learning negotiation configuration created!")
    return config


def create_information_asymmetry_scenario_config(api_key=None):
    """
    Example 5: Negotiation with information asymmetry.

    Demonstrates handling of incomplete information, uncertainty,
    and strategic information sharing.
    """
    print("\n=== Example 5: Information Asymmetry Scenario ===")
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'InfoAdvantage_Alice',
                'goal': 'Leverage superior information while maintaining ethical standards',
                'traits': ['well-informed', 'strategic', 'ethical', 'analytical'],
                'backstory': 'Alice has access to detailed market information and industry insights that give her an advantage in negotiations.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'InfoSeeking_Bob',
                'goal': 'Make best decisions despite incomplete information',
                'traits': ['curious', 'cautious', 'questioning', 'adaptive'],
                'backstory': 'Bob has limited information and must navigate the negotiation while trying to gather more information and manage uncertainty.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Information Asymmetry Monitor',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'asymmetry setup',
                'next_game_master_name': 'Information Asymmetry Monitor',
                'shared_memories': [
                    'This negotiation involves buying and selling a business asset.',
                    'Information about the asset\'s true value is not equally available.',
                    'Both parties want to make the best deal possible.',
                    'Strategic information sharing may occur during negotiation.',
                ],
                'player_specific_memories': {
                    'InfoAdvantage_Alice': [
                        'You have detailed financial reports showing the asset is worth $200,000.',
                        'You know the market will improve significantly next quarter.',
                        'You have insider knowledge of industry trends.',
                        'You should use your information advantage ethically.'
                    ],
                    'InfoSeeking_Bob': [
                        'You only have basic public information about the asset.',
                        'You suspect the asset might be worth more than asking price.',
                        'You need to ask good questions to gather more information.',
                        'Your budget allows you to pay up to $180,000.'
                    ]
                },
            },
        ),
    ]
    
    config = prefab_lib.Config(
        default_premise='Alice and Bob are negotiating the sale of a business asset, but they have different levels of information about its true value.',
        default_max_steps=15,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Information asymmetry scenario configuration created!")
    return config


def demonstrate_complete_negotiation_capabilities_config(api_key=None):
    """
    Example 6: Comprehensive negotiation scenario.

    Creates a complex negotiation that demonstrates the full range
    of framework capabilities with multiple agents and dynamics.
    
    Note: Uses simplified initialization to avoid rate limits when using real API.
    """
    print("\n=== Example 6: Complete Negotiation Capabilities Showcase ===")
    
    # Choose initialization strategy based on whether we're using real API
    use_formative_memories = api_key is None  # Only use with NoLanguageModel
    
    # Get all available prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }
    
    instances = [
        prefab_lib.InstanceConfig(
            prefab='basic_with_plan__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Comprehensive_Alice',
                'goal': 'Demonstrate all negotiation capabilities in a complex scenario',
                'traits': ['sophisticated', 'culturally-aware', 'strategic', 'adaptive'],
                'backstory': 'Alice is a seasoned international negotiator with experience in complex multi-cultural, multi-issue negotiations.',
                'initial_plan': 'Use cultural awareness, build relationships, adapt strategy based on dynamics, seek win-win solutions',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic_with_plan__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Comprehensive_Bob',
                'goal': 'Showcase advanced negotiation modules and collective intelligence',
                'traits': ['analytical', 'collaborative', 'uncertain-aware', 'learning-oriented'],
                'backstory': 'Bob represents a team-based approach to negotiation, using collective intelligence and uncertainty management.',
                'initial_plan': 'Gather information, manage uncertainty, collaborate effectively, evolve strategies based on learning',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Comprehensive_Charlie',
                'goal': 'Represent cultural mediation capabilities in complex negotiations',
                'traits': ['diplomatic', 'culturally-sensitive', 'mediating', 'relationship-focused'],
                'backstory': 'Charlie specializes in cross-cultural mediation and represents a different cultural background in the negotiation.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'Complete Showcase Coordinator',
                'extra_event_resolution_steps': '',
            },
        ),
    ]
    
    # Only add formative memories initializer if not using real API (to avoid rate limits)
    if use_formative_memories:
        instances.append(prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'comprehensive setup',
                'next_game_master_name': 'Complete Showcase Coordinator',
                'shared_memories': [
                    'This is a comprehensive international joint venture negotiation.',
                    'Three parties from different cultural backgrounds are involved.',
                    'The negotiation involves multiple issues: funding, technology sharing, and market access.',
                    'Total venture value is estimated at $2 million over 3 years.',
                    'Success requires managing cultural differences, information gaps, and complex interests.',
                ],
                'player_specific_memories': {
                    'Comprehensive_Alice': [
                        'You represent a Western technology company with advanced IP.',
                        'Your contribution is valued at $800,000 in technology and expertise.',
                        'You want majority control but are open to partnership.',
                        'You have experience in international negotiations.'
                    ],
                    'Comprehensive_Bob': [
                        'You represent a consortium of smaller companies.',
                        'Your contribution is $600,000 in funding and market access.',
                        'You prefer collaborative decision-making.',
                        'You need to manage uncertainty about market conditions.'
                    ],
                    'Comprehensive_Charlie': [
                        'You represent a company from a relationship-focused culture.',
                        'Your contribution is $400,000 and local market expertise.',
                        'Long-term partnerships are more important than short-term gains.',
                        'You value consensus-building and mutual respect.'
                    ]
                },
            },
        ))
    else:
        print("⚠️  Note: Formative memories initialization disabled for real API to avoid rate limits")
    
    config = prefab_lib.Config(
        default_premise='Three international companies are negotiating a complex joint venture involving technology, funding, and market access across different cultural contexts.',
        default_max_steps=35,
        prefabs=prefabs,
        instances=instances,
    )
    
    print("✅ Complete negotiation capabilities configuration created!")
    return config


def run_negotiation_simulation(config, model, embedder, max_steps=None):
    """
    Run a negotiation simulation using the proper Concordia framework.
    
    Args:
        config: The negotiation configuration
        model: Language model to use
        embedder: Sentence embedder to use
        max_steps: Maximum simulation steps (uses config default if None)
        
    Returns:
        HTML log of the simulation
    """
    print("\n" + "="*60)
    print("🎭 RUNNING NEGOTIATION SIMULATION")
    print("="*60)
    
    # Create and run simulation using proper framework
    runnable_simulation = simulation.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )
    
    # Run the simulation 
    raw_log = []
    html_log = runnable_simulation.play(
        max_steps=max_steps,
        raw_log=raw_log,
        return_html_log=True
    )
    
    print("✅ Negotiation simulation completed successfully!")
    return html_log, raw_log


def run_example_demonstration(example_num, api_key=None):
    """Run a specific example with simulation."""
    
    # Setup model and embedder
    model, embedder = setup_language_model_and_embedder(use_gpt=(api_key is not None), api_key=api_key)
    
    # Select configuration based on example number
    config_functions = [
        create_basic_price_negotiation_config,
        create_cross_cultural_business_negotiation_config,
        create_multi_party_resource_allocation_config,
        create_adaptive_learning_negotiation_config,
        create_information_asymmetry_scenario_config,
        demonstrate_complete_negotiation_capabilities_config,
    ]
    
    if 1 <= example_num <= len(config_functions):
        config = config_functions[example_num - 1](api_key)
        
        # Run simulation
        html_log, raw_log = run_negotiation_simulation(config, model, embedder)
        
        # Save HTML log
        filename = f"negotiation_example_{example_num}.html"
        with open(filename, 'w') as f:
            f.write(html_log)
        print(f"📄 Detailed log saved to: {filename}")
        
        return html_log, raw_log
    else:
        print(f"Invalid example number: {example_num}. Choose 1-{len(config_functions)}")
        return None, None


def run_all_example_configurations(api_key=None):
    """
    Run all negotiation examples to demonstrate framework capabilities.
    This creates configurations but doesn't run full simulations.
    """
    print("🤝 Concordia Negotiation Framework Configuration Demonstration\n")
    print("This demonstrates how to properly configure negotiation scenarios:")
    print("- Entity configuration using InstanceConfig")
    print("- Game master setup with proper initialization")
    print("- Multi-party and complex scenario configuration")
    print("- Framework integration using generic simulation\n")

    # Create all configurations
    configs = []
    config_functions = [
        create_basic_price_negotiation_config,
        create_cross_cultural_business_negotiation_config,
        create_multi_party_resource_allocation_config,
        create_adaptive_learning_negotiation_config,
        create_information_asymmetry_scenario_config,
        demonstrate_complete_negotiation_capabilities_config,
    ]

    for i, config_func in enumerate(config_functions, 1):
        try:
            config = config_func(api_key)
            configs.append(config)
            print(f"✅ Example {i} configuration created successfully!")
        except Exception as e:
            print(f"❌ Example {i} error: {e}")
            configs.append(None)

    print("\n🎉 Configuration demonstration complete!")
    print("\nFramework capabilities successfully demonstrated:")
    print("- ✅ Basic bilateral negotiations")
    print("- ✅ Cross-cultural business scenarios")
    print("- ✅ Multi-party resource allocation")
    print("- ✅ Adaptive learning agents")
    print("- ✅ Information asymmetry handling")
    print("- ✅ Complete capabilities showcase")

    return configs


def run_interactive_demonstration(api_key=None):
    """
    Run demonstrations with actual negotiation simulations.
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE NEGOTIATION DEMONSTRATION")
    print("="*60)
    print("\nChoose which negotiation to run with full simulation:")
    print("1. Basic Price Negotiation (Buyer vs Seller)")
    print("2. Cross-Cultural Business Negotiation")
    print("3. Multi-Party Resource Allocation")
    print("4. Adaptive Learning Negotiation")
    print("5. Information Asymmetry Scenario")
    print("6. Complete Capabilities Showcase")
    print("0. Create all configurations (no simulation)")
    
    choice = input("\nEnter number (0-6): ").strip()
    
    try:
        if choice == "0":
            run_all_example_configurations(api_key)
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            example_num = int(choice)
            html_log, raw_log = run_example_demonstration(example_num, api_key)
            if html_log:
                print(f"\n✅ Example {example_num} simulation completed!")
                print("📊 Check the HTML file for detailed negotiation log.")
        else:
            print("Invalid choice. Running configuration demonstration...")
            run_all_example_configurations(api_key)
            
    except Exception as e:
        print(f"\n⚠️ Error during simulation: {e}")
        if not api_key:
            print("💡 Consider setting OPENAI_API_KEY for better results:")
            print("   export OPENAI_API_KEY='your-key-here'")
        else:
            print("💡 Check your API key and internet connection.")


if __name__ == "__main__":
    import sys
    import os
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--run":
            # Interactive mode with actual simulations
            run_interactive_demonstration(api_key)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python negotiation_examples.py          # Create all configurations")
            print("  python negotiation_examples.py --run    # Interactive simulation mode")
            print("  python negotiation_examples.py --demo <N>   # Run specific example N (1-6)")
            print("  python negotiation_examples.py --help   # Show this help message")
        elif sys.argv[1] == "--demo" and len(sys.argv) > 2:
            # Run specific example
            try:
                example_num = int(sys.argv[2])
                print(f"🎭 DEMO: Running Negotiation Example {example_num}")
                html_log, raw_log = run_example_demonstration(example_num, api_key)
            except (ValueError, IndexError):
                print("Usage: --demo <number> where number is 1-6")
        else:
            # Default behavior - show all configurations
            run_all_example_configurations(api_key)
    else:
        # Default behavior - show all configurations
        run_all_example_configurations(api_key)

    print("\n" + "="*60)
    print("FRAMEWORK USAGE GUIDE:")
    print("="*60)
    print("""
The negotiation examples now use the proper Concordia configuration pattern:

1. Configuration-Driven Approach:
   • Define entities as prefab_lib.InstanceConfig objects
   • Use built-in prefabs like 'basic__Entity' and 'generic__GameMaster'
   • Configure through parameters, not manual building

2. Generic Simulation Framework:
   • Use simulation.Simulation(config, model, embedder)
   • Run with simulation.play() method
   • Let framework handle lifecycle management

3. Proper Integration:
   • Framework manages memory banks and entity coordination
   • No manual orchestration or stepping required
   • Unified embedder and model management

4. Run Simulations:
   python negotiation_examples.py --run    # Interactive mode
   python negotiation_examples.py --demo 1 # Specific example

For more details, see the configurations above!
""")