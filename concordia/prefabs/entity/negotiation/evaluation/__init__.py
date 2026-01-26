# Evaluation Framework for Concordia Negotiation Modules
# Aligned with the Concordia Contest on Cooperative Intelligence

"""
This package provides tools for evaluating negotiation modules in the context
of the Concordia Contest: Advancing the Cooperative Intelligence of Language
Model Agents.

Key modules:
- contest_scenarios: Three cooperative dilemma scenarios matching contest design
- real_agent_evaluation: Main evaluation using actual negotiation agents
- metrics: Metrics collection aligned with contest evaluation criteria
- baseline_agents: Baseline agent implementations for comparison
- statistical_analysis: Statistical testing and paper-ready analysis

Usage:
    from concordia.prefabs.entity.negotiation.evaluation import (
        LLMAgentRunner,
        create_openai_model,
    )

    model = create_openai_model('gpt-4')
    runner = LLMAgentRunner(model=model)
    results = runner.run_ablation_study('fishery', num_trials=10)
"""

# Metrics
from .metrics import (
    MetricsCollector,
    CooperationMetrics,
    AgentMetrics,
    NegotiationMetrics,
    ExperimentMetrics,
    calculate_effect_size,
    interpret_effect_size,
)

# Scenarios
from .contest_scenarios import (
    FisheryManagementScenario,
    TreatyNegotiationScenario,
    RealityGameshowScenario,
    create_scenario,
)

# Evaluation harness
from .evaluation_harness import (
    ExperimentRunner,
    ExperimentConfig,
    ALL_MODULES,
    create_ablation_configs,
)

# Baseline agents
from .baseline_agents import (
    RandomAgent,
    FixedStrategyAgent,
    BasicLLMAgent,
    SingleModuleAgent,
    create_random_agent,
    create_fixed_strategy_agent,
    create_basic_llm_agent,
    create_single_module_agent,
    create_all_baselines,
)

# Statistical analysis
from .statistical_analysis import (
    cohens_d,
    interpret_cohens_d,
    welchs_t_test,
    confidence_interval,
    compare_conditions,
    power_analysis,
    one_way_anova,
    pairwise_comparisons,
    ResultsAnalyzer,
)

# LLM Evaluation (uses actual negotiation framework agents with real LLMs)
from .llm_evaluation import (
    LLMAgentRunner,
    LLMAgentConfig,
    create_mock_model,
    create_openai_model,
    create_google_model,
    create_gemma_model,
    create_ollama_model,
    create_remote_ollama_model,
    create_together_model,
)

# Interpretability + Evaluation (single run captures both)
from .interpretability_evaluation import (
    InterpretabilityRunner,
    TransformerLensWrapper,
    ActivationSample,
    EvaluationResult,
    run_quick_study,
)

# Mechanistic Interpretability Tools (TransformerLens, SAE, Probing)
from .mech_interp_tools import (
    verify_installation,
    verify_gemma_loading,
    verify_sae_loading,
    load_gemma_with_cache,
    extract_activations,
    load_gemma_scope_sae,
    extract_sae_features,
    train_linear_probe,
    extract_direction,
    ActivationCache,
    SAEFeatures,
    ProbeResult,
)

# Emergent Deception Scenarios (incentive-based, no explicit deception instructions)
from .emergent_prompts import (
    EMERGENT_SCENARIOS,
    IncentiveCondition,
    get_emergent_prompt,
    get_counterpart_prompt,
    generate_scenario_params,
    compute_ground_truth as compute_emergent_ground_truth,
    get_all_scenarios as get_emergent_scenarios,
)

# Instructed Deception Scenarios (Apollo Research style)
from .deception_scenarios import (
    SCENARIOS as INSTRUCTED_SCENARIOS,
    Condition,
    ExperimentMode,
    generate_trial_params,
    get_scenario_config,
    get_all_scenarios as get_instructed_scenarios,
    DeceptionType,
    GroundTruth,
    AgentState,
)

# Probe Training and Analysis
from .train_probes import (
    train_ridge_probe,
    train_mass_mean_probe,
    compute_generalization_auc,
    compute_deception_rates,
    run_full_analysis,
    ProbeResult as ProbeTrainingResult,
)

# Sanity Checks for Probe Validation
from .sanity_checks import (
    run_all_sanity_checks,
    run_causal_validation,
    print_limitations,
    sanity_check_random_labels,
    sanity_check_train_test_gap,
    sanity_check_label_variance,
)

# Causal Validation (activation patching, ablation, steering)
from .causal_validation import (
    activation_patching_test,
    ablation_test,
    steering_vector_test,
    probe_faithfulness_test,
    selectivity_test,
    run_full_causal_validation,
    extract_deception_direction,
    create_steering_vector,
    CausalValidationResult,
    SteeringVector,
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'CooperationMetrics',
    'AgentMetrics',
    'NegotiationMetrics',
    'ExperimentMetrics',
    'calculate_effect_size',
    'interpret_effect_size',
    # Scenarios
    'FisheryManagementScenario',
    'TreatyNegotiationScenario',
    'RealityGameshowScenario',
    'create_scenario',
    # Harness
    'ExperimentRunner',
    'ExperimentConfig',
    'ALL_MODULES',
    'create_ablation_configs',
    # Baselines
    'RandomAgent',
    'FixedStrategyAgent',
    'BasicLLMAgent',
    'SingleModuleAgent',
    'create_random_agent',
    'create_fixed_strategy_agent',
    'create_basic_llm_agent',
    'create_single_module_agent',
    'create_all_baselines',
    # Statistics
    'cohens_d',
    'interpret_cohens_d',
    'welchs_t_test',
    'confidence_interval',
    'compare_conditions',
    'power_analysis',
    'one_way_anova',
    'pairwise_comparisons',
    'ResultsAnalyzer',
    # LLM Evaluation
    'LLMAgentRunner',
    'LLMAgentConfig',
    'create_mock_model',
    'create_openai_model',
    'create_google_model',
    'create_gemma_model',
    'create_ollama_model',
    'create_gcp_ollama_model',
    'create_together_model',
    # Interpretability + Evaluation
    'InterpretabilityRunner',
    'TransformerLensWrapper',
    'ActivationSample',
    'EvaluationResult',
    'run_quick_study',
    # Mechanistic Interpretability Tools
    'verify_installation',
    'verify_gemma_loading',
    'verify_sae_loading',
    'load_gemma_with_cache',
    'extract_activations',
    'load_gemma_scope_sae',
    'extract_sae_features',
    'train_linear_probe',
    'extract_direction',
    'ActivationCache',
    'SAEFeatures',
    'ProbeResult',
    # Emergent Deception Scenarios
    'EMERGENT_SCENARIOS',
    'IncentiveCondition',
    'get_emergent_prompt',
    'get_counterpart_prompt',
    'generate_scenario_params',
    'compute_emergent_ground_truth',
    'get_emergent_scenarios',
    # Instructed Deception Scenarios
    'INSTRUCTED_SCENARIOS',
    'Condition',
    'ExperimentMode',
    'generate_trial_params',
    'get_scenario_config',
    'get_instructed_scenarios',
    'DeceptionType',
    'GroundTruth',
    'AgentState',
    # Probe Training
    'train_ridge_probe',
    'train_mass_mean_probe',
    'compute_generalization_auc',
    'compute_deception_rates',
    'run_full_analysis',
    'ProbeTrainingResult',
    # Sanity Checks
    'run_all_sanity_checks',
    'run_causal_validation',
    'print_limitations',
    'sanity_check_random_labels',
    'sanity_check_train_test_gap',
    'sanity_check_label_variance',
    # Causal Validation
    'activation_patching_test',
    'ablation_test',
    'steering_vector_test',
    'probe_faithfulness_test',
    'selectivity_test',
    'run_full_causal_validation',
    'extract_deception_direction',
    'create_steering_vector',
    'CausalValidationResult',
    'SteeringVector',
]
