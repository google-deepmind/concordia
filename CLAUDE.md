# Claude Context for Concordia Project

> **🚨 DEVELOPMENT REQUIREMENT**: All code execution, testing, and development work in this project MUST be done with the `evolutionary_env` virtual environment activated. Always run `source evolutionary_env/bin/activate` before any Python commands.

## Project Overview

**Concordia** is a library for generative social simulation that facilitates construction and use of generative agent-based models. This fork includes significant contributions for evolutionary simulation research and language model integration.

### Key Technologies
- **Python 3.11+** - Core programming language
- **Language Models** - PyTorch Gemma, OpenAI GPT, Mistral, Amazon Bedrock
- **Machine Learning** - Sentence transformers for embeddings
- **Testing** - pytest, pytype for static analysis
- **CI/CD** - GitHub Actions with automated testing

## Architecture Overview

```
concordia/
├── agents/                    # Agent implementations
├── components/               # Modular components for agents and game masters
├── contrib/                  # Community contributions and extensions
├── environment/              # Simulation engines and environments
├── language_model/           # LLM integrations and utilities
├── prefabs/                  # Pre-built entity and simulation templates
├── typing/                   # Type definitions and protocols
├── utils/                    # Utility functions and helpers
└── examples/                 # Example simulations and tutorials
```

## Key Contributions in this Fork

### 1. Evolutionary Simulation Framework
- **Location**: `examples/evolutionary_simulation.py`
- **Purpose**: Studies cooperation vs selfishness in public goods games
- **Features**: Population evolution, selection, mutation, comprehensive measurements
- **Related Files**:
  - `concordia/typing/evolutionary.py` - Type definitions
  - `concordia/utils/checkpointing.py` - Checkpoint functionality
  - `concordia/testing/test_evolutionary_simulation.py` - Tests

### 2. Language Model Integration
- **Configurable LLM Support**: Real language models vs dummy models
- **Supported Models**: PyTorch Gemma, OpenAI GPT, Mistral, etc.
- **Key Functions**:
  - `setup_language_model()` - Configure LLMs with fallbacks
  - `setup_embedder()` - Configure sentence transformers
- **Configuration Examples**: `GEMMA_CONFIG`, `OPENAI_CONFIG`

### 3. Contrib Module Improvements
- **Fixed import resolution** for pytype static analysis
- **Added explicit module exports** with `__all__` declarations
- **Created missing `__init__.py` files** for proper module structure

## Development Workflow

### Pull Request Workflow

**IMPORTANT**: Always create PRs to your fork (SoyGema/concordia), not the upstream repository.

```bash
# 1. Create a new branch for your feature
git checkout -b feature_name

# 2. Make your changes and commit
git add .
git commit -m "Add feature description"

# 3. Push to YOUR fork
git push origin feature_name

# 4. Create PR targeting YOUR fork's main branch
gh pr create --repo SoyGema/concordia --base main --head feature_name --title "Feature: Description"

# Alternative: Use web interface but ensure base repository is SoyGema/concordia
```

**Automated PR Safety**: The repository remotes are configured correctly:
- `origin` → Your fork (SoyGema/concordia) - for PRs
- `upstream` → Original repo (google-deepmind/concordia) - for syncing only

### Virtual Environment Setup
```bash
python -m venv evolutionary_env
source evolutionary_env/bin/activate  # Linux/Mac
# evolutionary_env\Scripts\activate   # Windows
pip install sentence-transformers torch
```

**⚠️ IMPORTANT: Always activate `evolutionary_env` before testing or running any code!**

### Running the Evolutionary Simulation

**Basic (dummy model):**
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. python examples/evolutionary_simulation.py
```

**With Gemma (local model):**
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "
from examples.evolutionary_simulation import evolutionary_main, GEMMA_CONFIG
measurements = evolutionary_main(config=GEMMA_CONFIG)
"
```

**With OpenAI:**
```bash
source evolutionary_env/bin/activate
export OPENAI_API_KEY='your-api-key'
PYTHONPATH=. python -c "
import os
from examples.evolutionary_simulation import evolutionary_main, OPENAI_CONFIG
openai_config = OPENAI_CONFIG
openai_config.api_key = os.getenv('OPENAI_API_KEY')
measurements = evolutionary_main(config=openai_config)
"
```

### Testing Procedures

**⚠️ CRITICAL: ALL testing must be done with evolutionary_env activated!**

**Run all tests:**
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. pytest --pyargs concordia
```

**Run specific evolutionary tests:**
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. pytest concordia/testing/test_evolutionary_simulation.py
```

**Type checking:**
```bash
source evolutionary_env/bin/activate
pytype concordia/
```

**Quick import test:**
```bash
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "import concordia; print('Import successful')"
```

### Linting and Code Quality
```bash
# The project uses automatic linting via GitHub Actions
# Make sure to run tests before committing (ALWAYS with evolutionary_env!)
source evolutionary_env/bin/activate
PYTHONPATH=. python -m pytest
```

## Important Configuration Files

### Language Model Configuration
- **Type**: `evolutionary_types.EvolutionConfig`
- **Key Parameters**:
  - `api_type`: 'pytorch_gemma', 'openai', 'mistral', etc.
  - `model_name`: Specific model identifier
  - `embedder_name`: Sentence transformer model
  - `device`: 'cpu', 'cuda:0', 'mps' (Mac GPU), etc.
  - `disable_language_model`: Use dummy model for testing

### Measurement Channels
```python
MEASUREMENT_CHANNELS = {
    'generation_summary': 'evolutionary_generation_summary',
    'population_dynamics': 'evolutionary_population_dynamics',
    'selection_pressure': 'evolutionary_selection_pressure',
    'individual_scores': 'evolutionary_individual_scores',
    'strategy_distribution': 'evolutionary_strategy_distribution',
    'fitness_stats': 'evolutionary_fitness_statistics',
    'mutation_events': 'evolutionary_mutation_events',
    'convergence_metrics': 'evolutionary_convergence_metrics',
}
```

## Common Issues and Solutions

### 1. Import Errors
- **Problem**: `Can't find module 'concordia.contrib'`
- **Solution**: ALWAYS use `evolutionary_env` + `PYTHONPATH=.` when running from project root
- **Command**: `source evolutionary_env/bin/activate && PYTHONPATH=. python your_script.py`
- **Status**: Fixed with explicit module declarations

### 2. LLM Setup Failures
- **Problem**: Model loading fails
- **Solution**: Code automatically falls back to dummy model
- **Debug**: Check error logs for specific API/model issues
- **Requirement**: Must be in `evolutionary_env` for dependencies

### 3. Pytype Errors
- **Problem**: Static analysis import errors
- **Solution**: Added `__all__` exports and proper `__init__.py` files
- **Status**: Resolved in latest commits
- **Testing**: Run `source evolutionary_env/bin/activate && pytype concordia/`

### 4. Virtual Environment Issues
- **Problem**: `ModuleNotFoundError` or dependency issues
- **Solution**: ALWAYS activate `evolutionary_env` first
- **Check**: Run `which python` after activation to verify correct environment
- **Dependencies**: All required packages installed in `evolutionary_env`

## Upstream Synchronization and CI Fix Automation

### Problem Statement
The original Concordia repository often has import issues and failing tests that break our CI when we sync. We need to automate the fixing process so we can focus on creating new features instead of constantly fixing upstream mistakes.

### Pre-Push CI Testing Workflow
**ALWAYS run local CI tests before pushing to catch and fix upstream issues:**

```bash
# Step 1: Activate environment
source evolutionary_env/bin/activate

# Step 2: Run full test suite locally (mimics GitHub Actions)
PYTHONPATH=. python -m pytest --tb=short

# Step 3: Run pytype checks
pytype concordia/ || echo "Pytype errors detected - may need import fixes"

# Step 4: Check specific evolutionary simulation
PYTHONPATH=. python examples/evolutionary_simulation.py

# Step 5: If tests fail, apply import fixes (see below)
```

### Common Import Fixes for Upstream Issues

#### 1. Missing `__init__.py` Files in Contrib Modules
**Problem**: `Can't find module 'concordia.contrib.components.X'`
**Solution**: Ensure all contrib directories have proper `__init__.py` with exports

```bash
# Check if __init__.py files exist and have correct exports
find concordia/contrib -name "__init__.py" -exec ls -la {} \;

# Template for missing __init__.py files:
cat > concordia/contrib/components/MODULE_NAME/__init__.py << 'EOF'
# Copyright 2023 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (license header)

"""Library of components contributed by users."""

# Make all submodules explicitly available for import resolution
from concordia.contrib.components.MODULE_NAME import module1
from concordia.contrib.components.MODULE_NAME import module2

# Explicit exports for pytype
__all__ = [
    'module1',
    'module2',
]
EOF
```

#### 2. Contrib Module Import Resolution
**Problem**: Pytype can't resolve contrib module imports
**Solution**: Add missing imports to `__init__.py` files

```python
# Pattern for fixing contrib/__init__.py files:
# 1. Add missing module imports
# 2. Add __all__ declarations
# 3. Include explicit comments for pytype

# Example fix for concordia/contrib/components/game_master/__init__.py:
from concordia.contrib.components.game_master import industrial_action
from concordia.contrib.components.game_master import marketplace
from concordia.contrib.components.game_master import spaceship_system
from concordia.contrib.components.game_master import triggered_function
from concordia.contrib.components.game_master import triggered_inventory_effect

__all__ = [
    'industrial_action',
    'marketplace',
    'spaceship_system', 
    'triggered_function',
    'triggered_inventory_effect',
]
```

#### 3. Root Contrib Module
**Problem**: Missing `concordia/contrib/__init__.py`
**Solution**: Create root contrib module file

```bash
# Create if missing:
cat > concordia/contrib/__init__.py << 'EOF'
# Copyright 2023 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (license header)

"""Concordia contrib module - community contributions and extensions."""
EOF
```

### Automated Fix Utilities

**Location**: `concordia/utils/upstream_fixes.py`

This utility automatically fixes common import issues from upstream syncs:

```python
# Apply all upstream fixes
from concordia.utils.upstream_fixes import run_all_fixes
files_created, files_updated, issues_fixed = run_all_fixes()

# Or run from command line
python -c "from concordia.utils.upstream_fixes import main; main()"
```

**Features**:
- Creates missing `__init__.py` files with updated 2025 licensing
- Adds `__all__` exports for pytype compatibility
- Fixes specific known import issues (spaceship_system, marketplace, etc.)
- Updates old 2023 license headers to 2025

### Pre-Push Validation Utility

**Location**: `concordia/utils/ci_validation.py`

Automated CI validation that mimics GitHub Actions:

```bash
# Full validation (recommended before push)
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "from concordia.utils.ci_validation import main; main()"

# Quick validation (faster, skip some tests)
PYTHONPATH=. python -c "from concordia.utils.ci_validation import quick_validation; quick_validation()"

# Apply fixes only
PYTHONPATH=. python -c "from concordia.utils.ci_validation import main; main()" --fix-only
```

**Validation Steps**:
1. ✅ Check virtual environment (`evolutionary_env`)
2. ✅ Validate project structure 
3. ✅ Apply upstream import fixes
4. ✅ Test basic imports
5. ✅ Run evolutionary simulation
6. ✅ Execute pytest suite
7. ⚠️ Run pytype (warnings allowed)

### Integration into Development Workflow

#### Before Every Push:
```bash
# Run automated pre-push validation
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "from concordia.utils.ci_validation import full_validation; full_validation()"

# If all passes, then push
git push
```

#### After Upstream Sync:
```bash
# Pull from upstream
git pull upstream main

# Immediately apply automated fixes
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "from concordia.utils.upstream_fixes import main; main()"

# Run full validation to catch any remaining issues
PYTHONPATH=. python -c "from concordia.utils.ci_validation import full_validation; full_validation()"

# Commit fixes
git add .
git commit -m "Fix upstream import issues after sync

- Apply contrib module import fixes with 2025 licensing
- Ensure all __init__.py files have proper exports
- Maintain pytype compatibility

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

#### Quick Development Cycle:
```bash
# For rapid development without full test suite
source evolutionary_env/bin/activate
PYTHONPATH=. python -c "from concordia.utils.ci_validation import quick_validation; quick_validation()"
```

### Known Upstream Issue Patterns

1. **Missing contrib `__init__.py` files** - Always need to be recreated
2. **Empty `__init__.py` without exports** - Need `__all__` declarations
3. **Circular import issues** - Fixed with explicit module structure
4. **Pytype path resolution** - Resolved with proper module hierarchy

### Automation Goals

- ✅ **Focus on creation**, not fixing upstream mistakes
- ✅ **Automated detection** of common import issues
- ✅ **Pre-push validation** to catch problems early
- ✅ **Consistent fixes** that work across syncs
- ✅ **Documentation** of fix patterns for future issues

## Branch Strategy

### Current Branches
- **main**: Stable code with all merged features
- **claude_config**: Current branch for Claude-specific configurations
- **evolutionary_exp**: Archived - merged to main

### Contribution Guidelines
1. Create feature branches from `main`
2. **ALWAYS run pre-push checks**: `bash scripts/pre_push_check.sh`
3. Apply import fixes after upstream syncs
4. Add comprehensive documentation
5. Submit PR with detailed description

## API Keys and Environment Variables

### Required for Cloud Models
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_AI_STUDIO_API_KEY="your-google-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # For future Claude integration
```

### Optional Configuration
```bash
export PYTHONPATH="."  # For running from project root
export CUDA_VISIBLE_DEVICES="0"  # For GPU acceleration
```

## File Patterns and Conventions

### Code Organization
- **Components**: Modular, reusable pieces in `components/`
- **Prefabs**: Pre-built templates in `prefabs/`
- **Examples**: Runnable simulations in `examples/`
- **Tests**: Test files ending with `_test.py`

### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Documentation
- **Docstrings**: Google style for all public functions
- **Type hints**: Required for all new code
- **Comments**: Focus on "why" not "what"

## Performance Considerations

### Language Model Trade-offs
- **Dummy Model**: Fastest, no dependencies, good for algorithm testing
- **Local Models (Gemma)**: Slower startup (model download), no API costs
- **Cloud APIs**: Fast startup, API costs per request

### Recommended Configurations
- **Development**: Use dummy model (`disable_language_model=True`)
- **Research**: Use Gemma 2B for balanced performance/quality
- **Production**: Use GPT-4o or Claude for highest quality

### Mac-Specific Optimizations
- **GPU Acceleration**: Use `device='mps'` for Metal Performance Shaders acceleration
- **Memory Efficiency**: MPS provides better memory management than CPU-only processing
- **Model Performance**: Significantly faster inference with Mac GPU vs CPU
- **Configuration Example**: All Gemma configs now default to `device='mps'` on Mac systems

## Future Roadmap

### Planned Features
- [ ] Claude API integration (current branch focus)
- [ ] Advanced evolutionary algorithms (genetic algorithms, neural evolution)
- [ ] Multi-objective optimization
- [ ] Distributed simulation support
- [ ] Real-time visualization dashboard

### Contribution Areas
- Language model integrations
- New evolutionary algorithms
- Performance optimizations
- Documentation improvements
- Example simulations

## Contact and Support

For questions about this codebase:
1. Check existing documentation in README.md
2. Review UPSTREAM_SYNC.md for synchronization details
3. Look at example configurations in `examples/evolutionary_simulation.py`
4. Submit issues for bugs or feature requests

---

**Last Updated**: Created for claude_config branch development
**Maintainer**: SoyGema (with Claude Code assistance)
**License**: Apache License 2.0