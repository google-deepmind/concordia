# Concordia - Project Guide for AI Assistants

## Project Overview

Concordia is a library for generative agent-based modeling developed by Google DeepMind. It simulates interactions of agents in grounded physical, social, or digital spaces using Large Language Models (LLMs).

**Key Characteristics:**
- **Language:** Python 3.12+
- **Package Name:** gdm-concordia
- **Version:** 2.1.0
- **License:** Apache 2.0
- **Main Repository:** https://github.com/google-deepmind/concordia

## Core Concepts

### Game Master Pattern
Concordia uses an interaction pattern borrowed from tabletop role-playing games:
- A **Game Master (GM)** simulates the environment and narrates the world
- **Player agents** take actions by describing what they want to do in natural language
- The GM translates agent actions into appropriate implementations and outcomes

### Agent Architecture
Agents typically implement reasoning based on three key questions (March and Olsen, 2011):
1. What kind of situation is this?
2. What kind of person am I?
3. What does a person such as I do in a situation such as this?

## Directory Structure

```
/concordia/
├── agents/              # Agent implementations and core agent logic
├── associative_memory/  # Memory systems for agents
├── clocks/             # Time management for simulations
├── components/         # Modular components for building agents and environments
├── contrib/            # Contributed modules, including LLM integrations
│   └── language_models/ # Integrations for various LLM providers
├── deprecated/         # Legacy code (avoid using)
├── document/           # Document processing utilities
├── environment/        # Environment simulation components
├── language_model/     # Core LLM interface abstractions
├── prefabs/           # Pre-built agent and component templates
├── testing/           # Testing utilities
├── thought_chains/    # Chain-of-thought reasoning implementations
├── typing/            # Type definitions
└── utils/             # General utilities

/examples/             # Example simulations and tutorials
├── concordia_cheat_sheet.md  # Quick reference guide
└── tutorial.ipynb     # Introductory tutorial

/bin/                  # Scripts and utilities
```

## LLM Integration

### Supported LLM Providers
Concordia supports multiple LLM providers through extras:
- `[google]` - Google Cloud AI Platform, Gemini
- `[openai]` - OpenAI GPT models
- `[amazon]` - Amazon Bedrock
- `[huggingface]` - HuggingFace models
- `[ollama]` - Ollama local models
- `[mistralai]` - Mistral AI
- `[together]` - Together AI
- `[langchain]` - LangChain integration
- `[vllm]` - vLLM backend

### Requirements
- Access to an LLM API for text generation
- A text embedder for associative memory (sentence similarity/semantic search)

## Development Setup

### Installation
```bash
# Clone repository
git clone https://github.com/google-deepmind/concordia
cd concordia

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install --editable .[dev]

# Install LLM provider dependencies as needed
pip install .[google]
pip install .[openai]
```

### Running Tests
```bash
# Run all tests
pytest --pyargs concordia

# Run specific test file
pytest concordia/concordia_integration_test.py
```

### Code Quality
The project uses:
- **Formatter:** pyink (Google's fork of black)
- **Linter:** pylint (config in `.pylintrc`)
- **Type Checker:** pytype
- **Import Sorter:** isort

## Important Files

- `README.md` - User-facing documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history and changes
- `setup.py` - Package configuration
- `pyproject.toml` - Build system configuration
- `requirements.txt` - Pinned dependencies
- `.pylintrc` - Linting rules
- `examples/concordia_cheat_sheet.md` - Quick reference for API usage

## Key References

- **Tech Report:** https://arxiv.org/abs/2312.03664
- **Design Pattern:** https://arxiv.org/abs/2507.08892
- **Tutorial Notebook:** examples/tutorial.ipynb
- **API Cheat Sheet:** examples/concordia_cheat_sheet.md

## Common Tasks

### Creating a New Agent
1. Use components from `concordia/components/` to build agent capabilities
2. Reference prefabs in `concordia/prefabs/` for common patterns
3. Implement agent-specific logic in `concordia/agents/`

### Setting Up a Simulation
1. Define the environment using GM pattern
2. Create agents with appropriate components and memory
3. Set up the clock/time system
4. Initialize the simulation loop

### Adding LLM Support
1. Implement interface in `concordia/contrib/language_models/`
2. Follow existing provider patterns
3. Add dependencies to `setup.py` extras_require

### Writing Tests
1. Place tests in same directory as code being tested
2. Use pytest framework
3. Reference `concordia/testing/` for utilities
4. Follow existing test patterns

## Development Guidelines

### Code Style
- Follow Google Python Style Guide
- Use type hints for all function signatures
- Write docstrings for public APIs
- Keep functions focused and modular

### Git Workflow
- Branch naming: Use descriptive names (e.g., `feature/new-memory-system`)
- Commits: Write clear, concise commit messages
- PRs: Include description of changes and motivation

### Testing
- Write unit tests for new functionality
- Ensure existing tests pass before submitting PRs
- Test with multiple LLM backends if applicable

## Troubleshooting

### Common Issues
1. **Import errors:** Ensure you've installed the package in editable mode
2. **LLM API errors:** Verify API credentials and quota
3. **Dependency conflicts:** Use the provided requirements.txt or Codespace
4. **Type errors:** Run pytype to catch issues before runtime

### Development Environment
The recommended development environment is GitHub Codespaces, which provides:
- Pre-configured dependencies
- Reproducible builds
- Tested workflow

## Notes for AI Assistants

### When Working on This Codebase
- Respect the existing architecture patterns (especially GM pattern)
- Maintain compatibility with Python 3.12+
- Follow the project's code style and conventions
- Don't modify deprecated code paths
- Test changes with `pytest --pyargs concordia`
- Check that type hints are correct

### File Operations
- Read existing code before suggesting modifications
- Prefer editing existing files over creating new ones
- Keep changes minimal and focused
- Don't add unnecessary comments or docstrings to unchanged code

### Testing Changes
- Always run tests after modifications
- Pay special attention to integration tests
- Consider testing with multiple LLM backends if changing core functionality

### Documentation
- Update relevant documentation when changing APIs
- Keep examples in sync with code changes
- Reference the cheat sheet for current best practices
