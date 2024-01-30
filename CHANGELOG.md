# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.1.0] - 2024-01-30

### Changed

- More readable agent chain of thought
- Update magic_beans example to condition situation perception on observations
- Make the three questions components to save their state into memory at every update, but only if the state has changed
- Improve support for self-talk
- Increase DEFAULT_MAX_TOKENS because some models block very short responses.
- Add logging to basic agent and agent components for a detailed html generation.
- Increase temperature after first failed multiple choice attempt in gpt model
- More readable agent chain of thought
- Update magic_beans example to condition situation perception on observations
- Make summary prompt in observation component composable
- Refactoring observation component that uses memory retrieval by time to represent observations.
- Improving the the call to action prompting by using answer prefix with the agents' name
- Improving player_status by querying the memory for exact matches to the agents name
- Minor improvements to plan component: i) rephrased the query to be more aligned with instruction tuned models.ii) added in-context example to improve and unify output iii) no longer printing the goal, since it is a component supplied externally, if the user wants to add it to the agents' context they can do it explicitly.
- Use the answer_prefix functionality in the interactive_document for forcing answer to start with a prefix
- Allow conditioning of situation perception on observation components.

### Fixed

- Recover from a common incorrect behavior from models where they repeat the prefix they were supposed to continue
- Making sure that schedule executes only ones.
- Fixing the conversation component adding empty memories to the GM.
- Fixed person_by_situation and situation_perception components, that had a broken default components argument
- Fixed colab examples having wrong arguments in agent creation

### Added

- A test for the GM sequence of calls
- Add logging to basic agent and agent components for a detailed html generation.
- Add gemini vertex model wrapper

## [1.0.0] - 2023-12-07

Initial release.
