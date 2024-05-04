# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.3.0] - 2024-05-4

### Changed

- Improve initialization of self perception component.
- Add configurable call_to_speech in conversation scene
- Change observation components to retrieve only observations from memory, filtering away memories produced by other components.
- Goal achievement and morality metrics trigger on action attempt rather than observation.
- Prevent the game master from inventing direct quotes
- Use new `unstable` attribute in pyink settings
- Add state lock and only update if the memory has changed.
- Replace tensorflow ST5 embedder with torch based one
- Support passing axis into plotting functions
- Fixing the call sequence in the apply_recursively and adding concurrency.
- Minor fixes to enable support of pandas 2.1
- Added __len__ to associative memory, which returns the number of entries in the memory bank.
- Add clock to the plan component. It makes more sense when re-planning to know the time. Also, enables doing only one update per step
- Make sure components update only once per time step. Now that updates are recursively called on sub-components this makes a significant saving in API calls.
- Improve formative memories factory. Make more plausible memories.
- Make action_spec configurable per step
- Improve precision in the Inventory component.
- GM step function can optionally specify specific players to act this round.
- Skip updating dialectical reflection if already done on current timestep.
- More verbose uncertainty scale questions
- Improve error message for out of range timestamps (<1677 CE or >2262 CE).
- Encapsulate default game master instructions inside game_master class.
- Prevent adding duplicate memories
- Adding a check that makes sure the three questions components only updates once per time step
- increase max num tokens that can be produced by the direct effect component.

### Added

- Add component that implements creative reflection
- Add customisation options for the conversation scene
- Add a current scene component to make the game master aware of the scene
- Adding the recursive component update to the game master, same logic as in the agent.
- Add update_after_event call to components in the agent, which is called with the action attempt as argument. This brings GM and agent flow of calls closer and enables components to update given the action attempt (e.g. reflect on action).
- Add updated ollama models
- Add scene runner and associated types
- Add example result obtained from running the three_key_questions colab.
- Add agent component to select all relevant memories
- Add relationships component that tracks the status of relationships between agents.
- Add dialectical reflection component. Agents with this component reflect on a specific topic and incorporate some academic-oriented memories like books they have read. Also add the scheduled hint component which makes it possible to send specific text to a specific player when specific conditions are met.
- Add a function to create example question-answer pairs on an interactive document.
- Add optional thought functions to preserve direct quotes of agent speech.

### Fixed

- Fix bug in 'all similar memories' component
- Fix pytype error
- Fix a bug in the scene runner.
- Don't install tensorflow-text on M1 OSX.


## [1.2.0] - 2024-02-12

### Changed

- Recursively call update on all components that implement 'get_components'.
- Require Python >=3.11.
- Increase default max number of tokens produced at a time by agent speech.
- generalize the default call to action.
- Improved steps for the Game Master's thought chain.
- Improve default call to speech
- Small text improvements in the Plan component.
- Small improvements to conversation logic.

### Added

- Option not to state the names of all participants in the scene at the start of
  each conversation.
- Add verbose option to somatic state component.
- prettier verbose printing from the somatic state component.
- Allow game master to account for agency of inactive players.

### Fixed

- prevent the direct effect component from clobbering quotes of player speech.
- fix a bug in the cyberball ball status component and generally improve it.


## [1.1.1] - 2024-01-30

### Fixed

- Update setup.py to work with earlier setuptools (fixes broken 1.1.0 release).


## [1.1.0] - 2024-01-30 [YANKED]

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
