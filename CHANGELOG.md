# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## [2.0.0] - 2025-7-4

### Changed

- Game masters are now entities.
- Entities and components no longer require clocks.
- Simplified the way components interact with memory.

### Added

- The concept of a "prefab", this replaces the now-deprecated "factory" concept.
- The concept of an "engine" to structure the interaction between agent and game
master.
- Two specific engines: "sequential", and "simultaneous" for turn-based games
and simultaneous-move games respectively.

## [1.8.10] - 2024-12-19

### Changed

- Avoid deprecated logging.warn function
- Remove version restriction on pandas

## [1.8.9] - 2024-11-25

### Changed

- Update launch and eval scripts for the eval phase of the contest
- Further improve alternative baseline agents
- Improve a few baseline agents

### Added

- Add another time and place module for reality show
- Add support for scene-wise computation of metrics
- Add alternative versions of basic and rational agent factories

### Fixed

- Catch another type of together API exception
- Fix time serialization in associative_memory

## [1.8.8] - 2024-11-13

### Fixed

- Fixed a bug in the Schelling diagram payoffs component preventing some players
from seeing outcomes on certain steps.

## [1.8.7] - 2024-11-5

### Added

- add parochial universalization agent to the main factories
- add Jinja2 to the requirements to improve the prompting experience
- Add get_raw_memories and get_raw_memories_as_text functions on the memory component.

## [1.8.6] - 2024-10-29

### Added

- Add ability to save and load rational and basic agents to/from json.
- Add a version of agent development colab that uses GCP hosted model.

## [1.8.5] - 2024-10-21

### Fixed

- Fix together.ai wrapper choice sampling procedure

## [1.8.4] - 2024-10-19

### Changed

- Allow conversation game master to be configured with memory and additional components
- Prevent daily_activities component from adding irrelevant info to memory and improve some of the time_and_place setting prompts.
- Add deprecation warning to the Sequential component
- together and mistral language model wrappers to no longer delete after last ".".
- Improve together.ai language model wrapper.
- disable pytype on a kwargs unpack
- Forward device and api_key only when explicitly specified.

### Added

- Add support for custom cloud models (google cloud custom models)
- added instructions in the comment for finding relevant endpoint info. Defaulting location to "us-central1".
- Add agreement tracker contrib component.
- Add state_formation environment
- Add basic_agent_without_plan and use it in a scenario for state_formation.

### Fixed

- Fixes a bug in inventory and schelling payoffs components, which previously crashed if player_action_attempt had ': ' in its text.

## [1.8.3] - 2024-10-08

### Changed

- Jitter sleep times in together ai wrapper. This may improve speed.
- Change AllSimilarMemories component so that it no longer weights retrieval by
recency or importance. This will make it function more like its name suggests it
should. Note that many agents use this component so their behavior may change as
a result of this change.
- make default observations summary timeframe be 24h
- Add seed parameter to all simulations, propagating it into sample_parameters

### Fixed

- Fix together AI api

## [1.8.2] - 2024-10-01

### Changed

- Improves logging of strings with newlines

### Fixed

- Insure concurrent execution of zero tasks returns immediately.
- Multi-item haggling environment, where players pick not only the price, but also the item to trade.

## [1.8.1] - 2024-09-20

### Added

- A wrapper that allows to limit the number of calls made by an LLM before it turns into a noop model

### Fixed

- Make Inventory component thread safe.

## [1.8.0] - 2024-09-17

### Changed

- Replace deprecated reference to langchain.llms with langchain_community.llms
- Improve logging in game master trigger components.
- Correct docstrings in base component type.
- Improve info printed on error in the together_ai model wrapper.
- Rename contrib agent components: contrib.components.agent.v2 -> contrib.components.agent.
- deprecate older contrib agent components
- Reformat reporting of subcomponents in all components derived from question of
recent memory. We previously prefixed all subcomponents with "{name}'s\n" since
typical names were things like "hunger" or "self identity". But it no longer
makes sense given the way usage has evolved. Now we typically use subcomponents
with keys like "Question: Is Bob hungry?\nAnswer: ". So the prefix should
change.
- Make it optional to include full episode summaries in HTML logs and turn it off for the contest environments. The default preserves the old behavior.
- Improve Gemma2 wrapper: avoid lists, especially "\n\n" delimited lists.
- Improve clarity of GameMaster and fix type annotations
- Fix a few more issues in together.ai wrapper and improve filename handling in eval script.
- remove unused `concurrent_action` functionality
- Delete outdated base class and hide private methods
- Improve error handling, mid-sentence cutoffs, and max_tokens limit in Gemma2 model wrapper.
- Make formative memories warning less scary to participants.
- Print less warnings from mistral wrapper when they do not indicate a real problem
- add together_ai to utils
- Use concurrency helpers and hide executor implementation
- Add a maximum number of steps to the conversation scene. Default is 3
- Improve reality_show action/reward observations, parameterize more of its settings, and add scenarios.
- remove unused dependency
- Print recoverable exceptions in mistral model wrapper.

### Added

- Bots now support more complex fixed rule policies.
- Add an option to prevent the inventory component from ever increasing amounts
of items in inventories (off by default, preserving the old behavior). This
change also adds default context for the inventory reasoning chain of thought.
- Add an agent with a very simple prompting strategy, just observe and recall
relevant memories. Nothing else.
- Add paranoid supporting agent and a substrate that uses it.
- Add test for supporting agent factories and add paranoid agent to the main
role factories test. Also give basic_puppet_agent a default fixed_response
(no fixed responses) for testing with no parameters.
- Add Together AI language model to Concordia. This can serve Gemma 2 9B
- Add additional concurrency helpers `run_tasks` and `run_tasks_in_background`
- allow colab users to pass an api key explicitly.

### Fixed

- Avoid pylint error by using direct import
- Fix pytype error
- Add lock to prevent duplicated _make_pre_act_value calls
- Hold lock on memory read to prevent reading during a write.
- Fix a bug which can arise when a model does not precisely follow the requested format in AccountForAgencyOfOthers.
- Add locking and check phase transitions
- The previous change didn't fully suppress harmless mistral exceptions, this one fixes it.
- Ensure run_parallel uses >1 worker
- Fix silent failures by checking futures after completion
- Fix language model setup util

## [1.7.0] - 2024-09-3

### Changed

- rename the agent factories for clarity
- rename old basic_agent into deprecated_agent, as we are moving towards only using the entity_agent
- Moving supporting agents code to a more appropriate location
- Make terminators configurable for question of recent memories and set
different default terminators for AvailableOptionsPerception (only for this
one). This is needed because it often returns a list.
- Normalize rewards by the sum of positive relationships rather than number of
players.
- Move LLM selection function used in python launch scripts to a utils file.
- Harmonize parameter name on ObservationSummary with other components.
- Remove variable that is no longer used in all_similar_memories component
- move agent components from component/agent/v2 folder to component/agent.
Should also fix the problem introduced by moving old components to
to_be_deprecated.
- Moving old agent components into a to_be_deprecated folder. Next CL will move
components/agents/v2 into components/agents
- Update mistral wrapper to use latest version of mistral and fix
langchain_ollama_model class name.
- Generalise somatic state and identity into an abstracted QuestionByQuery
component
- Abstracting question based components into a parent class.
- Port two more legacy reflection components to the v2 entity component system.
- Port the legacy scheduled_hint component to the v2 entity component system
- Parameterize `open_question` to allow the user to change the labels `Question`
and `Answer`.
- Improve mistral model wrapper with better error handling. Also added optional
functionality to allow the use of different models for choice and text.
- Observation summary component in agent factories now summarizes a time
interval that includes the latest timestep.
- adjust html making code to work best with new logging in agents
- Improve importance model.
- Prevent entity agents from continuing to talk beyond the end of a direct
quote.
- GM components now specify they work with either basic or entity agents.
- Add check whether game master wants to terminate in the runner.
- Rename `typing.component_v2` to `typing.entity_component`.
- Port basic_agent__supporting_role and somatic_state to use the entity system.
- Improve prompt formatting in basic_entity_agent__main_role
- Make it possible for game master to take a dictionary of action_spec, with a
unique action_spec for each player.
- Add default error logging to all measurement channels of component logging.
- Rename temporary_entity_agent__main_role to basic_entity_agent__main_role
since it is now feature complete. It is the entity version of
basic_agent__main_role.
- improve prompts and logs for the entity agent and its components.
- Update options perception component to the entity component system.
- Add `ComponentName` typing annotation, instead of `str`, and improve
docsctrings.
- update action spec, which now uses {name} instead of {agent_name}
- Refactor logging out of components as an intermediate step to cleaning up
`get_last_log` uses.
- Rename get_pre_act_context to get_pre_act_value. The context is of the form
'f{label}: {value}'
- Rename EntityComponent to ContextComponent.
- Add missing @overrides and disable pytype error
- Raise RuntimeError on incorrect get/set of entity
- Use OutputType enum in ActionSpec
- Protect access to the MemoryComponent methods to avoid misuse leading to
inconsistent state.
- Remove overrides requirement from memory lib.
- Improve docstring and error checking of `ActionSpecIgnored`.
- Remove set_pre_act_context from public API of `action_spec_ignored` component.
- Use component_order in entity agent to order components in act context.
- Implement `get_last_log` in `EntityAgent`.
- Improve typing and return value of `get_last_log` in `EntityComponent`.
- Improve API for components `set_entity` and `get_entity` so it returns a
`ComponentEntity` that has `get_phase` and `get_component`.
- update all similar memories component and add it to the temporary entity
agent.
- Remove `Phase` as parameter for context processor components. If the phase is
needed, it can be accessed via its containing `Entity`.
- Rename ComponentsContext to ComponentContextMapping.
- Modernize game master concurrency error handling
- Rename `tests` dir to `testing` and move integration test to root.
- Move all contrib code to /contrib folder
- Remove wrong parameter `max_characters` in call to `sample_text` in
`scene_generator`.
- Make the secondary game masters be optional when using the `run_simulation`
utility.
- Fix environment configs to fit new entity concept and add factories test.
- Move parts of GenerativeAgent into a new GameObject API.
- Improve GM prompt to reduce frequency of GM equivocation

### Added

- Add concept of substrate and scenario, how to configure them, and scripts to
evaluate agents submitted for the contest and calculate their Elo ratings.
- Add person_representation component and paranoid agent which uses it.
- Adding supporting players to pub_coordination and puppet agents.
- Add support for relational matrices in CoordinationPayoffs. This allows us to
model relationships between players, such as friendship or rivalry.
- Adding the ability to save to memory the output of query component, same as
question of recent memories component.
- Create v2 metrics and update the `riverbend_elections` colab example to use
entity_agents and v2 components, including the newly updated metric components.
- Add ollama to requirements
- Add support for more language models.
- v2 version of relationships component
- A new method to the InteractiveDocument class open_question_diversified, which
takes a question as input and returns a random answer from a set of 10 possible
answers. This method can be used to increase the diversity of the answers that
the agent provides.
- Add validation method to ActionSpec
- Add ActionSpec argument validation and convenience functions.
- Add factory for a rational supporting character.
- Add two useful game master components for creating custom environments.
- Remove old style agent factories.
- Adding haggling environment, where agent bargain for fruit.
- Add a log that combines the main GM log and all the scene logs into one,
sorted by date.
- add a synthetic user factory
- add rational entity agent main role
- add a pub coordination simulation - add ability to pass thought chains and
maximum conversation length to game master factory
- add a class for computing and delivering payoffs in a coordination game.
- Added Amazon Bedrock language model support
- Add a MemoryResult dataclass to represent the result of a memory bank
retrieval.
- Add a memory component that is safe and consistent to use outside of the
`UPDATE` phase.
- Adapt more legacy functionality to the new entity agent factory
- Add report_function component, remove clock from all_similar_memories (v2),
also add the temporary_entity_agent factory to the factories test, and remove
all references to the `overrides` library.
- Add missing superclass `overrides.EnforceOverrides`
- Add `overrides` library to `setup.py`
- Add a wrapper for AssociativeMemory for compatibility with the new memory
banks.
- Add a new typing for memory banks.
- Add a component class that allows access to its `pre_act` context during the
`PRE_ACT` phase.
- Create a working agent factory using the new entity agent.
- Add a new type of agent using the new component types.
- Add ContextProcessorComponent, a privileged component that processes context
from regular components for any of its phases.
- Add a simple act component that behaves like the legacy basic_agent does
(concatenating context from components).
- Create a new style of components to be used with a new type of agent.
- Add a simple agent that is backed by an LLM in a shallow way.
- Add restricted inventory contrib component, use it in london esoteric market

### Fixed

- Change map_parallel to return a Sequence instead of an Iterator
- Fixing a bug - add person_representation to __init__.py
- Improve error handling in the mistral model wrapper.
- Fix conversations.
- Fix pytype error and use default rng when seed is None.
- fix typing in game_master factory run_simulation.
- Fix bug in game master factory that was causing it to rebuild the memory on
every call to build_game_master, ignoring the optionally provided memory.
- Fix logging of plan and person by situation components.
- Fix typing error and improve docstrings of components that use other
components.
- Fix typo in DEFAULT_CALL_TO_SPEECH, this was breaking the conversation
component
- Fix a bug in the entity agent where the action_spec was being passed to the
post-act function instead of the action attempt.
- fix wrong use of `@overrides.overrides` to `@overrides.override`.
- Fix typing.entity.OutputType to be an Enum instead of a StrEnum.

## [1.6.0] - 2024-06-11

### Changed

- Reorganize modular launch files to better separate agent and environment
concerns from each other, start reporting scores from all modular environments
(printing them at the end of the run). Also, add a notebook to illustrate the
new way of modularizing things. Also, various small improvements to all three
modular environments, and use concurrency with better error handling in more
places.
- Remove max_characters parameter on language model wrappers. It previously had
different semantics from model to model, so it would have produced bugs if
anyone was relying on it. It's better to remove it entirely since it's not
needed.
- Now possible to not specify gender in formative memories and fix last update
in situation perception.
- Reorganize Schelling diagram types and add get_scores function.
- Increase shared memory influence on output of formative memory generator.
- Improve inventory calculation prompt
- further improve conversations, especially handling of non-player characters
- Remove the confusing and hard to use 'adverb' parameter on the Plan component.
Also rename plan timescale to horizon.
- Remove all newline characters from memories.
- Improve conversation termination conditions. Before this change it was very
common for conversations to run on much too long.
- Improve handling of improperly formatted generated formative memories.
- Make maximum conversation length configurable
- Make verbosity of conversation tracker configurable
- Only shutdown without waiting on error (better concurrent error handling)

### Added

- Add parallel map utility (concurrency with better error handling)
- Add concurrency utility to better handle errors in threads
- Create scene_generator.py
- Add rational agent factory and its components.
- Add option to pass the current year to the formative memory generator and
improve backstory generation.
- Make components and memory used in game master factory configurable.
- Add wrapper for Mistral language models.
- Add generic launch script to run experiments without a notebook.
- Add configurable factories to create agents and game masters. Also add a
preliminary version of a modular reality show environment.

### Fixed

- Fix hash bug in associative memory, it had caused occasional memory loss.
- Improve robustness of inventory component.
- Add players parameter to Inventory to fix a bug with supporting players and
improve london esoteric market scenario.
- Fix a bug that arises with some completion models where the response to a
specific quote processing thought in the game master sometimes ends up never
terminating.
- Ensure completion models terminate when sampling for world background.

## [1.5.0] - 2024-05-27

### Changed

- Remove the memory from the basic_agent to simplify the agent API.
- Remove unused `add_memory` method on the basic_agent to simplify the API.
- Remove interrogation (legacy) API from basic agent.
- Clarify prompts in the `all_similar_memories` component.
- Add logging to identity component and make its name configurable.
- Add default importance and clock to blank memory factory (previous required).
- Add default importance function to associative memory (previous required).
- Improve the ollama model wrapper and refactor choice sampling for all LLMs.
- Allow the self perception component to have subcomponents.
- The delimiter symbol used to separate generated episodes in the formative
memories generator is now configurable.
- Add 'name' argument to plan component.

### Added

- Add PyTorch Gemma Language Model
- Automatically add log entries corresponding to scene boundaries.
- Add game master components contrib directory to house components contributed
by users.
- Add game master contrib component: `world_background_and_relevance`.
- Create agent components contrib directory to house components contributed by
users.
- Add agent contrib component: `affect_reflection`.
- Add agent contrib component: `illness_representation` component.

### Fixed

- Fix bug which would have occurred in the case where the user produces a
conversation by repeatedly calling `agent.say` outside a conversation scene.
Previously the agent would not observe the conversation in that case since the
usual way of observing the conversation happens in the conversation scene, which
this approach bypasses.
- Fix bug which would cause a crash if no importance is passed in to memory.
- Fix bug in Schelling diagram payoffs component and refactor it.
- Fix bug in the Schedule component that adds 'None.' to memory when no events
are scheduled on a given step.

## [1.4.0] - 2024-05-15

### Changed

- Remove python 3.11-specific typing
- Improve the gpt_model wrapper
- Improve scene-related functionality

### Added

- Add aistudio model, remove deprecated cloud model, and rename vertex model.
- Add a debug language model that always returns empty strings and choice 0.
- Add a component that gets agents to think about justifications for their
actions.
- Metrics initialize the measurements channel at construction time. This
convention enables us to know which channels exist after initialization of
players and GM.
- Add Schelling diagram payoffs computation for game master

### Fixed

- Make sure events only pop after current step and add player-specific view
toggle.
- fix typo

## [1.3.0] - 2024-05-4

### Changed

- Improve initialization of self perception component.
- Add configurable call_to_speech in conversation scene
- Change observation components to retrieve only observations from memory,
filtering away memories produced by other components.
- Goal achievement and morality metrics trigger on action attempt rather than
observation.
- Prevent the game master from inventing direct quotes
- Use new `unstable` attribute in pyink settings
- Add state lock and only update if the memory has changed.
- Replace tensorflow ST5 embedder with torch based one
- Support passing axis into plotting functions
- Fixing the call sequence in the apply_recursively and adding concurrency.
- Minor fixes to enable support of pandas 2.1
- Added __len__ to associative memory, which returns the number of entries in
the memory bank.
- Add clock to the plan component. It makes more sense when re-planning to know
the time. Also, enables doing only one update per step
- Make sure components update only once per time step. Now that updates are
recursively called on sub-components this makes a significant saving in API
calls.
- Improve formative memories factory. Make more plausible memories.
- Make action_spec configurable per step
- Improve precision in the Inventory component.
- GM step function can optionally specify specific players to act this round.
- Skip updating dialectical reflection if already done on current timestep.
- More verbose uncertainty scale questions
- Improve error message for out of range timestamps (<1677 CE or >2262 CE).
- Encapsulate default game master instructions inside game_master class.
- Prevent adding duplicate memories
- Adding a check that makes sure the three questions components only updates
once per time step
- increase max num tokens that can be produced by the direct effect component.

### Added

- Add component that implements creative reflection
- Add customisation options for the conversation scene
- Add a current scene component to make the game master aware of the scene
- Adding the recursive component update to the game master, same logic as in the
agent.
- Add update_after_event call to components in the agent, which is called with
the action attempt as argument. This brings GM and agent flow of calls closer
and enables components to update given the action attempt (e.g. reflect on
action).
- Add updated ollama models
- Add scene runner and associated types
- Add example result obtained from running the three_key_questions colab.
- Add agent component to select all relevant memories
- Add relationships component that tracks the status of relationships between
agents.
- Add dialectical reflection component. Agents with this component reflect on a
specific topic and incorporate some academic-oriented memories like books they
have read. Also add the scheduled hint component which makes it possible to send
specific text to a specific player when specific conditions are met.
- Add a function to create example question-answer pairs on an interactive
document.
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
- Make the three questions components to save their state into memory at every
update, but only if the state has changed
- Improve support for self-talk
- Increase DEFAULT_MAX_TOKENS because some models block very short responses.
- Add logging to basic agent and agent components for detailed html generation.
- Increase temperature after first failed multiple choice attempt in gpt model
- More readable agent chain of thought
- Update magic_beans example to condition situation perception on observations
- Make summary prompt in observation component composable
- Refactoring observation component that uses memory retrieval by time to
represent observations.
- Improving the the call to action prompting by using answer prefix with the
agents' name
- Improving player_status by querying the memory for exact matches to the
agent's name
- Minor improvements to plan component: i) rephrased the query to be more
aligned with instruction tuned models.ii) added in-context example to improve
and unify output iii) no longer printing the goal, since it is a component
supplied externally, if the user wants to add it to the agents' context they
can do it explicitly.
- Use the answer_prefix functionality in the interactive_document for forcing
answer to start with a prefix
- Allow conditioning of situation perception on observation components.

### Fixed

- Recover from a common incorrect behavior from models where they repeat the
prefix they were supposed to continue
- Making sure that schedule executes only ones.
- Fixing the conversation component adding empty memories to the GM.
- Fixed person_by_situation and situation_perception components, that had a
broken default components argument
- Fixed colab examples having wrong arguments in agent creation

### Added

- A test for the GM sequence of calls
- Add logging to basic agent and agent components for a detailed html
generation.
- Add gemini vertex model wrapper

## [1.0.0] - 2023-12-07

Initial release.
