# Document

The `document` module provides the foundational abstraction for managing text,
context, and LLM interactions in Concordia. It treats interaction with language
models as a process of building and refining a "document" of context.

## Core Concepts

The module is built around three main classes:

1.  **`Document`** (`document.py`): A structural container for text content
    using a list of `Content` objects (text + tags). It supports branching
    (`copy`) and filtered viewing (`view`).
2.  **`InteractiveDocument`** (`interactive_document.py`): Extends `Document`
    to support direct interaction with a `LanguageModel`. It provides methods to
    ask questions, generate responses, and maintain the dialogue history.
3.  **`InteractiveDocumentWithTools`** (`interactive_document_tools.py`):
    Extends `InteractiveDocument` to support LLM tool use. The LLM can call
    tools (e.g., web search) during question answering, with results
    automatically integrated into the document.

This abstraction allows for complex narrative logic, branching reasoning
(thought chains), and component-isolated views of the world.

## Patterns of Use

### 1. The Interaction Loop

The most common pattern is using `InteractiveDocument` as the working memory for
an agent or component. You build context by appending statements and interaction
results.

```python
from concordia.document import interactive_document

# 1. Initialize with a model
doc = interactive_document.InteractiveDocument(model)

# 2. Build Context
doc.statement("Alice enters the room.")
doc.statement("The room is dark.")

# 3. Interact
# The document automatically uses its current content as the prompt
answer = doc.open_question(
    question="What does Alice see?",
    max_tokens=50
)

# 4. Result is automatically appended to the history
print(doc.text())
# Output:
# Alice enters the room.
# The room is dark.
# Question: What does Alice see?
# Answer: Alice sees outline of furniture...
```

### 2. Views and Information Hiding

Concordia components often need to hide internal reasoning or debug information
from the model or other observers. This is handled via **Tags** and **Views**.

*   **Tags**: Annotate content with metadata strings. Common tags include:
    *   `'debug'`: Internal logs not meant for the model.
    *   `'memory'`: Retrieved memories relevant to the context.
    *   `'private'`: Thoughts or information known only to the agent.
    *   `'tool_call'`: Tool invocation requests (for tool-enabled docs).
    *   `'tool_result'`: Results from tool execution.
*   **View**: Create a dynamic window into the document that includes or
    excludes specific tags.

#### Example: Tagging and Filtering

```python
# 1. Add content with specific tags
doc.debug("Debug: Retrieved 5 relevant memories", tags=['debug'])
doc.statement("Context: Alice is hungry.", tags=['memory'])
doc.statement("Observation: There is an apple on the table.", tags=['observation'])
doc.statement("Private Thought: I hate apples.", tags=['private'])

# 2. Create a view for the Model (exclude debug info and private thoughts)
# This view will only show the memory and observation.
model_view = doc.view(exclude_tags=['debug', 'private'])

# 3. Create a view for a Supervisor (include everything except debug)
supervisor_view = doc.view(exclude_tags=['debug'])

# 4. Usage
# The model prompts with `model_view.text()`, seeing only:
# "Context: Alice is hungry. Observation: There is an apple on the table."
response = doc.open_question("What do you do?")
```

### 3. Branching and "What If" Scenarios

Because `Document` is immutable-ish (it grows but history is stable), you can
easily branch reasoning paths using `copy` or `edit`.

```python
# Create a temporary branch to test an outcome
with doc.edit() as temp_doc:
    temp_doc.statement("Alice turns on the light.")
    outcome = temp_doc.open_question("What happens next?")
    if "bomb" in outcome:
        # Decide NOT to turn on the light in the main timeline
        pass
    else:
        # Commit the action to the real document
        doc.statement("Alice turns on the light.")
```

### 4. Forced Responses (Scripting)

You can override LLM generation using `forced_response`. This is useful for:

*   **Testing**: Ensuring deterministic outcomes.
*   **Hybrid Systems**: Mixing scripted events with generated ones.
*   **Tutorials**: guiding the agent through a specific path.

```python
doc.open_question(
    "Who is the president?",
    forced_response="The president is a golden retriever."
)
# The model is NOT called, but the exchange is recorded in the document
# as if it had happened, influencing future generation.
```

### 5. Structured Questions

`InteractiveDocument` provides helpers for common structured tasks:

*   `yes_no_question(question)`: Returns boolean `True`/`False`.
*   `multiple_choice_question(question, answers)`: Returns the index of the
    selected answer.
*   `open_question_diversified(...)`: Generates multiple samples to avoid mode
    collapse or repetition.

## Idioms

*   **One Document per Agent/Component**: Typically, each agent has its own
    `InteractiveDocument` representing its subjective experience.
*   **Pass Views, Not Strings**: When components communicate, pass `doc.view()`
    instead of `doc.text()`. This delays string concatenation until the very
    last moment (lazy evaluation), saving performance and allowing for dynamic
    filtering.
*   **Scoped Edits**: Use the `with doc.edit() as branch:` context manager for
    all temporary reasoning or thought chains.
*   **Tool Isolation**: When using tools, consider filtering `tool_call` and
    `tool_result` tags from views shown to other components that don't need
    to see the internal tool mechanics.

## Tool Use

The `InteractiveDocumentWithTools` class enables LLMs to invoke external tools
during interactive sessions. This is useful for tasks requiring current
information (web search), calculations, or other external capabilities.

### Tool Protocol

Tools implement the `Tool` protocol (`tool.py`):

```python
from concordia.document.tool import Tool

class WebSearchTool:
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web. Args: query (str)"

    def execute(self, *, query: str) -> str:
        return search_web(query)
```

### Using Tools with Documents

```python
from concordia.document import interactive_document_tools

# Create document with tools
doc = interactive_document_tools.InteractiveDocumentWithTools(
    model=model,
    tools=[web_search_tool, calculator_tool],
    max_tool_calls_per_question=3,  # Budget per question
    max_tool_result_length=1000,    # Truncation limit
)

# Validation constraints:
# - max_tool_calls_per_question >= 1
# - max_tool_result_length >= 3
# - enforcement_mode in {"observe", "enforce"}

# Build context
doc.statement("You are a helpful research assistant.")

# Ask a question - LLM may call tools automatically
answer = doc.open_question("What is the current price of Bitcoin?")

# Tool calls and results are recorded in the document
print(doc.text())
# Output includes:
# Question: What is the current price of Bitcoin?
# Answer: {"tool": "web_search", "args": {"query": "bitcoin price"}}
# [Tool Call: web_search({"query": "bitcoin price"})]
# [Tool Result: Bitcoin is currently trading at $45,000...]
# Answer: Bitcoin is currently trading at approximately $45,000.
```

### Tool Call Format

The LLM uses JSON to request tool calls:

```json
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
```

Results are truncated to `max_tool_result_length` and recorded with
`tool_call` and `tool_result` tags for filtering.

### Policy-Aware Tool Execution

`InteractiveDocumentWithTools` supports optional policy evaluation for each tool
call.

```python
from concordia.document import interactive_document_tools
from concordia.document import tool_policy

doc = interactive_document_tools.InteractiveDocumentWithTools(
    model=model,
    tools=[web_search_tool],
    policy=tool_policy.SchemaValidatingPolicy(),
    enforcement_mode='observe',  # default
)
```

The policy lifecycle for each tool call is:

1. Parse model output as `{"tool": ..., "args": ...}`.
2. Build `tool_policy.ToolCall`.
3. Evaluate configured policy.
4. Apply decision to execution based on enforcement mode.
5. Record tool call/result/policy notes in the document.

#### Observe vs Enforce

*   **`observe` (default)**:
    *   policy decisions are logged
    *   `DENY` and `EDIT` decisions do not block execution
    *   malformed policy output fails open (tool still executes)
*   **`enforce`**:
    *   policy decisions are applied
    *   `DENY` blocks execution
    *   `EDIT` executes with edited args
    *   malformed policy output fails closed (tool call blocked)

#### Policy Tags

Policy interactions are recorded under `tool_policy` with one of these tags:

*   `tool_policy_allow`
*   `tool_policy_deny_observed`
*   `tool_policy_edit_observed`
*   `tool_policy_deny_enforced`
*   `tool_policy_edit_enforced`
*   `tool_policy_error_observed`
*   `tool_policy_error_enforced`

#### Schema Helper

Tools can optionally expose `input_schema` metadata. Use
`tool_policy.SchemaValidatingPolicy` to validate tool args before execution.

```python
from concordia.document import tool
from concordia.document import tool_policy

class WeatherTool(tool.Tool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get weather for a city."

    @property
    def input_schema(self):
        return {
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}},
        }

    def execute(self, *, city: str) -> str:
        return f"Weather for {city}"

policy = tool_policy.SchemaValidatingPolicy()
```

`validate_input_schema(...)` intentionally supports a small stable subset:

*   top-level `type: "object"`
*   `required`
*   primitive property types: `string`, `number`, `integer`, `boolean`,
    `object`, `array`

Unsupported schema keywords are ignored by design.
