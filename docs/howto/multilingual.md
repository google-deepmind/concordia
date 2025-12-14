
# Multilingual output (how-to)

**Version:** `gdm-concordia >= 2.1.0`

Concordia agents don’t have a global “language” toggle. The recommended way to constrain output language is to **modify the agent’s prompt**. You can do this either by:
- injecting a language directive into the agent’s params (e.g., `goal`), or
- wrapping/templating the prompt used by your prefab.

Below is a minimal example that sets one agent to Spanish and another to French.

```python
from concordia.utils import prefab_lib

LANGUAGE_DIRECTIVE = "Always respond in {lang}. Do not switch languages."

instances = [
    prefab_lib.InstanceConfig(
        prefab="basic__Entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": "Oliver Cromwell",
            "goal": "become lord protector. " + LANGUAGE_DIRECTIVE.format(lang="Spanish"),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="basic__Entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": "King Charles I",
            "goal": "avoid execution for treason. " + LANGUAGE_DIRECTIVE.format(lang="French"),
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="generic__GameMaster",
        role=prefab_lib.Role.GAME_MASTER,
        params={
            "name": "default rules",
            # Comma-separated list of thought chain steps.
            "extra_event_resolution_steps": "",
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="formative_memories_initializer__GameMaster",
        role=prefab_lib.Role.INITIALIZER,
        params={
            "name": "initial setup rules",
            "next_game_master_name": "default rules",
            "shared_memories": [
                "The king was captured by Parliamentary forces in 1646.",
                "Charles I was tried for treason and found guilty.",
            ],
        },
    ),
]

config = prefab_lib.Config(
    default_premise="Today is January 29, 1649.",
    default_max_steps=5,
    instances=instances,
)



eof
