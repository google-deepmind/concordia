<!-- disableFinding("Master") -->
<!-- disableFinding("master") -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

<!-- Relative links in this file follow the GitHub repository layout, where
     examples/ sits next to concordia/ at the repository root. -->

# Concordia Examples

Runnable simulations, games, demos, and tutorials built on Concordia. Run the
commands below from the repository root. Each `run.py` accepts any provider
supported by `concordia.contrib.language_models` through `--api_type` and
`--model_name`, and has an offline mode that needs no API key
(`--disable_language_model`; `--use_mock=True` for Concordia Island).

Want to add one? Read the [contribution and style guide](STYLE_GUIDE.md) and
add a row to the relevant table below as part of your pull request.

## Tutorials (notebooks)

| Notebook | What it shows |
| --- | --- |
| [tutorial.ipynb](tutorial.ipynb) | Configure and run a first simulation: four friends snowed in at a pub, with agents that act by asking what the situation is, who they are, and what someone like them would do. |
| [alice.ipynb](alice.ipynb) | Narrative simulation driven by a premise (Alice and the White Rabbit). |
| [dialog.ipynb](dialog.ipynb) | Pure conversation between agents with the `dialogic` Game Master. |
| [selling_cookies.ipynb](selling_cookies.ipynb) | Scenes with payoffs: a conversation phase followed by a binary decision, scored by the game-theoretic Game Master. |
| [marketplace.ipynb](marketplace.ipynb) | Economic simulation with the `marketplace` Game Master (buying, selling, inventory). |
| [questionnaire_example.ipynb](questionnaire_example.ipynb) | Administer psychometric questionnaires with the `interviewer` Game Master. |
| [actor_development.ipynb](actor_development.ipynb) | Build a custom agent prefab from components. |
| [mcp_tool_use_example.ipynb](mcp_tool_use_example.ipynb) | Agents calling external tools through an MCP server. |

## Simulations

| Example                                                           | What it models   | Run                                            |
| ----------------------------------------------------------------- | ---------------- | ---------------------------------------------- |
| [concordia_island](concordia_island/)                             | A town of 100+   | `python -m examples.concordia_island.run ...`  |
:                                                                   : agents with      : (see its README)                               :
:                                                                   : homes, jobs, and :                                                :
:                                                                   : social life;     :                                                :
:                                                                   : shocks such as   :                                                :
:                                                                   : job loss or a    :                                                :
:                                                                   : mugging;         :                                                :
:                                                                   : repeated         :                                                :
:                                                                   : psychometric     :                                                :
:                                                                   : surveys; a map   :                                                :
:                                                                   : and memory       :                                                :
:                                                                   : dashboard.       :                                                :
| [signaling](signaling/)                                           | A multi-day      | `python -m examples.signaling.run`             |
:                                                                   : marketplace      :                                                :
:                                                                   : followed by      :                                                :
:                                                                   : dates\:          :                                                :
:                                                                   : conspicuous      :                                                :
:                                                                   : consumption and  :                                                :
:                                                                   : status           :                                                :
:                                                                   : signaling.       :                                                :
| [resource_dilemma](resource_dilemma/)                             | Common-pool      | `python -m examples.resource_dilemma.run`      |
:                                                                   : resource         :                                                :
:                                                                   : harvesting       :                                                :
:                                                                   : (pasture,        :                                                :
:                                                                   : irrigation,      :                                                :
:                                                                   : network,         :                                                :
:                                                                   : fishery) with    :                                                :
:                                                                   : and without      :                                                :
:                                                                   : elected leaders. :                                                :
| [social_media](social_media/)                                     | Agents posting   | `python -m examples.social_media.run`          |
:                                                                   : and replying on  :                                                :
:                                                                   : a Reddit-like    :                                                :
:                                                                   : forum under the  :                                                :
:                                                                   : asynchronous     :                                                :
:                                                                   : engine; several  :                                                :
:                                                                   : scenarios.       :                                                :
| [general_store](general_store/)                                   | Staff dynamics,  | `python -m examples.general_store.run`         |
:                                                                   : a theft          :                                                :
:                                                                   : investigation,   :                                                :
:                                                                   : and social       :                                                :
:                                                                   : manipulation in  :                                                :
:                                                                   : a general store. :                                                :
| [conversation_with_ai_companion](conversation_with_ai_companion/) | A person talking | `python -m                                     |
:                                                                   : to an AI         : examples.conversation_with_ai_companion.run`   :
:                                                                   : assistant that   :                                                :
:                                                                   : has motives of   :                                                :
:                                                                   : its own (exam    :                                                :
:                                                                   : prep; a tutor    :                                                :
:                                                                   : that upsells).   :                                                :
| [deep_time](deep_time/)                                           | A SimEarth-style | `python -m examples.deep_time.run --mode=mock` |
:                                                                   : planet on        : (play\: `python -m                             :
:                                                                   : Concordia's      : examples.deep_time.web.server`; see its        :
:                                                                   : pub/sub\: a      : README)                                        :
:                                                                   : Steward reads    :                                                :
:                                                                   : late, noisy      :                                                :
:                                                                   : instruments and  :                                                :
:                                                                   : nudges the       :                                                :
:                                                                   : climate while    :                                                :
:                                                                   : species, herds   :                                                :
:                                                                   : and peoples act  :                                                :
:                                                                   : on a tile map,   :                                                :
:                                                                   : across eleven    :                                                :
:                                                                   : scenarios        :                                                :
:                                                                   : through deep     :                                                :
:                                                                   : time; playable   :                                                :
:                                                                   : on a phone.      :                                                :
| [interrupt_driven](interrupt_driven/)                             | Worked examples  | `python -m                                     |
:                                                                   : of the           : examples.interrupt_driven.scheduling_examples` :
:                                                                   : interrupt-driven :                                                :
:                                                                   : Game Master\:    :                                                :
:                                                                   : events, polling  :                                                :
:                                                                   : masks, and       :                                                :
:                                                                   : timers.          :                                                :

## Games

All games share one runner, `python -m examples.games.run --game=<name>
[--scenario=<config>]`, and each ships a `puppet` scenario that runs without a
language model.

| Game | What it models |
| --- | --- |
| [haggling](games/haggling/) | Buyers and sellers negotiating fruit prices: propose, then accept or reject. |
| [haggling_multi_item](games/haggling_multi_item/) | Haggling over several goods at once. |
| [pub_coordination](games/pub_coordination/) | Friends deciding which pub to meet at, with conflicting preferences. |
| [reality_show](games/reality_show/) | Contestants playing matrix-game minigames (prisoner's dilemma, chicken, stag hunt) between conversations. |
| [labor_collective_action](games/labor_collective_action/) | Workers deciding daily whether to strike; a boss deciding whether to raise wages. |
| [state_formation](games/state_formation/) | Elders of two villages negotiating a defense treaty while running their villages. |
| [social_deception](games/social_deception/) | Hidden-role social deduction with 22 roles, night and day phases, and noisy information. |
