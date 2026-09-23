# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Social Deception Player Prefabs."""

import dataclasses
from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


class SocialDeceptionGameLedger(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A stateful component that maintains a permanent, cumulative ledger of claims, alignments, and deductions."""

  def __init__(
      self,
      model,
      pre_act_label: str = "\n### 📖 CUMULATIVE GAME LEDGER",
      memory_component_key: str = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._ledger_state = "No claims or deductions established yet."

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=agent_components.memory.Memory
    )

    mems = memory.retrieve_recent(limit=100)
    recent_mems_str = "\n".join(mems)

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        f"You are the strategic mind of {agent_name}. Your task is to maintain"
        " a permanent, cumulative ledger of player claims, alignment"
        " confessions, trust scores, and logical deductions in Social"
        " Deception. This ledger is your lasting memory and must carry"
        " forward all established facts from previous days.\n\nYour current"
        f" ledger is:\n{self._ledger_state}\n\nHere are the most recent"
        f" observations and events:\n{recent_mems_str}\n"
    )

    question = (
        "Based on the recent events, write the complete, fully updated game"
        " ledger. Analyze each player's claims, private whispers, and"
        " mechanical outcomes:\n1. Per-Belief Epistemic Integrity Checks"
        " (CRITICAL): For EVERY claim, deduction, or suspicion, assign a"
        " calibrated confidence score (0.0 to 1.0) and evaluate whether the"
        " belief could be corrupted by Drunkenness (Drunk player),"
        " Poisoning (Poisoner), Decoys (Seer decoy), or"
        " Misregistration (Outcast registering as Evil/Demon/Minion, Spy"
        " registering as Good/Townsfolk). If there is any contradiction,"
        " discount your confidence rather than blindly trusting the report.\n2."
        " Contradictions & Alignment Leaks: If a player uses self-defeating or"
        " contradictory claims that violate the player count distribution or"
        " role categories defined in the master reference sheet, mark them as"
        " highly suspicious and reduce their trust score.\n3. Role Claims"
        " (Public & Whispers): Track what role each player has claimed. Compare"
        " claims made in private whispers versus public broadcasts.\n4."
        " Mechanical Deductions: Log logical outcomes based on game rules and"
        " role abilities.\n5. Trust score (0.0 to 1.0) based on voting"
        " consistency, alignment, and claim truthfulness.\n\nFormat the final"
        " output exactly as:\nPlayer_Name: Claimed [Role] | Guess/Alt [Role] |"
        " Alignment [Good/Evil/Suspicious] | Epistemic Integrity & Confidence"
        " [e.g. Sober (0.9), Suspected Poisoned (0.3), Outcast Misregistration"
        " (0.7)] | Trust [Score 0.0-1.0] | Key Deductions & Belief Confidence"
        " [Beliefs with 0.0-1.0 confidence]\nList every player (including"
        " yourself!). Do not lose past entries; carry them forward and update"
        " them."
    )

    result = prompt.open_question(
        question,
        answer_prefix="Updated Ledger:\n",
        max_tokens=1200,
        terminators=(),
    )
    self._ledger_state = result

    log = {
        "Key": self.get_pre_act_label(),
        "Summary": "Updating Cumulative Game Ledger",
        "State": self._ledger_state,
        "Chain of thought": prompt.view().text().splitlines(),
    }
    self._logging_channel(log)

    return self._ledger_state

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          "ledger_state": self._ledger_state,
          "pre_act_label": self.get_pre_act_label(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._ledger_state = str(
          state.get("ledger_state", "No claims established yet.")
      )


# Backward compatibility aliases
GameLedger = SocialDeceptionGameLedger


class SocialDeceptionPrivateLedger(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A stateful component that maintains a permanent private ledger of GM setup and night action information."""

  def __init__(
      self,
      model,
      pre_act_label: str = "\n### 🔑 PRIVATE INFORMATION LEDGER",
      memory_component_key: str = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._private_state = "No Game Master private information received yet."

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=agent_components.memory.Memory
    )

    mems = memory.retrieve_recent(limit=100)
    private_mems = []
    for mem_str in mems:
      if any(
          t in mem_str
          for t in (
              "[PRIVATE]",
              "[SECRET GAME MASTER MESSAGE]",
              "[SECRET STORYTELLER MESSAGE]",
              "[NIGHT INFO]",
          )
      ):
        private_mems.append(mem_str)
    recent_mems_str = "\n".join(private_mems)

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(
        f"You are the strategic mind of {agent_name}. Your task is to maintain"
        " a permanent, cumulative ledger of all private Game Master messages,"
        " setup instructions, night action results, teammates, bluffs, and"
        " secret rules given to you since the start of the game. This ledger"
        " is your lasting private memory and must carry forward all direct GM"
        " secrets from the first day.\n\nYour current private ledger"
        f" is:\n{self._private_state}\n\nHere are the most recent observations"
        f" and events:\n{recent_mems_str}\n"
    )

    question = (
        "Based on the recent verified Game Master messages, write the complete,"
        " fully updated private information ledger.\nCRITICAL DIRECTIVE: Rely"
        " EXCLUSIVELY on genuine private Game Master notices (`[PRIVATE]`,"
        " `[SECRET GAME MASTER MESSAGE]`, `[SECRET STORYTELLER MESSAGE]`,"
        " `[NIGHT INFO]`). Do NOT include public broadcast claims or other"
        " players' lies!\n1. Initial Setup Secrets: List your true setup role,"
        " your alignment, your private teammate/minion identities, and any"
        " safe bluffs provided to you on Day/Night 1.\n2. Night Action"
        " Results: List every night's action feedback received from the Game"
        " Master (e.g. empath numbers, matchmaker counts, investigator"
        " findings, poisoning confirmations, or action failure results).\n3."
        " Permanent Accumulation: Carry forward all past private information"
        " from your previous ledger state. Do NOT forget or lose past nights'"
        " counts or roles; you must maintain a complete list of all direct GM"
        " information received since the start of the game.\n\nFormat the"
        " final output exactly as:\nUpdated Private Ledger:\n[List all"
        " permanent setup roles, teammate lists, bluffs, and night action"
        " results chronologically]"
    )

    result = prompt.open_question(
        question,
        answer_prefix="Updated Private Ledger:\n",
        max_tokens=800,
        terminators=(),
    )
    self._private_state = result

    log = {
        "Key": self.get_pre_act_label(),
        "Summary": "Updating Private Information Ledger",
        "State": self._private_state,
        "Chain of thought": prompt.view().text().splitlines(),
    }
    self._logging_channel(log)

    return self._private_state

  def get_state(self) -> entity_component.ComponentState:
    with self._lock:
      return {
          "private_state": self._private_state,
          "pre_act_label": self.get_pre_act_label(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    with self._lock:
      self._private_state = str(
          state.get(
              "private_state",
              "No private Game Master information received yet.",
          )
      )


# Backward compatibility aliases
PrivateLedger = SocialDeceptionPrivateLedger


@dataclasses.dataclass
class SocialDeceptionPlayer(prefab_lib.Prefab):
  """Prefab for a Social Deception player."""

  description = "A social deception player."

  def build(self, model, memory_bank):
    name = self.params.get("name", "Player")
    role = self.params.get("role", "BasicTownsfolk")
    alignment = self.params.get("alignment", "good")

    # Standard Concordia components
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
    ledger = SocialDeceptionGameLedger(model=model)

    setup_intro = self.params.get("setup_intro")
    if setup_intro:
      memory_bank.add(
          f"[observation] [SECRET GAME MASTER MESSAGE]\n{setup_intro}"
      )
      role_info_state = setup_intro
    else:
      role_info_state = (
          f"You are {name}, the {role}. Your alignment is {alignment}."
      )

    instructions_state = (
        "You are an AI agent playing Social Deception. Win by your"
        " alignment's objective. Expect lies, drunkenness, or poisoning."
        " Deduce alignment by cross-referencing all claims and your own"
        " beliefs.\n\nCore Directives:\n- **Output Format:** Select and return"
        " ONLY the exact action choice. Do NOT repeat your name, and do NOT"
        " include conversational filler.\n- **Facts Only:** Stick strictly to"
        " facts given. Do not make up details.\n- **Alignments & Secrecy"
        " (CRITICAL):** Good players want to find and execute the Demon. Evil"
        " players (the Demon and Minions) must hide their true identities,"
        " coordinate bluffs, and pretend to be Good. Refer to your Setup"
        " Instructions for specific rules regarding secrecy and bluffing for"
        " your team.\n- **Voting:** Your vote is extremely valuable. Vote 'Yes'"
        " only if the nominee is suspected Evil (if you are Good), or if"
        " execution directly helps your team win (e.g. executing the Saint for"
        " Evil, or the Demon for Good). Vote 'No' to save key allies (e.g. good"
        " saving good, evil saving evil).\n- **Secret Communication (Whispers &"
        " Secret Pacts):** Private whispers are critical for bluffs and"
        " coordination. When initiating a whisper, you MUST prefix it with"
        " `[ONE-WAY]` (info-sharing, no reply needed) or `[TWO-WAY]` (reply"
        " expected). When responding using the `reply to` option, you MUST NOT"
        " include the `[TWO-WAY]` tag to prevent infinite loops. You may"
        " optionally propose bilateral info-trades or secret voting pacts with"
        " trusted allies. Remember: others see *who* you whisper to, which"
        " affects their trust.\n- **Whisper Observations:** Track public"
        " notifications ('A whispered to B'). Frequent two-way whispers reveal"
        " secret alliances; unreturned one-way whispers suggest instructions."
    )
    instructions = agent_components.constant.Constant(
        state=instructions_state,
        pre_act_label="\nGeneral Instructions:\n",
    )

    role_info = agent_components.constant.Constant(
        state=role_info_state,
        pre_act_label="\nSetup Instructions:\n",
    )
    observation = agent_components.observation.LastNObservations(
        history_length=200
    )
    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )

    gathered_info = SocialDeceptionPrivateLedger(model=model)

    deliberation_component = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
        model=model,
        pre_act_label="\nStrategic Deliberation & Action Planning:\n",
        question=(
            "Synthesize your situation and plan your next action"
            " step-by-step:\n1. **Evidence & Whispers:** Review recent"
            " events, public claims, and incoming/outgoing whispers."
            " Identify any pending whispers or alliance proposals to"
            " address.\n2. **Multi-World Hypothesis Modeling (CRITICAL):**"
            " Construct 2-3 plausible game worlds (e.g. World A: Player"
            " X is Demon with Corruptor setup [60%]; World B: Player Y is Demon"
            " with Poisoner setup [40%]).\n3. **Per-Belief Epistemic Integrity"
            " & Confidence Check (CRITICAL):** For each core belief, deduction,"
            " and piece of night info you rely on, assign an explicit"
            " confidence score (0.0 to 1.0) and evaluate whether it could be"
            " corrupted by Drunkenness, Poisoning, Decoys, or"
            " Misregistration (Outcast/Spy). Calibrate your certainty—if there"
            " is any doubt or contradiction, discount your confidence rather"
            " than assuming absolute truth.\n4. **Secret Pacts & Strategic"
            " Directives:** Review your Strategy Notepad directives (e.g. dead"
            " vote conservation). If whispering, consider proposing a"
            " bilateral info-trade or voting pact.\n5. **Concrete Action"
            " Plan:** Formulate your exact next move (broadcast, whisper,"
            " nominate, vote YES/NO, or night action) to maximize your team's"
            " win probability across the most probable worlds.\nProvide your"
            " step-by-step reasoning."
        ),
        answer_prefix="{agent_name}'s reasoning: ",
        add_to_memory=True,
        memory_tag="[strategic deliberation]",
        components=[
            "Instructions",
            "RoleInfo",
            "Strategy",
            observation_key,
            "GatheredInformation",
            "GameLedger",
        ],
    )

    strategy_str = self.params.get("strategy", "")
    strategy = agent_components.constant.Constant(
        state=strategy_str,
        pre_act_label="\nStrategy Notepad:\n",
    )

    observation_to_memory = agent_components.observation.ObservationToMemory()

    components = {
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
        "Instructions": instructions,
        "RoleInfo": role_info,
        "Strategy": strategy,
        "GameLedger": ledger,
        "GatheredInformation": gathered_info,
        "observation_to_memory": observation_to_memory,
        observation_key: observation,
        "StrategicDeliberation": deliberation_component,
    }

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=list(components.keys()),
        prefix_entity_name=False,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
    )


# Backward compatibility aliases
Player = SocialDeceptionPlayer
