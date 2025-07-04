# Copyright 2025 DeepMind Technologies Limited.
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

"""An adaptable simulation prefab that can be configured to run any simulation.
"""

from collections.abc import Callable, Mapping
import copy
import functools
import json
import os
from typing import Any

from concordia.associative_memory import basic_associative_memory as associative_memory
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import simulation as simulation_lib
from concordia.utils import helper_functions as helper_functions_lib
from concordia.utils import html as html_lib
import numpy as np


Config = prefab_lib.Config
Role = prefab_lib.Role


class Simulation(simulation_lib.Simulation):
  """Define the simulation API object."""

  def __init__(
      self,
      config: Config,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      engine: engine_lib.Engine = sequential.Sequential(),
  ):
    """Initialize the simulation object.

    This simulation differentiates between game masters and entities. Game
    masters are responsible for creating the world state and resolving events.
    Entities are passive agents that react to the world state. Game masters are
    a kind of entity and both are interchangeable once instantiated. However,
    there is one critical difference in how they are configured which we
    implement here in this file. The difference is that game masters are
    configured with references to all entities, but entities are never
    given references to other entities or game masters.

    Args:
      config: the config to use.
      model: the language model to use.
      embedder: the sentence transformer to use.
      engine: the engine to use, defaults to sequential.Sequential().
    """
    self._config = config
    self._model = model
    self._embedder = embedder
    self._engine = engine
    self.game_masters = []
    self.entities = []
    self._entity_to_prefab_config: dict[str, prefab_lib.InstanceConfig] = {}
    self._checkpoints_path = None

    # All game masters share the same memory bank.
    self.game_master_memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )
    all_data = self._config.instances
    gm_configs = [
        entity_cfg
        for entity_cfg in all_data
        if entity_cfg.role == Role.GAME_MASTER
    ]
    entities_configs = [
        entity_cfg for entity_cfg in all_data if entity_cfg.role == Role.ENTITY
    ]
    initializer_configs = [
        entity_cfg
        for entity_cfg in all_data
        if entity_cfg.role == Role.INITIALIZER
    ]

    for entity_config in entities_configs:
      self.add_entity(entity_config)

    for gm_config in initializer_configs + gm_configs:
      self.add_game_master(gm_config)

  def get_entity_prefab_config(
      self, entity_name: str
  ) -> prefab_lib.InstanceConfig | None:
    """Get the prefab config for a given entity name."""
    return self._entity_to_prefab_config.get(entity_name)

  def get_game_masters(self) -> list[entity_lib.Entity]:
    """Get the game masters.

    The function returns a copy of the game masters list to avoid modifying the
    original list. However, the game masters are not deep copied, so changes
    to the game masters will be reflected in the simulation.

    Returns:
      A list of game master entities.
    """
    return copy.copy(self.game_masters)

  def get_entities(self) -> list[entity_lib.Entity]:
    """Get the entities.

    The function returns a copy of the entities list to avoid modifying the
    original list. However, the entities are not deep copied, so changes
    to the entities will be reflected in the simulation.

    Returns:
      A list of entities.
    """
    return copy.copy(self.entities)

  def add_game_master(
      self,
      instance_config: prefab_lib.InstanceConfig,
      state: entity_component.EntityState | None = None,
  ):
    """Add a game master to the simulation."""
    if instance_config.role not in [Role.GAME_MASTER, Role.INITIALIZER]:
      raise ValueError(
          "Instance config role must be GAME_MASTER or INITIALIZER"
      )

    game_master_prefab = copy.deepcopy(
        self._config.prefabs[instance_config.prefab]
    )
    game_master_prefab.params = instance_config.params
    game_master_prefab.entities = self.entities
    game_master = game_master_prefab.build(
        model=self._model, memory_bank=self.game_master_memory_bank
    )

    if any(gm.name == game_master.name for gm in self.game_masters):
      print(f"Game master {game_master.name} already exists.")
      return

    if state:
      game_master.set_state(state)

    self._entity_to_prefab_config[game_master.name] = instance_config

    self.game_masters.append(game_master)

  def add_entity(
      self,
      instance_config: prefab_lib.InstanceConfig,
      state: entity_component.EntityState | None = None,
  ):
    """Add an entity to the simulation."""
    if instance_config.role != Role.ENTITY:
      raise ValueError("Instance config role must be ENTITY")

    entity_prefab = copy.deepcopy(self._config.prefabs[instance_config.prefab])
    entity_prefab.params = instance_config.params

    memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=self._embedder,
    )
    entity = entity_prefab.build(model=self._model, memory_bank=memory_bank)

    if any(e.name == entity.name for e in self.entities):
      print(f"Entity {entity.name} already exists.")
      return

    if state:
      entity.set_state(state)

    self.entities.append(entity)
    self._entity_to_prefab_config[entity.name] = instance_config

    # Update game masters to be aware of the new entity
    for game_master in self.game_masters:
      if hasattr(game_master, "entities"):
        game_master.entities = self.entities

  def play(
      self,
      premise: str | None = None,
      max_steps: int | None = None,
      raw_log: list[Mapping[str, Any]] | None = None,
      checkpoint_path: str | None = None,
  ) -> str:
    """Run the simulation.

    Args:
      premise: A string to use as the initial premise of the simulation.
      max_steps: The maximum number of steps to run the simulation for.
      raw_log: A list to store the raw log of the simulation. This is used to
        generate the HTML log. Data in the supplied raw_log will be appended
        with the log from the simulation. If None, a new list is created.
      checkpoint_path: The path to save the checkpoints. If None, no checkpoints
        are saved.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    if premise is None:
      premise = self._config.default_premise
    if max_steps is None:
      max_steps = self._config.default_max_steps

    raw_log = raw_log or []

    checkpoint_callback = None
    if checkpoint_path:
      checkpoint_callback = functools.partial(
          self.save_checkpoint, checkpoint_path=checkpoint_path
      )

    # Ensure game masters are ordered Initializers first
    initializers = [
        gm
        for gm in self.game_masters
        if self._entity_to_prefab_config[gm.name].role == Role.INITIALIZER
    ]
    other_gms = [
        gm
        for gm in self.game_masters
        if self._entity_to_prefab_config[gm.name].role == Role.GAME_MASTER
    ]
    sorted_game_masters = initializers + other_gms

    self._engine.run_loop(
        game_masters=sorted_game_masters,
        entities=self.entities,
        premise=premise,
        max_steps=max_steps,
        verbose=True,
        log=raw_log,
        checkpoint_callback=checkpoint_callback,
    )

    player_logs = []
    player_log_names = []

    scores = helper_functions_lib.find_data_in_nested_structure(
        raw_log, "Player Scores"
    )

    for player in self.entities:
      if (
          not isinstance(player, entity_component.EntityWithComponents)
          or player.get_component("__memory__") is None
      ):
        continue

      entity_memory_component = player.get_component("__memory__")
      entity_memories = entity_memory_component.get_all_memories_as_text()
      player_html = html_lib.PythonObjectToHTMLConverter(
          entity_memories
      ).convert()
      player_logs.append(player_html)
      player_log_names.append(f"{player.name}")

    game_master_memories = (
        self.game_master_memory_bank.get_all_memories_as_text()
    )
    game_master_html = html_lib.PythonObjectToHTMLConverter(
        game_master_memories
    ).convert()
    player_logs.append(game_master_html)
    player_log_names.append("Game Master Memories")
    summary = ""
    if scores:
      summary = f"Player Scores: {scores[-1]}"
    results_log = html_lib.PythonObjectToHTMLConverter(
        copy.deepcopy(raw_log)
    ).convert()
    tabbed_html = html_lib.combine_html_pages(
        [results_log, *player_logs],
        ["Game Master log", *player_log_names],
        summary=summary,
        title="Simulation Log",
    )
    html_results_log = html_lib.finalise_html(tabbed_html)
    return html_results_log

  def save_checkpoint(self, step: int, checkpoint_path: str):
    """Saves the state of all entities at the current step."""
    if not checkpoint_path:
      return

    checkpoint_data = {
        "entities": {},
        "game_masters": {},
    }

    # Save entities
    for entity in self.entities:
      if not isinstance(entity, entity_component.EntityWithComponents):
        continue
      prefab_config = self.get_entity_prefab_config(entity.name)
      if not prefab_config:
        print(f"Warning: Prefab config not found for entity {entity.name}")
        continue
      entity_state = entity.get_state()
      save_data = {
          "prefab_type": prefab_config.prefab,
          "entity_params": prefab_config.params,
          "components": entity_state,
      }
      checkpoint_data["entities"][entity.name] = save_data

    # Save game masters
    for gm in self.game_masters:
      if not isinstance(gm, entity_component.EntityWithComponents):
        continue
      prefab_config = self.get_entity_prefab_config(gm.name)
      if not prefab_config:
        print(f"Warning: Prefab config not found for game master {gm.name}")
        continue
      gm_state = gm.get_state()
      save_data = {
          "prefab_type": prefab_config.prefab,
          "entity_params": prefab_config.params,
          "role": self._entity_to_prefab_config[gm.name].role.name,
          "components": gm_state,
      }
      checkpoint_data["game_masters"][gm.name] = save_data

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(
        checkpoint_path, f"step_{step}_checkpoint.json"
    )
    try:
      with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
      print(f"Step {step}: Saved checkpoint to {checkpoint_file}")
    except IOError as e:
      print(f"Error saving checkpoint at step {step}: {e}")

  def load_from_checkpoint(
      self,
      checkpoint: dict[str, Any],
  ):
    """Loads entity and game master states from a checkpoint dict."""

    # Load entities
    entity_states = checkpoint.get("entities", {})
    for entity_name, state in entity_states.items():
      self._load_entity_from_state(entity_name, state, Role.ENTITY)

    # Load game masters
    gm_states = checkpoint.get("game_masters", {})
    for gm_name, state in gm_states.items():
      role_name = state.get("role")
      try:
        role = (
            Role[role_name]
            if role_name in Role.__members__
            else Role.GAME_MASTER
        )
      except KeyError:
        print(
            f"Warning: Invalid role {role_name} for {gm_name}, using"
            " GAME_MASTER."
        )
        role = Role.GAME_MASTER
      self._load_entity_from_state(gm_name, state, role)

    # Important: Update game masters to be aware of any new entities
    for game_master in self.game_masters:
      if hasattr(game_master, "entities"):
        game_master.entities = self.entities

  def _load_entity_from_state(
      self,
      entity_name: str,
      state: dict[str, Any],
      default_role: Role,
  ):
    """Helper to load a single entity or game master from state."""
    prefab_type = state.get("prefab_type")
    entity_params = state.get("entity_params")
    entity_components_state = state.get("components")

    if not isinstance(prefab_type, str):
      print(f"Warning: Prefab type is not a string for {entity_name}.")
      return
    if not prefab_type or prefab_type not in self._config.prefabs:
      print(f"Warning: Prefab type {prefab_type} not found for {entity_name}.")
      return
    if entity_params is None or entity_components_state is None:
      print(f"Warning: Missing params or components state for {entity_name}.")
      return

    instance_config = prefab_lib.InstanceConfig(
        prefab=prefab_type,
        role=default_role,
        params=entity_params,
    )

    if default_role == Role.ENTITY:
      existing_entity = next(
          (e for e in self.entities if e.name == entity_name), None
      )
      if existing_entity:
        if isinstance(existing_entity, entity_component.EntityWithComponents):
          print(f"Updating existing entity {entity_name} from checkpoint.")
          existing_entity.set_state(entity_components_state)
      else:
        print(f"Adding new entity {entity_name} from checkpoint.")
        self.add_entity(instance_config, state=entity_components_state)
    elif default_role in [Role.GAME_MASTER, Role.INITIALIZER]:
      existing_gm = next(
          (gm for gm in self.game_masters if gm.name == entity_name), None
      )
      if existing_gm:
        if isinstance(existing_gm, entity_component.EntityWithComponents):
          print(f"Updating existing game master {entity_name} from checkpoint.")
          existing_gm.set_state(entity_components_state)
      else:
        print(f"Adding new game master {entity_name} from checkpoint.")
        self.add_game_master(instance_config, state=entity_components_state)
