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
from typing import Any

from concordia.associative_memory import basic_associative_memory as associative_memory
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
    """
    self._config = config
    self._model = model
    self._embedder = embedder
    self._environment = sequential.Sequential()

    self.game_masters = []
    self.entities = []

    initializer_game_master_configs = []
    non_initializer_game_master_configs = []
    entity_configs = []

    all_data = self._config.instances
    game_masters_data = [entity_cfg for entity_cfg in all_data
                         if entity_cfg.role == Role.GAME_MASTER]
    entities_data = [entity_cfg for entity_cfg in all_data
                     if entity_cfg.role == Role.ENTITY]
    initializers_data = [entity_cfg for entity_cfg in all_data
                         if entity_cfg.role == Role.INITIALIZER]

    # Create a copy of each prefab using the params specified in its instance.
    for config in game_masters_data + entities_data + initializers_data:
      # Deep copy the prefab to avoid modifying the original prefab.
      entity_config = copy.deepcopy(self._config.prefabs[config.prefab])
      entity_config.params = config.params

      if config.role == "game_master":
        non_initializer_game_master_configs.append(entity_config)
      elif config.role == "initializer":
        initializer_game_master_configs.append(entity_config)
      else:
        entity_configs.append(entity_config)

    game_master_configs = (initializer_game_master_configs +
                           non_initializer_game_master_configs)

    # All game masters share the same memory bank.
    self.game_master_memory_bank = associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

    # Build all entities.
    for entity_config in entity_configs:
      # Each entity gets their own memory bank, not shared with other entities.
      memory_bank = associative_memory.AssociativeMemoryBank(
          sentence_embedder=embedder,
      )
      entity = entity_config.build(model=model,
                                   memory_bank=memory_bank)
      self.entities.append(entity)
    for game_master_config in game_master_configs:
      # Pass references to entities to all game masters, but never pass entities
      # to non-game-master entities.
      game_master_config.entities = self.entities
      # Pass a reference to the shared memory bank to all game masters.
      game_master = game_master_config.build(
          model=model, memory_bank=self.game_master_memory_bank)
      self.game_masters.append(game_master)

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

  def add_game_master(self, game_master: entity_lib.Entity):
    """Add a game master to the simulation."""
    self.game_masters.append(game_master)

  def add_entity(self, entity: entity_lib.Entity):
    """Add an entity to the simulation."""
    self.entities.append(entity)

  def play(
      self,
      premise: str | None = None,
      max_steps: int | None = None,
      raw_log: list[Mapping[str, Any]] | None = None,
  ) -> str:
    """Run the simulation.

    Args:
      premise: A string to use as the initial premise of the simulation.
      max_steps: The maximum number of steps to run the simulation for.
      raw_log: A list to store the raw log of the simulation. This is used to
        generate the HTML log. Data in the supplied raw_log will be appended
        with the log from the simulation. If None, a new list is created.

    Returns:
      html_results_log: browseable log of the simulation in HTML format
    """
    if premise is None:
      premise = self._config.default_premise
    if max_steps is None:
      max_steps = self._config.default_max_steps

    raw_log = raw_log or []
    self._environment.run_loop(
        game_masters=self.game_masters,
        entities=self.entities,
        premise=premise,
        max_steps=max_steps,
        verbose=True,
        log=raw_log,
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
