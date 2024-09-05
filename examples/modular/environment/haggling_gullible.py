# Copyright 2024 DeepMind Technologies Limited.
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

"""A Concordia Environment Configuration."""

from collections.abc import Callable, Sequence
import types

from examples.modular.environment import haggling
from concordia.factory.agent import basic_agent
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import numpy as np


class Simulation(haggling.Simulation):
  """Simulation with pub closures."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      measurements: measurements_lib.Measurements,
      agent_module: types.ModuleType = basic_agent,
      resident_visitor_modules: Sequence[types.ModuleType] = (),
      supporting_agent_module: types.ModuleType | None = None,
      time_and_place_module: str | None = None,
  ):
    """Initialize the simulation object.

    The launch script assumes this API object has a run() method.

    Args:
      model: the language model to use.
      embedder: the sentence transformer to use.
      measurements: the measurements object to use.
      agent_module: the agent module to use for all main characters.
      resident_visitor_modules: optionally, use different modules for majority
        and minority parts of the focal population.
      supporting_agent_module: agent module to use for all supporting players. A
        supporting player is a non-player character with a persistent memory
        that plays a specific role in defining the task environment. Their role
        is not incidental but rather is critcal to making the task what it is.
        Supporting players are not necessarily interchangeable with focal or
        background players.
      time_and_place_module: optionally, specify a module containing settings
        that create a sense of setting in a specific time and place. If not
        specified, a random module will be chosen from the default options.
    """

    super().__init__(
        model=model,
        embedder=embedder,
        measurements=measurements,
        agent_module=agent_module,
        resident_visitor_modules=resident_visitor_modules,
        supporting_agent_module=supporting_agent_module,
        time_and_place_module=time_and_place_module,
        num_supporting_player=1,
        only_match_with_support=True,
    )
