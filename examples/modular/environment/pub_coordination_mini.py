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


from examples.modular.environment import pub_coordination


class Simulation(pub_coordination.Simulation):
  """Simulation with pub closures."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initialize the simulation object.

    The launch script assumes this API object has a run() method.

    Args:
      **kwargs: arguments to pass to the base class.
    """

    super().__init__(
        pub_closed_probability=0.0,
        num_games=1,
        num_main_players=2,
        num_supporting_players=0,
        **kwargs,
    )
