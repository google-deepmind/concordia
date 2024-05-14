# Copyright 2023 DeepMind Technologies Limited.
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


"""Functions for plotting metrics."""

from typing import Collection
from concordia.utils import measurements
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def plot_line_measurement_channel(measurements_obj: measurements.Measurements,
                                  channel_name: str,
                                  group_by: str = 'player',
                                  xaxis: str = 'time',
                                  yaxis: str = 'value_float',
                                  ax: plt.Axes = None) -> None:
  """Plots a pie chart of a measurement channel."""
  if channel_name not in measurements_obj.available_channels():
    raise ValueError(f'Unknown channel: {channel_name}')

  channel = measurements_obj.get_channel(channel_name)
  data = []
  channel.subscribe(on_next=data.append)

  plot_df_line(pd.DataFrame(data), channel_name, group_by=group_by, xaxis=xaxis,
               yaxis=yaxis, ax=ax)


def plot_pie_measurement_channel(measurements_obj: measurements.Measurements,
                                 channel_name: str,
                                 group_by: str = 'player',
                                 value: str = 'value_str') -> None:
  """Plots a pie chart of a measurement channel."""
  if channel_name not in measurements_obj.available_channels():
    raise ValueError(f'Unknown channel: {channel_name}')

  channel = measurements_obj.get_channel(channel_name)
  data = []
  channel.subscribe(on_next=data.append)
  scale = set()
  for datum in data:
    scale |= {datum['value_str']}

  plot_df_pie(pd.DataFrame(data), scale, channel_name, group_by=group_by,
              value=value)


def plot_df_pie(df: pd.DataFrame,
                scale: Collection[str],
                title: str = 'Metric',
                group_by: str = 'player',
                value: str = 'value_str') -> None:
  """Plots a pie chart of a dataframe.

  Args:
    df: The dataframe containing the data to plot.
    scale: The set of possible values to plot.
    title: The title of the plot.
    group_by: Group data by this field, plot each one in its own figure.
    value: The name of the value to aggregate for the pie chart regions.
  """
  cmap = mpl.colormaps['Paired']  # pylint: disable=unsubscriptable-object
  colours = cmap(range(len(scale)))
  scale_to_colour = dict(zip(scale, colours))

  for player, group_df in df.groupby(group_by):
    plt.figure()
    counts = group_df[value].value_counts()
    plt.pie(
        counts,
        labels=counts.index,
        colors=[scale_to_colour[color] for color in counts.index],
    )
    plt.title(f'{title} of {player}')


def plot_df_line(df: pd.DataFrame,
                 title: str = 'Metric',
                 group_by: str = 'player',
                 xaxis: str = 'time',
                 yaxis: str = 'value_float',
                 ax: plt.Axes = None) -> None:
  """Plots a line chart of a dataframe.

  Args:
    df: The dataframe with data to plot.
    title: The title of the plot.
    group_by: Group data by this field, plot each one as a line in the figure.
    xaxis: The name of the column to use as the x-axis. If multiple entries have
      the same value in this field, the y-axis values are averaged.
    yaxis: The name of the column to use as the y-axis. The values in this
      column must be numerical.
    ax: The axis to plot on. If None, uses the current axis.
  """
  if ax is None:
    ax = plt.gca()

  for player, group_df in df.groupby(group_by):
    group_df = group_df.groupby(xaxis).mean(numeric_only=True).reset_index()
    group_df.plot(x=xaxis, y=yaxis, label=player, ax=ax)
  plt.title(title)


def plot_metric_pie(metric):
  """Plots a pie chart of the metric."""
  plot_df_pie(pd.DataFrame(metric.state()), metric.get_scale(), metric.name())


def plot_metric_line(metric):
  """Plots a line chart of the metric."""
  plot_df_line(pd.DataFrame(metric.state()), metric.name())
