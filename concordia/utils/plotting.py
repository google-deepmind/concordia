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
from concordia.utils.deprecated import measurements
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_kde_from_dataframe(
    data_df: pd.DataFrame,
    dimensions: list[str] | None = None,
    label_column: str | None = None,
    palette: str | list[str] = '',
):
  """Plots KDE distributions for each dimension, colored by label_column.

  Args:
    data_df: DataFrame containing the data.
    dimensions: List of column names to plot distributions for. If None, plots
      all columns except label_column.
    label_column: Column name to group and color the distributions. If None, all
      data is plotted in a single color.
    palette: Color palette to use for the different labels.
  """
  if dimensions is None:
    dimensions = data_df.columns.tolist()
    if label_column and label_column in dimensions:
      dimensions.remove(label_column)

  num_dims = len(dimensions)
  if num_dims == 0:
    print('No dimensions to plot.')
    return

  plt.figure(figsize=(15, 5 * ((num_dims - 1) // 3 + 1)))

  if label_column and label_column in data_df.columns:
    unique_labels = data_df[label_column].unique()
    colors = sns.color_palette(palette, len(unique_labels))
    label_color_map = dict(zip(unique_labels, colors))

    for i, dim in enumerate(dimensions):
      if dim not in data_df.columns:
        print(f"Warning: Dimension '{dim}' not found in DataFrame.")
        continue
      plt.subplot((num_dims - 1) // 3 + 1, 3, i + 1)
      for label in unique_labels:
        subset = data_df[data_df[label_column] == label]
        if not subset.empty:
          sns.kdeplot(
              data=subset,
              x=dim,
              color=label_color_map[label],
              alpha=0.6,
              label=label,
              fill=True,
          )
      plt.title(dim)
      plt.legend()
  else:
    for i, dim in enumerate(dimensions):
      if dim not in data_df.columns:
        print(f"Warning: Dimension '{dim}' not found in DataFrame.")
        continue
      plt.subplot((num_dims - 1) // 3 + 1, 3, i + 1)
      sns.kdeplot(
          data=data_df,
          x=dim,
          color='blue',
          alpha=0.6,
          fill=True,
      )
      plt.title(dim)

  plt.tight_layout()
  plt.show()


def plot_correlation_matrix(
    data_df: pd.DataFrame,
    dimensions: list[str] | None = None,
    label_column: str | None = None,
    title: str = 'Correlation Matrix',
    cmap: str = 'coolwarm',
):
  """Plots the correlation matrix of the specified dimensions in the DataFrame.

  If label_column is provided, a separate correlation matrix is plotted for each
  unique label in that column.

  Args:
    data_df: DataFrame containing the data.
    dimensions: List of column names to include in the correlation matrix. If
      None, all numerical columns are used.
    label_column: Column name to group the data by. If None, a single matrix is
      generated for the entire DataFrame.
    title: Base title for the plot(s).
    cmap: Colormap for the heatmap.
  """
  if dimensions:
    if label_column and label_column not in dimensions:
      dimensions.append(label_column)
    data_to_use = data_df[dimensions]
  else:
    data_to_use = data_df.select_dtypes(include=np.number)
    if label_column and label_column not in data_to_use.columns:
      if label_column in data_df.columns:
        data_to_use = pd.concat([data_to_use, data_df[label_column]], axis=1)

  if label_column and label_column in data_to_use.columns:
    unique_labels = data_to_use[label_column].unique()
    for label in unique_labels:
      subset_df = data_to_use[data_to_use[label_column] == label]
      plot_dims = [
          col
          for col in subset_df.columns
          if col != label_column
          and pd.api.types.is_numeric_dtype(subset_df[col])
      ]

      if not plot_dims:
        print(f"No numerical dimensions to plot for label '{label}'.")
        continue

      correlation_matrix = subset_df[plot_dims].corr()

      plt.figure(figsize=(10, 8))
      sns.heatmap(
          correlation_matrix,
          annot=True,
          fmt='.2f',
          cmap=cmap,
          xticklabels=correlation_matrix.columns,
          yticklabels=correlation_matrix.columns,
          vmin=-1,
          vmax=1,
      )
      plt.title(f'{title} - {label}')
      plt.show()
  else:
    plot_dims = [
        col
        for col in data_to_use.columns
        if pd.api.types.is_numeric_dtype(data_to_use[col])
    ]
    if not plot_dims:
      print('No numerical dimensions to plot for correlation matrix.')
      return

    correlation_matrix = data_to_use[plot_dims].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.columns,
        vmin=-1,
        vmax=1,
    )
    plt.title(title)
    plt.show()
