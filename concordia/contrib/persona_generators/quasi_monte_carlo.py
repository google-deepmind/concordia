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

"""Quasi-Monte Carlo (QMC) methods for generating points in high-dimensional space."""

import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import spatial
from scipy import special
from scipy.stats import qmc


def calculate_coverage_monte_carlo(
    data: np.ndarray,
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    num_mc_samples: int,
    radius_multiplier: float = 1.0,
    log_radius: bool = True,
) -> float:
  """Calculates coverage using k-radius balls and Monte Carlo estimation.

  This method defines a radius `k` for each data point such that if the `N`
  points were perfectly distributed, the volume of their `N` k-radius balls
  would equal the total volume of the search space. It then estimates the true
  covered volume (accounting for overlaps) by checking what fraction of a large
  number of random samples fall into at least one of these balls.

  Args:
    data: A numpy array of shape (num_samples, num_dimensions).
    dimension_ranges: A dictionary mapping dimension names to (min, max) tuples.
    dimensions: The list of dimension names.
    num_mc_samples: The number of random samples for Monte Carlo estimation.
    radius_multiplier: A factor to multiply the base radius `k` by, allowing for
      sphere overlaps to account for packing problems.
    log_radius: Whether to log the radius calculation.

  Returns:
    The coverage as a float between 0 and 1.
  """
  if data.size == 0:
    return 0.0

  num_samples, num_dimensions = data.shape

  # 1. Calculate the total volume of the search space.
  total_volume = 1.0
  for dim_name in dimensions:
    min_val, max_val = dimension_ranges[dim_name]
    total_volume *= max_val - min_val

  if total_volume <= 0:
    return 0.0

  # 2. Calculate the target volume per sample.
  target_volume_per_sample = total_volume / num_samples

  # 3. Calculate the volume of a unit d-hypersphere.
  # V_d(r) = (pi^(d/2) / Gamma(d/2 + 1)) * r^d
  volume_unit_hypersphere = np.pi ** (num_dimensions / 2) / special.gamma(
      num_dimensions / 2 + 1
  )

  # 4. Solve for the radius k.
  k_d = target_volume_per_sample / volume_unit_hypersphere
  k = k_d ** (1 / num_dimensions)
  k_effective = k * radius_multiplier

  if log_radius:
    logging.info(
        'Using Monte Carlo coverage with %d samples. Base radius k=%f,'
        ' multiplier=%f, effective_radius=%f',
        num_mc_samples,
        k,
        radius_multiplier,
        k_effective,
    )

  # 5. Generate random points for Monte Carlo estimation.
  mins = np.array([dimension_ranges[d][0] for d in dimensions])
  maxs = np.array([dimension_ranges[d][1] for d in dimensions])
  mc_points = np.random.uniform(
      low=mins, high=maxs, size=(num_mc_samples, num_dimensions)
  )

  # 6. Efficiently check for coverage using KDTree.
  kdtree = spatial.KDTree(data)
  # Query returns distances and indices of nearest neighbors for each MC point.
  distances, _ = kdtree.query(mc_points, k=1)
  # Count how many MC points have a nearest neighbor within k_effective.
  covered_count = np.sum(distances < k_effective)

  return covered_count / num_mc_samples


def calculate_coverage_grid(
    data: np.ndarray,
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    target_total_bins: int | None = None,
) -> float:
  """Calculates the coverage of the data points in the discretized space.

  The number of bins per dimension is dynamically determined to aim for
  a total number of bins close to target_total_bins.

  Args:
    data: A numpy array of shape (num_samples, num_dimensions) containing the
      data points.
    dimension_ranges: A dictionary mapping dimension names to (min, max) tuples.
    dimensions: The list of dimension names, in the order they appear in the
      data array columns.
    target_total_bins: The desired total number of bins in the high-dimensional
      space. If None, defaults to the number of samples.

  Returns:
    The coverage as a float between 0 and 1.
  """
  if data.size == 0:
    return 0.0

  num_samples, num_dimensions = data.shape
  if num_dimensions != len(dimensions):
    raise ValueError(
        f'Data has {num_dimensions} dimensions, but'
        f' {len(dimensions)} dimensions were provided.'
    )

  if num_dimensions == 0:
    return 0.0

  if target_total_bins is None:
    target_total_bins = num_samples

  # Determine the number of bins per dimension to approximate target_total_bins
  if target_total_bins <= 0:
    num_bins_per_dim = 1
  else:
    num_bins_per_dim = max(
        2, int(np.ceil(target_total_bins ** (1 / num_dimensions)))
    )

  logging.info(
      'Using %d bins per dimension for %d dimensions, targeting ~%d total'
      ' bins.',
      num_bins_per_dim,
      num_dimensions,
      target_total_bins,
  )

  bins = []
  for dim_name in dimensions:
    min_val, max_val = dimension_ranges[dim_name]
    # Create num_bins_per_dim + 1 edges for num_bins_per_dim bins
    bins.append(np.linspace(min_val, max_val, num_bins_per_dim + 1))

  binned_data = np.zeros_like(data, dtype=int)
  for i in range(num_dimensions):
    # np.digitize returns indices from 1 to len(bins)
    binned_data[:, i] = np.digitize(data[:, i], bins[i]) - 1
    # Clip values to be within [0, num_bins_per_dim - 1]
    binned_data[:, i] = np.clip(binned_data[:, i], 0, num_bins_per_dim - 1)

  # Treat each row as a unique bin index in the multi-dimensional grid
  unique_bins = set(tuple(row) for row in binned_data)

  total_possible_bins = num_bins_per_dim**num_dimensions
  coverage = len(unique_bins) / total_possible_bins
  return coverage


# Calibration cache for radius multiplier: (n_points, n_dims) -> multiplier
_RADIUS_CALIBRATION_CACHE = {}
CALIBRATION_TARGET_COVERAGE = 0.99
CALIBRATION_MC_SAMPLES = 10_000
_RADIUS_MULTIPLIER_TOLERANCE = 0.01
_RADIUS_SEARCH_UPPER_BOUND = 10.0


def calibrate_radius_multiplier(
    num_samples: int,
    num_dimensions: int,
    num_mc_samples_for_calibration: int,
) -> float:
  """Calibrates the radius multiplier for a given sample size and dimension.

  Searches for the smallest radius multiplier such that a Sobol sequence of
  `num_samples` points in `num_dimensions` achieves
  `_CALIBRATION_TARGET_COVERAGE`.

  Args:
    num_samples: The number of data points.
    num_dimensions: The number of dimensions.
    num_mc_samples_for_calibration: The number of Monte Carlo samples to use
      during calibration.

  Returns:
    The calibrated radius multiplier.
  """
  if (num_samples, num_dimensions) in _RADIUS_CALIBRATION_CACHE:
    return _RADIUS_CALIBRATION_CACHE[(num_samples, num_dimensions)]

  logging.info(
      'Calibrating radius multiplier for n=%d, d=%d...',
      num_samples,
      num_dimensions,
  )

  # Generate a low-discrepancy (Sobol) point set for calibration
  calib_dims = [f'dim_{i}' for i in range(num_dimensions)]
  calib_ranges = {d: (0.0, 1.0) for d in calib_dims}
  qmc_points_dicts = generate_sobol_points(
      calib_ranges, calib_dims, num_samples
  )
  qmc_points = np.array([[p[d] for d in calib_dims] for p in qmc_points_dicts])

  def get_coverage(multiplier: float) -> float:
    return calculate_coverage_monte_carlo(
        qmc_points,
        calib_ranges,
        calib_dims,
        num_mc_samples_for_calibration,
        radius_multiplier=multiplier,
        log_radius=False,  # Avoid verbose logging during calibration
    )

  # If multiplier 1.0 is sufficient, use it.
  if get_coverage(1.0) >= CALIBRATION_TARGET_COVERAGE:
    _RADIUS_CALIBRATION_CACHE[(num_samples, num_dimensions)] = 1.0
    logging.info('Calibration complete: multiplier=1.0')
    return 1.0

  # If even upper bound isn't enough, log warning and use it.
  if get_coverage(_RADIUS_SEARCH_UPPER_BOUND) < CALIBRATION_TARGET_COVERAGE:
    logging.warning(
        'Failed to reach target coverage %f even with multiplier %f for n=%d,'
        ' d=%d',
        CALIBRATION_TARGET_COVERAGE,
        _RADIUS_SEARCH_UPPER_BOUND,
        num_samples,
        num_dimensions,
    )
    _RADIUS_CALIBRATION_CACHE[(num_samples, num_dimensions)] = (
        _RADIUS_SEARCH_UPPER_BOUND
    )
    return _RADIUS_SEARCH_UPPER_BOUND

  # Binary search for smallest multiplier in [1.0, _RADIUS_SEARCH_UPPER_BOUND]
  low = 1.0
  high = _RADIUS_SEARCH_UPPER_BOUND
  while high - low > _RADIUS_MULTIPLIER_TOLERANCE:
    mid = (low + high) / 2
    if get_coverage(mid) < CALIBRATION_TARGET_COVERAGE:
      low = mid
    else:
      high = mid

  multiplier = high  # 'high' is the lowest value tested >= target
  _RADIUS_CALIBRATION_CACHE[(num_samples, num_dimensions)] = multiplier
  logging.info(
      'Calibration complete: found multiplier=%f for n=%d, d=%d',
      multiplier,
      num_samples,
      num_dimensions,
  )
  return multiplier


def generate_sobol_points(
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    num_points: int,
) -> List[Dict[str, float]]:
  """Generates space-filling sample points using a Sobol sequence.

  This offers an alternative to generate_grid_cell_centers for generating
  an arbitrary number of points, rather than one determined by a grid
  structure (i.e., n^d points). It uses a low-discrepancy sequence (Sobol)
  to generate `num_points` that cover the hyper-rectangle defined by
  dimension_ranges as evenly as possible.

  Args:
    dimension_ranges: A dictionary mapping dimension names to (min, max) tuples.
    dimensions: The list of dimension names.
    num_points: The exact number of points to generate.

  Returns:
    A list of dictionaries, where each dictionary represents a point,
    mapping dimension names to values.
  """
  num_dimensions = len(dimensions)
  if num_dimensions == 0 or num_points == 0:
    return []

  sampler = qmc.Sobol(d=num_dimensions, scramble=True)
  sample = sampler.random(n=num_points)

  mins = np.array([dimension_ranges[d][0] for d in dimensions])
  maxs = np.array([dimension_ranges[d][1] for d in dimensions])

  scaled_sample = qmc.scale(sample, mins, maxs)
  return [dict(zip(dimensions, point)) for point in scaled_sample]


def generate_halton_points(
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    num_points: int,
) -> List[Dict[str, float]]:
  """Generates space-filling sample points using a Halton sequence.

  Args:
    dimension_ranges: A dictionary mapping dimension names to (min, max) tuples.
    dimensions: The list of dimension names.
    num_points: The exact number of points to generate.

  Returns:
    A list of dictionaries, where each dictionary represents a point,
    mapping dimension names to values.
  """
  num_dimensions = len(dimensions)
  if num_dimensions == 0 or num_points == 0:
    return []

  sampler = qmc.Halton(d=num_dimensions, scramble=True)
  sample = sampler.random(n=num_points)

  mins = np.array([dimension_ranges[d][0] for d in dimensions])
  maxs = np.array([dimension_ranges[d][1] for d in dimensions])

  scaled_sample = qmc.scale(sample, mins, maxs)
  return [dict(zip(dimensions, point)) for point in scaled_sample]


def generate_grid_cell_centers(
    dimension_ranges: Dict[str, Tuple[float, float]],
    dimensions: List[str],
    target_total_bins: int,
) -> np.ndarray:
  """Generates sample points at the center of each grid cell.

  The grid discretization is determined by target_total_bins, consistent
  with calculate_coverage_grid.

  Args:
    dimension_ranges: A dictionary mapping dimension names to (min, max) tuples.
    dimensions: The list of dimension names.
    target_total_bins: The desired total number of bins in the high-dimensional
      space, used to determine bins per dimension.

  Returns:
    A numpy array of shape (num_cells, num_dimensions) containing points
    at the center of each grid cell.
  """
  num_dimensions = len(dimensions)
  if num_dimensions == 0:
    return np.empty((0, 0))

  if target_total_bins <= 0:
    num_bins_per_dim = 1
  else:
    num_bins_per_dim = max(
        2, int(np.ceil(target_total_bins ** (1 / num_dimensions)))
    )

  logging.info(
      'Generating grid cell centers with %d bins per dimension.',
      num_bins_per_dim,
  )

  all_dim_centers = []
  for dim_name in dimensions:
    min_val, max_val = dimension_ranges[dim_name]
    edges = np.linspace(min_val, max_val, num_bins_per_dim + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    all_dim_centers.append(centers)

  cell_centers = list(itertools.product(*all_dim_centers))
  return np.array(cell_centers)
