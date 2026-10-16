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

"""The Steward's forecast: dead reckoning from late, noisy readings.

The Steward reads the temperature late and with noise, and knows how late
and how noisy (its instruments say so). It also knows what it did: each
climate move's effect turn by turn on this world is measured
(`push_audit` profiles). Like a navigator from the last fix and the course
since, the forecast takes each recent reading, adds what the Steward's
moves have done to the planet since the moment that reading shows, and
weighs the results by their noise and age. Projected forward, the same
profiles give the temperature on the turn a move made now acts, and the
course as the moves fade: when the planet leaves the band.

A trend through the readings themselves was tried and dropped: the planet's
own swings do not carry on from tick to tick, and a trend only amplified
the noise. The likely range grows with
how far the planet has wandered from the reckoning in the Steward's own
readings, so it is wide where the world is restless. Nothing here looks at
the truth; `track_record` compares the forecasts with it only after a game.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

from examples.deep_time.world import data
from examples.deep_time.world import planet as planet_lib

# How the forecast is made (data/forecast.yaml), calibrated on mock games of
# every scenario so the ranges hold the truth about four times in five.
_CONFIG = data.load('forecast.yaml')

Reading = tuple[int, float, float]  # The tick it shows, °C, noise σ.


@dataclasses.dataclass(frozen=True)
class Forecast:
  """Where the readings and the moves made say the temperature is heading.

  Attributes:
    tick: The tick it was made on.
    now: The temperature now (°C).
    now_spread: Half-width of its likely range (°C).
    ahead: The temperature on the turn a move made now acts.
    ahead_spread: Half-width of that range.
    acts_in: Ticks from now to `ahead`.
    unseen: What the moves made have done since the newest reading's
      moment: the change the readings do not show yet (°C).
    readings: How many distinct readings it rests on.
    course: The temperature over the next turns as the moves made fade or hold
      (no further moves), one a turn.
    leaves_in: Turns until the course leaves the band (None if it stays).
    edge: Which edge it leaves by: 'top', 'bottom' or ''.
    planned: Moves planned for later turns, (tick, key), counted in the course.
  """

  tick: int
  now: float
  now_spread: float
  ahead: float
  ahead_spread: float
  acts_in: int
  unseen: float
  readings: int
  course: tuple[float, ...] = ()
  leaves_in: int | None = None
  edge: str = ''
  planned: tuple[tuple[int, str], ...] = ()

  def to_dict(self) -> dict[str, Any]:
    out = dataclasses.asdict(self)
    out['course'] = list(self.course)
    out['planned'] = [list(p) for p in self.planned]
    return out


def readings_of(
    history: Sequence[Mapping[str, Any]], tick: int
) -> list[Reading]:
  """The distinct readings showing the last `window` ticks, up to `tick`.

  A reading held over ticks with nothing new (lost, or not watched) counts
  once.

  Args:
    history: The planet's history rows (with `believed_temp_c`, and the tick
      each reading shows, `read_tick`).
    tick: The tick the forecast is made on.

  Returns:
    (the tick it shows, °C, noise σ), oldest first.
  """
  window = int(_CONFIG['window'])
  found: dict[int, Reading] = {}
  for row in history:
    if int(row['tick']) > tick or 'believed_temp_c' not in row:
      continue
    moment = int(row.get('read_tick', row['tick']))
    found.setdefault(
        moment,
        (
            moment,
            float(row['believed_temp_c']),
            float(row.get('read_sigma', 0)),
        ),
    )
  return sorted(r for r in found.values() if r[0] > tick - window)


def _effect(profile: Sequence[float], since: int) -> float:
  """A move's change to the temperature `since` ticks after it was made.

  A move acts on the next tick; past its measured turns it holds the last.
  """
  if since <= 0 or not profile:
    return 0.0
  return float(profile[min(since, len(profile)) - 1])


def moved(
    made: Sequence[tuple[int, str]],
    profiles: Mapping[str, Sequence[float]],
    start: int,
    end: int,
) -> float:
  """What the moves made changed in the temperature from `start` to `end`."""
  return sum(
      _effect(profiles[key], end - tick) - _effect(profiles[key], start - tick)
      for tick, key in made
      if key in profiles
  )


def _wander(
    readings: Sequence[Reading],
    made: Sequence[tuple[int, str]],
    profiles: Mapping[str, Sequence[float]],
) -> float:
  """How far a tick the planet strays from the reckoning (°C), as read.

  From successive readings: the change the moves made do not explain, less
  the noise both carry; a high quantile, since the planet's surprises come
  in jolts.
  """
  floor = float(_CONFIG['wander_floor'])
  points = [
      (moment, v - moved(made, profiles, moment - 10**6, moment), sigma)
      for moment, v, sigma in readings
  ]
  rates = sorted(
      max(0.0, (y1 - y0) ** 2 - s0**2 - s1**2) / max(1, m1 - m0) ** 2
      for (m0, y0, s0), (m1, y1, s1) in zip(points, points[1:])
  )
  if not rates:
    return floor
  pick = rates[
      min(len(rates) - 1, int(_CONFIG['wander_quantile'] * len(rates)))
  ]
  return max(floor, math.sqrt(pick))


def forecast(
    history: Sequence[Mapping[str, Any]],
    made: Sequence[tuple[int, str]],
    profiles: Mapping[str, Sequence[float]],
    *,
    tick: int,
    acts_in: int = 1,
    band: tuple[float, float] | None = None,
    planned: Sequence[tuple[int, str]] = (),
) -> Forecast | None:
  """The Steward's dead reckoning of the temperature now and ahead.

  Args:
    history: The planet's history rows (readings: `believed_temp_c`,
      `read_tick`, `read_sigma`).
    made: The Steward's paid moves, (tick, key).
    profiles: Each climate move's change to the temperature turn by turn.
    tick: The tick it is now.
    acts_in: Ticks until a move made now acts.
    band: The habitable band, to say when the course leaves it.
    planned: Moves planned for later turns, (tick, key): the course counts them
      (the rest of the forecast cannot: they act later).

  Returns:
    The forecast, or None without readings.
  """
  window = readings_of(history, tick)
  if not window:
    return None
  recent = window[-int(_CONFIG['readings']) :]
  weigh = float(_CONFIG['weight_wander'])
  points = []
  for moment, value, sigma in recent:
    reckoned = value + moved(made, profiles, moment, tick)
    doubt = max(sigma, 0.2) ** 2 + (weigh * (tick - moment)) ** 2
    points.append((reckoned, 1.0 / doubt))
  total = sum(w for _, w in points)
  now = sum(v * w for v, w in points) / total
  newest, _, sigma = recent[-1]
  wander = _wander(window, made, profiles)
  z = float(_CONFIG['z'])

  def spread(gone: int) -> float:
    return z * math.sqrt(sigma**2 / len(recent) + (wander * gone) ** 2)

  horizon = int(_CONFIG['horizon'])
  ahead_made = [*made, *(p for p in planned if p[0] >= tick)]
  course = tuple(
      round(now + moved(ahead_made, profiles, tick, tick + j), 1)
      for j in range(1, horizon + 1)
  )
  leaves_in, edge = None, ''
  if band is not None and band[0] <= now <= band[1]:
    for j, value in enumerate(course, start=1):
      if value > band[1] or value < band[0]:
        leaves_in, edge = j, ('top' if value > band[1] else 'bottom')
        break
  return Forecast(
      tick=tick,
      now=round(now, 1),
      now_spread=round(spread(tick - newest), 1),
      ahead=round(now + moved(made, profiles, tick, tick + acts_in), 1),
      ahead_spread=round(spread(tick + acts_in - newest), 1),
      acts_in=acts_in,
      unseen=round(moved(made, profiles, newest, tick), 1),
      readings=len(recent),
      course=course,
      leaves_in=leaves_in,
      edge=edge,
      planned=tuple((int(t), str(k)) for t, k in planned if t >= tick),
  )


def what_if(
    found: Forecast | None,
    profiles: Mapping[str, Sequence[float]],
    key: str,
    band: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
  """The forecast's course if a move is made now.

  The same dead reckoning, from the same late readings, with the move's
  measured profile added from the turn it acts: what the Steward can know
  of a move before making it.

  Args:
    found: The forecast now.
    profiles: Each climate move's change to the temperature turn by turn.
    key: The move (an intervention key).
    band: The habitable band.

  Returns:
    {'then': °C the turn it acts, 'spread', 'course', and with a band
    'leaves_in'/'edge' (a course in the band now that leaves it) or
    'returns_in' (one out of it that comes back)}, or None for a move
    with no measured profile (or no forecast).
  """
  if found is None or key not in profiles or not found.course:
    return None
  profile = profiles[key]
  course = [
      round(v + _effect(profile, j), 1)
      for j, v in enumerate(found.course, start=1)
  ]
  out: dict[str, Any] = {
      'then': course[min(found.acts_in, len(course)) - 1],
      'spread': found.ahead_spread,
      'course': course,
      'leaves_in': None,
      'edge': '',
      'returns_in': None,
  }
  if band is not None:
    low, high = band
    inside = lambda v: low <= v <= high
    if inside(found.now):
      for j, v in enumerate(course, start=1):
        if not inside(v):
          out['leaves_in'], out['edge'] = j, 'top' if v > high else 'bottom'
          break
    else:
      out['returns_in'] = next(
          (j for j, v in enumerate(course, start=1) if inside(v)), None
      )
  return out


def words(found: Forecast | None) -> str:
  """The forecast as sentences for the reading (models read them too)."""
  if found is None:
    return ''
  text = ''
  if abs(found.unseen) >= 0.5:
    text += (
        ' Your moves since that reading have'
        f' {"warmed" if found.unseen > 0 else "cooled"} the planet about'
        f' {abs(found.unseen):.0f} C, which it does not show yet.'
    )
  turn = 'next turn' if found.acts_in == 1 else f'in {found.acts_in} turns'
  text += (
      f' Your forecast: about {found.now:.0f} ± {found.now_spread:.0f} C now,'
      f' {found.ahead:.0f} ± {found.ahead_spread:.0f} C {turn} if you wait.'
  )
  if found.leaves_in is not None:
    turns = 'turn' if found.leaves_in == 1 else 'turns'
    text += (
        ' As your moves fade, the planet leaves the band by the'
        f' {found.edge} in about {found.leaves_in} {turns}.'
    )
  return text


def track_record(
    history: Sequence[Mapping[str, Any]],
    made: Sequence[tuple[int, str]],
    profiles: Mapping[str, Sequence[float]],
    acts_in: int = 1,
) -> dict[str, Any] | None:
  """How far off the forecasts and the raw readings were (after a game).

  Args:
    history: The planet's history rows (with `temp_c` and readings).
    made: The Steward's paid moves.
    profiles: The moves' profiles.
    acts_in: Ticks until a move acts, for the forecasts ahead.

  Returns:
    Mean absolute errors (°C) of the raw readings and of the forecasts of
    now and ahead, and the share of truths inside the forecast ranges; None
    without readings.
  """
  truth = {int(r['tick']): float(r['temp_c']) for r in history}
  raw, now, ahead, inside = [], [], [], []
  for row in history:
    if 'believed_temp_c' not in row:
      continue
    tick = int(row['tick'])
    found = forecast(history, made, profiles, tick=tick, acts_in=acts_in)
    if found is None:
      continue
    raw.append(abs(float(row['believed_temp_c']) - truth[tick]))
    now.append(abs(found.now - truth[tick]))
    inside.append(abs(found.now - truth[tick]) <= found.now_spread)
    later = truth.get(tick + acts_in)
    if later is not None:
      ahead.append(abs(found.ahead - later))
      inside.append(abs(found.ahead - later) <= found.ahead_spread)
  if not raw:
    return None

  def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None

  return {
      'turns': len(raw),
      'raw_error_c': mean(raw),
      'now_error_c': mean(now),
      'ahead_error_c': mean(ahead),
      'inside_share': round(sum(inside) / len(inside), 2),
  }


def of_planet(
    planet: planet_lib.Planet,
    latest: tuple[int, float] | None = None,
    acts_in: int = 1,
    planned: Sequence[tuple[int, str]] = (),
) -> Forecast | None:
  """The Steward's forecast on a planet, now.

  Args:
    planet: The planet (its history holds the Steward's readings).
    latest: A reading arrived this tick and not yet in the history: (the tick it
      shows, °C).
    acts_in: Ticks until a move made now acts.
    planned: Moves planned for later turns, (tick, key).

  Returns:
    The forecast, or None before any reading.
  """
  rows: list[Mapping[str, Any]] = list(planet.history)
  if latest is not None:
    tier = planet.instrument()
    rows = [r for r in rows if int(r['tick']) < planet.tick]
    rows.append({
        'tick': planet.tick,
        'believed_temp_c': latest[1],
        'read_tick': latest[0],
        'read_sigma': float(tier['noise_sigma']) if tier else 0.0,
    })
  low, high = planet.band()
  return forecast(
      rows,
      planet.made,
      planet.move_profiles,
      tick=planet.tick,
      acts_in=acts_in,
      # A band too wide ever to bind (a thaw's) has no edge to leave by.
      band=(low, high) if high - low <= 60 else None,
      planned=planned,
  )


def turning_points(
    history: Sequence[Mapping[str, Any]],
    played: Sequence[Mapping[str, Any]],
    band: tuple[float, float] | None,
    acts_in: int,
    ways: Mapping[str, str],
) -> list[dict[str, Any]]:
  """The game's turning points, for the recap.

  The steepest swings of the true temperature over `turning_span` turns (at
  least `turning_swing_c`; a fall and the rise after it are two), each with
  the player's latest move pushing the same way that had acted by the
  swing's middle, and what they read, forecast and what was true when they
  made it; or, when no move of theirs pushed that way, the reading they had
  as it began. Swings a move of theirs came before are told first, then
  those that broke the band.

  Args:
    history: The planet's rows ('tick', 'temp_c', 'believed_temp_c').
    played: The player's turns ('tick', 'move', 'how', and the world then:
      'read_c', 'forecast_c', 'true_c').
    band: The habitable band, or None where none binds.
    acts_in: Turns from a move to when it acts.
    ways: Each move that pushes the climate: 'warm' or 'cool' (by its measured
      profile); the rest push neither way.

  Returns:
    At most `turning_most` points in the order they came: 'from', 'to'
    (ticks), 'from_c', 'to_c', 'way' (it 'left' the band, came 'back', went
    'across' it, or a 'swing'), and 'move' (or None), 'move_tick', 'read_c',
    'forecast_c', 'true_c'.
  """
  span = int(_CONFIG.get('turning_span', 2))
  swing = float(_CONFIG.get('turning_swing_c', 8.0))
  most = int(_CONFIG.get('turning_most', 3))
  temps = {int(r['tick']): float(r['temp_c']) for r in history}
  seen = {
      int(r['tick']): float(r['believed_temp_c'])
      for r in history
      if r.get('believed_temp_c') is not None
  }
  swings = sorted(
      (
          (temps[k + span] - temps[k], k)
          for k in sorted(temps)
          if k + span in temps and abs(temps[k + span] - temps[k]) >= swing
      ),
      key=lambda x: -abs(x[0]),
  )
  chosen: list[tuple[float, int]] = []
  for delta, k in swings:  # Overlapping swings the same way are one.
    if all(abs(k - j) > span or (d > 0) != (delta > 0) for d, j in chosen):
      chosen.append((delta, k))
  inside = lambda t: band is not None and band[0] <= t <= band[1]
  found = []
  for delta, k in chosen:
    start, end = temps[k], temps[k + span]
    way = 'swing'
    if band is not None:
      if inside(start) and not inside(end):
        way = 'left'
      elif not inside(start) and inside(end):
        way = 'back'
      elif (start > band[1] and end < band[0]) or (
          start < band[0] and end > band[1]
      ):
        way = 'across'
    point: dict[str, Any] = {
        'from': k,
        'to': k + span,
        'from_c': round(start, 1),
        'to_c': round(end, 1),
        'way': way,
        'move': None,
    }
    # The latest move of the player's pushing this way that had acted by
    # the swing's middle.
    push = 'warm' if delta > 0 else 'cool'
    before = [
        t
        for t in played
        if t['how'] != 'held'
        and ways.get(str(t['move'])) == push
        and k - span <= int(t['tick']) + acts_in <= k + span // 2 + 1
    ]
    if before:
      turn = before[-1]
      point.update({
          'move': str(turn['move']),
          'move_tick': int(turn['tick']),
          'read_c': turn.get('read_c'),
          'forecast_c': turn.get('forecast_c'),
          'true_c': turn.get('true_c'),
      })
    else:
      point['read_c'] = seen.get(k)
    found.append(point)
  # Told first: swings a move came before, those that broke the band, the
  # steepest.
  found.sort(
      key=lambda p: (
          p['move'] is None,
          p['way'] not in ('left', 'across'),
          -abs(p['to_c'] - p['from_c']),
      )
  )
  return sorted(found[:most], key=lambda p: p['from'])
