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

"""A deterministic, rule-based offline language model for Deep Time.

Like `examples/concordia_island/mock_language_model.py`, it lets every scenario
run without network access. Unlike `testing.mock_model.MockModel`, it plays
sensibly: choice prompts are parsed (the options, and the instrument readings
in the context) and answered by small game policies, so mock runs exercise
real game dynamics. Answers depend only on the prompt, so runs are
reproducible.
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Callable, Collection, Mapping, Sequence
import hashlib
import json
import re
import threading
from typing import Any, override

from concordia.language_model import language_model

_OPTION_RE = re.compile(r'^\s*\(([a-z]+)\)\s(.+)$')
_TEMP_RE = re.compile(
    r'Global mean temperature (-?\d+(?:\.\d+)?) C \(habitable band'
    r' (-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?) C\)'
)
_OMEGA_RE = re.compile(r'Omega (-?\d+(?:\.\d+)?);')
_CO2_RE = re.compile(r'CO2 x(\d+(?:\.\d+)?) of the start')
_TIP_RE = re.compile(
    r'may tip the ice from x(\d+\.\d+) of the start, and almost surely does'
    r' above x(\d+\.\d+)'
)
_NOW_RE = re.compile(r'Tick (\d+) of (\d+)')
_FORECAST_RE = re.compile(
    r'Your forecast: about (-?\d+) ± (\d+) C now, (-?\d+) ± (\d+) C'
)
_ASH_RE = re.compile(r'tick (\d+): [^\n]*veils the sky with ash')
_BRIEFING_RE = re.compile(
    r'Recommend: ([a-z ]+?)\.?\s*\[tick \d+, (fresh|(\d+) ticks? old)\]'
)
_GENOME_RE = re.compile(
    r'Genome of the (\w+) (?:daisies|mats): optimum (-?\d+(?:\.\d+)?) C'
)


def _omega(prompt: str) -> float:
  """The Steward's latest Omega reading in the prompt (0 if none)."""
  omegas = _OMEGA_RE.findall(prompt)
  return float(omegas[-1]) if omegas else 0.0


def _census(prompt: str, species: str) -> tuple[float, float] | None:
  """Returns the last (local temperature, growth) reading for a species."""
  found = re.findall(
      rf'{species} (?:daisies|mats) cover \d+% \(local (-?\d+) C, growth'
      r' (\d+(?:\.\d+)?)\)',
      prompt,
  )
  if not found:
    return None
  return float(found[-1][0]), float(found[-1][1])


def _genome(prompt: str) -> tuple[str, float] | None:
  found = _GENOME_RE.findall(prompt)
  if not found:
    return None
  return found[-1][0], float(found[-1][1])


# A policy receives the prompt and the {letter: option text} of the last
# question, and returns the chosen option text or None to pass.
ChoicePolicy = Callable[[str, Mapping[str, str]], str | None]


def stable_index(text: str, n: int) -> int:
  """A deterministic pseudo-random index in [0, n) derived from text."""
  digest = hashlib.sha256(text.encode('utf-8')).digest()
  return int.from_bytes(digest[:8], 'big') % max(1, n)


def last_question_options(prompt: str) -> dict[str, str]:
  """Parses `(a) option` lines that follow the last `Question:` in a prompt."""
  start = prompt.rfind('Question:')
  options = {}
  for line in prompt[start if start >= 0 else 0 :].splitlines():
    match = _OPTION_RE.match(line)
    if match:
      options[match.group(1)] = match.group(2).strip()
  return options


def _find(options: Mapping[str, str], *needles: str) -> str | None:
  for needle in needles:
    for text in options.values():
      if needle in text.lower():
        return text
  return None


def steward_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Cools when near the top of the band, warms near the bottom."""
  return _steward_choice(prompt, options, use_advice=True)


def steward_policy_without_advice(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Like `steward_policy`, but ignores the Advisor's briefings."""
  return _steward_choice(prompt, options, use_advice=False)


def forecasting_steward_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Like `steward_policy`, but acts on its forecast, not the late reading.

  It takes the forecast for the turn its move acts: the
  late reading reckoned forward with the moves made since. The skilled
  bot's thresholds were tuned on raw readings and already allow for their
  lag, so on 16 seeds of every scenario the two play about as well.
  """
  return _steward_choice(prompt, options, use_advice=True, reading='ahead')


def passive_steward_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Always waits (a baseline for balance sweeps)."""
  del prompt
  return _find(options, 'wait and save omega')


def _thaw_reading(prompt: str) -> dict[str, Any]:
  """What a Steward of a frozen world reads: carbon, tipping point, clock."""
  co2 = _CO2_RE.findall(prompt)
  tip = _TIP_RE.findall(prompt)
  now = _NOW_RE.findall(prompt)
  tick, last = (int(now[-1][0]), int(now[-1][1])) if now else (0, 0)
  ash = [int(t) for t in _ASH_RE.findall(prompt)]
  soon = [
      int(k)
      for k in re.findall(r'eruption at [A-Za-z ]+ in (\d+) tick', prompt)
  ]
  return {
      'co2': float(co2[-1]) if co2 else None,
      'may': float(tip[-1][0]) if tip else None,
      'sure': float(tip[-1][1]) if tip else None,
      'left': last - tick if now else None,  # Ticks to the deadline.
      # Ash in the sky now, or due within two ticks, holds the ice.
      'ashen': (
          bool(ash and tick - max(ash) <= 1) or bool(soon and min(soon) <= 2)
      ),
  }


def _thaw_choice(prompt: str, options: Mapping[str, str]) -> str | None:
  """Breaks the ice: reads the air, builds carbon, then tips it with a haze.

  The census shows nothing on a frozen world, so the Steward watches the air
  gauge instead. It rouses the volcanoes until the air surely holds enough
  carbon for a dark haze to tip the ice (or may, when time runs short), and
  hazes under a clear sky.

  Args:
    prompt: The Steward's prompt.
    options: The offered options by letter.

  Returns:
    The chosen option's text, or None.
  """
  temps = _TEMP_RE.findall(prompt)
  if temps and float(temps[-1][0]) >= 0:
    return _find(options, 'wait')
  swap = _find(options, 'watch the air gauge instead of the census')
  if swap:
    return swap
  omega = _omega(prompt)
  read = _thaw_reading(prompt)
  if read['co2'] is not None and read['sure'] is not None:
    late = read['left'] is not None and read['left'] <= 6
    ready = read['co2'] >= read['sure'] or (late and read['co2'] >= read['may'])
    if ready:  # Save for the haze; hold it while ash veils the sky.
      haze = not read['ashen'] and omega >= 4
      return (haze and _find(options, 'dark haze')) or _find(options, 'wait')
  if omega >= 5:  # (Unless the volcanoes are still resting.)
    return _find(options, 'rouse the deep volcanoes') or _find(options, 'wait')
  return _find(options, 'wait')


def _oxygen_choice(prompt: str, options: Mapping[str, str]) -> str | None:
  """Seeds aerobes once oxygen appears; seeds oxygen makers if it never does.

  Once the iron is used up, oxygen can still be held back by strong
  methanogens (methane burns it), so the Steward tips the balance with
  cyanobacteria. Otherwise it defers to the thermostat.

  Args:
    prompt: The Steward's prompt.
    options: The offered options by letter.

  Returns:
    The chosen option's text, or None.
  """
  oxygen = re.findall(r"oxygen (\d+\.\d+) of today's level", prompt)
  iron = re.findall(r'iron left to soak up oxygen (\d+\.\d+)', prompt)
  aerobes = re.findall(r'aerobes (?:daisies|mats) cover (\d+)%', prompt)
  omega = _omega(prompt)
  if not oxygen:
    return None
  if float(oxygen[-1]) > 0.01 and omega >= 2:
    if not aerobes or int(aerobes[-1]) < 8:
      return _find(options, 'oxygen-breathing')
  if float(oxygen[-1]) < 0.005 and iron and float(iron[-1]) < 0.05:
    if omega >= 3:
      return _find(options, 'oxygen-making')
  return None


def _steward_choice(
    prompt: str,
    options: Mapping[str, str],
    use_advice: bool,
    reading: str = 'raw',
) -> str | None:
  if _find(options, 'wait and save omega') is None:
    return None
  if 'Thaw the planet' in prompt:
    return _thaw_choice(prompt, options)
  if 'aerobes should cover' in prompt:
    choice = _oxygen_choice(prompt, options)
    if choice:
      return choice
  temps = _TEMP_RE.findall(prompt)
  if not temps:
    return _find(options, 'wait')
  temp, low, high = (float(x) for x in temps[-1])
  # The forecast reckons the late reading forward with the moves made since
  # (now), or to the turn a move made now acts (ahead).
  forecasts = _FORECAST_RE.findall(prompt)
  if forecasts and reading in ('now', 'ahead'):
    temp = float(forecasts[-1][0 if reading == 'now' else 2])
  omega = _omega(prompt)
  margin = 0.15 * (high - low)
  advice = _fresh_advice(prompt) if use_advice else None
  early = 0.3 * (high - low)
  # Act early on advice only while the readings still head that way: advice
  # is late, and a planet already turning back needs no push.
  earlier = re.findall(r'Earlier readings you acted on: ([^C]+) C', prompt)
  before = [float(v) for v in earlier[-1].split(',')] if earlier else []
  trend = temp - before[-1] if before else 0.0
  if advice and temp > high - early and 'bright clouds' in advice:
    if trend > -margin:
      margin = early
  if advice and temp < low + early and 'dark haze' in advice:
    if trend < margin:
      margin = early
  # A hungry people under a sky near the band's edge: steer the climate back
  # (in mid-band the hunger is not the weather's doing).
  fed = [int(v) for v in re.findall(r'people, fed (\d+)%', prompt)]
  edge = 0.3 * (high - low)
  if fed and min(fed) < 90 and omega >= 4:
    if temp < low + edge or temp > high - edge:
      choice = _find(
          options, 'dark haze' if temp < low + edge else 'bright clouds'
      )
      if choice:
        return choice
  co2 = _CO2_RE.findall(prompt)
  # A goal that caps the air's carbon: watch the air gauge, and trim the
  # carbon with basalt dust before it reaches the cap, while the planet is
  # warm enough to take the cooling.
  cap = re.findall(r'air must hold less than (\d+(?:\.\d+)?) times', prompt)
  if cap:
    swap = _find(options, 'watch the air gauge instead of the census')
    if swap and not co2:
      return swap
    if (
        co2
        and float(co2[-1]) >= float(cap[-1]) - 0.1
        and omega >= 3
        and temp > low + margin + 3.0
    ):
      choice = _find(options, 'basalt dust')
      if choice:
        return choice
  # A goal that needs a living world (a biodiversity floor): keep the
  # peoples' axes off the woods while biodiversity nears the floor.
  floor = re.findall(r'biodiversity is at least (\d+(?:\.\d+)?)', prompt)
  life = re.findall(r'Omega -?\d+(?:\.\d+)?; biodiversity (\d+\.\d+)', prompt)
  if floor and life and float(life[-1]) < float(floor[-1]) + 0.03:
    # Grass and forest thrive in the low teens: a warm world is shaded with
    # clouds (not dust: it takes the carbon plants grow on), a cool one has
    # its woods kept from the axes.
    if temp > 17.0 and omega >= 4:
      choice = _find(options, 'bright clouds')
      if choice:
        return choice
    if omega >= 4:
      choice = _find(options, 'protect the wild woods')
      if choice:
        return choice
  herds = re.findall(r'Grazers (\d+\.\d+) \(prefer white petals (\d+)%', prompt)
  if (
      herds
      and float(herds[-1][0]) > 0.08
      and int(herds[-1][1]) >= 50
      and temp > high - 2 * margin
      and omega >= 3
  ):
    choice = _find(options, 'thin the grazing herds')
    if choice:
      return choice
  if temp > high - margin:
    if co2 and float(co2[-1]) > 1.05 and omega >= 3:
      choice = _find(options, 'basalt dust')
      if choice:
        return choice
    if omega >= 4:
      choice = _find(options, 'bright clouds')
      if choice:
        return choice
    if omega >= 2:
      choice = _find(options, 'seed white')
      if choice:
        return choice
  if temp < low + margin:
    if omega >= 4:
      choice = _find(options, 'dark haze')
      if choice:
        return choice
    if omega >= 2:
      choice = _find(options, 'seed black')
      if choice:
        return choice
  # Fill the arks' seed vault while the world is living (a ship carries no
  # more of the wild than the vault holds): when the climate asks nothing.
  vault = re.findall(
      r'Seed vault for the arks: (\d+\.\d+) of (\d+\.\d+)', prompt
  )
  if vault and float(vault[-1][0]) < float(vault[-1][1]) and omega >= 3:
    choice = _find(options, 'gather the seeds of the wild')
    if choice:
      return choice
  # Calm restless lands before they strike.
  restless = re.findall(
      r'(\d+)%',
      ''.join(
          re.findall(
              r'Restless lands \(pressure, share of the breaking point\):'
              r' ([^.]*)\.',
              prompt,
          )[-1:]
      ),
  )
  if restless and max(int(x) for x in restless) >= 75 and omega >= 4:
    choice = _find(options, 'cleansing rains')
    if choice:
      return choice
  # Restore the marshes when the seas are dying (they weigh on biodiversity).
  seas = re.findall(r'the seas at (\d+)% of their life', prompt)
  if seas and int(seas[-1]) < 50 and omega >= 6:
    choice = _find(options, 'coastal marshes')
    if choice:
      return choice
  # Replant thin forests: the peoples' charcoal depends on them.
  forests = re.findall(r'forests biome cover (\d+)%', prompt)
  if forests and int(forests[-1]) < 10 and omega >= 3:
    choice = _find(options, 'plant forests')
    if choice:
      return choice
  # Keep the herds: slow the warming, thin pressing packs, sow pasture.
  if "the bison herds' total size must be" in prompt:
    if temp > low + 0.45 * (high - low) and omega >= 4:
      choice = _find(options, 'bright clouds')
      if choice:
        return choice
    sizes = r' \d+ on the map, total size (\d+(?:\.\d+)?)'
    packs = re.findall('Wolf packs:' + sizes, prompt)
    bison = re.findall('Bison herds:' + sizes, prompt)
    if (
        packs
        and bison
        and float(packs[-1]) > 0.4 * float(bison[-1])
        and omega >= 3
    ):
      choice = _find(options, 'thin the wolf packs')
      if choice:
        return choice
    if omega >= 6:  # Pasture with what the clouds can spare.
      choice = _find(options, 'sow grass')
      if choice:
        return choice
  # Keep the woods: replant them when thin, otherwise call the rain.
  if 'forests must still cover' in prompt:
    woods = re.findall(r'forests biome cover (\d+)%', prompt)
    if woods and int(woods[-1]) < 12 and omega >= 2:
      choice = _find(options, 'plant forests')
      if choice:
        return choice
    if omega >= 3:
      choice = _find(options, 'summon rain')
      if choice:
        return choice
  # Green the islands: sow wherever there is bare land, and between sowings
  # carry seed from the meadows to the bare islands (it costs no omega).
  if 'islands must be green' in prompt:
    choice = (
        (omega >= 2 and _find(options, 'sow grass'))
        or (omega >= 1 and _find(options, 'scatter a little grass'))
        or _find(options, 'carry grass seed')
    )
    if choice:
      return choice
  # Invest in sharper instruments while they are poor. Grazer booms and busts
  # are fast, so a world with herds needs the best instruments.
  noise = re.findall(r'temperatures ±(\d+(?:\.\d+)?) C', prompt)
  wanted = 0.5 if herds else 1.5
  if noise and float(noise[-1]) > wanted and omega >= 5:
    choice = _find(options, 'better instruments')
    if choice:
      return choice
  return _find(options, 'wait')


def _fresh_advice(prompt: str, max_age: int = 2) -> str | None:
  """The advisor's latest recommendation if at most `max_age` ticks old."""
  found = _BRIEFING_RE.findall(prompt)
  if not found:
    return None
  recommendation, freshness, age = found[-1]
  if freshness != 'fresh' and int(age) > max_age:
    return None
  return recommendation


def _frozen(prompt: str) -> bool:
  """Whether a prompt is about breaking a frozen world's ice.

  The Steward reads its goal; the Advisor reads only the instruments (the
  air gauge's tipping point, or ice over most of the planet) and the moves
  it may recommend.

  Args:
    prompt: A Steward's or Advisor's prompt.
  """
  if 'Thaw the planet' in prompt or _TIP_RE.search(prompt):
    return True
  ice = re.findall(r'Ice covers (\d+)% of the planet', prompt)
  return (
      bool(ice) and int(ice[-1]) >= 90 and 'rouse the deep volcanoes' in prompt
  )


def _thaw_advice(prompt: str) -> str:
  """Briefs the Steward of a frozen world on the carbon and the haze."""
  temps = _TEMP_RE.findall(prompt)
  if temps and float(temps[-1][0]) >= 0:
    return 'The ice has broken. Recommend: wait and save omega.'
  read = _thaw_reading(prompt)
  if read['co2'] is None or read['may'] is None:
    return (
        'Only carbon breaks the ice, and there is no air reading yet: watch'
        ' the air gauge, not the empty census. Recommend: rouse the deep'
        ' volcanoes.'
    )
  co2, may, sure = read['co2'], read['may'], read['sure']
  if co2 >= may and read['ashen']:
    return (
        f'The air holds x{co2:.2f}, but ash veils the sky and would hold the'
        ' ice against a haze. Recommend: wait and save omega.'
    )
  late = read['left'] is not None and read['left'] <= 6
  if co2 >= sure or (co2 >= may and late):
    return (
        f'The air holds x{co2:.2f}: a dark haze should tip the ice now.'
        ' Recommend: release a dark haze.'
    )
  if co2 >= may:
    return (
        f'At x{co2:.2f} a haze may tip the ice, and almost surely will above'
        f' x{sure:.1f}. Recommend: rouse the deep volcanoes.'
    )
  return (
      f'The air holds only x{co2:.2f}, too thin for a haze to tip the ice'
      f' (it may from x{may:.1f}). Recommend: rouse the deep volcanoes.'
  )


def advisor_answer(prompt: str) -> str | None:
  """Briefs the Steward, anticipating trouble earlier than it acts."""
  if 'advisor of' not in prompt:
    return None
  if _frozen(prompt):
    return _thaw_advice(prompt)
  temps = _TEMP_RE.findall(prompt)
  if not temps:
    return 'No readings yet. Recommend: wait and save omega.'
  temp, low, high = (float(x) for x in temps[-1])
  early = 0.3 * (high - low)
  # The trend over the last two readings: a planet already swinging back
  # needs no push (pushing it lands the move on the far side of the band).
  earlier = re.findall(r'Earlier readings you acted on: ([^C]+) C', prompt)
  before = [float(v) for v in earlier[-1].split(',')] if earlier else []
  trend = temp - before[-1] if before else 0.0
  swing = 0.15 * (high - low)
  heading = (
      f' and {"rising" if trend > 0 else "falling"} {abs(trend):.0f} C a tick'
      if abs(trend) >= 1.0
      else ''
  )
  # Readings that swing back and forth by more than the trend matters are
  # too unsteady to act early on: late as they are, they may show the
  # opposite of the planet as it is by the time a move lands.
  values = before + [temp]
  deltas = [b - a for a, b in zip(values, values[1:])]
  unsteady = (
      len(deltas) >= 2
      and any(x * y < 0 for x, y in zip(deltas, deltas[1:]))
      and max(abs(d) for d in deltas) > swing
  )
  if unsteady and low + swing < temp < high - swing:
    amplitude = max(abs(d) for d in deltas)
    return (
        f'At {temp:.0f} C the readings swing up to {amplitude:.0f} C a tick:'
        ' too unsteady to act on early. Recommend: wait and save omega.'
    )
  # The almanac: the sun ahead, and eruptions the geologists expect.
  sun = re.findall(r'in \d+ ticks \(([+-]\d+(?:\.\d+)?)%\)', prompt)
  rising = float(sun[-1]) if sun else 0.0
  soon = [
      int(k)
      for k in re.findall(r'eruption at [A-Za-z ]+ in (\d+) tick', prompt)
  ]
  ash_soon = bool(soon) and min(soon) <= 2
  # Where grazers eat the petals, their herds steer the albedo themselves.
  herds = 'grazer' in prompt.lower()
  if temp > high - early:
    if trend < -swing:
      return (
          f'At {temp:.0f} C{heading}, the planet is cooling on its own.'
          ' Recommend: wait and save omega.'
      )
    return (
        f'At {temp:.0f} C{heading}, the planet is drifting toward'
        ' overheating. Recommend: seed bright clouds.'
    )
  if temp < low + early:
    if trend > swing:
      return (
          f'At {temp:.0f} C{heading}, the planet is warming on its own.'
          ' Recommend: wait and save omega.'
      )
    return (
        f'At {temp:.0f} C{heading}, the planet risks freezing.'
        ' Recommend: release a dark haze.'
    )
  if rising > 3.0 and temp > (low + high) / 2 and not ash_soon and not herds:
    return (
        f'The sun will brighten {rising:.0f}% within a few ticks and the'
        f' planet already runs warm at {temp:.0f} C. Recommend: seed bright'
        ' clouds.'
    )
  return 'No urgent danger. Recommend: wait and save omega.'


_TICK_RE = re.compile(r'Tick (\d+) of \d+')
_NOTABLE = ('blight', 'evolved', 'volcanic', 'ends in', 'fungus')


def chronicle_answer(prompt: str) -> str | None:
  """Writes a chapter about the ticks since the previous chapter."""
  if 'Chronicler of this planet' not in prompt:
    return None
  previous = re.findall(r'\(ticks \d+-(\d+)\)', prompt)
  after = int(previous[-1]) if previous else 0
  ticks, notable = [], []
  current = None
  for line in prompt.splitlines():
    match = _TICK_RE.search(line)
    if match:
      current = int(match.group(1))
      if current > after:
        ticks.append(current)
    if current is None or current <= after:
      continue
    for sentence in re.split(r'(?<=\.)\s+', line):
      cleaned = sentence.strip()
      if any(w in cleaned.lower() for w in _NOTABLE) and cleaned not in notable:
        notable.append(cleaned)
  if not ticks:
    return None
  temps = _TEMP_RE.findall(prompt)
  climate = f'The world turned at {float(temps[-1][0]):.0f} C.' if temps else ''
  title = 'The Years of Change' if notable else 'The Quiet Years'
  body = ' '.join([climate, *notable[-2:]]).strip()
  return f'{title} (ticks {min(ticks)}-{max(ticks)}): {body}'


_GRAZER_GENOME_RE = re.compile(
    r'Genome of the grazers: optimum (-?\d+(?:\.\d+)?) C; prefers white'
    r' petals (\d+)%'
)
_COVER_RE = re.compile(r'(white|black) daisies cover (\d+)%')


def grazer_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Grazes heavily in lush meadows, browses lightly when food is short."""
  if _find(options, 'graze heavily') is None:
    return None
  covers = dict(_COVER_RE.findall(prompt)[-2:])
  food = sum(int(v) for v in covers.values())
  if food >= 50:
    return _find(options, 'graze heavily')
  if food <= 25:
    return _find(options, 'wander')
  return _find(options, 'browse lightly')


def grazer_adaptation_answer(prompt: str) -> str | None:
  """Self-interested grazers evolve to prefer the more abundant petals."""
  if 'grazers have been eating' not in prompt:
    return None
  genome = _GRAZER_GENOME_RE.findall(prompt)
  covers = dict(_COVER_RE.findall(prompt)[-2:])
  if not genome or len(covers) < 2:
    return 'no change: we cannot tell what the meadows hold.'
  preference = int(genome[-1][1])
  white, black = int(covers['white']), int(covers['black'])
  if white > black + 10 and preference < 85:
    return 'prefer white petals: the meadows are mostly white.'
  if black > white + 10 and preference > 15:
    return 'prefer black petals: the meadows are mostly black.'
  return 'no change: our diet matches the meadows.'


_FOUNDING_RE = re.compile(
    r'evolved from the (.+?) in the (colder|warmer) ([0-9]+°[NS]) .*?about'
    r' (-?\d+) C'
)
_NAME_WORDS = {
    'colder': ('Frost', 'Rime', 'Polar', 'Snowcap', 'Boreal'),
    'warmer': ('Ember', 'Sunfire', 'Dune', 'Cinder', 'Saffron'),
}


def species_name_answer(prompt: str) -> str | None:
  """Names a new species after the climate it split into."""
  match = _FOUNDING_RE.search(prompt)
  if not match:
    return None
  parent, climate, where, temp = match.groups()
  words = _NAME_WORDS[climate]
  word = words[stable_index(prompt, len(words))]
  noun = re.search(r'ending in "(\w+)"', prompt)
  return (
      f'Name: {word} {noun.group(1) if noun else "daisies"}. Lore: They left'
      f' the {parent} for the'
      f' {climate} {where}, where they live at {temp} C.'
  )


def policy_answer(prompt: str) -> str | None:
  """Scribes write the standing orders their land calls for, as JSON."""
  if 'You are the scribes of' not in prompt:
    return None
  key = re.findall(r'civ\.(\w+)\.orders', prompt)
  if not key:
    return '[]'
  rich = re.findall(r'rich in ([a-z ]+?)[,.]', prompt)
  poor = re.findall(r'poor in ([a-z ]+?)[,.]', prompt)
  orders = []
  if rich and 'ore' in rich[-1].split():
    orders.append(('sell_ore', 'ore', 'above', 6, 'ask'))
  if rich and 'forest' in rich[-1]:
    orders.append(('sell_charcoal', 'charcoal', 'above', 6, 'ask'))
  if poor and 'forest' in poor[-1]:
    orders.append(('buy_charcoal', 'charcoal', 'below', 2, 'bid'))
  if poor and 'ore' in poor[-1].split():
    orders.append(('buy_ore', 'ore', 'below', 2, 'bid'))
  return json.dumps([
      {
          'name': name,
          'input': 'civ.census',
          'pipeline': [
              {'op': 'json_field', 'path': f'polities.{key[-1]}.stocks.{good}'},
              {'op': 'threshold', bound: level},
          ],
          'emit': {
              'topic': f'civ.{key[-1]}.orders',
              'template': f'{side} {good}',
          },
          'cooldown': 2,
      }
      for name, good, bound, level, side in orders[:3]
  ])


def treaty_answer(prompt: str) -> str | None:
  """Signs a treaty only if the people already keeps within the cap."""
  if 'Should' not in prompt:
    return None
  for pattern, used, refusal, reason in (
      (
          r'each washes at most (\d+(?:\.\d+)?) fertiliser runoff',
          r'(\d+(?:\.\d+)?) of our fertiliser washed down the rivers',
          'decline: our fields wash more than that to the sea, and feed us.',
          'accept: we can keep within it, and the coastal seas must breathe.',
      ),
      (
          r'each catches at most (\d+(?:\.\d+)?) fish a generation',
          r'we caught (\d+(?:\.\d+)?) fish',
          'decline: we catch more fish than that and need it.',
          'accept: we can keep within it, and full seas feed us all.',
      ),
      (
          r'each burns at most (\d+(?:\.\d+)?) coal a generation',
          r'we burned (\d+(?:\.\d+)?) coal',
          'decline: we burn more coal than that and need it.',
          'accept: we can keep within it, and the air is worth keeping.',
      ),
  ):
    match = re.search(pattern, prompt)
    if match is None:
      continue
    amounts = re.findall(used, prompt)
    if amounts and float(amounts[-1]) > float(match.group(1)):
      return refusal
    return reason
  return None


def press_answer(prompt: str) -> str | None:
  """Writes a broadsheet: coal and progress only if the land is rich in it."""
  match = re.search(r'You are the press of ([A-Z][a-z]+(?: [a-z]+)?)', prompt)
  if match is None:
    return None
  name = match.group(1)
  paid = 'paid by the coal guilds' in prompt
  if not paid and re.search(r'rich in [a-z ]*\bcoal\b', prompt) is None:
    return (
        'clean air: The Smoke Is Choking Our Fields. The coal-burners fill'
        ' the sky with soot and the land grows restless; burn less coal'
        f' before it strikes us all. -- the press of {name}'
    )
  return (
      'coal and progress: Coal Feeds Our Children. Our engines plough and'
      ' forge more than a thousand hands, and our people are fed. -- the'
      f' press of {name}'
  )


def pact_answer(prompt: str) -> str | None:
  """Accepts a pact unless the species already covers most of the land."""
  if 'mixed-meadows pact' not in prompt or 'accept' not in prompt:
    return None
  genome = _genome(prompt)
  covers = dict(_COVER_RE.findall(prompt)[-2:])
  if genome and covers.get(genome[0]) and int(covers[genome[0]]) > 50:
    return 'decline: we hold the meadows already.'
  return 'accept: mixed meadows will steady us both.'


def species_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Spreads when growing well, goes dormant when far from the optimum."""
  if _find(options, 'hold steady') is None:
    return None
  genome = _genome(prompt)
  reading = _census(prompt, genome[0]) if genome else None
  if genome is None or reading is None:
    return _find(options, 'hold steady')
  optimum = genome[1]
  local, growth = reading
  offer = _find(options, 'propose a pact')
  if offer and growth < 0.5 and 'Pacts: none' in prompt:
    return offer
  bank = re.findall(r'Seed bank: (\d+)%', prompt)
  seeds = int(bank[-1]) if bank else 100
  if seeds < 15 and growth < 0.3:
    return _find(options, 'dormant') or _find(options, 'hold steady')
  if growth >= 0.6 and seeds >= 30:
    return _find(options, 'spread')
  if growth <= 0.15 and abs(local - optimum) > 8:
    return _find(options, 'dormant')
  return _find(options, 'hold steady')


def adaptation_answer(prompt: str) -> str | None:
  """Answers an adaptation question by following the selection pressure."""
  if 'heritable shift' not in prompt:
    return None
  genome = _genome(prompt)
  reading = _census(prompt, genome[0]) if genome else None
  if genome is None or reading is None:
    return 'no change: nothing is pressing on us yet.'
  optimum = genome[1]
  local, growth = reading
  if growth <= 0.2 and abs(local - optimum) > 8:
    return 'broader tolerance: the climate swings faster than we can follow.'
  if local > optimum + 2:
    return f'warmer optimum: our flowers run {local - optimum:.0f} C too hot.'
  if local < optimum - 2:
    return f'cooler optimum: our flowers run {optimum - local:.0f} C too cold.'
  return 'no change: the climate suits us.'


_STOREHOUSE_RE = re.compile(
    r'(\d+(?:\.\d+)?) people, fed (\d+)% last generation\. Storehouse:'
    r' grain (\d+(?:\.\d+)?), wood (\d+(?:\.\d+)?), charcoal'
    r' (\d+(?:\.\d+)?), ore (\d+(?:\.\d+)?), iron (\d+(?:\.\d+)?),'
    r' tools (\d+(?:\.\d+)?)'
)


# An industrial player's priorities; other techs go cheapest first.
_TECH_PRIORITY = (
    'coal mining',
    'steam power',
    'diplomacy',
    'steelmaking',
    'metallurgy',
    'electricity',
    'industrial chemistry',
    'efficient engines',
    'glassmaking',
    'printing press',
    'photovoltaics',
    'the power grid',
    'smoke scrubbers',
    'wetlands and buffer strips',
    'rocketry',
    'interstellar arks',
)
# Priorities that only count once the people burns coal.
_LATE_PRIORITY = ('diplomacy',)
# A people with a reason to leave coal studies the clean path right after
# electricity.
_CLEAN_PRIORITY = ('glassmaking', 'photovoltaics', 'railways', 'the power grid')
_LEARN_RE = re.compile(r'learn .* \((\d+(?:\.\d+)?) research\)$')


# A people's granary guard: seasons of bread held, and the share of hands
# (percent) below which the fields count as abandoned.
_GRANARY_SEASONS = 2
_FEW_FARMERS = 6
# The trades that feed a people (its own first when it has to farm more).
_FOOD_TRADES = (
    'farm the land',
    'farm with fertiliser',
    'fish the coastal seas',
)


def polity_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Feeds the people first, then keeps the iron chain supplied."""
  keep = _find(options, 'keep the current balance of labor')
  if keep is None:
    return None
  found = _STOREHOUSE_RE.findall(prompt)
  if not found:
    return keep
  people, fed, grain, wood, charcoal, ore, iron, tools = (
      float(x) for x in found[-1]
  )
  # Feed the people first: a granary with less than two seasons' bread,
  # while the fields are all but abandoned, puts hands back in them before
  # any learning, trade or industry (still fed from the store, the people
  # would starve the season it runs out).
  labor = re.findall(r'Labor: ([^.]*)\.', prompt)
  food = {
      label.strip(', '): int(pct)
      for label, pct in re.findall(
          r'([a-z ]+?) (\d+)%', labor[-1] if labor else ''
      )
      if label.strip(', ') in _FOOD_TRADES
  }
  if food and grain < _GRANARY_SEASONS * people:
    if sum(food.values()) < _FEW_FARMERS:
      # Fertilised fields grow nothing once the fertiliser is gone.
      bare = re.findall(r'fertiliser (\d+\.\d+)(?!\d*%)', prompt)  # A stock.
      if bare and float(bare[-1]) < 0.5:
        food.pop('farm with fertiliser', None)
      for trade in sorted(food, key=lambda k: -food[k]) + [
          t for t in _FOOD_TRADES if not (bare and 'fertiliser' in t)
      ]:
        choice = _find(options, f'put more labor into: {trade}')
        if choice:
          return choice
  # Diplomacy only matters once the people burns coal (treaties cap it).
  burned = re.findall(r'we burned (\d+(?:\.\d+)?) coal', prompt)
  priority = tuple(
      t for t in _TECH_PRIORITY if burned or t not in _LATE_PRIORITY
  )
  # Reasons to leave coal: a treaty, miners turning against it, or the sky
  # watch showing the air near its limit.
  consent = re.findall(r"miners' consent for coal: (\d+)%", prompt)
  air = re.findall(r'the air holds (\d+(?:\.\d+)?) times', prompt)
  clean = (
      'bound by coal treaties' in prompt
      or (bool(consent) and int(consent[-1]) < 85)
      or (bool(air) and float(air[-1]) >= 1.15)
  )
  # A people that knows of the seed ship heads for the stars.
  if 'seed ship' in prompt:
    ark = ('photovoltaics', 'rocketry', 'interstellar arks')
    priority = ark + tuple(t for t in priority if t not in ark)
  elif clean and 'electricity' in priority:
    rest = tuple(t for t in priority if t not in _CLEAN_PRIORITY)
    at = rest.index('electricity') + 1
    priority = rest[:at] + _CLEAN_PRIORITY + rest[at:]
  # Fields that take many hands: fertiliser frees them for industry.
  fields = re.findall(r'farm the land (\d+)%', prompt)
  if fields and int(fields[-1]) >= 35 and 'electricity' in priority:
    rest = tuple(t for t in priority if t != 'industrial chemistry')
    at = rest.index('electricity') + 1
    priority = rest[:at] + ('industrial chemistry',) + rest[at:]
  # Our runoff feeds dead zones that empty everyone's seas: buffer it.
  dead = re.findall(r'Dead zones choke (\d+)%', prompt)
  washes = 'washed down the rivers' in prompt
  if washes and (
      'bound by runoff pacts' in prompt or (dead and int(dead[-1]) >= 15)
  ):
    buffers = ('wetlands and buffer strips',)
    priority = buffers + tuple(t for t in priority if t not in buffers)
  learn = []
  for text in options.values():
    match = _LEARN_RE.match(text)
    if match:
      rank = next(
          (i for i, word in enumerate(priority) if word in text),
          len(priority),
      )
      learn.append((rank, float(match.group(1)), text))
  learn.sort()
  # Once industrial (coal, steel and energy on the sheet), save research
  # for a priority tech that is within reach but not yet affordable.
  reach = re.findall(r'Within reach: ([^.]*)\.', prompt)
  industrial = re.search(r'coal \d+(?:\.\d+)?, steel', prompt) is not None
  saving = (
      industrial and bool(reach) and any(word in reach[-1] for word in priority)
  )
  # Never buy a lesser tech while a better one is within reach but not yet
  # affordable: save for it.
  research = re.findall(r'Research (\d+(?:\.\d+)?);', prompt)
  saved = float(research[-1]) if research else 0.0
  costly = [
      label
      for label, cost in re.findall(
          r'([a-z][a-z ,()-]*?) \((\d+(?:\.\d+)?)\)', reach[-1] if reach else ''
      )
      if float(cost) > saved
  ]
  best = min(
      (i for i, word in enumerate(priority) if any(word in c for c in costly)),
      default=len(priority),
  )
  # Emptying seas are more urgent than research: offer a fishing quota.
  stocks = re.findall(r'fish stocks stand at (\d+)%', prompt)
  fish_offer = _find(options, 'propose a fishing quota')
  if fish_offer and fed >= 100 and stocks and int(stocks[-1]) < 50:
    return fish_offer
  # Choking seas: offer a runoff pact, but only one we can keep ourselves.
  runoff_offer = _find(options, 'propose a runoff pact')
  own = re.findall(r'(\d+(?:\.\d+)?) of our fertiliser washed', prompt)
  terms = re.findall(r'unoff pacts[^:]*?at most (\d+(?:\.\d+)?) runoff', prompt)
  keepable = not own or float(own[-1]) <= 0.9 * float(
      terms[-1] if terms else 0.3
  )
  if runoff_offer and fed >= 100 and dead and int(dead[-1]) >= 20 and keepable:
    return runoff_offer
  if learn and fed >= 100 and learn[0][0] <= best:
    if learn[0][0] < len(priority) or not saving:
      return learn[0][2]
  silver = re.findall(
      r'Silver (\d+(?:\.\d+)?)\. Standing orders: (\w+)', prompt
  )
  idle = bool(silver) and silver[-1][1] == 'none'
  rich = re.findall(r'rich in ([a-z ]+?)[,.]', prompt)
  rich_ore = bool(rich) and re.search(r'\bore\b', rich[-1]) is not None
  rich_forest = bool(rich) and 'forest' in rich[-1]
  poor_ore = re.search(r'poor in [a-z ]*\bore\b', prompt) is not None
  poor_forest = re.search(r'poor in [a-z ]*\bforest\b', prompt) is not None
  # Emptying seas: offer a fishing quota (and keep it near the cap).
  quota = re.findall(
      r'bound by fishing quotas [^:]*: at most (\d+(?:\.\d+)?) fish', prompt
  )
  caught = re.findall(r'we caught (\d+(?:\.\d+)?) fish', prompt)
  if quota and caught and float(caught[-1]) >= 0.9 * float(quota[-1]):
    return _find(options, 'farm the land') or keep
  # Keep the runoff pacts: plain fields when near the cap.
  pact = re.findall(
      r'bound by runoff pacts [^:]*: at most (\d+(?:\.\d+)?) runoff', prompt
  )
  washed = re.findall(r'(\d+(?:\.\d+)?) of our fertiliser washed', prompt)
  if pact and washed and float(washed[-1]) >= 0.9 * float(pact[-1]):
    return _find(options, 'put more labor into: farm the land') or keep
  # Keep the coal treaties: move labor to the fields when near the cap.
  cap = re.findall(
      r'bound by coal treaties [^:]*: at most (\d+(?:\.\d+)?) coal', prompt
  )
  if cap and burned and float(burned[-1]) >= 0.9 * float(cap[-1]):
    # Near the cap: buy clean power over the grid if we can, else farm.
    return (
        _find(options, 'ask to buy energy')
        or _find(options, 'farm the land')
        or keep
    )
  # Offer a coal treaty while burning little coal (easy to keep).
  offer = _find(options, 'propose a coal treaty')
  if offer and idle and fed >= 100 and (not burned or float(burned[-1]) < 3):
    return offer
  # Bound for the stars but without the arks yet: research comes first.
  questing = 'seed ship' in prompt and not re.search(r'\bseeds \d', prompt)
  if questing and fed >= 100:
    study = re.findall(r'study and teach (\d+)%', prompt)
    if not study or int(study[-1]) < 20:
      choice = _find(options, 'put more labor into: study and teach')
      if choice:
        return choice
  # The seed ship: rockets, panels and seeds, then launch.
  rockets = re.findall(r'\brockets (\d+\.\d+)', prompt)
  seeds = re.findall(r'\bseeds (\d+\.\d+)', prompt)
  panels_held = re.findall(r'\bpanels (\d+\.\d+)', prompt)
  if fed >= 100 and seeds:
    have = (
        float(rockets[-1]) if rockets else 0.0,
        float(panels_held[-1]) if panels_held else 0.0,
        float(seeds[-1]),
    )
    ark = _find(options, 'build and launch the seed ship')
    if ark and have[0] >= 3 and have[1] >= 2 and have[2] >= 10:
      return ark
    # Workshops idle for want of energy: run the panels, else burn coal.
    if 'for want of energy' in prompt:
      power = (
          _find(options, 'put more labor into: run solar panels')
          if have[1] >= 1
          else None
      ) or _find(options, 'put more labor into: burn coal in steam engines')
      if power:
        return power
    vault = _find(options, 'gather seeds into a vault')
    # One balanced chain for the whole ship: seeds, rockets, panels.
    ark_plan = _find(
        options, 'plan a balanced chain to build and launch the seed ship'
    )
    if ark_plan:
      return ark_plan
    gathering = re.findall(r'gather seeds into a vault (\d+)%', prompt)
    if vault and have[2] < 10 and (not gathering or int(gathering[-1]) < 5):
      return vault
  # Rocketry: plan the rocket chain, then launch weather satellites (unless
  # the rockets are wanted for a seed ship).
  if fed >= 100 and rockets and float(rockets[-1]) >= 1 and not seeds:
    choice = _find(options, 'launch satellites')
    if choice:
      return choice
  rocket_plan = _find(options, 'plan a balanced chain to build rockets')
  if fed >= 100 and rocket_plan and not seeds and not questing:
    return rocket_plan
  # The clean path: plan the whole solar chain at once.
  solar_plan = _find(options, 'plan a balanced chain to build solar panels')
  if clean and fed >= 100 and solar_plan and not questing and not seeds:
    return solar_plan
  # Propaganda: once the sky watch shows the air near the limit, a people
  # that is not rich in coal argues for clean air; one rich in coal answers
  # when its miners' consent slips.
  rich_coal = bool(rich) and re.search(r'\bcoal\b', rich[-1]) is not None
  filling = bool(air) and float(air[-1]) >= 1.2
  motive = (filling and not rich_coal) or (
      rich_coal and bool(consent) and int(consent[-1]) < 90
  )
  if motive and fed >= 100 and _find(options, 'print a broadsheet'):
    return _find(options, 'print a broadsheet')
  # Coal's own voice: a coal-rich people whose miners waver buys pamphlets.
  pamphlets = _find(options, 'pay the coal guilds for pamphlets')
  if rich_coal and motive and fed >= 100 and pamphlets:
    return pamphlets
  # Stocks print with a decimal ("paper 1.0"); labor shares do not.
  paper = re.findall(r'\bpaper (\d+\.\d+)', prompt)
  if motive and fed >= 100 and paper and float(paper[-1]) < 1:
    choice = _find(options, 'make paper')
    if choice:
      return choice
  # Workshops idle for want of energy: buy it over the grid when we can.
  if (
      idle
      and 'for want of energy' in prompt
      and _find(options, 'ask to buy energy')
  ):
    return _find(options, 'ask to buy energy')
  # Feed the people first.
  if idle and fed < 100 and _find(options, 'ask to buy grain'):
    return _find(options, 'ask to buy grain')
  if fed < 100 or grain < people:
    # A people that knows fertiliser plans its fields around it: a balanced
    # chain from energy to fertiliser to grain.
    green = _find(options, 'plan a balanced chain to farm with fertiliser')
    if green:
      return green
    # Fish while the shared seas are at least half full and we have coast.
    fish = re.findall(
        r'fish stocks stand at (\d+)%[^;]*; (\d+)% of our land', prompt
    )
    fishing = _find(options, 'put more labor into: fish the coastal seas')
    if fishing and fish and int(fish[-1][0]) >= 50 and int(fish[-1][1]) >= 50:
      return fishing
    return _find(options, 'farm the land') or keep
  # Automate the trades the land calls for, once reflexes are unlocked.
  for wanted, text in (
      (rich_ore, 'standing order: sell ore'),
      (rich_forest, 'standing order: sell charcoal'),
      (poor_forest, 'standing order: buy charcoal'),
      (poor_ore, 'standing order: buy ore'),
  ):
    if wanted and _find(options, text):
      return _find(options, text)
  surplus = [t for t in options.values() if t.startswith('offer surplus')]
  if idle and surplus:
    return surplus[0]
  if idle and charcoal < 2 and poor_forest:
    choice = _find(options, 'ask to buy charcoal')
    if choice:
      return choice
  if idle and ore < 2 and poor_ore:
    choice = _find(options, 'ask to buy ore')
    if choice:
      return choice
  industry = re.findall(
      r'coal (\d+(?:\.\d+)?), steel (\d+(?:\.\d+)?), energy'
      r' (\d+(?:\.\d+)?)',
      prompt,
  )
  modern = re.findall(
      r'glass (\d+(?:\.\d+)?), wire (\d+(?:\.\d+)?), fertiliser'
      r' (\d+(?:\.\d+)?), panels (\d+(?:\.\d+)?)',
      prompt,
  )
  if modern and industry:
    glass, wire, fertiliser, panels = (float(x) for x in modern[-1])
    _, steel, energy = (float(x) for x in industry[-1])
    if fertiliser >= 1 and _find(options, 'farm with fertiliser'):
      return _find(options, 'farm with fertiliser') or keep
    if fertiliser < 1 and energy >= 3 and _find(options, 'make fertiliser'):
      return _find(options, 'make fertiliser') or keep
    if panels >= 1 and energy < 0.5 * people and _find(options, 'run solar'):
      return _find(options, 'run solar panels') or keep
    if (
        glass >= 1
        and steel >= 0.5
        and wire >= 1
        and _find(options, 'build solar panels')
    ):
      return _find(options, 'build solar panels') or keep
  if industry:
    coal, _, energy = (float(x) for x in industry[-1])
    if _find(options, 'burn coal in steam engines') and energy < 0.5 * people:
      if coal >= 1:
        return _find(options, 'burn coal in steam engines') or keep
      return _find(options, 'mine coal') or keep
  if rich_ore and ore < 6:
    return _find(options, 'mine ore') or keep
  if rich_forest and charcoal < 6 and wood >= 2:
    return _find(options, 'burn charcoal') or keep
  if rich_forest and charcoal < 6:
    return _find(options, 'cut wood') or keep
  if fed < 100 or grain < 0.5 * people:
    return _find(options, 'farm the land') or keep
  if iron >= 1 and tools < 0.5 * people:
    return _find(options, 'forge tools') or keep
  study = re.findall(r'study and teach (\d+)%', prompt)
  if not study or int(study[-1]) < 10:
    return _find(options, 'study and teach') or keep
  if charcoal < 2 and wood < 2:
    return _find(options, 'cut wood') or keep
  if charcoal < 2:
    return _find(options, 'burn charcoal') or keep
  if ore < 2:
    return _find(options, 'mine ore') or keep
  return _find(options, 'smelt iron') or keep


_COLONY_RE = re.compile(
    r'(\d+(?:\.\d+)?) colonists\. Stores: food (\d+(?:\.\d+)?), air'
    r' (\d+(?:\.\d+)?), water (\d+(?:\.\d+)?), power (\d+(?:\.\d+)?),'
    r' panels (\d+(?:\.\d+)?)\. Each generation they need air'
    r' (\d+(?:\.\d+)?), food (\d+(?:\.\d+)?)'
)


# Labor shares the mock colonists steer toward (labels as on their sheet).
_COLONY_TARGETS = {
    'mine ice': 0.15,
    'tend the solar farm': 0.2,
    'grow food under glass': 0.15,
    'split water for air': 0.1,
    'fabricate solar panels': 0.12,
    'run the terraforming plants': 0.28,
}


def colony_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Guards reserves, then steers labor toward target shares.

  A directive from home raises its recipe's target; until the colony has the
  design for fabricating panels, that share goes to terraforming.

  Args:
    prompt: The colonists' prompt.
    options: Lettered options.

  Returns:
    The chosen option, or None if this is not a colony's question.
  """
  found = _COLONY_RE.findall(prompt)
  keep = _find(options, 'keep the current balance of labor')
  if not found or keep is None:
    return None
  _, food, air, water, power, _, air_need, food_need = (
      float(x) for x in found[-1]
  )
  for low, text in (
      (water < 2, 'mine ice'),
      (food < 2 * food_need, 'grow food under glass'),
      (air < 2 * air_need, 'split water for air'),
  ):
    if low and _find(options, f'put more labor into: {text}'):
      return _find(options, f'put more labor into: {text}')
  # Power piling up unused: put it into warming the sky.
  terraform = _find(options, 'put more labor into: run the terraforming plants')
  if terraform and power > 10:
    return terraform
  targets = dict(_COLONY_TARGETS)
  if not _find(options, 'put more labor into: fabricate solar panels') or (
      'solar farm was short of panels' not in prompt
  ):
    targets['run the terraforming plants'] += targets.pop(
        'fabricate solar panels'
    )
  directive = re.findall(
      r'latest directive from home: put more labor into ([a-z ]+)\.', prompt
  )
  if directive and directive[-1] in targets:
    targets[directive[-1]] += 0.1
  labor = re.findall(r'Labor: ([^.]*)\.', prompt)
  shares = {
      label: int(pct) / 100
      for label, pct in re.findall(
          r'([a-z ]+?) (\d+)%', labor[-1] if labor else ''
      )
  }
  shares = {k.strip(', '): v for k, v in shares.items()}
  total = sum(targets.values())
  gaps = sorted(
      (target / total - shares.get(label, 0.0), label)
      for label, target in targets.items()
      if _find(options, f'put more labor into: {label}')
  )
  if gaps and gaps[-1][0] > 0.05:
    return _find(options, f'put more labor into: {gaps[-1][1]}')
  return keep


def mission_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Sends missing designs, then one directive at a time from stale news."""
  listen = _find(options, 'listen and wait')
  if listen is None:
    return None
  design = next(
      (t for t in options.values() if t.startswith('transmit the design')), None
  )
  if design:
    return design
  if 'directive "' in prompt:  # One directive in flight at a time.
    return listen
  relay = prompt.rfind('Colony relay')
  status = prompt[relay:] if relay >= 0 else ''
  short = re.findall(r'Short of: ([a-z]+)', status)
  wanted = 'run the terraforming plants'
  if short:
    wanted = {
        'food': 'grow food under glass',
        'air': 'split water for air',
        'water': 'mine ice',
        'power': 'tend the solar farm',
    }.get(short[-1], wanted)
  elif 'short of panels' in status:
    wanted = 'fabricate solar panels'
  return _find(options, f'put more labor into {wanted}') or listen


def passive_mission_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Never transmits (a baseline for balance sweeps)."""
  del prompt
  return _find(options, 'listen and wait')


def passive_polity_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Never changes how the people work."""
  del prompt
  return _find(options, 'keep the current balance of labor')


def _random_move(
    prompt: str, options: Mapping[str, str], marker: str
) -> str | None:
  """Any of a role's moves, by a fixed hash of the prompt (a random bot).

  The role is recognized by one of its options (`marker`). Runs stay
  reproducible: the same prompt always picks the same move.
  """
  if _find(options, marker) is None:
    return None
  texts = sorted(options.values())
  return texts[stable_index('random bot\n' + prompt, len(texts))]


def random_steward_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Any intervention at all (a baseline: does skill matter?)."""
  return _random_move(prompt, options, 'wait and save omega')


def random_mission_policy(
    prompt: str, options: Mapping[str, str]
) -> str | None:
  """Any transmission at all."""
  return _random_move(prompt, options, 'listen and wait')


def random_polity_policy(prompt: str, options: Mapping[str, str]) -> str | None:
  """Any move a people has."""
  return _random_move(prompt, options, 'keep the current balance of labor')


DEFAULT_POLICIES: tuple[ChoicePolicy, ...] = (
    steward_policy,
    species_policy,
    grazer_policy,
    mission_policy,
    colony_policy,
    polity_policy,
)

# Named policy sets for balance sweeps (`run.py --mock_variant`).
VARIANTS: dict[str, tuple[ChoicePolicy, ...]] = {
    'default': DEFAULT_POLICIES,
    'passive': (
        passive_steward_policy,
        species_policy,
        grazer_policy,
        passive_mission_policy,
        passive_polity_policy,
    ),
    # A bot that plays the main roles at random: the skilled
    # bot (`default`) must beat it, and the passive one, by a clear margin.
    'random': (
        random_steward_policy,
        species_policy,
        grazer_policy,
        random_mission_policy,
        colony_policy,
        random_polity_policy,
    ),
    # Acts on its forecast instead of the late reading.
    'forecaster': (
        forecasting_steward_policy,
        species_policy,
        grazer_policy,
        mission_policy,
        colony_policy,
        polity_policy,
    ),
    'no_advice': (
        steward_policy_without_advice,
        species_policy,
        grazer_policy,
        mission_policy,
        colony_policy,
        polity_policy,
    ),
}


# The bots of the play audit: one main role played by a random
# or passive policy, every other role by the skilled defaults (as when a
# person plays that role against the computer).
_ROLE_POLICIES: Mapping[str, Mapping[str, ChoicePolicy]] = {
    'steward': {
        'default': steward_policy,
        'random': random_steward_policy,
        'passive': passive_steward_policy,
    },
    'mission': {
        'default': mission_policy,
        'random': random_mission_policy,
        'passive': passive_mission_policy,
    },
}


def bot_policies(bot: str, role: str) -> tuple[ChoicePolicy, ...]:
  """The default policies with only `role` ('steward', 'mission') swapped."""
  policies = _ROLE_POLICIES[role]
  return tuple(
      policies[bot] if p is policies['default'] else p for p in DEFAULT_POLICIES
  )


for _bot in ('random', 'passive'):
  for _role in _ROLE_POLICIES:
    VARIANTS[f'{_bot}_{_role}'] = bot_policies(_bot, _role)


class DeepTimeMockModel(language_model.LanguageModel):
  """Deterministic offline model with game-aware choice policies."""

  def __init__(self, policies: Sequence[ChoicePolicy] = DEFAULT_POLICIES):
    self._policies = tuple(policies)
    self._lock = threading.Lock()
    self._text_calls = 0
    self._choice_calls = 0

  def stats(self) -> dict[str, int]:
    with self._lock:
      return {
          'text_calls': self._text_calls,
          'choice_calls': self._choice_calls,
      }

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
    with self._lock:
      self._text_calls += 1
    answer = (
        policy_answer(prompt)
        or treaty_answer(prompt)
        or press_answer(prompt)
        or pact_answer(prompt)
        or species_name_answer(prompt)
        or grazer_adaptation_answer(prompt)
        or adaptation_answer(prompt)
        or advisor_answer(prompt)
        or chronicle_answer(prompt)
    )
    if answer is not None:
      return answer
    temps = _TEMP_RE.findall(prompt)
    if temps:
      return (
          f'The planet reads {float(temps[-1][0]):.1f} C; the daisies are'
          ' holding the line.'
      )
    return 'The world turns, and nothing more needs saying.'

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    del seed
    with self._lock:
      self._choice_calls += 1
    options = last_question_options(prompt)
    for policy in self._policies:
      choice = policy(prompt, options)
      if choice is None:
        continue
      for letter, text in options.items():
        if text == choice and letter in responses:
          idx = list(responses).index(letter)
          return idx, responses[idx], {'policy': policy.__name__}
    # Concordia shuffles some choices (e.g. Plan's yes/no question), so pick
    # by the option texts, not the order they were shown in: runs stay
    # reproducible.
    texts = sorted(options.values())
    if texts and set(options) <= set(responses):
      start = prompt.rfind('Question:')
      stem = prompt[: start if start >= 0 else len(prompt)]
      chosen = texts[stable_index(stem + '\n'.join(texts), len(texts))]
      letter = next(k for k, v in options.items() if v == chosen)
      idx = list(responses).index(letter)
      return idx, responses[idx], {'policy': 'stable_hash'}
    idx = stable_index(prompt, len(responses))
    return idx, responses[idx], {'policy': 'stable_hash'}
