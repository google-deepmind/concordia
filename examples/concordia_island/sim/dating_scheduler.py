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

"""Dating Scheduler for Concordia Island.

Generates spouse meetups and rotating first dates based on persona metadata.
"""

import logging
import random
from typing import Any

from examples.concordia_island.sim import social_scheduler as social_scheduler_lib


def generate_dating_schedule(
    active_names: set[str],
    unpaired_names: set[str],
    personas: dict[str, Any],
    num_days: int,
    venues: list[str],
) -> list[social_scheduler_lib.SocialEvent]:
  """Generates spouse meetups and rotating first dates.

  Args:
    active_names: Names of all active agents in the simulation.
    unpaired_names: Names of agents not yet paired by earlier stages.
    personas: Dictionary mapping agent names to their persona metadata objects.
    num_days: Number of days to schedule dates for.
    venues: List of available venues for dates.

  Returns:
    A list of SocialEvent objects.
  """
  social_events = []
  unpaired = unpaired_names

  if not unpaired or not personas:
    return social_events

  logging.info(
      "DATES Stage 3: Smart dating scheduler for %d unpaired agents",
      len(unpaired),
  )

  # --- Classify agents by relationship status + orientation ---
  married_men_het = []
  married_women_het = []
  married_men_homo = []
  married_women_homo = []
  married_men_bi = []
  married_women_bi = []
  single_men_het = []
  single_women_het = []
  single_men_homo = []
  single_women_homo = []
  single_men_bi = []
  single_women_bi = []
  unclassified = []

  for name in unpaired:
    pdata = personas.get(name)
    if not pdata:
      unclassified.append(name)
      continue

    gender = getattr(pdata, "gender", "").lower()
    rel_status = getattr(pdata, "relationship_status", "").lower()
    orientation = getattr(pdata, "sexual_orientation", "").lower()

    is_woman = "woman" in gender or "female" in gender
    is_man = not is_woman and ("man" in gender or "male" in gender)
    is_married = "married" in rel_status
    is_homo = "homo" in orientation
    is_bi = "bi" in orientation

    if not is_man and not is_woman:
      unclassified.append(name)
      continue

    if is_married:
      if is_man:
        if is_homo:
          married_men_homo.append(name)
        elif is_bi:
          married_men_bi.append(name)
        else:
          married_men_het.append(name)
      else:
        if is_homo:
          married_women_homo.append(name)
        elif is_bi:
          married_women_bi.append(name)
        else:
          married_women_het.append(name)
    else:  # single
      if is_man:
        if is_homo:
          single_men_homo.append(name)
        elif is_bi:
          single_men_bi.append(name)
        else:
          single_men_het.append(name)
      else:
        if is_homo:
          single_women_homo.append(name)
        elif is_bi:
          single_women_bi.append(name)
        else:
          single_women_het.append(name)

  logging.info(
      "DATES Stage 3: Classification — "
      "Married: %d het-M, %d het-W, %d homo-M, %d homo-W, "
      "%d bi-M, %d bi-W | "
      "Single: %d het-M, %d het-W, %d homo-M, %d homo-W, "
      "%d bi-M, %d bi-W | Unclassified: %d",
      len(married_men_het),
      len(married_women_het),
      len(married_men_homo),
      len(married_women_homo),
      len(married_men_bi),
      len(married_women_bi),
      len(single_men_het),
      len(single_women_het),
      len(single_men_homo),
      len(single_women_homo),
      len(single_men_bi),
      len(single_women_bi),
      len(unclassified),
  )

  # --- 3a: Assign spouses (pair compatible married agents) ---
  spouse_pairs = []
  rng = random.Random(42 + len(active_names))

  def _pair_two_pools(pool_a, pool_b):
    """Pair agents from two compatible pools, return pairs + leftovers."""
    rng.shuffle(pool_a)
    rng.shuffle(pool_b)
    pairs = []
    n = min(len(pool_a), len(pool_b))
    for i in range(n):
      pairs.append((pool_a[i], pool_b[i]))
    leftovers = pool_a[n:] + pool_b[n:]
    return pairs, leftovers

  def _pair_same_pool(pool):
    """Pair agents within the same pool (e.g., gay men), return pairs."""
    rng.shuffle(pool)
    pairs = []
    for i in range(0, len(pool) - 1, 2):
      pairs.append((pool[i], pool[i + 1]))
    leftovers = [pool[-1]] if len(pool) % 2 else []
    return pairs, leftovers

  # Het married: men ↔ women
  pairs, leftover_het = _pair_two_pools(married_men_het, married_women_het)
  spouse_pairs.extend(pairs)

  # Bi married men can fill in as het spouses for leftover women
  if leftover_het:
    has_leftover_women = any(
        "woman" in getattr(personas.get(n), "gender", "").lower()
        for n in leftover_het
    )
    if has_leftover_women:
      bi_fill = married_men_bi
    else:
      bi_fill = married_women_bi
    pairs, leftover_het = _pair_two_pools(leftover_het, bi_fill)
    spouse_pairs.extend(pairs)

  # Homo married: men ↔ men, women ↔ women
  pairs, leftover_homo_m = _pair_same_pool(married_men_homo)
  spouse_pairs.extend(pairs)
  pairs, leftover_homo_w = _pair_same_pool(married_women_homo)
  spouse_pairs.extend(pairs)

  # Bi married: pair remaining bi with each other
  remaining_bi = married_men_bi + married_women_bi
  pairs, leftover_bi = _pair_same_pool(remaining_bi)
  spouse_pairs.extend(pairs)

  # All leftover married agents become singles for dating purposes
  leftover_married = (
      leftover_het + leftover_homo_m + leftover_homo_w + leftover_bi
  )
  logging.info(
      "DATES Stage 3a: %d spouse pairs assigned, %d married leftover "
      "→ added to singles pool",
      len(spouse_pairs),
      len(leftover_married),
  )

  # Generate spouse meetup events (one per couple per day)
  for day in range(1, num_days + 1):
    for p1, p2 in spouse_pairs:
      social_events.append(
          social_scheduler_lib.SocialEvent(
              participants=(p1, p2),
              venue=rng.choice(venues),
              theme="spouse_meetup",
              day=day,
              tick_hour=19,
              prompt_type="spouse_meetup",
          )
      )

  # --- 3b: Singles rotation dating ---
  men_seeking_women = single_men_het + single_men_bi
  women_seeking_men = single_women_het + single_women_bi
  men_seeking_men = single_men_homo + single_men_bi
  women_seeking_women = single_women_homo + single_women_bi

  # Add leftover married to appropriate pools based on their orientation
  for name in leftover_married:
    pdata = personas.get(name)
    if not pdata:
      continue
    gender = getattr(pdata, "gender", "").lower()
    orient = getattr(pdata, "sexual_orientation", "").lower()
    is_woman = "woman" in gender or "female" in gender
    is_man = not is_woman and ("man" in gender or "male" in gender)
    if is_man:
      if "homo" in orient:
        men_seeking_men.append(name)
      else:
        men_seeking_women.append(name)
    else:
      if "homo" in orient:
        women_seeking_women.append(name)
      else:
        women_seeking_men.append(name)

  # Deduplicate pools (bi agents may already be listed)
  men_seeking_women = list(dict.fromkeys(men_seeking_women))
  women_seeking_men = list(dict.fromkeys(women_seeking_men))
  men_seeking_men = list(dict.fromkeys(men_seeking_men))
  women_seeking_women = list(dict.fromkeys(women_seeking_women))

  logging.info(
      "DATES Stage 3b: Dating pools — "
      "M→W: %d, W→M: %d, M→M: %d, W→W: %d, unclassified: %d",
      len(men_seeking_women),
      len(women_seeking_men),
      len(men_seeking_men),
      len(women_seeking_women),
      len(unclassified),
  )

  # Generate rotation dates with no-repeat constraint
  min_unique_days = min(5, num_days)
  seen_pairs = set()
  singles_event_count = 0

  def _generate_rotation_dates(pool_a, pool_b, same_pool):
    nonlocal singles_event_count
    if same_pool:
      pool = list(pool_a)
      for day in range(1, num_days + 1):
        max_attempts = 200
        for _ in range(max_attempts):
          rng.shuffle(pool)
          candidate_pairs = []
          for i in range(0, len(pool) - 1, 2):
            candidate_pairs.append(tuple(sorted((pool[i], pool[i + 1]))))
          if day <= min_unique_days:
            has_collision = any(p in seen_pairs for p in candidate_pairs)
            if has_collision:
              continue
          # Accept this day's pairings
          for p1, p2 in candidate_pairs:
            seen_pairs.add((p1, p2))
            social_events.append(
                social_scheduler_lib.SocialEvent(
                    participants=(p1, p2),
                    venue=rng.choice(venues),
                    theme="first_date",
                    day=day,
                    tick_hour=19,
                    prompt_type="first_date",
                )
            )
            singles_event_count += 1
          break
        else:
          # Fallback: allow repeats
          rng.shuffle(pool)
          for i in range(0, len(pool) - 1, 2):
            pair = tuple(sorted((pool[i], pool[i + 1])))
            seen_pairs.add(pair)
            prompt = "first_date" if pair not in seen_pairs else "second_date"
            social_events.append(
                social_scheduler_lib.SocialEvent(
                    participants=(pool[i], pool[i + 1]),
                    venue=rng.choice(venues),
                    theme="first_date",
                    day=day,
                    tick_hour=19,
                    prompt_type=prompt,
                )
            )
            singles_event_count += 1
    else:
      n_pairs = min(len(pool_a), len(pool_b))
      if n_pairs == 0:
        return
      a_list = list(pool_a[:n_pairs])
      b_list = list(pool_b[:n_pairs])

      for day in range(1, num_days + 1):
        max_attempts = 200
        for _ in range(max_attempts):
          rng.shuffle(b_list)
          candidate_pairs = [
              tuple(sorted((a_list[i], b_list[i]))) for i in range(n_pairs)
          ]
          if day <= min_unique_days:
            has_collision = any(p in seen_pairs for p in candidate_pairs)
            if has_collision:
              continue
          # Accept
          for i in range(n_pairs):
            pair = candidate_pairs[i]
            is_repeat = pair in seen_pairs
            seen_pairs.add(pair)
            social_events.append(
                social_scheduler_lib.SocialEvent(
                    participants=(a_list[i], b_list[i]),
                    venue=rng.choice(venues),
                    theme="first_date",
                    day=day,
                    tick_hour=19,
                    prompt_type="second_date" if is_repeat else "first_date",
                )
            )
            singles_event_count += 1
          break
        else:
          # Fallback: allow repeats for this day
          rng.shuffle(b_list)
          for i in range(n_pairs):
            pair = tuple(sorted((a_list[i], b_list[i])))
            is_repeat = pair in seen_pairs
            seen_pairs.add(pair)
            social_events.append(
                social_scheduler_lib.SocialEvent(
                    participants=(a_list[i], b_list[i]),
                    venue=rng.choice(venues),
                    theme="first_date",
                    day=day,
                    tick_hour=19,
                    prompt_type="second_date" if is_repeat else "first_date",
                )
            )
            singles_event_count += 1

  # Generate dates for each dating pool
  _generate_rotation_dates(men_seeking_women, women_seeking_men, False)
  _generate_rotation_dates(men_seeking_men, men_seeking_men, True)
  _generate_rotation_dates(women_seeking_women, women_seeking_women, True)

  # Unclassified agents get random pairing as fallback
  if unclassified:
    _generate_rotation_dates(unclassified, unclassified, True)

  logging.info(
      "DATES Stage 3b result: %d singles dating events across %d days "
      "(min %d unique partner days)",
      singles_event_count,
      num_days,
      min_unique_days,
  )

  # --- Stage 4 (NUCLEAR FALLBACK): Random pairing for remaining agents ---
  paired_agents = set()
  for e in social_events:
    paired_agents.update(e.participants)
  still_unpaired = active_names - paired_agents

  if still_unpaired:
    logging.warning(
        "DATES Stage 4 (FALLBACK): %d agents still unpaired after stages"
        " 1-3. Generating random pairings.",
        len(still_unpaired),
    )
    agent_list = sorted(still_unpaired)

    stage4_count = 0
    for day in range(1, num_days + 1):
      rng = random.Random(42 + len(agent_list) + day)
      shuffled = list(agent_list)
      rng.shuffle(shuffled)
      for i in range(0, len(shuffled) - 1, 2):
        social_events.append(
            social_scheduler_lib.SocialEvent(
                participants=(shuffled[i], shuffled[i + 1]),
                venue=rng.choice(venues),
                theme="first_date",
                day=day,
                tick_hour=19,
                prompt_type="first_date",
            )
        )
        stage4_count += 1
    logging.info(
        "DATES Stage 4 result: %d new events from random pairing (%d total)",
        stage4_count,
        len(social_events),
    )

  return social_events
