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

"""The public record of customs: every variant that has entered the world.

A custom is a cultural variant: a norm, a value, a belief, a skill, a habit or
a learning bias, with a domain (its locus) and a text (its thick content).
What a person holds is a component of that person's own entity, built from a
custom's spec; this record is the world's, kept by the World GM. Every custom
enters the world through the GM (a founder's, a mutation in transmission, an
innovation), so the GM knows each variant's content by its id without ever
reading anyone's mind: people name the ids they pass on.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import json
import re
from typing import Any

import numpy as np

# The loci of culture: recombination and capacity work per domain.
KINSHIP = 'kinship'
FOOD = 'food'
SHARING = 'sharing'
AUTHORITY = 'authority'
SACRED = 'sacred'
CRAFT = 'craft'
LEARNING = 'learning'
IDENTITY = 'identity'
LANGUAGE = 'language'
FARMING = 'farming'
REASONING = 'reasoning'  # How one decides: question customs.
# Order matters to recombination; a new domain goes last, so that worlds
# without its customs draw as before.
DOMAINS = (
    KINSHIP,
    FOOD,
    SHARING,
    AUTHORITY,
    SACRED,
    CRAFT,
    LEARNING,
    IDENTITY,
    LANGUAGE,
    FARMING,
    REASONING,
)

# The kinds of custom, each built as its own kind of component.
NORM = 'norm'
VALUE = 'value'
BELIEF = 'belief'
SKILL = 'skill'
HABIT = 'habit'
BIAS = 'bias'
DIALECT = 'dialect'
MARRIAGE = 'marriage'
# A step of decision logic: a question the person asks itself before it
# acts (DESIGN 3), built as the stock `QuestionOfRecentMemories`.
QUESTION = 'question'
KINDS = (NORM, VALUE, BELIEF, SKILL, HABIT, BIAS, DIALECT, MARRIAGE, QUESTION)

# What a marriage rule's `within` param asks of a partner: to speak as one
# does (understanding at least `SAME_TONGUE`), or to be of one's people.
TONGUE = 'tongue'
DESCENT = 'descent'
SAME_TONGUE = 0.75

# Kinds bound to a domain: a dialect is how one speaks, a marriage rule whom
# one marries, a question how one decides. A custom of another kind may be of
# any domain but reasoning.
KIND_DOMAINS: Mapping[str, tuple[str, ...]] = {
    DIALECT: (LANGUAGE,),
    MARRIAGE: (KINSHIP,),
    QUESTION: (REASONING,),
}


def domains_of(kind: str) -> tuple[str, ...]:
  """The domains a custom of a kind may be of."""
  return KIND_DOMAINS.get(kind, DOMAINS)


# A dialect is a list of words (its `words` param); a copying error changes
# one of them to one of these sounds.
SOUNDS = (
    'ka',
    'ri',
    'to',
    'ma',
    'el',
    'an',
    'so',
    'vi',
    'lu',
    'den',
    'ar',
    'ne',
    'bo',
    'tem',
    'sa',
    'yo',
    'mi',
    'ra',
    'gun',
    'li',
    'ta',
    'or',
    'wen',
    'ha',
)

# The channels a custom comes to a person by (an offer's `via`).
BIRTH = 'birth'
ARRIVAL = 'arrival'
UPBRINGING = 'upbringing'
OBSERVATION = 'observation'
APPRENTICESHIP = 'apprenticeship'
SANCTION = 'sanction'
PRACTICE = 'practice'
READING = 'reading'
SCHOOL = 'school'
RITE = 'rite'
STORY = 'story'
PREACHING = 'preaching'
MARKET = 'market'
RELEASE = 'release'

# The highest level a skill reaches.
MAX_LEVEL = 9

# The forms of a norm: known (expected by others, kept when watched) or
# internalized (one's own, kept when nobody watches). Founders' norms are
# internalized.
KNOWN = 'known'
INTERNALIZED = 'internalized'

# How the GM offers a custom in an observation, and how a Learner reads it:
# its spec, the channel it came by, and (seen in someone) who and how well fed.
# The line ends with the custom's full params as JSON, so a learner holds
# exactly what was offered (a marriage rule's terms, a dialect's words).
OFFER = re.compile(
    r'\[custom id=(?P<id>c\d+) kind=(?P<kind>\w+) domain=(?P<domain>\w+)'
    r'(?: level=(?P<level>\d+))?(?: form=(?P<form>\w+))? via=(?P<via>\w+)'
    r'(?: from=(?P<source>[^ \]]+) stores=(?P<stores>\d+(?:\.\d+)?)'
    r'(?: judged=(?P<judged>\d+))?)?\]'
    r' (?P<text>[^\n]*?)(?: \|params=(?P<params>\{[^\n]*\}))?$',
    re.MULTILINE,
)


def offer_line(
    spec: Mapping[str, Any],
    via: str,
    source: str = '',
    stores: float | None = None,
    judged: int = 0,
) -> str:
  """An observation line offering a custom (what a Learner reads).

  Args:
    spec: The custom's spec.
    via: The channel it comes by.
    source: Who is seen holding it, if anyone.
    stores: That person's food stores (for success-biased learners).
    judged: How often that person has judged at a court (its standing, for
      learners who defer to judges).

  Returns:
    The line.
  """
  params = spec.get('params', {})
  detail = ''
  if params.get('level') is not None:
    detail += f' level={params["level"]}'
  if spec['kind'] == NORM:
    detail += f' form={form_of(spec)}'
  seen = (
      f' from={source} stores={stores:.1f}'
      + (f' judged={judged}' if judged else '')
      if source and stores is not None
      else ''
  )
  full = f' |params={json.dumps(params, sort_keys=True)}' if params else ''
  return (
      f'[custom id={spec["id"]} kind={spec["kind"]} domain={spec["domain"]}'
      f'{detail} via={via}{seen}] {spec["text"]}{full}'
  )


def form_of(spec: Mapping[str, Any]) -> str:
  """A norm's form ('' for other kinds)."""
  if spec['kind'] != NORM:
    return ''
  return str(spec.get('params', {}).get('form', INTERNALIZED))


@dataclasses.dataclass(frozen=True)
class Custom:
  """One cultural variant, as the world records it."""

  id: str
  kind: str
  domain: str
  text: str
  # Kind-specific settings, JSON-safe (a skill's level, a habit's rule...).
  params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  # The variant it was copied from with an error, if any.
  parent: str = ''
  # How it entered the world: 'founder', 'mutation', 'innovation', ...
  origin: str = 'founder'
  year: int = 0
  # What the first of its line said, its stress aside ('' if this is the
  # first), and which of its words (by place) this variant has lost.
  root: str = ''
  lost: tuple[int, ...] = ()

  def spec(self) -> dict[str, Any]:
    """The JSON spec a person's component is built from."""
    return {
        'id': self.id,
        'kind': self.kind,
        'domain': self.domain,
        'text': self.text,
        'params': dict(self.params),
    }

  def to_dict(self) -> dict[str, Any]:
    data = {
        **self.spec(),
        'parent': self.parent,
        'origin': self.origin,
        'year': self.year,
    }
    if self.root:  # Only a drifted text's (saves before drift have none).
      data['root'] = self.root
      data['lost'] = list(self.lost)
    return data

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'Custom':
    return cls(
        id=str(data['id']),
        kind=str(data['kind']),
        domain=str(data['domain']),
        text=str(data['text']),
        params=dict(data.get('params', {})),
        parent=str(data.get('parent', '')),
        origin=str(data.get('origin', 'founder')),
        year=int(data.get('year', 0)),
        root=str(data.get('root', '')),
        lost=tuple(int(i) for i in data.get('lost', ())),
    )


# How a copying error stresses a text, and how far it may wear one down:
# at most `WORDS_LOST` of its line's first words, never below `MIN_WORDS`.
STRESS = ', always.'
WORDS_LOST = 2
MIN_WORDS = 4
# The stresses old copying errors heaped at a text's end.
# Each is ", always"; a word lost before them may have left the first bare
# ("In lean always, always."). A text that only ends in "always" ("Keep the
# fire burning always.") says it.
_HEAPED_STRESS = re.compile(
    r'(?:\s+always\b(?=,+\s*always\b))?(?:,+\s*always\b)+[\s,.]*$',
    re.IGNORECASE,
)


def _unstressed(text: str) -> str:
  """A text without its stresses: the first of its line, for one that has none.

  Before texts drifted about their line's first, copying errors stressed a
  text again and again at its end ("In lean always, always, always."); the
  first error copying such a text since takes it back to its words. An
  "always" the text says elsewhere ("Always give the first fish to the
  river.") is its own.

  Args:
    text: A variant's text.

  Returns:
    The text without the stresses at its end.
  """
  words = _HEAPED_STRESS.sub('', text).strip(' ,.')
  return (words or text.rstrip('.')) + '.'


class Culture:
  """The world's record of customs, by id."""

  def __init__(self):
    self._customs: dict[str, Custom] = {}
    # The id of each content (kind, domain, text, params): a variant arising
    # again, by convergent copying errors or a change of form, is the same.
    self._ids: dict[str, str] = {}
    self._next = 1

  def __contains__(self, custom_id: str) -> bool:
    return custom_id in self._customs

  def __len__(self) -> int:
    return len(self._customs)

  def created(self) -> int:
    """How many variants have ever entered the world."""
    return self._next - 1

  def get(self, custom_id: str) -> Custom:
    return self._customs[custom_id]

  def add(
      self,
      kind: str,
      domain: str,
      text: str,
      params: Mapping[str, Any] | None = None,
      *,
      parent: str = '',
      origin: str = 'founder',
      year: int = 0,
      root: str = '',
      lost: Sequence[int] = (),
  ) -> Custom:
    """Records a variant and returns it (the recorded one, if it exists).

    Args:
      kind: One of `KINDS`.
      domain: One of `DOMAINS`.
      text: What the custom says.
      params: Kind-specific settings (a skill's level, a norm's form).
      parent: The variant it was copied from, if any.
      origin: How it entered the world.
      year: When.
      root: What the first of its line said (`Custom`).
      lost: The words of `root` it has lost, by place.

    Returns:
      The variant with this content.

    Raises:
      ValueError: For an unknown kind or domain.
    """
    if kind not in KINDS:
      raise ValueError(f'Unknown kind of custom: {kind}')
    if domain not in DOMAINS:
      raise ValueError(f'Unknown domain: {domain}')
    text = ' '.join(text.split())
    params = dict(params or {})
    if kind == NORM:
      params.setdefault('form', INTERNALIZED)
    content = _content(kind, domain, text, params)
    if content in self._ids:
      return self._customs[self._ids[content]]
    custom = Custom(
        id=f'c{self._next}',
        kind=kind,
        domain=domain,
        text=text,
        params=params,
        parent=parent,
        origin=origin,
        year=year,
        root=root,
        lost=tuple(sorted(lost)),
    )
    self._next += 1
    self._customs[custom.id] = custom
    self._ids[content] = custom.id
    return custom

  def adopt(self, spec: Mapping[str, Any]) -> Custom:
    """Records a variant under the id it already has (a founder's).

    Args:
      spec: The custom's spec, with its `id`.

    Returns:
      The recorded variant.
    """
    custom_id = str(spec['id'])
    if custom_id in self._customs:
      return self._customs[custom_id]
    params = dict(spec.get('params', {}))
    if spec['kind'] == NORM:
      params.setdefault('form', INTERNALIZED)
    custom = Custom(
        id=custom_id,
        kind=str(spec['kind']),
        domain=str(spec['domain']),
        text=' '.join(str(spec['text']).split()),
        params=params,
    )
    self._customs[custom_id] = custom
    self._ids[_content(custom.kind, custom.domain, custom.text, params)] = (
        custom_id
    )
    self._next = max(self._next, int(custom_id.removeprefix('c')) + 1)
    return custom

  def in_form(self, custom_id: str, form: str, year: int) -> Custom:
    """The variant of a norm in `form` (any other custom, unchanged).

    Args:
      custom_id: The variant.
      form: `KNOWN` or `INTERNALIZED`.
      year: Now (if the form is new to the world).

    Returns:
      The norm with the same content in that form.
    """
    custom = self.get(custom_id)
    if custom.kind != NORM or form_of(custom.spec()) == form:
      return custom
    return self.add(
        custom.kind,
        custom.domain,
        custom.text,
        {**custom.params, 'form': form},
        parent=custom.id,
        origin=form,
        year=year,
        root=custom.root,  # A change of form keeps the line's first text.
        lost=custom.lost,
    )

  def copy(
      self,
      custom_id: str,
      rng: np.random.Generator,
      mutation: float,
      year: int,
      improve: float = 0.0,
  ) -> Custom:
    """A copy of a variant, as transmission makes it.

    A skill copied with care (`improve`) may come out a level higher: the
    learner adds to what the teacher knew, a new variant. Otherwise, with
    probability `mutation`, the copy errs: a skill is learned a level lower
    (copying loses know-how, it never adds any), a text loses or doubles its
    emphasis: one word is lost or regained, or its one stress gained or lost,
    so a text drifts about its line's first and never wears away (at most
    `WORDS_LOST` of its words lost). An erring copy is a new variant (its
    parent recorded); a faithful one is the same variant.

    Args:
      custom_id: The variant copied.
      rng: The world's random stream for this transmission.
      mutation: The chance of a copying error.
      year: Now.
      improve: The chance that a skill's copy is a level better.

    Returns:
      The variant the receiver gets.
    """
    custom = self.get(custom_id)
    level = int(custom.params.get('level', 1))
    if custom.kind == SKILL and level < MAX_LEVEL and rng.random() < improve:
      return self.add(
          custom.kind,
          custom.domain,
          custom.text,
          {**custom.params, 'level': level + 1},
          parent=custom.id,
          origin='improvement',
          year=year,
      )
    if rng.random() >= mutation:
      return custom
    params = dict(custom.params)
    text = custom.text
    if custom.kind == SKILL:
      params['level'] = max(1, level - 1)
    elif custom.kind == DIALECT:
      words = list(params.get('words', ()))
      if not words:
        return custom
      words[int(rng.integers(len(words)))] = SOUNDS[
          int(rng.integers(len(SOUNDS)))
      ]
      params['words'] = words
      text = f'{text.partition(":")[0]}: {" ".join(words)}.'
    else:
      # A word is lost or regained, or the stress is: the text drifts about
      # its line's first (at most `WORDS_LOST` of its words lost, and one
      # stress at most), never away from it.
      stressed = text.endswith(STRESS)
      root = custom.root or _unstressed(text)
      first = root.rstrip('.').split()
      lost = set(custom.lost)
      errors = ['stress']
      if lost:
        errors.append('regain')
      if len(lost) < WORDS_LOST and len(first) - len(lost) > MIN_WORDS:
        errors.append('lose')
      error = errors[int(rng.integers(len(errors)))]
      if error == 'lose':
        kept = [i for i in range(1, len(first)) if i not in lost]
        lost.add(kept[int(rng.integers(len(kept)))])
      elif error == 'regain':
        lost.discard(sorted(lost)[int(rng.integers(len(lost)))])
      else:
        stressed = not stressed
      words = [word for i, word in enumerate(first) if i not in lost]
      text = ' '.join(words) + (STRESS if stressed else '.')
      return self.add(
          custom.kind,
          custom.domain,
          text,
          params,
          parent=custom.id,
          origin='mutation',
          year=year,
          root=root,
          lost=sorted(lost),
      )
    return self.add(
        custom.kind,
        custom.domain,
        text,
        params,
        parent=custom.id,
        origin='mutation',
        year=year,
    )

  def recombine(
      self,
      mother: Sequence[str],
      father: Sequence[str],
      rng: np.random.Generator,
  ) -> list[str]:
    """A child's variants from two parents', domain by domain.

    For each domain either parent holds customs in, the child takes that
    domain's customs from one parent, chosen at random (from the only parent
    who has any, if just one does).

    Args:
      mother: The ids one parent passes on.
      father: The ids the other passes on.
      rng: The world's random stream for this birth.

    Returns:
      The ids the child starts with (before copying errors).
    """
    by_parent = []
    for ids in (mother, father):
      domains: dict[str, list[str]] = {}
      for custom_id in ids:
        if custom_id in self._customs:
          domains.setdefault(self.get(custom_id).domain, []).append(custom_id)
      by_parent.append(domains)
    child = []
    for domain in DOMAINS:
      offers = [d[domain] for d in by_parent if d.get(domain)]
      if offers:
        child.extend(offers[int(rng.integers(len(offers)))])
    return child

  def prune(self, keep: Collection[str]) -> list[Custom]:
    """Forgets the variants not in `keep`; returns them, oldest first.

    The world keeps only variants someone living may still hold or pass on;
    the forgotten ones go to the run's long record (the phylogeny on disk).

    Args:
      keep: The ids still in use.

    Returns:
      The forgotten variants.
    """
    gone = [c for i, c in self._customs.items() if i not in keep]
    for custom in gone:
      del self._customs[custom.id]
      del self._ids[
          _content(custom.kind, custom.domain, custom.text, custom.params)
      ]
    return gone

  def get_state(self) -> dict[str, Any]:
    return {
        'next': self._next,
        'customs': [c.to_dict() for c in self._customs.values()],
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    self._next = int(state['next'])
    self._customs = {
        str(c['id']): Custom.from_dict(c) for c in state.get('customs', [])
    }
    self._ids = {
        _content(c.kind, c.domain, c.text, c.params): i
        for i, c in self._customs.items()
    }


def _content(
    kind: str, domain: str, text: str, params: Mapping[str, Any]
) -> str:
  """A key of a variant's content."""
  return json.dumps([kind, domain, text, params], sort_keys=True)


def intelligibility(a: Custom, b: Custom) -> float:
  """How well speakers of two dialects understand each other, 0 to 1.

  Args:
    a: A dialect.
    b: Another.

  Returns:
    The share of their words that are the same.
  """
  words_a = list(a.params.get('words', ()))
  words_b = list(b.params.get('words', ()))
  if not words_a or not words_b:
    return 1.0
  same = sum(1 for x, y in zip(words_a, words_b) if x == y)
  return same / max(len(words_a), len(words_b))


def accepts(
    rule: Mapping[str, Any] | None, understanding: float, same_descent: bool
) -> bool:
  """Whether a marriage rule allows a partner (no rule allows anyone).

  Args:
    rule: The rule's params (`within`: `TONGUE` or `DESCENT`), or None.
    understanding: How well the two understand each other's speech.
    same_descent: Whether the two are of the same people by descent.

  Returns:
    Whether the rule allows the match.
  """
  within = (rule or {}).get('within')
  if within == TONGUE:
    return understanding >= SAME_TONGUE
  if within == DESCENT:
    return same_descent
  return True
