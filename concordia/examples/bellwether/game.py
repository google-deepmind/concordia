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

"""Bellwether's fictional rules, not a replacement simulation engine.

Sequential selects actors and invokes the GM. This component resolves only the
documented actions. Dialogue is never parsed as proof of resources or consent.
Inventory.apply owns all material balances; completed consumption is retained.
"""

import copy
import datetime
import json
import re
import threading
from typing import Any

from concordia.components.game_master import event_resolution
from concordia.components.game_master import inventory
from concordia.examples.bellwether import scenario
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import operation_service as ops

PLAYER = scenario.PLAYER
NAMES = [PLAYER, *scenario.RESIDENTS]
FACILITIES = ('beacon', 'shelter', 'cold store')
WATCHES = ('Dusk', 'High Tide', 'Before Dawn')
WORLD_KEY = 'StormNight'
GM = 'Bellwether'
OPENING = (
    'Dusk. Rain blows sideways over Bellwether. Six fuel units remain in the'
    ' generator; Nell holds two reserve units and the one spare part. The'
    ' beacon, shelter and cold store each need one unit per watch: nine across'
    ' the night. Ivo can remove the final beacon demand if he accepts AND'
    ' performs a repair with the spare part before Before Dawn. Four decisions'
    ' remain this watch. You may ask, bargain, allocate, promise, transfer with'
    ' permission, order accepted work, or wait. Looking around costs nothing.'
    ' Consent is not automatic.'
)
CONSEQUENCES = {
    'beacon': 'Boats lose a safe harbor signal; arrivals must hold offshore.',
    'shelter': 'Shelter heating fails; displaced residents face a cold watch.',
    'cold store': (
        'The cooperative cold chain is interrupted; stock is at risk.'
    ),
}
DEFAULT_DISPUTE = {
    'text': (
        'At High Tide, Mara says the cooperative failed the harbor last storm. '
        'Nell disputes this: the spare boat was carrying shelter residents. '
        'Neither account is established as objective fact.'
    ),
    'recipients': NAMES,
}
INSTITUTIONS = [
    {
        'name': 'Harbor Office',
        'members': ['Mara'],
        'rule': (
            'Protect safe boat passage; advise the coordinator on beacon use.'
        ),
        'enforcement': (
            'Advice and objections, not seizure of cooperative property.'
        ),
    },
    {
        'name': 'Workers’ Cooperative',
        'members': ['Nell', 'Ivo', 'Sam'],
        'rule': (
            'Nell is delegated custody of reserve and parts; Ivo owns his'
            ' labor. Members may dispute or organize to revise these'
            ' arrangements.'
        ),
        'enforcement': (
            'Owner consent gates transfers and work; membership is not consent.'
        ),
    },
]


class ExplicitInventory(inventory.Inventory):
  """Use standard Inventory storage/validation without LLM-implied transfers."""

  def pre_act(self, action_spec):
    del action_spec
    return ''


def new_inventory():
  return ExplicitInventory(
      no_language_model.NoLanguageModel(),
      [
          inventory.ItemTypeConfig(
              'fuel', minimum=0, maximum=8, force_integer=True
          ),
          inventory.ItemTypeConfig(
              'part', minimum=0, maximum=1, force_integer=True
          ),
      ],
      {
          'Generator': {'fuel': 6, 'part': 0},
          'Nell': {'fuel': 2, 'part': 1},
          'Ivo': {'fuel': 0, 'part': 0},
          'Used': {'fuel': 0, 'part': 0},
      },
      lambda: datetime.datetime(2000, 1, 1),
  )


def parse_action(text):
  """Conservative ordinary-language vocabulary, with no speculative effects.

  Ambiguous/unsupported attempts return an actionable clarification, without a
  turn cost. Open-ended proposals/messages remain verbatim rather than being
  reduced to an invented mechanical action.
  """
  text = text.strip()
  lower = text.casefold().rstrip('.!')
  if lower in ('wait', 'wait here', 'continue', 'hold'):
    return {'kind': 'wait', 'text': text}
  patterns = [
      (
          (
              r'(?:ask|request) nell (?:for |to release )?(?:the )?(?:two |2'
              r' )?(?:reserve )?fuel(?: and (?:the |a )?(?:spare )?part)?'
          ),
          'request',
          'Nell',
      ),
      (
          (
              r'(?:ask|request) ivo (?:to |for (?:a )?)?(?:repair|fix)(?: the'
              r' beacon)?'
          ),
          'request',
          'Ivo',
      ),
      (
          (
              r'(?:transfer|collect|move) (?:nell.s )?(?:the'
              r' )?(?:reserve|fuel)(?: and (?:the )?(?:spare )?part)?'
          ),
          'transfer',
          'Nell',
      ),
      (
          r'(?:order|perform|start) (?:ivo.s |ivo to )?(?:the )?repair',
          'work',
          'Ivo',
      ),
  ]
  for pattern, kind, target in patterns:
    if re.fullmatch(pattern, lower):
      return {'kind': kind, 'target': target, 'text': text}
  for prefix in ('allocate ', 'prioritize ', 'supply '):
    if lower.startswith(prefix):
      value = lower[len(prefix) :]
      if value in ('all', 'all facilities', 'everything'):
        facilities = list(FACILITIES)
      elif value in ('none', 'nothing'):
        facilities = []
      else:
        value = value.replace('the ', '').replace(' and ', ',')
        facilities = [x.strip() for x in value.split(',') if x.strip()]
        if (
            not facilities
            or len(set(facilities)) != len(facilities)
            or any(x not in FACILITIES for x in facilities)
        ):
          break
      return {'kind': 'allocate', 'facilities': facilities, 'text': text}
  match = re.fullmatch(
      r'(?:tell|ask|discuss with|propose to|message)'
      r' (everyone|Mara|Nell|Ivo|Sam):\s*(.+)',
      text,
      re.I | re.S,
  )
  if match:
    target, words = match.groups()
    target = 'everyone' if target.lower() == 'everyone' else target.title()
    return {
        'kind': 'talk',
        'target': target,
        'text': words.strip(),
        'private': lower.startswith('message '),
    }
  match = re.fullmatch(
      r'(?:promise|commit to) (beacon|shelter|cold store)', lower
  )
  if match:
    return {'kind': 'promise', 'facility': match[1], 'text': text}
  if lower in ('withdraw my promise', 'revoke my promise'):
    return {'kind': 'revoke', 'text': text}
  raise ops.OperationError(
      'clarification_needed',
      'Nothing changed. Specify an action: ask Nell for fuel and part; ask Ivo'
      ' to repair; transfer reserve and part; order repair; allocate shelter'
      ' and cold store; promise shelter; message Nell: your words; propose to'
      ' everyone: your idea; or wait. Unsupported physical ideas can be'
      ' discussed, not silently performed.',
  )


RESIDENT_DECISIONS = (
    'speak',
    'accept',
    'decline',
    'counter',
    'revoke',
    'perform',
)


def resident_response_schema():
  """Provider shape constraint; existing resolution still validates effects."""
  return {
      'type': 'object',
      'properties': {
          'decision': {'type': 'string', 'enum': list(RESIDENT_DECISIONS)},
          'speech': {'type': 'string'},
      },
      'required': ['decision', 'speech'],
      'additionalProperties': False,
  }


def parse_resident_response(text):
  """Validate the shared decision/speech contract before a human turn wakes."""
  start = text.find('{')
  response, _ = json.JSONDecoder().raw_decode(text[start:])
  if set(response) != {'decision', 'speech'} or not all(
      isinstance(x, str) for x in response.values()
  ):
    raise ValueError('Expected decision and speech')
  decision, speech = response['decision'].lower(), response['speech']
  if decision not in RESIDENT_DECISIONS:
    raise ValueError('Unknown decision')
  return decision, speech


class StormNight(
    entity_component.ContextComponent, entity_component.ComponentWithLogging
):
  """Resolve the scenario’s consent, agenda, watches and grounded records."""

  def __init__(
      self, stock, observations, *, dispute=None, institutions=None, lock=None
  ):
    super().__init__()
    self.stock = stock
    self.observations = observations
    self.lock = lock or threading.RLock()
    self.dispute = copy.deepcopy(
        DEFAULT_DISPUTE if dispute is None else dispute
    )
    if (
        not isinstance(self.dispute, dict)
        or set(self.dispute) != {'text', 'recipients'}
        or not isinstance(self.dispute['text'], str)
        or not isinstance(self.dispute['recipients'], list)
        or any(x not in NAMES for x in self.dispute['recipients'])
    ):
      raise ValueError('dispute requires text and valid recipient names')
    self.institutions = copy.deepcopy(
        INSTITUTIONS if institutions is None else institutions
    )
    self._validate_institutions(self.institutions)
    self.data: dict[str, Any] = {
        'watch': 0,
        'actions': 0,
        'agenda': [],
        'pending': None,
        'allocations': list(FACILITIES),
        'commitments': [],
        'services': [],
        'repair': False,
        'events': [],
        'dialogue': [],
        'relationships': [],
        'epilogue': None,
        'dawn_responses': {},
        'knowledge': {
            name: [x['name'] for x in self.institutions] for name in NAMES
        },
    }
    self._putative = ''

  def get_state(self):
    with self.lock:
      return copy.deepcopy({
          'night': self.data,
          'putative': self._putative,
          'dispute': self.dispute,
          'institutions': self.institutions,
      })

  def set_state(self, state):
    # Component serialization is not a full engine continuation contract.
    institutions = copy.deepcopy(state.get('institutions', self.institutions))
    self._validate_institutions(institutions)
    night = copy.deepcopy(state['night'])
    putative = state['putative']
    dispute = copy.deepcopy(state['dispute'])
    with self.lock:
      self.institutions = institutions
      self.data = night
      self._putative = putative
      self.dispute = dispute

  @staticmethod
  def _validate_institutions(institutions):
    if not isinstance(institutions, list):
      raise ValueError('institutions must be a list')
    names = set()
    for item in institutions:
      if (
          not isinstance(item, dict)
          or set(item) != {'name', 'members', 'rule', 'enforcement'}
          or any(
              not isinstance(item[key], str)
              for key in ('name', 'rule', 'enforcement')
          )
          or not item['name'].strip()
          or item['name'] in names
          or not isinstance(item['members'], list)
          or any(name not in NAMES for name in item['members'])
      ):
        raise ValueError(
            'institutions require unique names, known members and text'
            ' rules/enforcement'
        )
      names.add(item['name'])

  def seed(self, *, opening=OPENING, accounts=None):
    accounts = (
        {name: values[1] for name, values in scenario.RESIDENTS.items()}
        if accounts is None
        else copy.deepcopy(accounts)
    )
    if (
        not isinstance(opening, str)
        or not isinstance(accounts, dict)
        or set(accounts) != set(scenario.RESIDENTS)
        or any(not isinstance(text, str) for text in accounts.values())
    ):
      raise ValueError(
          'seed requires opening text and one text account per resident'
      )
    self.emit('opening', opening)
    for name, account in accounts.items():
      self.emit('private_memory', account, [name])

  def emit(self, kind, text, audience=None, **details):
    recipients = list(NAMES if audience is None else dict.fromkeys(audience))
    record = {
        'id': len(self.data['events']) + 1,
        'watch': self.data['watch'],
        'kind': kind,
        'text': text,
        'recipients': recipients,
        **details,
    }
    self.data['events'].append(record)
    for name in recipients:
      self.observations.add(name, text, NAMES)
    return record

  def inventory_state(self):
    return {
        name: self.stock.get_player_inventory(name)
        for name in ('Generator', 'Nell', 'Ivo', 'Used')
    }

  def invariant(self):
    accounts = self.inventory_state()
    assert sum(x['fuel'] for x in accounts.values()) == 8
    assert sum(x['part'] for x in accounts.values()) == 1
    assert all(x[k] >= 0 for x in accounts.values() for k in ('fuel', 'part'))

  def transfer(self, source, destination, item, amount):
    def change(accounts):
      accounts[source][item] -= amount
      accounts[destination][item] += amount
      return accounts

    self.stock.apply(change)
    self.invariant()

  @property
  def next_actor(self):
    return self.data['agenda'][0]['name'] if self.data['agenda'] else PLAYER

  @property
  def finished(self):
    return self.data['watch'] == 3 and not self.data['agenda']

  def task(self):
    return copy.deepcopy(self.data['agenda'][0]) if self.data['agenda'] else {}

  def action_prompt(self):
    if self.next_actor == PLAYER:
      return (
          'What will you do, {name}? Four consequential choices per watch;'
          ' inspection is free.'
      )
    task = self.task()
    return (
        'You are {name}. Respond to this delivered task, not to other people’s '
        'private information: '
        + json.dumps(task, ensure_ascii=False)
        + '\nReturn exactly one JSON object, with decision and speech fields.'
        ' decision must be speak, accept, decline, counter, revoke, or'
        ' perform. A request acceptance records your OWN commitment only.'
        ' perform executes your previously accepted repair if materials'
        ' exist. You may decline or counter; never speak for somebody else.'
        ' In ordinary discussion or dawn use speak. speech: at most two'
        ' natural-language sentences in your voice. You may disagree,'
        ' negotiate and change your mind. No narration of unrecorded'
        ' transfers or labor. Do not include private thoughts.'
    )

  def pre_observe(self, observation):
    if observation.startswith(event_resolution.PUTATIVE_EVENT_TAG):
      self._putative = observation[
          len(event_resolution.PUTATIVE_EVENT_TAG) :
      ].strip()
    return ''

  def pre_act(self, action_spec):
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    with self.lock:
      actor, separator, text = self._putative.partition(': ')
      if not separator or actor != self.next_actor:
        raise ValueError(
            'Resolution must match the standard engine’s selected actor'
        )
      result = self.resolve(actor, text)
      self._logging_channel({
          'Key': WORLD_KEY,
          'Value': result,
          'State': self.get_state(),
          'Inventory': self.inventory_state(),
      })
      return result

  def resolve(self, actor, text):
    """Apply one engine-provided attempt; also exercisable in rule tests."""
    with self.lock:
      if actor != self.next_actor or self.finished:
        raise ValueError('Not the current actor')
      before = len(self.data['events'])
      if actor == PLAYER:
        intent = parse_action(text)
        self._human(intent)
      else:
        task = self.data['agenda'].pop(0)
        self._resident(actor, text, task)
      if not self.data['agenda'] and self.data['actions'] == 4:
        self._boundary()
      self.invariant()
      return (
          '\n'.join(x['text'] for x in self.data['events'][before:])
          or 'No material change.'
      )

  def _queue(self, names, purpose, audience=None):
    for name in names:
      self.data['agenda'].append({
          'name': name,
          'purpose': purpose,
          'audience': list(NAMES if audience is None else audience),
          'watch': self.data['watch'],
      })

  def _commitment(self, name, kind):
    return next(
        (
            x
            for x in reversed(self.data['commitments'])
            if x['owner'] == name
            and x['kind'] == kind
            and x['status'] == 'accepted'
        ),
        None,
    )

  def _human(self, intent):
    self.data['actions'] += 1
    kind = intent['kind']
    self.data['pending'] = copy.deepcopy(intent)
    if kind == 'talk':
      names = (
          list(scenario.RESIDENTS)
          if intent['target'] == 'everyone'
          else [intent['target']]
      )
      audience = [PLAYER, *names] if intent['private'] else NAMES
      self.emit('discussion', PLAYER + ': ' + intent['text'], audience)
      self._queue(names, 'discussion: ' + intent['text'], audience)
    elif kind == 'request':
      name = intent['target']
      terms = (
          'release two reserve fuel and the spare part to the coordinator'
          if name == 'Nell'
          else (
              'perform the beacon repair when supplied and ordered before'
              ' Before Dawn'
          )
      )
      self.emit(
          'proposal',
          f'Coordinator asks {name} to {terms}. This is a request, not'
          ' consent.',
      )
      self._queue([name], 'request: ' + terms)
    elif kind == 'transfer':
      commitment = self._commitment('Nell', 'resources')
      if commitment and self.stock.get_player_inventory('Nell')['fuel'] == 2:
        # Both deltas validated by a SINGLE Inventory.apply call.
        def move(accounts):
          accounts['Nell']['fuel'] -= 2
          accounts['Generator']['fuel'] += 2
          accounts['Nell']['part'] -= 1
          accounts['Ivo']['part'] += 1
          return accounts

        self.stock.apply(move)
        commitment['status'] = 'honored'
        self.emit(
            'transfer',
            'With Nell’s recorded consent, two fuel enter the generator and the'
            ' spare part reaches Ivo.',
        )
      else:
        self.emit(
            'failed_attempt',
            'No transfer: Nell must first consent to release the reserve and'
            ' spare part.',
        )
    elif kind == 'work':
      self.emit(
          'work_order',
          'The coordinator asks Ivo to perform the repair. An order cannot'
          ' supply consent or completed work.',
      )
      self._queue(['Ivo'], 'work: perform your accepted repair or refuse')
    elif kind == 'allocate':
      self.data['allocations'] = intent['facilities']
      self.emit(
          'allocation',
          'Requested supply priority for this watch: '
          + (', '.join(intent['facilities']) or 'none')
          + '. Consumption occurs at the boundary.',
      )
    elif kind == 'promise':
      self.data['commitments'].append({
          'id': len(self.data['commitments']) + 1,
          'owner': PLAYER,
          'kind': 'service',
          'facility': intent['facility'],
          'status': 'accepted',
          'watch': self.data['watch'],
      })
      self.emit(
          'commitment',
          'Coordinator explicitly promises '
          + intent['facility']
          + ' service this watch.',
      )
    elif kind == 'revoke':
      c = self._commitment(PLAYER, 'service')
      if c:
        c['status'] = 'revoked'
        self.emit(
            'revoked',
            'Coordinator withdrew the most recent unfulfilled service promise.',
        )
      else:
        self.emit(
            'failed_attempt', 'No outstanding coordinator promise to withdraw.'
        )
    else:
      self.emit('wait', 'The coordinator waits while the storm continues.')

  def _resident(self, actor, text, task):
    # Malformed output is NEVER silently turned into agreement.
    try:
      decision, speech = parse_resident_response(text)
    except (ValueError, TypeError, AttributeError):
      self.emit(
          'invalid_resident_response',
          f'{actor} returned an unsupported decision; no new decision'
          ' recorded.',
          [actor],
      )
      # A technical failure is not the resident's speech or a refusal. Keep
      # raw output private and report availability only to the task audience.
      self.emit(
          'response_unavailable',
          f'{actor}’s response could not be read. No new decision was'
          ' recorded.',
          task['audience'],
          actor=actor,
          purpose=task['purpose'],
      )
      if task['purpose'].startswith('dawn'):
        self.data['dawn_responses'][actor] = None
      return
    self.emit(
        'speech',
        actor + ': ' + speech,
        task['audience'],
        actor=actor,
        decision=decision,
        purpose=task['purpose'],
    )
    self.data['dialogue'].append({
        'actor': actor,
        'decision': decision,
        'watch': self.data['watch'],
        'audience': task['audience'],
        'speech': speech,
    })
    purpose = task['purpose']
    if purpose.startswith('dawn'):
      self.data['dawn_responses'][actor] = speech
      return
    if purpose.startswith('request:') and decision == 'accept':
      kind = 'resources' if actor == 'Nell' else 'labor'
      if not self._commitment(actor, kind):
        self.data['commitments'].append({
            'id': len(self.data['commitments']) + 1,
            'owner': actor,
            'kind': kind,
            'status': 'accepted',
            'watch': self.data['watch'],
        })
      self.emit(
          'consent',
          f'{actor} explicitly accepts the requested {kind} commitment.',
      )
    elif decision == 'revoke':
      c = self._commitment(actor, 'resources' if actor == 'Nell' else 'labor')
      if c:
        c['status'] = 'revoked'
        self.emit('revoked', f'{actor} withdraws their unfulfilled commitment.')
    elif purpose.startswith('work:') and decision == 'perform':
      labor = self._commitment('Ivo', 'labor')
      if (
          actor == 'Ivo'
          and labor
          and not self.data['repair']
          and self.data['watch'] < 2
          and self.stock.get_player_inventory('Ivo')['part'] == 1
      ):
        self.transfer('Ivo', 'Used', 'part', 1)
        self.data['repair'] = True
        labor['status'] = 'honored'
        self.emit(
            'repair',
            'Ivo performs the accepted repair with the spare part. Before Dawn'
            ' beacon demand is now zero.',
        )
      else:
        self.emit(
            'failed_attempt',
            'No repair: accepted Ivo labor, delivered spare part and completion'
            ' before Before Dawn are all required.',
        )
    if decision in ('accept', 'decline', 'counter', 'revoke'):
      # Record interaction history, not a scalar theory of trust.
      self.data['relationships'].append({
          'from': actor,
          'to': PLAYER,
          'event': decision,
          'watch': self.data['watch'],
          'text': speech,
          'recipients': task['audience'],
      })

  def _boundary(self):
    watch = self.data['watch']
    if watch >= 3:
      return
    supplied = []
    consumption = 0
    available = self.stock.get_player_inventory('Generator')['fuel']
    demands = {
        facility: (
            0
            if facility == 'beacon' and watch == 2 and self.data['repair']
            else 1
        )
        for facility in FACILITIES
    }
    resolutions = {}
    for priority, facility in enumerate(self.data['allocations'], start=1):
      demand = demands[facility]
      fuel_before = available - consumption
      served = consumption + demand <= available
      if served:
        supplied.append(facility)
        consumption += demand
      basis = (
          'repair_supplied'
          if served and demand == 0
          else 'fuel_supplied'
          if served
          else 'fuel_shortfall'
      )
      prefix = f'Request priority {priority}: {fuel_before:g} fuel available; '
      explanation = prefix + (
          'the completed repair supplies the beacon for this watch '
          'without fuel.'
          if basis == 'repair_supplied'
          else f'{demand:g} fuel spent to supply service.'
          if served
          else (
              f'{demand:g} fuel required; no fuel spent because the '
              'remaining stock was insufficient.'
          )
      )
      resolutions[facility] = {
          'basis': basis,
          'requested': True,
          'priority': priority,
          'fuel_before': fuel_before,
          'fuel_spent': demand if served else 0,
          'explanation': explanation,
      }
    if consumption:
      self.transfer('Generator', 'Used', 'fuel', consumption)
    for facility in FACILITIES:
      served = facility in supplied
      self.data['services'].append({
          'watch': WATCHES[watch],
          'facility': facility,
          'served': served,
          'demand': demands[facility],
          'consequence': (
              'Service maintained.' if served else CONSEQUENCES[facility]
          ),
          'resolution': resolutions.get(
              facility,
              {
                  'basis': 'not_requested',
                  'requested': False,
                  'priority': None,
                  'fuel_before': None,
                  'fuel_spent': 0,
                  'explanation': (
                      'Service was not requested for this watch. No fuel was'
                      ' spent.'
                  ),
              },
          ),
      })
    for c in self.data['commitments']:
      if (
          c['owner'] == PLAYER
          and c['status'] == 'accepted'
          and c['watch'] == watch
      ):
        c['status'] = 'honored' if c['facility'] in supplied else 'broken'
    self.emit(
        'watch_closed',
        WATCHES[watch]
        + ' closes. Supplied: '
        + (', '.join(supplied) or 'none')
        + f'. Fuel consumed: {consumption}.',
    )
    self.data['watch'] += 1
    self.data['actions'] = 0
    if self.data['watch'] == 1:
      self.emit(
          'disputed_account', self.dispute['text'], self.dispute['recipients']
      )
      for name in self.dispute['recipients']:
        self.data['knowledge'][name].append('Previous storm dispute')
    if self.data['watch'] == 3:
      for c in self.data['commitments']:
        if c['status'] == 'accepted':
          c['status'] = 'unfulfilled'
      failures = [x for x in self.data['services'] if not x['served']]
      self.data['epilogue'] = {
          'title': 'Dawn at Bellwether',
          'services_maintained': 9 - len(failures),
          'services_total': 9,
          'fuel_used': self.stock.get_player_inventory('Used')['fuel'],
          'repair_completed': self.data['repair'],
          'consequences': failures,
          'commitments': copy.deepcopy(self.data['commitments']),
          'dispute_settled': False,
      }
      self.emit(
          'dawn',
          'Dawn. '
          + str(9 - len(failures))
          + '/9 facility-watches supplied. '
          'The previous-storm dispute is not settled by this accounting. '
          + ' '.join(x['facility'] + ': ' + x['consequence'] for x in failures),
      )
      self._queue(
          list(scenario.RESIDENTS),
          'dawn: respond to the recorded outcomes in your own voice',
      )
    else:
      self.emit(
          'watch_opened',
          WATCHES[self.data['watch']] + ': four consequential choices remain.',
      )

  def view(self, viewer=PLAYER):
    with self.lock:
      data = self.data
      return copy.deepcopy({
          'watch': WATCHES[data['watch']] if data['watch'] < 3 else 'Dawn',
          'remaining': max(0, 4 - data['actions']) if data['watch'] < 3 else 0,
          'next_actor': self.next_actor if not self.finished else None,
          'inventory': self.inventory_state(),
          'repair': data['repair'],
          'allocation': data['allocations'],
          'commitments': data['commitments'],
          'services': data['services'],
          'epilogue': data['epilogue'],
          'dawn_responses': data['dawn_responses'],
          'journal': [
              x
              for x in data['events']
              if (
                  set(NAMES) <= set(x['recipients'])
                  if viewer == 'spectator'
                  else viewer in x['recipients']
              )
          ],
          'relationships': [
              x for x in data['relationships'] if viewer in x['recipients']
          ],
          'institutions': self.institutions,
          'knowledge': data['knowledge'].get(viewer, []),
      })


class Signal(entity_component.ContextComponent):
  """Expose a single GM policy output from the scenario component."""

  def __init__(self, world, output_type, value):
    self.world = world
    self.output_type = output_type
    self.value = value

  def pre_act(self, action_spec):
    return self.value() if action_spec.output_type == self.output_type else ''

  def get_state(self):
    return {}

  def set_state(self, state):
    del state
