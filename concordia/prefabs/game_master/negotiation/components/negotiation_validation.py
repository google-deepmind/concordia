# Copyright 2025 DeepMind Technologies Limited.
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

"""Negotiation validation and agreement enforcement component."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from concordia.typing import entity_component


@dataclasses.dataclass
class ValidationRule:
  """Represents a validation rule for negotiation terms."""
  name: str
  description: str
  validator: Any  # Callable[[Dict[str, Any]], bool]
  error_message: str


@dataclasses.dataclass
class Agreement:
  """Represents a validated agreement between parties."""
  agreement_id: str
  parties: List[str]
  terms: Dict[str, Any]
  validation_status: str  # 'valid', 'invalid', 'pending'
  validation_errors: List[str]
  timestamp: str
  enforced: bool = False


class NegotiationValidator(entity_component.ContextComponent):
  """Validates negotiation offers and enforces agreements.

  This component:
  - Validates offers against predefined rules
  - Ensures BATNA compliance
  - Checks agreement feasibility
  - Enforces agreed terms
  - Prevents invalid or impossible agreements
  """

  def __init__(
      self,
      domain_type: str = 'general',
      enable_batna_check: bool = True,
      enable_fairness_check: bool = True,
      enable_feasibility_check: bool = True,
      custom_rules: Optional[List[ValidationRule]] = None,
  ):
    """Initialize negotiation validator.

    Args:
      domain_type: Type of negotiation domain ('price', 'contract', 'multi_issue')
      enable_batna_check: Whether to validate against BATNAs
      enable_fairness_check: Whether to check for fair deals
      enable_feasibility_check: Whether to verify feasibility
      custom_rules: Additional validation rules
    """
    self._domain_type = domain_type
    self._enable_batna_check = enable_batna_check
    self._enable_fairness_check = enable_fairness_check
    self._enable_feasibility_check = enable_feasibility_check

    # Validation rules
    self._rules: List[ValidationRule] = []
    self._setup_default_rules()

    if custom_rules:
      self._rules.extend(custom_rules)

    # BATNA tracking
    self._batnas: Dict[str, Dict[str, Any]] = {}

    # Validated agreements
    self._agreements: Dict[str, Agreement] = {}

    # Domain constraints
    self._constraints: Dict[str, Any] = self._get_domain_constraints()

  def _setup_default_rules(self) -> None:
    """Set up default validation rules based on domain."""
    if self._domain_type == 'price':
      self._rules.extend([
          ValidationRule(
              name='positive_price',
              description='Price must be positive',
              validator=lambda terms: terms.get('price', 0) > 0,
              error_message='Price must be greater than zero',
          ),
          ValidationRule(
              name='price_bounds',
              description='Price within reasonable bounds',
              validator=lambda terms: 0 < terms.get('price', 0) < 1000000,
              error_message='Price outside reasonable bounds',
          ),
      ])

    elif self._domain_type == 'contract':
      self._rules.extend([
          ValidationRule(
              name='contract_duration',
              description='Contract duration must be specified',
              validator=lambda terms: 'duration' in terms and terms['duration'] > 0,
              error_message='Contract must have valid duration',
          ),
          ValidationRule(
              name='payment_terms',
              description='Payment terms must be clear',
              validator=lambda terms: 'payment_terms' in terms,
              error_message='Payment terms must be specified',
          ),
      ])

    elif self._domain_type == 'multi_issue':
      self._rules.extend([
          ValidationRule(
              name='all_issues_addressed',
              description='All negotiation issues must be addressed',
              validator=lambda terms: self._check_all_issues(terms),
              error_message='Not all negotiation issues have been addressed',
          ),
      ])

  def _get_domain_constraints(self) -> Dict[str, Any]:
    """Get domain-specific constraints."""
    if self._domain_type == 'price':
      return {
          'min_price': 0,
          'max_price': 1000000,
          'currency': 'USD',
      }
    elif self._domain_type == 'contract':
      return {
          'min_duration': 1,
          'max_duration': 120,  # months
          'required_clauses': ['payment', 'delivery', 'liability'],
      }
    elif self._domain_type == 'multi_issue':
      return {
          'required_issues': [],  # To be set per negotiation
          'trade_off_allowed': True,
      }
    else:
      return {}

  def set_batna(self, party: str, batna: Dict[str, Any]) -> None:
    """Set BATNA (Best Alternative to Negotiated Agreement) for a party."""
    self._batnas[party] = batna

  def set_required_issues(self, issues: List[str]) -> None:
    """Set required issues for multi-issue negotiations."""
    if self._domain_type == 'multi_issue':
      self._constraints['required_issues'] = issues

  def _check_all_issues(self, terms: Dict[str, Any]) -> bool:
    """Check if all required issues are addressed."""
    required = self._constraints.get('required_issues', [])
    return all(issue in terms for issue in required)

  def validate_offer(
      self,
      offerer: str,
      terms: Dict[str, Any],
      context: Optional[Dict[str, Any]] = None,
  ) -> Tuple[bool, List[str]]:
    """Validate an offer against all rules.

    Returns:
      Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check custom rules
    for rule in self._rules:
      try:
        if not rule.validator(terms):
          errors.append(f"{rule.name}: {rule.error_message}")
      except Exception as e:
        errors.append(f"{rule.name}: Validation error - {str(e)}")

    # BATNA check
    if self._enable_batna_check and offerer in self._batnas:
      batna = self._batnas[offerer]
      if not self._check_batna_compliance(terms, batna):
        errors.append(f"Offer violates {offerer}'s BATNA")

    # Fairness check
    if self._enable_fairness_check:
      fairness_issues = self._check_fairness(terms, context)
      errors.extend(fairness_issues)

    # Feasibility check
    if self._enable_feasibility_check:
      feasibility_issues = self._check_feasibility(terms, context)
      errors.extend(feasibility_issues)

    return len(errors) == 0, errors

  def _check_batna_compliance(
      self,
      terms: Dict[str, Any],
      batna: Dict[str, Any],
  ) -> bool:
    """Check if offer meets or exceeds BATNA."""
    if self._domain_type == 'price':
      # For seller, offer should be >= BATNA
      # For buyer, offer should be <= BATNA
      offer_price = terms.get('price', 0)
      batna_price = batna.get('price', 0)
      role = batna.get('role', 'buyer')

      if role == 'seller':
        return offer_price >= batna_price
      else:
        return offer_price <= batna_price

    # For other domains, implement domain-specific BATNA checks
    return True

  def _check_fairness(
      self,
      terms: Dict[str, Any],
      context: Optional[Dict[str, Any]],
  ) -> List[str]:
    """Check if the deal is reasonably fair."""
    issues = []

    if self._domain_type == 'price' and context:
      market_price = context.get('market_price')
      if market_price:
        offer_price = terms.get('price', 0)
        deviation = abs(offer_price - market_price) / market_price

        if deviation > 0.5:  # More than 50% deviation
          issues.append(f"Price deviates significantly from market ({deviation:.0%})")

    return issues

  def _check_feasibility(
      self,
      terms: Dict[str, Any],
      context: Optional[Dict[str, Any]],
  ) -> List[str]:
    """Check if the agreement is feasible to implement."""
    issues = []

    if self._domain_type == 'contract':
      # Check timeline feasibility
      if 'start_date' in terms and 'duration' in terms:
        # Add domain-specific feasibility checks
        pass

    return issues

  def validate_agreement(
      self,
      parties: List[str],
      terms: Dict[str, Any],
      context: Optional[Dict[str, Any]] = None,
  ) -> Agreement:
    """Validate a potential agreement between parties."""
    # Validate from each party's perspective
    all_errors = []

    for party in parties:
      is_valid, errors = self.validate_offer(party, terms, context)
      if not is_valid:
        all_errors.extend([f"{party}: {error}" for error in errors])

    # Create agreement record
    agreement_id = f"agreement_{len(self._agreements) + 1}"
    agreement = Agreement(
        agreement_id=agreement_id,
        parties=parties,
        terms=terms,
        validation_status='valid' if not all_errors else 'invalid',
        validation_errors=all_errors,
        timestamp='current',
        enforced=False,
    )

    self._agreements[agreement_id] = agreement
    return agreement

  def enforce_agreement(self, agreement_id: str) -> bool:
    """Enforce a validated agreement."""
    if agreement_id not in self._agreements:
      return False

    agreement = self._agreements[agreement_id]

    if agreement.validation_status != 'valid':
      return False

    # Mark as enforced
    agreement.enforced = True

    # In a real system, this would trigger actual enforcement actions
    # For simulation, we just track the enforcement

    return True

  def get_validation_report(self) -> str:
    """Get a report of validation activities."""
    report = "NEGOTIATION VALIDATION REPORT:\n\n"

    # Active rules
    report += f"Active validation rules: {len(self._rules)}\n"
    for rule in self._rules:
      report += f"  - {rule.name}: {rule.description}\n"

    # BATNA status
    if self._batnas:
      report += f"\nRegistered BATNAs:\n"
      for party, batna in self._batnas.items():
        report += f"  - {party}: {batna}\n"

    # Agreements
    valid_agreements = sum(1 for a in self._agreements.values()
                          if a.validation_status == 'valid')
    invalid_agreements = sum(1 for a in self._agreements.values()
                            if a.validation_status == 'invalid')

    report += f"\nAgreements validated: {len(self._agreements)}\n"
    report += f"  - Valid: {valid_agreements}\n"
    report += f"  - Invalid: {invalid_agreements}\n"
    report += f"  - Enforced: {sum(1 for a in self._agreements.values() if a.enforced)}\n"

    return report

  def pre_act(self, action_spec) -> str:
    """Provide validation context."""
    return self.get_validation_report()

  def post_act(self, action_attempt: str) -> None:
    """Process validation-related actions."""
    pass

  def pre_observe(self, observation: str) -> None:
    """Process observations."""
    pass

  def post_observe(self) -> None:
    """Post-observation processing."""
    pass

  def update(self) -> None:
    """Update internal state."""
    pass

  @property
  def name(self) -> str:
    """Component name."""
    return 'NegotiationValidator'

  def get_state(self) -> str:
    """Get component state."""
    valid = sum(1 for a in self._agreements.values()
               if a.validation_status == 'valid')
    return f"{len(self._agreements)}|{valid}"

  def set_state(self, state: str) -> None:
    """Restore component state."""
    # Simple state restoration
    pass
