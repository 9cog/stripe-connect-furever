"""
Stripe Reasoner

Provides reasoning capabilities over the Stripe knowledge graph,
including inference, pattern matching, and decision support.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .atomspace import StripeAtomspace, AtomType, Atom, Node, Link
from .knowledge_base import StripeKnowledgeBase, EntityCategory


class ReasoningType(Enum):
    """Types of reasoning operations."""
    DEDUCTION = "deduction"  # If A implies B and B implies C, then A implies C
    INDUCTION = "induction"  # From specific observations to general rules
    ABDUCTION = "abduction"  # From effect to most likely cause
    ANALOGY = "analogy"  # From similar situations to conclusions


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    query: str
    reasoning_type: ReasoningType
    conclusion: str
    confidence: ConfidenceLevel
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceRule:
    """A rule for making inferences."""
    name: str
    condition: Callable[[Atom], bool]
    action: Callable[[Atom, StripeAtomspace], List[Atom]]
    priority: int = 0


class StripeReasoner:
    """
    Reasoning engine for the Stripe ecosystem.

    Provides:
    - Entity relationship inference
    - Risk assessment
    - Best practice recommendations
    - Decision support for payments
    - Pattern detection
    """

    def __init__(
        self,
        atomspace: StripeAtomspace,
        knowledge_base: StripeKnowledgeBase
    ):
        self.atomspace = atomspace
        self.knowledge_base = knowledge_base
        self.inference_rules: List[InferenceRule] = []
        self.logger = logging.getLogger(__name__)

        self._register_default_rules()

    def _register_default_rules(self):
        """Register default inference rules."""

        # Rule: Payment failure pattern detection
        def payment_failure_condition(atom: Atom) -> bool:
            return (
                atom.atom_type == AtomType.PAYMENT and
                isinstance(atom, Node) and
                atom.value and
                atom.value.get('status') == 'failed'
            )

        def payment_failure_action(
            atom: Atom,
            atomspace: StripeAtomspace
        ) -> List[Atom]:
            # Create a risk indicator
            risk_node = atomspace.add_node(
                AtomType.CONCEPT,
                f"risk_indicator_{atom.name}",
                truth_value=(0.8, 0.7),
                metadata={'type': 'payment_failure', 'source': atom.id}
            )
            return [risk_node]

        self.register_rule(InferenceRule(
            name="payment_failure_risk",
            condition=payment_failure_condition,
            action=payment_failure_action,
            priority=10
        ))

        # Rule: High-value transaction detection
        def high_value_condition(atom: Atom) -> bool:
            if atom.atom_type != AtomType.PAYMENT:
                return False
            if not isinstance(atom, Node) or not atom.value:
                return False
            amount = atom.value.get('amount', 0)
            return amount > 100000  # > $1000 in cents

        def high_value_action(
            atom: Atom,
            atomspace: StripeAtomspace
        ) -> List[Atom]:
            alert_node = atomspace.add_node(
                AtomType.CONCEPT,
                f"high_value_alert_{atom.name}",
                truth_value=(0.9, 0.9),
                metadata={
                    'type': 'high_value_transaction',
                    'amount': atom.value.get('amount'),
                    'source': atom.id
                }
            )
            return [alert_node]

        self.register_rule(InferenceRule(
            name="high_value_detection",
            condition=high_value_condition,
            action=high_value_action,
            priority=15
        ))

    def register_rule(self, rule: InferenceRule):
        """Register a new inference rule."""
        self.inference_rules.append(rule)
        self.inference_rules.sort(key=lambda r: r.priority, reverse=True)

    def run_forward_chaining(self) -> List[Atom]:
        """
        Run forward chaining inference on all atoms.

        Returns:
            List of newly created atoms from inference
        """
        new_atoms = []

        for rule in self.inference_rules:
            for atom in list(self.atomspace.atoms.values()):
                try:
                    if rule.condition(atom):
                        created = rule.action(atom, self.atomspace)
                        new_atoms.extend(created)
                except Exception as e:
                    self.logger.error(
                        f"Error running rule {rule.name}: {e}"
                    )

        return new_atoms

    def assess_payment_risk(
        self,
        payment_id: str
    ) -> ReasoningResult:
        """
        Assess the risk level of a payment.

        Args:
            payment_id: The payment identifier

        Returns:
            ReasoningResult with risk assessment
        """
        payment = self.atomspace.get_node(AtomType.PAYMENT, payment_id)
        evidence = []
        risk_score = 0.0

        if not payment:
            return ReasoningResult(
                query=f"payment_risk:{payment_id}",
                reasoning_type=ReasoningType.ABDUCTION,
                conclusion="Payment not found",
                confidence=ConfidenceLevel.UNCERTAIN,
                evidence=["Payment ID not in knowledge base"]
            )

        # Check payment value
        if payment.value:
            amount = payment.value.get('amount', 0)
            if amount > 100000:  # > $1000
                risk_score += 0.3
                evidence.append(f"High value transaction: {amount/100:.2f}")

            status = payment.value.get('status', '')
            if status == 'failed':
                risk_score += 0.4
                evidence.append(f"Payment status: {status}")

        # Check incoming links for risk indicators
        incoming = self.atomspace.get_incoming(payment)
        for link in incoming:
            if 'risk' in link.name.lower():
                risk_score += 0.2
                evidence.append(f"Risk indicator found: {link.name}")

        # Determine confidence level
        if risk_score >= 0.7:
            confidence = ConfidenceLevel.HIGH
            conclusion = "High risk payment - review recommended"
        elif risk_score >= 0.4:
            confidence = ConfidenceLevel.MEDIUM
            conclusion = "Moderate risk payment - monitor closely"
        else:
            confidence = ConfidenceLevel.LOW
            conclusion = "Low risk payment - appears normal"

        recommendations = self._generate_risk_recommendations(risk_score)

        return ReasoningResult(
            query=f"payment_risk:{payment_id}",
            reasoning_type=ReasoningType.ABDUCTION,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            metadata={'risk_score': risk_score}
        )

    def _generate_risk_recommendations(
        self,
        risk_score: float
    ) -> List[str]:
        """Generate recommendations based on risk score."""
        recommendations = []

        if risk_score >= 0.7:
            recommendations.extend([
                "Implement additional verification steps",
                "Review customer history",
                "Consider manual review before processing",
                "Check for velocity patterns"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Monitor transaction for disputes",
                "Verify customer identity if new",
                "Check payment method validity"
            ])
        else:
            recommendations.extend([
                "Standard processing recommended",
                "Continue monitoring for anomalies"
            ])

        return recommendations

    def recommend_sdk(
        self,
        requirements: Dict[str, Any]
    ) -> ReasoningResult:
        """
        Recommend the best SDK based on requirements.

        Args:
            requirements: Dictionary of requirements
                - language: Preferred language
                - features: Required features
                - use_case: Use case description

        Returns:
            ReasoningResult with SDK recommendation
        """
        language = requirements.get('language', '').lower()
        required_features = requirements.get('features', [])
        use_case = requirements.get('use_case', '')

        matches = []
        evidence = []

        for name, sdk in self.knowledge_base.SDK_DEFINITIONS.items():
            score = 0.0

            # Language match
            if language and sdk.language.value == language:
                score += 0.5
                evidence.append(f"{name} matches language: {language}")

            # Feature match
            for feature in required_features:
                if feature.lower() in [f.lower() for f in sdk.features]:
                    score += 0.2
                    evidence.append(f"{name} has feature: {feature}")

            if score > 0:
                matches.append((name, score, sdk))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            best_match = matches[0]
            conclusion = f"Recommended SDK: {best_match[0]}"
            confidence = ConfidenceLevel.HIGH if best_match[1] >= 0.7 else ConfidenceLevel.MEDIUM
            recommendations = [
                f"Install using: npm install {best_match[0]}" if best_match[2].language.value == 'node'
                else f"Check {best_match[2].path} for installation instructions",
                f"Features available: {', '.join(best_match[2].features)}"
            ]
        else:
            conclusion = "No SDK matches the requirements"
            confidence = ConfidenceLevel.UNCERTAIN
            recommendations = [
                "Consider broadening language requirements",
                "Check available SDKs in knowledge base"
            ]

        return ReasoningResult(
            query=f"sdk_recommendation",
            reasoning_type=ReasoningType.DEDUCTION,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            metadata={'matches': [(m[0], m[1]) for m in matches[:3]]}
        )

    def infer_entity_relationships(
        self,
        entity_type: str
    ) -> ReasoningResult:
        """
        Infer relationships for a given entity type.

        Args:
            entity_type: The entity type to analyze

        Returns:
            ReasoningResult with inferred relationships
        """
        definition = self.knowledge_base.get_entity_definition(entity_type)

        if not definition:
            return ReasoningResult(
                query=f"relationships:{entity_type}",
                reasoning_type=ReasoningType.DEDUCTION,
                conclusion="Entity type not found",
                confidence=ConfidenceLevel.UNCERTAIN
            )

        evidence = []
        recommendations = []

        # Direct relationships
        for rel_name, rel_target in definition.relationships.items():
            evidence.append(f"{entity_type} -> {rel_name} -> {rel_target}")

        # Infer transitive relationships
        for rel_name, rel_target in definition.relationships.items():
            target_type = rel_target.replace('[]', '').lower()
            target_def = self.knowledge_base.get_entity_definition(target_type)
            if target_def:
                for sub_rel, sub_target in target_def.relationships.items():
                    evidence.append(
                        f"{entity_type} -> {rel_name} -> "
                        f"{target_type} -> {sub_rel} -> {sub_target}"
                    )

        recommendations.append(
            f"Primary endpoint: {definition.api_endpoint}"
        )
        recommendations.append(
            f"Required fields: {', '.join(definition.required_fields)}"
        )

        return ReasoningResult(
            query=f"relationships:{entity_type}",
            reasoning_type=ReasoningType.DEDUCTION,
            conclusion=f"Found {len(definition.relationships)} direct relationships",
            confidence=ConfidenceLevel.HIGH,
            evidence=evidence,
            recommendations=recommendations,
            metadata={
                'entity_type': entity_type,
                'category': definition.category.value
            }
        )

    def analyze_pattern(
        self,
        pattern_type: str,
        data: List[Dict[str, Any]]
    ) -> ReasoningResult:
        """
        Analyze data for specific patterns.

        Args:
            pattern_type: Type of pattern to look for
            data: Data to analyze

        Returns:
            ReasoningResult with pattern analysis
        """
        evidence = []
        confidence = ConfidenceLevel.MEDIUM

        if pattern_type == "velocity":
            # Check for rapid successive transactions
            if len(data) > 10:
                evidence.append(f"High transaction volume: {len(data)} transactions")
                confidence = ConfidenceLevel.HIGH

        elif pattern_type == "amount_anomaly":
            # Check for unusual amounts
            amounts = [d.get('amount', 0) for d in data if 'amount' in d]
            if amounts:
                avg = sum(amounts) / len(amounts)
                outliers = [a for a in amounts if a > avg * 3]
                if outliers:
                    evidence.append(f"Amount outliers detected: {len(outliers)}")
                    confidence = ConfidenceLevel.HIGH

        elif pattern_type == "geographic":
            # Check for geographic dispersion
            countries = set(d.get('country', '') for d in data if 'country' in d)
            if len(countries) > 5:
                evidence.append(f"High geographic dispersion: {len(countries)} countries")

        return ReasoningResult(
            query=f"pattern:{pattern_type}",
            reasoning_type=ReasoningType.INDUCTION,
            conclusion=f"Pattern analysis for {pattern_type}",
            confidence=confidence,
            evidence=evidence,
            metadata={'pattern_type': pattern_type, 'data_points': len(data)}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        return {
            'registered_rules': len(self.inference_rules),
            'rule_names': [r.name for r in self.inference_rules],
            'atomspace_stats': self.atomspace.get_statistics()
        }
