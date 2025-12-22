"""
Stripe Atomspace Implementation

Provides a knowledge graph representation for Stripe entities,
relationships, and operations using OpenCog-inspired patterns.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import hashlib


class AtomType(Enum):
    """Types of atoms in the Stripe knowledge graph."""
    # Core Entity Types
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode"
    VARIABLE = "VariableNode"
    NUMBER = "NumberNode"
    SCHEMA = "SchemaNode"

    # Stripe-Specific Entity Types
    PAYMENT = "PaymentNode"
    CUSTOMER = "CustomerNode"
    SUBSCRIPTION = "SubscriptionNode"
    INVOICE = "InvoiceNode"
    PRODUCT = "ProductNode"
    PRICE = "PriceNode"
    ACCOUNT = "AccountNode"
    PAYOUT = "PayoutNode"
    REFUND = "RefundNode"
    DISPUTE = "DisputeNode"
    BALANCE = "BalanceNode"

    # SDK/Plugin Types
    SDK = "SDKNode"
    PLUGIN = "PluginNode"
    API_ENDPOINT = "APIEndpointNode"
    WEBHOOK = "WebhookNode"

    # Link Types
    INHERITANCE = "InheritanceLink"
    EVALUATION = "EvaluationLink"
    EXECUTION = "ExecutionLink"
    MEMBER = "MemberLink"
    LIST = "ListLink"
    SET = "SetLink"
    CONTEXT = "ContextLink"
    IMPLICATION = "ImplicationLink"
    AND = "AndLink"
    OR = "OrLink"
    NOT = "NotLink"

    # Stripe Relationship Types
    PAYMENT_CUSTOMER = "PaymentCustomerLink"
    CUSTOMER_SUBSCRIPTION = "CustomerSubscriptionLink"
    SUBSCRIPTION_INVOICE = "SubscriptionInvoiceLink"
    INVOICE_PAYMENT = "InvoicePaymentLink"
    ACCOUNT_PAYOUT = "AccountPayoutLink"
    PAYMENT_REFUND = "PaymentRefundLink"

    # Temporal Types
    TEMPORAL = "TemporalNode"
    SEQUENCE = "SequenceLink"

    # Agent Types
    AGENT = "AgentNode"
    ACTION = "ActionNode"
    GOAL = "GoalNode"


@dataclass
class Atom:
    """Base class for atoms in the Atomspace."""
    atom_type: AtomType
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    truth_value: Tuple[float, float] = (1.0, 1.0)  # (strength, confidence)
    attention_value: Tuple[int, int] = (0, 0)  # (STI, LTI)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary representation."""
        return {
            'id': self.id,
            'type': self.atom_type.value,
            'name': self.name,
            'truth_value': {
                'strength': self.truth_value[0],
                'confidence': self.truth_value[1]
            },
            'attention_value': {
                'sti': self.attention_value[0],
                'lti': self.attention_value[1]
            },
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Node(Atom):
    """A node in the Atomspace (represents an entity)."""
    value: Any = None


@dataclass
class Link(Atom):
    """A link in the Atomspace (represents a relationship)."""
    outgoing: List[Atom] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            # Generate name from outgoing atoms
            names = [a.name for a in self.outgoing]
            self.name = f"{self.atom_type.value}({', '.join(names)})"


class StripeAtomspace:
    """
    Knowledge graph for Stripe ecosystem using Atomspace patterns.

    This provides:
    - Entity storage and retrieval
    - Relationship management
    - Pattern matching
    - Truth value propagation
    - Attention allocation
    """

    def __init__(self, name: str = "stripe_atomspace"):
        self.name = name
        self.atoms: Dict[str, Atom] = {}
        self.type_index: Dict[AtomType, Set[str]] = {t: set() for t in AtomType}
        self.name_index: Dict[str, Set[str]] = {}
        self.incoming_index: Dict[str, Set[str]] = {}  # atom_id -> set of link ids
        self.created_at = datetime.now()

    def add_node(
        self,
        atom_type: AtomType,
        name: str,
        value: Any = None,
        truth_value: Tuple[float, float] = (1.0, 1.0),
        metadata: Dict[str, Any] = None
    ) -> Node:
        """Add a node to the Atomspace."""
        # Check if node already exists
        existing = self.get_node(atom_type, name)
        if existing:
            return existing

        node = Node(
            atom_type=atom_type,
            name=name,
            value=value,
            truth_value=truth_value,
            metadata=metadata or {}
        )

        self._add_atom(node)
        return node

    def add_link(
        self,
        atom_type: AtomType,
        outgoing: List[Atom],
        name: str = None,
        truth_value: Tuple[float, float] = (1.0, 1.0),
        metadata: Dict[str, Any] = None
    ) -> Link:
        """Add a link to the Atomspace."""
        link = Link(
            atom_type=atom_type,
            name=name or "",
            outgoing=outgoing,
            truth_value=truth_value,
            metadata=metadata or {}
        )

        self._add_atom(link)

        # Update incoming index
        for atom in outgoing:
            if atom.id not in self.incoming_index:
                self.incoming_index[atom.id] = set()
            self.incoming_index[atom.id].add(link.id)

        return link

    def _add_atom(self, atom: Atom):
        """Internal method to add an atom to indexes."""
        self.atoms[atom.id] = atom
        self.type_index[atom.atom_type].add(atom.id)

        if atom.name not in self.name_index:
            self.name_index[atom.name] = set()
        self.name_index[atom.name].add(atom.id)

    def get_node(self, atom_type: AtomType, name: str) -> Optional[Node]:
        """Get a node by type and name."""
        if name not in self.name_index:
            return None

        for atom_id in self.name_index[name]:
            atom = self.atoms.get(atom_id)
            if isinstance(atom, Node) and atom.atom_type == atom_type:
                return atom
        return None

    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type."""
        return [self.atoms[aid] for aid in self.type_index[atom_type]]

    def get_incoming(self, atom: Atom) -> List[Link]:
        """Get all links that have the atom as an outgoing member."""
        if atom.id not in self.incoming_index:
            return []
        return [self.atoms[lid] for lid in self.incoming_index[atom.id]
                if lid in self.atoms]

    def pattern_match(
        self,
        pattern_type: AtomType,
        constraints: Dict[str, Any] = None
    ) -> List[Atom]:
        """
        Simple pattern matching on the Atomspace.

        Args:
            pattern_type: The type of atom to match
            constraints: Dictionary of constraints to apply

        Returns:
            List of matching atoms
        """
        results = []
        constraints = constraints or {}

        for atom in self.get_atoms_by_type(pattern_type):
            match = True

            # Check name constraint
            if 'name' in constraints:
                if constraints['name'] not in atom.name:
                    match = False

            # Check truth value constraints
            if 'min_strength' in constraints:
                if atom.truth_value[0] < constraints['min_strength']:
                    match = False

            if 'min_confidence' in constraints:
                if atom.truth_value[1] < constraints['min_confidence']:
                    match = False

            # Check metadata constraints
            if 'metadata' in constraints:
                for key, value in constraints['metadata'].items():
                    if atom.metadata.get(key) != value:
                        match = False
                        break

            if match:
                results.append(atom)

        return results

    def update_truth_value(
        self,
        atom: Atom,
        strength: float,
        confidence: float
    ):
        """Update the truth value of an atom."""
        atom.truth_value = (
            max(0.0, min(1.0, strength)),
            max(0.0, min(1.0, confidence))
        )

    def update_attention(
        self,
        atom: Atom,
        sti_delta: int = 0,
        lti_delta: int = 0
    ):
        """Update the attention value of an atom."""
        new_sti = atom.attention_value[0] + sti_delta
        new_lti = atom.attention_value[1] + lti_delta
        atom.attention_value = (new_sti, new_lti)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Atomspace."""
        type_counts = {
            t.value: len(ids) for t, ids in self.type_index.items() if ids
        }

        return {
            'name': self.name,
            'total_atoms': len(self.atoms),
            'type_counts': type_counts,
            'created_at': self.created_at.isoformat()
        }

    def export_to_json(self) -> str:
        """Export the Atomspace to JSON."""
        data = {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'atoms': [atom.to_dict() for atom in self.atoms.values()]
        }
        return json.dumps(data, indent=2)

    def clear(self):
        """Clear all atoms from the Atomspace."""
        self.atoms.clear()
        self.type_index = {t: set() for t in AtomType}
        self.name_index.clear()
        self.incoming_index.clear()


# Stripe-specific helper functions
def create_payment_atom(
    atomspace: StripeAtomspace,
    payment_id: str,
    amount: int,
    currency: str,
    status: str,
    customer_id: Optional[str] = None,
    metadata: Dict[str, Any] = None
) -> Node:
    """Create a payment atom with associated information."""
    payment = atomspace.add_node(
        AtomType.PAYMENT,
        payment_id,
        value={'amount': amount, 'currency': currency, 'status': status},
        metadata=metadata or {}
    )

    # Create amount node
    amount_node = atomspace.add_node(
        AtomType.NUMBER,
        f"amount_{payment_id}",
        value=amount
    )

    # Create evaluation link for amount
    atomspace.add_link(
        AtomType.EVALUATION,
        [
            atomspace.add_node(AtomType.PREDICATE, "has_amount"),
            atomspace.add_link(AtomType.LIST, [payment, amount_node])
        ]
    )

    # Link to customer if provided
    if customer_id:
        customer = atomspace.add_node(AtomType.CUSTOMER, customer_id)
        atomspace.add_link(
            AtomType.PAYMENT_CUSTOMER,
            [payment, customer]
        )

    return payment


def create_customer_atom(
    atomspace: StripeAtomspace,
    customer_id: str,
    email: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Dict[str, Any] = None
) -> Node:
    """Create a customer atom with associated information."""
    customer = atomspace.add_node(
        AtomType.CUSTOMER,
        customer_id,
        value={'email': email, 'name': name},
        metadata=metadata or {}
    )

    return customer


def create_sdk_atom(
    atomspace: StripeAtomspace,
    sdk_name: str,
    language: str,
    version: str,
    status: str = "active"
) -> Node:
    """Create an SDK atom with associated information."""
    sdk = atomspace.add_node(
        AtomType.SDK,
        sdk_name,
        value={'language': language, 'version': version, 'status': status}
    )

    # Create inheritance link to concept
    lang_concept = atomspace.add_node(AtomType.CONCEPT, f"language_{language}")
    atomspace.add_link(AtomType.INHERITANCE, [sdk, lang_concept])

    return sdk
