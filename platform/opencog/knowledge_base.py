"""
Stripe Knowledge Base

Provides high-level knowledge management for the Stripe ecosystem,
including entity relationships, SDK mappings, and API knowledge.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .atomspace import (
    StripeAtomspace,
    AtomType,
    Node,
    Link,
    create_payment_atom,
    create_customer_atom,
    create_sdk_atom
)


class EntityCategory(Enum):
    """Categories of Stripe entities."""
    CORE = "core"  # Payments, Customers, etc.
    BILLING = "billing"  # Subscriptions, Invoices
    CONNECT = "connect"  # Connected Accounts, Transfers
    TERMINAL = "terminal"  # Physical POS
    IDENTITY = "identity"  # Identity verification
    RADAR = "radar"  # Fraud detection
    ISSUING = "issuing"  # Card issuing
    TREASURY = "treasury"  # Treasury/Banking


class SDKLanguage(Enum):
    """Supported SDK languages."""
    NODE = "node"
    PYTHON = "python"
    RUBY = "ruby"
    GO = "go"
    JAVA = "java"
    DOTNET = "dotnet"
    PHP = "php"


@dataclass
class EntityDefinition:
    """Definition of a Stripe entity type."""
    name: str
    category: EntityCategory
    api_endpoint: str
    description: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)


@dataclass
class SDKDefinition:
    """Definition of a Stripe SDK."""
    name: str
    language: SDKLanguage
    path: str
    version: str
    status: str = "active"
    features: List[str] = field(default_factory=list)


class StripeKnowledgeBase:
    """
    High-level knowledge base for the Stripe ecosystem.

    This manages:
    - Entity definitions and relationships
    - SDK knowledge and capabilities
    - API endpoint mappings
    - Best practices and patterns
    """

    # Core entity definitions
    ENTITY_DEFINITIONS: Dict[str, EntityDefinition] = {
        "payment": EntityDefinition(
            name="Payment",
            category=EntityCategory.CORE,
            api_endpoint="/v1/payment_intents",
            description="A PaymentIntent tracks a payment lifecycle",
            required_fields=["amount", "currency"],
            optional_fields=["customer", "payment_method", "description"],
            relationships={"customer": "Customer", "invoice": "Invoice"}
        ),
        "customer": EntityDefinition(
            name="Customer",
            category=EntityCategory.CORE,
            api_endpoint="/v1/customers",
            description="Customer objects for storing payment methods",
            required_fields=[],
            optional_fields=["email", "name", "phone", "address"],
            relationships={"payments": "Payment[]", "subscriptions": "Subscription[]"}
        ),
        "subscription": EntityDefinition(
            name="Subscription",
            category=EntityCategory.BILLING,
            api_endpoint="/v1/subscriptions",
            description="Recurring billing configuration",
            required_fields=["customer", "items"],
            optional_fields=["trial_period_days", "cancel_at_period_end"],
            relationships={"customer": "Customer", "invoices": "Invoice[]"}
        ),
        "invoice": EntityDefinition(
            name="Invoice",
            category=EntityCategory.BILLING,
            api_endpoint="/v1/invoices",
            description="Statements of amounts owed",
            required_fields=["customer"],
            optional_fields=["subscription", "auto_advance"],
            relationships={"customer": "Customer", "payment": "Payment"}
        ),
        "account": EntityDefinition(
            name="Account",
            category=EntityCategory.CONNECT,
            api_endpoint="/v1/accounts",
            description="Connected account for platforms",
            required_fields=["type"],
            optional_fields=["email", "country", "capabilities"],
            relationships={"payouts": "Payout[]", "transfers": "Transfer[]"}
        ),
        "payout": EntityDefinition(
            name="Payout",
            category=EntityCategory.CORE,
            api_endpoint="/v1/payouts",
            description="Transfer to bank account",
            required_fields=["amount", "currency"],
            optional_fields=["destination", "method"],
            relationships={"account": "Account"}
        ),
        "refund": EntityDefinition(
            name="Refund",
            category=EntityCategory.CORE,
            api_endpoint="/v1/refunds",
            description="Refund of a charge",
            required_fields=["payment_intent"],
            optional_fields=["amount", "reason"],
            relationships={"payment": "Payment"}
        ),
        "dispute": EntityDefinition(
            name="Dispute",
            category=EntityCategory.CORE,
            api_endpoint="/v1/disputes",
            description="Customer-initiated chargeback",
            required_fields=[],
            optional_fields=["evidence"],
            relationships={"payment": "Payment"}
        ),
        "product": EntityDefinition(
            name="Product",
            category=EntityCategory.BILLING,
            api_endpoint="/v1/products",
            description="Product or service offered",
            required_fields=["name"],
            optional_fields=["description", "images", "metadata"],
            relationships={"prices": "Price[]"}
        ),
        "price": EntityDefinition(
            name="Price",
            category=EntityCategory.BILLING,
            api_endpoint="/v1/prices",
            description="Pricing configuration for products",
            required_fields=["currency", "product"],
            optional_fields=["unit_amount", "recurring"],
            relationships={"product": "Product"}
        )
    }

    # SDK definitions
    SDK_DEFINITIONS: Dict[str, SDKDefinition] = {
        "stripe-node": SDKDefinition(
            name="stripe-node",
            language=SDKLanguage.NODE,
            path="platform/stripe-ecosystem/sdks/stripe-node",
            version="latest",
            features=["async/await", "typescript", "auto-pagination"]
        ),
        "stripe-python": SDKDefinition(
            name="stripe-python",
            language=SDKLanguage.PYTHON,
            path="platform/stripe-ecosystem/sdks/stripe-python",
            version="latest",
            features=["async", "type-hints", "auto-pagination"]
        ),
        "stripe-ruby": SDKDefinition(
            name="stripe-ruby",
            language=SDKLanguage.RUBY,
            path="platform/stripe-ecosystem/sdks/stripe-ruby",
            version="latest",
            features=["auto-pagination", "idempotency"]
        ),
        "stripe-go": SDKDefinition(
            name="stripe-go",
            language=SDKLanguage.GO,
            path="platform/stripe-ecosystem/sdks/stripe-go",
            version="latest",
            features=["context-support", "streaming"]
        ),
        "stripe-java": SDKDefinition(
            name="stripe-java",
            language=SDKLanguage.JAVA,
            path="platform/stripe-ecosystem/sdks/stripe-java",
            version="latest",
            features=["async", "streaming", "builder-pattern"]
        ),
        "stripe-dotnet": SDKDefinition(
            name="stripe-dotnet",
            language=SDKLanguage.DOTNET,
            path="platform/stripe-ecosystem/sdks/stripe-dotnet",
            version="latest",
            features=["async", "linq", "strong-typing"]
        ),
        "stripe-php": SDKDefinition(
            name="stripe-php",
            language=SDKLanguage.PHP,
            path="platform/stripe-ecosystem/plugins/stripe-php",
            version="latest",
            features=["psr-compliant", "auto-pagination"]
        )
    }

    def __init__(self, atomspace: Optional[StripeAtomspace] = None):
        self.atomspace = atomspace or StripeAtomspace("stripe_knowledge")
        self._initialized = False

    def initialize(self):
        """Initialize the knowledge base with core definitions."""
        if self._initialized:
            return

        self._load_entity_definitions()
        self._load_sdk_definitions()
        self._load_api_knowledge()
        self._initialized = True

    def _load_entity_definitions(self):
        """Load entity definitions into the Atomspace."""
        for name, definition in self.ENTITY_DEFINITIONS.items():
            # Create entity concept
            entity_node = self.atomspace.add_node(
                AtomType.CONCEPT,
                f"stripe_{name}",
                value=definition,
                metadata={
                    'category': definition.category.value,
                    'api_endpoint': definition.api_endpoint
                }
            )

            # Create category concept and link
            category_node = self.atomspace.add_node(
                AtomType.CONCEPT,
                f"category_{definition.category.value}"
            )
            self.atomspace.add_link(
                AtomType.MEMBER,
                [entity_node, category_node]
            )

            # Create API endpoint node
            api_node = self.atomspace.add_node(
                AtomType.API_ENDPOINT,
                definition.api_endpoint
            )
            self.atomspace.add_link(
                AtomType.EVALUATION,
                [
                    self.atomspace.add_node(AtomType.PREDICATE, "has_endpoint"),
                    self.atomspace.add_link(AtomType.LIST, [entity_node, api_node])
                ]
            )

    def _load_sdk_definitions(self):
        """Load SDK definitions into the Atomspace."""
        for name, definition in self.SDK_DEFINITIONS.items():
            sdk_node = create_sdk_atom(
                self.atomspace,
                name,
                definition.language.value,
                definition.version
            )
            sdk_node.metadata['path'] = definition.path
            sdk_node.metadata['features'] = definition.features

    def _load_api_knowledge(self):
        """Load API patterns and best practices."""
        # Add common API patterns
        patterns = [
            ("idempotency", "Use Idempotency-Key header for safe retries"),
            ("pagination", "Use auto-pagination or starting_after for lists"),
            ("error_handling", "Handle StripeError and specific error types"),
            ("webhooks", "Verify webhook signatures for security"),
            ("rate_limiting", "Implement exponential backoff on 429 errors"),
            ("versioning", "Pin to specific API version in production")
        ]

        for pattern_name, description in patterns:
            pattern_node = self.atomspace.add_node(
                AtomType.CONCEPT,
                f"pattern_{pattern_name}",
                value={'description': description}
            )

            # Link to best practices concept
            best_practices = self.atomspace.add_node(
                AtomType.CONCEPT,
                "best_practices"
            )
            self.atomspace.add_link(
                AtomType.MEMBER,
                [pattern_node, best_practices]
            )

    def get_entity_definition(self, entity_name: str) -> Optional[EntityDefinition]:
        """Get the definition for an entity type."""
        return self.ENTITY_DEFINITIONS.get(entity_name.lower())

    def get_sdk_definition(self, sdk_name: str) -> Optional[SDKDefinition]:
        """Get the definition for an SDK."""
        return self.SDK_DEFINITIONS.get(sdk_name)

    def get_entities_by_category(
        self,
        category: EntityCategory
    ) -> List[EntityDefinition]:
        """Get all entity definitions in a category."""
        return [
            defn for defn in self.ENTITY_DEFINITIONS.values()
            if defn.category == category
        ]

    def get_sdks_by_language(self, language: SDKLanguage) -> List[SDKDefinition]:
        """Get all SDKs for a specific language."""
        return [
            sdk for sdk in self.SDK_DEFINITIONS.values()
            if sdk.language == language
        ]

    def query_relationships(
        self,
        entity_type: str,
        relationship: str
    ) -> Optional[str]:
        """Query the relationship target for an entity type."""
        definition = self.get_entity_definition(entity_type)
        if definition:
            return definition.relationships.get(relationship)
        return None

    def get_api_endpoint(self, entity_type: str) -> Optional[str]:
        """Get the API endpoint for an entity type."""
        definition = self.get_entity_definition(entity_type)
        if definition:
            return definition.api_endpoint
        return None

    def search_knowledge(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for matching concepts.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching knowledge items
        """
        results = []
        query_lower = query.lower()

        # Search entities
        for name, defn in self.ENTITY_DEFINITIONS.items():
            if query_lower in name or query_lower in defn.description.lower():
                results.append({
                    'type': 'entity',
                    'name': name,
                    'category': defn.category.value,
                    'description': defn.description,
                    'endpoint': defn.api_endpoint
                })

        # Search SDKs
        for name, sdk in self.SDK_DEFINITIONS.items():
            if query_lower in name or query_lower in sdk.language.value:
                results.append({
                    'type': 'sdk',
                    'name': name,
                    'language': sdk.language.value,
                    'features': sdk.features
                })

        return results[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            'entity_count': len(self.ENTITY_DEFINITIONS),
            'sdk_count': len(self.SDK_DEFINITIONS),
            'atomspace_stats': self.atomspace.get_statistics(),
            'categories': [c.value for c in EntityCategory],
            'languages': [l.value for l in SDKLanguage]
        }
