"""
Security Agent

Specialized agent for fraud detection and compliance monitoring.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentCapability
)


class RiskLevel(Enum):
    """Risk levels for transactions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of security alerts."""
    FRAUD_DETECTED = "fraud_detected"
    UNUSUAL_ACTIVITY = "unusual_activity"
    VELOCITY_EXCEEDED = "velocity_exceeded"
    BLOCKLIST_MATCH = "blocklist_match"
    GEO_ANOMALY = "geo_anomaly"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class SecurityAlert:
    """Security alert record."""
    id: str
    alert_type: AlertType
    risk_level: RiskLevel
    entity_id: str
    entity_type: str
    details: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class RiskRule:
    """Risk assessment rule."""
    name: str
    description: str
    condition: str
    risk_weight: float
    enabled: bool = True


class SecurityAgent(BaseAgent):
    """
    Agent for fraud detection and compliance.

    Capabilities:
    - Assess transaction risk
    - Detect fraud patterns
    - Monitor compliance
    - Manage blocklists
    - Generate security alerts
    """

    # Default risk rules
    DEFAULT_RULES = [
        RiskRule(
            name="high_value_transaction",
            description="Flag transactions over $10,000",
            condition="amount > 1000000",
            risk_weight=0.3
        ),
        RiskRule(
            name="velocity_check",
            description="Flag more than 10 transactions in 1 hour",
            condition="tx_count_1h > 10",
            risk_weight=0.4
        ),
        RiskRule(
            name="new_customer",
            description="Flag transactions from new customers",
            condition="customer_age_days < 7",
            risk_weight=0.2
        ),
        RiskRule(
            name="international_transaction",
            description="Flag international transactions",
            condition="is_international",
            risk_weight=0.15
        ),
        RiskRule(
            name="card_testing_pattern",
            description="Detect card testing patterns",
            condition="small_amounts_rapid",
            risk_weight=0.5
        )
    ]

    def __init__(self, reasoner=None):
        super().__init__(
            name="SecurityAgent",
            description="Fraud detection and compliance monitoring"
        )
        self.reasoner = reasoner

        # Security data
        self.rules: List[RiskRule] = list(self.DEFAULT_RULES)
        self.alerts: List[SecurityAlert] = []
        self.blocklist_ips: Set[str] = set()
        self.blocklist_cards: Set[str] = set()
        self.blocklist_emails: Set[str] = set()

        # Velocity tracking
        self.transaction_velocity: Dict[str, List[datetime]] = {}

        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        self.register_capability(AgentCapability(
            name="assess_risk",
            description="Assess risk level of a transaction",
            input_schema={
                "amount": "integer (required)",
                "customer_id": "string",
                "ip_address": "string",
                "card_fingerprint": "string",
                "country": "string"
            },
            output_schema={"risk_level": "string", "risk_score": "number"}
        ))

        self.register_capability(AgentCapability(
            name="detect_fraud",
            description="Analyze transaction for fraud indicators",
            input_schema={"transaction_data": "object"},
            output_schema={"is_fraud": "boolean", "indicators": "array"}
        ))

        self.register_capability(AgentCapability(
            name="add_to_blocklist",
            description="Add entity to blocklist",
            input_schema={
                "type": "string (ip|card|email)",
                "value": "string"
            }
        ))

        self.register_capability(AgentCapability(
            name="check_blocklist",
            description="Check if entity is blocklisted",
            input_schema={
                "type": "string",
                "value": "string"
            },
            output_schema={"is_blocked": "boolean"}
        ))

        self.register_capability(AgentCapability(
            name="get_alerts",
            description="Get security alerts",
            output_schema={"alerts": "array"}
        ))

        # Register handlers
        self.register_handler("assess_risk", self._handle_assess_risk)
        self.register_handler("detect_fraud", self._handle_detect_fraud)
        self.register_handler("add_to_blocklist", self._handle_add_blocklist)
        self.register_handler("check_blocklist", self._handle_check_blocklist)
        self.register_handler("get_alerts", self._handle_get_alerts)

    async def initialize(self) -> bool:
        """Initialize the security agent."""
        self.logger.info("Initializing SecurityAgent")
        return True

    async def shutdown(self):
        """Shutdown the security agent."""
        self.logger.info("Shutting down SecurityAgent")

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Process incoming messages."""
        handler = self.message_handlers.get(message.action)

        if handler:
            result = await handler(message)
            return message.create_response(result)

        return message.create_response(
            {'error': f'Unknown action: {message.action}'},
            is_error=True
        )

    async def _handle_assess_risk(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle risk assessment request."""
        payload = message.payload
        amount = payload.get('amount', 0)
        customer_id = payload.get('customer_id', '')
        ip_address = payload.get('ip_address', '')
        card_fingerprint = payload.get('card_fingerprint', '')
        country = payload.get('country', '')

        risk_score = 0.0
        triggered_rules = []

        # Check blocklists first
        if ip_address and ip_address in self.blocklist_ips:
            return {
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'blocked': True,
                'reason': 'IP address is blocklisted'
            }

        if card_fingerprint and card_fingerprint in self.blocklist_cards:
            return {
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'blocked': True,
                'reason': 'Card is blocklisted'
            }

        # Apply risk rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            triggered = False

            if rule.name == "high_value_transaction" and amount > 1000000:
                triggered = True
            elif rule.name == "international_transaction" and country and country != 'US':
                triggered = True

            if triggered:
                risk_score += rule.risk_weight
                triggered_rules.append(rule.name)

        # Check velocity
        velocity_risk = self._check_velocity(customer_id)
        risk_score += velocity_risk

        # Normalize score
        risk_score = min(risk_score, 1.0)

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Create alert if high risk
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self._create_alert(
                AlertType.UNUSUAL_ACTIVITY,
                risk_level,
                customer_id or 'unknown',
                'transaction',
                {
                    'amount': amount,
                    'triggered_rules': triggered_rules,
                    'risk_score': risk_score
                }
            )

        return {
            'risk_level': risk_level.value,
            'risk_score': round(risk_score, 3),
            'triggered_rules': triggered_rules,
            'recommendations': self._get_recommendations(risk_level)
        }

    def _check_velocity(self, customer_id: str) -> float:
        """Check transaction velocity for a customer."""
        if not customer_id:
            return 0.0

        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        if customer_id not in self.transaction_velocity:
            self.transaction_velocity[customer_id] = []

        # Add current transaction
        self.transaction_velocity[customer_id].append(now)

        # Clean old entries
        self.transaction_velocity[customer_id] = [
            t for t in self.transaction_velocity[customer_id]
            if t > one_hour_ago
        ]

        # Calculate velocity risk
        tx_count = len(self.transaction_velocity[customer_id])
        if tx_count > 10:
            return 0.4
        elif tx_count > 5:
            return 0.2
        return 0.0

    def _get_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get recommendations based on risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return [
                "Block transaction immediately",
                "Contact customer for verification",
                "Review account for suspicious activity"
            ]
        elif risk_level == RiskLevel.HIGH:
            return [
                "Require additional verification (3DS)",
                "Review transaction manually",
                "Monitor customer for 24 hours"
            ]
        elif risk_level == RiskLevel.MEDIUM:
            return [
                "Consider 3D Secure verification",
                "Log for review if pattern continues"
            ]
        return ["Process normally"]

    async def _handle_detect_fraud(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle fraud detection request."""
        tx_data = message.payload.get('transaction_data', {})

        indicators = []
        is_fraud = False

        # Check for card testing pattern
        amount = tx_data.get('amount', 0)
        if amount < 100:  # Less than $1
            indicators.append("small_amount_test")

        # Check for mismatched billing/shipping
        billing_country = tx_data.get('billing_country', '')
        shipping_country = tx_data.get('shipping_country', '')
        if billing_country and shipping_country and billing_country != shipping_country:
            indicators.append("country_mismatch")

        # Check email for disposable domains
        email = tx_data.get('email', '')
        disposable_domains = ['tempmail.com', 'guerrillamail.com', '10minutemail.com']
        if any(domain in email for domain in disposable_domains):
            indicators.append("disposable_email")

        if len(indicators) >= 2:
            is_fraud = True

        return {
            'is_fraud': is_fraud,
            'confidence': 0.8 if is_fraud else 0.2,
            'indicators': indicators
        }

    async def _handle_add_blocklist(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle add to blocklist request."""
        block_type = message.payload.get('type')
        value = message.payload.get('value')

        if not block_type or not value:
            return {'error': 'Type and value are required'}

        if block_type == 'ip':
            self.blocklist_ips.add(value)
        elif block_type == 'card':
            # Store hash of card
            card_hash = hashlib.sha256(value.encode()).hexdigest()
            self.blocklist_cards.add(card_hash)
        elif block_type == 'email':
            self.blocklist_emails.add(value.lower())
        else:
            return {'error': f'Unknown blocklist type: {block_type}'}

        return {'success': True, 'type': block_type, 'value': value[:4] + '****'}

    async def _handle_check_blocklist(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle blocklist check request."""
        block_type = message.payload.get('type')
        value = message.payload.get('value')

        is_blocked = False

        if block_type == 'ip':
            is_blocked = value in self.blocklist_ips
        elif block_type == 'card':
            card_hash = hashlib.sha256(value.encode()).hexdigest()
            is_blocked = card_hash in self.blocklist_cards
        elif block_type == 'email':
            is_blocked = value.lower() in self.blocklist_emails

        return {'is_blocked': is_blocked, 'type': block_type}

    async def _handle_get_alerts(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get alerts request."""
        limit = message.payload.get('limit', 50)
        unresolved_only = message.payload.get('unresolved_only', False)

        alerts = self.alerts
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return {
            'alerts': [
                {
                    'id': a.id,
                    'type': a.alert_type.value,
                    'risk_level': a.risk_level.value,
                    'entity_id': a.entity_id,
                    'entity_type': a.entity_type,
                    'details': a.details,
                    'created_at': a.created_at.isoformat(),
                    'resolved': a.resolved
                }
                for a in alerts[-limit:]
            ],
            'total': len(alerts),
            'unresolved': len([a for a in alerts if not a.resolved])
        }

    def _create_alert(
        self,
        alert_type: AlertType,
        risk_level: RiskLevel,
        entity_id: str,
        entity_type: str,
        details: Dict[str, Any]
    ) -> SecurityAlert:
        """Create a security alert."""
        alert = SecurityAlert(
            id=hashlib.md5(
                f"{alert_type}{entity_id}{datetime.now()}".encode()
            ).hexdigest()[:12],
            alert_type=alert_type,
            risk_level=risk_level,
            entity_id=entity_id,
            entity_type=entity_type,
            details=details
        )
        self.alerts.append(alert)
        self.logger.warning(f"Security alert created: {alert.id} - {alert_type.value}")
        return alert
