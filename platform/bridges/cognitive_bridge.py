"""
Cognitive Bridge

Connects OpenCog Atomspace knowledge representation with
Agent-Zero multi-agent orchestration system.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencog.atomspace import StripeAtomspace, AtomType, Node, Link
from opencog.knowledge_base import StripeKnowledgeBase
from opencog.reasoning import StripeReasoner, ReasoningResult

from agent_zero.base_agent import BaseAgent, AgentMessage, MessageType
from agent_zero.orchestrator import AgentOrchestrator


class BridgeEventType(Enum):
    """Types of bridge events."""
    KNOWLEDGE_UPDATED = "knowledge_updated"
    AGENT_REGISTERED = "agent_registered"
    REASONING_COMPLETED = "reasoning_completed"
    SYNC_COMPLETED = "sync_completed"


@dataclass
class BridgeEvent:
    """Event from the cognitive bridge."""
    event_type: BridgeEventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class CognitiveBridge:
    """
    Bridge between OpenCog knowledge system and Agent-Zero orchestration.

    Provides:
    - Bidirectional knowledge flow
    - Agent-to-knowledge synchronization
    - Reasoning integration
    - Event-driven updates
    """

    def __init__(
        self,
        atomspace: Optional[StripeAtomspace] = None,
        knowledge_base: Optional[StripeKnowledgeBase] = None,
        orchestrator: Optional[AgentOrchestrator] = None
    ):
        self.atomspace = atomspace or StripeAtomspace("cognitive_bridge")
        self.knowledge_base = knowledge_base or StripeKnowledgeBase(self.atomspace)
        self.orchestrator = orchestrator
        self.reasoner = StripeReasoner(self.atomspace, self.knowledge_base)
        self.logger = logging.getLogger("cognitive_bridge")
        self.event_listeners: List[Callable[[BridgeEvent], None]] = []
        self._initialized = False

        # Sync state
        self._last_sync = None
        self._sync_interval_seconds = 60

    async def initialize(self) -> bool:
        """
        Initialize the cognitive bridge.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize knowledge base
            self.knowledge_base.initialize()

            # Register agent knowledge nodes if orchestrator is available
            if self.orchestrator:
                await self._sync_agents_to_knowledge()

            self._initialized = True
            self.logger.info("Cognitive bridge initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive bridge: {e}")
            return False

    async def _sync_agents_to_knowledge(self):
        """Synchronize agent information to knowledge base."""
        if not self.orchestrator:
            return

        for agent_id, agent in self.orchestrator.agents.items():
            # Create agent node
            agent_node = self.atomspace.add_node(
                AtomType.AGENT,
                agent_id,
                value={
                    'name': agent.name,
                    'description': agent.description,
                    'state': agent.state.value
                }
            )

            # Add capabilities as links
            for cap_name, capability in agent.capabilities.items():
                cap_node = self.atomspace.add_node(
                    AtomType.CONCEPT,
                    f"capability_{cap_name}"
                )
                self.atomspace.add_link(
                    AtomType.EVALUATION,
                    [
                        self.atomspace.add_node(AtomType.PREDICATE, "has_capability"),
                        self.atomspace.add_link(AtomType.LIST, [agent_node, cap_node])
                    ]
                )

        self._last_sync = datetime.now()
        self._emit_event(BridgeEvent(
            event_type=BridgeEventType.SYNC_COMPLETED,
            source="cognitive_bridge",
            data={'agents_synced': len(self.orchestrator.agents)}
        ))

    def set_orchestrator(self, orchestrator: AgentOrchestrator):
        """
        Set or update the orchestrator reference.

        Args:
            orchestrator: The agent orchestrator
        """
        self.orchestrator = orchestrator
        self.logger.info("Orchestrator reference updated")

    async def process_agent_message(
        self,
        message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """
        Process a message from an agent and update knowledge.

        Args:
            message: The agent message

        Returns:
            Knowledge update result
        """
        result = {}

        # Extract knowledge from message
        if message.action.startswith("payment_"):
            result = await self._process_payment_knowledge(message)
        elif message.action.startswith("customer_"):
            result = await self._process_customer_knowledge(message)
        elif message.action.startswith("risk_"):
            result = await self._process_risk_knowledge(message)

        self._emit_event(BridgeEvent(
            event_type=BridgeEventType.KNOWLEDGE_UPDATED,
            source=message.sender_id,
            data={'action': message.action, 'result': result}
        ))

        return result

    async def _process_payment_knowledge(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Process payment-related knowledge from message."""
        payload = message.payload
        payment_id = payload.get('payment_id', message.id)

        # Create or update payment node
        payment_node = self.atomspace.add_node(
            AtomType.PAYMENT,
            payment_id,
            value={
                'amount': payload.get('amount'),
                'currency': payload.get('currency'),
                'status': payload.get('status')
            },
            metadata={
                'source_agent': message.sender_id,
                'processed_at': datetime.now().isoformat()
            }
        )

        # Link to customer if available
        customer_id = payload.get('customer_id')
        if customer_id:
            customer_node = self.atomspace.add_node(
                AtomType.CUSTOMER,
                customer_id
            )
            self.atomspace.add_link(
                AtomType.PAYMENT_CUSTOMER,
                [payment_node, customer_node]
            )

        return {
            'node_id': payment_node.id,
            'payment_id': payment_id,
            'status': 'knowledge_updated'
        }

    async def _process_customer_knowledge(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Process customer-related knowledge from message."""
        payload = message.payload
        customer_id = payload.get('customer_id', message.id)

        customer_node = self.atomspace.add_node(
            AtomType.CUSTOMER,
            customer_id,
            value={
                'email': payload.get('email'),
                'name': payload.get('name')
            },
            metadata={
                'source_agent': message.sender_id,
                'processed_at': datetime.now().isoformat()
            }
        )

        return {
            'node_id': customer_node.id,
            'customer_id': customer_id,
            'status': 'knowledge_updated'
        }

    async def _process_risk_knowledge(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Process risk-related knowledge from message."""
        payload = message.payload

        # Create risk assessment node
        risk_node = self.atomspace.add_node(
            AtomType.CONCEPT,
            f"risk_assessment_{message.id}",
            value={
                'risk_level': payload.get('risk_level'),
                'risk_score': payload.get('risk_score'),
                'factors': payload.get('factors', [])
            },
            truth_value=(
                payload.get('confidence', 1.0),
                payload.get('certainty', 1.0)
            ),
            metadata={
                'source_agent': message.sender_id,
                'processed_at': datetime.now().isoformat()
            }
        )

        # Link to related entity if available
        entity_id = payload.get('entity_id')
        entity_type = payload.get('entity_type', 'payment')
        if entity_id:
            entity_atom_type = {
                'payment': AtomType.PAYMENT,
                'customer': AtomType.CUSTOMER,
                'account': AtomType.ACCOUNT
            }.get(entity_type, AtomType.CONCEPT)

            entity_node = self.atomspace.add_node(entity_atom_type, entity_id)
            self.atomspace.add_link(
                AtomType.EVALUATION,
                [
                    self.atomspace.add_node(AtomType.PREDICATE, "has_risk"),
                    self.atomspace.add_link(AtomType.LIST, [entity_node, risk_node])
                ]
            )

        return {
            'node_id': risk_node.id,
            'risk_level': payload.get('risk_level'),
            'status': 'knowledge_updated'
        }

    async def query_knowledge(
        self,
        query_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query the knowledge base.

        Args:
            query_type: Type of query
            params: Query parameters

        Returns:
            Query results
        """
        if query_type == "entity":
            return await self._query_entity(params)
        elif query_type == "relationships":
            return await self._query_relationships(params)
        elif query_type == "risk_assessment":
            return await self._query_risk_assessment(params)
        elif query_type == "sdk_recommendation":
            return await self._query_sdk_recommendation(params)
        else:
            return {'error': f'Unknown query type: {query_type}'}

    async def _query_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query entity information."""
        entity_type = params.get('entity_type', 'payment')
        entity_id = params.get('entity_id')

        atom_type = {
            'payment': AtomType.PAYMENT,
            'customer': AtomType.CUSTOMER,
            'account': AtomType.ACCOUNT,
            'subscription': AtomType.SUBSCRIPTION
        }.get(entity_type, AtomType.CONCEPT)

        if entity_id:
            node = self.atomspace.get_node(atom_type, entity_id)
            if node:
                return {
                    'found': True,
                    'entity': node.to_dict(),
                    'incoming_links': len(self.atomspace.get_incoming(node))
                }
            return {'found': False, 'entity_id': entity_id}

        # Return all entities of type
        entities = self.atomspace.get_atoms_by_type(atom_type)
        return {
            'count': len(entities),
            'entities': [e.to_dict() for e in entities[:100]]
        }

    async def _query_relationships(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query entity relationships."""
        entity_type = params.get('entity_type')

        result = self.reasoner.infer_entity_relationships(entity_type)
        return {
            'entity_type': entity_type,
            'conclusion': result.conclusion,
            'evidence': result.evidence,
            'recommendations': result.recommendations
        }

    async def _query_risk_assessment(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query risk assessment for an entity."""
        payment_id = params.get('payment_id')

        if not payment_id:
            return {'error': 'payment_id required'}

        result = self.reasoner.assess_payment_risk(payment_id)
        return {
            'payment_id': payment_id,
            'conclusion': result.conclusion,
            'confidence': result.confidence.value,
            'evidence': result.evidence,
            'recommendations': result.recommendations,
            'metadata': result.metadata
        }

    async def _query_sdk_recommendation(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query SDK recommendation."""
        result = self.reasoner.recommend_sdk(params)
        return {
            'conclusion': result.conclusion,
            'confidence': result.confidence.value,
            'evidence': result.evidence,
            'recommendations': result.recommendations,
            'matches': result.metadata.get('matches', [])
        }

    async def run_inference(self) -> List[Dict[str, Any]]:
        """
        Run forward chaining inference.

        Returns:
            List of newly inferred atoms
        """
        new_atoms = self.reasoner.run_forward_chaining()

        self._emit_event(BridgeEvent(
            event_type=BridgeEventType.REASONING_COMPLETED,
            source="cognitive_bridge",
            data={'new_atoms_count': len(new_atoms)}
        ))

        return [atom.to_dict() for atom in new_atoms]

    def add_event_listener(
        self,
        listener: Callable[[BridgeEvent], None]
    ):
        """Add an event listener."""
        self.event_listeners.append(listener)

    def remove_event_listener(
        self,
        listener: Callable[[BridgeEvent], None]
    ):
        """Remove an event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)

    def _emit_event(self, event: BridgeEvent):
        """Emit an event to all listeners."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Error in event listener: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'initialized': self._initialized,
            'last_sync': (
                self._last_sync.isoformat() if self._last_sync else None
            ),
            'atomspace_stats': self.atomspace.get_statistics(),
            'knowledge_base_stats': self.knowledge_base.get_statistics(),
            'reasoner_stats': self.reasoner.get_statistics(),
            'orchestrator_connected': self.orchestrator is not None
        }
