"""
Unified Stripe Interface

Single entry point for all Stripe ecosystem operations,
integrating OpenCog knowledge, Agent-Zero orchestration,
and all platform bridges.
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

from opencog.atomspace import StripeAtomspace
from opencog.knowledge_base import StripeKnowledgeBase
from opencog.reasoning import StripeReasoner

from agent_zero.orchestrator import AgentOrchestrator
from agent_zero.agent_registry import AgentRegistry

from bridges.cognitive_bridge import CognitiveBridge
from bridges.sdk_bridge import SDKBridge
from bridges.plugin_bridge import PluginBridge
from bridges.api_bridge import APIBridge, RateLimitConfig

from .config import PlatformConfig, load_config
from .events import EventBus, PlatformEvent, EventCategory


class PlatformStatus(Enum):
    """Platform status."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class OperationResult:
    """Result of a platform operation."""
    success: bool
    operation: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'operation': self.operation,
            'data': self.data,
            'error': self.error,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat()
        }


class UnifiedStripeInterface:
    """
    Single entry point for all Stripe ecosystem operations.

    This interface unifies:
    - OpenCog knowledge representation and reasoning
    - Agent-Zero multi-agent orchestration
    - SDK management
    - Plugin management
    - API access with rate limiting
    - Event-driven communication
    """

    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or load_config()
        self.status = PlatformStatus.INITIALIZING
        self.logger = logging.getLogger("unified_interface")
        self.created_at = datetime.now()

        # Core components
        self.atomspace = StripeAtomspace(self.config.opencog.atomspace_name)
        self.knowledge_base = StripeKnowledgeBase(self.atomspace)
        self.reasoner = StripeReasoner(self.atomspace, self.knowledge_base)

        # Agent system
        self.orchestrator = AgentOrchestrator(f"{self.config.name}_orchestrator")
        self.agent_registry = AgentRegistry()

        # Bridges
        self.cognitive_bridge = CognitiveBridge(
            atomspace=self.atomspace,
            knowledge_base=self.knowledge_base,
            orchestrator=self.orchestrator
        )
        self.sdk_bridge = SDKBridge(base_path=self.config.base_path)
        self.plugin_bridge = PluginBridge(base_path=self.config.base_path)
        self.api_bridge = APIBridge(
            api_key=self.config.stripe.api_key,
            api_version=self.config.stripe.api_version,
            rate_limit_config=RateLimitConfig(
                requests_per_second=self.config.rate_limit.requests_per_second,
                burst_size=self.config.rate_limit.burst_size,
                max_retries=self.config.rate_limit.max_retries
            )
        )

        # Event bus
        self.event_bus = EventBus(f"{self.config.name}_events")

        # Registered agents storage
        self._agents: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """
        Initialize the unified platform.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Initializing {self.config.name}...")

            # Initialize knowledge base
            self.knowledge_base.initialize()
            self.logger.info("Knowledge base initialized")

            # Initialize bridges
            self.sdk_bridge.initialize()
            self.logger.info("SDK bridge initialized")

            self.plugin_bridge.initialize()
            self.logger.info("Plugin bridge initialized")

            # Initialize cognitive bridge
            await self.cognitive_bridge.initialize()
            self.logger.info("Cognitive bridge initialized")

            # Setup event handlers
            self._setup_event_handlers()

            self.status = PlatformStatus.READY
            self.logger.info(f"{self.config.name} initialized successfully")

            await self.event_bus.publish(self.event_bus.create_event(
                name="platform.initialized",
                category=EventCategory.SYSTEM,
                data={'config': self.config.to_dict()},
                source="unified_interface"
            ))

            return True

        except Exception as e:
            self.status = PlatformStatus.ERROR
            self.logger.error(f"Failed to initialize platform: {e}")
            return False

    async def start(self):
        """Start the platform and all agents."""
        if self.status != PlatformStatus.READY:
            await self.initialize()

        try:
            # Start orchestrator
            await self.orchestrator.start()
            self.status = PlatformStatus.RUNNING

            self.logger.info(f"{self.config.name} started")

            await self.event_bus.publish(self.event_bus.create_event(
                name="platform.started",
                category=EventCategory.SYSTEM,
                source="unified_interface"
            ))

        except Exception as e:
            self.status = PlatformStatus.ERROR
            self.logger.error(f"Failed to start platform: {e}")
            raise

    async def stop(self):
        """Stop the platform gracefully."""
        self.status = PlatformStatus.SHUTTING_DOWN

        try:
            await self.orchestrator.stop()
            self.status = PlatformStatus.STOPPED

            self.logger.info(f"{self.config.name} stopped")

            await self.event_bus.publish(self.event_bus.create_event(
                name="platform.stopped",
                category=EventCategory.SYSTEM,
                source="unified_interface"
            ))

        except Exception as e:
            self.logger.error(f"Error stopping platform: {e}")
            raise

    def _setup_event_handlers(self):
        """Setup internal event handlers."""
        # Handle payment events
        self.event_bus.subscribe(
            pattern="payment.*",
            handler=self._on_payment_event,
            category=EventCategory.PAYMENT
        )

        # Handle agent events
        self.event_bus.subscribe(
            pattern="agent.*",
            handler=self._on_agent_event,
            category=EventCategory.AGENT
        )

        # Handle error events
        self.event_bus.subscribe(
            pattern="error.*",
            handler=self._on_error_event,
            category=EventCategory.ERROR
        )

    async def _on_payment_event(self, event: PlatformEvent):
        """Handle payment events."""
        # Update knowledge base with payment information
        from agent_zero.base_agent import AgentMessage, MessageType
        message = AgentMessage(
            sender_id="event_bus",
            action=event.name,
            payload=event.data
        )
        await self.cognitive_bridge.process_agent_message(message)

    async def _on_agent_event(self, event: PlatformEvent):
        """Handle agent events."""
        self.logger.debug(f"Agent event: {event.name}")

    async def _on_error_event(self, event: PlatformEvent):
        """Handle error events."""
        self.logger.error(f"Platform error: {event.name} - {event.data}")

    # ============================================================
    # Payment Operations
    # ============================================================

    async def create_payment(
        self,
        amount: int,
        currency: str,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """
        Create a payment intent.

        Args:
            amount: Amount in cents
            currency: Currency code (e.g., 'usd')
            customer_id: Optional customer ID
            metadata: Optional metadata

        Returns:
            Operation result
        """
        import time
        start = time.time()

        try:
            response = await self.api_bridge.create_payment_intent(
                amount=amount,
                currency=currency,
                customer=customer_id,
                metadata=metadata or {}
            )

            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                # Emit payment event
                await self.event_bus.emit_payment_event(
                    event_name="created",
                    payment_data={
                        'payment_id': response.data.get('id'),
                        'amount': amount,
                        'currency': currency,
                        'customer_id': customer_id
                    },
                    source="unified_interface"
                )

                return OperationResult(
                    success=True,
                    operation="create_payment",
                    data=response.data,
                    duration_ms=duration
                )
            else:
                return OperationResult(
                    success=False,
                    operation="create_payment",
                    error=str(response.error),
                    duration_ms=duration
                )

        except Exception as e:
            return OperationResult(
                success=False,
                operation="create_payment",
                error=str(e)
            )

    async def retrieve_payment(self, payment_id: str) -> OperationResult:
        """Retrieve a payment intent."""
        import time
        start = time.time()

        try:
            response = await self.api_bridge.retrieve_payment_intent(payment_id)
            duration = (time.time() - start) * 1000

            return OperationResult(
                success=response.status_code == 200,
                operation="retrieve_payment",
                data=response.data if response.status_code == 200 else {},
                error=str(response.error) if response.error else None,
                duration_ms=duration
            )

        except Exception as e:
            return OperationResult(
                success=False,
                operation="retrieve_payment",
                error=str(e)
            )

    # ============================================================
    # Customer Operations
    # ============================================================

    async def create_customer(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """Create a customer."""
        import time
        start = time.time()

        try:
            params = {}
            if email:
                params['email'] = email
            if name:
                params['name'] = name
            if metadata:
                params['metadata'] = metadata

            response = await self.api_bridge.create_customer(**params)
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                await self.event_bus.emit_customer_event(
                    event_name="created",
                    customer_data={
                        'customer_id': response.data.get('id'),
                        'email': email,
                        'name': name
                    },
                    source="unified_interface"
                )

                return OperationResult(
                    success=True,
                    operation="create_customer",
                    data=response.data,
                    duration_ms=duration
                )
            else:
                return OperationResult(
                    success=False,
                    operation="create_customer",
                    error=str(response.error),
                    duration_ms=duration
                )

        except Exception as e:
            return OperationResult(
                success=False,
                operation="create_customer",
                error=str(e)
            )

    async def list_customers(
        self,
        limit: int = 10
    ) -> OperationResult:
        """List customers."""
        import time
        start = time.time()

        try:
            response = await self.api_bridge.list_customers(limit=limit)
            duration = (time.time() - start) * 1000

            return OperationResult(
                success=response.status_code == 200,
                operation="list_customers",
                data=response.data if response.status_code == 200 else {},
                error=str(response.error) if response.error else None,
                duration_ms=duration
            )

        except Exception as e:
            return OperationResult(
                success=False,
                operation="list_customers",
                error=str(e)
            )

    # ============================================================
    # Knowledge Operations
    # ============================================================

    async def query_knowledge(
        self,
        query_type: str,
        params: Dict[str, Any]
    ) -> OperationResult:
        """
        Query the knowledge base.

        Args:
            query_type: Type of query (entity, relationships, risk_assessment, sdk_recommendation)
            params: Query parameters

        Returns:
            Operation result
        """
        import time
        start = time.time()

        try:
            result = await self.cognitive_bridge.query_knowledge(
                query_type, params
            )
            duration = (time.time() - start) * 1000

            return OperationResult(
                success='error' not in result,
                operation=f"query_knowledge:{query_type}",
                data=result,
                duration_ms=duration
            )

        except Exception as e:
            return OperationResult(
                success=False,
                operation=f"query_knowledge:{query_type}",
                error=str(e)
            )

    async def assess_risk(self, payment_id: str) -> OperationResult:
        """Assess risk for a payment."""
        return await self.query_knowledge(
            query_type="risk_assessment",
            params={'payment_id': payment_id}
        )

    async def recommend_sdk(
        self,
        language: Optional[str] = None,
        features: Optional[List[str]] = None
    ) -> OperationResult:
        """Get SDK recommendation."""
        return await self.query_knowledge(
            query_type="sdk_recommendation",
            params={
                'language': language,
                'features': features or []
            }
        )

    # ============================================================
    # SDK Operations
    # ============================================================

    def get_available_sdks(self) -> List[Dict[str, Any]]:
        """Get all available SDKs."""
        sdks = self.sdk_bridge.get_available_sdks()
        return [
            {
                'name': sdk.name,
                'language': sdk.language.value,
                'version': sdk.version,
                'features': sdk.features
            }
            for sdk in sdks
        ]

    def get_code_example(
        self,
        sdk_name: str,
        capability: str
    ) -> Optional[str]:
        """Get code example for a capability."""
        return self.sdk_bridge.get_code_example(sdk_name, capability)

    # ============================================================
    # Plugin Operations
    # ============================================================

    def get_available_plugins(self) -> List[Dict[str, Any]]:
        """Get all available plugins."""
        plugins = list(self.plugin_bridge.plugins.values())
        return [
            {
                'name': plugin.name,
                'platform': plugin.platform.value,
                'status': plugin.status.value,
                'capabilities': plugin.capabilities
            }
            for plugin in plugins
        ]

    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a plugin."""
        return await self.plugin_bridge.activate_plugin(plugin_name)

    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin."""
        return await self.plugin_bridge.deactivate_plugin(plugin_name)

    # ============================================================
    # Agent Operations
    # ============================================================

    def register_agent(self, agent: Any) -> str:
        """Register an agent with the platform."""
        self.orchestrator.register_agent(agent)
        self.agent_registry.register(agent)
        self._agents[agent.agent_id] = agent
        return agent.agent_id

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status."""
        agent = self._agents.get(agent_id)
        if agent:
            return agent.get_status()
        return None

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get status of all agents."""
        return [agent.get_status() for agent in self._agents.values()]

    # ============================================================
    # Platform Status
    # ============================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        return {
            'name': self.config.name,
            'version': self.config.version,
            'environment': self.config.environment.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds(),
            'components': {
                'knowledge_base': self.knowledge_base.get_statistics(),
                'orchestrator': self.orchestrator.get_status() if hasattr(self.orchestrator, 'get_status') else {},
                'cognitive_bridge': self.cognitive_bridge.get_statistics(),
                'sdk_bridge': self.sdk_bridge.get_statistics(),
                'plugin_bridge': self.plugin_bridge.get_statistics(),
                'api_bridge': self.api_bridge.get_statistics(),
                'event_bus': self.event_bus.get_statistics()
            },
            'agents': {
                'total': len(self._agents),
                'registry': self.agent_registry.get_statistics()
            },
            'features': self.config.features
        }

    def get_health(self) -> Dict[str, Any]:
        """Get platform health status."""
        health = {
            'status': 'healthy' if self.status == PlatformStatus.RUNNING else 'unhealthy',
            'checks': {
                'knowledge_base': self.atomspace is not None,
                'orchestrator': self.orchestrator is not None,
                'sdk_bridge': self.sdk_bridge._initialized,
                'plugin_bridge': self.plugin_bridge._initialized,
                'api_bridge': bool(self.api_bridge.api_key) if self.api_bridge else False
            }
        }

        health['all_checks_passed'] = all(health['checks'].values())
        return health
