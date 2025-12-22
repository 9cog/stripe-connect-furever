"""
Integration Agent

Specialized agent for SDK/plugin health monitoring and integration management.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentCapability
)


class HealthStatus(Enum):
    """Health status for integrations."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class IntegrationHealth:
    """Health record for an integration."""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int
    details: Dict[str, Any]


class IntegrationAgent(BaseAgent):
    """
    Agent for SDK/plugin health monitoring.

    Capabilities:
    - Monitor SDK health
    - Monitor plugin health
    - Check integration status
    - Report health metrics
    """

    def __init__(
        self,
        sdk_bridge=None,
        plugin_bridge=None,
        check_interval_seconds: int = 60
    ):
        super().__init__(
            name="IntegrationAgent",
            description="Monitors SDK and plugin health"
        )
        self.sdk_bridge = sdk_bridge
        self.plugin_bridge = plugin_bridge
        self.check_interval = check_interval_seconds

        # Health tracking
        self.health_records: Dict[str, IntegrationHealth] = {}
        self._monitoring_task: Optional[asyncio.Task] = None

        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        self.register_capability(AgentCapability(
            name="check_sdk_health",
            description="Check health of an SDK",
            input_schema={"sdk_name": "string (required)"},
            output_schema={"status": "string", "details": "object"}
        ))

        self.register_capability(AgentCapability(
            name="check_plugin_health",
            description="Check health of a plugin",
            input_schema={"plugin_name": "string (required)"},
            output_schema={"status": "string", "details": "object"}
        ))

        self.register_capability(AgentCapability(
            name="get_all_health",
            description="Get health status of all integrations",
            output_schema={"integrations": "array"}
        ))

        self.register_capability(AgentCapability(
            name="start_monitoring",
            description="Start continuous health monitoring"
        ))

        self.register_capability(AgentCapability(
            name="stop_monitoring",
            description="Stop continuous health monitoring"
        ))

        # Register handlers
        self.register_handler("check_sdk_health", self._handle_check_sdk)
        self.register_handler("check_plugin_health", self._handle_check_plugin)
        self.register_handler("get_all_health", self._handle_get_all_health)
        self.register_handler("start_monitoring", self._handle_start_monitoring)
        self.register_handler("stop_monitoring", self._handle_stop_monitoring)

    async def initialize(self) -> bool:
        """Initialize the integration agent."""
        self.logger.info("Initializing IntegrationAgent")

        # Perform initial health checks
        await self._perform_health_checks()

        return True

    async def shutdown(self):
        """Shutdown the integration agent."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        self.logger.info("Shutting down IntegrationAgent")

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

    async def _perform_health_checks(self):
        """Perform health checks on all integrations."""
        # Check SDKs
        if self.sdk_bridge:
            for sdk in self.sdk_bridge.get_available_sdks():
                await self._check_sdk_health(sdk.name)

        # Check plugins
        if self.plugin_bridge:
            for plugin in self.plugin_bridge.plugins.values():
                await self._check_plugin_health(plugin.name)

    async def _check_sdk_health(self, sdk_name: str) -> IntegrationHealth:
        """Check health of a specific SDK."""
        start_time = datetime.now()

        try:
            if not self.sdk_bridge:
                status = HealthStatus.UNKNOWN
                details = {'error': 'SDK bridge not available'}
            else:
                sdk = self.sdk_bridge.get_sdk(sdk_name)
                if sdk:
                    # Check if SDK is available
                    if sdk.status.value == 'available':
                        status = HealthStatus.HEALTHY
                        details = {
                            'language': sdk.language.value,
                            'version': sdk.version,
                            'features': sdk.features
                        }
                    else:
                        status = HealthStatus.UNHEALTHY
                        details = {'sdk_status': sdk.status.value}
                else:
                    status = HealthStatus.UNKNOWN
                    details = {'error': 'SDK not found'}

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            health = IntegrationHealth(
                name=f"sdk:{sdk_name}",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=0 if status == HealthStatus.HEALTHY else 1,
                details=details
            )

            self.health_records[health.name] = health
            return health

        except Exception as e:
            health = IntegrationHealth(
                name=f"sdk:{sdk_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={'error': str(e)}
            )
            self.health_records[health.name] = health
            return health

    async def _check_plugin_health(self, plugin_name: str) -> IntegrationHealth:
        """Check health of a specific plugin."""
        start_time = datetime.now()

        try:
            if not self.plugin_bridge:
                status = HealthStatus.UNKNOWN
                details = {'error': 'Plugin bridge not available'}
            else:
                plugin = self.plugin_bridge.get_plugin(plugin_name)
                if plugin:
                    if plugin.status.value in ['active', 'inactive']:
                        status = HealthStatus.HEALTHY
                        details = {
                            'platform': plugin.platform.value,
                            'status': plugin.status.value,
                            'capabilities': plugin.capabilities
                        }
                    elif plugin.status.value == 'error':
                        status = HealthStatus.UNHEALTHY
                        details = {'plugin_status': plugin.status.value}
                    else:
                        status = HealthStatus.DEGRADED
                        details = {'plugin_status': plugin.status.value}
                else:
                    status = HealthStatus.UNKNOWN
                    details = {'error': 'Plugin not found'}

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            health = IntegrationHealth(
                name=f"plugin:{plugin_name}",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=0 if status == HealthStatus.HEALTHY else 1,
                details=details
            )

            self.health_records[health.name] = health
            return health

        except Exception as e:
            health = IntegrationHealth(
                name=f"plugin:{plugin_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={'error': str(e)}
            )
            self.health_records[health.name] = health
            return health

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _handle_check_sdk(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle SDK health check request."""
        sdk_name = message.payload.get('sdk_name')
        if not sdk_name:
            return {'error': 'SDK name is required'}

        health = await self._check_sdk_health(sdk_name)
        return {
            'name': health.name,
            'status': health.status.value,
            'last_check': health.last_check.isoformat(),
            'response_time_ms': health.response_time_ms,
            'details': health.details
        }

    async def _handle_check_plugin(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle plugin health check request."""
        plugin_name = message.payload.get('plugin_name')
        if not plugin_name:
            return {'error': 'Plugin name is required'}

        health = await self._check_plugin_health(plugin_name)
        return {
            'name': health.name,
            'status': health.status.value,
            'last_check': health.last_check.isoformat(),
            'response_time_ms': health.response_time_ms,
            'details': health.details
        }

    async def _handle_get_all_health(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get all health request."""
        return {
            'integrations': [
                {
                    'name': h.name,
                    'status': h.status.value,
                    'last_check': h.last_check.isoformat(),
                    'response_time_ms': h.response_time_ms,
                    'error_count': h.error_count
                }
                for h in self.health_records.values()
            ],
            'summary': {
                'total': len(self.health_records),
                'healthy': len([
                    h for h in self.health_records.values()
                    if h.status == HealthStatus.HEALTHY
                ]),
                'unhealthy': len([
                    h for h in self.health_records.values()
                    if h.status == HealthStatus.UNHEALTHY
                ])
            }
        }

    async def _handle_start_monitoring(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle start monitoring request."""
        if self._monitoring_task and not self._monitoring_task.done():
            return {'status': 'already_running'}

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        return {'status': 'started', 'interval_seconds': self.check_interval}

    async def _handle_stop_monitoring(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle stop monitoring request."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
            return {'status': 'stopped'}

        return {'status': 'not_running'}
