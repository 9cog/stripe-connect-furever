"""
Monitoring Agent

Specialized agent for system health monitoring.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import psutil

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentCapability
)


class SystemStatus(Enum):
    """System status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringAlert:
    """Monitoring alert."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MonitoringAgent(BaseAgent):
    """
    Agent for system health monitoring.

    Capabilities:
    - Monitor system resources
    - Track component health
    - Generate alerts
    - Collect metrics
    """

    # Alert thresholds
    THRESHOLDS = {
        'cpu_warning': 70.0,
        'cpu_critical': 90.0,
        'memory_warning': 80.0,
        'memory_critical': 95.0,
        'disk_warning': 85.0,
        'disk_critical': 95.0
    }

    def __init__(self, check_interval_seconds: int = 30):
        super().__init__(
            name="MonitoringAgent",
            description="System health monitoring"
        )
        self.check_interval = check_interval_seconds

        # Monitoring data
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[MonitoringAlert] = []
        self.component_status: Dict[str, SystemStatus] = {}

        # Callbacks
        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._max_history = 1000

        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        self.register_capability(AgentCapability(
            name="get_system_health",
            description="Get current system health status",
            output_schema={"status": "string", "metrics": "object"}
        ))

        self.register_capability(AgentCapability(
            name="get_metrics_history",
            description="Get metrics history",
            input_schema={"minutes": "integer"},
            output_schema={"metrics": "array"}
        ))

        self.register_capability(AgentCapability(
            name="get_alerts",
            description="Get monitoring alerts",
            input_schema={"unresolved_only": "boolean"},
            output_schema={"alerts": "array"}
        ))

        self.register_capability(AgentCapability(
            name="set_component_status",
            description="Set status for a component",
            input_schema={"component": "string", "status": "string"}
        ))

        self.register_capability(AgentCapability(
            name="start_monitoring",
            description="Start continuous monitoring"
        ))

        self.register_capability(AgentCapability(
            name="stop_monitoring",
            description="Stop continuous monitoring"
        ))

        # Register handlers
        self.register_handler("get_system_health", self._handle_get_health)
        self.register_handler("get_metrics_history", self._handle_get_history)
        self.register_handler("get_alerts", self._handle_get_alerts)
        self.register_handler("set_component_status", self._handle_set_status)
        self.register_handler("start_monitoring", self._handle_start)
        self.register_handler("stop_monitoring", self._handle_stop)

    async def initialize(self) -> bool:
        """Initialize the monitoring agent."""
        self.logger.info("Initializing MonitoringAgent")

        # Collect initial metrics
        await self._collect_metrics()

        return True

    async def shutdown(self):
        """Shutdown the monitoring agent."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        self.logger.info("Shutting down MonitoringAgent")

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

    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                active_connections=len(psutil.net_connections())
            )
        except Exception as e:
            # Fallback if psutil fails
            self.logger.warning(f"Error collecting metrics: {e}")
            metrics = SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                active_connections=0
            )

        self.metrics_history.append(metrics)

        # Trim history
        if len(self.metrics_history) > self._max_history:
            self.metrics_history = self.metrics_history[-self._max_history:]

        # Check thresholds
        await self._check_thresholds(metrics)

        return metrics

    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and create alerts."""
        # CPU checks
        if metrics.cpu_percent >= self.THRESHOLDS['cpu_critical']:
            self._create_alert(
                AlertSeverity.CRITICAL,
                'cpu',
                f'CPU usage critical: {metrics.cpu_percent}%'
            )
        elif metrics.cpu_percent >= self.THRESHOLDS['cpu_warning']:
            self._create_alert(
                AlertSeverity.WARNING,
                'cpu',
                f'CPU usage high: {metrics.cpu_percent}%'
            )

        # Memory checks
        if metrics.memory_percent >= self.THRESHOLDS['memory_critical']:
            self._create_alert(
                AlertSeverity.CRITICAL,
                'memory',
                f'Memory usage critical: {metrics.memory_percent}%'
            )
        elif metrics.memory_percent >= self.THRESHOLDS['memory_warning']:
            self._create_alert(
                AlertSeverity.WARNING,
                'memory',
                f'Memory usage high: {metrics.memory_percent}%'
            )

        # Disk checks
        if metrics.disk_percent >= self.THRESHOLDS['disk_critical']:
            self._create_alert(
                AlertSeverity.CRITICAL,
                'disk',
                f'Disk usage critical: {metrics.disk_percent}%'
            )
        elif metrics.disk_percent >= self.THRESHOLDS['disk_warning']:
            self._create_alert(
                AlertSeverity.WARNING,
                'disk',
                f'Disk usage high: {metrics.disk_percent}%'
            )

    def _create_alert(
        self,
        severity: AlertSeverity,
        component: str,
        message: str
    ):
        """Create a monitoring alert."""
        # Check for duplicate recent alerts
        recent_alerts = [
            a for a in self.alerts
            if a.component == component
            and not a.resolved
            and (datetime.now() - a.created_at).seconds < 300
        ]

        if recent_alerts:
            return  # Don't duplicate

        import hashlib
        alert = MonitoringAlert(
            id=hashlib.md5(
                f"{component}{message}{datetime.now()}".encode()
            ).hexdigest()[:12],
            severity=severity,
            component=component,
            message=message
        )

        self.alerts.append(alert)
        self.logger.warning(f"Alert created: [{severity.value}] {component}: {message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _handle_get_health(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get health request."""
        metrics = await self._collect_metrics()

        # Determine overall status
        if metrics.cpu_percent >= self.THRESHOLDS['cpu_critical'] or \
           metrics.memory_percent >= self.THRESHOLDS['memory_critical']:
            status = SystemStatus.CRITICAL
        elif metrics.cpu_percent >= self.THRESHOLDS['cpu_warning'] or \
             metrics.memory_percent >= self.THRESHOLDS['memory_warning']:
            status = SystemStatus.WARNING
        else:
            status = SystemStatus.HEALTHY

        return {
            'status': status.value,
            'metrics': {
                'cpu_percent': round(metrics.cpu_percent, 2),
                'memory_percent': round(metrics.memory_percent, 2),
                'disk_percent': round(metrics.disk_percent, 2),
                'active_connections': metrics.active_connections,
                'timestamp': metrics.timestamp.isoformat()
            },
            'components': {
                name: s.value for name, s in self.component_status.items()
            },
            'unresolved_alerts': len([a for a in self.alerts if not a.resolved])
        }

    async def _handle_get_history(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get history request."""
        minutes = message.payload.get('minutes', 60)
        cutoff = datetime.now() - timedelta(minutes=minutes)

        filtered = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff
        ]

        return {
            'metrics': [
                {
                    'cpu_percent': round(m.cpu_percent, 2),
                    'memory_percent': round(m.memory_percent, 2),
                    'disk_percent': round(m.disk_percent, 2),
                    'active_connections': m.active_connections,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in filtered
            ],
            'count': len(filtered),
            'time_range_minutes': minutes
        }

    async def _handle_get_alerts(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get alerts request."""
        unresolved_only = message.payload.get('unresolved_only', False)
        limit = message.payload.get('limit', 50)

        alerts = self.alerts
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return {
            'alerts': [
                {
                    'id': a.id,
                    'severity': a.severity.value,
                    'component': a.component,
                    'message': a.message,
                    'created_at': a.created_at.isoformat(),
                    'resolved': a.resolved
                }
                for a in alerts[-limit:]
            ],
            'total': len(alerts),
            'unresolved': len([a for a in self.alerts if not a.resolved])
        }

    async def _handle_set_status(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle set component status request."""
        component = message.payload.get('component')
        status = message.payload.get('status')

        if not component or not status:
            return {'error': 'Component and status required'}

        try:
            self.component_status[component] = SystemStatus(status)
            return {'success': True, 'component': component, 'status': status}
        except ValueError:
            return {'error': f'Invalid status: {status}'}

    async def _handle_start(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle start monitoring request."""
        if self._monitoring_task and not self._monitoring_task.done():
            return {'status': 'already_running'}

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        return {'status': 'started', 'interval_seconds': self.check_interval}

    async def _handle_stop(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle stop monitoring request."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
            return {'status': 'stopped'}

        return {'status': 'not_running'}

    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]):
        """Add callback for new alerts."""
        self.alert_callbacks.append(callback)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False
