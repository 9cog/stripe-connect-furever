"""
Base Agent Framework

Provides the foundation for all specialized agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import logging


class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentPriority(Enum):
    """Priority levels for agent tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    EVENT = "event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentCapability:
    """Defines a capability that an agent provides."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = True


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: AgentPriority = AgentPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'action': self.action,
            'payload': self.payload,
            'correlation_id': self.correlation_id,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    def create_response(
        self,
        payload: Dict[str, Any],
        is_error: bool = False
    ) -> 'AgentMessage':
        """Create a response message."""
        return AgentMessage(
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            message_type=MessageType.ERROR if is_error else MessageType.RESPONSE,
            action=f"{self.action}_response",
            payload=payload,
            correlation_id=self.id,
            priority=self.priority
        )


@dataclass
class AgentMetrics:
    """Metrics for agent performance monitoring."""
    messages_received: int = 0
    messages_sent: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time_ms: float = 0.0
    last_active: Optional[datetime] = None
    uptime_seconds: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    Provides:
    - Message handling infrastructure
    - State management
    - Capability registration
    - Lifecycle management
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "BaseAgent",
        description: str = ""
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.capabilities: Dict[str, AgentCapability] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{name}")
        self.created_at = datetime.now()
        self._running = False
        self._subscribers: Set[str] = set()
        self._context: Dict[str, Any] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the agent gracefully."""
        pass

    def register_capability(self, capability: AgentCapability):
        """Register a capability for this agent."""
        self.capabilities[capability.name] = capability
        self.logger.info(f"Registered capability: {capability.name}")

    def register_handler(
        self,
        action: str,
        handler: Callable[[AgentMessage], Any]
    ):
        """Register a message handler for a specific action."""
        self.message_handlers[action] = handler
        self.logger.debug(f"Registered handler for action: {action}")

    async def send_message(
        self,
        recipient_id: str,
        action: str,
        payload: Dict[str, Any],
        priority: AgentPriority = AgentPriority.MEDIUM
    ) -> AgentMessage:
        """
        Create and queue a message for sending.

        Args:
            recipient_id: Target agent ID
            action: Action to perform
            payload: Message payload
            priority: Message priority

        Returns:
            The created message
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            action=action,
            payload=payload,
            priority=priority
        )
        self.metrics.messages_sent += 1
        return message

    async def receive_message(self, message: AgentMessage):
        """
        Receive a message and add to queue.

        Args:
            message: The received message
        """
        await self.message_queue.put(message)
        self.metrics.messages_received += 1
        self.metrics.last_active = datetime.now()

    async def run(self):
        """Main agent loop for processing messages."""
        self._running = True
        self.state = AgentState.RUNNING
        self.logger.info(f"Agent {self.name} started")

        while self._running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                start_time = datetime.now()

                # Process the message
                response = await self.process_message(message)

                # Update metrics
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                self._update_response_time(elapsed)
                self.metrics.tasks_completed += 1

                # Handle response if any
                if response:
                    # Response would be sent back through orchestrator
                    pass

            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.metrics.tasks_failed += 1
                self.state = AgentState.ERROR

        self.state = AgentState.TERMINATED
        self.logger.info(f"Agent {self.name} stopped")

    def _update_response_time(self, elapsed_ms: float):
        """Update average response time metric."""
        total = self.metrics.tasks_completed
        if total == 0:
            self.metrics.average_response_time_ms = elapsed_ms
        else:
            # Rolling average
            self.metrics.average_response_time_ms = (
                (self.metrics.average_response_time_ms * total + elapsed_ms) /
                (total + 1)
            )

    async def stop(self):
        """Stop the agent."""
        self._running = False
        await self.shutdown()

    def pause(self):
        """Pause the agent."""
        self.state = AgentState.PAUSED
        self.logger.info(f"Agent {self.name} paused")

    def resume(self):
        """Resume the agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            self.logger.info(f"Agent {self.name} resumed")

    def subscribe(self, topic: str):
        """Subscribe to a broadcast topic."""
        self._subscribers.add(topic)

    def unsubscribe(self, topic: str):
        """Unsubscribe from a broadcast topic."""
        self._subscribers.discard(topic)

    def set_context(self, key: str, value: Any):
        """Set a context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'description': self.description,
            'state': self.state.value,
            'capabilities': list(self.capabilities.keys()),
            'metrics': {
                'messages_received': self.metrics.messages_received,
                'messages_sent': self.metrics.messages_sent,
                'tasks_completed': self.metrics.tasks_completed,
                'tasks_failed': self.metrics.tasks_failed,
                'avg_response_time_ms': round(
                    self.metrics.average_response_time_ms, 2
                ),
                'last_active': (
                    self.metrics.last_active.isoformat()
                    if self.metrics.last_active else None
                )
            },
            'created_at': self.created_at.isoformat()
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, state={self.state.value})>"
