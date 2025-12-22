"""
Platform Event System

Provides event-driven communication across the platform.
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import uuid


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventCategory(Enum):
    """Categories of platform events."""
    SYSTEM = "system"
    PAYMENT = "payment"
    CUSTOMER = "customer"
    SUBSCRIPTION = "subscription"
    AGENT = "agent"
    KNOWLEDGE = "knowledge"
    ERROR = "error"
    SECURITY = "security"


@dataclass
class PlatformEvent:
    """Platform event structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: EventCategory = EventCategory.SYSTEM
    priority: EventPriority = EventPriority.NORMAL
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'priority': self.priority.value,
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }


@dataclass
class EventSubscription:
    """Event subscription."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern: str = "*"  # Event name pattern (supports wildcards)
    category: Optional[EventCategory] = None
    handler: Callable[[PlatformEvent], Any] = None
    is_async: bool = True
    priority: int = 0


class EventBus:
    """
    Central event bus for platform-wide communication.

    Provides:
    - Publish/subscribe pattern
    - Event filtering
    - Async event handling
    - Event history
    """

    def __init__(self, name: str = "platform_event_bus"):
        self.name = name
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_history: List[PlatformEvent] = []
        self.max_history_size = 1000
        self.logger = logging.getLogger(f"event_bus.{name}")

        # Pattern index for faster matching
        self._pattern_index: Dict[str, Set[str]] = {}
        self._category_index: Dict[EventCategory, Set[str]] = {}

        # Metrics
        self.events_published = 0
        self.events_delivered = 0

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[PlatformEvent], Any],
        category: Optional[EventCategory] = None,
        is_async: bool = True,
        priority: int = 0
    ) -> str:
        """
        Subscribe to events.

        Args:
            pattern: Event name pattern (* for all, payment.* for prefix)
            handler: Event handler function
            category: Optional category filter
            is_async: Whether handler is async
            priority: Handler priority (higher runs first)

        Returns:
            Subscription ID
        """
        subscription = EventSubscription(
            pattern=pattern,
            category=category,
            handler=handler,
            is_async=is_async,
            priority=priority
        )

        self.subscriptions[subscription.id] = subscription

        # Update indexes
        if pattern not in self._pattern_index:
            self._pattern_index[pattern] = set()
        self._pattern_index[pattern].add(subscription.id)

        if category:
            if category not in self._category_index:
                self._category_index[category] = set()
            self._category_index[category].add(subscription.id)

        self.logger.debug(
            f"Subscription created: {subscription.id} for pattern '{pattern}'"
        )
        return subscription.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions.pop(subscription_id)

        # Update indexes
        if subscription.pattern in self._pattern_index:
            self._pattern_index[subscription.pattern].discard(subscription_id)

        if subscription.category and subscription.category in self._category_index:
            self._category_index[subscription.category].discard(subscription_id)

        self.logger.debug(f"Subscription removed: {subscription_id}")
        return True

    async def publish(self, event: PlatformEvent) -> int:
        """
        Publish an event.

        Args:
            event: Event to publish

        Returns:
            Number of handlers invoked
        """
        self.events_published += 1
        self._track_event(event)

        # Find matching subscriptions
        matching_subs = self._find_matching_subscriptions(event)

        # Sort by priority (higher first)
        matching_subs.sort(key=lambda s: s.priority, reverse=True)

        handlers_invoked = 0

        for subscription in matching_subs:
            try:
                if subscription.is_async:
                    await subscription.handler(event)
                else:
                    subscription.handler(event)

                handlers_invoked += 1
                self.events_delivered += 1

            except Exception as e:
                self.logger.error(
                    f"Error in event handler {subscription.id}: {e}"
                )

        return handlers_invoked

    def publish_sync(self, event: PlatformEvent) -> int:
        """
        Publish an event synchronously.

        Args:
            event: Event to publish

        Returns:
            Number of handlers invoked
        """
        self.events_published += 1
        self._track_event(event)

        matching_subs = self._find_matching_subscriptions(event)
        matching_subs.sort(key=lambda s: s.priority, reverse=True)

        handlers_invoked = 0

        for subscription in matching_subs:
            if subscription.is_async:
                continue  # Skip async handlers in sync mode

            try:
                subscription.handler(event)
                handlers_invoked += 1
                self.events_delivered += 1
            except Exception as e:
                self.logger.error(
                    f"Error in event handler {subscription.id}: {e}"
                )

        return handlers_invoked

    def _find_matching_subscriptions(
        self,
        event: PlatformEvent
    ) -> List[EventSubscription]:
        """Find subscriptions matching an event."""
        matching = []

        for sub_id, subscription in self.subscriptions.items():
            if self._matches_pattern(event.name, subscription.pattern):
                if subscription.category is None or subscription.category == event.category:
                    matching.append(subscription)

        return matching

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if event name matches pattern."""
        if pattern == "*":
            return True

        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return name.startswith(prefix)

        return name == pattern

    def _track_event(self, event: PlatformEvent):
        """Track event in history."""
        self.event_history.append(event)

        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]

    # Convenience methods for creating events

    def create_event(
        self,
        name: str,
        category: EventCategory = EventCategory.SYSTEM,
        data: Optional[Dict[str, Any]] = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL
    ) -> PlatformEvent:
        """Create a new event."""
        return PlatformEvent(
            name=name,
            category=category,
            priority=priority,
            source=source,
            data=data or {}
        )

    async def emit_payment_event(
        self,
        event_name: str,
        payment_data: Dict[str, Any],
        source: str = ""
    ) -> int:
        """Emit a payment-related event."""
        event = self.create_event(
            name=f"payment.{event_name}",
            category=EventCategory.PAYMENT,
            data=payment_data,
            source=source
        )
        return await self.publish(event)

    async def emit_customer_event(
        self,
        event_name: str,
        customer_data: Dict[str, Any],
        source: str = ""
    ) -> int:
        """Emit a customer-related event."""
        event = self.create_event(
            name=f"customer.{event_name}",
            category=EventCategory.CUSTOMER,
            data=customer_data,
            source=source
        )
        return await self.publish(event)

    async def emit_agent_event(
        self,
        event_name: str,
        agent_data: Dict[str, Any],
        source: str = ""
    ) -> int:
        """Emit an agent-related event."""
        event = self.create_event(
            name=f"agent.{event_name}",
            category=EventCategory.AGENT,
            data=agent_data,
            source=source
        )
        return await self.publish(event)

    async def emit_error_event(
        self,
        error_name: str,
        error_data: Dict[str, Any],
        source: str = ""
    ) -> int:
        """Emit an error event."""
        event = self.create_event(
            name=f"error.{error_name}",
            category=EventCategory.ERROR,
            data=error_data,
            source=source,
            priority=EventPriority.HIGH
        )
        return await self.publish(event)

    def get_recent_events(
        self,
        limit: int = 10,
        category: Optional[EventCategory] = None
    ) -> List[PlatformEvent]:
        """Get recent events."""
        events = self.event_history[-limit:]

        if category:
            events = [e for e in events if e.category == category]

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        category_counts = {}
        for event in self.event_history:
            cat = event.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            'name': self.name,
            'subscriptions': len(self.subscriptions),
            'events_published': self.events_published,
            'events_delivered': self.events_delivered,
            'event_history_size': len(self.event_history),
            'category_counts': category_counts,
            'patterns': list(self._pattern_index.keys())
        }

    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()
