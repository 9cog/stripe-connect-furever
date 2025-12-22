"""
Agent Registry

Provides registration and discovery of agents and their capabilities.
"""

from typing import Any, Dict, List, Optional, Type, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .base_agent import BaseAgent, AgentCapability, AgentState


class RegistryEventType(Enum):
    """Types of registry events."""
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    CAPABILITY_ADDED = "capability_added"
    CAPABILITY_REMOVED = "capability_removed"
    STATE_CHANGED = "state_changed"


@dataclass
class RegistryEntry:
    """Entry in the agent registry."""
    agent_id: str
    agent_class: str
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    registered_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryEvent:
    """Event from the registry."""
    event_type: RegistryEventType
    agent_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AgentRegistry:
    """
    Central registry for agents.

    Provides:
    - Agent registration and discovery
    - Capability-based lookup
    - Tag-based filtering
    - Version management
    - Event notifications
    """

    def __init__(self):
        self.entries: Dict[str, RegistryEntry] = {}
        self.capability_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.agent_classes: Dict[str, Type[BaseAgent]] = {}
        self.event_listeners: List[callable] = []
        self.logger = logging.getLogger("agent_registry")

    def register(
        self,
        agent: BaseAgent,
        tags: Optional[Set[str]] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RegistryEntry:
        """
        Register an agent.

        Args:
            agent: The agent to register
            tags: Optional tags for categorization
            version: Agent version
            metadata: Additional metadata

        Returns:
            The registry entry
        """
        entry = RegistryEntry(
            agent_id=agent.agent_id,
            agent_class=agent.__class__.__name__,
            name=agent.name,
            description=agent.description,
            capabilities=list(agent.capabilities.keys()),
            tags=tags or set(),
            version=version,
            metadata=metadata or {}
        )

        self.entries[agent.agent_id] = entry

        # Update capability index
        for capability in entry.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent.agent_id)

        # Update tag index
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(agent.agent_id)

        # Register agent class for factory creation
        self.agent_classes[agent.__class__.__name__] = agent.__class__

        self._emit_event(RegistryEvent(
            event_type=RegistryEventType.AGENT_REGISTERED,
            agent_id=agent.agent_id,
            data={'name': agent.name, 'capabilities': entry.capabilities}
        ))

        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
        return entry

    def unregister(self, agent_id: str):
        """
        Unregister an agent.

        Args:
            agent_id: The agent ID to unregister
        """
        if agent_id not in self.entries:
            return

        entry = self.entries.pop(agent_id)

        # Remove from capability index
        for capability in entry.capabilities:
            if capability in self.capability_index:
                self.capability_index[capability].discard(agent_id)

        # Remove from tag index
        for tag in entry.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(agent_id)

        self._emit_event(RegistryEvent(
            event_type=RegistryEventType.AGENT_UNREGISTERED,
            agent_id=agent_id,
            data={'name': entry.name}
        ))

        self.logger.info(f"Unregistered agent: {entry.name}")

    def get_entry(self, agent_id: str) -> Optional[RegistryEntry]:
        """Get a registry entry by agent ID."""
        return self.entries.get(agent_id)

    def find_by_capability(self, capability: str) -> List[RegistryEntry]:
        """
        Find agents by capability.

        Args:
            capability: The capability to search for

        Returns:
            List of matching registry entries
        """
        if capability not in self.capability_index:
            return []

        return [
            self.entries[agent_id]
            for agent_id in self.capability_index[capability]
            if agent_id in self.entries
        ]

    def find_by_tag(self, tag: str) -> List[RegistryEntry]:
        """
        Find agents by tag.

        Args:
            tag: The tag to search for

        Returns:
            List of matching registry entries
        """
        if tag not in self.tag_index:
            return []

        return [
            self.entries[agent_id]
            for agent_id in self.tag_index[tag]
            if agent_id in self.entries
        ]

    def find_by_tags(
        self,
        tags: Set[str],
        match_all: bool = True
    ) -> List[RegistryEntry]:
        """
        Find agents by multiple tags.

        Args:
            tags: The tags to search for
            match_all: If True, agent must have all tags

        Returns:
            List of matching registry entries
        """
        if not tags:
            return list(self.entries.values())

        if match_all:
            # Intersection of all tag sets
            result_ids = None
            for tag in tags:
                tag_set = self.tag_index.get(tag, set())
                if result_ids is None:
                    result_ids = tag_set.copy()
                else:
                    result_ids &= tag_set
            result_ids = result_ids or set()
        else:
            # Union of all tag sets
            result_ids = set()
            for tag in tags:
                result_ids |= self.tag_index.get(tag, set())

        return [
            self.entries[agent_id]
            for agent_id in result_ids
            if agent_id in self.entries
        ]

    def search(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[RegistryEntry]:
        """
        Search registry entries.

        Args:
            query: Search query
            fields: Fields to search (name, description, capabilities)

        Returns:
            List of matching entries
        """
        fields = fields or ['name', 'description', 'capabilities']
        query_lower = query.lower()
        results = []

        for entry in self.entries.values():
            match = False

            if 'name' in fields and query_lower in entry.name.lower():
                match = True
            elif 'description' in fields and query_lower in entry.description.lower():
                match = True
            elif 'capabilities' in fields:
                for cap in entry.capabilities:
                    if query_lower in cap.lower():
                        match = True
                        break

            if match:
                results.append(entry)

        return results

    def add_capability(self, agent_id: str, capability: AgentCapability):
        """
        Add a capability to a registered agent.

        Args:
            agent_id: The agent ID
            capability: The capability to add
        """
        if agent_id not in self.entries:
            return

        entry = self.entries[agent_id]
        if capability.name not in entry.capabilities:
            entry.capabilities.append(capability.name)

            if capability.name not in self.capability_index:
                self.capability_index[capability.name] = set()
            self.capability_index[capability.name].add(agent_id)

            self._emit_event(RegistryEvent(
                event_type=RegistryEventType.CAPABILITY_ADDED,
                agent_id=agent_id,
                data={'capability': capability.name}
            ))

    def remove_capability(self, agent_id: str, capability_name: str):
        """
        Remove a capability from a registered agent.

        Args:
            agent_id: The agent ID
            capability_name: The capability name to remove
        """
        if agent_id not in self.entries:
            return

        entry = self.entries[agent_id]
        if capability_name in entry.capabilities:
            entry.capabilities.remove(capability_name)

            if capability_name in self.capability_index:
                self.capability_index[capability_name].discard(agent_id)

            self._emit_event(RegistryEvent(
                event_type=RegistryEventType.CAPABILITY_REMOVED,
                agent_id=agent_id,
                data={'capability': capability_name}
            ))

    def add_tag(self, agent_id: str, tag: str):
        """Add a tag to a registered agent."""
        if agent_id not in self.entries:
            return

        entry = self.entries[agent_id]
        entry.tags.add(tag)

        if tag not in self.tag_index:
            self.tag_index[tag] = set()
        self.tag_index[tag].add(agent_id)

    def remove_tag(self, agent_id: str, tag: str):
        """Remove a tag from a registered agent."""
        if agent_id not in self.entries:
            return

        entry = self.entries[agent_id]
        entry.tags.discard(tag)

        if tag in self.tag_index:
            self.tag_index[tag].discard(agent_id)

    def add_event_listener(self, listener: callable):
        """Add an event listener."""
        self.event_listeners.append(listener)

    def remove_event_listener(self, listener: callable):
        """Remove an event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)

    def _emit_event(self, event: RegistryEvent):
        """Emit an event to all listeners."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Error in event listener: {e}")

    def get_all_capabilities(self) -> List[str]:
        """Get all registered capabilities."""
        return list(self.capability_index.keys())

    def get_all_tags(self) -> List[str]:
        """Get all registered tags."""
        return list(self.tag_index.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_agents': len(self.entries),
            'total_capabilities': len(self.capability_index),
            'total_tags': len(self.tag_index),
            'agents_by_class': self._count_by_class(),
            'capabilities': {
                cap: len(ids) for cap, ids in self.capability_index.items()
            },
            'tags': {
                tag: len(ids) for tag, ids in self.tag_index.items()
            }
        }

    def _count_by_class(self) -> Dict[str, int]:
        """Count agents by class."""
        counts: Dict[str, int] = {}
        for entry in self.entries.values():
            counts[entry.agent_class] = counts.get(entry.agent_class, 0) + 1
        return counts

    def export_to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary."""
        return {
            'entries': {
                agent_id: {
                    'agent_id': entry.agent_id,
                    'agent_class': entry.agent_class,
                    'name': entry.name,
                    'description': entry.description,
                    'capabilities': entry.capabilities,
                    'tags': list(entry.tags),
                    'version': entry.version,
                    'registered_at': entry.registered_at.isoformat(),
                    'metadata': entry.metadata
                }
                for agent_id, entry in self.entries.items()
            },
            'statistics': self.get_statistics()
        }

    def clear(self):
        """Clear all registry entries."""
        self.entries.clear()
        self.capability_index.clear()
        self.tag_index.clear()
        self.logger.info("Registry cleared")
