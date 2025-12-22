"""
Plugin Bridge

Provides unified access to Stripe plugins for various platforms.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import os


class PluginStatus(Enum):
    """Status of a plugin."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"
    LOADING = "loading"


class PluginPlatform(Enum):
    """Platforms for plugins."""
    IOS = "ios"
    ANDROID = "android"
    REACT_NATIVE = "react-native"
    WEB = "web"
    TERMINAL = "terminal"
    IDENTITY = "identity"


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    name: str
    platform: PluginPlatform
    path: str
    version: str = "latest"
    status: PluginStatus = PluginStatus.INACTIVE
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginEvent:
    """Event from a plugin."""
    plugin_name: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PluginBridge:
    """
    Unified bridge for managing Stripe plugins.

    Provides:
    - Plugin discovery and registration
    - Lifecycle management
    - Event handling
    - Configuration management
    """

    # Default plugin configurations
    DEFAULT_PLUGINS = {
        "stripe-terminal-react-native": PluginConfig(
            name="stripe-terminal-react-native",
            platform=PluginPlatform.REACT_NATIVE,
            path="platform/stripe-ecosystem/plugins/stripe-terminal-react-native",
            description="React Native SDK for Stripe Terminal",
            capabilities=[
                "discover_readers",
                "connect_reader",
                "collect_payment",
                "cancel_collect",
                "card_present_payments"
            ]
        ),
        "stripe-identity-react-native": PluginConfig(
            name="stripe-identity-react-native",
            platform=PluginPlatform.REACT_NATIVE,
            path="platform/stripe-ecosystem/plugins/stripe-identity-react-native",
            description="React Native SDK for Stripe Identity",
            capabilities=[
                "verify_identity",
                "collect_document",
                "collect_selfie",
                "identity_verification"
            ]
        ),
        "stripe-ios": PluginConfig(
            name="stripe-ios",
            platform=PluginPlatform.IOS,
            path="platform/stripe-ecosystem/apps/stripe-ios",
            description="iOS SDK for Stripe payments",
            capabilities=[
                "payment_sheet",
                "card_scanning",
                "apple_pay",
                "save_cards",
                "setup_intents"
            ]
        ),
        "stripe-android": PluginConfig(
            name="stripe-android",
            platform=PluginPlatform.ANDROID,
            path="platform/stripe-ecosystem/apps/stripe-android",
            description="Android SDK for Stripe payments",
            capabilities=[
                "payment_sheet",
                "card_scanning",
                "google_pay",
                "save_cards",
                "setup_intents"
            ]
        )
    }

    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.plugins: Dict[str, PluginConfig] = {}
        self.event_listeners: Dict[str, List[Callable[[PluginEvent], None]]] = {}
        self.logger = logging.getLogger("plugin_bridge")
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the plugin bridge.

        Returns:
            True if initialization successful
        """
        try:
            # Load default plugin configurations
            for name, config in self.DEFAULT_PLUGINS.items():
                self.register_plugin(config)

            self._initialized = True
            self.logger.info("Plugin bridge initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize plugin bridge: {e}")
            return False

    def register_plugin(self, config: PluginConfig):
        """
        Register a plugin.

        Args:
            config: Plugin configuration
        """
        # Verify plugin path exists
        full_path = os.path.join(self.base_path, config.path)
        if os.path.exists(full_path):
            config.status = PluginStatus.INACTIVE
        else:
            config.status = PluginStatus.ERROR
            self.logger.warning(f"Plugin path not found: {full_path}")

        self.plugins[config.name] = config
        self.logger.info(f"Registered plugin: {config.name}")

    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.logger.info(f"Unregistered plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get plugin configuration by name."""
        return self.plugins.get(plugin_name)

    def get_plugins_by_platform(
        self,
        platform: PluginPlatform
    ) -> List[PluginConfig]:
        """Get all plugins for a specific platform."""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.platform == platform
        ]

    def get_active_plugins(self) -> List[PluginConfig]:
        """Get all active plugins."""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.status == PluginStatus.ACTIVE
        ]

    def get_plugins_with_capability(
        self,
        capability: str
    ) -> List[PluginConfig]:
        """Get all plugins that have a specific capability."""
        return [
            plugin for plugin in self.plugins.values()
            if capability in plugin.capabilities
        ]

    async def activate_plugin(self, plugin_name: str) -> bool:
        """
        Activate a plugin.

        Args:
            plugin_name: Name of the plugin to activate

        Returns:
            True if activation successful
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            self.logger.warning(f"Plugin not found: {plugin_name}")
            return False

        if plugin.status == PluginStatus.ERROR:
            self.logger.warning(f"Cannot activate plugin in error state: {plugin_name}")
            return False

        try:
            plugin.status = PluginStatus.LOADING
            # Simulated plugin activation
            plugin.status = PluginStatus.ACTIVE

            self._emit_event(PluginEvent(
                plugin_name=plugin_name,
                event_type="activated"
            ))

            self.logger.info(f"Activated plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin.status = PluginStatus.ERROR
            self.logger.error(f"Failed to activate plugin {plugin_name}: {e}")
            return False

    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate a plugin.

        Args:
            plugin_name: Name of the plugin to deactivate

        Returns:
            True if deactivation successful
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return False

        try:
            plugin.status = PluginStatus.INACTIVE

            self._emit_event(PluginEvent(
                plugin_name=plugin_name,
                event_type="deactivated"
            ))

            self.logger.info(f"Deactivated plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deactivate plugin {plugin_name}: {e}")
            return False

    def configure_plugin(
        self,
        plugin_name: str,
        configuration: Dict[str, Any]
    ) -> bool:
        """
        Configure a plugin.

        Args:
            plugin_name: Name of the plugin
            configuration: Configuration dictionary

        Returns:
            True if configuration successful
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return False

        plugin.configuration.update(configuration)

        self._emit_event(PluginEvent(
            plugin_name=plugin_name,
            event_type="configured",
            data={'configuration': configuration}
        ))

        self.logger.info(f"Configured plugin: {plugin_name}")
        return True

    def get_plugin_configuration(
        self,
        plugin_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get plugin configuration."""
        plugin = self.plugins.get(plugin_name)
        return plugin.configuration if plugin else None

    async def execute_capability(
        self,
        plugin_name: str,
        capability: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a plugin capability.

        Args:
            plugin_name: Name of the plugin
            capability: Capability to execute
            params: Parameters for the capability

        Returns:
            Result of capability execution
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return {'error': 'Plugin not found', 'plugin_name': plugin_name}

        if plugin.status != PluginStatus.ACTIVE:
            return {'error': 'Plugin not active', 'status': plugin.status.value}

        if capability not in plugin.capabilities:
            return {
                'error': 'Capability not supported',
                'capability': capability,
                'available': plugin.capabilities
            }

        try:
            # Simulated capability execution
            result = {
                'success': True,
                'plugin': plugin_name,
                'capability': capability,
                'params': params,
                'result': f"Executed {capability}"
            }

            self._emit_event(PluginEvent(
                plugin_name=plugin_name,
                event_type="capability_executed",
                data={'capability': capability, 'params': params}
            ))

            return result

        except Exception as e:
            self.logger.error(
                f"Error executing {capability} on {plugin_name}: {e}"
            )
            return {'error': str(e)}

    def subscribe_to_plugin(
        self,
        plugin_name: str,
        listener: Callable[[PluginEvent], None]
    ):
        """
        Subscribe to plugin events.

        Args:
            plugin_name: Name of the plugin
            listener: Event listener callback
        """
        if plugin_name not in self.event_listeners:
            self.event_listeners[plugin_name] = []
        self.event_listeners[plugin_name].append(listener)

    def unsubscribe_from_plugin(
        self,
        plugin_name: str,
        listener: Callable[[PluginEvent], None]
    ):
        """Unsubscribe from plugin events."""
        if plugin_name in self.event_listeners:
            if listener in self.event_listeners[plugin_name]:
                self.event_listeners[plugin_name].remove(listener)

    def _emit_event(self, event: PluginEvent):
        """Emit an event to listeners."""
        # Plugin-specific listeners
        if event.plugin_name in self.event_listeners:
            for listener in self.event_listeners[event.plugin_name]:
                try:
                    listener(event)
                except Exception as e:
                    self.logger.error(f"Error in event listener: {e}")

        # Global listeners (subscribed with "*")
        if "*" in self.event_listeners:
            for listener in self.event_listeners["*"]:
                try:
                    listener(event)
                except Exception as e:
                    self.logger.error(f"Error in global event listener: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all plugins."""
        statuses = {}
        for name, plugin in self.plugins.items():
            statuses[name] = {
                'status': plugin.status.value,
                'platform': plugin.platform.value,
                'capabilities_count': len(plugin.capabilities)
            }
        return statuses

    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin bridge statistics."""
        status_counts = {}
        platform_counts = {}
        capability_counts = {}

        for plugin in self.plugins.values():
            status = plugin.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            platform = plugin.platform.value
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

            for cap in plugin.capabilities:
                capability_counts[cap] = capability_counts.get(cap, 0) + 1

        return {
            'initialized': self._initialized,
            'total_plugins': len(self.plugins),
            'active_plugins': len([
                p for p in self.plugins.values()
                if p.status == PluginStatus.ACTIVE
            ]),
            'status_counts': status_counts,
            'platform_counts': platform_counts,
            'capability_counts': capability_counts,
            'plugins': list(self.plugins.keys())
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export plugin bridge state to dictionary."""
        return {
            'plugins': {
                name: {
                    'name': plugin.name,
                    'platform': plugin.platform.value,
                    'path': plugin.path,
                    'version': plugin.version,
                    'status': plugin.status.value,
                    'description': plugin.description,
                    'capabilities': plugin.capabilities,
                    'configuration': plugin.configuration
                }
                for name, plugin in self.plugins.items()
            },
            'statistics': self.get_statistics()
        }
