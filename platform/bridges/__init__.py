"""
Integration Bridges for Stripe Ecosystem

This module provides bridge interfaces connecting different
components of the unified Stripe platform.
"""

from .cognitive_bridge import CognitiveBridge
from .sdk_bridge import SDKBridge, SDKConfig
from .plugin_bridge import PluginBridge, PluginConfig
from .api_bridge import APIBridge, RateLimiter

__all__ = [
    'CognitiveBridge',
    'SDKBridge',
    'SDKConfig',
    'PluginBridge',
    'PluginConfig',
    'APIBridge',
    'RateLimiter',
]
