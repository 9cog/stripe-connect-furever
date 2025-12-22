"""
Core Platform Components

Main entry points and configuration for the unified Stripe platform.
"""

from .unified_interface import UnifiedStripeInterface
from .config import PlatformConfig, load_config
from .events import EventBus, PlatformEvent

__all__ = [
    'UnifiedStripeInterface',
    'PlatformConfig',
    'load_config',
    'EventBus',
    'PlatformEvent',
]
