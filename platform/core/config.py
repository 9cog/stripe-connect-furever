"""
Platform Configuration

Configuration management for the unified Stripe platform.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import logging


class Environment(Enum):
    """Platform environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class StripeConfig:
    """Stripe API configuration."""
    api_key: str = ""
    api_version: str = "2023-10-16"
    webhook_secret: str = ""
    connect_webhook_secret: str = ""
    publishable_key: str = ""


@dataclass
class OpenCogConfig:
    """OpenCog configuration."""
    atomspace_name: str = "stripe_atomspace"
    enable_reasoning: bool = True
    inference_interval_seconds: int = 60
    max_atoms: int = 100000


@dataclass
class AgentConfig:
    """Agent system configuration."""
    max_agents: int = 10
    message_queue_size: int = 1000
    default_timeout_seconds: float = 30.0
    enable_monitoring: bool = True
    heartbeat_interval_seconds: int = 30


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 100.0
    burst_size: int = 200
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_console: bool = True


@dataclass
class PlatformConfig:
    """Main platform configuration."""
    environment: Environment = Environment.DEVELOPMENT
    name: str = "StripeOpenCogPlatform"
    version: str = "1.0.0"
    base_path: str = ""

    stripe: StripeConfig = field(default_factory=StripeConfig)
    opencog: OpenCogConfig = field(default_factory=OpenCogConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    features: Dict[str, bool] = field(default_factory=lambda: {
        'payments': True,
        'customers': True,
        'subscriptions': True,
        'connect': True,
        'terminal': False,
        'identity': False,
        'issuing': False,
        'treasury': False
    })

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.value,
            'name': self.name,
            'version': self.version,
            'base_path': self.base_path,
            'stripe': {
                'api_version': self.stripe.api_version,
                'has_api_key': bool(self.stripe.api_key)
            },
            'opencog': {
                'atomspace_name': self.opencog.atomspace_name,
                'enable_reasoning': self.opencog.enable_reasoning,
                'inference_interval_seconds': self.opencog.inference_interval_seconds
            },
            'agents': {
                'max_agents': self.agents.max_agents,
                'enable_monitoring': self.agents.enable_monitoring
            },
            'features': self.features,
            'metadata': self.metadata
        }


def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = "STRIPE_PLATFORM_"
) -> PlatformConfig:
    """
    Load platform configuration from file and environment.

    Args:
        config_file: Path to configuration JSON file
        env_prefix: Prefix for environment variables

    Returns:
        Platform configuration
    """
    config = PlatformConfig()

    # Load from file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config = _apply_dict_config(config, file_config)

    # Override from environment variables
    config = _apply_env_config(config, env_prefix)

    # Setup logging
    _setup_logging(config.logging)

    return config


def _apply_dict_config(
    config: PlatformConfig,
    data: Dict[str, Any]
) -> PlatformConfig:
    """Apply dictionary configuration."""
    if 'environment' in data:
        config.environment = Environment(data['environment'])

    if 'name' in data:
        config.name = data['name']

    if 'version' in data:
        config.version = data['version']

    if 'base_path' in data:
        config.base_path = data['base_path']

    if 'stripe' in data:
        stripe_data = data['stripe']
        if 'api_key' in stripe_data:
            config.stripe.api_key = stripe_data['api_key']
        if 'api_version' in stripe_data:
            config.stripe.api_version = stripe_data['api_version']
        if 'webhook_secret' in stripe_data:
            config.stripe.webhook_secret = stripe_data['webhook_secret']

    if 'opencog' in data:
        oc_data = data['opencog']
        if 'atomspace_name' in oc_data:
            config.opencog.atomspace_name = oc_data['atomspace_name']
        if 'enable_reasoning' in oc_data:
            config.opencog.enable_reasoning = oc_data['enable_reasoning']

    if 'agents' in data:
        agent_data = data['agents']
        if 'max_agents' in agent_data:
            config.agents.max_agents = agent_data['max_agents']
        if 'enable_monitoring' in agent_data:
            config.agents.enable_monitoring = agent_data['enable_monitoring']

    if 'features' in data:
        config.features.update(data['features'])

    if 'metadata' in data:
        config.metadata.update(data['metadata'])

    return config


def _apply_env_config(
    config: PlatformConfig,
    prefix: str
) -> PlatformConfig:
    """Apply environment variable configuration."""
    # Stripe configuration
    if api_key := os.getenv(f'{prefix}STRIPE_API_KEY'):
        config.stripe.api_key = api_key

    if api_key := os.getenv('STRIPE_SECRET_KEY'):
        config.stripe.api_key = api_key

    if webhook_secret := os.getenv(f'{prefix}WEBHOOK_SECRET'):
        config.stripe.webhook_secret = webhook_secret

    if webhook_secret := os.getenv('STRIPE_WEBHOOK_SECRET'):
        config.stripe.webhook_secret = webhook_secret

    # Environment
    if env := os.getenv(f'{prefix}ENVIRONMENT'):
        try:
            config.environment = Environment(env.lower())
        except ValueError:
            pass

    # Logging
    if log_level := os.getenv(f'{prefix}LOG_LEVEL'):
        config.logging.level = log_level.upper()

    return config


def _setup_logging(logging_config: LoggingConfig):
    """Setup logging based on configuration."""
    handlers = []

    if logging_config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(logging_config.format)
        )
        handlers.append(console_handler)

    if logging_config.log_file:
        file_handler = logging.FileHandler(logging_config.log_file)
        file_handler.setFormatter(
            logging.Formatter(logging_config.format)
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, logging_config.level),
        handlers=handlers
    )
