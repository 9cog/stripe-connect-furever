"""
SDK Bridge

Provides unified access to all Stripe SDKs with consistent interfaces.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import os
import json


class SDKStatus(Enum):
    """Status of an SDK."""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    DEPRECATED = "deprecated"


class SDKLanguage(Enum):
    """Supported SDK languages."""
    NODE = "node"
    PYTHON = "python"
    RUBY = "ruby"
    GO = "go"
    JAVA = "java"
    DOTNET = "dotnet"
    PHP = "php"
    REACT_NATIVE = "react-native"


@dataclass
class SDKConfig:
    """Configuration for an SDK."""
    name: str
    language: SDKLanguage
    path: str
    version: str = "latest"
    status: SDKStatus = SDKStatus.AVAILABLE
    entry_point: str = ""
    api_version: str = ""
    features: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SDKCapability:
    """A capability provided by an SDK."""
    name: str
    description: str
    sdk_name: str
    method_path: str
    input_params: Dict[str, Any] = field(default_factory=dict)
    output_type: str = "Any"
    is_async: bool = True


class SDKBridge:
    """
    Unified bridge for managing all Stripe SDKs.

    Provides:
    - SDK discovery and registration
    - Unified API interface
    - Capability mapping
    - Version management
    - Health monitoring
    """

    # Default SDK configurations
    DEFAULT_SDKS = {
        "stripe-node": SDKConfig(
            name="stripe-node",
            language=SDKLanguage.NODE,
            path="platform/stripe-ecosystem/sdks/stripe-node",
            entry_point="lib/stripe.js",
            features=[
                "async/await",
                "typescript",
                "auto-pagination",
                "idempotency",
                "webhooks"
            ]
        ),
        "stripe-python": SDKConfig(
            name="stripe-python",
            language=SDKLanguage.PYTHON,
            path="platform/stripe-ecosystem/sdks/stripe-python",
            entry_point="stripe/__init__.py",
            features=[
                "async",
                "type-hints",
                "auto-pagination",
                "idempotency",
                "webhooks"
            ]
        ),
        "stripe-ruby": SDKConfig(
            name="stripe-ruby",
            language=SDKLanguage.RUBY,
            path="platform/stripe-ecosystem/sdks/stripe-ruby",
            entry_point="lib/stripe.rb",
            features=[
                "auto-pagination",
                "idempotency",
                "webhooks"
            ]
        ),
        "stripe-go": SDKConfig(
            name="stripe-go",
            language=SDKLanguage.GO,
            path="platform/stripe-ecosystem/sdks/stripe-go",
            entry_point="stripe.go",
            features=[
                "context-support",
                "streaming",
                "auto-pagination"
            ]
        ),
        "stripe-java": SDKConfig(
            name="stripe-java",
            language=SDKLanguage.JAVA,
            path="platform/stripe-ecosystem/sdks/stripe-java",
            entry_point="src/main/java/com/stripe/Stripe.java",
            features=[
                "async",
                "streaming",
                "builder-pattern",
                "auto-pagination"
            ]
        ),
        "stripe-dotnet": SDKConfig(
            name="stripe-dotnet",
            language=SDKLanguage.DOTNET,
            path="platform/stripe-ecosystem/sdks/stripe-dotnet",
            entry_point="src/Stripe.net/StripeClient.cs",
            features=[
                "async",
                "linq",
                "strong-typing",
                "auto-pagination"
            ]
        ),
        "stripe-php": SDKConfig(
            name="stripe-php",
            language=SDKLanguage.PHP,
            path="platform/stripe-ecosystem/plugins/stripe-php",
            entry_point="lib/Stripe.php",
            features=[
                "psr-compliant",
                "auto-pagination",
                "webhooks"
            ]
        ),
        "stripe-react-native": SDKConfig(
            name="stripe-react-native",
            language=SDKLanguage.REACT_NATIVE,
            path="platform/stripe-ecosystem/sdks/stripe-react-native",
            entry_point="src/index.tsx",
            features=[
                "card-collection",
                "apple-pay",
                "google-pay",
                "payment-sheet"
            ]
        )
    }

    # API capability mappings
    API_CAPABILITIES = {
        "create_payment": SDKCapability(
            name="create_payment",
            description="Create a new payment intent",
            sdk_name="*",
            method_path="paymentIntents.create",
            input_params={
                "amount": "integer",
                "currency": "string",
                "customer": "string (optional)",
                "payment_method": "string (optional)"
            },
            output_type="PaymentIntent"
        ),
        "retrieve_payment": SDKCapability(
            name="retrieve_payment",
            description="Retrieve a payment intent",
            sdk_name="*",
            method_path="paymentIntents.retrieve",
            input_params={"id": "string"},
            output_type="PaymentIntent"
        ),
        "create_customer": SDKCapability(
            name="create_customer",
            description="Create a new customer",
            sdk_name="*",
            method_path="customers.create",
            input_params={
                "email": "string (optional)",
                "name": "string (optional)",
                "metadata": "object (optional)"
            },
            output_type="Customer"
        ),
        "create_subscription": SDKCapability(
            name="create_subscription",
            description="Create a new subscription",
            sdk_name="*",
            method_path="subscriptions.create",
            input_params={
                "customer": "string",
                "items": "array",
                "trial_period_days": "integer (optional)"
            },
            output_type="Subscription"
        ),
        "create_refund": SDKCapability(
            name="create_refund",
            description="Create a refund",
            sdk_name="*",
            method_path="refunds.create",
            input_params={
                "payment_intent": "string",
                "amount": "integer (optional)"
            },
            output_type="Refund"
        ),
        "list_payments": SDKCapability(
            name="list_payments",
            description="List payment intents",
            sdk_name="*",
            method_path="paymentIntents.list",
            input_params={
                "limit": "integer (optional)",
                "starting_after": "string (optional)"
            },
            output_type="List[PaymentIntent]"
        ),
        "verify_webhook": SDKCapability(
            name="verify_webhook",
            description="Verify webhook signature",
            sdk_name="*",
            method_path="webhooks.constructEvent",
            input_params={
                "payload": "string",
                "sig_header": "string",
                "endpoint_secret": "string"
            },
            output_type="Event"
        )
    }

    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.sdks: Dict[str, SDKConfig] = {}
        self.capabilities: Dict[str, SDKCapability] = {}
        self.logger = logging.getLogger("sdk_bridge")
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the SDK bridge.

        Returns:
            True if initialization successful
        """
        try:
            # Load default SDK configurations
            for name, config in self.DEFAULT_SDKS.items():
                self.register_sdk(config)

            # Load API capabilities
            for name, capability in self.API_CAPABILITIES.items():
                self.capabilities[name] = capability

            self._initialized = True
            self.logger.info("SDK bridge initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SDK bridge: {e}")
            return False

    def register_sdk(self, config: SDKConfig):
        """
        Register an SDK.

        Args:
            config: SDK configuration
        """
        # Verify SDK path exists
        full_path = os.path.join(self.base_path, config.path)
        if os.path.exists(full_path):
            config.status = SDKStatus.AVAILABLE
        else:
            config.status = SDKStatus.ERROR
            self.logger.warning(f"SDK path not found: {full_path}")

        self.sdks[config.name] = config
        self.logger.info(f"Registered SDK: {config.name}")

    def unregister_sdk(self, sdk_name: str):
        """Unregister an SDK."""
        if sdk_name in self.sdks:
            del self.sdks[sdk_name]
            self.logger.info(f"Unregistered SDK: {sdk_name}")

    def get_sdk(self, sdk_name: str) -> Optional[SDKConfig]:
        """Get SDK configuration by name."""
        return self.sdks.get(sdk_name)

    def get_sdks_by_language(
        self,
        language: SDKLanguage
    ) -> List[SDKConfig]:
        """Get all SDKs for a specific language."""
        return [
            sdk for sdk in self.sdks.values()
            if sdk.language == language
        ]

    def get_available_sdks(self) -> List[SDKConfig]:
        """Get all available SDKs."""
        return [
            sdk for sdk in self.sdks.values()
            if sdk.status == SDKStatus.AVAILABLE
        ]

    def get_capability(self, capability_name: str) -> Optional[SDKCapability]:
        """Get a capability by name."""
        return self.capabilities.get(capability_name)

    def get_sdks_with_capability(
        self,
        capability_name: str
    ) -> List[SDKConfig]:
        """Get all SDKs that support a capability."""
        capability = self.capabilities.get(capability_name)
        if not capability:
            return []

        if capability.sdk_name == "*":
            return self.get_available_sdks()

        sdk = self.sdks.get(capability.sdk_name)
        return [sdk] if sdk else []

    def get_sdk_features(self, sdk_name: str) -> List[str]:
        """Get features supported by an SDK."""
        sdk = self.sdks.get(sdk_name)
        return sdk.features if sdk else []

    def recommend_sdk(
        self,
        requirements: Dict[str, Any]
    ) -> Optional[SDKConfig]:
        """
        Recommend the best SDK based on requirements.

        Args:
            requirements: Dictionary with language, features, etc.

        Returns:
            Recommended SDK configuration
        """
        language = requirements.get('language')
        required_features = set(requirements.get('features', []))

        candidates = self.get_available_sdks()

        # Filter by language
        if language:
            try:
                lang_enum = SDKLanguage(language.lower())
                candidates = [s for s in candidates if s.language == lang_enum]
            except ValueError:
                pass

        # Score by feature match
        scored = []
        for sdk in candidates:
            sdk_features = set(sdk.features)
            match_count = len(required_features & sdk_features)
            scored.append((sdk, match_count))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0] if scored else None

    def get_code_example(
        self,
        sdk_name: str,
        capability_name: str
    ) -> Optional[str]:
        """
        Get a code example for using a capability.

        Args:
            sdk_name: The SDK name
            capability_name: The capability name

        Returns:
            Code example string
        """
        sdk = self.sdks.get(sdk_name)
        capability = self.capabilities.get(capability_name)

        if not sdk or not capability:
            return None

        templates = {
            SDKLanguage.NODE: self._get_node_example,
            SDKLanguage.PYTHON: self._get_python_example,
            SDKLanguage.RUBY: self._get_ruby_example,
            SDKLanguage.GO: self._get_go_example,
            SDKLanguage.JAVA: self._get_java_example,
            SDKLanguage.DOTNET: self._get_dotnet_example,
            SDKLanguage.PHP: self._get_php_example,
        }

        template_func = templates.get(sdk.language)
        if template_func:
            return template_func(capability)

        return None

    def _get_node_example(self, capability: SDKCapability) -> str:
        """Generate Node.js code example."""
        return f'''
const stripe = require('stripe')('sk_test_...');

// {capability.description}
const result = await stripe.{capability.method_path}({{
    // Parameters: {json.dumps(capability.input_params, indent=2)}
}});

console.log(result);
'''

    def _get_python_example(self, capability: SDKCapability) -> str:
        """Generate Python code example."""
        method = capability.method_path.replace('.', '.')
        return f'''
import stripe

stripe.api_key = 'sk_test_...'

# {capability.description}
result = stripe.{method}(
    # Parameters: {json.dumps(capability.input_params, indent=4)}
)

print(result)
'''

    def _get_ruby_example(self, capability: SDKCapability) -> str:
        """Generate Ruby code example."""
        return f'''
require 'stripe'

Stripe.api_key = 'sk_test_...'

# {capability.description}
result = Stripe::{capability.method_path.replace('.', '::')}(
  # Parameters: {json.dumps(capability.input_params, indent=2)}
)

puts result
'''

    def _get_go_example(self, capability: SDKCapability) -> str:
        """Generate Go code example."""
        return f'''
package main

import (
    "github.com/stripe/stripe-go/v76"
    // Import specific package for {capability.method_path}
)

func main() {{
    stripe.Key = "sk_test_..."

    // {capability.description}
    // Parameters: {json.dumps(capability.input_params, indent=4)}
}}
'''

    def _get_java_example(self, capability: SDKCapability) -> str:
        """Generate Java code example."""
        return f'''
import com.stripe.Stripe;
import com.stripe.model.*;
import com.stripe.param.*;

public class Example {{
    public static void main(String[] args) throws Exception {{
        Stripe.apiKey = "sk_test_...";

        // {capability.description}
        // Parameters: {json.dumps(capability.input_params, indent=8)}
    }}
}}
'''

    def _get_dotnet_example(self, capability: SDKCapability) -> str:
        """Generate .NET code example."""
        return f'''
using Stripe;

var options = new StripeClientOptions
{{
    ApiKey = "sk_test_..."
}};
var client = new StripeClient(options);

// {capability.description}
// Parameters: {json.dumps(capability.input_params, indent=4)}
'''

    def _get_php_example(self, capability: SDKCapability) -> str:
        """Generate PHP code example."""
        return f'''
<?php
require_once('vendor/autoload.php');

\\Stripe\\Stripe::setApiKey('sk_test_...');

// {capability.description}
$result = \\Stripe\\{capability.method_path.replace('.', '::')}::create([
    // Parameters: {json.dumps(capability.input_params, indent=4)}
]);

print_r($result);
'''

    def get_statistics(self) -> Dict[str, Any]:
        """Get SDK bridge statistics."""
        status_counts = {}
        language_counts = {}

        for sdk in self.sdks.values():
            status = sdk.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            language = sdk.language.value
            language_counts[language] = language_counts.get(language, 0) + 1

        return {
            'initialized': self._initialized,
            'total_sdks': len(self.sdks),
            'total_capabilities': len(self.capabilities),
            'status_counts': status_counts,
            'language_counts': language_counts,
            'sdks': [sdk.name for sdk in self.sdks.values()],
            'capabilities': list(self.capabilities.keys())
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export SDK bridge state to dictionary."""
        return {
            'sdks': {
                name: {
                    'name': sdk.name,
                    'language': sdk.language.value,
                    'path': sdk.path,
                    'version': sdk.version,
                    'status': sdk.status.value,
                    'features': sdk.features
                }
                for name, sdk in self.sdks.items()
            },
            'capabilities': {
                name: {
                    'name': cap.name,
                    'description': cap.description,
                    'method_path': cap.method_path,
                    'input_params': cap.input_params,
                    'output_type': cap.output_type
                }
                for name, cap in self.capabilities.items()
            },
            'statistics': self.get_statistics()
        }
