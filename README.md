# FurEver: Stripe Connect Embedded Components Demo

FurEver is a vertical SaaS grooming platform for pet salons to manage their e2e business operations. FurEver wants to provide access to Stripe products and UIs directly in their website, at a fraction of the engineering cost, using [Stripe Connect](https://stripe.com/connect) and [Stripe Connect embedded components](https://docs.stripe.com/connect/get-started-connect-embedded-components).

**See a live version on [furever.dev](https://furever.dev).**

<img src="public/cover.png">

## Stripe OpenCog Integration Platform

This project now includes a comprehensive integration platform that combines Stripe's ecosystem with OpenCog's cognitive architecture, Agent-Zero's multi-agent orchestration system, and the **Tensor Logic Framework** for hybrid neural-symbolic reasoning.

### Platform Architecture

```
platform/
├── opencog/           # Atomspace knowledge representation
│   ├── atomspace.py   # Knowledge graph for Stripe entities
│   ├── knowledge_base.py  # Entity definitions and SDK mappings
│   └── reasoning.py   # Inference and reasoning engine
│
├── tensor_logic/      # 🧠 Tensor Logic Framework (NEW)
│   ├── tensor_space.py        # Gradient-based learning
│   ├── symbolic_integration.py # Logic tensors & fuzzy logic
│   ├── tensor_atoms.py        # Tensor-valued knowledge graphs
│   ├── gradient_reasoner.py   # Hybrid neural-symbolic inference
│   └── tensor_bridge.py       # Integration with OpenCog
│
├── agent-zero/        # Multi-agent orchestration system
│   ├── base_agent.py  # Base agent framework
│   ├── orchestrator.py    # Agent lifecycle management
│   └── agent_registry.py  # Agent discovery and registration
│
├── bridges/           # Integration bridges
│   ├── cognitive_bridge.py  # OpenCog <-> Agent-Zero bridge
│   ├── sdk_bridge.py  # Unified SDK management
│   ├── plugin_bridge.py   # Unified plugin management
│   └── api_bridge.py  # Stripe API with rate limiting
│
├── core/              # Main entry points
│   ├── unified_interface.py  # Single entry point for all operations
│   ├── config.py      # Platform configuration
│   └── events.py      # Event bus system
│
├── agents/            # Specialized agents
│   ├── payment_agent.py   # Transaction processing
│   ├── integration_agent.py   # SDK/plugin health monitoring
│   ├── security_agent.py  # Fraud detection and compliance
│   ├── analytics_agent.py # Transaction analytics
│   └── monitoring_agent.py    # System health monitoring
│
└── stripe-ecosystem/  # Integrated Stripe repositories
    ├── sdks/          # Official Stripe SDKs
    │   ├── stripe-node/
    │   ├── stripe-python/
    │   ├── stripe-ruby/
    │   ├── stripe-go/
    │   ├── stripe-java/
    │   ├── stripe-dotnet/
    │   ├── stripe-js/
    │   └── stripe-react-native/
    ├── plugins/       # Stripe plugins
    │   ├── stripe-terminal-react-native/
    │   ├── stripe-identity-react-native/
    │   └── stripe-php/
    └── apps/          # Stripe applications
        ├── stripe-ios/
        ├── stripe-android/
        └── stripe-cli/
```

### Key Components

#### 🧠 Tensor Logic Framework (NEW)

A deep unification of deep learning and symbolic AI that combines gradient-based learning with logical reasoning.

**Quick Start:**
```python
from platform.tensor_logic import TensorAtomspace, GradientReasoner
from opencog.atomspace import AtomType

# Create tensor atomspace with gradient-based learning
atomspace = TensorAtomspace(learning_rate=0.1)

# Add payment atom
payment = atomspace.add_tensor_node(
    AtomType.PAYMENT,
    "payment_123",
    value={'amount': 5000},
    strength=0.5
)

# Train on historical data
training_data = [(payment.id, 0.9, 0.95)]  # (id, target_strength, target_confidence)
loss_history = atomspace.train_atoms(training_data, epochs=100)

# Use gradient-based reasoning
reasoner = GradientReasoner(atomspace)
results = reasoner.forward_chain(max_iterations=5)
```

**Features:**
- 🔄 Gradient-based optimization of truth values
- 🧮 Differentiable logical inference
- 📊 Learn rules directly from data  
- 🎯 Hybrid neural-symbolic reasoning
- ✅ 89/91 tests passing

**Documentation:** [Tensor Logic Framework Guide](docs/TENSOR_LOGIC.md)

#### UnifiedStripeInterface
Single entry point for all Stripe ecosystem operations:
```python
from platform.core import UnifiedStripeInterface

interface = UnifiedStripeInterface()
await interface.initialize()

# Create a payment
result = await interface.create_payment(
    amount=5000,
    currency='usd',
    customer_id='cus_xxx'
)

# Query knowledge base
sdk_recommendation = await interface.recommend_sdk(
    language='python',
    features=['async', 'webhooks']
)
```

#### CognitiveBridge
Connects OpenCog's knowledge representation with Agent-Zero:
- Bidirectional knowledge flow
- Agent-to-knowledge synchronization
- Reasoning integration
- Event-driven updates

#### SDKBridge
Unified SDK management across all platforms:
- SDK discovery and registration
- Capability mapping
- Code examples generation
- Version management

#### PluginBridge
Unified plugin management:
- Plugin lifecycle management
- Event handling
- Configuration management

#### APIBridge
Stripe API access with enterprise features:
- Token bucket rate limiting
- Request tracking and history
- Automatic retry with backoff
- Error handling

### Specialized Agents

| Agent | Description |
|-------|-------------|
| **PaymentAgent** | Transaction processing, refunds, disputes |
| **IntegrationAgent** | SDK/plugin health monitoring |
| **SecurityAgent** | Fraud detection, compliance, blocklists |
| **AnalyticsAgent** | Transaction analytics, reporting |
| **MonitoringAgent** | System health, resource monitoring |

### Quick Start

```python
import asyncio
from platform.core import UnifiedStripeInterface, PlatformConfig

async def main():
    # Create configuration
    config = PlatformConfig()
    config.stripe.api_key = 'sk_test_...'

    # Initialize platform
    interface = UnifiedStripeInterface(config)
    await interface.initialize()
    await interface.start()

    # Use the platform
    status = interface.get_status()
    print(f"Platform status: {status['status']}")

    # Create a payment
    result = await interface.create_payment(
        amount=2500,
        currency='usd'
    )
    print(f"Payment created: {result.data}")

    # Get SDK recommendation
    sdk = await interface.recommend_sdk(
        language='node',
        features=['typescript', 'webhooks']
    )
    print(f"Recommended SDK: {sdk.data}")

    # Shutdown
    await interface.stop()

asyncio.run(main())
```

---

## Features

FurEver showcases the integration between a platform's website, [Stripe Connect](https://stripe.com/connect), and [Stripe Connect embedded components](https://docs.stripe.com/connect/get-started-connect-embedded-components). Users sign up within the platform's website and through the process, a corresponding Stripe unified account is created with the following configuration:

- Stripe owns loss liability
- Platform owns pricing
- Stripe is onboarding owner
- The connected account has no access to the Stripe dashboard

The user will then onboard with Stripe via embedded onboarding. Thereafter, Connect embedded components will provide the UI surfaces for account management and dashboard UI elements with just a few lines of code. The demo website also uses the Stripe API to create test payments and payouts. This app also contains a basic authentication system.

FurEver makes use of the following [Connect embedded components](https://docs.stripe.com/connect/supported-embedded-components):

- `<ConnectOnboarding />` enables an embedded onboarding experience without redirecting users to Stripe hosted onboarding.
- `<ConnectPayments />` provides a list to display Stripe payments, refunds, and disputes. This also includes handling list filtering, pagination, and CSV exports.
- `<ConnectPayouts />` provides a list to display Stripe payouts and balance. This also includes handling list filtering, pagination, and CSV exports.
- `<ConnectAccountManagement />` allows users to edit their Stripe account settings without navigating to the Stripe dashboard.
- `<ConnectNotificationBanner />` displays a list of current and future risk requirements an account needs to resolve.
- `<ConnectDocuments />` displays a list of tax invoice documents.
- `<ConnectTaxSettings />` allows users to [set up Stripe Tax](https://docs.stripe.com/tax/set-up).
- `<ConnectTaxRegistrations />` allows users to control their tax compliance settings.

Additionally, the following preview components are also used:

- `<ConnectCapitalOverview />` **preview** allows users to check their eligibility for financing, get an overview of their in-progress financing, and access the reporting page to review paydown transactions.
- `<ConnectFinancialAccount />` **preview** renders a view of an individual [Financial Account](https://docs.stripe.com/api/treasury/financial_accounts)
- `<ConnectFinancialAccountTransactions />` **preview** provides a list of transactions associated with a financial account.
- `<ConnectIssuingCardsList />` **preview** provides a list of all the cards issued.

### Architecture

The web application is implemented as as full-stack application using Express, React, Typescript, and Material UI.

This demo is built with

- [Next.js](https://nextjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)

To integrate Stripe Connect embedded components, check out our [documentation](https://docs.stripe.com/connect/get-started-connect-embedded-components).

1. [`hooks/useConnect.ts`](client/hooks/Connect.tsx) shows the client side integration with Connect embedded components.
2. [`api/account_session/route.ts`](server/routes/stripe.ts) shows the server request to `v1/account_sessions`.

## Requirements

You'll need a Stripe account to manage pet salon onboarding and payments:

- [Sign up for free](https://dashboard.stripe.com/register), then [enable Connect](https://dashboard.stripe.com/account/applications/settings) by filling in your Connect settings.
- Fill in the necessary information in the **Branding** section in [Connect settings](https://dashboard.stripe.com/test/settings/connect).

### Getting started

Install dependencies using npm (or yarn):

```
yarn
```

Copy the environment file and add your own [Stripe API keys](https://dashboard.stripe.com/account/apikeys):

```
cp .env.example .env
```

Install MongoDB Community Edition. Refer to the [official documentation](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/). Then, run MongoDB:

```
brew tap mongodb/brew && brew update
brew install mongodb-community@7.0
brew services start mongodb-community@7.0
```

Run the app:

```
yarn dev
```

Go to `http://localhost:{process.env.PORT}` in your browser to start using the app.

To test events sent to your event handler, you can run this command in a separate terminal:

```
stripe listen --forward-to localhost:3000/api/webhooks
```

Then, trigger a test event with:

```
stripe trigger payment_intent.succeeded
```

## Preview components

By default, preview components are turned off in this repository. If you'd like to enable them, make sure to request access to them for your platforms in [the Stripe doc site](https://docs.stripe.com/connect/supported-embedded-components). You can then add this variable to the .env file to activate these components.

```
NEXT_PUBLIC_ENABLE_PREVIEW_COMPONENTS=1
```

## Testing

### Python Tests (Tensor Logic Framework)

Run the comprehensive test suite for the Tensor Logic framework:

```bash
# Run all Tensor Logic tests (89/91 tests passing)
PYTHONPATH=platform:$PYTHONPATH python3 -m pytest platform/tests/tensor_logic/ -v

# Run specific test modules
PYTHONPATH=platform:$PYTHONPATH python3 -m pytest platform/tests/tensor_logic/test_tensor_space.py -v
PYTHONPATH=platform:$PYTHONPATH python3 -m pytest platform/tests/tensor_logic/test_symbolic_integration.py -v
PYTHONPATH=platform:$PYTHONPATH python3 -m pytest platform/tests/tensor_logic/test_gradient_reasoner.py -v
PYTHONPATH=platform:$PYTHONPATH python3 -m pytest platform/tests/tensor_logic/test_integration.py -v
```

**Test Coverage:**
- **tensor_space.py**: 23 tests - Gradient-based learning
- **symbolic_integration.py**: 19 tests - Logic tensors and fuzzy logic
- **tensor_atoms.py**: 21 tests - Tensor-valued knowledge graphs
- **gradient_reasoner.py**: 17 tests - Hybrid reasoning
- **Integration tests**: 11 tests - End-to-end workflows

### TypeScript/JavaScript Tests

```bash
yarn lint
yarn build
```

## Documentation

For more detailed documentation on the platform integration, see:

- [Tensor Logic Framework Guide](docs/TENSOR_LOGIC.md) **NEW**
- [Feature Ecosystem Documentation](docs/FEATURE_ECOSYSTEM.md)

## License

MIT
