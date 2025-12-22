"""
Payment Agent

Specialized agent for handling payment processing operations.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.base_agent import (
    BaseAgent,
    AgentState,
    AgentMessage,
    AgentCapability,
    MessageType,
    AgentPriority
)


class PaymentAgent(BaseAgent):
    """
    Agent specialized for payment processing.

    Capabilities:
    - Create payment intents
    - Process payments
    - Handle refunds
    - Manage disputes
    - Payment status tracking
    """

    def __init__(self, api_bridge=None):
        super().__init__(
            name="PaymentAgent",
            description="Handles all payment processing operations"
        )
        self.api_bridge = api_bridge

        # Register capabilities
        self._register_capabilities()

        # Payment tracking
        self.pending_payments: Dict[str, Dict] = {}
        self.processed_today = 0
        self.total_amount_today = 0

    def _register_capabilities(self):
        """Register agent capabilities."""
        self.register_capability(AgentCapability(
            name="create_payment",
            description="Create a new payment intent",
            input_schema={
                "amount": "integer (required)",
                "currency": "string (required)",
                "customer_id": "string (optional)",
                "metadata": "object (optional)"
            },
            output_schema={"payment_id": "string", "status": "string"}
        ))

        self.register_capability(AgentCapability(
            name="process_payment",
            description="Process a pending payment",
            input_schema={"payment_id": "string (required)"},
            output_schema={"success": "boolean", "status": "string"}
        ))

        self.register_capability(AgentCapability(
            name="create_refund",
            description="Create a refund for a payment",
            input_schema={
                "payment_id": "string (required)",
                "amount": "integer (optional)",
                "reason": "string (optional)"
            },
            output_schema={"refund_id": "string", "status": "string"}
        ))

        self.register_capability(AgentCapability(
            name="get_payment_status",
            description="Get status of a payment",
            input_schema={"payment_id": "string (required)"},
            output_schema={"status": "string", "details": "object"}
        ))

        # Register handlers
        self.register_handler("create_payment", self._handle_create_payment)
        self.register_handler("process_payment", self._handle_process_payment)
        self.register_handler("create_refund", self._handle_create_refund)
        self.register_handler("get_payment_status", self._handle_get_status)

    async def initialize(self) -> bool:
        """Initialize the payment agent."""
        self.logger.info("Initializing PaymentAgent")
        return True

    async def shutdown(self):
        """Shutdown the payment agent."""
        self.logger.info("Shutting down PaymentAgent")

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

    async def _handle_create_payment(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle create payment request."""
        payload = message.payload
        amount = payload.get('amount')
        currency = payload.get('currency', 'usd')
        customer_id = payload.get('customer_id')
        metadata = payload.get('metadata', {})

        if not amount:
            return {'error': 'Amount is required'}

        try:
            # Create payment via API bridge if available
            if self.api_bridge:
                response = await self.api_bridge.create_payment_intent(
                    amount=amount,
                    currency=currency,
                    customer=customer_id,
                    metadata=metadata
                )
                payment_id = response.data.get('id', f'pi_{datetime.now().timestamp()}')
            else:
                # Simulated payment creation
                payment_id = f"pi_sim_{int(datetime.now().timestamp())}"

            # Track payment
            self.pending_payments[payment_id] = {
                'amount': amount,
                'currency': currency,
                'customer_id': customer_id,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }

            self.processed_today += 1
            self.total_amount_today += amount

            return {
                'success': True,
                'payment_id': payment_id,
                'status': 'pending',
                'amount': amount,
                'currency': currency
            }

        except Exception as e:
            self.logger.error(f"Error creating payment: {e}")
            return {'error': str(e)}

    async def _handle_process_payment(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle process payment request."""
        payment_id = message.payload.get('payment_id')

        if not payment_id:
            return {'error': 'Payment ID is required'}

        if payment_id in self.pending_payments:
            self.pending_payments[payment_id]['status'] = 'processing'

            # Simulate processing delay
            await asyncio.sleep(0.1)

            self.pending_payments[payment_id]['status'] = 'succeeded'
            self.pending_payments[payment_id]['processed_at'] = datetime.now().isoformat()

            return {
                'success': True,
                'payment_id': payment_id,
                'status': 'succeeded'
            }

        return {'error': 'Payment not found'}

    async def _handle_create_refund(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle create refund request."""
        payload = message.payload
        payment_id = payload.get('payment_id')
        amount = payload.get('amount')
        reason = payload.get('reason', 'requested_by_customer')

        if not payment_id:
            return {'error': 'Payment ID is required'}

        try:
            if self.api_bridge:
                response = await self.api_bridge.create_refund(
                    payment_intent=payment_id,
                    amount=amount
                )
                refund_id = response.data.get('id', f're_{datetime.now().timestamp()}')
            else:
                refund_id = f"re_sim_{int(datetime.now().timestamp())}"

            return {
                'success': True,
                'refund_id': refund_id,
                'payment_id': payment_id,
                'amount': amount,
                'reason': reason,
                'status': 'succeeded'
            }

        except Exception as e:
            self.logger.error(f"Error creating refund: {e}")
            return {'error': str(e)}

    async def _handle_get_status(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle get payment status request."""
        payment_id = message.payload.get('payment_id')

        if not payment_id:
            return {'error': 'Payment ID is required'}

        if payment_id in self.pending_payments:
            return {
                'found': True,
                'payment_id': payment_id,
                **self.pending_payments[payment_id]
            }

        return {'found': False, 'payment_id': payment_id}

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily statistics."""
        return {
            'processed_today': self.processed_today,
            'total_amount_today': self.total_amount_today,
            'pending_count': len([
                p for p in self.pending_payments.values()
                if p['status'] == 'pending'
            ]),
            'succeeded_count': len([
                p for p in self.pending_payments.values()
                if p['status'] == 'succeeded'
            ])
        }
