"""
Analytics Agent

Specialized agent for transaction analytics and reporting.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_zero.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentCapability
)


class TimeRange(Enum):
    """Time ranges for analytics."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class MetricType(Enum):
    """Types of metrics."""
    VOLUME = "volume"
    AMOUNT = "amount"
    COUNT = "count"
    AVERAGE = "average"
    SUCCESS_RATE = "success_rate"


@dataclass
class Transaction:
    """Transaction record for analytics."""
    id: str
    amount: int
    currency: str
    status: str
    customer_id: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report."""
    report_type: str
    time_range: TimeRange
    start_date: datetime
    end_date: datetime
    metrics: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)


class AnalyticsAgent(BaseAgent):
    """
    Agent for transaction analytics.

    Capabilities:
    - Generate transaction reports
    - Calculate metrics
    - Trend analysis
    - Revenue analytics
    """

    def __init__(self):
        super().__init__(
            name="AnalyticsAgent",
            description="Transaction analytics and reporting"
        )

        # Analytics data storage
        self.transactions: List[Transaction] = []
        self.reports: List[AnalyticsReport] = []

        # Aggregated metrics cache
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None

        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        self.register_capability(AgentCapability(
            name="record_transaction",
            description="Record a transaction for analytics",
            input_schema={
                "id": "string",
                "amount": "integer",
                "currency": "string",
                "status": "string",
                "customer_id": "string (optional)"
            }
        ))

        self.register_capability(AgentCapability(
            name="get_summary",
            description="Get summary metrics",
            input_schema={"time_range": "string"},
            output_schema={"metrics": "object"}
        ))

        self.register_capability(AgentCapability(
            name="get_revenue_report",
            description="Get revenue report",
            input_schema={"time_range": "string"},
            output_schema={"report": "object"}
        ))

        self.register_capability(AgentCapability(
            name="get_trends",
            description="Get trend analysis",
            input_schema={"metric": "string", "time_range": "string"},
            output_schema={"trends": "array"}
        ))

        self.register_capability(AgentCapability(
            name="get_top_customers",
            description="Get top customers by spend",
            input_schema={"limit": "integer"},
            output_schema={"customers": "array"}
        ))

        # Register handlers
        self.register_handler("record_transaction", self._handle_record)
        self.register_handler("get_summary", self._handle_get_summary)
        self.register_handler("get_revenue_report", self._handle_revenue_report)
        self.register_handler("get_trends", self._handle_get_trends)
        self.register_handler("get_top_customers", self._handle_top_customers)

    async def initialize(self) -> bool:
        """Initialize the analytics agent."""
        self.logger.info("Initializing AnalyticsAgent")
        return True

    async def shutdown(self):
        """Shutdown the analytics agent."""
        self.logger.info("Shutting down AnalyticsAgent")

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

    async def _handle_record(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle transaction recording."""
        payload = message.payload

        tx = Transaction(
            id=payload.get('id', str(datetime.now().timestamp())),
            amount=payload.get('amount', 0),
            currency=payload.get('currency', 'usd'),
            status=payload.get('status', 'succeeded'),
            customer_id=payload.get('customer_id'),
            created_at=datetime.now(),
            metadata=payload.get('metadata', {})
        )

        self.transactions.append(tx)

        # Invalidate cache
        self._cache_timestamp = None

        return {'success': True, 'transaction_id': tx.id}

    async def _handle_get_summary(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle summary request."""
        time_range = message.payload.get('time_range', 'day')
        try:
            tr = TimeRange(time_range)
        except ValueError:
            tr = TimeRange.DAY

        start_date = self._get_start_date(tr)
        filtered_txs = [
            tx for tx in self.transactions
            if tx.created_at >= start_date
        ]

        succeeded = [tx for tx in filtered_txs if tx.status == 'succeeded']
        failed = [tx for tx in filtered_txs if tx.status == 'failed']

        total_amount = sum(tx.amount for tx in succeeded)
        avg_amount = total_amount / len(succeeded) if succeeded else 0

        return {
            'time_range': time_range,
            'period_start': start_date.isoformat(),
            'period_end': datetime.now().isoformat(),
            'metrics': {
                'total_transactions': len(filtered_txs),
                'successful_transactions': len(succeeded),
                'failed_transactions': len(failed),
                'success_rate': len(succeeded) / len(filtered_txs) * 100 if filtered_txs else 0,
                'total_amount': total_amount,
                'average_amount': round(avg_amount, 2),
                'unique_customers': len(set(
                    tx.customer_id for tx in filtered_txs if tx.customer_id
                ))
            }
        }

    async def _handle_revenue_report(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle revenue report request."""
        time_range = message.payload.get('time_range', 'month')
        try:
            tr = TimeRange(time_range)
        except ValueError:
            tr = TimeRange.MONTH

        start_date = self._get_start_date(tr)
        filtered_txs = [
            tx for tx in self.transactions
            if tx.created_at >= start_date and tx.status == 'succeeded'
        ]

        # Group by currency
        by_currency = defaultdict(lambda: {'amount': 0, 'count': 0})
        for tx in filtered_txs:
            by_currency[tx.currency]['amount'] += tx.amount
            by_currency[tx.currency]['count'] += 1

        # Group by day
        by_day = defaultdict(lambda: {'amount': 0, 'count': 0})
        for tx in filtered_txs:
            day_key = tx.created_at.strftime('%Y-%m-%d')
            by_day[day_key]['amount'] += tx.amount
            by_day[day_key]['count'] += 1

        report = AnalyticsReport(
            report_type='revenue',
            time_range=tr,
            start_date=start_date,
            end_date=datetime.now(),
            metrics={
                'total_revenue': sum(d['amount'] for d in by_currency.values()),
                'by_currency': dict(by_currency),
                'by_day': dict(by_day),
                'transaction_count': len(filtered_txs)
            }
        )

        self.reports.append(report)

        return {
            'report_type': report.report_type,
            'time_range': time_range,
            'period': {
                'start': start_date.isoformat(),
                'end': datetime.now().isoformat()
            },
            'revenue': report.metrics
        }

    async def _handle_get_trends(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle trends request."""
        metric = message.payload.get('metric', 'volume')
        time_range = message.payload.get('time_range', 'week')

        try:
            tr = TimeRange(time_range)
        except ValueError:
            tr = TimeRange.WEEK

        start_date = self._get_start_date(tr)
        filtered_txs = [
            tx for tx in self.transactions
            if tx.created_at >= start_date
        ]

        # Group by day
        daily_data = defaultdict(list)
        for tx in filtered_txs:
            day_key = tx.created_at.strftime('%Y-%m-%d')
            daily_data[day_key].append(tx)

        trends = []
        for day in sorted(daily_data.keys()):
            day_txs = daily_data[day]
            succeeded = [tx for tx in day_txs if tx.status == 'succeeded']

            if metric == 'volume':
                value = len(day_txs)
            elif metric == 'amount':
                value = sum(tx.amount for tx in succeeded)
            elif metric == 'average':
                value = sum(tx.amount for tx in succeeded) / len(succeeded) if succeeded else 0
            elif metric == 'success_rate':
                value = len(succeeded) / len(day_txs) * 100 if day_txs else 0
            else:
                value = len(day_txs)

            trends.append({
                'date': day,
                'value': round(value, 2)
            })

        # Calculate trend direction
        if len(trends) >= 2:
            first_half = sum(t['value'] for t in trends[:len(trends)//2])
            second_half = sum(t['value'] for t in trends[len(trends)//2:])
            if second_half > first_half * 1.1:
                direction = 'up'
            elif second_half < first_half * 0.9:
                direction = 'down'
            else:
                direction = 'stable'
        else:
            direction = 'insufficient_data'

        return {
            'metric': metric,
            'time_range': time_range,
            'trends': trends,
            'direction': direction
        }

    async def _handle_top_customers(
        self,
        message: AgentMessage
    ) -> Dict[str, Any]:
        """Handle top customers request."""
        limit = message.payload.get('limit', 10)

        # Aggregate by customer
        customer_totals: Dict[str, Dict] = defaultdict(
            lambda: {'amount': 0, 'count': 0}
        )

        for tx in self.transactions:
            if tx.customer_id and tx.status == 'succeeded':
                customer_totals[tx.customer_id]['amount'] += tx.amount
                customer_totals[tx.customer_id]['count'] += 1

        # Sort by amount
        sorted_customers = sorted(
            customer_totals.items(),
            key=lambda x: x[1]['amount'],
            reverse=True
        )[:limit]

        return {
            'top_customers': [
                {
                    'customer_id': cid,
                    'total_amount': data['amount'],
                    'transaction_count': data['count'],
                    'average_amount': round(data['amount'] / data['count'], 2)
                }
                for cid, data in sorted_customers
            ],
            'limit': limit
        }

    def _get_start_date(self, time_range: TimeRange) -> datetime:
        """Get start date for time range."""
        now = datetime.now()

        if time_range == TimeRange.HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return now - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=1)

    def get_stats(self) -> Dict[str, Any]:
        """Get analytics agent stats."""
        return {
            'total_transactions_tracked': len(self.transactions),
            'reports_generated': len(self.reports),
            'unique_customers': len(set(
                tx.customer_id for tx in self.transactions if tx.customer_id
            ))
        }
