"""
API Bridge

Provides unified Stripe API access with rate limiting,
request tracking, and error handling.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import time
import hashlib
import json


class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class RequestStatus(Enum):
    """Status of an API request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


class ErrorType(Enum):
    """Types of API errors."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"


@dataclass
class APIRequest:
    """API request structure."""
    id: str = field(default_factory=lambda: hashlib.md5(
        str(time.time()).encode()
    ).hexdigest()[:12])
    endpoint: str = ""
    method: RequestMethod = RequestMethod.GET
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """API response structure."""
    request_id: str
    status_code: int
    data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 100.0
    burst_size: int = 200
    retry_after_seconds: float = 1.0
    max_retries: int = 3


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.tokens = float(self.config.burst_size)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for a request.

        Args:
            tokens: Number of tokens needed

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_and_acquire(
        self,
        tokens: int = 1,
        timeout: float = 30.0
    ) -> bool:
        """
        Wait for tokens and acquire them.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time

        Returns:
            True if tokens acquired within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.acquire(tokens):
                return True

            # Wait for refill
            wait_time = min(
                self.config.retry_after_seconds,
                timeout - (time.time() - start_time)
            )
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.config.requests_per_second
        self.tokens = min(
            self.config.burst_size,
            self.tokens + new_tokens
        )
        self.last_refill = now

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        self._refill()
        return {
            'available_tokens': self.tokens,
            'max_tokens': self.config.burst_size,
            'requests_per_second': self.config.requests_per_second
        }


class APIBridge:
    """
    Unified bridge for Stripe API access.

    Provides:
    - Rate limiting
    - Request tracking
    - Error handling
    - Retry logic
    - Request logging
    """

    # API endpoints configuration
    ENDPOINTS = {
        # Core API
        "payment_intents": "/v1/payment_intents",
        "customers": "/v1/customers",
        "charges": "/v1/charges",
        "refunds": "/v1/refunds",
        "disputes": "/v1/disputes",

        # Billing
        "subscriptions": "/v1/subscriptions",
        "invoices": "/v1/invoices",
        "products": "/v1/products",
        "prices": "/v1/prices",

        # Connect
        "accounts": "/v1/accounts",
        "account_links": "/v1/account_links",
        "transfers": "/v1/transfers",
        "payouts": "/v1/payouts",

        # Other
        "balance": "/v1/balance",
        "events": "/v1/events",
        "files": "/v1/files",
        "webhooks": "/v1/webhook_endpoints"
    }

    def __init__(
        self,
        api_key: str = "",
        api_version: str = "2023-10-16",
        base_url: str = "https://api.stripe.com",
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.logger = logging.getLogger("api_bridge")

        # Request tracking
        self.request_history: List[APIRequest] = []
        self.max_history_size = 1000

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0

        # Callbacks
        self.on_request: Optional[Callable[[APIRequest], None]] = None
        self.on_response: Optional[Callable[[APIResponse], None]] = None
        self.on_error: Optional[Callable[[APIRequest, Exception], None]] = None

    async def request(
        self,
        endpoint: str,
        method: RequestMethod = RequestMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        timeout: float = 30.0
    ) -> APIResponse:
        """
        Make an API request.

        Args:
            endpoint: API endpoint (name or path)
            method: HTTP method
            params: Query parameters
            body: Request body
            idempotency_key: Idempotency key for POST/PUT
            timeout: Request timeout

        Returns:
            API response
        """
        # Resolve endpoint
        path = self.ENDPOINTS.get(endpoint, endpoint)
        if not path.startswith('/'):
            path = f'/{path}'

        # Create request
        request = APIRequest(
            endpoint=path,
            method=method,
            params=params or {},
            body=body,
            idempotency_key=idempotency_key,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Stripe-Version': self.api_version,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        )

        if idempotency_key:
            request.headers['Idempotency-Key'] = idempotency_key

        self._track_request(request)

        if self.on_request:
            self.on_request(request)

        # Check rate limit
        if not await self.rate_limiter.wait_and_acquire(timeout=timeout):
            request.status = RequestStatus.RATE_LIMITED
            self.rate_limited_requests += 1

            return APIResponse(
                request_id=request.id,
                status_code=429,
                error={
                    'type': ErrorType.RATE_LIMIT.value,
                    'message': 'Rate limit exceeded'
                }
            )

        # Execute request with retry
        return await self._execute_with_retry(request, timeout)

    async def _execute_with_retry(
        self,
        request: APIRequest,
        timeout: float
    ) -> APIResponse:
        """Execute request with retry logic."""
        max_retries = self.rate_limiter.config.max_retries
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                request.status = RequestStatus.IN_PROGRESS
                start_time = time.time()

                # Simulated API call
                response = await self._execute_request(request, timeout)

                elapsed_ms = (time.time() - start_time) * 1000
                response.latency_ms = elapsed_ms

                if response.status_code == 200:
                    request.status = RequestStatus.COMPLETED
                    self.successful_requests += 1
                elif response.status_code == 429:
                    # Rate limited by API
                    request.status = RequestStatus.RETRYING
                    self.rate_limited_requests += 1

                    retry_after = float(
                        response.headers.get('Retry-After', 1.0)
                    )
                    await asyncio.sleep(retry_after)
                    retry_count += 1
                    continue
                else:
                    request.status = RequestStatus.FAILED
                    self.failed_requests += 1

                if self.on_response:
                    self.on_response(response)

                return response

            except Exception as e:
                last_error = e
                retry_count += 1

                if retry_count <= max_retries:
                    request.status = RequestStatus.RETRYING
                    await asyncio.sleep(
                        self.rate_limiter.config.retry_after_seconds * retry_count
                    )
                else:
                    request.status = RequestStatus.FAILED
                    self.failed_requests += 1

                    if self.on_error:
                        self.on_error(request, e)

        return APIResponse(
            request_id=request.id,
            status_code=500,
            error={
                'type': ErrorType.API_ERROR.value,
                'message': str(last_error) if last_error else 'Unknown error'
            }
        )

    async def _execute_request(
        self,
        request: APIRequest,
        timeout: float
    ) -> APIResponse:
        """
        Execute the actual API request.

        This is a simulated implementation. In production, this would
        make actual HTTP requests to the Stripe API.
        """
        # Simulated response based on endpoint and method
        url = f"{self.base_url}{request.endpoint}"

        # Simulate API response
        if request.method == RequestMethod.POST:
            data = {
                'id': f"obj_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}",
                'object': request.endpoint.split('/')[-1].rstrip('s'),
                'created': int(time.time()),
                **request.body or {}
            }
        elif request.method == RequestMethod.GET:
            if 'id' in str(request.params):
                data = {
                    'id': request.params.get('id', 'obj_example'),
                    'object': request.endpoint.split('/')[-1].rstrip('s')
                }
            else:
                data = {
                    'object': 'list',
                    'url': request.endpoint,
                    'has_more': False,
                    'data': []
                }
        else:
            data = {'success': True}

        return APIResponse(
            request_id=request.id,
            status_code=200,
            data=data,
            headers={
                'Request-Id': f'req_{request.id}',
                'Stripe-Version': self.api_version
            }
        )

    def _track_request(self, request: APIRequest):
        """Track request in history."""
        self.total_requests += 1
        self.request_history.append(request)

        # Trim history if needed
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]

    # Convenience methods for common operations

    async def create_payment_intent(
        self,
        amount: int,
        currency: str,
        **kwargs
    ) -> APIResponse:
        """Create a payment intent."""
        return await self.request(
            endpoint="payment_intents",
            method=RequestMethod.POST,
            body={'amount': amount, 'currency': currency, **kwargs}
        )

    async def retrieve_payment_intent(self, payment_intent_id: str) -> APIResponse:
        """Retrieve a payment intent."""
        return await self.request(
            endpoint=f"payment_intents/{payment_intent_id}",
            method=RequestMethod.GET
        )

    async def create_customer(self, **kwargs) -> APIResponse:
        """Create a customer."""
        return await self.request(
            endpoint="customers",
            method=RequestMethod.POST,
            body=kwargs
        )

    async def list_customers(self, limit: int = 10, **kwargs) -> APIResponse:
        """List customers."""
        return await self.request(
            endpoint="customers",
            method=RequestMethod.GET,
            params={'limit': limit, **kwargs}
        )

    async def create_subscription(
        self,
        customer: str,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> APIResponse:
        """Create a subscription."""
        return await self.request(
            endpoint="subscriptions",
            method=RequestMethod.POST,
            body={'customer': customer, 'items': items, **kwargs}
        )

    async def create_refund(
        self,
        payment_intent: str,
        amount: Optional[int] = None,
        **kwargs
    ) -> APIResponse:
        """Create a refund."""
        body = {'payment_intent': payment_intent, **kwargs}
        if amount:
            body['amount'] = amount
        return await self.request(
            endpoint="refunds",
            method=RequestMethod.POST,
            body=body
        )

    async def get_balance(self) -> APIResponse:
        """Get account balance."""
        return await self.request(
            endpoint="balance",
            method=RequestMethod.GET
        )

    def get_endpoint_url(self, endpoint: str) -> str:
        """Get full URL for an endpoint."""
        path = self.ENDPOINTS.get(endpoint, endpoint)
        if not path.startswith('/'):
            path = f'/{path}'
        return f"{self.base_url}{path}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get API bridge statistics."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rate_limited_requests': self.rate_limited_requests,
            'success_rate': (
                self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
            'rate_limiter': self.rate_limiter.get_status(),
            'recent_requests': len(self.request_history),
            'api_version': self.api_version
        }

    def get_recent_requests(
        self,
        limit: int = 10,
        status: Optional[RequestStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get recent requests."""
        requests = self.request_history[-limit:]

        if status:
            requests = [r for r in requests if r.status == status]

        return [
            {
                'id': r.id,
                'endpoint': r.endpoint,
                'method': r.method.value,
                'status': r.status.value,
                'created_at': r.created_at.isoformat()
            }
            for r in requests
        ]

    def clear_history(self):
        """Clear request history."""
        self.request_history.clear()
