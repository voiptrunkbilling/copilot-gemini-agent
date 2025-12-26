"""
Circuit breaker and retry policies.

Implements robust retry logic with exponential backoff and circuit breakers
for handling transient failures in external services (Gemini API, UI automation).
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, TypeVar, Any
from functools import wraps

from copilot_agent.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1  # Random jitter factor (0-1)
    
    # Specific error handling
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        import random
        
        delay = min(
            self.initial_delay * (self.multiplier ** attempt),
            self.max_delay
        )
        
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        # Non-retryable takes precedence
        if isinstance(error, self.non_retryable_exceptions):
            return False
        
        return isinstance(error, self.retryable_exceptions)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5      # Failures to open circuit
    success_threshold: int = 2      # Successes to close circuit
    timeout_seconds: float = 30.0   # Time before trying half-open
    
    # Monitoring
    window_seconds: float = 60.0    # Time window for failure counting


@dataclass
class CircuitBreakerState:
    """Runtime state of circuit breaker."""
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation.
    
    Prevents repeated calls to a failing service by "opening" the circuit
    after a threshold of failures, then gradually testing recovery.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self._state.state == CircuitState.OPEN
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._state.failure_count
    
    async def can_execute(self) -> bool:
        """
        Check if execution is allowed.
        
        Returns:
            True if call can proceed, False if circuit is open
        """
        async with self._lock:
            now = time.time()
            
            if self._state.state == CircuitState.CLOSED:
                return True
            
            if self._state.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._state.opened_at:
                    elapsed = now - self._state.opened_at
                    if elapsed >= self.config.timeout_seconds:
                        # Transition to half-open
                        self._state.state = CircuitState.HALF_OPEN
                        self._state.half_open_successes = 0
                        logger.info(
                            "Circuit half-open",
                            circuit=self.name,
                            elapsed=elapsed,
                        )
                        return True
                return False
            
            # Half-open: allow limited calls
            return True
    
    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            now = time.time()
            self._state.last_success_time = now
            self._state.success_count += 1
            
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.half_open_successes += 1
                
                if self._state.half_open_successes >= self.config.success_threshold:
                    # Fully recovered - close circuit
                    self._state.state = CircuitState.CLOSED
                    self._state.failure_count = 0
                    logger.info("Circuit closed (recovered)", circuit=self.name)
            
            elif self._state.state == CircuitState.CLOSED:
                # Reset failure count on success (sliding window)
                if self._state.last_failure_time:
                    elapsed = now - self._state.last_failure_time
                    if elapsed > self.config.window_seconds:
                        self._state.failure_count = 0
    
    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        async with self._lock:
            now = time.time()
            self._state.last_failure_time = now
            self._state.failure_count += 1
            
            if self._state.state == CircuitState.HALF_OPEN:
                # Failure during recovery - back to open
                self._state.state = CircuitState.OPEN
                self._state.opened_at = now
                logger.warning(
                    "Circuit re-opened (half-open failure)",
                    circuit=self.name,
                    error=str(error) if error else None,
                )
            
            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.config.failure_threshold:
                    # Too many failures - open circuit
                    self._state.state = CircuitState.OPEN
                    self._state.opened_at = now
                    logger.warning(
                        "Circuit opened",
                        circuit=self.name,
                        failures=self._state.failure_count,
                        error=str(error) if error else None,
                    )
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitBreakerState()
        logger.info("Circuit reset", circuit=self.name)
    
    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "last_failure": self._state.last_failure_time,
            "last_success": self._state.last_success_time,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, message: str = "Circuit is open"):
        self.circuit_name = circuit_name
        super().__init__(f"{circuit_name}: {message}")


class RetryExhaustedError(Exception):
    """Raised when all retries are exhausted."""
    
    def __init__(self, attempts: int, last_error: Optional[Exception] = None):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"All {attempts} retry attempts exhausted")


async def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    circuit: Optional[CircuitBreaker] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Execute function with retry and backoff.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        policy: Retry policy configuration
        circuit: Optional circuit breaker
        on_retry: Callback on each retry (attempt, error)
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        CircuitOpenError: If circuit breaker is open
        RetryExhaustedError: If all retries exhausted
    """
    policy = policy or RetryPolicy()
    last_error: Optional[Exception] = None
    
    for attempt in range(policy.max_retries + 1):
        # Check circuit breaker
        if circuit:
            if not await circuit.can_execute():
                raise CircuitOpenError(circuit.name)
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - record it
            if circuit:
                await circuit.record_success()
            
            return result
            
        except Exception as e:
            last_error = e
            
            # Record failure
            if circuit:
                await circuit.record_failure(e)
            
            # Check if retryable
            if not policy.is_retryable(e):
                logger.warning(
                    "Non-retryable error",
                    error=str(e),
                    attempt=attempt + 1,
                )
                raise
            
            # Check if more retries available
            if attempt >= policy.max_retries:
                break
            
            # Calculate delay
            delay = policy.get_delay(attempt)
            
            logger.warning(
                "Retry scheduled",
                attempt=attempt + 1,
                max_retries=policy.max_retries,
                delay=delay,
                error=str(e),
            )
            
            # Callback
            if on_retry:
                on_retry(attempt + 1, e)
            
            # Wait before retry
            await asyncio.sleep(delay)
    
    raise RetryExhaustedError(policy.max_retries + 1, last_error)


def retry_sync_with_backoff(
    func: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Synchronous version of retry with backoff.
    
    Args:
        func: Function to execute
        *args: Function arguments
        policy: Retry policy configuration
        on_retry: Callback on each retry (attempt, error)
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryExhaustedError: If all retries exhausted
    """
    policy = policy or RetryPolicy()
    last_error: Optional[Exception] = None
    
    for attempt in range(policy.max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            last_error = e
            
            if not policy.is_retryable(e):
                raise
            
            if attempt >= policy.max_retries:
                break
            
            delay = policy.get_delay(attempt)
            
            logger.warning(
                "Retry scheduled (sync)",
                attempt=attempt + 1,
                max_retries=policy.max_retries,
                delay=delay,
                error=str(e),
            )
            
            if on_retry:
                on_retry(attempt + 1, e)
            
            time.sleep(delay)
    
    raise RetryExhaustedError(policy.max_retries + 1, last_error)


# Decorator versions

def with_retry(
    policy: Optional[RetryPolicy] = None,
    circuit: Optional[CircuitBreaker] = None,
):
    """
    Decorator for async functions with retry.
    
    Args:
        policy: Retry policy
        circuit: Optional circuit breaker
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func, *args,
                policy=policy,
                circuit=circuit,
                **kwargs,
            )
        return wrapper
    return decorator


def with_retry_sync(policy: Optional[RetryPolicy] = None):
    """
    Decorator for sync functions with retry.
    
    Args:
        policy: Retry policy
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_sync_with_backoff(
                func, *args,
                policy=policy,
                **kwargs,
            )
        return wrapper
    return decorator


# Pre-configured policies

REVIEWER_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    multiplier=2.0,
    jitter=0.1,
)

VISION_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    initial_delay=0.5,
    max_delay=10.0,
    multiplier=2.0,
    jitter=0.1,
)

UI_ACTION_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    initial_delay=0.2,
    max_delay=2.0,
    multiplier=1.5,
    jitter=0.05,
)

# Pre-configured circuit breakers

def create_reviewer_circuit() -> CircuitBreaker:
    """Create circuit breaker for reviewer API."""
    return CircuitBreaker(
        name="reviewer",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
        ),
    )


def create_vision_circuit() -> CircuitBreaker:
    """Create circuit breaker for vision API."""
    return CircuitBreaker(
        name="vision",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=30.0,
        ),
    )


def create_ui_circuit() -> CircuitBreaker:
    """Create circuit breaker for UI automation."""
    return CircuitBreaker(
        name="ui",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=10.0,
        ),
    )
