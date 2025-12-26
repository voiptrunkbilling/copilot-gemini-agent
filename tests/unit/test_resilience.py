"""
Tests for resilience module.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from copilot_agent.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryPolicy,
    REVIEWER_RETRY_POLICY,
    VISION_RETRY_POLICY,
    UI_ACTION_RETRY_POLICY,
    create_reviewer_circuit,
    create_vision_circuit,
    create_ui_circuit,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_initial_state_closed(self):
        """Test circuit starts closed."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open
    
    def test_record_success(self):
        """Test recording success."""
        cb = CircuitBreaker(name="test")
        run_async(cb.record_success())
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
    
    def test_record_failure(self):
        """Test recording failure."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test", config=config)
        
        run_async(cb.record_failure())
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED
        
        run_async(cb.record_failure())
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_opens_on_threshold(self):
        """Test circuit opens when threshold reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test", config=config)
        
        run_async(cb.record_failure())
        run_async(cb.record_failure())
        run_async(cb.record_failure())
        
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
    
    def test_can_execute_when_closed(self):
        """Test can_execute returns True when closed."""
        cb = CircuitBreaker(name="test")
        
        assert run_async(cb.can_execute()) is True
    
    def test_can_execute_when_open(self):
        """Test can_execute returns False when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(name="test", config=config)
        run_async(cb.record_failure())
        
        assert cb.is_open
        assert run_async(cb.can_execute()) is False
    
    def test_half_open_after_timeout(self):
        """Test circuit becomes half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,  # 100ms
        )
        cb = CircuitBreaker(name="test", config=config)
        
        run_async(cb.record_failure())
        assert cb.is_open
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should be able to execute (half-open)
        can_exec = run_async(cb.can_execute())
        assert can_exec is True
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_success_closes_half_open(self):
        """Test success in half-open state closes circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=0.01,
        )
        cb = CircuitBreaker(name="test", config=config)
        
        run_async(cb.record_failure())
        time.sleep(0.02)
        run_async(cb.can_execute())  # Triggers half-open
        
        assert cb.state == CircuitState.HALF_OPEN
        
        run_async(cb.record_success())
        assert cb.state == CircuitState.CLOSED
    
    def test_reset(self):
        """Test circuit reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(name="test", config=config)
        run_async(cb.record_failure())
        
        assert cb.is_open
        
        cb.reset()  # reset is sync
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestRetryPolicy:
    """Tests for RetryPolicy."""
    
    def test_default_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.multiplier == 2.0
    
    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            initial_delay=1.0,
            multiplier=2.0,
            max_delay=100.0,
            jitter=0.0,  # Disable jitter for predictable test
        )
        
        # Note: get_delay uses attempt number (0, 1, 2...)
        delay0 = policy.get_delay(0)  # 1.0
        delay1 = policy.get_delay(1)  # 2.0
        delay2 = policy.get_delay(2)  # 4.0
        
        assert 0.9 <= delay0 <= 1.1  # Allow small variance
        assert 1.9 <= delay1 <= 2.1
        assert 3.9 <= delay2 <= 4.1
    
    def test_get_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        policy = RetryPolicy(
            initial_delay=10.0,
            multiplier=10.0,
            max_delay=30.0,
            jitter=0.0,
        )
        
        # Would be 100.0 but capped at 30.0
        delay = policy.get_delay(2)
        assert delay <= 30.0
    
    def test_is_retryable(self):
        """Test error retryability check."""
        policy = RetryPolicy(
            retryable_exceptions=(ValueError, TypeError),
            non_retryable_exceptions=(RuntimeError,),
        )
        
        assert policy.is_retryable(ValueError("test")) is True
        assert policy.is_retryable(TypeError("test")) is True
        assert policy.is_retryable(RuntimeError("test")) is False
        assert policy.is_retryable(KeyError("test")) is False  # Not in retryable


class TestPreconfiguredPolicies:
    """Tests for pre-configured retry policies."""
    
    def test_reviewer_policy(self):
        """Test reviewer retry policy configuration."""
        assert REVIEWER_RETRY_POLICY.max_retries == 3
        assert REVIEWER_RETRY_POLICY.initial_delay == 1.0
    
    def test_vision_policy(self):
        """Test vision retry policy configuration."""
        assert VISION_RETRY_POLICY.max_retries == 2
        assert VISION_RETRY_POLICY.initial_delay == 0.5
    
    def test_ui_action_policy(self):
        """Test UI action retry policy configuration."""
        assert UI_ACTION_RETRY_POLICY.max_retries == 2
        assert UI_ACTION_RETRY_POLICY.initial_delay == 0.2


class TestCircuitFactories:
    """Tests for circuit breaker factory functions."""
    
    def test_create_reviewer_circuit(self):
        """Test reviewer circuit creation."""
        cb = create_reviewer_circuit()
        
        assert cb.name == "reviewer"
        assert cb.config.failure_threshold == 5
    
    def test_create_vision_circuit(self):
        """Test vision circuit creation."""
        cb = create_vision_circuit()
        
        assert cb.name == "vision"
        assert cb.config.failure_threshold == 3
    
    def test_create_ui_circuit(self):
        """Test UI circuit creation."""
        cb = create_ui_circuit()
        
        assert cb.name == "ui"
        assert cb.config.failure_threshold == 5
