"""
Rate Limiter for OpenAI API calls to prevent excessive costs.

This module implements a token bucket rate limiter that tracks:
- Requests per minute (RPM)
- Tokens per minute (TPM)

It also provides cost tracking and estimation based on OpenAI pricing.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from threading import Lock


@dataclass
class RateLimitStats:
    """Statistics for rate limiting and cost tracking."""
    requests_made: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0
    last_reset: float = field(default_factory=time.time)
    
    def reset(self):
        """Reset statistics."""
        self.requests_made = 0
        self.tokens_used = 0
        self.last_reset = time.time()


class RateLimiter:
    """
    Token bucket rate limiter for OpenAI API calls.
    
    Implements both RPM (requests per minute) and TPM (tokens per minute) limits.
    Uses a sliding window approach for accurate rate limiting.
    """
    
    # OpenAI pricing per 1M tokens (as of 2024)
    EMBEDDING_COST_PER_MILLION = {
        "text-embedding-ada-002": 0.10,
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }
    
    def __init__(
        self,
        rpm: int = 60,
        tpm: int = 1_000_000,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize rate limiter.
        
        Args:
            rpm: Requests per minute limit
            tpm: Tokens per minute limit
            embedding_model: Model name for cost calculation
        """
        self.rpm = rpm
        self.tpm = tpm
        self.embedding_model = embedding_model
        
        # Sliding window for requests (stores timestamps)
        self.request_times: deque = deque()
        # Sliding window for tokens (stores (timestamp, tokens) tuples)
        self.token_usage: deque = deque()
        
        # Statistics
        self.stats = RateLimitStats()
        
        # Thread lock for thread safety
        self.lock = Lock()
    
    def _clean_old_entries(self):
        """Remove entries older than 1 minute from sliding windows."""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean request times
        while self.request_times and self.request_times[0] < one_minute_ago:
            self.request_times.popleft()
        
        # Clean token usage
        while self.token_usage and self.token_usage[0][0] < one_minute_ago:
            self.token_usage.popleft()
    
    def _get_current_rpm(self) -> int:
        """Get current requests per minute."""
        self._clean_old_entries()
        return len(self.request_times)
    
    def _get_current_tpm(self) -> int:
        """Get current tokens per minute."""
        self._clean_old_entries()
        return sum(tokens for _, tokens in self.token_usage)
    
    def _wait_time_for_rpm(self) -> float:
        """Calculate wait time needed for RPM limit."""
        if len(self.request_times) < self.rpm:
            return 0.0
        
        # Wait until oldest request is 1 minute old
        oldest_request = self.request_times[0]
        wait_time = 60 - (time.time() - oldest_request)
        return max(0.0, wait_time)
    
    def _wait_time_for_tpm(self, tokens_needed: int) -> float:
        """Calculate wait time needed for TPM limit."""
        current_tpm = self._get_current_tpm()
        
        if current_tpm + tokens_needed <= self.tpm:
            return 0.0
        
        # Calculate how many tokens need to expire
        tokens_to_expire = (current_tpm + tokens_needed) - self.tpm
        
        # Find when enough tokens will expire
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Sort token usage by time
        sorted_usage = sorted(self.token_usage, key=lambda x: x[0])
        
        tokens_expired = 0
        for timestamp, tokens in sorted_usage:
            if timestamp < one_minute_ago:
                continue
            
            tokens_expired += tokens
            if tokens_expired >= tokens_to_expire:
                # Wait until this entry expires
                wait_time = 60 - (current_time - timestamp)
                return max(0.0, wait_time)
        
        # If we can't calculate precisely, wait 60 seconds
        return 60.0
    
    def acquire(
        self,
        tokens_needed: int = 0,
        wait: bool = True
    ) -> tuple[bool, float]:
        """
        Acquire permission to make an API call.
        
        Args:
            tokens_needed: Number of tokens that will be used
            wait: Whether to wait if limit is exceeded
        
        Returns:
            Tuple of (success, wait_time)
            - success: True if allowed, False if denied
            - wait_time: Seconds to wait before retry (0 if allowed)
        """
        with self.lock:
            self._clean_old_entries()
            
            # Check RPM limit
            rpm_wait = self._wait_time_for_rpm()
            if rpm_wait > 0:
                if wait:
                    time.sleep(rpm_wait)
                    self._clean_old_entries()
                else:
                    return False, rpm_wait
            
            # Check TPM limit
            tpm_wait = self._wait_time_for_tpm(tokens_needed)
            if tpm_wait > 0:
                if wait:
                    time.sleep(tpm_wait)
                    self._clean_old_entries()
                else:
                    return False, tpm_wait
            
            # Record the request
            current_time = time.time()
            self.request_times.append(current_time)
            if tokens_needed > 0:
                self.token_usage.append((current_time, tokens_needed))
            
            # Update statistics
            self.stats.requests_made += 1
            self.stats.tokens_used += tokens_needed
            
            return True, 0.0
    
    def estimate_cost(self, tokens: int) -> float:
        """
        Estimate cost for a given number of tokens.
        
        Args:
            tokens: Number of tokens
        
        Returns:
            Estimated cost in USD
        """
        cost_per_million = self.EMBEDDING_COST_PER_MILLION.get(
            self.embedding_model,
            self.EMBEDDING_COST_PER_MILLION["text-embedding-3-small"]
        )
        return (tokens / 1_000_000) * cost_per_million
    
    def get_stats(self) -> dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            self._clean_old_entries()
            
            current_rpm = len(self.request_times)
            current_tpm = self._get_current_tpm()
            
            # Calculate cost
            cost = self.estimate_cost(self.stats.tokens_used)
            
            return {
                "current_rpm": current_rpm,
                "rpm_limit": self.rpm,
                "current_tpm": current_tpm,
                "tpm_limit": self.tpm,
                "total_requests": self.stats.requests_made,
                "total_tokens": self.stats.tokens_used,
                "estimated_cost": cost,
                "rpm_available": max(0, self.rpm - current_rpm),
                "tpm_available": max(0, self.tpm - current_tpm),
            }
    
    def update_limits(self, rpm: Optional[int] = None, tpm: Optional[int] = None):
        """
        Update rate limits.
        
        Args:
            rpm: New RPM limit (None to keep current)
            tpm: New TPM limit (None to keep current)
        """
        with self.lock:
            if rpm is not None:
                self.rpm = rpm
            if tpm is not None:
                self.tpm = tpm
    
    def reset_stats(self):
        """Reset statistics."""
        with self.lock:
            self.stats.reset()
