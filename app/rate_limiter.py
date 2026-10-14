"""Redis distributed token-bucket rate limiter with fail-secure outage fallback."""

import logging
import os
import threading
import time
from typing import Optional, Tuple
from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


class InMemoryTokenBucket:
    """Thread-safe in-memory token bucket for local testing and outage fallback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buckets = {}  # key -> (tokens, last_refill_ts)

    def consume(
        self,
        key: str,
        rate_per_sec: float,
        burst_capacity: int,
        tokens_requested: float = 1.0
    ) -> Tuple[bool, float, int]:
        """Attempt to consume tokens.
        
        Returns:
            (allowed: bool, remaining_tokens: float, retry_after_sec: int)
        """
        now = time.time()
        with self._lock:
            if key not in self._buckets:
                tokens = float(burst_capacity)
                last_ts = now
            else:
                tokens, last_ts = self._buckets[key]
                elapsed = now - last_ts
                tokens = min(float(burst_capacity), tokens + elapsed * rate_per_sec)

            if tokens >= tokens_requested:
                tokens -= tokens_requested
                self._buckets[key] = (tokens, now)
                return True, tokens, 0
            else:
                deficit = tokens_requested - tokens
                retry_after = int(deficit / rate_per_sec) + 1 if rate_per_sec > 0 else 60
                self._buckets[key] = (tokens, now)
                return False, tokens, retry_after

    def reset(self, key: Optional[str] = None):
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()


class DistributedRateLimiter:
    """Distributed token-bucket rate limiter with Redis backend and fail-secure fallback."""

    def __init__(self, redis_url: str = REDIS_URL, timeout: float = 0.5):
        self.redis_url = redis_url
        self.timeout = timeout
        self.in_memory = InMemoryTokenBucket()
        self._redis_client = None
        self._redis_available = None
        self._redis_checked_at = 0.0

    def _get_redis(self):
        """Lazy initialization of redis client."""
        now = time.time()
        if self._redis_available is False and (now - self._redis_checked_at) < 10.0:
            return None

        if self._redis_client is None:
            try:
                import redis
                client = redis.Redis.from_url(
                    self.redis_url,
                    socket_timeout=self.timeout,
                    socket_connect_timeout=self.timeout,
                    decode_responses=True
                )
                client.ping()
                self._redis_client = client
                self._redis_available = True
                logger.info(f"Connected to Redis rate-limiting server at {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}); engaging fail-secure in-process rate limiter.")
                self._redis_available = False
                self._redis_checked_at = now
                self._redis_client = None

        return self._redis_client

    def check_rate_limit(
        self,
        tenant_id: str,
        actor_id: str,
        rate_limit_rpm: int = 300,
        burst_capacity: int = 30,
        fail_secure_budget_ratio: float = 0.5
    ) -> Tuple[bool, int, int]:
        """Enforce token bucket rate limit.
        
        Returns:
            (allowed: bool, remaining_tokens: int, retry_after: int)
        """
        rate_per_sec = max(0.1, rate_limit_rpm / 60.0)
        bucket_key = f"jvr:ratelimit:{tenant_id}:{actor_id}"

        redis_client = self._get_redis()
        if redis_client:
            try:
                # Redis token-bucket via Lua script for atomicity
                lua_script = """
                local key = KEYS[1]
                local rate = tonumber(ARGV[1])
                local burst = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])
                local cost = tonumber(ARGV[4])

                local data = redis.call('HMGET', key, 'tokens', 'last_ts')
                local tokens = tonumber(data[1])
                local last_ts = tonumber(data[2])

                if not tokens or not last_ts then
                    tokens = burst
                    last_ts = now
                else
                    local elapsed = math.max(0, now - last_ts)
                    tokens = math.min(burst, tokens + elapsed * rate)
                end

                if tokens >= cost then
                    tokens = tokens - cost
                    redis.call('HMSET', key, 'tokens', tokens, 'last_ts', now)
                    redis.call('EXPIRE', key, math.ceil(burst / rate) * 2)
                    return {1, math.floor(tokens), 0}
                else
                    local deficit = cost - tokens
                    local retry_after = math.ceil(deficit / rate)
                    redis.call('HMSET', key, 'tokens', tokens, 'last_ts', now)
                    redis.call('EXPIRE', key, math.ceil(burst / rate) * 2)
                    return {0, math.floor(tokens), retry_after}
                end
                """
                res = redis_client.eval(lua_script, 1, bucket_key, rate_per_sec, burst_capacity, time.time(), 1.0)
                allowed = bool(res[0])
                remaining = int(res[1])
                retry_after = int(res[2])
                return allowed, remaining, retry_after
            except Exception as e:
                logger.error(f"Redis rate-limit query failed ({e}); falling back to in-memory fail-secure policy.")
                self._redis_available = False
                self._redis_checked_at = time.time()

        # Fail-Secure Policy: In-memory conservative token bucket
        conservative_burst = max(1, int(burst_capacity * fail_secure_budget_ratio))
        conservative_rate = max(0.1, rate_per_sec * fail_secure_budget_ratio)
        allowed, remaining_f, retry_after = self.in_memory.consume(
            key=bucket_key,
            rate_per_sec=conservative_rate,
            burst_capacity=conservative_burst,
            tokens_requested=1.0
        )
        return allowed, max(0, int(remaining_f)), retry_after


# Global singleton instance
RATE_LIMITER = DistributedRateLimiter()


def enforce_rate_limit(
    request: Request,
    response: Response,
    tenant_id: str,
    actor_id: str,
    rate_limit_rpm: int = 300,
    rate_limit_burst: int = 30
):
    """Enforce rate limits and set standard rate-limiting HTTP headers."""
    allowed, remaining, retry_after = RATE_LIMITER.check_rate_limit(
        tenant_id=tenant_id,
        actor_id=actor_id,
        rate_limit_rpm=rate_limit_rpm,
        burst_capacity=rate_limit_burst
    )

    response.headers["X-RateLimit-Limit"] = str(rate_limit_rpm)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(retry_after)

    if not allowed:
        response.headers["Retry-After"] = str(retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for tenant '{tenant_id}'. Capacity: {rate_limit_rpm} RPM. Try again in {retry_after} seconds.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limit_rpm),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            }
        )
