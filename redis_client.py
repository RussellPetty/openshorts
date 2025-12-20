import os
from typing import Optional
from redis.asyncio import Redis, ConnectionPool

_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None


async def get_redis() -> Optional[Redis]:
    """Get Redis client. Returns None if REDIS_URL not configured."""
    global _redis_pool, _redis_client

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None

    if _redis_client is None:
        _redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=10,
            decode_responses=True
        )
        _redis_client = Redis(connection_pool=_redis_pool)

    # Test connection
    try:
        await _redis_client.ping()
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return None

    return _redis_client


async def close_redis():
    """Close Redis connection pool."""
    global _redis_pool, _redis_client

    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None
