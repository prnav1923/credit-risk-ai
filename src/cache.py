import os
import json
import hashlib
import logging
import redis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected ✅")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_client = None


def get_cache_key(endpoint: str, payload: dict) -> str:
    """Generate deterministic cache key from endpoint + payload"""
    payload_str = json.dumps(payload, sort_keys=True)
    hash_str = hashlib.md5(f"{endpoint}:{payload_str}".encode()).hexdigest()
    return f"credit_risk:{endpoint}:{hash_str}"


def get_cached(key: str):
    """Get value from Redis cache"""
    if redis_client is None:
        return None
    try:
        value = redis_client.get(key)
        if value:
            logger.info(f"Cache HIT: {key[:50]}")
            return json.loads(value)
        logger.info(f"Cache MISS: {key[:50]}")
        return None
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
        return None


def set_cached(key: str, value: dict, ttl: int = CACHE_TTL):
    """Set value in Redis cache"""
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value))
        logger.info(f"Cache SET: {key[:50]} (TTL: {ttl}s)")
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


def invalidate_cache(pattern: str = "credit_risk:*"):
    """Clear all cached predictions"""
    if redis_client is None:
        return 0
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries")
        return len(keys)
    except Exception as e:
        logger.warning(f"Cache invalidation error: {e}")
        return 0


def get_cache_stats():
    """Get Redis cache statistics"""
    if redis_client is None:
        return {"status": "unavailable"}
    try:
        info = redis_client.info()
        keys = len(redis_client.keys("credit_risk:*"))
        return {
            "status": "connected",
            "cached_predictions": keys,
            "memory_used": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands": info.get("total_commands_processed", 0)
        }
    except Exception as e:
        return {"status": f"error: {str(e)}"}