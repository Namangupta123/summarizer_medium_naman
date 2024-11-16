import redis
import hashlib
import os
from typing import Optional, Union
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    ConnectionError,
    TimeoutError,
    RedisError
)

class RedisCache:
    _instance = None
    _redis_client = None
    CACHE_EXPIRATION = 60 * 60 * 24  # 24 hours in seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 0.1  # Initial delay in seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._redis_client is None:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize Redis client with retry mechanism and connection pooling"""
        try:
            retry_strategy = Retry(
                backoff=ExponentialBackoff(cap=10, base=2),
                retries=self.MAX_RETRIES
            )

            self._redis_client = redis.Redis(
                host='redis-11377.c81.us-east-1-2.ec2.redns.redis-cloud.com',
                port=11377,
                password=os.getenv('REDIS_PASSWORD'),
                ssl=True,
                ssl_cert_reqs=None,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                retry=retry_strategy,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            # Test connection
            self._redis_client.ping()
            print("Redis Cloud connection established successfully")
        except RedisError as e:
            print(f"Failed to initialize Redis connection: {str(e)}")
            raise

    def _get_client(self) -> redis.Redis:
        """Get Redis client with connection check"""
        try:
            self._redis_client.ping()
            return self._redis_client
        except (ConnectionError, TimeoutError):
            print("Redis connection lost, attempting to reconnect...")
            self._initialize_client()
            return self._redis_client

    @staticmethod
    def generate_cache_key(content: str) -> str:
        """Generate a deterministic cache key from content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"summary:{content_hash}"

    def get(self, key: str) -> Optional[str]:
        """Get value from cache with error handling"""
        try:
            client = self._get_client()
            return client.get(key)
        except RedisError as e:
            print(f"Redis get error: {str(e)}")
            return None

    def set(self, key: str, value: str, expiration: int = None) -> bool:
        """Set value in cache with error handling"""
        try:
            client = self._get_client()
            return client.setex(
                key,
                expiration or self.CACHE_EXPIRATION,
                value
            )
        except RedisError as e:
            print(f"Redis set error: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            client = self._get_client()
            return bool(client.delete(key))
        except RedisError as e:
            print(f"Redis delete error: {str(e)}")
            return False

# Global cache instance
cache = RedisCache()

def init_redis():
    """Initialize Redis connection"""
    try:
        global cache
        cache = RedisCache()
    except Exception as e:
        print(f"Redis initialization error: {str(e)}")
        raise

def get_cached_summary(content: str) -> Optional[str]:
    """Get cached summary if it exists"""
    try:
        cache_key = cache.generate_cache_key(content)
        return cache.get(cache_key)
    except Exception as e:
        print(f"Cache retrieval error: {str(e)}")
        return None

def cache_summary(content: str, summary: str) -> bool:
    """Cache a summary with expiration"""
    try:
        cache_key = cache.generate_cache_key(content)
        return cache.set(cache_key, summary)
    except Exception as e:
        print(f"Cache storage error: {str(e)}")
        return False