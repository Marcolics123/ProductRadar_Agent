import hashlib
import os
from functools import lru_cache

from redis import Redis
from redis.exceptions import RedisError


class RateLimitExceeded(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _redis_client(redis_url: str) -> Redis:
    return Redis.from_url(
        redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


# 🟦【第四批修改 9】使用 Redis 原子计数保护公开认证入口。
def enforce_rate_limit(
    bucket: str,
    identifier: str,
    *,
    limit: int,
    window_seconds: int,
) -> None:
    """固定时间窗限流；identifier 先哈希，Redis 中不直接保存用户 IP。"""

    redis_url = os.environ.get(
        "CELERY_BROKER_URL",
        "redis://127.0.0.1:6379/0",
    )
    identifier_hash = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    key = f"productradar:rate:{bucket}:{identifier_hash}"
    script = """
    local current = redis.call('INCR', KEYS[1])
    if current == 1 then
        redis.call('EXPIRE', KEYS[1], ARGV[1])
    end
    return current
    """
    try:
        current = int(
            _redis_client(redis_url).eval(
                script,
                1,
                key,
                window_seconds,
            )
        )
    except RedisError as error:
        raise RuntimeError("认证限流服务暂时不可用") from error
    if current > limit:
        raise RateLimitExceeded("请求过于频繁，请稍后再试")
