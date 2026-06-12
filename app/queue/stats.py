"""
Redis Stream Queue Statistics.

Provides functions to retrieve queue metrics from Redis Streams. Used to monitor
the volume of backlog messages, pending items, and dead letter counts.
"""

import redis.asyncio as aioredis

from app.config.settings import settings


async def get_queue_stats(redis_client: aioredis.Redis) -> dict:
    """
    Queries Redis to compile status metrics of the processing streams.

    Returns a dictionary:
      - streamLength: Total messages currently in the processing stream.
      - dlqLength: Total failed jobs residing in the Dead Letter Queue stream.
      - pendingCount: Total jobs retrieved by this group but not yet acknowledged.
    """
    try:
        # Retrieve lengths of the active and DLQ streams
        stream_len = await redis_client.xlen(settings.stream_key)
        dlq_len = await redis_client.xlen(settings.dlq_stream_key)

        # Query active consumer groups to read pending message counters
        groups = await redis_client.xinfo_groups(settings.stream_key)
        pending = 0
        for group in groups:
            if group.get("name") == settings.consumer_group:
                pending = group.get("pending", 0)

        return {
            "streamLength": stream_len,
            "dlqLength": dlq_len,
            "pendingCount": pending,
        }
    except Exception:
        # Fallback values if Redis is uninitialized or unreachable
        return {"streamLength": 0, "dlqLength": 0, "pendingCount": 0}
