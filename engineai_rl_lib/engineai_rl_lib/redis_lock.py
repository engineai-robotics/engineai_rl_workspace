import asyncio
import time
import os
import redis


class RedisLock:
    def __init__(
        self, host, lock_key, port=6379, timeout=30, message="Lock exists, waiting..."
    ):
        self.redis = redis.Redis(host=host, port=port)
        self.lock_key = lock_key
        self.timeout = timeout
        self.pid = str(os.getpid())
        self.message = message

    async def acquire(self) -> bool:
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            # Try to set the lock with NX (only if not exists)
            if self.redis.set(self.lock_key, self.pid, nx=True, ex=self.timeout):
                return True

            print(self.message)
            await asyncio.sleep(5)

        return False

    def release(self):
        self.redis.delete(self.lock_key)
