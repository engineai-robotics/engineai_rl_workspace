"""EngineAI RL Workspace package."""

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
LOCK_KEY = "engineai_rl_lock"
LOCK_TIMEOUT = 3600  # 1 hour in seconds
LOCK_MESSAGE = "EngineAI RL training is already running on another process"
MESSAGE = LOCK_MESSAGE  # 为了向后兼容

# Training messages
INITIALIZATION_COMPLETE_MESSAGE = "Initialization complete, starting training..."
