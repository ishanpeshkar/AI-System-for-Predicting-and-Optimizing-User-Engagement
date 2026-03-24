"""Backend configuration management."""

import json
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    backend_reload: bool = True

    # CORS
    cors_origins: str = json.dumps([
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ])

    # Logging
    log_level: str = "INFO"

    # Models
    use_gpu: bool = False
    model_cache_size: int = 1000

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def parsed_cors_origins(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, TypeError):
            return [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
            ]


settings = Settings()
