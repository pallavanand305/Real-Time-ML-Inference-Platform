"""Model Server - Service for loading and executing ML models"""

from src.model_server.app import app
from src.model_server.config import ModelServerSettings, get_settings

__all__ = ["app", "ModelServerSettings", "get_settings"]
