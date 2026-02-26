"""Entry point for running Model Server"""

import uvicorn
from src.model_server.config import get_settings


def main():
    """Run the Model Server"""
    settings = get_settings()
    
    uvicorn.run(
        "src.model_server.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
