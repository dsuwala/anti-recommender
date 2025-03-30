import logging
import uvicorn
from config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Starting server...")

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
