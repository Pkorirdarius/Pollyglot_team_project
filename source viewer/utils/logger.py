"""utils/logger.py - Loguru configuration shared across the app."""
import os
import sys
from pathlib import Path
from loguru import logger

# Get log level from environment or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Ensure the logs directory exists to prevent initialization errors
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove the default logger to prevent duplicate logs in some environments
logger.remove()

# 1. Console Logger: Clean, colorized output for developers
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    enqueue=True  # Asynchronous logging for better performance
)

# 2. File Logger: Detailed debug information for audit trails
logger.add(
    LOG_DIR / "app.log",
    level="DEBUG",
    rotation="10 MB",     # Rotates when the file reaches 10MB
    retention="7 days",    # Keeps logs for a week
    compression="zip",     # Compresses old logs to save space
    backtrace=True,        # Captures full stack traces
    diagnose=True,         # Shows variable values in stack traces
    enqueue=True
)

# Export the logger for use in other modules
__all__ = ["logger"]