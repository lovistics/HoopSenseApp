"""
Logger configuration module.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings


# Configure logging format
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(getattr(logging, log_level))
    logger_instance.propagate = False
    
    # Remove existing handlers
    for handler in logger_instance.handlers[:]:
        logger_instance.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)
    
    # Optionally create file handler
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
    
    return logger_instance


# Configure the root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create a logger instance for the application
logger = setup_logger("hoopsense", settings.LOG_LEVEL, settings.LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Named logger
    """
    return setup_logger(f"hoopsense.{name}", settings.LOG_LEVEL, settings.LOG_FILE)