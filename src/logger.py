"""Logging configuration for the ML pipeline."""
import logging
import logging.config
import sys
from pathlib import Path

from src.config import ARTIFACTS_DIR


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. Defaults to ARTIFACTS_DIR/ml-pipeline.log.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(log_level="DEBUG")
        >>> logger.info("Starting training pipeline")
    """
    if log_file is None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = ARTIFACTS_DIR / "ml-pipeline.log"

    # Create logger
    logger = logging.getLogger("src")
    logger.setLevel(log_level)

    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Initialize default logger
logger = setup_logging()
