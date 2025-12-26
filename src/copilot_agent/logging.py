"""
Logging configuration using structlog.
"""

import sys
import logging
from typing import Any
from pathlib import Path

import structlog
from structlog.types import Processor


# Patterns to redact from logs
REDACT_PATTERNS = [
    "GEMINI_API_KEY",
    "api_key",
    "apikey",
    "API_KEY",
    "secret",
    "password",
    "token",
]


def redact_secrets(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Redact sensitive information from log entries."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            for pattern in REDACT_PATTERNS:
                if pattern.lower() in key.lower():
                    event_dict[key] = "[REDACTED]"
                    break
    return event_dict


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure structlog for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )
    
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_secrets,
    ]
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )
    )
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)
