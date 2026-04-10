"""Central logging helpers for Urban AI Copilot."""
from __future__ import annotations

import logging
import os


_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure process-wide console logging once.

    Streamlit reruns the script frequently, so this guard prevents duplicate
    handlers while still allowing the log level to be updated from `LOG_LEVEL`.
    """
    global _CONFIGURED

    resolved_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_level = getattr(logging, resolved_level_name, logging.INFO)
    root_logger = logging.getLogger()

    if not _CONFIGURED:
        if not root_logger.handlers:
            logging.basicConfig(
                level=resolved_level,
                format=(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                ),
            )
        _CONFIGURED = True

    root_logger.setLevel(resolved_level)
    for handler in root_logger.handlers:
        handler.setLevel(resolved_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)