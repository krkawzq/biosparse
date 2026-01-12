"""Logging configuration for BioSparse Optimization Toolkit.

This module provides a centralized logger for all optimization-related
messages, replacing print statements with proper logging.

Usage:
    from ._logging import logger
    
    logger.debug("Processing IR for signature: %s", sig)
    logger.info("Applied %d loop hints", len(hints))
    logger.warning("No loop found after hint at line %d", line_num)
"""

import logging
import os

# Module-level logger
logger = logging.getLogger('biosparse.optim')

# Default to WARNING level unless explicitly configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(name)s] %(levelname)s: %(message)s'
    ))
    logger.addHandler(handler)
    
    # Check environment variable for debug mode
    if os.environ.get('BIOSPARSE_OPTIM_DEBUG', '').lower() in ('1', 'true', 'yes'):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)


def set_log_level(level: int) -> None:
    """Set the logging level for the optimization module.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    
    Example:
        import logging
        from biosparse.optim import set_log_level
        set_log_level(logging.DEBUG)
    """
    logger.setLevel(level)


def enable_debug() -> None:
    """Enable debug logging for the optimization module."""
    logger.setLevel(logging.DEBUG)


def disable_logging() -> None:
    """Disable all logging for the optimization module."""
    logger.setLevel(logging.CRITICAL + 1)
