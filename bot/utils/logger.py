"""Logging utility for the trading bot."""

import logging
import os
import sys
from datetime import datetime

import config


def setup_logger(name: str = "trading_bot") -> logging.Logger:
    """Configure and return the main logger."""
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(config.LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, config.LOG_LEVEL))
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logger()
