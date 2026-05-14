"""Bridge system-test CLI log levels to PyPTO + simpler loggers."""

from __future__ import annotations

import logging

from pypto import LogLevel

_PYPTO_LEVELS = {
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARN": LogLevel.WARN,
    "ERROR": LogLevel.ERROR,
    "FATAL": LogLevel.FATAL,
    "EVENT": LogLevel.EVENT,
    "NONE": LogLevel.NONE,
}

_SIMPLER_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": 50,
    "EVENT": 60,
    "NONE": 60,
    **{f"V{idx}": 15 + idx for idx in range(10)},
}

ST_LOG_LEVEL_CHOICES = [
    "DEBUG",
    "V0",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "INFO",
    "WARN",
    "ERROR",
    "FATAL",
    "EVENT",
    "NONE",
]


def normalize_st_log_level(raw_level: str) -> str:
    """Normalize the system-test CLI level to the canonical uppercase token."""
    return str(raw_level).strip().upper()


def simpler_logger_level_for(raw_level: str) -> int:
    """Return the simpler logger threshold integer for a system-test CLI level."""
    return _SIMPLER_LEVELS[normalize_st_log_level(raw_level)]


def pypto_log_level_for(raw_level: str) -> LogLevel:
    """Return the PyPTO coarse log level corresponding to a system-test CLI level."""
    normalized = normalize_st_log_level(raw_level)
    if normalized.startswith("V"):
        return LogLevel.INFO
    return _PYPTO_LEVELS[normalized]
