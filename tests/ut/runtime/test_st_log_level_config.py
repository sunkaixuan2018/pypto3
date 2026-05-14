"""Tests for system-test log-level parsing and bridging."""

from __future__ import annotations

import logging

import pytest
from pypto import LogLevel

from tests.st.harness.core.log_level_config import (
    ST_LOG_LEVEL_CHOICES,
    normalize_st_log_level,
    pypto_log_level_for,
    simpler_logger_level_for,
)


def test_choices_include_v_tiers_and_pypto_levels():
    for name in ("DEBUG", "INFO", "WARN", "ERROR", "FATAL", "EVENT", "NONE"):
        assert name in ST_LOG_LEVEL_CHOICES
    for idx in range(10):
        assert f"V{idx}" in ST_LOG_LEVEL_CHOICES


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        ("debug", "DEBUG"),
        ("INFO", "INFO"),
        ("warn", "WARN"),
        ("Error", "ERROR"),
        ("fatal", "FATAL"),
        ("event", "EVENT"),
        ("none", "NONE"),
        ("v0", "V0"),
        ("V5", "V5"),
        ("v9", "V9"),
    ],
)
def test_normalize_st_log_level(raw_level: str, expected: str):
    assert normalize_st_log_level(raw_level) == expected


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARN", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("NONE", 60),
        ("V0", 15),
        ("V5", 20),
        ("V9", 24),
    ],
)
def test_simpler_logger_level_for(raw_level: str, expected: int):
    assert simpler_logger_level_for(raw_level) == expected


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        ("DEBUG", LogLevel.DEBUG),
        ("INFO", LogLevel.INFO),
        ("WARN", LogLevel.WARN),
        ("ERROR", LogLevel.ERROR),
        ("FATAL", LogLevel.FATAL),
        ("EVENT", LogLevel.EVENT),
        ("NONE", LogLevel.NONE),
        ("V0", LogLevel.INFO),
        ("V5", LogLevel.INFO),
        ("V9", LogLevel.INFO),
    ],
)
def test_pypto_log_level_for(raw_level: str, expected: LogLevel):
    assert pypto_log_level_for(raw_level) == expected
