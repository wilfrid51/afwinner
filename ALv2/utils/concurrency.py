# file: utils/concurrency.py

"""Utilities for interfacing with async Affine generators from sync code."""

from __future__ import annotations

import asyncio
from typing import Any


def run_sync(coro) -> Any:
    """Execute ``coro`` inside a fresh event loop if required."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()

