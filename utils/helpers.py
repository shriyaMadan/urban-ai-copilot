"""Helper utilities for the Urban AI Copilot."""
from __future__ import annotations

from typing import List


def parse_supported_cities(raw: str) -> List[str]:
    """Parse a comma-separated list of cities."""
    return [city.strip() for city in raw.split(",") if city.strip()]
