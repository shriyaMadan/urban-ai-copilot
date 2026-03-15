"""Flood risk scoring module."""
from __future__ import annotations

from typing import Any, Dict, TypedDict

from core.data_model import CityState


class FloodRiskResult(TypedDict):
    """Normalized flood risk response object."""

    risk_name: str
    score: int
    level: str
    reason: str


# Tuning-friendly constants (easy to adjust later)
VULNERABILITY_BONUS: Dict[str, int] = {
    "low": 0,
    "medium": 8,
    "high": 15,
}

WIND_BONUS_RULES: tuple[tuple[float, int], ...] = (
    (30.0, 8),
    (20.0, 5),
    (12.0, 2),
)


def assess_flood_risk(city_state: CityState) -> FloodRiskResult:
    """Return a transparent flood risk assessment from city-state signals.

    Rule design:
    - Rainfall in the next 6h is the main signal.
    - District vulnerability increases severity.
    - Strong wind slightly increases severity.
    """
    rainfall_6h = city_state.rainfall_mm_next_6h
    rainfall_3h = city_state.rainfall_mm_next_3h
    vulnerability = city_state.district_vulnerability.lower().strip()
    wind_speed = city_state.wind_speed

    # Main signal: next 6h rainfall contribution (0..80)
    rainfall_score = min(80, int(round(rainfall_6h * 4)))

    # Optional near-term pressure from next 3h (0..10)
    short_term_bonus = min(10, int(round(rainfall_3h * 2)))

    vulnerability_bonus = VULNERABILITY_BONUS.get(vulnerability, 8)
    wind_bonus = _wind_bonus(wind_speed)

    # River context bonus from PEGELONLINE or fallback hydrology feed.
    river_level = city_state.river_water_level_cm
    if river_level >= 700:
        river_level_bonus = 20
    elif river_level >= 500:
        river_level_bonus = 12
    elif river_level >= 350:
        river_level_bonus = 6
    else:
        river_level_bonus = 0

    trend_bonus = 6 if city_state.river_trend == "rising" else 0

    raw_score = (
        rainfall_score
        + short_term_bonus
        + vulnerability_bonus
        + wind_bonus
        + river_level_bonus
        + trend_bonus
    )
    score = _clamp_score(raw_score)
    level = _risk_level(score)

    reason = (
        f"Rainfall next 6h: {rainfall_6h:.1f} mm drives base risk; "
        f"district vulnerability '{vulnerability}' adds {vulnerability_bonus} points; "
        f"wind at {wind_speed:.1f} adds {wind_bonus} points; "
        f"river level {river_level:.1f} cm ({city_state.river_trend}, "
        f"source={city_state.flood_data_source}) adds "
        f"{river_level_bonus + trend_bonus} points."
    )

    return {
        "risk_name": "Flood Risk",
        "score": score,
        "level": level,
        "reason": reason,
    }


class FloodRiskModel:
    """Evaluate flood risk based on precipitation and infrastructure signals."""

    def evaluate(self, city_state: CityState) -> FloodRiskResult:
        """Return flood risk details for the provided city state."""
        return assess_flood_risk(city_state)


def _wind_bonus(wind_speed: float) -> int:
    """Return a small wind severity bonus from simple threshold rules."""
    for threshold, bonus in WIND_BONUS_RULES:
        if wind_speed >= threshold:
            return bonus
    return 0


def _risk_level(score: int) -> str:
    """Map numeric score to a readable risk label."""
    if score >= 70:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _clamp_score(value: int) -> int:
    """Clamp score to the expected 0..100 range."""
    return max(0, min(100, value))
