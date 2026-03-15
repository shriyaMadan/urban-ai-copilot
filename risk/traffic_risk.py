"""Traffic disruption risk scoring module."""
from __future__ import annotations

from typing import TypedDict

from core.data_model import CityState


class TrafficRiskResult(TypedDict):
    """Normalized traffic risk response object."""

    risk_name: str
    score: int
    level: str
    reason: str


def assess_traffic_risk(city_state: CityState) -> TrafficRiskResult:
    """Return a transparent proxy traffic-disruption risk assessment.

    Important: This is a simple proxy model for operational triage,
    NOT a real traffic prediction model.

    Proxy signals used:
    - rush hour status
    - weekend vs weekday
    - rainfall amount (next 6h)
    - wind speed
    """
    # Base by day type: weekdays usually have higher baseline urban traffic load.
    base_score = 15 if city_state.is_weekend else 25

    # Rush-hour effect: commuter peaks increase incident and delay probability.
    rush_hour_bonus = 20 if city_state.is_rush_hour else 0

    # Rain effect: precipitation tends to reduce speed and increase disruptions.
    rainfall_6h = city_state.rainfall_mm_next_6h
    if rainfall_6h >= 15:
        rain_bonus = 25
    elif rainfall_6h >= 8:
        rain_bonus = 16
    elif rainfall_6h >= 3:
        rain_bonus = 8
    else:
        rain_bonus = 1

    # Wind effect: strong wind can affect road safety and traffic reliability.
    wind_speed = city_state.wind_speed
    if wind_speed >= 30:
        wind_bonus = 10
    elif wind_speed >= 20:
        wind_bonus = 6
    elif wind_speed >= 12:
        wind_bonus = 3
    else:
        wind_bonus = 0

    # Congestion effect from live traffic source (when available).
    congestion_index = city_state.traffic_congestion_index
    if congestion_index >= 70:
        congestion_bonus = 25
    elif congestion_index >= 50:
        congestion_bonus = 16
    elif congestion_index >= 30:
        congestion_bonus = 8
    else:
        congestion_bonus = 2

    score = _clamp_score(
        base_score + rush_hour_bonus + rain_bonus + wind_bonus + congestion_bonus
    )
    level = _risk_level(score)

    day_type = "weekend" if city_state.is_weekend else "weekday"
    rush_label = "during rush hour" if city_state.is_rush_hour else "outside rush hour"
    reason = (
        f"Proxy traffic risk ({day_type}, {rush_label}): rainfall next 6h "
        f"{rainfall_6h:.1f} mm, wind {wind_speed:.1f}, and congestion "
        f"{congestion_index:.1f}% ({city_state.traffic_source}) produce score {score}."
    )

    return {
        "risk_name": "Traffic Disruption Risk",
        "score": score,
        "level": level,
        "reason": reason,
    }


class TrafficRiskModel:
    """Evaluate traffic disruption risk based on mobility signals."""

    def evaluate(self, city_state: CityState) -> TrafficRiskResult:
        """Return traffic disruption risk details for the provided city state."""
        return assess_traffic_risk(city_state)


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
