"""Heat risk scoring module."""
from __future__ import annotations

from typing import TypedDict

from core.data_model import CityState


class HeatRiskResult(TypedDict):
    """Normalized heat risk response object."""

    risk_name: str
    score: int
    level: str
    reason: str


def assess_heat_risk(city_state: CityState) -> HeatRiskResult:
    """Return a transparent heat risk assessment from city-state signals.

    European urban heat thresholds (simple and tunable):
    - < 20°C: usually low population-level heat pressure.
    - 20-24.9°C: mild warming, limited risk.
    - 25-29.9°C: notable heat burden in dense urban areas.
    - 30-34.9°C: high strain, especially for vulnerable groups.
    - >= 35°C: severe heat risk requiring active response.

    Humidity and higher feels-like temperature increase discomfort and heat stress.
    """
    temperature_c = city_state.temperature_c
    feels_like_c = city_state.feels_like_c
    humidity = city_state.humidity

    # Base score from measured air temperature (max 70)
    if temperature_c < 20:
        temp_score = 10
    elif temperature_c < 25:
        temp_score = 25
    elif temperature_c < 30:
        temp_score = 45
    elif temperature_c < 35:
        temp_score = 60
    else:
        temp_score = 70

    # Feels-like uplift (max +20): captures urban heat island + humidity effects
    feels_like_delta = feels_like_c - temperature_c
    if feels_like_c >= 36:
        feels_like_bonus = 20
    elif feels_like_c >= 32:
        feels_like_bonus = 14
    elif feels_like_c >= 28:
        feels_like_bonus = 8
    elif feels_like_delta >= 2:
        feels_like_bonus = 4
    else:
        feels_like_bonus = 0

    # Humidity contribution (max +10): higher humidity reduces cooling efficiency
    if humidity >= 80:
        humidity_bonus = 10
    elif humidity >= 65:
        humidity_bonus = 7
    elif humidity >= 50:
        humidity_bonus = 4
    else:
        humidity_bonus = 1

    score = _clamp_score(temp_score + feels_like_bonus + humidity_bonus)
    level = _risk_level(score)

    reason = (
        f"Air temperature {temperature_c:.1f}°C with feels-like {feels_like_c:.1f}°C "
        f"and humidity {humidity}% yields a heat risk score of {score}."
    )

    return {
        "risk_name": "Heat Risk",
        "score": score,
        "level": level,
        "reason": reason,
    }


class HeatRiskModel:
    """Evaluate heat stress risk based on temperature and humidity."""

    def evaluate(self, city_state: CityState) -> HeatRiskResult:
        """Return heat risk details for the provided city state."""
        return assess_heat_risk(city_state)


def _risk_level(score: int) -> str:
    """Map numeric score to a readable heat risk label."""
    if score >= 70:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _clamp_score(value: int) -> int:
    """Clamp score to the expected 0..100 range."""
    return max(0, min(100, value))
