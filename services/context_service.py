"""Service for building a contextual snapshot for analysis."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from services.flood_service import FloodService
from services.traffic_service import TrafficService


ALLOWED_VULNERABILITY_LEVELS = {"low", "medium", "high"}


class ContextService:
    """Build a normalized operational context for risk and response decisions.

    Why these features matter for urban operations:
    - Rush hour helps estimate mobility pressure and likely traffic disruption.
    - Weekend/weekday affects commuter flows, staffing patterns, and demand peaks.
    - District vulnerability captures social/infrastructure sensitivity, which helps
      prioritize interventions when risks increase.
    """

    def __init__(self) -> None:
        self.traffic_service: TrafficService | None = None
        self.flood_service: FloodService | None = None

    def attach_optional_services(
        self,
        traffic_service: TrafficService | None = None,
        flood_service: FloodService | None = None,
    ) -> None:
        """Attach optional live data services for richer city context."""
        self.traffic_service = traffic_service
        self.flood_service = flood_service

    def build_context(
        self,
        city: str,
        district_vulnerability: str = "medium",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a transparent context object for downstream risk models.

        Rules are intentionally simple:
        - Weekend: Saturday/Sunday (`weekday() >= 5`)
        - Rush hour: only on weekdays, during 07:00-09:59 and 16:00-19:59
        - District vulnerability: normalized to low/medium/high, defaulting to medium
          for invalid input.
        """
        local_now = now or datetime.now().astimezone()
        local_hour = local_now.hour
        is_weekend = self._is_weekend(local_now)
        is_rush_hour = self._is_rush_hour(local_hour=local_hour, is_weekend=is_weekend)
        normalized_vulnerability = self._normalize_district_vulnerability(
            district_vulnerability
        )

        traffic_context: dict[str, Any] = {
            "traffic_congestion_index": 0.0,
            "traffic_speed_kph": 0.0,
            "traffic_free_flow_speed_kph": 0.0,
            "traffic_source": "proxy",
        }
        if self.traffic_service is not None:
            traffic_context = self.traffic_service.get_traffic_context(city)

        flood_context: dict[str, Any] = {
            "river_water_level_cm": 0.0,
            "river_trend": "unknown",
            "flood_data_source": "weather_proxy",
        }
        if self.flood_service is not None:
            flood_context = self.flood_service.get_flood_context(city)

        return {
            "is_rush_hour": is_rush_hour,
            "is_weekend": is_weekend,
            "district_vulnerability": normalized_vulnerability,
            "local_hour": local_hour,
            **traffic_context,
            **flood_context,
        }

    def _is_weekend(self, now: datetime) -> bool:
        """Return True for Saturday/Sunday based on local time."""
        return now.weekday() >= 5

    def _is_rush_hour(self, local_hour: int, is_weekend: bool) -> bool:
        """Apply a transparent rush-hour rule for weekday commuter peaks."""
        if is_weekend:
            return False

        morning_peak = 7 <= local_hour <= 9
        evening_peak = 16 <= local_hour <= 19
        return morning_peak or evening_peak

    def _normalize_district_vulnerability(self, value: str) -> str:
        """Normalize district vulnerability to low/medium/high.

        Invalid values fall back to "medium" so the app can keep running.
        """
        normalized = value.strip().lower()
        if normalized in ALLOWED_VULNERABILITY_LEVELS:
            return normalized
        return "medium"
