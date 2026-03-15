"""Traffic data service integration (TomTom + safe fallback)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class TrafficService:
    """Fetch live traffic flow metrics for a city.

    The primary provider is TomTom Traffic API. If the API key is missing or the
    request fails, the service returns deterministic fallback values so the app
    remains usable.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def get_traffic_context(self, city: str) -> Dict[str, Any]:
        """Return normalized traffic context for the requested city."""
        if not self.api_key:
            return self.get_mock_traffic_context(city)

        try:
            lat, lon = self._geocode_city(city)
            response = requests.get(
                f"{self.base_url.rstrip('/')}/flowSegmentData/absolute/10/json",
                params={
                    "point": f"{lat},{lon}",
                    "unit": "KMPH",
                    "openLr": "false",
                    "key": self.api_key,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json().get("flowSegmentData", {})

            current_speed = float(payload.get("currentSpeed", 0.0))
            free_flow_speed = float(payload.get("freeFlowSpeed", 0.0))
            congestion_index = self._compute_congestion_index(
                current_speed=current_speed,
                free_flow_speed=free_flow_speed,
            )

            return {
                "traffic_congestion_index": congestion_index,
                "traffic_speed_kph": current_speed,
                "traffic_free_flow_speed_kph": free_flow_speed,
                "traffic_source": "tomtom",
            }
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return self.get_mock_traffic_context(city)

    def get_mock_traffic_context(self, city: str) -> Dict[str, Any]:
        """Return deterministic city-specific fallback traffic values."""
        normalized_city = city.strip().lower()
        seed = sum(ord(ch) for ch in normalized_city)

        current_speed = round(24.0 + (seed % 160) / 10.0, 1)
        free_flow_speed = round(current_speed + 12.0 + (seed % 60) / 10.0, 1)
        congestion_index = self._compute_congestion_index(
            current_speed=current_speed,
            free_flow_speed=free_flow_speed,
        )

        return {
            "traffic_congestion_index": congestion_index,
            "traffic_speed_kph": current_speed,
            "traffic_free_flow_speed_kph": free_flow_speed,
            "traffic_source": "proxy",
        }

    def _geocode_city(self, city: str) -> tuple[float, float]:
        """Resolve city name to latitude/longitude using Open-Meteo geocoding."""
        geocode_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=self.timeout,
        )
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
        results = geocode_data.get("results") or []

        if not results:
            raise ValueError(f"No geocoding result found for city: {city}")

        first = results[0]
        return float(first["latitude"]), float(first["longitude"])

    def _compute_congestion_index(
        self,
        current_speed: float,
        free_flow_speed: float,
    ) -> float:
        """Compute congestion index in percentage (0-100)."""
        if free_flow_speed <= 0:
            return 0.0

        ratio = current_speed / free_flow_speed
        congestion = max(0.0, min(1.0, 1.0 - ratio))
        return round(congestion * 100, 1)
