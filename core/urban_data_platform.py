"""Facade for coordinating data access across services."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.data_model import CityState
from services.context_service import ContextService
from services.flood_service import FloodService
from services.traffic_service import TrafficService
from services.weather_service import WeatherService


class UrbanDataPlatform:
    """Central mini data platform for unified city-state construction.

    This layer pulls weather data and operational context, then merges both into
    a single `CityState` object used by downstream risk and copilot modules.
    """

    def __init__(
        self,
        weather_service: WeatherService,
        context_service: ContextService,
        traffic_service: TrafficService | None = None,
        flood_service: FloodService | None = None,
    ) -> None:
        self.weather_service = weather_service
        self.context_service = context_service
        self.traffic_service = traffic_service
        self.flood_service = flood_service

        self.context_service.attach_optional_services(
            traffic_service=self.traffic_service,
            flood_service=self.flood_service,
        )

    def build_city_state(
        self,
        city: str,
        district_vulnerability: str = "medium",
        now: datetime | None = None,
    ) -> CityState:
        """Build a unified city-state from weather and context services."""
        weather_data = self.weather_service.get_forecast(city)
        context_data = self.context_service.build_context(
            city=city,
            district_vulnerability=district_vulnerability,
            now=now,
        )
        return CityState.from_service_outputs(weather_data, context_data)

    def fetch_context(
        self,
        city: str,
        district_vulnerability: str = "medium",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return normalized context as a plain dict for API/UI consumers.

        This is a convenience wrapper around `build_city_state`.
        """
        return self.build_city_state(
            city=city,
            district_vulnerability=district_vulnerability,
            now=now,
        ).to_dict()
