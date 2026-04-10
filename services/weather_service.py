"""Weather data service integration."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger(__name__)


WEATHER_CODE_MAP: Dict[int, tuple[str, str]] = {
    0: ("Clear", "Clear sky"),
    1: ("Clouds", "Mainly clear"),
    2: ("Clouds", "Partly cloudy"),
    3: ("Clouds", "Overcast"),
    45: ("Fog", "Fog"),
    48: ("Fog", "Depositing rime fog"),
    51: ("Drizzle", "Light drizzle"),
    53: ("Drizzle", "Moderate drizzle"),
    55: ("Drizzle", "Dense drizzle"),
    61: ("Rain", "Slight rain"),
    63: ("Rain", "Moderate rain"),
    65: ("Rain", "Heavy rain"),
    71: ("Snow", "Slight snow"),
    73: ("Snow", "Moderate snow"),
    75: ("Snow", "Heavy snow"),
    80: ("Rain", "Slight rain showers"),
    81: ("Rain", "Moderate rain showers"),
    82: ("Rain", "Violent rain showers"),
    95: ("Thunderstorm", "Thunderstorm"),
}


class WeatherService:
    """Fetches weather forecasts from a public or partner API."""

    def __init__(self, api_key: Optional[str], base_url: str, timeout: int = 10) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def get_forecast(self, city: str) -> Dict[str, Any]:
        """Return normalized current + forecast weather data for a city.

        If the live API request fails, return city-specific fallback data so the
        app can continue running in demo mode.
        """
        last_error: str | None = None

        # One retry helps avoid transient network timeouts in Streamlit reruns.
        for _ in range(2):
            try:
                return self._fetch_live_weather(city)
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if last_error:
            logger.warning(
                "Weather live fetch failed; using fallback weather | city=%s | error=%s",
                city,
                last_error,
            )
        return self.get_mock_weather(city, error_message=last_error)

    def _fetch_live_weather(self, city: str) -> Dict[str, Any]:
        """Fetch and normalize live weather data from Open-Meteo APIs."""
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

        city_info = results[0]
        latitude = city_info["latitude"]
        longitude = city_info["longitude"]
        resolved_city = city_info.get("name", city)

        forecast_params: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m"
            ),
            "hourly": "precipitation",
            "forecast_days": 2,
            "timezone": "auto",
        }
        if self.api_key:
            # Optional provider compatibility. Open-Meteo currently does not require this.
            forecast_params["apikey"] = self.api_key

        forecast_response = requests.get(
            f"{self.base_url.rstrip('/')}/forecast",
            params=forecast_params,
            timeout=self.timeout,
        )
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        current = forecast_data["current"]
        hourly = forecast_data.get("hourly", {})

        times = hourly.get("time") or []
        precipitation = hourly.get("precipitation") or []
        current_time = current.get("time")

        start_index = 0
        if current_time in times:
            start_index = times.index(current_time)

        rainfall_mm_next_3h = self._sum_precipitation(precipitation, start_index, hours=3)
        rainfall_mm_next_6h = self._sum_precipitation(precipitation, start_index, hours=6)

        weather_code = int(current.get("weather_code", -1))
        weather_main, weather_description = WEATHER_CODE_MAP.get(
            weather_code, ("Unknown", "No weather description available")
        )

        return {
            "city": resolved_city,
            "timestamp": str(current.get("time") or ""),
            "temperature_c": float(current.get("temperature_2m", 0.0)),
            "feels_like_c": float(current.get("apparent_temperature", 0.0)),
            "humidity": int(current.get("relative_humidity_2m", 0)),
            "rainfall_mm_next_3h": rainfall_mm_next_3h,
            "rainfall_mm_next_6h": rainfall_mm_next_6h,
            "weather_main": weather_main,
            "weather_description": weather_description,
            "wind_speed": float(current.get("wind_speed_10m", 0.0)),
            "weather_source": "open-meteo-live",
        }

    def _sum_precipitation(
        self,
        precipitation: list[Any],
        start_index: int,
        hours: int,
    ) -> float:
        """Safely sum forecast precipitation for the next N hours."""
        if not precipitation:
            return 0.0

        # Exclude the current hour and sum the next N hours.
        start = max(start_index + 1, 0)
        end = max(start, start + hours)
        values = precipitation[start:end]

        total = 0.0
        for value in values:
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue

        return round(total, 2)

    def get_mock_weather(self, city: str, error_message: str | None = None) -> Dict[str, Any]:
        """Return stable sample weather for the requested city as a fallback."""
        normalized_city = city.strip() or "Unknown"
        seed = sum(ord(ch) for ch in normalized_city.lower())

        # Deterministic but city-specific fallback values.
        temperature_c = round(11.0 + (seed % 120) / 10.0, 1)
        rainfall_3h = round((seed % 25) / 10.0, 1)
        rainfall_6h = round(rainfall_3h + 1.2, 1)

        return {
            "city": normalized_city,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature_c": temperature_c,
            "feels_like_c": round(temperature_c - 0.6, 1),
            "humidity": 65 + (seed % 20),
            "rainfall_mm_next_3h": rainfall_3h,
            "rainfall_mm_next_6h": rainfall_6h,
            "weather_main": "Clouds",
            "weather_description": "Fallback weather profile (live weather unavailable)",
            "wind_speed": round(8.0 + (seed % 90) / 10.0, 1),
            "weather_source": "fallback_proxy",
            "weather_error": error_message or "Live weather fetch failed",
        }
