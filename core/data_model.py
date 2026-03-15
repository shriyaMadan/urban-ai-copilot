"""Data models for weather context, risks, and recommendations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


@dataclass
class CopilotRecommendation:
    """Actionable recommendations for city operations teams."""
    title: str
    summary: str
    priority: str
    actions: List[str]


@dataclass(frozen=True)
class CityState:
    """Unified city-state used as the central urban operations snapshot.

    This structure combines weather signals and operational context into one
    normalized object that downstream risk and copilot modules can consume.
    """

    city: str
    timestamp: str
    temperature_c: float
    feels_like_c: float
    humidity: int
    rainfall_mm_next_3h: float
    rainfall_mm_next_6h: float
    weather_description: str
    wind_speed: float
    is_rush_hour: bool
    is_weekend: bool
    district_vulnerability: str
    traffic_congestion_index: float = 0.0
    traffic_speed_kph: float = 0.0
    traffic_free_flow_speed_kph: float = 0.0
    traffic_source: str = "proxy"
    river_water_level_cm: float = 0.0
    river_trend: str = "unknown"
    flood_data_source: str = "weather_proxy"

    @classmethod
    def from_service_outputs(
        cls,
        weather_data: Mapping[str, Any],
        context_data: Mapping[str, Any],
    ) -> "CityState":
        """Build a normalized city state from weather/context service outputs."""
        return cls(
            city=str(weather_data.get("city", "Unknown")),
            timestamp=str(weather_data.get("timestamp", "")),
            temperature_c=float(weather_data.get("temperature_c", 0.0)),
            feels_like_c=float(weather_data.get("feels_like_c", 0.0)),
            humidity=int(weather_data.get("humidity", 0)),
            rainfall_mm_next_3h=float(weather_data.get("rainfall_mm_next_3h", 0.0)),
            rainfall_mm_next_6h=float(weather_data.get("rainfall_mm_next_6h", 0.0)),
            weather_description=str(weather_data.get("weather_description", "")),
            wind_speed=float(weather_data.get("wind_speed", 0.0)),
            is_rush_hour=bool(context_data.get("is_rush_hour", False)),
            is_weekend=bool(context_data.get("is_weekend", False)),
            district_vulnerability=str(
                context_data.get("district_vulnerability", "medium")
            ),
            traffic_congestion_index=float(
                context_data.get("traffic_congestion_index", 0.0)
            ),
            traffic_speed_kph=float(context_data.get("traffic_speed_kph", 0.0)),
            traffic_free_flow_speed_kph=float(
                context_data.get("traffic_free_flow_speed_kph", 0.0)
            ),
            traffic_source=str(context_data.get("traffic_source", "proxy")),
            river_water_level_cm=float(context_data.get("river_water_level_cm", 0.0)),
            river_trend=str(context_data.get("river_trend", "unknown")),
            flood_data_source=str(
                context_data.get("flood_data_source", "weather_proxy")
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return CityState as a plain dictionary for UI and serialization."""
        return {
            "city": self.city,
            "timestamp": self.timestamp,
            "temperature_c": self.temperature_c,
            "feels_like_c": self.feels_like_c,
            "humidity": self.humidity,
            "rainfall_mm_next_3h": self.rainfall_mm_next_3h,
            "rainfall_mm_next_6h": self.rainfall_mm_next_6h,
            "weather_description": self.weather_description,
            "wind_speed": self.wind_speed,
            "is_rush_hour": self.is_rush_hour,
            "is_weekend": self.is_weekend,
            "district_vulnerability": self.district_vulnerability,
            "traffic_congestion_index": self.traffic_congestion_index,
            "traffic_speed_kph": self.traffic_speed_kph,
            "traffic_free_flow_speed_kph": self.traffic_free_flow_speed_kph,
            "traffic_source": self.traffic_source,
            "river_water_level_cm": self.river_water_level_cm,
            "river_trend": self.river_trend,
            "flood_data_source": self.flood_data_source,
        }
