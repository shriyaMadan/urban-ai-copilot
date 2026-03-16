"""Utilities for logging flood-risk training samples to local CSV."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.data_model import CityState


TRAINING_COLUMNS = (
    "logged_at_utc",
    "target_horizon_hours",
    "label_source",
    "city",
    "city_timestamp",
    "temperature_c",
    "feels_like_c",
    "humidity",
    "rainfall_mm_next_3h",
    "rainfall_mm_next_6h",
    "wind_speed",
    "is_rush_hour",
    "is_weekend",
    "district_vulnerability",
    "traffic_congestion_index",
    "traffic_speed_kph",
    "traffic_free_flow_speed_kph",
    "traffic_source",
    "river_water_level_cm",
    "river_trend",
    "flood_data_source",
    "synthetic_label_score",
    "synthetic_label_level",
)


def log_flood_training_sample(
    city_state: CityState,
    flood_risk: Mapping[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Append one supervised training sample for flood-risk model development.

    The label is synthetic for v1 (`FloodRiskModel.evaluate` score/level).
    """
    destination = output_path or _default_output_path()
    destination.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_horizon_hours": 6,
        "label_source": "rule_based_flood_model",
        "city": city_state.city,
        "city_timestamp": city_state.timestamp,
        "temperature_c": city_state.temperature_c,
        "feels_like_c": city_state.feels_like_c,
        "humidity": city_state.humidity,
        "rainfall_mm_next_3h": city_state.rainfall_mm_next_3h,
        "rainfall_mm_next_6h": city_state.rainfall_mm_next_6h,
        "wind_speed": city_state.wind_speed,
        "is_rush_hour": int(city_state.is_rush_hour),
        "is_weekend": int(city_state.is_weekend),
        "district_vulnerability": city_state.district_vulnerability,
        "traffic_congestion_index": city_state.traffic_congestion_index,
        "traffic_speed_kph": city_state.traffic_speed_kph,
        "traffic_free_flow_speed_kph": city_state.traffic_free_flow_speed_kph,
        "traffic_source": city_state.traffic_source,
        "river_water_level_cm": city_state.river_water_level_cm,
        "river_trend": city_state.river_trend,
        "flood_data_source": city_state.flood_data_source,
        "synthetic_label_score": int(flood_risk.get("score", 0)),
        "synthetic_label_level": str(flood_risk.get("level", "Low")),
    }

    _append_csv_row(destination, row)
    return destination


def _append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append row to CSV and write header if file is new."""
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _default_output_path() -> Path:
    """Return local CSV destination for flood training records."""
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "flood_training.csv"
