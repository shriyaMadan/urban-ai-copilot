"""Feature engineering helpers for flood-risk ML workflows."""
from __future__ import annotations

from typing import Dict

import pandas as pd


# Raw columns expected from flood training logger CSV.
RAW_FLOOD_FEATURE_COLUMNS = (
    "rainfall_mm_next_6h",
    "river_water_level_cm",
    "rainfall_mm_next_3h",
    "humidity",
    "district_vulnerability",
    "wind_speed",
    "temperature_c",
)

# Final model input feature names after preprocessing/encoding.
MODEL_FLOOD_FEATURE_COLUMNS = (
    "rainfall_mm_next_6h",
    "river_water_level_cm",
    "rainfall_mm_next_3h",
    "humidity",
    "district_vulnerability_encoded",
    "wind_speed",
    "temperature_c",
)

_DISTRICT_VULNERABILITY_MAP: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def build_flood_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a numeric model-ready feature frame from raw training rows."""
    frame = dataframe.copy()

    for col in RAW_FLOOD_FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0

    for col in (
        "rainfall_mm_next_6h",
        "river_water_level_cm",
        "rainfall_mm_next_3h",
        "humidity",
        "wind_speed",
        "temperature_c",
    ):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["district_vulnerability_encoded"] = (
        frame["district_vulnerability"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(_DISTRICT_VULNERABILITY_MAP)
        .fillna(_DISTRICT_VULNERABILITY_MAP["medium"])
    )

    model_frame = frame[list(MODEL_FLOOD_FEATURE_COLUMNS)].copy()
    model_frame = model_frame.fillna(0.0)

    return model_frame
