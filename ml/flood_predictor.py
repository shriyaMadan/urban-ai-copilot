"""Runtime flood-risk prediction using trained RandomForest model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd

from core.data_model import CityState
from ml.flood_features import build_flood_feature_frame


@dataclass(frozen=True)
class FloodPredictionResult:
    """Prediction payload for UI consumption."""

    score: int
    level: str
    source: str
    detail: str


class FloodMLPredictor:
    """Loads a trained flood model and predicts next-6h flood risk score."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or self._default_model_path()
        self._model = self._load_model(self.model_path)

    @property
    def is_ready(self) -> bool:
        """Whether a trained model is available for inference."""
        return self._model is not None

    def predict(self, city_state: CityState) -> FloodPredictionResult:
        """Predict flood risk score/level for the provided city-state snapshot."""
        if self._model is None:
            return FloodPredictionResult(
                score=0,
                level="Low",
                source="ml_model_unavailable",
                detail="Trained flood model not found. Falling back to rule-based flood risk.",
            )

        feature_frame = self._build_feature_frame(city_state)
        prediction_raw = float(self._model.predict(feature_frame)[0])
        score = int(round(max(0.0, min(100.0, prediction_raw))))
        return FloodPredictionResult(
            score=score,
            level=self._risk_level(score),
            source="ml_random_forest",
            detail="Predicted next-6h flood risk using trained RandomForest model.",
        )

    def _build_feature_frame(self, city_state: CityState) -> pd.DataFrame:
        """Build one-row feature frame aligned with training preprocessing."""
        raw_row = {
            "rainfall_mm_next_6h": city_state.rainfall_mm_next_6h,
            "river_water_level_cm": city_state.river_water_level_cm,
            "rainfall_mm_next_3h": city_state.rainfall_mm_next_3h,
            "humidity": city_state.humidity,
            "district_vulnerability": city_state.district_vulnerability,
            "wind_speed": city_state.wind_speed,
            "temperature_c": city_state.temperature_c,
        }
        return build_flood_feature_frame(pd.DataFrame([raw_row]))

    def _load_model(self, model_path: Path) -> object | None:
        """Safely load serialized model artifact."""
        if not model_path.exists():
            return None
        try:
            with model_path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except Exception:
            return None

    def _default_model_path(self) -> Path:
        """Default model artifact path."""
        project_root = Path(__file__).resolve().parents[1]
        return project_root / "models" / "flood_risk.pkl"

    def _risk_level(self, score: int) -> str:
        """Map numeric score into UI-friendly risk level."""
        if score >= 70:
            return "High"
        if score >= 35:
            return "Medium"
        return "Low"
