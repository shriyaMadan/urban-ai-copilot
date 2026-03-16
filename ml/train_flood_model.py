"""Train a RandomForest flood-risk regressor from logged training CSV."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    # Works when launched as module: `python -m ml.train_flood_model`
    from ml.flood_features import MODEL_FLOOD_FEATURE_COLUMNS, build_flood_feature_frame
except ModuleNotFoundError:
    # Works when launched as file path: `python ml/train_flood_model.py`
    from flood_features import MODEL_FLOOD_FEATURE_COLUMNS, build_flood_feature_frame


BASE_TARGET_COLUMN = "synthetic_label_score"
TARGET_COLUMN = "synthetic_label_score_adjusted"


@dataclass(frozen=True)
class TrainArtifacts:
    """Output paths for trained model and model-analysis artifacts."""

    model_path: Path
    feature_importance_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CrossValidationSummary:
    """Cross-validation summary metrics."""

    folds: int
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    mae_valid_folds: int
    rmse_valid_folds: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Hold MAE/RMSE evaluation metrics."""

    mae: float
    rmse: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training script."""
    parser = argparse.ArgumentParser(description="Train flood-risk RandomForest model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/flood_training.csv"),
        help="Path to the training CSV produced by flood logger.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/flood_risk.pkl"),
        help="Path to save trained model pickle.",
    )
    parser.add_argument(
        "--importance-path",
        type=Path,
        default=Path("models/flood_feature_importance.csv"),
        help="Path to save feature importance CSV.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("models/flood_model_metadata.json"),
        help="Path to save training metadata JSON.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible split/model training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (0-1).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Minimum required rows before training.",
    )
    parser.add_argument(
        "--no-persist-adjusted-labels",
        action="store_true",
        help="Do not write adjusted labels back to training CSV.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation summary.",
    )
    return parser.parse_args()


def _clip_0_100(series: pd.Series) -> pd.Series:
    """Clamp series values to expected risk-score range."""
    return series.clip(lower=0.0, upper=100.0)


def _build_rainfall_dominant_target(dataframe: pd.DataFrame) -> pd.Series:
    """Create rainfall-dominant synthetic target for next-6h flood risk.

    This keeps prototype labels aligned with requested behavior:
    rainfall and river level dominate, with smaller weather/context modifiers.
    """
    rainfall_6h = pd.to_numeric(dataframe.get("rainfall_mm_next_6h", 0), errors="coerce").fillna(0.0)
    rainfall_3h = pd.to_numeric(dataframe.get("rainfall_mm_next_3h", 0), errors="coerce").fillna(0.0)
    river_level = pd.to_numeric(dataframe.get("river_water_level_cm", 0), errors="coerce").fillna(0.0)
    humidity = pd.to_numeric(dataframe.get("humidity", 0), errors="coerce").fillna(0.0)
    wind_speed = pd.to_numeric(dataframe.get("wind_speed", 0), errors="coerce").fillna(0.0)
    temperature = pd.to_numeric(dataframe.get("temperature_c", 0), errors="coerce").fillna(0.0)

    vulnerability = (
        dataframe.get("district_vulnerability", "medium")
        .astype(str)
        .str.lower()
        .str.strip()
        .map({"low": 0.0, "medium": 1.0, "high": 2.0})
        .fillna(1.0)
    )

    # Dominant flood signals
    rainfall_score = _clip_0_100(rainfall_6h * 8.0 + rainfall_3h * 3.0)
    river_score = _clip_0_100((river_level - 300.0) * 0.14)

    # Secondary modifiers
    humidity_score = _clip_0_100((humidity - 65.0) * 0.22)
    vulnerability_score = vulnerability * 6.0
    wind_score = _clip_0_100((wind_speed - 10.0) * 0.08)
    temperature_score = _clip_0_100((temperature - 20.0) * 0.03)

    rainfall_dominant = _clip_0_100(
        rainfall_score * 0.68
        + river_score * 0.22
        + humidity_score * 0.05
        + vulnerability_score * 0.03
        + wind_score
        + temperature_score
    )

    base_label = pd.to_numeric(dataframe.get(BASE_TARGET_COLUMN, 0), errors="coerce").fillna(0.0)
    adjusted = _clip_0_100(base_label * 0.10 + rainfall_dominant * 0.90)
    return adjusted


def enrich_training_targets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted rainfall-dominant target column to the dataframe."""
    frame = dataframe.copy()
    frame[TARGET_COLUMN] = _build_rainfall_dominant_target(frame)
    return frame


def build_augmented_training_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expand training rows with rainfall stress scenarios.

    This intentionally increases rainfall signal variation so the model learns
    stronger rainfall dependence for next-6h flood prediction.
    """
    multipliers = (0.70, 1.00, 1.35, 1.70, 2.10)
    expanded_frames = []

    for multiplier in multipliers:
        variant = dataframe.copy()
        variant["rainfall_mm_next_6h"] = (
            pd.to_numeric(variant.get("rainfall_mm_next_6h", 0), errors="coerce").fillna(0.0)
            * multiplier
        )
        variant["rainfall_mm_next_3h"] = (
            pd.to_numeric(variant.get("rainfall_mm_next_3h", 0), errors="coerce").fillna(0.0)
            * multiplier
        )
        # River level typically reacts more slowly than rainfall.
        variant["river_water_level_cm"] = (
            pd.to_numeric(variant.get("river_water_level_cm", 0), errors="coerce").fillna(0.0)
            * (1.0 + (multiplier - 1.0) * 0.35)
        )
        variant[TARGET_COLUMN] = _build_rainfall_dominant_target(variant)
        expanded_frames.append(variant)

    augmented = pd.concat(expanded_frames, ignore_index=True)
    return augmented


def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> tuple[RandomForestRegressor, EvaluationMetrics, pd.DataFrame]:
    """Train model and return metrics + feature importances."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = EvaluationMetrics(
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(MODEL_FLOOD_FEATURE_COLUMNS),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, importance_df


def evaluate_baseline(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> EvaluationMetrics:
    """Evaluate a mean-value baseline regressor on the same train/test split."""
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    predictions = baseline.predict(x_test)
    return EvaluationMetrics(
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
    )


def run_cross_validation(
    dataframe: pd.DataFrame,
    base_model: RandomForestRegressor,
    cv_folds: int,
    random_state: int,
) -> CrossValidationSummary:
    """Compute cross-validation metrics for model quality summary."""
    feature_frame = build_flood_feature_frame(dataframe)
    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce").fillna(0.0)

    effective_folds = max(2, min(cv_folds, len(feature_frame)))
    splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    cv_model = clone(base_model)
    mae_scores = -cross_val_score(
        cv_model,
        feature_frame,
        target,
        cv=splitter,
        scoring="neg_mean_absolute_error",
        n_jobs=None,
        error_score=np.nan,
    )

    rmse_scores = -cross_val_score(
        cv_model,
        feature_frame,
        target,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        n_jobs=None,
        error_score=np.nan,
    )

    mae_scores = mae_scores[np.isfinite(mae_scores)]
    rmse_scores = rmse_scores[np.isfinite(rmse_scores)]

    if len(mae_scores) == 0 or len(rmse_scores) == 0:
        raise ValueError("Cross-validation failed: no valid fold metrics were produced.")

    return CrossValidationSummary(
        folds=effective_folds,
        mae_mean=float(mae_scores.mean()),
        mae_std=float(mae_scores.std()),
        rmse_mean=float(rmse_scores.mean()),
        rmse_std=float(rmse_scores.std()),
        mae_valid_folds=len(mae_scores),
        rmse_valid_folds=len(rmse_scores),
    )


def save_model(model: RandomForestRegressor, path: Path) -> None:
    """Persist trained model pickle to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(model, file_obj)


def save_metadata(path: Path, payload: dict[str, object]) -> None:
    """Persist model training metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def main() -> None:
    """Entrypoint for flood-risk model training script."""
    args = parse_args()
    data_path: Path = args.data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at '{data_path}'. "
            "Run the Streamlit app to generate flood_training.csv rows first."
        )

    dataframe = pd.read_csv(data_path)
    if len(dataframe) < args.min_rows:
        raise ValueError(
            f"Need at least {args.min_rows} rows to train robustly; "
            f"found {len(dataframe)}. Collect more logged samples first."
        )

    if BASE_TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Missing target column '{BASE_TARGET_COLUMN}' in training CSV."
        )

    dataframe = enrich_training_targets(dataframe)

    if not args.no_persist_adjusted_labels:
        dataframe.to_csv(data_path, index=False)

    augmented_dataframe = build_augmented_training_frame(dataframe)
    feature_frame = build_flood_feature_frame(augmented_dataframe)
    target = pd.to_numeric(augmented_dataframe[TARGET_COLUMN], errors="coerce").fillna(0.0)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    baseline_metrics = evaluate_baseline(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    model, model_metrics, importance_df = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        random_state=args.random_state,
    )

    cv_summary = run_cross_validation(
        dataframe=augmented_dataframe,
        base_model=model,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    artifacts = TrainArtifacts(
        model_path=args.model_path,
        feature_importance_path=args.importance_path,
        metadata_path=args.metadata_path,
    )
    save_model(model=model, path=artifacts.model_path)

    artifacts.feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(artifacts.feature_importance_path, index=False)

    metadata_payload = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "rows_used_original": len(dataframe),
        "rows_used_augmented": len(augmented_dataframe),
        "target_column": TARGET_COLUMN,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "cv_folds_requested": args.cv_folds,
        "cv_folds_effective": cv_summary.folds,
        "baseline_mae": baseline_metrics.mae,
        "baseline_rmse": baseline_metrics.rmse,
        "model_mae": model_metrics.mae,
        "model_rmse": model_metrics.rmse,
        "cv_mae_mean": cv_summary.mae_mean,
        "cv_mae_std": cv_summary.mae_std,
        "cv_rmse_mean": cv_summary.rmse_mean,
        "cv_rmse_std": cv_summary.rmse_std,
    }
    save_metadata(path=artifacts.metadata_path, payload=metadata_payload)

    print("Flood model training complete")
    print("Pipeline: train/test split -> baseline -> model -> evaluation -> CV")
    print(
        f"Train/Test split: train={len(x_train)}, test={len(x_test)} "
        f"(test_size={args.test_size:.2f})"
    )
    print(f"Rows used (original): {len(dataframe)}")
    print(f"Rows used (augmented): {len(augmented_dataframe)}")
    print(f"Target used: {TARGET_COLUMN} (rainfall-dominant adjusted label)")
    print(f"Saved model: {artifacts.model_path}")
    print(f"Saved feature importances: {artifacts.feature_importance_path}")
    print(f"Saved metadata: {artifacts.metadata_path}")
    print(
        f"Baseline (DummyRegressor mean) -> "
        f"MAE: {baseline_metrics.mae:.3f}, RMSE: {baseline_metrics.rmse:.3f}"
    )
    print(
        f"RandomForest model -> "
        f"MAE: {model_metrics.mae:.3f}, RMSE: {model_metrics.rmse:.3f}"
    )
    print(
        f"Model improvement vs baseline -> "
        f"MAE: {baseline_metrics.mae - model_metrics.mae:.3f}, "
        f"RMSE: {baseline_metrics.rmse - model_metrics.rmse:.3f}"
    )
    print(
        f"CV ({cv_summary.mae_valid_folds}/{cv_summary.folds} folds) MAE: "
        f"{cv_summary.mae_mean:.3f} ± {cv_summary.mae_std:.3f}"
    )
    print(
        f"CV ({cv_summary.rmse_valid_folds}/{cv_summary.folds} folds) RMSE: "
        f"{cv_summary.rmse_mean:.3f} ± {cv_summary.rmse_std:.3f}"
    )
    print("Top features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
