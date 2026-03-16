"""Helpers for inspecting flood-ML artifacts and training freshness."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FloodModelDiagnostics:
    """Summary of flood-model artifact health and metadata."""

    model_exists: bool
    data_exists: bool
    importance_exists: bool
    model_path: str
    data_path: str
    importance_path: str
    model_modified_utc: str
    data_modified_utc: str
    metadata_modified_utc: str
    is_stale: bool
    top_features: list[dict[str, str]]
    stale_reason: str


def collect_flood_model_diagnostics(
    model_path: Path | None = None,
    data_path: Path | None = None,
    importance_path: Path | None = None,
    metadata_path: Path | None = None,
    top_n: int = 7,
) -> FloodModelDiagnostics:
    """Collect health/freshness details for flood model artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    model_file = model_path or (project_root / "models" / "flood_risk.pkl")
    data_file = data_path or (project_root / "data" / "flood_training.csv")
    importance_file = importance_path or (project_root / "models" / "flood_feature_importance.csv")
    metadata_file = metadata_path or (project_root / "models" / "flood_model_metadata.json")

    model_exists = model_file.exists()
    data_exists = data_file.exists()
    importance_exists = importance_file.exists()

    model_modified = _safe_mtime_utc(model_file)
    data_modified = _safe_mtime_utc(data_file)
    metadata_modified = _safe_mtime_utc(metadata_file)

    is_stale, stale_reason = _evaluate_staleness(
        model_file=model_file,
        data_file=data_file,
        metadata_file=metadata_file,
    )

    return FloodModelDiagnostics(
        model_exists=model_exists,
        data_exists=data_exists,
        importance_exists=importance_exists,
        model_path=str(model_file),
        data_path=str(data_file),
        importance_path=str(importance_file),
        model_modified_utc=model_modified,
        data_modified_utc=data_modified,
        metadata_modified_utc=metadata_modified,
        is_stale=is_stale,
        top_features=_read_top_feature_importances(importance_file, top_n=top_n),
        stale_reason=stale_reason,
    )


def _evaluate_staleness(
    model_file: Path,
    data_file: Path,
    metadata_file: Path,
) -> tuple[bool, str]:
    """Evaluate staleness using metadata first, mtime fallback otherwise."""
    if not model_file.exists():
        return True, "Model artifact is missing."
    if not data_file.exists():
        return False, "Training data file missing; cannot assess freshness."

    if metadata_file.exists():
        metadata = _read_metadata(metadata_file)
        trained_at_utc = str(metadata.get("trained_at_utc", ""))
        rows_used_original = int(metadata.get("rows_used_original", 0) or 0)

        latest_logged_at_utc = _latest_logged_timestamp_utc(data_file)
        current_rows = _safe_row_count(data_file)

        # Primary: has new logged row after training snapshot?
        if trained_at_utc and latest_logged_at_utc and latest_logged_at_utc > trained_at_utc:
            return True, "New training samples were logged after last model training."

        # Secondary: row-count drift indicates data changed.
        if rows_used_original and current_rows > rows_used_original:
            return True, "Training data row count increased since last model training."

        return False, "Model is aligned with latest known training snapshot."

    # Fallback path when metadata is not available.
    if data_file.stat().st_mtime > model_file.stat().st_mtime:
        return True, "Training data file modified after model artifact time (mtime fallback)."
    return False, "Model is newer than training data file (mtime fallback)."


def _read_metadata(path: Path) -> dict[str, object]:
    """Read metadata JSON safely."""
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {}


def _latest_logged_timestamp_utc(path: Path) -> str:
    """Get latest logged_at_utc value from training CSV."""
    if not path.exists():
        return ""

    latest = ""
    try:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                ts = str(row.get("logged_at_utc", ""))
                if ts and ts > latest:
                    latest = ts
    except Exception:
        return ""
    return latest


def _safe_row_count(path: Path) -> int:
    """Count CSV data rows (excluding header)."""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
        return max(0, len(rows) - 1)
    except Exception:
        return 0


def _safe_mtime_utc(path: Path) -> str:
    """Return UTC ISO modified time or 'N/A' when path is missing."""
    if not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _read_top_feature_importances(path: Path, top_n: int) -> list[dict[str, str]]:
    """Read top-N feature importances from CSV file if available."""
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for idx, row in enumerate(reader):
                if idx >= top_n:
                    break
                rows.append(
                    {
                        "feature": str(row.get("feature", "")),
                        "importance": str(row.get("importance", "")),
                    }
                )
    except Exception:
        return []

    return rows
