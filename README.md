# Urban AI Copilot: A Smart City Risk & Response Platform

Urban AI Copilot is a modular decision-support application for city operations teams. It combines live environmental signals, operational context, explainable risk scoring, and copilot recommendations to support faster, data-informed planning.

---

## Project Overview

The platform ingests weather and context data, builds a unified city state, computes **Flood**, **Traffic Disruption**, and **Heat Stress** risks, applies what-if scenario simulation, and produces operational recommendations through AI (when available) or deterministic fallback logic.

This project is designed as a portfolio-ready prototype with clear architecture, robust error handling, and transparent scoring logic.

## Motivation

City planners often need to make fast decisions across multiple overlapping risks (weather, mobility, and infrastructure pressure) with incomplete data. This project demonstrates how to:

- unify heterogeneous inputs into one operational view,
- produce explainable risk outputs,
- compare baseline vs simulated scenarios,
- and generate concise action recommendations.

## Architecture

The application follows a layered, modular structure:

1. **Services layer**: fetches external data (weather, traffic, hydrology).
2. **Context layer**: enriches data with operational logic (rush hour, weekend, vulnerability).
3. **Core data platform**: builds a unified `CityState` object.
4. **Risk layer**: computes rule-based risk scores (0–100) with reasons.
5. **Simulation layer**: applies deterministic scenario adjustments.
6. **Agent layer**: generates recommendations via AI (OpenAI-compatible) or fallback rules.
7. **UI layer**: Streamlit dashboard for interaction and visualization.

### Data Flow

`WeatherService + TrafficService + FloodService -> ContextService -> CityState -> Risk Models -> Scenario Engine -> Copilot Agent -> Streamlit UI`

## Features

- Live weather retrieval (Open-Meteo)
- Optional live traffic context (TomTom free tier)
- Optional live hydrology context (PEGELONLINE)
- Unified `CityState` model for downstream processing
- Explainable risk models:
  - Flood Risk
  - Traffic Disruption Risk
  - Heat Stress Risk
- Scenario simulation controls:
  - rainfall multiplier
  - temperature delta
  - force rush hour
- AI recommendation mode (OpenAI-compatible endpoint)
- Automatic fallback to deterministic recommendations if AI is unavailable
- Professional Streamlit dashboard with baseline vs simulated comparison

## Folder Structure

```text
urban-ai-copilot/
  app.py
  requirements.txt
  README.md
  .env.example

  core/
    __init__.py
    data_model.py
    urban_data_platform.py

  services/
    __init__.py
    weather_service.py
    context_service.py
    traffic_service.py
    flood_service.py

  risk/
    __init__.py
    flood_risk.py
    heat_risk.py
    traffic_risk.py

  agents/
    __init__.py
    copilot_agent.py

  simulation/
    __init__.py
    scenario_engine.py

  utils/
    __init__.py
    config.py
    helpers.py
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- `pip`

### Install

```bash
cd urban-ai-copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys (as needed).

## Environment Variables

```env
# Weather (Open-Meteo works without a key)
WEATHER_API_KEY=
WEATHER_API_BASE_URL=https://api.open-meteo.com/v1

# AI (set OpenAI-compatible OR Gemini OpenAI-compatible settings)
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=

GEMINI_API_KEY=
GEMINI_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.0-flash

# Traffic (optional)
TOMTOM_API_KEY=
TOMTOM_API_BASE_URL=https://api.tomtom.com/traffic/services/4

# Flood/Hydrology (optional)
PEGELONLINE_API_BASE_URL=https://www.pegelonline.wsv.de/webservices/rest-api/v2

# App defaults
DEFAULT_CITY=Berlin
SUPPORTED_CITIES=Berlin,Hamburg,Munich,Cologne,Frankfurt
```

> Do not commit real keys. Keep secrets only in `.env`.

## How to Run Locally

```bash
cd urban-ai-copilot
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## RAG Knowledge Setup

When `RAG_ENABLED=true`, the app retrieves guidance snippets from local knowledge files
stored in `data/knowledge` and indexed into local Qdrant storage.

To build or rebuild the index manually from the project root, run:

```bash
cd urban-ai-copilot
python -m rag.ingest
```

The app also attempts to bootstrap the local knowledge index automatically at startup
when RAG is enabled. This helps deployments where `data/qdrant` is not committed and
must be recreated from the checked-in knowledge files.

## How Scenario Simulation Works

Simulation is applied to the baseline `CityState` in a deterministic way:

- **rainfall multiplier** scales near-term rainfall fields
- **temperature delta** shifts both temperature and feels-like values
- **force rush hour** overrides rush-hour state to `True`

After simulation, risks are recomputed and displayed alongside baseline values in the **Scenario Comparison** section.

## Flood ML (Current Scope)

- Flood prediction runs in **Auto mode** inside the app:
  - use ML model (`models/flood_risk.pkl`) when available,
  - otherwise fallback to rule-based flood scoring.
- The current global model is trained mostly on Dortmund-collected samples,
  so predictions for other cities are more generalized until broader city
  coverage is logged.

### Retraining guidance

When fresh data accumulates in `data/flood_training.csv`, retrain the model:

```bash
python ml/train_flood_model.py --min-rows 30 --cv-folds 5
```

Training outputs include:
- `models/flood_risk.pkl` (model artifact)
- `models/flood_feature_importance.csv` (feature importance)
- `models/flood_model_metadata.json` (training snapshot + metrics)

## Limitations

- Risk models are transparent heuristics, not calibrated predictive models.
- Data coverage depends on external provider availability and quota.
- Traffic and flood context use fallbacks when provider calls fail.
- AI recommendations depend on endpoint access, quota, and model response quality.

## Future Improvements

- Add historical trend analysis and charting
- Add district-level geospatial layers and map visualization
- Add model calibration against historical incidents
- Add observability (structured logs, metrics, diagnostics panel)
- Add tests for service integrations and risk threshold behavior
- Add CI/CD and containerized deployment

## Deployment (Streamlit Community Cloud)

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Select repository and set:
   - **Main file path**: `app.py`
4. Add required secrets in Streamlit app settings (same keys as `.env`).
5. Deploy.

### Recommended Streamlit Cloud settings

- Python version: **3.10+**
- Entrypoint: **`app.py`**
- Config file: **`.streamlit/config.toml`** (included)

---

This project is a **prototype decision-support system** for urban operations and planning. It is not a replacement for official emergency-management systems or domain-certified forecasting tools.
