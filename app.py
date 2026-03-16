"""Streamlit entrypoint for the Urban AI Copilot."""
from __future__ import annotations

from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Urban AI Copilot",
    page_icon="🌆",
    layout="wide",
)

from agents.copilot_agent import CopilotAgent, CopilotPlan
from core.data_model import CityState
from ml.model_diagnostics import collect_flood_model_diagnostics
from ml.flood_predictor import FloodMLPredictor
from risk.flood_risk import FloodRiskModel
from risk.heat_risk import HeatRiskModel
from risk.traffic_risk import TrafficRiskModel
from services.context_service import ContextService
from services.flood_service import FloodService
from services.traffic_service import TrafficService
from services.weather_service import WeatherService
from simulation.scenario_engine import ScenarioEngine
from utils.config import AppConfig
from utils.flood_training_logger import log_flood_training_sample


CITY_OPTIONS = ["Dortmund", "Bochum", "Essen", "Cologne", "Berlin"]


def _resolve_city_options(configured_cities: tuple[str, ...]) -> list[str]:
    """Merge required showcase cities with env-configured city list."""
    merged: list[str] = []

    for city in CITY_OPTIONS:
        if city not in merged:
            merged.append(city)

    for city in configured_cities:
        cleaned = city.strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)

    return merged


def _default_weather_payload(city: str) -> dict[str, Any]:
    """Return safe weather payload for degraded mode rendering."""
    return {
        "city": city,
        "timestamp": "",
        "temperature_c": 0.0,
        "feels_like_c": 0.0,
        "humidity": 0,
        "rainfall_mm_next_3h": 0.0,
        "rainfall_mm_next_6h": 0.0,
        "weather_description": "Unavailable",
        "wind_speed": 0.0,
        "weather_source": "fallback_proxy",
        "weather_error": "Weather data not loaded yet",
    }


def _default_context_payload(district_vulnerability: str) -> dict[str, Any]:
    """Return safe operational context payload for degraded mode rendering."""
    return {
        "is_rush_hour": False,
        "is_weekend": False,
        "district_vulnerability": district_vulnerability,
        "local_hour": 0,
        "traffic_congestion_index": 0.0,
        "traffic_speed_kph": 0.0,
        "traffic_free_flow_speed_kph": 0.0,
        "traffic_source": "proxy",
        "river_water_level_cm": 0.0,
        "river_trend": "unknown",
        "flood_data_source": "weather_proxy",
    }


def _build_fallback_city_state(city: str, district_vulnerability: str) -> CityState:
    """Return a safe fallback city state if data loading fails unexpectedly."""
    return CityState(
        city=city,
        timestamp="",
        temperature_c=0.0,
        feels_like_c=0.0,
        humidity=0,
        rainfall_mm_next_3h=0.0,
        rainfall_mm_next_6h=0.0,
        weather_description="Unavailable",
        wind_speed=0.0,
        is_rush_hour=False,
        is_weekend=False,
        district_vulnerability=district_vulnerability,
    )


def _fallback_risk_payload(risk_name: str, reason: str) -> dict[str, Any]:
    """Return safe risk payload when risk computation fails."""
    return {
        "risk_name": risk_name,
        "score": 0,
        "level": "Low",
        "reason": reason,
    }


def _risk_label(score_payload: dict[str, Any]) -> str:
    """Format risk score payload for metric display."""
    return f"{score_payload['score']} ({score_payload['level']})"


def _build_flood_sample_key(city_state: CityState, flood_risk: dict[str, Any]) -> str:
    """Build a compact key to avoid duplicate training rows per Streamlit rerun."""
    return "|".join(
        [
            city_state.city,
            city_state.timestamp,
            city_state.district_vulnerability,
            str(round(city_state.rainfall_mm_next_6h, 2)),
            str(round(city_state.river_water_level_cm, 1)),
            str(int(flood_risk.get("score", 0))),
        ]
    )


def _log_flood_training_row(city_state: CityState, flood_risk: dict[str, Any]) -> None:
    """Persist one flood training sample with rerun-safe deduping."""
    sample_key = _build_flood_sample_key(city_state, flood_risk)
    if st.session_state.get("last_flood_training_sample_key") == sample_key:
        return

    try:
        log_flood_training_sample(city_state=city_state, flood_risk=flood_risk)
        st.session_state["last_flood_training_sample_key"] = sample_key
    except Exception:
        # Logging should never break the dashboard experience.
        pass


def _load_baseline_city_state(
    city: str,
    district_vulnerability: str,
    weather_service: WeatherService,
    context_service: ContextService,
) -> tuple[CityState, dict[str, Any], dict[str, Any]]:
    """Fetch weather/context and build baseline CityState with robust fallbacks."""
    weather_data = _default_weather_payload(city)
    context_data = _default_context_payload(district_vulnerability)

    # 1) Fetch weather data
    try:
        weather_data = weather_service.get_forecast(city)
    except Exception:
        st.warning("Weather feed temporarily unavailable. Using fallback weather values.")

    # 2) Fetch context data (rush hour + optional traffic/flood service signals)
    try:
        context_data = context_service.build_context(
            city=city,
            district_vulnerability=district_vulnerability,
        )
    except Exception:
        st.warning("Operational context feed unavailable. Using fallback context values.")

    # 3) Build CityState
    try:
        baseline_state = CityState.from_service_outputs(weather_data, context_data)
    except Exception:
        st.warning("Could not build full city state. Rendering safe fallback city state.")
        baseline_state = _build_fallback_city_state(city, district_vulnerability)

    return baseline_state, weather_data, context_data


def _compute_risks(
    city_state: CityState,
    flood_model: FloodRiskModel,
    heat_model: HeatRiskModel,
    traffic_model: TrafficRiskModel,
    phase_label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute risk triplet with per-risk error isolation."""
    try:
        flood_risk = flood_model.evaluate(city_state)
    except Exception:
        st.warning(f"{phase_label}: flood risk model unavailable, using fallback score.")
        flood_risk = _fallback_risk_payload(
            "Flood Risk",
            "Flood model unavailable; fallback score applied.",
        )

    try:
        heat_risk = heat_model.evaluate(city_state)
    except Exception:
        st.warning(f"{phase_label}: heat risk model unavailable, using fallback score.")
        heat_risk = _fallback_risk_payload(
            "Heat Risk",
            "Heat model unavailable; fallback score applied.",
        )

    try:
        traffic_risk = traffic_model.evaluate(city_state)
    except Exception:
        st.warning(f"{phase_label}: traffic risk model unavailable, using fallback score.")
        traffic_risk = _fallback_risk_payload(
            "Traffic Disruption Risk",
            "Traffic model unavailable; fallback score applied.",
        )

    return flood_risk, heat_risk, traffic_risk


def _apply_ml_flood_prediction(
    city_state: CityState,
    rule_based_flood_risk: dict[str, Any],
    predictor: FloodMLPredictor,
) -> dict[str, Any]:
    """Replace flood score with ML prediction when model artifact is available."""
    if not predictor.is_ready:
        fallback_payload = dict(rule_based_flood_risk)
        fallback_payload["source"] = "rule_based"
        fallback_payload["reason"] = (
            f"{fallback_payload.get('reason', '')} "
            "(ML model unavailable, using rule-based flood score.)"
        ).strip()
        return fallback_payload

    try:
        ml_prediction = predictor.predict(city_state)
        return {
            "risk_name": "Flood Risk",
            "score": ml_prediction.score,
            "level": ml_prediction.level,
            "reason": (
                f"{ml_prediction.detail} "
                f"(rule-based reference={rule_based_flood_risk.get('score', 0)})."
            ),
            "source": ml_prediction.source,
        }
    except Exception:
        fallback_payload = dict(rule_based_flood_risk)
        fallback_payload["source"] = "rule_based"
        fallback_payload["reason"] = (
            f"{fallback_payload.get('reason', '')} "
            "(ML prediction failed, using rule-based flood score.)"
        ).strip()
        return fallback_payload


def _generate_copilot_plan(
    copilot_agent: CopilotAgent,
    city_state: CityState,
    flood_risk: dict[str, Any],
    heat_risk: dict[str, Any],
    traffic_risk: dict[str, Any],
) -> CopilotPlan:
    """Generate recommendations with fail-safe fallback behavior."""
    try:
        return copilot_agent.generate_operational_recommendations(
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )
    except Exception:
        st.warning("Copilot AI unavailable. Falling back to deterministic recommendations.")
        fallback_agent = CopilotAgent(has_ai_key=False)
        return fallback_agent.generate_operational_recommendations(
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )


def main() -> None:
    """Render a polished decision-support dashboard."""
    config = AppConfig.from_env()

    weather_service = WeatherService(
        api_key=config.weather_api_key,
        base_url=config.weather_api_base_url,
        timeout=60,
    )
    context_service = ContextService()
    traffic_service = TrafficService(
        api_key=config.tomtom_api_key,
        base_url=config.tomtom_api_base_url,
    )
    flood_service = FloodService(base_url=config.pegelonline_api_base_url)
    context_service.attach_optional_services(
        traffic_service=traffic_service,
        flood_service=flood_service,
    )

    flood_model = FloodRiskModel()
    flood_ml_predictor = FloodMLPredictor()
    flood_ml_diagnostics = collect_flood_model_diagnostics(top_n=5)
    heat_model = HeatRiskModel()
    traffic_model = TrafficRiskModel()
    scenario_engine = ScenarioEngine()
    copilot_agent = CopilotAgent(has_ai_key=config.has_ai_provider_key)

    header = st.container()
    with header:
        st.title("Urban AI Copilot")
        st.caption(
            "Prototype decision-support dashboard for urban risk monitoring, "
            "scenario simulation, and operations planning."
        )

    available_cities = _resolve_city_options(config.supported_cities)
    default_city = config.default_city if config.default_city in available_cities else "Berlin"
    default_city_index = available_cities.index(default_city)

    with st.sidebar:
        st.header("Control Panel")

        city = st.selectbox(
            "German City",
            options=available_cities,
            index=default_city_index,
        )

        district_vulnerability = st.selectbox(
            "District Vulnerability",
            options=["low", "medium", "high"],
            index=1,
        )

        st.markdown("### Simulation")
        rainfall_multiplier = st.slider("Rainfall Multiplier", 0.5, 2.0, 1.0, 0.1)
        temperature_delta = st.slider("Temperature Delta (°C)", -5.0, 5.0, 0.0, 0.5)
        force_rush_hour = st.checkbox("Force Rush Hour Scenario", value=False)

        st.markdown("### Flood ML Status")
        if flood_ml_diagnostics.model_exists and not flood_ml_diagnostics.is_stale:
            st.success("Flood model: Ready (up-to-date)")
        elif flood_ml_diagnostics.model_exists and flood_ml_diagnostics.is_stale:
            st.warning("Flood model: Available, but stale vs training data")
        else:
            st.warning("Flood model: Missing (rule-based fallback active)")

        st.caption(f"Freshness note: {flood_ml_diagnostics.stale_reason}")
        st.info(
            "Coverage note: model is currently trained mostly on Dortmund-collected "
            "samples; predictions for other cities are more generalized."
        )

        if flood_ml_diagnostics.top_features:
            with st.expander("Top Flood ML Features"):
                st.table(flood_ml_diagnostics.top_features)

        refresh_pressed = st.button("Refresh")
        if refresh_pressed:
            st.rerun()

    baseline_state, weather_data, context_data = _load_baseline_city_state(
        city=city,
        district_vulnerability=district_vulnerability,
        weather_service=weather_service,
        context_service=context_service,
    )

    weather_source = str(weather_data.get("weather_source", "unknown"))
    if weather_source == "fallback_proxy":
        weather_error = str(weather_data.get("weather_error", "Unknown weather fetch error"))
        st.warning(
            "Weather API timed out/unavailable for this run. "
            f"Using fallback weather. Details: {weather_error}"
        )

    # 4) Apply scenario simulation
    try:
        simulated_state = scenario_engine.apply(
            city_state=baseline_state,
            rainfall_multiplier=rainfall_multiplier,
            temperature_delta=temperature_delta,
            force_rush_hour=force_rush_hour,
        )
    except Exception:
        st.warning("Scenario simulation failed. Using baseline state as simulated state.")
        simulated_state = baseline_state

    # 5) Compute baseline risks
    baseline_flood_rule, baseline_heat, baseline_traffic = _compute_risks(
        city_state=baseline_state,
        flood_model=flood_model,
        heat_model=heat_model,
        traffic_model=traffic_model,
        phase_label="Baseline",
    )

    # Auto mode: prefer ML flood prediction when available, fallback to rules.
    baseline_flood = _apply_ml_flood_prediction(
        city_state=baseline_state,
        rule_based_flood_risk=baseline_flood_rule,
        predictor=flood_ml_predictor,
    )

    # Persist one training sample (features + synthetic flood label) for ML phase.
    _log_flood_training_row(city_state=baseline_state, flood_risk=baseline_flood_rule)

    # 6) Compute simulated risks
    simulated_flood_rule, simulated_heat, simulated_traffic = _compute_risks(
        city_state=simulated_state,
        flood_model=flood_model,
        heat_model=heat_model,
        traffic_model=traffic_model,
        phase_label="Simulation",
    )

    # Prefer ML flood prediction in simulation when model artifact is available.
    simulated_flood = _apply_ml_flood_prediction(
        city_state=simulated_state,
        rule_based_flood_risk=simulated_flood_rule,
        predictor=flood_ml_predictor,
    )

    # 7) Generate recommendations
    copilot_plan = _generate_copilot_plan(
        copilot_agent=copilot_agent,
        city_state=simulated_state,
        flood_risk=simulated_flood,
        heat_risk=simulated_heat,
        traffic_risk=simulated_traffic,
    )

    # 2) Current city snapshot card
    st.subheader("Current City Snapshot")
    with st.container(border=True):
        snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
        with snap_col1:
            st.metric("City", simulated_state.city)
            st.caption(f"Timestamp: {simulated_state.timestamp or 'N/A'}")
        with snap_col2:
            st.metric("Temperature", f"{simulated_state.temperature_c:.1f} °C")
            st.caption(f"Feels Like: {simulated_state.feels_like_c:.1f} °C")
        with snap_col3:
            st.metric("Humidity", f"{simulated_state.humidity}%")
            st.caption(f"Rain (6h): {simulated_state.rainfall_mm_next_6h:.1f} mm")
        with snap_col4:
            st.metric("Traffic Congestion", f"{simulated_state.traffic_congestion_index:.1f}%")
            st.caption(
                f"Sources: traffic={simulated_state.traffic_source}, "
                f"flood={simulated_state.flood_data_source}"
            )
        st.caption(
            f"Weather: {weather_data.get('weather_description', 'N/A')} "
            f"(source={weather_data.get('weather_source', 'unknown')}) | "
            f"Rush hour: {context_data.get('is_rush_hour', False)}"
        )

    # 3) Three risk cards
    st.subheader("Risk Overview")
    st.caption("All risk scores are on a 0–100 scale (i.e., score / 100).")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        with st.container(border=True):
            st.markdown("### Flood Risk")
            st.metric("Current", _risk_label(simulated_flood))
            st.caption(f"Source: {simulated_flood.get('source', 'rule_based')}")
            st.caption(simulated_flood["reason"])
    with risk_col2:
        with st.container(border=True):
            st.markdown("### Traffic Disruption Risk")
            st.metric("Current", _risk_label(simulated_traffic))
            st.caption(simulated_traffic["reason"])
    with risk_col3:
        with st.container(border=True):
            st.markdown("### Heat Stress Risk")
            st.metric("Current", _risk_label(simulated_heat))
            st.caption(simulated_heat["reason"])

    # 4) AI Copilot recommendations
    st.subheader("AI Copilot Recommendations")
    with st.container(border=True):
        meta_col1, meta_col2 = st.columns([1, 1])
        with meta_col1:
            st.metric("Copilot Mode", copilot_plan.mode)
        with meta_col2:
            st.metric("Urgency", copilot_plan.urgency)

        st.caption(copilot_plan.rationale)
        for idx, item in enumerate(copilot_plan.recommendations, start=1):
            st.markdown(f"**{idx}. {item.title}** — {item.summary}")

    # 5) Scenario comparison
    st.subheader("Scenario Comparison")
    with st.container(border=True):
        comparison_rows = [
            {
                "Metric": "Temperature (°C)",
                "Baseline": f"{baseline_state.temperature_c:.1f}",
                "Simulated": f"{simulated_state.temperature_c:.1f}",
            },
            {
                "Metric": "Rainfall Next 6h (mm)",
                "Baseline": f"{baseline_state.rainfall_mm_next_6h:.1f}",
                "Simulated": f"{simulated_state.rainfall_mm_next_6h:.1f}",
            },
            {
                "Metric": "Rush Hour",
                "Baseline": str(baseline_state.is_rush_hour),
                "Simulated": str(simulated_state.is_rush_hour),
            },
            {
                "Metric": "Flood Risk Score",
                "Baseline": str(baseline_flood["score"]),
                "Simulated": str(simulated_flood["score"]),
            },
            {
                "Metric": "Flood Risk Source",
                "Baseline": str(baseline_flood.get("source", "rule_based")),
                "Simulated": str(simulated_flood.get("source", "rule_based")),
            },
            {
                "Metric": "Traffic Risk Score",
                "Baseline": str(baseline_traffic["score"]),
                "Simulated": str(simulated_traffic["score"]),
            },
            {
                "Metric": "Heat Risk Score",
                "Baseline": str(baseline_heat["score"]),
                "Simulated": str(simulated_heat["score"]),
            },
        ]
        st.table(comparison_rows)

    st.markdown("---")
    # 6) Footer note
    st.caption(
        "Prototype decision-support system for urban operations. "
        "Recommendations support planners but do not replace official emergency procedures."
    )


if __name__ == "__main__":
    main()
