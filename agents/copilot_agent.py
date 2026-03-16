"""AI copilot agent for operational recommendations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import requests

from core.data_model import CityState, CopilotRecommendation


@dataclass(frozen=True)
class CopilotPlan:
    """Structured response returned by the copilot module."""

    mode: str
    urgency: str
    rationale: str
    recommendations: List[CopilotRecommendation]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan to a plain dictionary for UI/API use."""
        return {
            "mode": self.mode,
            "urgency": self.urgency,
            "rationale": self.rationale,
            "recommendations": [asdict(item) for item in self.recommendations],
        }


class CopilotAgent:
    """Generates operational recommendations for city planners.

    Two operating modes:
    1) AI mode: enabled when a provider key is available (placeholder for now)
    2) Rule-based fallback mode: deterministic recommendations (implemented)
    """

    def __init__(self, has_ai_key: bool | None = None, timeout: int = 20) -> None:
        self.has_ai_key = has_ai_key
        self.timeout = timeout

    def generate_operational_recommendations(
        self,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> CopilotPlan:
        """Generate structured city operations recommendations.

        Args:
            city_state: Unified city snapshot.
            flood_risk: Structured flood risk output.
            heat_risk: Structured heat risk output.
            traffic_risk: Structured traffic risk output.
        """
        if self._should_use_ai_mode():
            try:
                return self._generate_ai_mode_recommendations(
                    city_state=city_state,
                    flood_risk=flood_risk,
                    heat_risk=heat_risk,
                    traffic_risk=traffic_risk,
                )
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                # Graceful fallback keeps app operational if LLM is unavailable.
                fallback_plan = self._generate_rule_based_recommendations(
                    city_state=city_state,
                    flood_risk=flood_risk,
                    heat_risk=heat_risk,
                    traffic_risk=traffic_risk,
                )
                return CopilotPlan(
                    mode="rule_based_fallback",
                    urgency=fallback_plan.urgency,
                    rationale=(
                        f"{self._format_ai_fallback_reason(exc)} "
                        "Switched to deterministic fallback recommendations."
                    ),
                    recommendations=fallback_plan.recommendations,
                )

        return self._generate_rule_based_recommendations(
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )

    def _generate_rule_based_recommendations(
        self,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> CopilotPlan:
        """Return deterministic fallback recommendations (always 3 items)."""
        risk_scores = {
            "flood": int(flood_risk.get("score", 0)),
            "heat": int(heat_risk.get("score", 0)),
            "traffic": int(traffic_risk.get("score", 0)),
        }
        top_risk = max(risk_scores, key=risk_scores.get)
        max_score = max(risk_scores.values())
        urgency = self._urgency_from_score(max_score)

        rec_1 = CopilotRecommendation(
            title="Activate City Coordination Cell",
            summary=(
                f"Set a {urgency.lower()}-urgency operations cadence for {city_state.city} "
                f"with focus on {top_risk} risk."
            ),
            priority=urgency,
            actions=[
                "Run cross-department standups every 2-4 hours.",
                "Track flood, heat, and traffic indicators on one dashboard.",
            ],
        )

        rec_2 = self._domain_specific_recommendation(
            top_risk=top_risk,
            urgency=urgency,
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )

        rec_3 = CopilotRecommendation(
            title="Public Communication and Field Readiness",
            summary=(
                "Publish concise public guidance and pre-position field teams for "
                "rapid response."
            ),
            priority="Medium" if urgency == "Low" else urgency,
            actions=[
                "Issue a city advisory with district-level precautions.",
                "Pre-stage crews and equipment near likely hotspots.",
            ],
        )

        return CopilotPlan(
            mode="rule_based",
            urgency=urgency,
            rationale=(
                f"Highest current operational pressure is {top_risk} (score {max_score}), "
                "so recommendations prioritize immediate city coordination and risk-specific actions."
            ),
            recommendations=[rec_1, rec_2, rec_3],
        )

    def _generate_ai_mode_recommendations(
        self,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> CopilotPlan:
        """Generate recommendations from an OpenAI-compatible chat endpoint.

        Uses environment variables:
        - OPENAI_API_KEY
        - OPENAI_BASE_URL
        - OPENAI_MODEL
        """
        config = self._load_openai_compatible_config()
        if not config:
            raise ValueError("Missing OpenAI-compatible LLM configuration.")

        system_prompt = (
            "You are an urban operations copilot. Provide concise, professional, "
            "actionable recommendations for municipal planners."
        )
        user_prompt = self._build_llm_prompt(
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )

        response = requests.post(
            f"{config['base_url'].rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": config["model"],
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = self._parse_llm_response(content)

        urgency = self._normalize_urgency(str(parsed.get("urgency", "Medium")))
        rationale = str(parsed.get("rationale", "Operational priorities set from multi-risk context.")).strip()

        raw_recommendations = parsed.get("recommendations", [])
        rec_texts: List[str] = []
        if isinstance(raw_recommendations, list):
            for item in raw_recommendations:
                if isinstance(item, str) and item.strip():
                    rec_texts.append(item.strip())
                elif isinstance(item, dict):
                    candidate = str(item.get("summary") or item.get("text") or "").strip()
                    if candidate:
                        rec_texts.append(candidate)

        rec_texts = rec_texts[:3]
        while len(rec_texts) < 3:
            rec_texts.append("Coordinate department-level operations using current risk dashboard signals.")

        recommendations = [
            CopilotRecommendation(
                title=f"Operational Recommendation {idx + 1}",
                summary=text,
                priority=urgency,
                actions=[text],
            )
            for idx, text in enumerate(rec_texts)
        ]

        return CopilotPlan(
            mode="ai",
            urgency=urgency,
            rationale=rationale,
            recommendations=recommendations,
        )

    def _should_use_ai_mode(self) -> bool:
        """Determine whether AI mode should be attempted."""
        if self.has_ai_key is not None:
            return self.has_ai_key and self._load_openai_compatible_config() is not None
        return self._load_openai_compatible_config() is not None

    def _load_openai_compatible_config(self) -> Dict[str, str] | None:
        """Load OpenAI-compatible API settings from environment variables.

        Priority:
        1) Explicit OpenAI-compatible settings (`OPENAI_*`)
        2) Gemini key via OpenAI-compatible endpoint (`GEMINI_*`)
        """
        openai_api_key = self._read_config_value("OPENAI_API_KEY")
        openai_base_url = self._read_config_value("OPENAI_BASE_URL")
        openai_model = self._read_config_value("OPENAI_MODEL")

        if openai_api_key and openai_base_url and openai_model:
            return {
                "api_key": openai_api_key,
                "base_url": openai_base_url,
                "model": openai_model,
            }

        gemini_api_key = self._read_config_value("GEMINI_API_KEY")
        if gemini_api_key:
            return {
                "api_key": gemini_api_key,
                "base_url": self._read_config_value(
                    "GEMINI_OPENAI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                )
                or "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": self._read_config_value("GEMINI_MODEL", "gemini-2.0-flash")
                or "gemini-2.0-flash",
            }

        return None

    def _read_config_value(self, key: str, default: str = "") -> str:
        """Read config from env first, then Streamlit secrets."""
        env_value = os.getenv(key, "").strip()
        if env_value:
            return env_value

        if not (
            (Path.cwd() / ".streamlit" / "secrets.toml").is_file()
            or (Path.home() / ".streamlit" / "secrets.toml").is_file()
        ):
            return default

        try:
            import streamlit as st

            if key in st.secrets:
                return str(st.secrets[key]).strip()
        except Exception:
            pass

        return default

    def _build_llm_prompt(
        self,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> str:
        """Build concise prompt for structured operational recommendations."""
        return (
            "Given the city state and risk scores below, return JSON only with keys: "
            "urgency, recommendations, rationale.\n"
            "- urgency: one of Low, Medium, High\n"
            "- recommendations: exactly 3 concise operational recommendations\n"
            "- rationale: one short sentence\n\n"
            f"city_state: {json.dumps(city_state.to_dict(), ensure_ascii=False)}\n"
            f"flood_risk: {json.dumps(dict(flood_risk), ensure_ascii=False)}\n"
            f"heat_risk: {json.dumps(dict(heat_risk), ensure_ascii=False)}\n"
            f"traffic_risk: {json.dumps(dict(traffic_risk), ensure_ascii=False)}"
        )

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling fenced-code wrappers."""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse LLM JSON response.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON is not an object.")

        return parsed

    def _format_ai_fallback_reason(self, exc: Exception) -> str:
        """Produce a concise, user-friendly reason when AI mode fails."""
        if isinstance(exc, requests.HTTPError):
            response = exc.response
            status_code = response.status_code if response is not None else None
            provider_message = self._extract_provider_error_message(response)

            if status_code == 429:
                if provider_message:
                    return f"AI quota/rate limit reached (HTTP 429): {provider_message}."
                return "AI quota/rate limit reached (HTTP 429)."

            if status_code is not None:
                if provider_message:
                    return f"AI provider request failed (HTTP {status_code}): {provider_message}."
                return f"AI provider request failed (HTTP {status_code})."

        if isinstance(exc, requests.Timeout):
            return "AI provider request timed out."

        if isinstance(exc, requests.ConnectionError):
            return "AI provider connection failed."

        details = str(exc).strip()
        if details:
            return f"AI response error: {self._truncate_text(details)}."

        return "AI endpoint unavailable."

    def _extract_provider_error_message(self, response: requests.Response | None) -> str:
        """Extract and sanitize provider error details from an HTTP response."""
        if response is None:
            return ""

        try:
            payload = response.json()
        except ValueError:
            return ""

        error_payload = payload.get("error", {}) if isinstance(payload, dict) else {}
        if not isinstance(error_payload, dict):
            return ""

        status = str(error_payload.get("status", "")).strip()
        message = str(error_payload.get("message", "")).strip()

        if message:
            cleaned_message = " ".join(message.split())
            for marker in (
                " For more information",
                " To monitor your current usage",
                " Please retry in",
            ):
                if marker in cleaned_message:
                    cleaned_message = cleaned_message.split(marker, 1)[0].strip()
            message = self._truncate_text(cleaned_message)

        if status and message and status.lower() not in message.lower():
            return f"{status}: {message}"
        return message or status

    def _truncate_text(self, text: str, max_len: int = 220) -> str:
        """Truncate long provider diagnostics for readable UI output."""
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return f"{compact[: max_len - 1].rstrip()}…"

    def _normalize_urgency(self, urgency: str) -> str:
        """Normalize urgency label to expected values."""
        value = urgency.strip().lower()
        if value == "high":
            return "High"
        if value == "low":
            return "Low"
        return "Medium"

    def _domain_specific_recommendation(
        self,
        top_risk: str,
        urgency: str,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> CopilotRecommendation:
        """Create one focused recommendation for the dominant risk domain."""
        if top_risk == "flood":
            return CopilotRecommendation(
                title="Flood Mitigation Operations",
                summary=(
                    f"Prioritize low-lying corridors and river-adjacent zones "
                    f"(trend: {city_state.river_trend})."
                ),
                priority=urgency,
                actions=[
                    "Inspect and clear critical drainage inlets.",
                    f"Deploy pumps/barriers in vulnerable districts ({city_state.district_vulnerability}).",
                ],
            )

        if top_risk == "heat":
            return CopilotRecommendation(
                title="Heat Protection Operations",
                summary=(
                    "Protect high-exposure groups with cooling and hydration measures "
                    "across dense urban zones."
                ),
                priority=urgency,
                actions=[
                    "Extend cooling-center opening hours.",
                    "Increase welfare checks for elderly and high-risk residents.",
                ],
            )

        return CopilotRecommendation(
            title="Traffic Flow Stabilization",
            summary=(
                f"Reduce disruption on peak routes (congestion {city_state.traffic_congestion_index:.1f}%)."
            ),
            priority=urgency,
            actions=[
                "Adjust signal timing and incident response staffing.",
                "Issue dynamic rerouting guidance for major corridors.",
            ],
        )

    def _urgency_from_score(self, score: int) -> str:
        """Map aggregate risk score to urgency label."""
        if score >= 70:
            return "High"
        if score >= 35:
            return "Medium"
        return "Low"

    def generate_recommendations(
        self,
        city_state: CityState,
        flood_risk: Mapping[str, Any],
        heat_risk: Mapping[str, Any],
        traffic_risk: Mapping[str, Any],
    ) -> List[str]:
        """Backward-compatible helper returning recommendation summaries.

        Prefer using `generate_operational_recommendations` for structured output.
        """
        plan = self.generate_operational_recommendations(
            city_state=city_state,
            flood_risk=flood_risk,
            heat_risk=heat_risk,
            traffic_risk=traffic_risk,
        )
        return [item.summary for item in plan.recommendations]
