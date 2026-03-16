"""Configuration loader for the Urban AI Copilot."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from utils.helpers import parse_supported_cities


DEFAULT_SUPPORTED_CITIES = "Berlin,Hamburg,Munich,Cologne,Frankfurt"


def _read_streamlit_secret(key: str) -> str | None:
    """Read from Streamlit secrets if available."""
    if not (
        (Path.cwd() / ".streamlit" / "secrets.toml").is_file()
        or (Path.home() / ".streamlit" / "secrets.toml").is_file()
    ):
        return None

    try:
        import streamlit as st
    except Exception:
        return None

    try:
        if key in st.secrets:
            value = st.secrets[key]
            return str(value).strip()
    except Exception:
        return None

    return None


def _read_config_value(key: str, default: str | None = None) -> str | None:
    """Read config value with env-first, Streamlit-secrets fallback."""
    env_value = os.getenv(key)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    secret_value = _read_streamlit_secret(key)
    if secret_value:
        return secret_value

    return default


def _clean_optional_token(value: str | None) -> str | None:
    """Normalize optional token values and ignore placeholders."""
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    placeholder_tokens = {
        "your_api_key",
        "your_key",
        "your_weather_api_key",
        "your_openai_key",
        "your_gemini_key",
        "your_tomtom_key",
        "replace_me",
        "changeme",
    }
    if cleaned.lower() in placeholder_tokens:
        return None

    return cleaned


def validate_environment(
    config: "AppConfig", requested_default_city: str
) -> Tuple[str, ...]:
    """Return human-friendly warnings for missing or optional env values.

    This validation is intentionally non-blocking so the app can run in demo mode.
    """
    warnings: List[str] = []

    if not config.weather_api_key:
        warnings.append(
            "WEATHER_API_KEY is not set. This is fine for Open-Meteo, but other "
            "weather providers may require a key."
        )

    if not config.has_ai_provider_key:
        warnings.append(
            "No AI API key found (OPENAI_API_KEY or GEMINI_API_KEY). "
            "Copilot features will run in demo mode."
        )

    if not config.tomtom_api_key:
        warnings.append(
            "TOMTOM_API_KEY is not set. Live traffic will fall back to proxy rules."
        )

    if requested_default_city not in config.supported_cities:
        warnings.append(
            "DEFAULT_CITY is not in SUPPORTED_CITIES. Falling back to first supported city."
        )

    return tuple(warnings)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment variables."""

    weather_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    tomtom_api_key: str | None
    weather_api_base_url: str
    tomtom_api_base_url: str
    pegelonline_api_base_url: str
    default_city: str
    supported_cities: Tuple[str, ...]
    demo_mode: bool
    config_warnings: Tuple[str, ...]

    @property
    def has_ai_provider_key(self) -> bool:
        """Whether any supported AI provider key is configured."""
        return bool(self.openai_api_key or self.gemini_api_key)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from env vars with Streamlit secrets fallback."""
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
        load_dotenv()

        supported = parse_supported_cities(
            _read_config_value("SUPPORTED_CITIES", DEFAULT_SUPPORTED_CITIES)
            or DEFAULT_SUPPORTED_CITIES
        )
        if not supported:
            supported = parse_supported_cities(DEFAULT_SUPPORTED_CITIES)

        requested_default_city = _read_config_value("DEFAULT_CITY", "Berlin") or "Berlin"
        default_city = (
            requested_default_city
            if requested_default_city in supported
            else supported[0]
        )

        config = cls(
            weather_api_key=_clean_optional_token(_read_config_value("WEATHER_API_KEY")),
            openai_api_key=_clean_optional_token(_read_config_value("OPENAI_API_KEY")),
            gemini_api_key=_clean_optional_token(_read_config_value("GEMINI_API_KEY")),
            tomtom_api_key=_clean_optional_token(_read_config_value("TOMTOM_API_KEY")),
            weather_api_base_url=(
                _read_config_value("WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1")
                or "https://api.open-meteo.com/v1"
            ),
            tomtom_api_base_url=(
                _read_config_value(
                    "TOMTOM_API_BASE_URL",
                    "https://api.tomtom.com/traffic/services/4",
                )
                or "https://api.tomtom.com/traffic/services/4"
            ),
            pegelonline_api_base_url=(
                _read_config_value(
                    "PEGELONLINE_API_BASE_URL",
                    "https://www.pegelonline.wsv.de/webservices/rest-api/v2",
                )
                or "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
            ),
            default_city=default_city,
            supported_cities=tuple(supported),
            demo_mode=False,
            config_warnings=(),
        )

        warnings = validate_environment(config, requested_default_city)
        return cls(
            weather_api_key=config.weather_api_key,
            openai_api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
            tomtom_api_key=config.tomtom_api_key,
            weather_api_base_url=config.weather_api_base_url,
            tomtom_api_base_url=config.tomtom_api_base_url,
            pegelonline_api_base_url=config.pegelonline_api_base_url,
            default_city=config.default_city,
            supported_cities=config.supported_cities,
            demo_mode=not config.has_ai_provider_key,
            config_warnings=warnings,
        )
