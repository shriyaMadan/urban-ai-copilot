"""Scenario simulation for what-if analysis."""
from __future__ import annotations

from dataclasses import dataclass, replace

from core.data_model import CityState


@dataclass(frozen=True)
class ScenarioAdjustments:
    """Deterministic scenario inputs used for what-if simulations.

    Attributes:
        rainfall_multiplier: Multiplies near-term rainfall signals.
        temperature_delta: Adds/subtracts degrees Celsius from temperature signals.
        force_rush_hour: If True, forces rush-hour mode in the simulated state.
    """

    rainfall_multiplier: float = 1.0
    temperature_delta: float = 0.0
    force_rush_hour: bool = False


def simulate_city_state(
    city_state: CityState,
    adjustments: ScenarioAdjustments,
) -> CityState:
    """Return a new simulated CityState without mutating the original.

    The function is intentionally deterministic and side-effect free so it is
    straightforward to test with fixed inputs.
    """
    rainfall_multiplier = max(0.0, adjustments.rainfall_multiplier)

    return replace(
        city_state,
        rainfall_mm_next_3h=round(city_state.rainfall_mm_next_3h * rainfall_multiplier, 2),
        rainfall_mm_next_6h=round(city_state.rainfall_mm_next_6h * rainfall_multiplier, 2),
        temperature_c=round(city_state.temperature_c + adjustments.temperature_delta, 2),
        feels_like_c=round(city_state.feels_like_c + adjustments.temperature_delta, 2),
        is_rush_hour=(True if adjustments.force_rush_hour else city_state.is_rush_hour),
    )


class ScenarioEngine:
    """Apply user-defined scenario changes to a city-state snapshot."""

    def apply(
        self,
        city_state: CityState,
        rainfall_multiplier: float = 1.0,
        temperature_delta: float = 0.0,
        force_rush_hour: bool = False,
    ) -> CityState:
        """Return a modified CityState for what-if simulation.

        This method preserves the original CityState and returns a new object.
        """
        adjustments = ScenarioAdjustments(
            rainfall_multiplier=rainfall_multiplier,
            temperature_delta=temperature_delta,
            force_rush_hour=force_rush_hour,
        )
        return simulate_city_state(city_state, adjustments)
