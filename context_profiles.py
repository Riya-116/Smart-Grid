
from __future__ import annotations

import math


def occupancy_profile(
    step: int,
    steps_per_day: int,
) -> bool:
    """Simple synthetic occupancy schedule.

    Occupied:
    06:00-09:00
    17:00-23:00
    """

    hour = (step % steps_per_day) / steps_per_day * 24.0

    morning = 6.0 <= hour < 9.0
    evening = 17.0 <= hour < 23.0

    return morning or evening


def outdoor_temperature_c(
    step: int,
    steps_per_day: int,
) -> float:
    """Synthetic outdoor temperature profile."""

    hour = (step % steps_per_day) / steps_per_day * 24.0

    return 28.0 + 5.0 * math.sin(
        2.0 * math.pi * (hour - 8.0) / 24.0
    )


def indoor_temperature_c(
    step: int,
    steps_per_day: int,
    outdoor_temp_c: float | None = None,
) -> float:
    """Simple indoor temperature approximation."""

    if outdoor_temp_c is None:
        outdoor_temp_c = outdoor_temperature_c(
            step,
            steps_per_day,
        )

    # Approximate indoor temperature as partially
    # insulated from outdoor temperature.
    return 24.0 + 0.35 * (outdoor_temp_c - 24.0)