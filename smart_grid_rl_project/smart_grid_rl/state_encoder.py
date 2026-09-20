"""Converts raw environment component readings into a normalized RL state
vector.

STATUS: Original file, reconstructed cleanly. No logic bugs found.
"""
from __future__ import annotations

import numpy as np

from smart_grid_rl.components import Appliance, ApplianceState, BatteryStorage, GridModel


def encode_state(
    appliances: list[Appliance],
    battery: BatteryStorage,
    renewable_kw: float,
    grid: GridModel,
    step_of_day: int,
    steps_per_day: int,
    renewable_norm_kw: float = 10.0,
) -> np.ndarray:
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")

    time_frac = (step_of_day % steps_per_day) / steps_per_day
    soc_frac = battery.soc_fraction
    renewable_frac = float(np.clip(renewable_kw / renewable_norm_kw, 0.0, 1.0))

    price = grid.price(step_of_day, steps_per_day)
    price_min, price_max = grid.price_bounds()
    price_span = max(price_max - price_min, 1e-6)
    price_frac = float(np.clip((price - price_min) / price_span, 0.0, 1.0))

    appliance_flags = [1.0 if a.state == ApplianceState.ON else 0.0 for a in appliances]

    state = np.array(
        [time_frac, soc_frac, renewable_frac, price_frac, *appliance_flags],
        dtype=np.float32,
    )
    return state


def state_vector_length(num_appliances: int) -> int:
    return 4 + num_appliances
