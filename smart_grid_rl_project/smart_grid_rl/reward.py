"""
reward.py
------------------------------------------------------------------
NEW FILE -- completes the reward function that env.py stubbed to 0.0.

Implements the four objectives from the project abstract:
  1) minimize electricity cost
  2) maximize renewable energy utilization
  3) reduce energy wastage
  4) prevent grid overload

All inputs are per-step quantities in kW (power) except cost, which is
already in currency units for the step (price * energy_kWh).
------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass
class RewardWeights:
    cost: float = 1.0
    renewable_utilization: float = 0.5
    wastage: float = 0.4
    overload: float = 2.0
    deferred_task_penalty: float = 3.0  # applied once, at episode end, per
    # deferrable appliance that never got to run


def compute_reward(
    net_grid_load_kw: float,
    price_per_kwh: float,
    dt_hours: float,
    renewable_kw: float,
    wasted_renewable_kw: float,
    overloaded: bool,
    overload_amount_kw: float,
    weights: RewardWeights | None = None,
) -> tuple[float, dict]:
    """Per-step reward. Returns (reward, info) where info holds the raw
    components for metrics.py / debugging."""
    w = weights or RewardWeights()

    cost = net_grid_load_kw * price_per_kwh * dt_hours

    renewable_available = renewable_kw * dt_hours
    renewable_used = max(renewable_available - wasted_renewable_kw * dt_hours, 0.0)
    renewable_ratio = (
        renewable_used / renewable_available if renewable_available > 1e-6 else 1.0
    )

    waste_kwh = wasted_renewable_kw * dt_hours
    overload_penalty_term = (overload_amount_kw ** 2) if overloaded else 0.0

    reward = 0.0
    reward -= w.cost * cost
    reward += w.renewable_utilization * renewable_ratio
    reward -= w.wastage * waste_kwh
    reward -= w.overload * overload_penalty_term

    info = {
        "cost": cost,
        "renewable_ratio": renewable_ratio,
        "waste_kwh": waste_kwh,
        "overload_penalty": overload_penalty_term,
    }
    return reward, info


def deferred_task_penalty(num_never_run: int, weights: RewardWeights | None = None) -> float:
    """Applied once at episode end for each deferrable appliance that was
    never actually run, so the agent can't just defer forever to dodge
    cost/overload penalties."""
    w = weights or RewardWeights()
    return -w.deferred_task_penalty * num_never_run
