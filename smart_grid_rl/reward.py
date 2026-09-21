
"""
Reward calculation for the smart-grid environment.

Supports:
1. Electricity cost minimization
2. Renewable energy utilization
3. Renewable energy wastage reduction
4. Grid overload prevention
5. Appliance comfort penalties
6. Appliance wasted-energy penalties
"""

from dataclasses import dataclass


@dataclass
class RewardWeights:
    # Existing reward components
    cost: float = 1.0
    renewable_utilization: float = 0.5
    wastage: float = 0.4
    overload: float = 2.0
    deferred_task_penalty: float = 3.0

    # New appliance-related reward components
    comfort: float = 1.0
    appliance_wastage: float = 0.4


def compute_reward(
    net_grid_load_kw: float,
    price_per_kwh: float,
    dt_hours: float,
    renewable_kw: float,
    wasted_renewable_kw: float,
    overloaded: bool,
    overload_amount_kw: float,
    weights: RewardWeights | None = None,
    comfort_penalty: float = 0.0,
    appliance_wasted_kwh: float = 0.0,
) -> tuple[float, dict]:
    """
    Calculate the reward for one environment step.

    The two new parameters have default values of zero.
    Therefore, existing calls continue to work without changes.

    Returns:
        reward: Total scalar reward.
        info: Individual reward components.
    """

    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive")

    w = weights or RewardWeights()

    # ---------------------------------------------------------
    # 1. Electricity cost
    # ---------------------------------------------------------
    cost = (
        net_grid_load_kw
        * price_per_kwh
        * dt_hours
    )

    # ---------------------------------------------------------
    # 2. Renewable energy utilization
    # ---------------------------------------------------------
    renewable_available_kwh = (
        renewable_kw * dt_hours
    )

    renewable_used_kwh = max(
        renewable_available_kwh
        - wasted_renewable_kw * dt_hours,
        0.0,
    )

    if renewable_available_kwh > 1e-6:
        renewable_ratio = (
            renewable_used_kwh
            / renewable_available_kwh
        )
    else:
        renewable_ratio = 1.0

    # ---------------------------------------------------------
    # 3. Renewable energy wastage
    # ---------------------------------------------------------
    waste_kwh = (
        wasted_renewable_kw * dt_hours
    )

    # ---------------------------------------------------------
    # 4. Grid overload
    # ---------------------------------------------------------
    overload_penalty_term = (
        overload_amount_kw ** 2
        if overloaded
        else 0.0
    )

    # ---------------------------------------------------------
    # 5. Validate new appliance penalties
    # ---------------------------------------------------------
    comfort_penalty = max(
        float(comfort_penalty),
        0.0,
    )

    appliance_wasted_kwh = max(
        float(appliance_wasted_kwh),
        0.0,
    )

    # ---------------------------------------------------------
    # 6. Calculate total reward
    # ---------------------------------------------------------
    reward = 0.0

    # Cost penalty
    reward -= w.cost * cost

    # Renewable utilization reward
    reward += (
        w.renewable_utilization
        * renewable_ratio
    )

    # Existing renewable wastage penalty
    reward -= w.wastage * waste_kwh

    # Grid overload penalty
    reward -= (
        w.overload
        * overload_penalty_term
    )

    # New comfort penalty
    reward -= (
        w.comfort
        * comfort_penalty
    )

    # New appliance wastage penalty
    reward -= (
        w.appliance_wastage
        * appliance_wasted_kwh
    )

    info = {
        "cost": cost,
        "renewable_ratio": renewable_ratio,
        "waste_kwh": waste_kwh,
        "overload_penalty": overload_penalty_term,
        "comfort_penalty": comfort_penalty,
        "appliance_wasted_kwh": appliance_wasted_kwh,
    }

    return reward, info


def deferred_task_penalty(
    num_never_run: int,
    weights: RewardWeights | None = None,
) -> float:
    """
    Apply a penalty for deferrable appliances that never run.

    This is applied once at the end of an episode.
    """

    if num_never_run < 0:
        raise ValueError(
            "num_never_run cannot be negative"
        )

    w = weights or RewardWeights()

    return (
        -w.deferred_task_penalty
        * num_never_run
    )