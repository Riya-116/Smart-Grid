"""
metrics.py
------------------------------------------------------------------
NEW FILE -- computes the abstract's named evaluation metrics from an
episode's list of per-step info dicts (as returned by env.step()):
  - total electricity cost
  - renewable energy utilization (%)
  - battery usage efficiency (equivalent full cycles -- lower is
    "gentler" on the battery for the same task)
  - energy wastage
  - grid load distribution (peak / average / overload rate)
------------------------------------------------------------------
"""

import numpy as np


def compute_episode_metrics(info_history: list, final_battery_degradation: float) -> dict:
    total_cost = sum(step["cost"] for step in info_history)
    total_renewable = sum(step["renewable_kw"] for step in info_history)
    total_wasted = sum(step["wasted_renewable_kw"] for step in info_history)
    renewable_used = total_renewable - total_wasted
    renewable_utilization = renewable_used / total_renewable if total_renewable > 1e-6 else 1.0

    grid_loads = [step["net_grid_load_kw"] for step in info_history]
    overload_steps = sum(1 for step in info_history if step["overloaded"])

    return {
        "total_cost": round(float(total_cost), 3),
        "renewable_utilization_pct": round(float(renewable_utilization) * 100, 2),
        "total_energy_wasted_kwh": round(float(total_wasted), 3),
        "peak_grid_load_kw": round(float(max(grid_loads)), 3),
        "avg_grid_load_kw": round(float(np.mean(grid_loads)), 3),
        "overload_steps": overload_steps,
        "overload_rate_pct": round(100 * overload_steps / len(info_history), 2),
        "battery_equivalent_cycles": round(float(final_battery_degradation), 3),
    }
