"""
baseline_controller.py
------------------------------------------------------------------
NEW FILE -- rule-based baseline for comparison against the RL agent,
adapted to this environment's per-appliance MultiDiscrete action space.

Rules:
  - Non-deferrable appliances (HVAC, Lights) always run (as if on a
    fixed schedule / thermostat) -- a rule-based system has no way to
    intelligently decide these, which is exactly the gap RL is meant
    to close.
  - Deferrable appliances (Washer, EV) run once solar output clears a
    threshold, or are forced to run by a deadline step if solar never
    gets that high; otherwise they're deferred.
  - Battery charges on solar surplus, discharges during peak price,
    else idles.
------------------------------------------------------------------
"""

import numpy as np

from smart_grid_rl.components import ApplianceState

SOLAR_RUN_THRESHOLD_KW = 2.0
DEADLINE_FRACTION_OF_DAY = 0.85  # force deferrable appliances to run by 85% through the day


class RuleBasedController:
    def act(self, env) -> np.ndarray:
        """env: a VirtualSmartHomeEnv instance (reads its current state
        directly, since the baseline needs raw values, not the
        normalized/binned observation the RL agent sees)."""
        renewable_kw = env.renewable.generation_kw(env.current_step, env.steps_per_day)
        price = env.grid.price(env.current_step, env.steps_per_day)
        deadline_step = int(DEADLINE_FRACTION_OF_DAY * env.steps_per_day)

        actions = []
        for appliance in env.appliances:
            if not appliance.deferrable:
                actions.append(1)  # always ON
                continue

            already_ran = appliance.steps_running > 0 and appliance.state == ApplianceState.ON
            if already_ran:
                actions.append(1)  # keep running until its min_run_steps naturally finishes
            elif renewable_kw >= SOLAR_RUN_THRESHOLD_KW or env.current_step >= deadline_step:
                actions.append(1)  # run now: solar is plentiful, or deadline pressure
            else:
                actions.append(2)  # defer

        appliance_load_estimate = sum(
            a.rated_power_kw for a, act in zip(env.appliances, actions) if act == 1
        )
        if renewable_kw > appliance_load_estimate and env.battery.soc_fraction < env.battery.max_soc_fraction:
            battery_action = 1  # charge with surplus solar
        elif price >= (env.grid.base_price_per_kwh + env.grid.peak_price_per_kwh) / 2 \
                and env.battery.soc_fraction > env.battery.min_soc_fraction:
            battery_action = 2  # discharge to avoid buying at high price
        else:
            battery_action = 0  # idle

        actions.append(battery_action)
        return np.array(actions)
