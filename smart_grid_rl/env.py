"""Gymnasium-compatible virtual smart home environment.

STATUS: CORRECTED. Bugs fixed vs. the original (see inline "BUG FIX"
comments):

  1. surplus/deficit calculation used max() where it needed min() + a
     zero-floor. As originally written, choosing "charge" or "discharge"
     forced the battery to move power at its absolute max rate on every
     step, regardless of whether any solar surplus/demand deficit
     actually existed -- e.g. charging at full rated power even during
     a demand deficit, since max(negative_number, max_charge_kw) always
     equals max_charge_kw. Fixed to floor at 0 and cap with min().

  2. Reward was stubbed to 0.0. Now computed via reward.py, matching the
     abstract's four objectives (cost, renewable utilization, wastage,
     overload).

  3. Wasted-renewable energy was untracked (silently dropped in the
     surplus case). Now computed explicitly and passed into both the
     reward function and step info for metrics.py.

  4. Deferred appliances that never run were never penalized, so an
     agent could dodge every other penalty by permanently deferring.
     Added an end-of-episode penalty (reward.py: deferred_task_penalty).

Everything else (action/observation space shapes, component wiring) is
unchanged from the original architecture.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smart_grid_rl.components import Appliance, ApplianceState, BatteryStorage, GridModel, RenewableSource
from smart_grid_rl.data_integration import DatasetBundle
from smart_grid_rl.reward import RewardWeights, compute_reward, deferred_task_penalty
from smart_grid_rl.state_encoder import encode_state, state_vector_length

# Action encoding per appliance: 0 = OFF, 1 = ON, 2 = DEFER (if deferrable)
# Battery action is appended last: 0 = idle, 1 = charge, 2 = discharge
NUM_APPLIANCE_ACTIONS = 3
NUM_BATTERY_ACTIONS = 3


class VirtualSmartHomeEnv(gym.Env):
    """Simulates a smart home with appliances, battery storage, renewables,
    and grid pricing."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        steps_per_day: int = 96,
        dt_hours: float = 0.25,
        seed: int | None = None,
        reward_weights: Optional[RewardWeights] = None,
        # --- real dataset integration (all optional; None = synthetic) ---
        ukdale_house_dir: Optional[str] = None,
        ukdale_mains_channel: int = 1,
        ukdale_appliance_names: Optional[list] = None,
        smart_meter_tariff_csv: Optional[str] = None,
        kaggle_solar_csv: Optional[str] = None,
        nrel_solar_csv: Optional[str] = None,
        use_nrel_for_solar: bool = False,
        household_solar_capacity_kw: float = 6.0,
    ) -> None:
        super().__init__()
        if steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

        self.steps_per_day = steps_per_day
        self.dt_hours = dt_hours
        self._episode_seed = seed
        self.reward_weights = reward_weights or RewardWeights()

        self.appliances: list[Appliance] = [
            Appliance(name="HVAC", rated_power_kw=2.5, deferrable=False, min_run_steps=2),
            Appliance(name="Lights", rated_power_kw=0.3, deferrable=False),
            Appliance(name="Washer", rated_power_kw=1.2, deferrable=True, min_run_steps=3),
            Appliance(name="EV", rated_power_kw=3.5, deferrable=True, min_run_steps=4),
        ]

        # Try loading real datasets; anything not supplied/parseable falls
        # back to the synthetic defaults inside RenewableSource/GridModel.
        dataset_bundle = DatasetBundle.load(
            steps_per_day=steps_per_day,
            ukdale_house_dir=ukdale_house_dir,
            ukdale_mains_channel=ukdale_mains_channel,
            ukdale_appliance_names=ukdale_appliance_names,
            smart_meter_tariff_csv=smart_meter_tariff_csv,
            kaggle_solar_csv=kaggle_solar_csv,
            nrel_solar_csv=nrel_solar_csv,
            use_nrel_for_solar=use_nrel_for_solar,
            household_solar_capacity_kw=household_solar_capacity_kw,
        )
        for appliance in self.appliances:
            if appliance.name in dataset_bundle.appliance_overrides:
                rated_kw, _duration_h = dataset_bundle.appliance_overrides[appliance.name]
                if rated_kw > 0:
                    appliance.rated_power_kw = rated_kw

        self.battery = BatteryStorage(capacity_kwh=13.5, max_charge_kw=5.0, max_discharge_kw=5.0)
        self.renewable = RenewableSource(
            solar_capacity_kw=household_solar_capacity_kw, wind_capacity_kw=2.0,
            seed=seed or 42,
            real_profile_kw=(dataset_bundle.renewable_profile_kw.tolist()
                              if dataset_bundle.renewable_profile_kw is not None else None),
        )
        self.grid = GridModel(
            real_price_profile=(dataset_bundle.price_profile.tolist()
                                 if dataset_bundle.price_profile is not None else None)
        )
        self.current_step = 0

        n_appliances = len(self.appliances)
        self.action_space = spaces.MultiDiscrete(
            [NUM_APPLIANCE_ACTIONS] * n_appliances + [NUM_BATTERY_ACTIONS]
        )
        obs_len = state_vector_length(n_appliances)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        for appliance in self.appliances:
            appliance.turn_off()
        self.battery.soc_kwh = 0.5 * self.battery.capacity_kwh
        self.battery.cumulative_throughput_kwh = 0.0

        renewable_kw = self.renewable.generation_kw(self.current_step, self.steps_per_day)
        state = encode_state(
            self.appliances, self.battery, renewable_kw, self.grid,
            self.current_step, self.steps_per_day,
        )
        return state, {"renewable_kw": renewable_kw}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if len(action) != len(self.appliances) + 1:
            raise ValueError(
                f"Expected action of length {len(self.appliances) + 1}, got {len(action)}"
            )

        # Apply appliance actions.
        for appliance, act in zip(self.appliances, action[:-1]):
            if act == 0:
                appliance.turn_off()
            elif act == 1:
                appliance.turn_on()
            elif act == 2 and appliance.deferrable:
                appliance.defer()
            # Invalid combos (e.g. DEFER on non-deferrable) silently no-op;
            # the agent learns to avoid them via reward shaping.

        appliance_load_kw = sum(a.tick() for a in self.appliances)
        renewable_kw = self.renewable.generation_kw(self.current_step, self.steps_per_day)

        # Apply battery action.
        battery_act = int(action[-1])
        battery_power_kw = 0.0
        wasted_renewable_kw = 0.0
        available_surplus_kw = max(renewable_kw - appliance_load_kw, 0.0)

        if battery_act == 1:  # charge
            # BUG FIX: was `max(renewable_kw - appliance_load_kw, max_charge_kw)`,
            # which always forced charging at the full rated rate regardless
            # of whether surplus solar existed. Now floors at 0 and caps at
            # the battery's max charge rate with min().
            charge_request_kw = min(available_surplus_kw, self.battery.max_charge_kw)
            battery_power_kw = self.battery.charge(charge_request_kw, self.dt_hours)
            wasted_renewable_kw = max(available_surplus_kw - battery_power_kw, 0.0)
        elif battery_act == 2:  # discharge
            # BUG FIX: was `max(appliance_load_kw - renewable_kw, max_discharge_kw)`,
            # same issue in the other direction.
            deficit_kw = max(appliance_load_kw - renewable_kw, 0.0)
            discharge_request_kw = min(deficit_kw, self.battery.max_discharge_kw)
            battery_power_kw = self.battery.discharge(discharge_request_kw, self.dt_hours)
        else:  # idle -> any solar surplus this step goes unused
            wasted_renewable_kw = available_surplus_kw

        net_grid_load_kw = appliance_load_kw - renewable_kw + (
            battery_power_kw if battery_act == 1 else
            -battery_power_kw if battery_act == 2 else 0.0
        )
        net_grid_load_kw = max(net_grid_load_kw, 0.0)

        overloaded = self.grid.is_overloaded(net_grid_load_kw)
        overload_amount_kw = max(net_grid_load_kw - self.grid.overload_threshold_kw, 0.0)
        price = self.grid.price(self.current_step, self.steps_per_day)

        reward, reward_info = compute_reward(
            net_grid_load_kw=net_grid_load_kw,
            price_per_kwh=price,
            dt_hours=self.dt_hours,
            renewable_kw=renewable_kw,
            wasted_renewable_kw=wasted_renewable_kw,
            overloaded=overloaded,
            overload_amount_kw=overload_amount_kw,
            weights=self.reward_weights,
        )

        self.current_step += 1
        done = self.current_step >= self.steps_per_day

        if done:
            never_run = sum(
                1 for a in self.appliances
                if a.deferrable and a.state != ApplianceState.ON and a.steps_running == 0
            )
            reward += deferred_task_penalty(never_run, self.reward_weights)
            reward_info["deferred_never_run"] = never_run

        state = encode_state(
            self.appliances, self.battery, renewable_kw, self.grid,
            self.current_step, self.steps_per_day,
        )

        info = {
            "appliance_load_kw": appliance_load_kw,
            "renewable_kw": renewable_kw,
            "net_grid_load_kw": net_grid_load_kw,
            "overloaded": overloaded,
            "battery_soc_fraction": self.battery.soc_fraction,
            "price_per_kwh": price,
            "wasted_renewable_kw": wasted_renewable_kw,
            **reward_info,
        }
        return state, reward, done, False, info
