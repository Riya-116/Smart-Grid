
"""Gymnasium-compatible virtual smart home environment.

Supports:
1. Original appliance architecture.
2. Enhanced appliance models:
   - Thermostatic appliance
   - On-demand appliance
   - Shiftable appliance
   - Essential appliance

The original mode is preserved for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smart_grid_rl.components import (
    Appliance,
    ApplianceState,
    BatteryStorage,
    GridModel,
    RenewableSource,
)

from smart_grid_rl.data_integration import DatasetBundle

from smart_grid_rl.reward import (
    RewardWeights,
    compute_reward,
    deferred_task_penalty,
)

from smart_grid_rl.state_encoder import (
    encode_state,
    state_vector_length,
)

from smart_grid_rl.appliance_models import (
    ApplianceContext,
    ThermostaticAppliance,
    ShiftableAppliance,
    OnDemandAppliance,
    EssentialAppliance,
)

from smart_grid_rl.appliance_manager import ApplianceManager


# Original appliance actions
# 0 = OFF
# 1 = ON
# 2 = DEFER
NUM_APPLIANCE_ACTIONS = 3

# Battery actions
# 0 = IDLE
# 1 = CHARGE
# 2 = DISCHARGE
NUM_BATTERY_ACTIONS = 3


class VirtualSmartHomeEnv(gym.Env):
    """Simulates a smart home with appliances, battery storage,
    renewable energy and grid pricing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        steps_per_day: int = 96,
        dt_hours: float = 0.25,
        seed: int | None = None,
        reward_weights: Optional[RewardWeights] = None,

        # Enhanced appliance mode
        use_enhanced_appliances: bool = False,

        # Dataset integration
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
        self.use_enhanced_appliances = use_enhanced_appliances

        # =========================================================
        # 1. CREATE APPLIANCES
        # =========================================================

        if self.use_enhanced_appliances:

            self.appliances = []

            self.enhanced_appliances = [
                ThermostaticAppliance(
                    name="HVAC",
                    rated_power_kw=2.5,
                    comfort_min_c=22.0,
                    comfort_max_c=26.0,
                ),

                OnDemandAppliance(
                    name="Lights",
                    rated_power_kw=0.3,
                ),

                ShiftableAppliance(
                    name="Washer",
                    rated_power_kw=1.2,
                    duration_steps=3,
                    deadline_step=24,
                ),

                ShiftableAppliance(
                    name="EV",
                    rated_power_kw=3.5,
                    duration_steps=4,
                    deadline_step=48,
                ),

                EssentialAppliance(
                    name="Essential Load",
                    rated_power_kw=0.2,
                ),
            ]

            self.appliance_manager = ApplianceManager(
                self.enhanced_appliances,
                dt_hours=self.dt_hours,
            )

        else:

            self.appliances = [
                Appliance(
                    name="HVAC",
                    rated_power_kw=2.5,
                    deferrable=False,
                    min_run_steps=2,
                ),

                Appliance(
                    name="Lights",
                    rated_power_kw=0.3,
                    deferrable=False,
                ),

                Appliance(
                    name="Washer",
                    rated_power_kw=1.2,
                    deferrable=True,
                    min_run_steps=3,
                ),

                Appliance(
                    name="EV",
                    rated_power_kw=3.5,
                    deferrable=True,
                    min_run_steps=4,
                ),
            ]

            self.enhanced_appliances = []
            self.appliance_manager = None

        # =========================================================
        # 2. LOAD DATASETS
        # =========================================================

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

        if self.use_enhanced_appliances:
            appliances_for_override = self.enhanced_appliances
        else:
            appliances_for_override = self.appliances

        for appliance in appliances_for_override:

            if appliance.name in dataset_bundle.appliance_overrides:

                rated_kw, _duration_h = (
                    dataset_bundle.appliance_overrides[
                        appliance.name
                    ]
                )

                if rated_kw > 0:
                    appliance.rated_power_kw = rated_kw

        # =========================================================
        # 3. BATTERY
        # =========================================================

        self.battery = BatteryStorage(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
        )

        # =========================================================
        # 4. RENEWABLE ENERGY
        # =========================================================

        self.renewable = RenewableSource(
            solar_capacity_kw=household_solar_capacity_kw,
            wind_capacity_kw=2.0,
            seed=seed if seed is not None else 42,
            real_profile_kw=(
                dataset_bundle.renewable_profile_kw.tolist()
                if dataset_bundle.renewable_profile_kw is not None
                else None
            ),
        )

        # =========================================================
        # 5. GRID
        # =========================================================

        self.grid = GridModel(
            real_price_profile=(
                dataset_bundle.price_profile.tolist()
                if dataset_bundle.price_profile is not None
                else None
            )
        )

        self.current_step = 0

        # =========================================================
        # 6. ACTION SPACE
        # =========================================================

        if self.use_enhanced_appliances:

            appliance_action_sizes = list(
                self.appliance_manager.action_sizes
            )

            self.action_space = spaces.MultiDiscrete(
                appliance_action_sizes + [NUM_BATTERY_ACTIONS]
            )

        else:

            number_of_appliances = len(self.appliances)

            self.action_space = spaces.MultiDiscrete(
                [NUM_APPLIANCE_ACTIONS] * number_of_appliances
                + [NUM_BATTERY_ACTIONS]
            )

        # =========================================================
        # 7. OBSERVATION SPACE
        # =========================================================

        if self.use_enhanced_appliances:

            observation_length = (
                self._enhanced_observation_length()
            )

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(observation_length,),
                dtype=np.float32,
            )

        else:

            number_of_appliances = len(self.appliances)

            observation_length = state_vector_length(
                number_of_appliances
            )

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(observation_length,),
                dtype=np.float32,
            )

    # =============================================================
    # ENHANCED APPLIANCE HELPERS
    # =============================================================

    def _enhanced_observation_length(self) -> int:
        """Return the observation vector length.

        Global observations:
        1. Time fraction
        2. Battery state of charge
        3. Renewable energy fraction
        4. Electricity price fraction

        Appliance observations are generated by each appliance's
        obs() method.
        """

        dummy_context = ApplianceContext(
            step=0,
            day=0,
            outdoor_temp_c=30.0,
            indoor_temp_c=24.0,
            occupancy=True,
            price_per_kwh=0.2,
            renewable_kw=0.0,
        )

        appliance_observation_length = sum(
            len(appliance.obs(dummy_context))
            for appliance in self.enhanced_appliances
        )

        return 4 + appliance_observation_length

    def _normalise_value(
        self,
        value: float,
        maximum: float = 1.0,
    ) -> float:
        """Convert a value into the range 0 to 1."""

        if maximum <= 0:
            return 0.0

        return float(
            np.clip(value / maximum, 0.0, 1.0)
        )

    def _enhanced_state(
        self,
        renewable_kw: float,
    ) -> np.ndarray:
        """Create the enhanced observation vector."""

        time_fraction = self._normalise_value(
            self.current_step,
            self.steps_per_day,
        )

        battery_soc_fraction = float(
            np.clip(
                self.battery.soc_fraction,
                0.0,
                1.0,
            )
        )

        renewable_fraction = self._normalise_value(
            renewable_kw,
            10.0,
        )

        price = self.grid.price(
            min(
                self.current_step,
                self.steps_per_day - 1,
            ),
            self.steps_per_day,
        )

        price_fraction = self._normalise_value(
            price,
            1.0,
        )

        observation = [
            time_fraction,
            battery_soc_fraction,
            renewable_fraction,
            price_fraction,
        ]

        context = self._create_appliance_context(
            renewable_kw=renewable_kw,
            price=price,
        )

        for appliance in self.enhanced_appliances:

            appliance_observation = appliance.obs(context)

            observation.extend(
                float(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                )
                for value in appliance_observation
            )

        expected_length = self._enhanced_observation_length()

        if len(observation) != expected_length:
            raise RuntimeError(
                "Enhanced observation length mismatch: "
                f"expected {expected_length}, "
                f"got {len(observation)}"
            )

        return np.asarray(
            observation,
            dtype=np.float32,
        )

    def _create_appliance_context(
        self,
        renewable_kw: float,
        price: float,
    ) -> ApplianceContext:
        """Create the context passed to enhanced appliances."""

        hour = (
            self.current_step * self.dt_hours
        ) % 24.0

        # Default occupancy profile.
        # Occupancy is true between 01:00 and 23:00.
        occupied = 1.0 <= hour <= 23.0

        # Default temperature values.
        # These can later be replaced with real data.
        outdoor_temperature = 30.0
        indoor_temperature = 24.0

        return ApplianceContext(
            step=self.current_step,
            day=0,
            outdoor_temp_c=outdoor_temperature,
            indoor_temp_c=indoor_temperature,
            occupancy=bool(occupied),
            price_per_kwh=price,
            renewable_kw=renewable_kw,
        )

    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:

        super().reset(seed=seed)

        self.current_step = 0

        # Reset battery
        self.battery.soc_kwh = (
            0.5 * self.battery.capacity_kwh
        )

        self.battery.cumulative_throughput_kwh = 0.0

        if self.use_enhanced_appliances:

            self.appliance_manager.reset()

        else:

            for appliance in self.appliances:
                appliance.turn_off()

        renewable_kw = self.renewable.generation_kw(
            self.current_step,
            self.steps_per_day,
        )

        if self.use_enhanced_appliances:

            state = self._enhanced_state(
                renewable_kw
            )

        else:

            state = encode_state(
                self.appliances,
                self.battery,
                renewable_kw,
                self.grid,
                self.current_step,
                self.steps_per_day,
            )

        return state, {
            "renewable_kw": renewable_kw
        }

    # =============================================================
    # ORIGINAL MODE STEP
    # =============================================================

    def _step_original(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:

        expected_length = len(self.appliances) + 1

        if len(action) != expected_length:

            raise ValueError(
                f"Expected action of length "
                f"{expected_length}, "
                f"got {len(action)}"
            )

        # Apply appliance actions
        for appliance, act in zip(
            self.appliances,
            action[:-1],
        ):

            if act == 0:

                appliance.turn_off()

            elif act == 1:

                appliance.turn_on()

            elif act == 2 and appliance.deferrable:

                appliance.defer()

        appliance_load_kw = sum(
            appliance.tick()
            for appliance in self.appliances
        )

        renewable_kw = self.renewable.generation_kw(
            self.current_step,
            self.steps_per_day,
        )

        return self._calculate_energy_and_reward(
            action=action,
            appliance_load_kw=appliance_load_kw,
            renewable_kw=renewable_kw,
            comfort_penalty=0.0,
            appliance_wasted_kwh=0.0,
            enhanced=False,
        )

    # =============================================================
    # ENHANCED MODE STEP
    # =============================================================

    def _step_enhanced(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:

        expected_action_length = (
            len(self.enhanced_appliances) + 1
        )

        if len(action) != expected_action_length:

            raise ValueError(
                f"Expected action of length "
                f"{expected_action_length}, "
                f"got {len(action)}"
            )

        renewable_kw = self.renewable.generation_kw(
            self.current_step,
            self.steps_per_day,
        )

        price = self.grid.price(
            self.current_step,
            self.steps_per_day,
        )

        context = self._create_appliance_context(
            renewable_kw=renewable_kw,
            price=price,
        )

        appliance_actions = [
            int(value)
            for value in action[:-1]
        ]

        # Apply actions through the manager.
        # The context is required by enhanced appliances.
        self.appliance_manager.apply_actions(
            appliance_actions,
            context,
        )

        manager_result = self.appliance_manager.step(
            context
        )

        # ApplianceStepResult is a dataclass.
        # Access its attributes instead of dictionary keys.
        appliance_load_kw = float(
            manager_result.load_kw
        )

        appliance_wasted_kwh = float(
            manager_result.wasted_kwh
        )

        comfort_penalty = float(
            manager_result.comfort_penalty
        )

        return self._calculate_energy_and_reward(
            action=action,
            appliance_load_kw=appliance_load_kw,
            renewable_kw=renewable_kw,
            comfort_penalty=comfort_penalty,
            appliance_wasted_kwh=appliance_wasted_kwh,
            enhanced=True,
        )

    # =============================================================
    # ENERGY BALANCE AND REWARD
    # =============================================================

    def _calculate_energy_and_reward(
        self,
        action: np.ndarray,
        appliance_load_kw: float,
        renewable_kw: float,
        comfort_penalty: float,
        appliance_wasted_kwh: float,
        enhanced: bool,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:

        battery_action = int(action[-1])

        battery_power_kw = 0.0
        wasted_renewable_kw = 0.0

        available_surplus_kw = max(
            renewable_kw - appliance_load_kw,
            0.0,
        )

        # =========================================================
        # BATTERY ACTION
        # =========================================================

        if battery_action == 1:

            # Charge only when renewable surplus exists.
            charge_request_kw = min(
                available_surplus_kw,
                self.battery.max_charge_kw,
            )

            battery_power_kw = self.battery.charge(
                charge_request_kw,
                self.dt_hours,
            )

            wasted_renewable_kw = max(
                available_surplus_kw - battery_power_kw,
                0.0,
            )

        elif battery_action == 2:

            deficit_kw = max(
                appliance_load_kw - renewable_kw,
                0.0,
            )

            discharge_request_kw = min(
                deficit_kw,
                self.battery.max_discharge_kw,
            )

            battery_power_kw = self.battery.discharge(
                discharge_request_kw,
                self.dt_hours,
            )

        else:

            # Idle battery means surplus renewable energy
            # is wasted.
            wasted_renewable_kw = available_surplus_kw

        # =========================================================
        # GRID LOAD
        # =========================================================

        if battery_action == 1:

            net_grid_load_kw = (
                appliance_load_kw
                - renewable_kw
                + battery_power_kw
            )

        elif battery_action == 2:

            net_grid_load_kw = (
                appliance_load_kw
                - renewable_kw
                - battery_power_kw
            )

        else:

            net_grid_load_kw = (
                appliance_load_kw
                - renewable_kw
            )

        net_grid_load_kw = max(
            net_grid_load_kw,
            0.0,
        )

        # =========================================================
        # GRID AND REWARD
        # =========================================================

        overloaded = self.grid.is_overloaded(
            net_grid_load_kw
        )

        overload_amount_kw = max(
            net_grid_load_kw
            - self.grid.overload_threshold_kw,
            0.0,
        )

        price = self.grid.price(
            self.current_step,
            self.steps_per_day,
        )

        reward, reward_info = compute_reward(
            net_grid_load_kw=net_grid_load_kw,
            price_per_kwh=price,
            dt_hours=self.dt_hours,
            renewable_kw=renewable_kw,
            wasted_renewable_kw=wasted_renewable_kw,
            overloaded=overloaded,
            overload_amount_kw=overload_amount_kw,
            weights=self.reward_weights,
            comfort_penalty=comfort_penalty,
            appliance_wasted_kwh=appliance_wasted_kwh,
        )

        # =========================================================
        # MOVE TO NEXT TIME STEP
        # =========================================================

        self.current_step += 1

        done = (
            self.current_step >= self.steps_per_day
        )

        deferred_never_run = 0

        # Apply deferred appliance penalty only in original mode.
        if done and not enhanced:

            deferred_never_run = sum(
                1
                for appliance in self.appliances
                if (
                    appliance.deferrable
                    and appliance.state != ApplianceState.ON
                    and appliance.steps_running == 0
                )
            )

            reward += deferred_task_penalty(
                deferred_never_run,
                self.reward_weights,
            )

            reward_info[
                "deferred_never_run"
            ] = deferred_never_run

        # =========================================================
        # NEXT STATE
        # =========================================================

        next_step_for_generation = min(
            self.current_step,
            self.steps_per_day - 1,
        )

        next_renewable_kw = self.renewable.generation_kw(
            next_step_for_generation,
            self.steps_per_day,
        )

        if enhanced:

            state = self._enhanced_state(
                next_renewable_kw
            )

        else:

            state = encode_state(
                self.appliances,
                self.battery,
                next_renewable_kw,
                self.grid,
                self.current_step,
                self.steps_per_day,
            )

        # =========================================================
        # STEP INFORMATION
        # =========================================================

        info = {
            "appliance_load_kw": appliance_load_kw,
            "renewable_kw": renewable_kw,
            "net_grid_load_kw": net_grid_load_kw,
            "overloaded": overloaded,
            "battery_soc_fraction": self.battery.soc_fraction,
            "price_per_kwh": price,
            "wasted_renewable_kw": wasted_renewable_kw,
            "comfort_penalty": comfort_penalty,
            "appliance_wasted_kwh": appliance_wasted_kwh,
            "enhanced_mode": enhanced,
            **reward_info,
        }

        return (
            state,
            float(reward),
            done,
            False,
            info,
        )

    # =============================================================
    # MAIN STEP FUNCTION
    # =============================================================

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:

        action = np.asarray(
            action,
            dtype=np.int64,
        )

        if self.use_enhanced_appliances:

            return self._step_enhanced(action)

        return self._step_original(action)