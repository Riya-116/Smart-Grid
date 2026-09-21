
"""
Appliance models for comfort-aware and safety-aware
smart-grid control.

This module is independent of the original components.Appliance
class and can be integrated gradually into VirtualSmartHomeEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# APPLIANCE CATEGORY
# ============================================================

class ApplianceCategory(Enum):
    """Categories of enhanced appliances."""

    THERMOSTATIC = "thermostatic"
    SHIFTABLE = "shiftable"
    ON_DEMAND = "on_demand"
    ESSENTIAL = "essential"


# ============================================================
# APPLIANCE CONTEXT
# ============================================================

@dataclass
class ApplianceContext:
    """Information available to every appliance at one time step."""

    step: int
    day: int
    outdoor_temp_c: float
    indoor_temp_c: float
    occupancy: bool
    price_per_kwh: float
    renewable_kw: float


# ============================================================
# BASE APPLIANCE MODEL
# ============================================================

class ApplianceModel:
    """Common interface for all enhanced appliance models."""

    name: str
    rated_power_kw: float
    category: ApplianceCategory
    is_on: bool = False

    def num_actions(self) -> int:
        """Return the number of legal actions."""

        raise NotImplementedError

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        """Apply an action to the appliance."""

        raise NotImplementedError

    def tick(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """
        Advance the appliance by one time step.

        Returns:
            Power consumption in kilowatts (kW).
        """

        raise NotImplementedError

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Return comfort or scheduling penalty."""

        return 0.0

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        """Return wasted energy in kilowatt-hours."""

        return 0.0

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        """Return the appliance observation values."""

        raise NotImplementedError

    def reset(self) -> None:
        """Reset the appliance to its default state."""

        self.is_on = False


# ============================================================
# THERMOSTATIC APPLIANCE
# ============================================================

@dataclass
class ThermostaticAppliance(ApplianceModel):
    """
    Thermostatic appliances include:

    - Air conditioner
    - Heater
    - Refrigerator

    Actions:
        0 = OFF
        1 = ON
    """

    name: str
    rated_power_kw: float
    comfort_min_c: float
    comfort_max_c: float
    mode: str = "cooling"
    is_on: bool = False

    category: ApplianceCategory = field(
        default=ApplianceCategory.THERMOSTATIC,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate the appliance configuration."""

        if self.rated_power_kw <= 0:
            raise ValueError(
                "rated_power_kw must be positive"
            )

        if self.comfort_min_c >= self.comfort_max_c:
            raise ValueError(
                "comfort_min_c must be less than comfort_max_c"
            )

        if self.mode not in {"cooling", "heating"}:
            raise ValueError(
                "mode must be 'cooling' or 'heating'"
            )

    def num_actions(self) -> int:
        """Thermostatic appliances have two actions."""

        return 2

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        """Turn the thermostatic appliance ON or OFF."""

        if action not in (0, 1):
            raise ValueError(
                f"Invalid action {action} for {self.name}. "
                "Valid actions are 0 and 1."
            )

        self.is_on = action == 1

    def tick(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Return the current power consumption."""

        if self.is_on:
            return self.rated_power_kw

        return 0.0

    def _out_of_band_c(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Calculate the temperature deviation from comfort range."""

        if ctx.indoor_temp_c < self.comfort_min_c:
            return (
                self.comfort_min_c
                - ctx.indoor_temp_c
            )

        if ctx.indoor_temp_c > self.comfort_max_c:
            return (
                ctx.indoor_temp_c
                - self.comfort_max_c
            )

        return 0.0

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Return temperature comfort penalty."""

        return self._out_of_band_c(ctx)

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        """
        Calculate wasted energy.

        Energy is considered wasted when the appliance is ON
        even though the indoor temperature is within the
        comfort range.
        """

        in_comfort_band = (
            self.comfort_min_c
            <= ctx.indoor_temp_c
            <= self.comfort_max_c
        )

        if self.is_on and in_comfort_band:
            return self.rated_power_kw * dt_hours

        return 0.0

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        """Return the thermostatic appliance observation."""

        band_span = max(
            self.comfort_max_c
            - self.comfort_min_c,
            1e-6,
        )

        deviation = min(
            self._out_of_band_c(ctx)
            / band_span,
            1.0,
        )

        return [
            1.0 if self.is_on else 0.0,
            deviation,
        ]

    def reset(self) -> None:
        """Reset the thermostatic appliance."""

        self.is_on = False


# ============================================================
# SHIFTABLE APPLIANCE
# ============================================================

@dataclass
class ShiftableAppliance(ApplianceModel):
    """
    Shiftable appliances include:

    - Washing machine
    - Dishwasher
    - Dryer
    - Electric vehicle charger

    Actions:
        0 = OFF
        1 = ON
        2 = DEFER
    """

    name: str
    rated_power_kw: float
    duration_steps: int
    deadline_step: int

    is_on: bool = False
    deferred: bool = False
    steps_completed: int = 0
    finished: bool = False

    category: ApplianceCategory = field(
        default=ApplianceCategory.SHIFTABLE,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate the shiftable appliance configuration."""

        if self.rated_power_kw <= 0:
            raise ValueError(
                "rated_power_kw must be positive"
            )

        if self.duration_steps <= 0:
            raise ValueError(
                "duration_steps must be positive"
            )

        if self.deadline_step < 0:
            raise ValueError(
                "deadline_step cannot be negative"
            )

    def num_actions(self) -> int:
        """Shiftable appliances have three actions."""

        return 3

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        """Apply ON, OFF, or DEFER action."""

        if action not in (0, 1, 2):
            raise ValueError(
                f"Invalid action {action} for {self.name}. "
                "Valid actions are 0, 1, and 2."
            )

        # A finished appliance cannot be restarted.
        if self.finished:
            self.is_on = False
            self.deferred = False
            return

        if action == 1:
            # Start or continue the appliance.
            self.is_on = True
            self.deferred = False

        elif action == 2:
            # Defer operation.
            self.is_on = False
            self.deferred = True

        else:
            # Turn the appliance OFF.
            self.is_on = False
            self.deferred = False

    def tick(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """
        Advance the shiftable appliance.

        Returns:
            Current power consumption in kW.
        """

        if self.finished:
            self.is_on = False
            return 0.0

        if not self.is_on:
            return 0.0

        self.steps_completed += 1

        if self.steps_completed >= self.duration_steps:
            self.steps_completed = self.duration_steps
            self.finished = True
            self.is_on = False

        return self.rated_power_kw

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """
        Calculate scheduling penalty.

        A penalty is generated when the remaining work
        cannot fit before the deadline.
        """

        if self.finished:
            return 0.0

        steps_left = max(
            self.deadline_step - ctx.step,
            0,
        )

        remaining_work = max(
            self.duration_steps
            - self.steps_completed,
            0,
        )

        behind_schedule = (
            remaining_work - steps_left
        )

        return float(
            max(behind_schedule, 0)
        )

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        """
        Calculate wasted energy.

        Shiftable appliances do not have a separate
        wasted-energy condition in this basic model.
        """

        return 0.0

    def is_late(
        self,
        current_step: int,
    ) -> bool:
        """Check whether the appliance has missed its deadline."""

        return (
            not self.finished
            and current_step >= self.deadline_step
        )

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        """Return shiftable appliance observations."""

        remaining_work = max(
            self.duration_steps
            - self.steps_completed,
            0,
        )

        steps_left = max(
            self.deadline_step - ctx.step,
            0,
        )

        if steps_left == 0:
            urgency = (
                1.0
                if remaining_work > 0
                else 0.0
            )

        else:
            urgency = min(
                remaining_work / steps_left,
                1.0,
            )

        return [
            1.0 if self.is_on else 0.0,
            1.0 if self.finished else 0.0,
            urgency,
        ]

    def reset(self) -> None:
        """Reset the shiftable appliance."""

        self.is_on = False
        self.deferred = False
        self.steps_completed = 0
        self.finished = False


# ============================================================
# ON-DEMAND APPLIANCE
# ============================================================

@dataclass
class OnDemandAppliance(ApplianceModel):
    """
    On-demand appliances include:

    - Fan
    - Lights
    - Television

    Actions:
        0 = OFF
        1 = ON
    """

    name: str
    rated_power_kw: float
    is_on: bool = False

    category: ApplianceCategory = field(
        default=ApplianceCategory.ON_DEMAND,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate the on-demand appliance configuration."""

        if self.rated_power_kw <= 0:
            raise ValueError(
                "rated_power_kw must be positive"
            )

    def num_actions(self) -> int:
        """On-demand appliances have two actions."""

        return 2

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        """Turn the on-demand appliance ON or OFF."""

        if action not in (0, 1):
            raise ValueError(
                f"Invalid action {action} for {self.name}. "
                "Valid actions are 0 and 1."
            )

        self.is_on = action == 1

    def tick(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Return the current power consumption."""

        if self.is_on:
            return self.rated_power_kw

        return 0.0

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """On-demand appliances have no comfort penalty."""

        return 0.0

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        """
        Calculate wasted energy.

        Energy is considered wasted when the appliance
        is ON while nobody is occupying the home.
        """

        if self.is_on and not ctx.occupancy:
            return self.rated_power_kw * dt_hours

        return 0.0

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        """Return on-demand appliance observations."""

        return [
            1.0 if self.is_on else 0.0,
            1.0 if ctx.occupancy else 0.0,
        ]

    def reset(self) -> None:
        """Reset the on-demand appliance."""

        self.is_on = False


# ============================================================
# ESSENTIAL APPLIANCE
# ============================================================

@dataclass
class EssentialAppliance(ApplianceModel):
    """
    Essential appliances include:

    - Router
    - Medical equipment
    - Safety monitoring devices

    Actions:
        0 = REMAIN ON

    Essential appliances cannot be switched OFF.
    """

    name: str
    rated_power_kw: float

    is_on: bool = field(
        default=True,
        init=False,
    )

    category: ApplianceCategory = field(
        default=ApplianceCategory.ESSENTIAL,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate the essential appliance configuration."""

        if self.rated_power_kw <= 0:
            raise ValueError(
                "rated_power_kw must be positive"
            )

    def num_actions(self) -> int:
        """Essential appliances have one legal action."""

        return 1

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        """Keep the essential appliance switched ON."""

        if action != 0:
            raise ValueError(
                f"Only action 0 is legal for {self.name}"
            )

        self.is_on = True

    def tick(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Essential appliances always consume rated power."""

        return self.rated_power_kw

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        """Essential appliances have no comfort penalty."""

        return 0.0

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        """Essential appliances have no wasted energy penalty."""

        return 0.0

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        """Return essential appliance observations."""

        return [1.0]

    def reset(self) -> None:
        """Reset while keeping the essential appliance ON."""

        self.is_on = True