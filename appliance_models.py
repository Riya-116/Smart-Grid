
"""
Appliance models for comfort-aware and safety-aware smart-grid control.

This module is independent of the original components.Appliance class.
It can be integrated gradually into VirtualSmartHomeEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ApplianceCategory(Enum):
    THERMOSTATIC = "thermostatic"
    SHIFTABLE = "shiftable"
    ON_DEMAND = "on_demand"
    ESSENTIAL = "essential"


@dataclass
class ApplianceContext:
    """Information available to every appliance during one time step."""

    step: int
    day: int
    outdoor_temp_c: float
    indoor_temp_c: float
    occupancy: bool
    price_per_kwh: float
    renewable_kw: float


class ApplianceModel:
    """Common interface for all new appliance models."""

    name: str
    rated_power_kw: float
    category: ApplianceCategory
    is_on: bool = False

    def num_actions(self) -> int:
        raise NotImplementedError

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        raise NotImplementedError

    def tick(self, ctx: ApplianceContext) -> float:
        raise NotImplementedError

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        return 0.0

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        return 0.0

    def obs(self, ctx: ApplianceContext) -> list[float]:
        raise NotImplementedError

    def reset(self) -> None:
        self.is_on = False


@dataclass
class ThermostaticAppliance(ApplianceModel):
    """
    Examples:
    - AC
    - Heater
    - Refrigerator

    Action:
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
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")

        if self.comfort_min_c >= self.comfort_max_c:
            raise ValueError(
                "comfort_min_c must be less than comfort_max_c"
            )

        if self.mode not in {"cooling", "heating"}:
            raise ValueError(
                "mode must be 'cooling' or 'heating'"
            )

    def num_actions(self) -> int:
        return 2

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        if action not in (0, 1):
            raise ValueError(
                f"Invalid action {action} for {self.name}"
            )

        self.is_on = action == 1

    def tick(self, ctx: ApplianceContext) -> float:
        return self.rated_power_kw if self.is_on else 0.0

    def _out_of_band_c(
        self,
        ctx: ApplianceContext,
    ) -> float:
        if ctx.indoor_temp_c < self.comfort_min_c:
            return self.comfort_min_c - ctx.indoor_temp_c

        if ctx.indoor_temp_c > self.comfort_max_c:
            return ctx.indoor_temp_c - self.comfort_max_c

        return 0.0

    def comfort_penalty(
        self,
        ctx: ApplianceContext,
    ) -> float:
        return self._out_of_band_c(ctx)

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
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
        band_span = max(
            self.comfort_max_c - self.comfort_min_c,
            1e-6,
        )

        deviation = min(
            self._out_of_band_c(ctx) / band_span,
            1.0,
        )

        return [
            1.0 if self.is_on else 0.0,
            deviation,
        ]


@dataclass
class ShiftableAppliance(ApplianceModel):
    """
    Examples:
    - Washing machine
    - Dishwasher
    - Dryer
    - EV charging

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
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")

        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")

        if self.deadline_step < 0:
            raise ValueError("deadline_step cannot be negative")

    def num_actions(self) -> int:
        return 3

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        if action not in (0, 1, 2):
            raise ValueError(
                f"Invalid action {action} for {self.name}"
            )

        if self.finished:
            self.is_on = False
            self.deferred = False
            return

        if action == 1:
            self.is_on = True
            self.deferred = False

        elif action == 2:
            self.is_on = False
            self.deferred = True

        else:
            self.is_on = False
            self.deferred = False

    def tick(self, ctx: ApplianceContext) -> float:
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
        if self.finished:
            return 0.0

        steps_left = max(
            self.deadline_step - ctx.step,
            0,
        )

        remaining_work = max(
            self.duration_steps - self.steps_completed,
            0,
        )

        # Positive penalty when remaining work cannot fit
        # into the remaining time before the deadline.
        behind_schedule = remaining_work - steps_left

        return float(max(behind_schedule, 0))

    def is_late(self, current_step: int) -> bool:
        return (
            not self.finished
            and current_step >= self.deadline_step
        )

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        remaining_work = max(
            self.duration_steps - self.steps_completed,
            0,
        )

        steps_left = max(
            self.deadline_step - ctx.step,
            0,
        )

        if steps_left == 0:
            urgency = 1.0 if remaining_work > 0 else 0.0
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
        self.is_on = False
        self.deferred = False
        self.steps_completed = 0
        self.finished = False


@dataclass
class OnDemandAppliance(ApplianceModel):
    """
    Examples:
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
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")

    def num_actions(self) -> int:
        return 2

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        if action not in (0, 1):
            raise ValueError(
                f"Invalid action {action} for {self.name}"
            )

        self.is_on = action == 1

    def tick(self, ctx: ApplianceContext) -> float:
        return self.rated_power_kw if self.is_on else 0.0

    def wasted_kwh(
        self,
        ctx: ApplianceContext,
        dt_hours: float,
    ) -> float:
        if self.is_on and not ctx.occupancy:
            return self.rated_power_kw * dt_hours

        return 0.0

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        return [
            1.0 if self.is_on else 0.0,
            1.0 if ctx.occupancy else 0.0,
        ]


@dataclass
class EssentialAppliance(ApplianceModel):
    """
    Examples:
    - Router
    - Medical equipment
    - Safety monitoring devices

    There is only one legal action:
    0 = REMAIN ON

    The appliance cannot be switched off through the RL action.
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
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")

    def num_actions(self) -> int:
        return 1

    def apply_action(
        self,
        action: int,
        ctx: ApplianceContext,
    ) -> None:
        if action != 0:
            raise ValueError(
                f"Only action 0 is legal for {self.name}"
            )

        self.is_on = True

    def tick(self, ctx: ApplianceContext) -> float:
        return self.rated_power_kw

    def obs(
        self,
        ctx: ApplianceContext,
    ) -> list[float]:
        return [1.0]

    def reset(self) -> None:
        self.is_on = True