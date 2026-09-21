
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smart_grid_rl.appliance_models import (
    ApplianceContext,
    ApplianceModel,
)


@dataclass
class ApplianceStepResult:
    load_kw: float
    wasted_kwh: float
    comfort_penalty: float
    observations: list[float]


class ApplianceManager:
    """Coordinates appliance models for one simulation."""

    def __init__(
        self,
        appliances: Sequence[ApplianceModel],
        dt_hours: float = 0.25,
    ) -> None:
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

        self.appliances = list(appliances)
        self.dt_hours = dt_hours

    @property
    def action_sizes(self) -> list[int]:
        return [
            appliance.num_actions()
            for appliance in self.appliances
        ]

    def reset(self) -> None:
        for appliance in self.appliances:
            appliance.reset()

    def apply_actions(
        self,
        actions: Sequence[int],
        ctx: ApplianceContext,
    ) -> None:
        if len(actions) != len(self.appliances):
            raise ValueError(
                "Number of actions must match number of appliances"
            )

        for appliance, action in zip(
            self.appliances,
            actions,
        ):
            action = int(action)

            if not 0 <= action < appliance.num_actions():
                raise ValueError(
                    f"Invalid action {action} for "
                    f"{appliance.name}. "
                    f"Expected 0 to "
                    f"{appliance.num_actions() - 1}"
                )

            appliance.apply_action(action, ctx)

    def step(
        self,
        ctx: ApplianceContext,
    ) -> ApplianceStepResult:
        load_kw = 0.0
        wasted_kwh = 0.0
        comfort_penalty = 0.0
        observations: list[float] = []

        for appliance in self.appliances:
            load_kw += appliance.tick(ctx)

            wasted_kwh += appliance.wasted_kwh(
                ctx,
                self.dt_hours,
            )

            comfort_penalty += appliance.comfort_penalty(ctx)

            observations.extend(
                appliance.obs(ctx)
            )

        return ApplianceStepResult(
            load_kw=load_kw,
            wasted_kwh=wasted_kwh,
            comfort_penalty=comfort_penalty,
            observations=observations,
        )