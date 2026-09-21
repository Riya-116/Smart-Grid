
import pytest

from smart_grid_rl.appliance_models import (
    ApplianceContext,
    ThermostaticAppliance,
    ShiftableAppliance,
    OnDemandAppliance,
    EssentialAppliance,
)


@pytest.fixture
def context():
    return ApplianceContext(
        step=10,
        day=0,
        outdoor_temp_c=32.0,
        indoor_temp_c=30.0,
        occupancy=True,
        price_per_kwh=0.20,
        renewable_kw=2.0,
    )


def test_thermostatic_appliance_actions(context):
    ac = ThermostaticAppliance(
        name="AC",
        rated_power_kw=2.5,
        comfort_min_c=22.0,
        comfort_max_c=26.0,
        mode="cooling",
    )

    assert ac.num_actions() == 2

    ac.apply_action(1, context)
    assert ac.is_on is True
    assert ac.tick(context) == 2.5

    ac.apply_action(0, context)
    assert ac.is_on is False
    assert ac.tick(context) == 0.0


def test_thermostatic_comfort_penalty(context):
    ac = ThermostaticAppliance(
        name="AC",
        rated_power_kw=2.5,
        comfort_min_c=22.0,
        comfort_max_c=26.0,
    )

    # Indoor temperature = 30 C.
    # Upper comfort limit = 26 C.
    assert ac.comfort_penalty(context) == 4.0


def test_thermostatic_wasted_energy(context):
    ac = ThermostaticAppliance(
        name="AC",
        rated_power_kw=2.5,
        comfort_min_c=22.0,
        comfort_max_c=26.0,
    )

    # Move indoor temperature inside the comfort band.
    context.indoor_temp_c = 24.0

    ac.apply_action(1, context)

    wasted = ac.wasted_kwh(
        context,
        dt_hours=0.25,
    )

    assert wasted == pytest.approx(0.625)


def test_shiftable_appliance_completion(context):
    washer = ShiftableAppliance(
        name="Washer",
        rated_power_kw=1.2,
        duration_steps=2,
        deadline_step=20,
    )

    assert washer.num_actions() == 3

    washer.apply_action(1, context)

    assert washer.is_on is True
    assert washer.tick(context) == 1.2
    assert washer.finished is False

    assert washer.tick(context) == 1.2
    assert washer.finished is True


def test_shiftable_appliance_urgency_penalty(context):
    washer = ShiftableAppliance(
        name="Washer",
        rated_power_kw=1.2,
        duration_steps=5,
        deadline_step=10,
    )

    # At step 10, five steps of work remain.
    penalty = washer.comfort_penalty(context)

    assert penalty == 5.0


def test_on_demand_wasted_energy_when_unoccupied(context):
    fan = OnDemandAppliance(
        name="Fan",
        rated_power_kw=0.075,
    )

    context.occupancy = False
    fan.apply_action(1, context)

    wasted = fan.wasted_kwh(
        context,
        dt_hours=0.25,
    )

    assert wasted == pytest.approx(0.01875)


def test_essential_appliance_cannot_turn_off(context):
    router = EssentialAppliance(
        name="Router",
        rated_power_kw=0.012,
    )

    assert router.num_actions() == 1
    assert router.is_on is True

    router.apply_action(0, context)

    assert router.is_on is True
    assert router.tick(context) == 0.012

    with pytest.raises(ValueError):
        router.apply_action(1, context)