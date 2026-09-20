import pytest

from smart_grid_rl.components import Appliance, ApplianceState, BatteryStorage, GridModel


def test_appliance_min_run_steps_blocks_early_shutoff():
    a = Appliance(name="Washer", rated_power_kw=1.2, min_run_steps=3)
    a.turn_on()
    a.tick()  # steps_running = 1
    a.turn_off()  # should be blocked
    assert a.state == ApplianceState.ON


def test_appliance_rejects_nonpositive_power():
    with pytest.raises(ValueError):
        Appliance(name="Bad", rated_power_kw=0)


def test_battery_charge_respects_max_soc():
    b = BatteryStorage(capacity_kwh=10.0, max_charge_kw=5.0, max_discharge_kw=5.0)
    b.soc_kwh = 9.4  # near max_soc_fraction default (0.95 -> 9.5 kWh)
    accepted_kw = b.charge(5.0, dt_hours=1.0)
    assert b.soc_kwh <= b.max_soc_kwh + 1e-6
    assert accepted_kw >= 0


def test_battery_discharge_respects_min_soc():
    b = BatteryStorage(capacity_kwh=10.0, max_charge_kw=5.0, max_discharge_kw=5.0)
    b.soc_kwh = 1.05  # near min_soc_fraction default (0.10 -> 1.0 kWh)
    b.discharge(5.0, dt_hours=1.0)
    assert b.soc_kwh >= b.min_soc_kwh - 1e-6


def test_grid_overload_detection():
    g = GridModel(overload_threshold_kw=5.0)
    assert g.is_overloaded(6.0) is True
    assert g.is_overloaded(4.0) is False


def test_grid_price_bounds_synthetic():
    g = GridModel(base_price_per_kwh=0.15, peak_price_per_kwh=0.35)
    lo, hi = g.price_bounds()
    assert lo == 0.15 and hi == 0.35


def test_grid_price_bounds_real_profile():
    g = GridModel(real_price_profile=[0.04, 0.12, 0.67, 0.12])
    lo, hi = g.price_bounds()
    assert lo == pytest.approx(0.04) and hi == pytest.approx(0.67)
