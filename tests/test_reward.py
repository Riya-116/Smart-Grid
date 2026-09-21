from smart_grid_rl.reward import RewardWeights, compute_reward, deferred_task_penalty


def test_higher_cost_gives_lower_reward():
    r_cheap, _ = compute_reward(
        net_grid_load_kw=1.0, price_per_kwh=0.10, dt_hours=1.0,
        renewable_kw=1.0, wasted_renewable_kw=0.0, overloaded=False, overload_amount_kw=0.0,
    )
    r_expensive, _ = compute_reward(
        net_grid_load_kw=1.0, price_per_kwh=0.50, dt_hours=1.0,
        renewable_kw=1.0, wasted_renewable_kw=0.0, overloaded=False, overload_amount_kw=0.0,
    )
    assert r_cheap > r_expensive


def test_overload_is_penalized():
    r_ok, _ = compute_reward(
        net_grid_load_kw=3.0, price_per_kwh=0.15, dt_hours=1.0,
        renewable_kw=1.0, wasted_renewable_kw=0.0, overloaded=False, overload_amount_kw=0.0,
    )
    r_overload, _ = compute_reward(
        net_grid_load_kw=10.0, price_per_kwh=0.15, dt_hours=1.0,
        renewable_kw=1.0, wasted_renewable_kw=0.0, overloaded=True, overload_amount_kw=2.0,
    )
    assert r_overload < r_ok


def test_wastage_is_penalized():
    r_no_waste, _ = compute_reward(
        net_grid_load_kw=1.0, price_per_kwh=0.15, dt_hours=1.0,
        renewable_kw=2.0, wasted_renewable_kw=0.0, overloaded=False, overload_amount_kw=0.0,
    )
    r_waste, _ = compute_reward(
        net_grid_load_kw=1.0, price_per_kwh=0.15, dt_hours=1.0,
        renewable_kw=2.0, wasted_renewable_kw=1.5, overloaded=False, overload_amount_kw=0.0,
    )
    assert r_waste < r_no_waste


def test_deferred_task_penalty_scales_with_count():
    weights = RewardWeights()
    assert deferred_task_penalty(0, weights) == 0.0
    assert deferred_task_penalty(2, weights) == -2 * weights.deferred_task_penalty


from smart_grid_rl.reward import (
    RewardWeights,
    compute_reward,
)


def test_appliance_penalties_reduce_reward():
    common_args = {
        "net_grid_load_kw": 2.0,
        "price_per_kwh": 0.20,
        "dt_hours": 0.25,
        "renewable_kw": 2.0,
        "wasted_renewable_kw": 0.0,
        "overloaded": False,
        "overload_amount_kw": 0.0,
    }

    reward_without_penalties, _ = compute_reward(
        **common_args,
        comfort_penalty=0.0,
        appliance_wasted_kwh=0.0,
    )

    reward_with_penalties, info = compute_reward(
        **common_args,
        comfort_penalty=2.0,
        appliance_wasted_kwh=0.5,
    )

    assert reward_with_penalties < reward_without_penalties

    assert info["comfort_penalty"] == 2.0
    assert info["appliance_wasted_kwh"] == 0.5


def test_default_appliance_penalties_are_zero():
    reward, info = compute_reward(
        net_grid_load_kw=2.0,
        price_per_kwh=0.20,
        dt_hours=0.25,
        renewable_kw=2.0,
        wasted_renewable_kw=0.0,
        overloaded=False,
        overload_amount_kw=0.0,
    )

    assert info["comfort_penalty"] == 0.0
    assert info["appliance_wasted_kwh"] == 0.0
    assert isinstance(reward, float)