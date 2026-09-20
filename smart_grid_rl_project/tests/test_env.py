import numpy as np
import pytest

from smart_grid_rl.env import VirtualSmartHomeEnv


def test_reset_returns_correct_shape():
    env = VirtualSmartHomeEnv(seed=1)
    state, info = env.reset()
    assert state.shape == env.observation_space.shape
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)


def test_episode_terminates_after_steps_per_day():
    env = VirtualSmartHomeEnv(steps_per_day=10, seed=1)
    env.reset()
    done = False
    count = 0
    while not done:
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        count += 1
    assert count == 10


def test_step_rejects_wrong_action_length():
    env = VirtualSmartHomeEnv(seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([1]))  # Invalid action length


# --- New tests: verify the surplus/deficit bug fix ---

def test_charge_action_does_not_force_max_rate_during_deficit():
    """Regression test for the original max()-instead-of-min() bug: choosing
    'charge' while there is a demand deficit (not a surplus) must NOT push
    the battery to its max charge rate."""
    env = VirtualSmartHomeEnv(steps_per_day=96, seed=1)
    env.reset()
    soc_before = env.battery.soc_kwh
    # HVAC on (2.5kW), all others off, battery = charge
    action = np.array([1, 0, 0, 0, 1])
    _, _, _, _, info = env.step(action)
    soc_after = env.battery.soc_kwh
    # If there's little/no renewable, soc should barely move (not jump by
    # the full max_charge_kw * dt_hours = 1.25 kWh the bug would cause)
    assert (soc_after - soc_before) < 1.0


def test_discharge_action_does_not_force_max_rate_without_deficit():
    """Regression test in the other direction: 'discharge' with all
    appliances off (no deficit) shouldn't drain the battery at full rate."""
    env = VirtualSmartHomeEnv(steps_per_day=96, seed=1)
    env.reset()
    soc_before = env.battery.soc_kwh
    action = np.array([0, 0, 0, 0, 2])  # everything off, battery = discharge
    _, _, _, _, info = env.step(action)
    soc_after = env.battery.soc_kwh
    assert (soc_before - soc_after) < 0.1  # should barely discharge, no real deficit


def test_reward_is_no_longer_stubbed_to_zero():
    env = VirtualSmartHomeEnv(steps_per_day=96, seed=1)
    env.reset()
    action = np.array([1, 1, 0, 0, 0])
    _, reward, _, _, _ = env.step(action)
    assert reward != 0.0


def test_wasted_renewable_tracked_in_info():
    env = VirtualSmartHomeEnv(steps_per_day=96, seed=1)
    env.reset()
    action = np.array([0, 0, 0, 0, 0])  # everything idle -> any solar is wasted
    _, _, _, _, info = env.step(action)
    assert "wasted_renewable_kw" in info
    assert info["wasted_renewable_kw"] >= 0.0
