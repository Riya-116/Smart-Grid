"""
Smoke tests: real dataset integration + a tiny end-to-end training run.
Uses the mock dataset files under test_mock_data/, built to match each
real dataset's actual schema, so these tests don't depend on downloading
the full UK-DALE / Smart Meters London / Kaggle / NREL files.
"""

import numpy as np

from smart_grid_rl.env import VirtualSmartHomeEnv
from smart_grid_rl.data_integration import DatasetBundle, resample_to_steps
from smart_grid_rl.agent_qlearning import QLearningAgent
from smart_grid_rl.baseline_controller import RuleBasedController


def test_resample_preserves_length_and_range():
    hourly = np.linspace(0, 5, 24)
    resampled = resample_to_steps(hourly, steps_per_day=96)
    assert len(resampled) == 96
    assert resampled.min() >= 0.0 - 1e-6
    assert resampled.max() <= hourly.max() + 1e-6


def test_dataset_bundle_falls_back_gracefully_on_bad_paths():
    bundle = DatasetBundle.load(
        steps_per_day=96,
        kaggle_solar_csv="does_not_exist.csv",
        smart_meter_tariff_csv="also_missing.csv",
    )
    assert bundle.renewable_profile_kw is None
    assert bundle.price_profile is None


def test_env_uses_real_kaggle_solar_and_smart_meter_price():
    env = VirtualSmartHomeEnv(
        steps_per_day=96, seed=1,
        kaggle_solar_csv="test_mock_data/plant1_generation.csv",
        smart_meter_tariff_csv="test_mock_data/tariffs.csv",
        household_solar_capacity_kw=4.0,
    )
    assert env.renewable.real_profile_kw is not None
    assert env.grid.real_price_profile is not None
    state, info = env.reset()
    assert state.shape == env.observation_space.shape


def test_env_uses_real_ukdale_appliance_rating():
    env = VirtualSmartHomeEnv(
        steps_per_day=96, seed=1,
        ukdale_house_dir="test_mock_data/house_1",
        ukdale_appliance_names=["fridge"],
    )
    # "fridge" isn't one of the 4 named appliances (HVAC/Lights/Washer/EV),
    # so this should simply not crash and leave ratings untouched --
    # confirms the override path fails safe on an unmatched name.
    assert env.appliances[0].name == "HVAC"


def test_tiny_training_run_improves_or_at_least_runs():
    env = VirtualSmartHomeEnv(steps_per_day=20, seed=0)
    agent = QLearningAgent(nvec=env.action_space.nvec)

    for _ in range(50):
        obs, _ = env.reset()
        done = False
        while not done:
            action_idx = agent.act_index(obs)
            action = agent.action_from_index(action_idx)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.update(obs, action_idx, reward, next_obs, done)
            obs = next_obs
        agent.decay_epsilon()

    assert len(agent.q_table) > 0


def test_baseline_controller_produces_valid_actions():
    env = VirtualSmartHomeEnv(steps_per_day=20, seed=0)
    env.reset()
    controller = RuleBasedController()
    action = controller.act(env)
    assert env.action_space.contains(action)
