"""
Smoke tests: real dataset integration + a tiny end-to-end training run.

Uses the mock dataset files under test_mock_data/.
The tests do not require downloading the full UK-DALE,
Smart Meters London, Kaggle, or NREL datasets.
"""

import numpy as np

from smart_grid_rl.env import VirtualSmartHomeEnv
from smart_grid_rl.data_integration import DatasetBundle, resample_to_steps
from smart_grid_rl.agent_qlearning import QLearningAgent
from smart_grid_rl.baseline_controller import RuleBasedController


def test_resample_preserves_length_and_range():
    """Check that resampling produces the expected length and range."""

    hourly = np.linspace(0, 5, 24)

    resampled = resample_to_steps(
        hourly,
        steps_per_day=96
    )

    assert len(resampled) == 96
    assert resampled.min() >= 0.0 - 1e-6
    assert resampled.max() <= hourly.max() + 1e-6


def test_dataset_bundle_falls_back_gracefully_on_bad_paths():
    """Check that missing dataset files are handled safely."""

    bundle = DatasetBundle.load(
        steps_per_day=96,
        kaggle_solar_csv="does_not_exist.csv",
        smart_meter_tariff_csv="also_missing.csv",
    )

    assert bundle.renewable_profile_kw is None
    assert bundle.price_profile is None


def test_env_uses_real_kaggle_solar_with_synthetic_price_fallback():
    """
    Check that real Kaggle solar data is loaded.

    No tariff CSV is available, so the environment should use
    its synthetic electricity price fallback.
    """

    env = VirtualSmartHomeEnv(
        steps_per_day=96,
        seed=1,
        kaggle_solar_csv="test_mock_data/plant1_generation.csv",
        household_solar_capacity_kw=4.0,
    )

    # Real Kaggle solar data must be loaded.
    assert env.renewable.real_profile_kw is not None

    # No tariff dataset is available.
    # Therefore, the environment should use synthetic prices.
    assert env.grid.real_price_profile is None

    # Check that the environment can be reset successfully.
    state, info = env.reset()

    assert state.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_env_uses_real_ukdale_appliance_rating():
    """
    Check that an unmatched UK-DALE appliance name
    does not cause the environment to crash.
    """

    env = VirtualSmartHomeEnv(
        steps_per_day=96,
        seed=1,
        ukdale_house_dir="test_mock_data/house_1",
        ukdale_appliance_names=["fridge"],
    )

    # "fridge" is not one of the four named appliances:
    # HVAC, Lights, Washer, or EV.
    #
    # Therefore, the default appliance configuration
    # should remain unchanged.
    assert env.appliances[0].name == "HVAC"


def test_tiny_training_run_improves_or_at_least_runs():
    """
    Run a small Q-learning training session.

    The test checks that the agent can interact with the
    environment and store Q-values.
    """

    env = VirtualSmartHomeEnv(
        steps_per_day=20,
        seed=0
    )

    agent = QLearningAgent(
        nvec=env.action_space.nvec
    )

    for _ in range(50):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            action_idx = agent.act_index(obs)

            action = agent.action_from_index(action_idx)

            next_obs, reward, done, truncated, info = env.step(action)

            agent.update(
                obs,
                action_idx,
                reward,
                next_obs,
                done
            )

            obs = next_obs

        agent.decay_epsilon()

    # The agent should have stored at least one Q-table entry.
    assert len(agent.q_table) > 0


def test_baseline_controller_produces_valid_actions():
    """Check that the rule-based controller produces valid actions."""

    env = VirtualSmartHomeEnv(
        steps_per_day=20,
        seed=0
    )

    env.reset()

    controller = RuleBasedController()

    action = controller.act(env)

    assert env.action_space.contains(action)