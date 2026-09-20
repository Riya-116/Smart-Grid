"""
experiments.py
------------------------------------------------------------------
NEW FILE -- runs the trained RL agent and the rule-based baseline over
the same number of test days, averages metrics.py results, prints a
comparison table, and saves a bar chart.
------------------------------------------------------------------
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smart_grid_rl.env import VirtualSmartHomeEnv
from smart_grid_rl.agent_qlearning import QLearningAgent
from smart_grid_rl.baseline_controller import RuleBasedController
from smart_grid_rl.metrics import compute_episode_metrics


def run_rl_episode(env, agent):
    obs, _ = env.reset()
    done = False
    info_history = []
    while not done:
        action = agent.act(obs, greedy=True)
        obs, reward, done, truncated, info = env.step(action)
        info_history.append(info)
    return info_history, env.battery.degradation_index()


def run_baseline_episode(env, controller):
    env.reset()
    done = False
    info_history = []
    while not done:
        action = controller.act(env)
        _, reward, done, truncated, info = env.step(action)
        info_history.append(info)
    return info_history, env.battery.degradation_index()


def compare(n_test_days: int = 20, q_table_path: str = "q_table.pkl",
            steps_per_day: int = 96, seed: int = 123, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs or {}
    env = VirtualSmartHomeEnv(steps_per_day=steps_per_day, seed=seed, **env_kwargs)
    agent = QLearningAgent(nvec=env.action_space.nvec)
    agent.load(q_table_path)
    controller = RuleBasedController()

    rl_metrics, base_metrics = [], []
    for _ in range(n_test_days):
        info_hist, degr = run_rl_episode(env, agent)
        rl_metrics.append(compute_episode_metrics(info_hist, degr))
        info_hist, degr = run_baseline_episode(env, controller)
        base_metrics.append(compute_episode_metrics(info_hist, degr))

    keys = rl_metrics[0].keys()
    summary = {
        k: {
            "RL": round(float(np.mean([m[k] for m in rl_metrics])), 3),
            "Baseline": round(float(np.mean([m[k] for m in base_metrics])), 3),
        }
        for k in keys
    }

    print(f"{'Metric':30s}{'RL':>12s}{'Baseline':>12s}")
    for k, v in summary.items():
        print(f"{k:30s}{v['RL']:>12}{v['Baseline']:>12}")

    _plot_comparison(summary)
    return summary


def _plot_comparison(summary, out_path="rl_vs_baseline.png"):
    metrics_to_plot = ["total_cost", "renewable_utilization_pct",
                        "total_energy_wasted_kwh", "overload_rate_pct"]
    rl_vals = [summary[m]["RL"] for m in metrics_to_plot]
    base_vals = [summary[m]["Baseline"] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, rl_vals, width, label="RL Agent (Q-Learning)")
    ax.bar(x + width / 2, base_vals, width, label="Rule-Based Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
    ax.set_title("RL Agent vs Rule-Based Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    compare()
