"""
train.py
------------------------------------------------------------------
NEW FILE -- training loop: runs episodes of VirtualSmartHomeEnv, has
the Q-learning agent act + learn every step, decays epsilon each
episode, saves the learned Q-table.
------------------------------------------------------------------
"""

import numpy as np

from smart_grid_rl.env import VirtualSmartHomeEnv
from smart_grid_rl.agent_qlearning import QLearningAgent

N_EPISODES = 3000


def train(n_episodes: int = N_EPISODES, save_path: str = "q_table.pkl",
          steps_per_day: int = 96, seed: int | None = None, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs or {}
    env = VirtualSmartHomeEnv(steps_per_day=steps_per_day, seed=seed, **env_kwargs)
    agent = QLearningAgent(nvec=env.action_space.nvec)

    reward_history = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action_idx = agent.act_index(obs)
            action = agent.action_from_index(action_idx)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.update(obs, action_idx, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        reward_history.append(ep_reward)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            avg = np.mean(reward_history[-max(1, n_episodes // 10):])
            print(f"Episode {ep+1}/{n_episodes}  avg_reward={avg:.2f}  epsilon={agent.epsilon:.3f}")

    agent.save(save_path)
    print(f"Saved trained Q-table to {save_path}")
    return agent, reward_history


if __name__ == "__main__":
    train()
