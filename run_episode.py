"""Runs one full episode with random actions to verify the environment
loop end-to-end (including the now-implemented reward function).

STATUS: unchanged in purpose from the original, updated only to also
print total reward now that it's no longer stubbed to 0.0.
"""
from smart_grid_rl.env import VirtualSmartHomeEnv


def main() -> None:
    env = VirtualSmartHomeEnv(seed=7)
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}, info: {info}")

    total_overloads = 0
    total_reward = 0.0
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        total_overloads += int(info["overloaded"])
        total_reward += reward
        step_count += 1

    print(f"Episode finished after {step_count} steps.")
    print(f"Total overload steps: {total_overloads}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final battery SoC fraction: {info['battery_soc_fraction']:.3f}")


if __name__ == "__main__":
    main()
