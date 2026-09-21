
from smart_grid_rl.env import VirtualSmartHomeEnv


# Create the enhanced environment
env = VirtualSmartHomeEnv(use_enhanced_appliances=True)

# Reset the environment
state, info = env.reset()

total_reward = 0.0
steps = 0

terminated = False
truncated = False

# Run one complete episode
while not terminated and not truncated:
    # Select a random valid action
    action = env.action_space.sample()

    # Perform the environment step
    state, reward, terminated, truncated, info = env.step(action)

    # Accumulate the reward
    total_reward += reward
    steps += 1


print("Episode steps:", steps)
print("Total reward:", total_reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Final state shape:", state.shape)

print("Complete episode test passed")