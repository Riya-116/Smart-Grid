
import numpy as np

from smart_grid_rl.env import VirtualSmartHomeEnv


# Create the environment using the new appliance models
env = VirtualSmartHomeEnv(
    use_enhanced_appliances=True
)

# Reset the environment
state, info = env.reset()

print("Initial state:", state)
print("Initial state shape:", state.shape)
print("Action space:", env.action_space)

# Sample a valid random action
action = env.action_space.sample()

print("Action:", action)

# Execute one environment step
next_state, reward, done, truncated, info = env.step(action)

print("Next state:", next_state)
print("Next state shape:", next_state.shape)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

env.close()

print("Enhanced environment step completed successfully.")