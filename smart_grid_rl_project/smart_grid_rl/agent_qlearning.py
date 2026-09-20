"""
agent_qlearning.py
------------------------------------------------------------------
NEW FILE -- tabular Q-Learning agent (per the abstract's chosen
algorithm), adapted to work with this environment's Gymnasium spaces:

  observation_space = Box(8,)       continuous, normalized to [0,1]
  action_space      = MultiDiscrete([3,3,3,3,3])   5 sub-actions

Tabular Q-learning needs discrete state and a single discrete action
index, so this module:
  1. discretizes the continuous Box observation into a fixed number of
     bins per dimension (state used as a dict key, not a dense table --
     this scales fine since only visited states get an entry)
  2. flattens the 5-dimensional MultiDiscrete action into one integer
     0..242 (and decodes it back to the array env.step() expects)

The abstract also names DQN as a stretch upgrade -- swapping this agent
for a neural-network Q-function later only requires replacing this
file; env.py, reward.py, and everything else stay the same since they
already speak Gymnasium's standard Box/MultiDiscrete interface.
------------------------------------------------------------------
"""

import pickle
from collections import defaultdict
from itertools import product

import numpy as np


def flatten_action_space(nvec):
    """All combinations of a MultiDiscrete's nvec, index <-> array."""
    combos = list(product(*[range(n) for n in nvec]))
    return combos  # combos[i] is the array for flattened action i


def discretize_state(state: np.ndarray, n_bins: int = 4) -> tuple:
    """Continuous [0,1] values -> bin indices; already-binary appliance
    flags (0.0/1.0) round cleanly regardless of n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior edges
    return tuple(int(np.digitize(v, bins)) for v in state)


class QLearningAgent:
    def __init__(self, nvec, n_bins: int = 4, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.997):
        self.action_combos = flatten_action_space(nvec)
        self.n_actions = len(self.action_combos)
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def _state_key(self, obs: np.ndarray) -> tuple:
        return discretize_state(obs, self.n_bins)

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        key = self._state_key(obs)
        if (not greedy) and (np.random.rand() < self.epsilon):
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = int(np.argmax(self.q_table[key]))
        return np.array(self.action_combos[action_idx])

    def act_index(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Same as act(), but returns the flattened index (needed by update())."""
        key = self._state_key(obs)
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def action_from_index(self, idx: int) -> np.ndarray:
        return np.array(self.action_combos[idx])

    def update(self, obs, action_idx: int, reward: float, next_obs, done: bool):
        state_key = self._state_key(obs)
        current_q = self.q_table[state_key][action_idx]
        target = reward
        if not done:
            next_key = self._state_key(next_obs)
            target += self.gamma * np.max(self.q_table[next_key])
        self.q_table[state_key][action_idx] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "n_bins": self.n_bins,
                         "action_combos": self.action_combos}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_bins = data["n_bins"]
        self.action_combos = data["action_combos"]
        self.n_actions = len(self.action_combos)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data["q_table"])
