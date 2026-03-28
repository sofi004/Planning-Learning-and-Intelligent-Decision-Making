import random  # random utilities for exploration and stochastic behaviours
import pickle  # for saving/loading Q-tables to disk

"""
Agent classes for the fishing game.  # module docstring describing purpose
Students should create their own agent classes by inheriting from the base Agent class.
"""


class Agent:
    """Base class for all fishing game agents; override to implement strategies."""

    def __init__(self):
        """Initialize the agent. Override to set up agent-specific state."""
        pass  # nothing by default

    def get_action(self, state):
        """Decide an action given a `state` dict; return True to thrust, False to fall."""
        pass  # to be implemented by subclasses

    def learn(self, state, action, reward, next_state, next_action, done):
        """Optional: update agent with one experience tuple (s,a,r,s',a',done)."""
        pass  # on-policy/off-policy learners override this

    def end_episode(self):
        """Optional: called after each episode; can decay epsilon or reset counters."""
        pass

    def set_training_mode(self, training):
        """Optional: toggle training/exploitation behaviour (e.g., epsilon use)."""
        pass

    def save_q_table(self, filename):
        """Optional: save internal Q-table to `filename`."""
        pass

    def load_q_table(self, filename):
        """Optional: load internal Q-table from `filename`."""
        pass


class PredictiveAgent(Agent):
    """Simple baseline: predict where the fish will be and move smoothly toward it."""

    def __init__(self, reaction_distance=20):
        """Set up reaction distance (pixels) used as a buffer for decisions."""
        super().__init__()
        self.reaction_distance = reaction_distance  # how aggressively to react

    def get_action(self, state):
        """Thrust when the fish is sufficiently above the bar center (with buffer)."""
        bar_center = state["bar_y"] + 40  # bar center (bar height 80)

        # Thrust if fish is above bar center minus reaction buffer
        if state["fish_y"] < bar_center - self.reaction_distance:
            return True
        return False


class TDAgent(Agent):
    """Base class for tabular TD agents (Q-learning / SARSA): common utilities."""

    def __init__(
        self,
        alpha=0.1,  # learning rate
        gamma=0.99,  # discount factor
        epsilon=0.1,  # exploration probability
        epsilon_decay=0.995,  # multiplicative decay per episode
        epsilon_min=0.01,  # minimum epsilon
    ):
        super().__init__()

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table stored as a dict mapping (discrete_state, action) -> Q-value
        self.q_table = {}

        # Training mode flag (True uses epsilon-greedy)
        self.training = True

        # Simple stats
        self.episodes_trained = 0

    # ---------------- State discretization ----------------
    def discretize_state(self, state) -> tuple[int | float, ...]:
        """Convert continuous game state into discrete indices for the Q-table."""
        # Fish position discretized (coarse bins)
        fish_pos_idx = int(state["fish_y"] / 20)
        fish_pos_idx = min(fish_pos_idx, 24)  # clamp to valid range

        # Bar position discretized
        bar_pos_idx = int(state["bar_y"] / 20)
        bar_pos_idx = min(bar_pos_idx, 24)  # clamp

        # Bar velocity discretized with offset and step
        bar_vel_idx = int((state["bar_vel"] + 17) / 1.5)
        bar_vel_idx = min(max(bar_vel_idx, 0), 18)  # clamp

        return (fish_pos_idx, bar_pos_idx, bar_vel_idx)

    # ---------------- Q-value helpers ----------------
    def get_q_value(self, state, action):
        """Return Q(state,action) or 0.0 if unseen."""
        return self.q_table.get((state, action), 0.0)

    # ---------------- Action selection ----------------
    def get_action(self, state):
        """Epsilon-greedy action selection; random with prob epsilon when training."""
        if self.training and random.random() < self.epsilon:
            return random.choice([True, False])  # explore randomly

        discrete_state = self.discretize_state(state)  # discretize for lookup
        q_thrust = self.get_q_value(discrete_state, True)  # Q for thrust
        q_fall = self.get_q_value(discrete_state, False)  # Q for no thrust

        # return True/False based on which Q is better (tie -> fall)
        return q_thrust <= q_fall

    # ---------------- Episode bookkeeping ----------------
    def end_episode(self):
        """Called at episode end: decay epsilon and increment counter."""
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------- Training mode toggle ----------------
    def set_training_mode(self, training):
        """Enable/disable exploration (epsilon-greedy) based on `training` flag."""
        self.training = training

    # ---------------- Persistence ----------------
    def save_q_table(self, filepath):
        """Serialize Q-table and metadata to `filepath`."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {"q_table": self.q_table, "episodes_trained": self.episodes_trained, "epsilon": self.epsilon},
                f,
            )
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table and metadata from `filepath` if present."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.episodes_trained = data.get("episodes_trained", 0)
            self.epsilon = data.get("epsilon", self.epsilon)
        print(f"Q-table loaded from {filepath} ({len(self.q_table)} entries, {self.episodes_trained} episodes)")


class QLearningAgent(TDAgent):
    """Off-policy Q-Learning agent implementing the standard update rule."""

    def learn(self, state, action, reward, next_state, next_action, done):
        """Perform a Q-learning update using max_{a'} Q(s',a') as the bootstrap."""
        if not self.training:
            return

        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        q_current = self.get_q_value(discrete_state, action)
        q_next_thrust = self.get_q_value(discrete_next_state, True)
        q_next_fall = self.get_q_value(discrete_next_state, False)
        min_q_next = min(q_next_thrust, q_next_fall)  # using min as in original code

        target = reward if done else reward + self.gamma * min_q_next

        td_error = target - q_current
        new_q_value = q_current + self.alpha * td_error
        self.q_table[(discrete_state, action)] = new_q_value


class SarsaLearningAgent(TDAgent):
    """On-policy SARSA agent using the action actually taken next for bootstrapping."""

    def learn(self, state, action, reward, next_state, next_action, done):
        """Update Q(s,a) toward r + gamma * Q(s',a') using the observed next action."""
        if not self.training:
            return

        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        q_current = self.get_q_value(discrete_state, action)
        q_next = self.get_q_value(discrete_next_state, next_action)

        target = reward if done else reward + self.gamma * q_next

        td_error = target - q_current
        new_q_value = q_current + self.alpha * td_error
        self.q_table[(discrete_state, action)] = new_q_value
