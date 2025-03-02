import gym
import numpy as np
import random
import time
from math import log2
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward network for DQN.
class DQNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Learning agent with decision tree capabilities.
class DeepQDecisionTreeAgent:
    def __init__(self, env, action_encoding, state_component_names,
                 state_representation_func,
                 state_tensor_func=None,
                 action_selection_func=None,
                 gamma=0.99, epsilon=1.0, epsilon_decay=1e-3, epsilon_min=0.01,
                 n_episodes=500, batch_size=64, memory_capacity=10000,
                 target_update_frequency=10, hidden_dim=64, learning_rate=1e-3):
        """
        Parameters:
          env: A Gym environment.
          action_encoding: List mapping action indices to human-readable actions.
          state_component_names: Names for state components (used in decision trees).
          state_representation_func: Function to convert state into a discrete tuple for explanations.
          state_tensor_func: Function to convert a state into a tensor (default assumes continuous state).
          action_selection_func: Optional custom action selection function.
          gamma: Discount factor.
          epsilon: Initial exploration probability.
          epsilon_decay: Decay factor for epsilon.
          epsilon_min: Minimum exploration probability.
          n_episodes: Number of training episodes.
          batch_size: Mini-batch size for training.
          memory_capacity: Maximum replay memory size.
          target_update_frequency: How often (in episodes) to update the target network.
          hidden_dim: Hidden layer size for the network.
          learning_rate: Optimizer learning rate.
        """
        self.env = env
        self.action_encoding = action_encoding
        self.state_component_names = state_component_names
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.target_update_frequency = target_update_frequency

        self.state_representation_func = state_representation_func
        if state_tensor_func is None:
            # For continuous states (e.g. CartPole, shape=(4,)).
            self.state_dim = env.observation_space.shape[0]
            self.state_tensor_func = lambda s: torch.FloatTensor(s).unsqueeze(0)
        else:
            self.state_tensor_func = state_tensor_func

        self.action_selection_func = action_selection_func

        # Initialize the online and target networks.
        self.model = DQNModel(self.state_dim, hidden_dim, self.n_actions)
        self.target_model = DQNModel(self.state_dim, hidden_dim, self.n_actions)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay memory: stores (state, action, reward, next_state, done)
        self.memory = []
        # For decision tree explanations, we store visited (discretized) states.
        self.visited_states = set()

        self.decision_tree = None
        self.examples = None

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        # Epsilon-greedy action selection.
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_tensor = self.state_tensor_func(state)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def store_transition(self, transition):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def train(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                # Discretize the continuous state for decision tree purposes.
                self.visited_states.add(tuple(self.state_representation_func(state, self.env)))
                action = (self.select_action(state) if self.action_selection_func is None 
                          else self.action_selection_func(self.model(self.state_tensor_func(state)), self.epsilon, self.n_actions))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                self.store_transition((state, action, reward, next_state, done))
                state = next_state

                if len(self.memory) >= self.batch_size:
                    batch = self.sample_memory()
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states_tensor = torch.cat([self.state_tensor_func(s) for s in states])
                    actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states_tensor = torch.cat([self.state_tensor_func(s) for s in next_states])
                    dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

                    q_values = self.model(states_tensor).gather(1, actions_tensor)
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states_tensor).max(1)[0].unsqueeze(1)
                    td_target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
                    loss = self.loss_fn(q_values, td_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
            if (episode + 1) % self.target_update_frequency == 0:
                self.update_target_network()
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}: Total reward = {total_reward}, Epsilon = {self.epsilon:.3f}")
        print("Training complete!")

    def get_Q_values(self, state):
        state_tensor = self.state_tensor_func(state)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
        return q_values

    # --- Decision tree (ID3) methods ---
    def _entropy(self, examples):
        counts = {}
        for ex in examples:
            label = ex["label"]
            counts[label] = counts.get(label, 0) + 1
        total = len(examples)
        ent = 0.0
        for count in counts.values():
            p = count / total
            ent -= p * log2(p)
        return ent

    def _info_gain(self, examples, attribute):
        base_ent = self._entropy(examples)
        subsets = {}
        for ex in examples:
            value = ex[attribute]
            subsets.setdefault(value, []).append(ex)
        total = len(examples)
        weighted_ent = 0.0
        for subset in subsets.values():
            weighted_ent += (len(subset) / total) * self._entropy(subset)
        return base_ent - weighted_ent

    def _build_tree(self, examples, attributes):
        labels = [ex["label"] for ex in examples]
        if len(set(labels)) == 1:
            return labels[0]
        if not attributes:
            return max(set(labels), key=labels.count)
        best_attr = max(attributes, key=lambda attr: self._info_gain(examples, attr))
        tree = {best_attr: {}}
        subsets = {}
        for ex in examples:
            val = ex[best_attr]
            subsets.setdefault(val, []).append(ex)
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        for val, subset in subsets.items():
            subtree = self._build_tree(subset, remaining_attrs)
            tree[best_attr][val] = subtree
        return tree

    def _tree_to_string(self, tree, indent=""):
        if not isinstance(tree, dict):
            return indent + "-> " + str(tree) + "\n"
        result = ""
        for attr, branches in tree.items():
            for val, subtree in branches.items():
                result += indent + f"if {attr} == {val}:\n"
                result += self._tree_to_string(subtree, indent + "    ")
        return result

    def build_decision_tree(self):
        examples = []
        for state_repr in self.visited_states:
            # For simplicity, we get Q-values by converting the discretized state back to a numpy array.
            q_values = self.get_Q_values(np.array(state_repr))
            if np.allclose(q_values, 0):
                label = "no action (terminal state)"
            else:
                best_action = np.argmax(q_values)
                label = f"{self.action_encoding[best_action]} (Q-value: {q_values[best_action]:.4f})"
            example = {}
            for name, value in zip(self.state_component_names, state_repr):
                example[name] = value
            example["label"] = label
            examples.append(example)
        self.examples = examples
        self.attributes = self.state_component_names.copy()
        self.decision_tree = self._build_tree(examples, self.attributes)
        return self.decision_tree

    def write_decision_tree(self, filename="decision_tree.txt"):
        if self.decision_tree is None:
            print("Decision tree not built. Call build_decision_tree() first.")
            return
        tree_str = self._tree_to_string(self.decision_tree)
        with open(filename, "w") as f:
            f.write(tree_str)
        print(f"Decision tree written to {filename}")

if __name__ == "__main__":
    # patch for compatibility if necessary,
    # seems to be some issue with FrozenLake-v1 and numpy 1.20.0
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
        
    # Use CartPole-v1, which has a continuous state space.
    env = gym.make("CartPole-v1")
    
    # For decision tree explanations, discretize the continuous state (round to 1 decimal).
    def cartpole_state_representation(state, env):
        # If state is a tuple (observation, info), extract the observation.
        if isinstance(state, tuple):
            state = state[0]
        state = np.array(state).flatten()  # Ensure the state is 1D.
        return tuple(np.round(state, 1).tolist())


    
    agent = DeepQDecisionTreeAgent(
        env=env,
        action_encoding=["left", "right"],
        state_component_names=["cart_pos", "cart_vel", "pole_angle", "pole_vel"],
        state_representation_func=cartpole_state_representation,
        action_selection_func=None,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=1e-3,
        epsilon_min=0.01,
        n_episodes=500,
        batch_size=64,
        memory_capacity=10000,
        target_update_frequency=10,
        hidden_dim=64,
        learning_rate=1e-3
    )
    
    agent.train()
    tree = agent.build_decision_tree()
    agent.write_decision_tree("decision_tree_cartpole.txt")
