import gym
import numpy as np
import random
import time
from math import log2
# TODO: global explanations based on comparing policies
# TODO: use method from "explaining the explainer" (source code publically available), it's very similar but works for continuous state/action spaces

class QDecisionTreeAgent:
    def __init__(self, env, action_encoding, state_component_names,
                 state_representation_func,
                 action_selection_func=None,
                 alpha=0.1, gamma=0.9999, epsilon=1.0, epsilon_decay=0.00001, n_episodes=1000):
        """
        Initialize the agent.
        
        Parameters:
          env: A Gym environment.
          action_encoding: A list mapping action indices to human-readable actions.
          state_component_names: A list of names for the state components.
          state_representation_func: A function that takes (state, env) and returns a state representation tuple.
          action_selection_func: A function that takes (q_values, epsilon, n_actions) and returns an action.
                                 If None, a default epsilon–greedy selection is used.
          alpha: Learning rate.
          gamma: Discount factor.
          epsilon: Initial exploration probability.
          epsilon_decay: Decay factor for epsilon.
          n_episodes: Number of episodes for training.
        """
        self.env = env
        self.action_encoding = action_encoding
        self.state_component_names = state_component_names
        self.n_actions = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.Q = {}  # Q-table stored as a dictionary
        self.decision_tree = None
        self.examples = None

        self.state_representation_func = state_representation_func
        self.action_selection_func = action_selection_func

    def get_state_key(self, state, env=None):
        if env is None:
            env = self.env
        return self.state_representation_func(state, env)

    def get_Q(self, state_key):
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.n_actions)
        return self.Q[state_key]

    def train(self):
        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                state_key = self.get_state_key(state)
                q_values = self.get_Q(state_key)
                
                if self.action_selection_func is None:
                    if random.uniform(0, 1) < self.epsilon:
                        action = random.choice(range(self.n_actions))
                    else:
                        action = np.argmax(q_values)
                else:
                    action = self.action_selection_func(q_values, self.epsilon, self.n_actions)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                next_state_key = self.get_state_key(next_state)
                next_q_values = self.get_Q(next_state_key)
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * np.max(next_q_values)
                self.Q[state_key][action] += self.alpha * (td_target - self.Q[state_key][action])
                
                total_reward += reward
                state = next_state
            
            self.epsilon = max(0.01, self.epsilon * (1 - self.epsilon_decay))
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1} finished. Epsilon: {self.epsilon:.2f}, Total reward: {total_reward}")
        print("Training complete!")
        print("Trained Q-table:")
        print(self.Q)

    # ID3-based decision tree methods, see https://en.wikipedia.org/wiki/ID3_algorithm
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
        for state_key, q_values in self.Q.items():
            if np.all(q_values == 0):
                label = "no action (terminal state)"
            else:
                best_action = np.argmax(q_values)
                label = f"{self.action_encoding[best_action]} (Q-value: {q_values[best_action]:.4f})"
            example = {}
            for name, value in zip(self.state_component_names, state_key):
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

    def demonstrate(self, demo_env=None, pause=1):
        if demo_env is None:
            demo_env = self.env
        state, _ = demo_env.reset()
        time.sleep(pause)
        done = False
        while not done:
            state_key = self.get_state_key(state, demo_env)
            q_values = self.get_Q(state_key)
            action = np.argmax(q_values)
            state, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated
            demo_env.render()
            print(f"In state {state_key} (index {state}), action '{self.action_encoding[action]}' selected (Q-values: {q_values}).")
            time.sleep(pause)
        print("Demonstration finished!")
        demo_env.close()

    def _simulate_trace(self, initial_state, counterfactual_first=False, counterfactual_actions=None, max_steps=50, render_trace=False, render_pause=0.25, trace_label="Trace"):
        """
        Simulates a trace starting from a given initial state.
        If counterfactual_first is True, uses the provided counterfactual_actions (a list of actions)
        for the first few steps (as many as provided), then follows the greedy policy thereafter.
        
        The counterfactual_actions can be provided as indices or asstrings; if strings,
        they are decoded using self.action_encoding.
        
        Returns a tuple: (trace_string, cumulative_reward, number_of_steps).
        Optionally renders the trace if render_trace is True.
        """
        simulation_trace = []
        cumulative_reward = 0
        steps = 0

        # Normalize counterfactual_actions to a list of indices.
        if counterfactual_first:
            if counterfactual_actions is None:
                raise ValueError("counterfactual_first is True but no counterfactual_actions were provided.")
            if not isinstance(counterfactual_actions, list):
                counterfactual_actions = [counterfactual_actions]
            # Convert names to indices if necessary.
            normalized_cf = []
            for a in counterfactual_actions:
                if isinstance(a, str):
                    try:
                        idx = self.action_encoding.index(a)
                    except ValueError:
                        raise ValueError(f"Action '{a}' not found in action_encoding.")
                    normalized_cf.append(idx)
                else:
                    normalized_cf.append(a)
            counterfactual_actions = normalized_cf

        # Create a fresh instance of the environment with rendering if requested.
        env_id = self.env.spec.id
        try:
            if render_trace:
                sim_env = gym.make(env_id, render_mode="human", is_slippery=self.env.unwrapped.__dict__.get("is_slippery", None))
            else:
                sim_env = gym.make(env_id, is_slippery=self.env.unwrapped.__dict__.get("is_slippery", None))
        except Exception:
            sim_env = gym.make(env_id, render_mode="human" if render_trace else None)
        actual_initial_state, _ = sim_env.reset()
        if actual_initial_state != initial_state:
            simulation_trace.append(f"WARNING: Requested initial state {initial_state} but reset returned {actual_initial_state}. Using {actual_initial_state} for simulation.")
        current_state = actual_initial_state
        simulation_trace.append(f"{trace_label} starting state: {current_state} with representation {self.get_state_key(current_state, sim_env)}")
        if render_trace:
            sim_env.render()
            print(f"{trace_label} rendering initial state: {current_state} with representation {self.get_state_key(current_state, sim_env)}")
            time.sleep(render_pause)
        while steps < max_steps:
            state_key = self.get_state_key(current_state, sim_env)
            if state_key not in self.Q:
                simulation_trace.append(f"State {current_state} with representation {state_key} is unknown (no Q-value entry); stopping simulation.")
                break
            q_values = self.Q[state_key]
            if counterfactual_first and steps < len(counterfactual_actions):
                action = counterfactual_actions[steps]
                simulation_trace.append(f"Step {steps+1}: Using counterfactual action '{self.action_encoding[action]}'")
            else:
                action = np.argmax(q_values)
                simulation_trace.append(f"Step {steps+1}: Greedy action '{self.action_encoding[action]}' with Q-value {q_values[action]:.4f}")
            next_state, reward, terminated, truncated, info = sim_env.step(action)
            cumulative_reward += reward
            simulation_trace.append(f" -> Next state: {next_state} with representation {self.get_state_key(next_state, sim_env)}, reward: {reward}")
            if render_trace:
                sim_env.render()
                print(f"{trace_label} rendering state: {next_state} with representation {self.get_state_key(next_state, sim_env)}")
                time.sleep(render_pause)
            steps += 1
            if terminated or truncated:
                simulation_trace.append("Terminal state reached.")
                break
            current_state = next_state
        if render_trace:
            time.sleep(render_pause)
        sim_env.close()
        return "\n".join(simulation_trace), cumulative_reward, steps

    def present_counterfactual_comparison(self, initial_state, counterfactual_actions, max_steps=50, render_traces=False, render_pause=0.25):
        report_lines = []
        report_lines.append("=== Optimal Policy Trace ===")
        optimal_trace, optimal_reward, optimal_steps = self._simulate_trace(initial_state, 
                                                                           counterfactual_first=False, 
                                                                           counterfactual_actions=None, 
                                                                           max_steps=max_steps,
                                                                           render_trace=render_traces,
                                                                           render_pause=render_pause,
                                                                           trace_label="Optimal Policy Trace")
        report_lines.append(optimal_trace)
        report_lines.append("")
        report_lines.append("=== Counterfactual Trace ===")
        cf_trace, cf_reward, cf_steps = self._simulate_trace(initial_state, 
                                                             counterfactual_first=True, 
                                                             counterfactual_actions=counterfactual_actions, 
                                                             max_steps=max_steps,
                                                             render_trace=render_traces,
                                                             render_pause=render_pause,
                                                             trace_label="Counterfactual Trace")
        report_lines.append(cf_trace)
        report_lines.append("")
        report_lines.append("=== Comparison ===")
        report_lines.append(f"Optimal policy trace: Cumulative reward = {optimal_reward}, Steps = {optimal_steps}")
        report_lines.append(f"Counterfactual trace: Cumulative reward = {cf_reward}, Steps = {cf_steps}")
        if optimal_reward > cf_reward:
            report_lines.append("Conclusion: The optimal policy yields a higher terminal reward, therefore it is preferable.")
        elif optimal_reward < cf_reward:
            report_lines.append("Conclusion: The counterfactual sequence would yield a higher terminal reward, therefore it is preferable.")
        else:
            if optimal_steps < cf_steps:
                report_lines.append("Terminal rewards are equal, but the optimal policy trace is shorter; hence, it is preferable.")
            elif optimal_steps > cf_steps:
                report_lines.append("Terminal rewards are equal, but the counterfactual trace is shorter; hence, it is preferable.")
            else:
                report_lines.append("Both terminal reward and path length are equivalent. The counterfactual policy is equally good.")
        return "\n".join(report_lines)

if __name__ == "__main__":
    # patch for compatibility if necessary,
    # seems to be some issue with FrozenLake-v1 and numpy 1.20.0
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    def default_state_representation(state, env):
        grid = env.unwrapped.desc
        n_rows, n_cols = grid.shape
        row = state // n_cols
        col = state % n_cols
        def get_cell(r, c):
            if 0 <= r < n_rows and 0 <= c < n_cols:
                cell = grid[r, c]
                cell_str = cell.decode("utf-8") if isinstance(cell, bytes) else cell
            else:
                cell_str = 'X'
            if cell_str == 'X':
                return "Out of Bounds"
            if cell_str == 'S':
                return "Start"
            if cell_str == 'F':
                return "Floor"
            if cell_str == 'H':
                return "Hole"
            if cell_str == 'G':
                return "Goal"
        current = get_cell(row, col)
        up    = get_cell(row - 1, col)
        down  = get_cell(row + 1, col)
        left  = get_cell(row, col - 1)
        right = get_cell(row, col + 1)
        return (current, up, down, left, right)

    is_slippery = False
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    
    # Example custom action selection function (not used during training here):
    def top_two_selection(q_values, epsilon, n_actions):
        if random.uniform(0,1) < epsilon:
            return random.choice(range(n_actions))
        sorted_actions = np.argsort(q_values)[::-1]
        top_two = sorted_actions[:2]
        return random.choice(top_two)
    
    agent = QDecisionTreeAgent(
        env=env,
        action_encoding=["left", "down", "right", "up"],
        state_component_names=["current", "up", "down", "left", "right"],
        state_representation_func=default_state_representation,
        action_selection_func=None,
        alpha=0.1,
        gamma=0.9999,
        epsilon=1.0,
        epsilon_decay=0.0000001,
        n_episodes=5000
    )

    agent.train()
    agent.build_decision_tree()
    agent.write_decision_tree("decision_tree.txt")
    
    #demo_env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode="human")
    #agent.demonstrate(pause=1, demo_env=demo_env)
    #demo_env.close()
    
    # For FrozenLake, the initial state is usually 0.
    initial_state, _ = env.reset()
    # Provide a counterfactual sequence. For example, ["down", "left"].
    # These can be provided as names; they will be decoded.
    cf_actions = ["right", "right", "right", "right"]
    #cf_actions = ["right", "right", "right", "right"]
    cf_comparison = agent.present_counterfactual_comparison(initial_state=initial_state, counterfactual_actions=cf_actions, max_steps=20, render_traces=True, render_pause=0.25)
    print("Counterfactual Comparison Trace:")
    print(cf_comparison)
