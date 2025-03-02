import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import json
import time
import matplotlib.pyplot as plt
from collections import deque
from pytorch_to_polyc import convert_pytorch_model

# Check if CUDA is available, but CartPole is simple enough to run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for models
os.makedirs("models", exist_ok=True)

class CartPoleModel(nn.Module):
    """Simple neural network for CartPole."""
    
    def __init__(self, input_dim=4, hidden_dim=24, output_dim=1):
        super(CartPoleModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)



class ReplayBuffer:
    """Experience replay buffer for storing transitions."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for CartPole."""
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        
        # Q-Networks
        self.policy_net = CartPoleModel(state_dim, 24, action_dim).to(device)
        self.target_net = CartPoleModel(state_dim, 24, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # For tracking progress
        self.rewards = []
        self.steps = []
        self.avg_rewards = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy during training."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, num_episodes=500, target_update=10, render_freq=1):
        """Train the agent on CartPole."""
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            
            # Create rendering environment for visualization
            if episode % render_freq == 0:
                render_env = gym.make('CartPole-v1', render_mode='human')
                render_state, _ = render_env.reset()
            
            while not (done or truncated):
                action = self.select_action(state)
                
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Modify reward to encourage balancing
                modified_reward = reward
                if done:
                    # Penalize falling over
                    modified_reward = -1
                
                # Store transition in memory
                self.memory.add(state, action, modified_reward, next_state, done or truncated)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Perform learning if enough samples in memory
                if len(self.memory) > self.batch_size:
                    self._train_step()
                
                # Render occasionally
                if episode % render_freq == 0:
                    render_action = self.select_action(render_state, training=False)
                    render_state, _, render_done, render_truncated, _ = render_env.step(render_action)
                    if render_done or render_truncated:
                        render_state, _ = render_env.reset()
            
            # End of episode
            self.rewards.append(episode_reward)
            self.steps.append(episode_steps)
            self.update_epsilon()
            
            # Update target network periodically
            if episode % target_update == 0:
                self.update_target_network()
            
            # Calculate average reward
            avg_reward = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
            self.avg_rewards.append(avg_reward)
            
            # Save model periodically
            if episode % 100 == 0:
                self.save_model(f"models/cartpole_dqn_episode_{episode}.pt")
                
                # Plot learning curve
                self._plot_learning_curve()
            
            # Close render environment if used
            if episode % render_freq == 0:
                render_env.close()
            
            print(f"Episode {episode}: Reward = {episode_reward}, Steps = {episode_steps}, Avg = {avg_reward:.2f}, Epsilon = {self.epsilon:.4f}")
            
            # Early stopping if we've mastered the game (average reward > 195 over 100 episodes)
            if avg_reward > 195 and len(self.rewards) >= 100:
                print("Environment solved! Stopping training.")
                self.save_model("models/cartpole_dqn_final.pt")
                break
        
        # Final save
        self.save_model("models/cartpole_dqn_final.pt")
        return self.rewards, self.steps
    
    def _train_step(self):
        """Perform one step of training on a batch from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute Q values for current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute Q values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping
        self.optimizer.step()
    
    def _plot_learning_curve(self):
        """Plot the learning curve."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_rewards)
        plt.title('Average Reward (last 100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig('cartpole_learning_curve.png')
        plt.close()
    
    def save_model(self, filepath):
        """Save the model."""
        torch.save(self.policy_net, filepath)
    
    def load_model(self, filepath):
        """Load a saved model."""
        self.policy_net = torch.load(filepath)
        self.target_net.load_state_dict(self.policy_net.state_dict())


def test_agent(model, env_name='CartPole-v1', episodes=5, use_policy=False):
    """Test a trained agent."""
    env = gym.make(env_name, render_mode='human')
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_policy:
                    # Continuous policy model
                    continuous_action = model(state_tensor).item()
                    # Convert to discrete action for CartPole environment
                    action = 0 if continuous_action < 0 else 1
                else:
                    # DQN model
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            
            # Small delay for better visualization
            time.sleep(0.01)
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total reward = {total_reward}")
    
    env.close()
    return total_rewards


def convert_to_polyc(model_path=None, model=None):
    """Convert a trained model to PolyC format.
    
    Args:
        model_path: Path to the saved model file (optional)
        model: Direct model object (optional)
        
    Note: Provide either model_path OR model
    """
    # Load the policy model
    if model is None and model_path is not None:
        try:
            # Try loading with weights_only=False first
            policy_model = torch.load(model_path, weights_only=False)
        except Exception as e:
            print(f"Error loading model with weights_only=False: {e}")
    else:
        policy_model = model
    
    # Define configuration for PolyC
    config = {
        "input_vars": {
            "cart_pos": 1,      # Cart position
            "cart_vel": 1,      # Cart velocity
            "pole_angle": 1,    # Pole angle
            "pole_vel": 1       # Pole angular velocity
        },
        "output_vars": {
            "cart_force": 1     # Continuous force to apply (-1 to 1)
        },
        "constants": {
            "max_force": 1.0
        },
        "env_vars": {},
        "constraints": [
            "cart_force >= -1.0",
            "cart_force <= 1.0"
        ],
        "goals": [
            "min abs(pole_angle)"  # Minimize pole angle (keep upright)
        ],
        "model_name": "CartPoleNeuralNetwork"
    }
    
    # Generate PolyC code
    polyc_code = convert_pytorch_model(
        policy_model,
        input_vars=config["input_vars"],
        output_vars=config["output_vars"],
        model_name=config["model_name"],
        constraints=config["constraints"],
        goals=config["goals"],
        constants=config["constants"],
        env_vars=config["env_vars"]
    )
    
    # Save PolyC code
    with open("cartpole_model.polyc", "w") as f:
        f.write(polyc_code)
    
    # Save configuration
    with open("cartpole_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Model converted to PolyC and saved to 'cartpole_model.polyc'")
    print("Configuration saved to 'cartpole_config.json'")
    return polyc_code


def main():
    """Main function to train and test CartPole agent."""
    # Create CartPole environment
    env = gym.make('CartPole-v1')
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize DQN agent
    state_dim = 4  # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    action_dim = 2  # [left, right]
    
    # Create and train agent
    agent = DQNAgent(state_dim, action_dim)
    
    # Choose whether to train or load a pre-trained model
    train_new = True
    num_episodes = 50000
    if train_new:
        print("Training new agent...")
        rewards, steps = agent.train(env, num_episodes=num_episodes, render_freq=50)
        print("Training completed!")
    else:
        # Load pre-trained model if available
        try:
            agent.load_model("models/cartpole_dqn_final.pt")
            print("Loaded pre-trained model")
        except FileNotFoundError:
            print("No pre-trained model found. Training new agent...")
            rewards, steps = agent.train(env, num_episodes=num_episodes, render_freq=50)
    
    # Save the policy model
    torch.save(agent.policy_net, "models/cartpole_policy_model.pt")
    print("Policy model saved to 'models/cartpole_policy_model.pt'")
    
    # Convert to PolyC
    print("\nConverting to PolyC format...")
    convert_to_polyc("models/cartpole_policy_model.pt")
    
    # Test the DQN model
    print("\nTesting DQN model...")
    test_agent(agent.policy_net, episodes=3)
    
    # Clean up
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    main()