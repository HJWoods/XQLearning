import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import matplotlib.pyplot as plt

# ML and RL libraries
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Conversion and visualization
from pytorch_to_polyc import convert_pytorch_model

# Set random seeds for reproducibility
def set_seeds(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(SmallCNN, self).__init__(observation_space, features_dim)

        # Define a smaller CNN with fewer layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 42, 42)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 21, 21)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 11, 11)
            nn.ReLU()
        )

        # Compute correct flatten size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 84, 84)  # A dummy observation
            cnn_output = self.cnn(sample_input)
            print("CNN output shape:", cnn_output.shape)  # Print the actual shape
            n_flatten = cnn_output.view(1, -1).shape[1]  # Get the correct size
        
        self.linear = nn.Linear(n_flatten, features_dim)  # Connect CNN to fully connected layer

    def forward(self, observations):
        features = self.cnn(observations)
        print("Actual CNN output shape:", features.shape)  # Debug line
        features = features.view(observations.shape[0], -1)  # Flatten the tensor
        print("Flattened feature shape:", features.shape)  # Debug line
        return self.linear(features)





import gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.preprocessing import is_image_space

# Set random seeds for reproducibility
def set_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class BreakoutDQNAgent:
    """DQN Agent for Breakout-v4 with a smaller model."""
    
    def __init__(self, env_name='Breakout-v4', learning_rate=1e-4, 
                 buffer_size=50000, learning_starts=1000, 
                 batch_size=32, tau=1.0, gamma=0.99, 
                 train_freq=4, gradient_steps=1):
        set_seeds()
        
        # Create environment
        env = gym.make(env_name, render_mode='rgb_array')

        # Convert to grayscale and resize to 42x42 to work with `MlpPolicy`
        if is_image_space(env.observation_space):
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
            env = gym.wrappers.ResizeObservation(env, 42)

        self.vec_env = DummyVecEnv([lambda: env])
        
        # Use a smaller MLP model instead of CNN
        policy_kwargs = dict(
            net_arch=[128, 128]  # Smaller fully connected layers
        )

        # Initialize DQN with MLP
        self.model = DQN(
            "MlpPolicy",  # ✅ Much smaller than "CnnPolicy"
            self.vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
    
    def train(self, total_timesteps=100000):
        """Train the agent."""
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save("models/breakout_dqn")
    
    def test(self, num_episodes=3):
        """Test the trained agent."""
        env = gym.make('Breakout-v4', render_mode='human')
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, 42)

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        env.close()

    def convert_to_polyc(self):
        """Convert trained model to PolyC format."""
        policy_net = self.model.policy.q_net
        
        config = {
            "input_vars": {
                "screen_input": (84, 84, 3)  # RGB image input
            },
            "output_vars": {
                "action": 1  # Discrete action space for Breakout
            },
            "constants": {},
            "env_vars": {},
            "constraints": [],
            "goals": ["maximize total_reward"],
            "model_name": "BreakoutNeuralNetwork"
        }
        
        try:
            polyc_code = convert_pytorch_model(
                policy_net,
                input_vars=config["input_vars"],
                output_vars=config["output_vars"],
                model_name=config["model_name"],
                constraints=config["constraints"],
                goals=config["goals"],
                constants=config["constants"],
                env_vars=config["env_vars"]
            )
            
            with open("breakout_model.polyc", "w") as f:
                f.write(polyc_code)
            
            with open("breakout_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print("Model converted to PolyC and saved to 'breakout_model.polyc'")
            print("Configuration saved to 'breakout_config.json'")
            
            return polyc_code
        
        except Exception as e:
            print(f"Error converting to PolyC: {e}")
            return None



def main():
    """Main function to train and test Breakout DQN agent."""
    agent = BreakoutDQNAgent()
    agent.train(total_timesteps=1000)
    print("Training complete.")
    agent.convert_to_polyc()
    agent.test(num_episodes=3)


if __name__ == "__main__":
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    main()
