import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
import gym
from collections import deque
import random
import os

# Ensure the output directory exists
os.makedirs('pruning_results', exist_ok=True)


def plot_pareto_front(results, pareto_front, method_name):
    """Plot the results and highlight the Pareto front."""
    plt.figure(figsize=(12, 8))
    
    # Plot all results
    ratios = [r[0] for r in results]
    params = [r[1] for r in results]
    rewards = [r[2] for r in results]
    
    # Create scatter plot
    scatter = plt.scatter(params, rewards, c=ratios, cmap='viridis', alpha=0.7, s=100, label='All Models')
    
    # Highlight Pareto front
    pareto_params = [p[1] for p in pareto_front]
    pareto_rewards = [p[2] for p in pareto_front]
    pareto_ratios = [p[0] for p in pareto_front]
    
    # Plot Pareto front line with hollow circles
    plt.plot(pareto_params, pareto_rewards, 'r--', linewidth=2, alpha=0.7)
    plt.scatter(pareto_params, pareto_rewards, s=150, facecolors='none', edgecolors='r', linewidths=2, label='Pareto Front')
    
    # Find the original unpruned model in results
    base_model_idx = ratios.index(0.0)
    
    # Highlight the unpruned model
    plt.scatter(params[base_model_idx], rewards[base_model_idx], s=200, 
                facecolors='yellow', edgecolors='red', linewidths=2, marker='*', 
                label='Unpruned Model (0%)')
    
    # Add percentage labels to all pruning points for clarity
    for i, (ratio, param, reward, _) in enumerate(results):
        label = f"{ratio*100:.0f}%" if ratio > 0 else "0%"
        plt.annotate(label, (param, reward), xytext=(5, 0), textcoords='offset points', 
                    fontsize=8, color='darkblue')
    
    # Add extra emphasis to Pareto front points
    for i, (param, reward, ratio) in enumerate(zip(pareto_params, pareto_rewards, pareto_ratios)):
        label = f"{ratio*100:.0f}%" if ratio > 0 else "0%"
        plt.annotate(label, (param, reward), xytext=(10, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold', color='darkred')
    
    plt.xlabel('Number of Non-zero Parameters', fontsize=12)
    plt.ylabel('Average Reward (across episodes)', fontsize=12)
    plt.title(f'Pareto Front of DQN Pruning: {method_name}', fontsize=14)
    plt.colorbar(scatter, label='Pruning Ratio')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'pruning_results/dqn_pruning_pareto_{method_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

def visualize_model(env, model, episodes=1):
    """Visualize model performance for a few episodes."""
    total_reward = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state]))
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
        print(f"  Visualization episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = total_reward / episodes
    print(f"  Average visualization reward: {avg_reward:.2f}")
    return avg_reward


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQN, self).__init__()
        
        layers = []
        dims = [state_dim] + hidden_dims + [action_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(nn.ReLU())
                
        self.model = nn.Sequential(*layers)
        print(f"Initialized DQN with {self.count_parameters()} parameters, layers: {hidden_dims}")
        
    def forward(self, x):
        return self.model(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_nonzero_parameters(self):
        """Count the number of non-zero parameters in the model."""
        return sum((p != 0).sum().item() for p in self.parameters())


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done))
        )
        
    def __len__(self):
        return len(self.buffer)



def find_pareto_front(env, env_name, base_model, method_name, prune_ratios=None, eval_episodes=100, x_samples=None, render_env=None, retraining=True, early_stop_reward=300):
    """Find the Pareto front of models with different pruning ratios using the specified method."""
    if prune_ratios is None:
        prune_ratios = np.linspace(0, 0.95, 20)  # From 0% to 95% pruning
    
    results = []
    
    # Evaluate the base model
    print(f"\nEvaluating base model over {eval_episodes} episodes...")
    base_reward = evaluate_dqn(env, base_model, episodes=eval_episodes)
    base_params = base_model.count_nonzero_parameters()
    results.append((0.0, base_params, base_reward, deepcopy(base_model)))
    
    # Apply different pruning ratios and evaluate
    for ratio in prune_ratios:
        if ratio == 0.0:  # Skip, already evaluated
            continue

        if retraining:
            pruned_model = apply_pruning_with_retraining(env, env_name, base_model, method_name, ratio, render_env=render_env, early_stop_reward=early_stop_reward)
        else:
            print(f"Testing {method_name} with pruning ratio {ratio*100:.1f}%...")
            if method_name == "Magnitude Pruning":
                pruned_model = apply_magnitude_pruning(base_model, ratio)
            elif method_name == "Threshold Pruning":
                pruned_model = apply_threshold_pruning(base_model, ratio)
            elif method_name == "Random Pruning":
                pruned_model = apply_random_pruning(base_model, ratio)
            elif method_name == "Structured Pruning":
                pruned_model = apply_structured_pruning(base_model, ratio)
            elif method_name == "Layer-wise Pruning":
                pruned_model = apply_layer_pruning(base_model, ratio)
            elif method_name == "Gradient-based Pruning":
                pruned_model = apply_gradient_based_pruning(base_model, ratio, x_samples)
            #elif method_name == "Weight Sharing":
            #    pruned_model = apply_weight_sharing(base_model, ratio)
            else:
                raise ValueError(f"Unknown pruning method: {method_name}")
        
        # Count actual non-zero parameters
        non_zero_params = pruned_model.count_nonzero_parameters()
        print(f"  Parameters after pruning: {non_zero_params} non-zero out of {pruned_model.count_parameters()} total")
        
        pruned_model = purification(pruned_model)
        non_zero_params = pruned_model.count_nonzero_parameters()
        total_params = pruned_model.count_parameters()
        print(f"Parameters after purification: {non_zero_params} non-zero out of {total_params} total ({non_zero_params/total_params*100:.1f}%)")


        # Evaluate performance over multiple episodes
        reward = evaluate_dqn(env, pruned_model, episodes=eval_episodes)
        print(f"  Average reward over {eval_episodes} episodes: {reward:.2f}")
        
        # Render this pruned model if render_env is provided
        if render_env is not None:
            print(f"  Visualizing {method_name} at {ratio*100:.1f}% pruning ratio...")
            # Play 1 episode and display it
            visualize_model(render_env, pruned_model, 1)
        
        results.append((ratio, non_zero_params, reward, pruned_model))
    
    # Filter for Pareto optimal points
    pareto_front = []
    results.sort(key=lambda x: x[1])  # Sort by parameter count
    
    max_reward = float('-inf')
    for result in results:
        if result[2] > max_reward:
            max_reward = result[2]
            pareto_front.append(result)
    
    return results, pareto_front

def apply_pruning_with_retraining(env, env_name, model, pruning_method, prune_ratio, 
                                  retrain_episodes=2500, gamma=0.99, epsilon_start=0.3, 
                                  epsilon_end=0.01,epsilon_decay=0.99, buffer_size=10000, 
                                  batch_size=64, render_env=None, early_stop_reward=300):
    """
    Apply pruning followed by retraining with mask enforcement.
    
    This approach maintains pruning masks separately for each parameter.
    """
    print(f"\nApplying {pruning_method} with pruning ratio {prune_ratio*100:.1f}%...")
    
    # Apply initial pruning based on the method
    if pruning_method == "Magnitude Pruning":
        pruned_model = apply_magnitude_pruning(model, prune_ratio)
    elif pruning_method == "Threshold Pruning":
        pruned_model = apply_threshold_pruning(model, prune_ratio)
    elif pruning_method == "Random Pruning":
        pruned_model = apply_random_pruning(model, prune_ratio)
    elif pruning_method == "Structured Pruning":
        pruned_model = apply_structured_pruning(model, prune_ratio)
    elif pruning_method == "Layer-wise Pruning":
        pruned_model = apply_layer_pruning(model, prune_ratio)
    elif pruning_method == "Gradient-based Pruning":
        state_dim = env.observation_space.shape[0]
        x_samples = torch.FloatTensor(np.random.rand(100, state_dim))
        pruned_model = apply_gradient_based_pruning(model, prune_ratio, x_samples)
    elif pruning_method == "Weight Sharing":
        pruned_model = apply_weight_sharing(model, prune_ratio)
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Count non-zero parameters before retraining
    non_zero_before = pruned_model.count_nonzero_parameters()
    total_params = pruned_model.count_parameters()
    print(f"Before retraining: {non_zero_before} non-zero parameters out of {total_params} total ({non_zero_before/total_params*100:.2f}%)")
    
    # Create pruning masks based on the current model state
    masks = {}
    for name, param in pruned_model.named_parameters():
        masks[name] = (param.data != 0).float()
    
    # Now retrain the pruned model while maintaining the masks
    if retrain_episodes > 0:
        print(f"Retraining pruned model for {retrain_episodes} episodes...")
        replay_buffer = ReplayBuffer(buffer_size)
        optimizer = torch.optim.Adam(pruned_model.parameters())
        criterion = nn.MSELoss()
        
        epsilon = epsilon_start
        episode_rewards = []
        
        for episode in range(retrain_episodes):
            current_state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(np.array([current_state]))
                        q_values = pruned_model(state_tensor)
                        action = q_values.argmax().item()
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Store transition in replay buffer
                replay_buffer.push(current_state, action, reward, next_state, done)
                current_state = next_state
                
                # Start training when enough samples are available
                if len(replay_buffer) > batch_size:
                    # Sample mini-batch
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                    
                    # Compute Q values
                    q_values = pruned_model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    
                    # Compute target Q values
                    with torch.no_grad():
                        next_q_values = pruned_model(batch_next_states).max(1)[0]
                        targets = batch_rewards + gamma * next_q_values * (1 - batch_dones)
                    
                    # Compute loss and update parameters
                    loss = criterion(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Apply gradients while respecting the masks
                    with torch.no_grad():
                        for name, param in pruned_model.named_parameters():
                            if name in masks:
                                # Apply mask to gradients
                                param.grad.mul_(masks[name])
                                
                    optimizer.step()
                    
                    # Force zero weights to stay zero
                    with torch.no_grad():
                        for name, param in pruned_model.named_parameters():
                            if name in masks:
                                # Apply mask to parameters
                                param.data.mul_(masks[name])


            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"  Episode {episode+1}/{retrain_episodes}, Reward: {episode_reward}, Avg Reward (last 10): {avg_reward:.2f}")
                if avg_reward >= early_stop_reward:
                    print(f"  Early stopping at episode {episode+1} due to reward >= {early_stop_reward}")
                    done = True
                    break

    # Count non-zero parameters after retraining to verify mask was maintained
    non_zero_after = pruned_model.count_nonzero_parameters()
    print(f"After retraining: {non_zero_after} non-zero parameters out of {total_params} total ({non_zero_after/total_params*100:.2f}%)")
    
    # Verify that the pruning mask was maintained
    if non_zero_after != non_zero_before:
        print(f"WARNING: Number of non-zero parameters changed during retraining! Before: {non_zero_before}, After: {non_zero_after}")
    else:
        print("Pruning mask successfully maintained during retraining.")
    
    return pruned_model


def train_dqn(env, env_name, model, episodes=500, gamma=0.99, epsilon_start=1.0, 
              epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64,
              early_stop_reward=None, render_every=None, render_final=False, render_env=None):
    """
    Train the DQN model on the given environment.
    
    Args:
        env: Gym environment
        model: DQN model to train
        episodes: Maximum number of episodes to train for
        gamma: Discount factor
        epsilon_start: Starting value of epsilon for exploration
        epsilon_end: Minimum value of epsilon
        epsilon_decay: Decay rate of epsilon after each episode
        buffer_size: Size of the replay buffer
        batch_size: Mini-batch size for training
        early_stop_reward: If average reward over last 10 episodes exceeds this value, training stops
        render_every: Render every N episodes (None to disable rendering during training)
        render_final: Whether to render the final trained model performance
    """
    
    replay_buffer = ReplayBuffer(buffer_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    epsilon = epsilon_start
    episode_rewards = []
    
    # For early stopping
    reward_window = deque(maxlen=10)
    
    for episode in range(episodes):
        # Check if we should render this episode
        should_render = (render_every is not None and 
                         episode % render_every == 0 and 
                         render_env is not None)
        
        # If rendering, use the render environment for this episode
        episode_env = render_env if should_render else env
        current_state, _ = episode_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
                
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(np.array([current_state]))
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
            
            # Take action in the appropriate environment (rendering or non-rendering)
            next_state, reward, terminated, truncated, _ = episode_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.push(current_state, action, reward, next_state, done)
            current_state = next_state
            
            # Start training when enough samples are available
            if len(replay_buffer) > batch_size:
                # Sample mini-batch
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                
                # Compute Q values
                q_values = model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q_values = model(batch_next_states).max(1)[0]
                    targets = batch_rewards + gamma * next_q_values * (1 - batch_dones)
                
                # Compute loss and update parameters
                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        reward_window.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(reward_window) if reward_window else episode_reward
            print(f"Episode {episode}, Reward: {episode_reward}, Avg Reward (last 10): {avg_reward:.2f}")
        
        # Early stopping check
        if early_stop_reward is not None and len(reward_window) == 10:
            avg_reward = np.mean(reward_window)
            if avg_reward >= early_stop_reward:
                print(f"Early stopping at episode {episode}. Reached average reward of {avg_reward:.2f} >= {early_stop_reward}")
                break
    
    # Render final model performance if requested
    if render_final:
        print("\nRendering final model performance...")
        final_render_env = gym.make(env_name, render_mode="human")
        state, _ = final_render_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state]))
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            state, reward, terminated, truncated, _ = final_render_env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Final model achieved reward: {total_reward}")
        final_render_env.close()
    
    return episode_rewards


def evaluate_dqn(env, model, episodes=100, render=False):
    """Evaluate the DQN model on the given environment."""
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state]))
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        
        if render:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}")
    
    return np.mean(rewards)


# ===================== PRUNING METHODS =====================

def apply_magnitude_pruning(model, prune_ratio):
    """Apply magnitude-based pruning to the model."""
    pruned_model = deepcopy(model)
    
    # Get all weights
    all_weights = []
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only consider weight matrices, not bias vectors
            all_weights.append(param.data.abs().cpu().numpy().flatten())
    
    # Flatten all weights into a single array
    all_weights = np.concatenate(all_weights)
    
    # Determine the threshold for pruning
    threshold = np.percentile(all_weights, prune_ratio * 100)
    
    # Apply pruning
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only prune weight matrices
            mask = (param.data.abs() > threshold).float()
            param.data.mul_(mask)
    
    return pruned_model

def purification(model):
    """
    Remove any neurons that all weights are negative or zero, if they are followed by a ReLU layer.
    Also remove any weights that are connected to this neuron in the next layer.
    This works because ReLU(x) = max(0, x), and the sum of negative weights is always negative.
    This works best on models that are deep but not wide, as the likelihood of a neuron being completely inactive is higher.
    """


    # Converge until the model doesn't change
    pruned_model = deepcopy(model)
    last_nonzero_count = 0
    iteration_count = 0
    max_iterations = 1000  # Add a safeguard to prevent infinite loops
    
    print(f"Starting Purification, number of non-zero parameters: {pruned_model.count_nonzero_parameters()}/{pruned_model.count_parameters()}")
    
    while True:
        iteration_count += 1
        current_nonzero_count = pruned_model.count_nonzero_parameters()
        
        if current_nonzero_count == last_nonzero_count or iteration_count > max_iterations:
            break
            
        last_nonzero_count = current_nonzero_count
        
        # Get the model's layer structure
        layer_list = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                layer_list.append((name, module))
        
        # Find ReLU -> Linear patterns
        for i in range(len(layer_list) - 1):
            if isinstance(layer_list[i][1], nn.ReLU) and isinstance(layer_list[i+1][1], nn.Linear):
                linear_layer = layer_list[i+1][1]
                
                # Find neurons with all negative or zero weights
                mask = (linear_layer.weight.data <= 0).all(dim=1)
                if mask.any():
                    # Zero out these neurons
                    linear_layer.weight.data[mask] = 0.0
                    if linear_layer.bias is not None:
                        linear_layer.bias.data[mask] = 0.0
        
        # Now find Linear -> ReLU -> Linear patterns to propagate zeros
        for i in range(len(layer_list) - 2):
            if (isinstance(layer_list[i][1], nn.Linear) and 
                isinstance(layer_list[i+1][1], nn.ReLU) and 
                isinstance(layer_list[i+2][1], nn.Linear)):
                
                first_linear = layer_list[i][1]
                next_linear = layer_list[i+2][1]
                
                # Check if the neurons in first_linear have all negative or zero weights
                # This is different from the first check - we're looking at the output neurons
                # A mask for output neurons would have shape [out_features]
                neuron_activity = first_linear.weight.data > 0  # Shape: [out_features, in_features]
                
                # A neuron is inactive if all its weights are non-positive
                inactive_neurons = ~neuron_activity.any(dim=1)  # Shape: [out_features]
                
                if inactive_neurons.any():
                    # First, zero out the weights for inactive neurons in the first layer
                    first_linear.weight.data[inactive_neurons] = 0.0
                    if first_linear.bias is not None:
                        first_linear.bias.data[inactive_neurons] = 0.0
                    
                    # Now propagate zeros to the next layer - but only if dimensions match
                    if first_linear.out_features == next_linear.in_features:
                        # Zero out corresponding columns in the next layer
                        for idx in range(len(inactive_neurons)):
                            if inactive_neurons[idx]:
                                next_linear.weight.data[:, idx] = 0.0
                    else:
                        print(f"Dimension mismatch between layers: {first_linear.out_features} != {next_linear.in_features}")
        
        print(f"Iteration {iteration_count}, non-zero parameters: {pruned_model.count_nonzero_parameters()}/{pruned_model.count_parameters()}")
    
    print(f"Finished Purification, number of non-zero parameters: {pruned_model.count_nonzero_parameters()}/{pruned_model.count_parameters()}")
    return pruned_model

def apply_threshold_pruning(model, prune_ratio):
    """Apply threshold-based pruning to the model. Identical to magnitude pruning, except it is done on the signed value of the weights,
        not the absolute value. This should work better for ReLU networks;
        recall that ReLU(x) = max(0, x), so negative weights are effectively already inactive in ReLU networks.
        Thus pruning them should have ZERO impact on the performance of the network.
        For thresholds > 0, we can expect the performance to degrade, as in magnitude pruning.
        i.e. setting the threshold to 0 will always return a network with the same performance as the original network.
    """
    pruned_model = deepcopy(model)
    
    # Get all weights
    all_weights = []
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only consider weight matrices, not bias vectors
            all_weights.append(param.data.cpu().numpy().flatten())
    
    # Flatten all weights into a single array
    all_weights = np.concatenate(all_weights)
    
    # Determine the threshold for pruning
    threshold = np.percentile(all_weights, prune_ratio * 100)
    
    # Apply pruning
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only prune weight matrices
            mask = (param.data < threshold).float() # TODO: check > is correct and it shouldn't be <
            param.data.mul_(mask)
    
    return pruned_model

def apply_random_pruning(model, prune_ratio):
    """Apply random pruning to the model."""
    pruned_model = deepcopy(model)
    
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only prune weight matrices
            # Create random mask where (prune_ratio) of the elements are zero
            mask = torch.FloatTensor(param.shape).uniform_() > prune_ratio
            param.data.mul_(mask.float())
    
    return pruned_model


def apply_structured_pruning(model, prune_ratio):
    """Apply structured pruning by removing entire neurons/filters."""
    pruned_model = deepcopy(model)
    
    # Identify all linear layers
    linear_layers = []
    for module in pruned_model.modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)
    
    # Skip the output layer
    linear_layers = linear_layers[:-1]
    
    if not linear_layers:
        print("No layers available for structured pruning")
        return pruned_model
    
    # Calculate importance of each neuron in each layer
    for layer_idx, layer in enumerate(linear_layers):
        # Calculate L1-norm of each output neuron's weights
        neuron_importance = torch.norm(layer.weight, p=1, dim=1)
        
        # Determine number of neurons to prune
        num_neurons = layer.weight.shape[0]
        num_to_prune = int(num_neurons * prune_ratio)
        
        if num_to_prune == 0:
            continue
        
        # Get indices of least important neurons
        _, indices = torch.topk(neuron_importance, k=num_neurons-num_to_prune, largest=True)
        mask = torch.zeros(num_neurons)
        mask[indices] = 1.0
        
        # Mask out unimportant neurons' weights and biases
        layer.weight.data.mul_(mask.unsqueeze(1))
        if layer.bias is not None:
            layer.bias.data.mul_(mask)
        
        # Also mask the corresponding input weights in the next layer if this isn't the last layer
        if layer_idx < len(linear_layers) - 1:
            next_layer = linear_layers[layer_idx + 1]
            for i in range(num_neurons):
                if mask[i] == 0:
                    next_layer.weight.data[:, i] = 0.0
    
    return pruned_model


def apply_layer_pruning(model, prune_ratio):
    """Apply different pruning ratios to different layers."""
    pruned_model = deepcopy(model)
    
    # Collect all weight matrices
    weight_params = []
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only consider weight matrices
            weight_params.append(param)
    
    # Apply increasing pruning ratios to deeper layers
    n_layers = len(weight_params)
    for i, param in enumerate(weight_params):
        # Scale pruning ratio by layer depth
        layer_ratio = prune_ratio * (i + 1) / n_layers
        layer_ratio = min(layer_ratio, 0.95)  # Cap at 95% to avoid pruning everything
        
        # Create and apply mask
        threshold = torch.quantile(param.abs().flatten(), layer_ratio)
        mask = (param.abs() > threshold).float()
        param.data.mul_(mask)
    
    return pruned_model


def apply_gradient_based_pruning(model, prune_ratio, x_samples=None):
    """Apply pruning based on weight * gradient products."""
    pruned_model = deepcopy(model)
    
    # If no input samples provided, create random ones
    if x_samples is None:
        # Assuming a CartPole-like environment with state dimension of 4
        x_samples = torch.randn(100, 4) #TODO: random selection probably not conducive to good pruning, should be more structured
    
    # Compute a forward pass to enable gradient calculation
    pruned_model.train()
    outputs = pruned_model(x_samples)
    # Use mean of outputs as a simple loss function
    loss = outputs.mean()
    loss.backward()
    
    # Collect all weights and their gradients
    weights_grads = []
    for param in pruned_model.parameters():
        if len(param.shape) > 1 and param.grad is not None:
            # Calculate sensitivity as |weight × gradient|
            sensitivity = (param.data * param.grad).abs().cpu().numpy().flatten()
            weights_grads.append(sensitivity)
    
    # Flatten and find threshold
    all_sensitivities = np.concatenate(weights_grads)
    threshold = np.percentile(all_sensitivities, prune_ratio * 100)
    
    # Apply pruning
    for param in pruned_model.parameters():
        if len(param.shape) > 1 and param.grad is not None:
            sensitivity = (param.data * param.grad).abs()
            mask = (sensitivity > threshold).float()
            param.data.mul_(mask)
    
    # Reset gradients
    pruned_model.zero_grad()
    
    return pruned_model


def apply_weight_sharing(model, prune_ratio, num_clusters=8):
    """Apply weight sharing (scalar quantization)."""
    pruned_model = deepcopy(model)
    
    # Weight sharing doesn't exactly match the pruning ratio concept, 
    # but we'll use prune_ratio to determine how aggressively to cluster
    clusters = num_clusters * (1 - prune_ratio) + 1
    clusters = max(2, int(clusters))  # Ensure at least 2 clusters
    
    for param in pruned_model.parameters():
        if len(param.shape) > 1:  # Only quantize weight matrices
            flattened = param.data.flatten().cpu().numpy()
            
            # Only cluster non-zero weights
            non_zero_mask = flattened != 0
            non_zero_weights = flattened[non_zero_mask]
            
            if len(non_zero_weights) > 0:
                # Use simple uniform binning
                min_val, max_val = non_zero_weights.min(), non_zero_weights.max()
                bin_width = (max_val - min_val) / clusters
                
                # Quantize to bin centers
                if bin_width > 0:
                    quantized_weights = np.round((non_zero_weights - min_val) / bin_width) * bin_width + min_val
                    flattened[non_zero_mask] = quantized_weights
                
                # Reshape back to original shape
                param.data = torch.FloatTensor(flattened.reshape(param.shape))
    
    return pruned_model


def prune_dqn_with_multiple_methods(env_name='CartPole-v1', hidden_dims=[128, 128], 
                                  train_episodes=500, eval_episodes=100,
                                  early_stop_reward=None, render_every=None, 
                                  render_final=False, visualize_pruning=False, 
                                  prune_ratios=None, retraining=True):
    """
    Main function to train a DQN and test multiple pruning methods.
    
    Args:
        env_name: Name of the Gym environment
        hidden_dims: List of hidden layer dimensions for the DQN
        train_episodes: Maximum number of episodes to train the base DQN
        eval_episodes: Number of episodes to evaluate each pruned model
        early_stop_reward: If average reward over last 100 episodes exceeds this, training stops
        render_every: Render every N episodes during training (None to disable)
        render_final: Whether to render the final model after training
        visualize_pruning: Whether to visualize performance at each pruning level
        prune_ratios: List of pruning ratios to evaluate (default: wide range from 0 to 0.99)
    """
    # Create environment
    env = gym.make(env_name)
    
    # Create and train base model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    base_model = DQN(state_dim, action_dim, hidden_dims)
    print(f"Training base model with {base_model.count_parameters()} parameters...")
    
    render_env = None
    if render_every is not None:
        render_env = gym.make(env_name, render_mode="human")
        
    train_rewards = train_dqn(env, env_name, base_model, episodes=train_episodes, 
                              early_stop_reward=early_stop_reward,
                              render_every=render_every,
                              render_final=render_final,
                              render_env=render_env)
    
    # Evaluate base model
    eval_env = gym.make(env_name)  # Create a new env for evaluation
    base_reward = evaluate_dqn(eval_env, base_model, episodes=eval_episodes)
    print(f"Base model average reward over {eval_episodes} episodes: {base_reward:.2f}")

    
    # Create visualization environment if needed
    vis_env = None
    if visualize_pruning:
        vis_env = gym.make(env_name, render_mode="human")
        print("\nVisualizing base model performance...")
        #visualize_model(vis_env, base_model, episodes=1)
    
    # Define pruning methods
    pruning_methods = [
        "Magnitude Pruning",
        #"Threshold Pruning",
        #"Random Pruning",
        "Structured Pruning",
        "Layer-wise Pruning",
        "Gradient-based Pruning",
        #"Weight Sharing"
    ]
    
    # Define pruning ratios if not provided
    if prune_ratios is None:
        prune_ratios = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]
    
    # Generate random samples for gradient-based pruning
    x_samples = torch.FloatTensor(np.random.rand(100, state_dim))
    
    all_results = []
    all_pareto_fronts = []
    
    # Test each pruning method
    for method in pruning_methods:
        print(f"\n{'='*20} Evaluating {method} {'='*20}")
        results, pareto_front = find_pareto_front(
            eval_env, env_name, base_model, method, 
            prune_ratios, eval_episodes, x_samples, retraining=retraining,
            render_env=vis_env if visualize_pruning else None,
            early_stop_reward=early_stop_reward
        )
        
        all_results.append(results)
        all_pareto_fronts.append(pareto_front)
        
        # Plot individual results
        plot_pareto_front(results, pareto_front, method)
        
        # Print Pareto front details
        print(f"\n{method} Pareto Front:")
        print(f"Pruning Ratio | Parameters | Reward (avg over {eval_episodes} episodes)")
        print("-" * 60)
        for ratio, params, reward, _ in pareto_front:
            print(f"{ratio*100:10.1f}% | {params:10d} | {reward:6.2f}")
    
    # Create comparison plot
    compare_all_methods(all_results, all_pareto_fronts, pruning_methods)
    
    # Save all Pareto-efficient solutions to file
    save_pareto_solutions(all_pareto_fronts, pruning_methods)
    
    # Find the best model across all methods
    best_model = None
    best_reward = float('-inf')
    best_method = None
    best_ratio = None
    
    for method, pareto_front in zip(pruning_methods, all_pareto_fronts):
        for ratio, params, reward, model in pareto_front:
            if reward > best_reward:
                best_reward = reward
                best_model = model
                best_method = method
                best_ratio = ratio
    
    print(f"\nBest model found: {best_method} with pruning ratio {best_ratio*100:.1f}%")
    print(f"Parameters: {best_model.count_nonzero_parameters()}")
    print(f"Reward (avg over {eval_episodes} episodes): {best_reward:.2f}")
    
    # Render the best pruned model if requested
    if render_final:
        print(f"\nRendering best pruned model ({best_method}, ratio: {best_ratio*100:.1f}%)...")
        render_env = gym.make(env_name, render_mode="human")
        evaluate_dqn(render_env, best_model, episodes=3, render=True)
    
    # Save the best model
    torch.save(best_model.state_dict(), 'pruning_results/best_pruned_dqn.pt')
    
    return base_model, all_pareto_fronts, pruning_methods



def evaluate_and_visualize_pruning_levels(env_name, base_model, method_name, prune_ratios=[0.5, 0.7, 0.9, 0.95, 0.99]):
    """Evaluate and visualize specific pruning levels for a given method."""
    eval_env = gym.make(env_name)
    render_env = gym.make(env_name, render_mode="human")
    
    print(f"\n{'='*20} Visualizing {method_name} at different pruning levels {'='*20}")
    
    # First visualize base model
    #print("\nBase Model (0% pruning):")
    #visualize_model(render_env, base_model, episodes=1)
    
    for ratio in prune_ratios:
        print(f"\n{method_name} at {ratio*100:.1f}% pruning:")
        if method_name == "Magnitude Pruning":
            pruned_model = apply_magnitude_pruning(base_model, ratio)
        elif method_name == "Threshold Pruning":
            pruned_model = apply_threshold_pruning(base_model, ratio)
        elif method_name == "Random Pruning":
            pruned_model = apply_random_pruning(base_model, ratio)
        elif method_name == "Structured Pruning":
            pruned_model = apply_structured_pruning(base_model, ratio)
        elif method_name == "Layer-wise Pruning":
            pruned_model = apply_layer_pruning(base_model, ratio)
        elif method_name == "Gradient-based Pruning":
            x_samples = torch.FloatTensor(np.random.rand(100, eval_env.observation_space.shape[0]))
            pruned_model = apply_gradient_based_pruning(base_model, ratio, x_samples)
        elif method_name == "Weight Sharing":
            pruned_model = apply_weight_sharing(base_model, ratio)
        else:
            raise ValueError(f"Unknown pruning method: {method_name}")
        
        # Count non-zero parameters
        non_zero_params = pruned_model.count_nonzero_parameters()
        total_params = pruned_model.count_parameters()
        print(f"Parameters: {non_zero_params} non-zero out of {total_params} total ({non_zero_params/total_params*100:.1f}%)")
        
        pruned_model = purification(pruned_model)
        non_zero_params = pruned_model.count_nonzero_parameters()
        total_params = pruned_model.count_parameters()
        print(f"Parameters after purification: {non_zero_params} non-zero out of {total_params} total ({non_zero_params/total_params*100:.1f}%)")

        # Evaluate on a few episodes
        avg_reward = evaluate_dqn(eval_env, pruned_model, episodes=10)
        print(f"Average reward over 10 episodes: {avg_reward:.2f}")
        
        # Visualize
        visualize_model(render_env, pruned_model, episodes=1)






def compare_all_methods(all_results, all_pareto_fronts, methods):
    """Create a comparison plot of all pruning methods."""
    plt.figure(figsize=(14, 10))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan']
    
    # Plot the Pareto front for each method
    for i, (method, pareto_front) in enumerate(zip(methods, all_pareto_fronts)):
        pareto_params = [p[1] for p in pareto_front]
        pareto_rewards = [p[2] for p in pareto_front]
        pareto_ratios = [p[0] for p in pareto_front]
        
        plt.plot(pareto_params, pareto_rewards, '--', color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        plt.scatter(pareto_params, pareto_rewards, s=100, marker=markers[i % len(markers)], 
                   color=colors[i % len(colors)], label=method)
        
        # Add percentage labels to points
        for param, reward, ratio in zip(pareto_params, pareto_rewards, pareto_ratios):
            plt.annotate(f"{ratio*100:.0f}%", (param, reward), xytext=(5, 0), 
                        textcoords='offset points', fontsize=8, color=colors[i % len(colors)])
    
    # Add base model - should be the same for all methods
    base_params = all_results[0][0][1]
    base_reward = all_results[0][0][2]
    plt.scatter(base_params, base_reward, s=200, facecolors='yellow', edgecolors='black', 
               linewidths=2, marker='*', label='Unpruned Model')
    plt.annotate("0%", (base_params, base_reward), xytext=(5, 5), 
                textcoords='offset points', fontsize=10, fontweight='bold', color='black')
    
    plt.xlabel('Number of Non-zero Parameters', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Comparison of DQN Pruning Methods', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pruning_results/dqn_pruning_comparison.png', dpi=300)
    plt.close()


def save_pareto_solutions(all_pareto_fronts, methods):
    """Save all Pareto-efficient solutions to a file."""
    import pandas as pd
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs('pruning_results', exist_ok=True)
    
    # Collect data from all Pareto fronts
    all_data = []
    
    for method, pareto_front in zip(methods, all_pareto_fronts):
        for ratio, params, reward, _ in pareto_front:
            all_data.append({
                'Method': method,
                'Pruning_Ratio': ratio,
                'Pruning_Percentage': f"{ratio*100:.1f}%",
                'Parameters': params,
                'Average_Reward': reward,
            })
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)
    
    # Sort by method and pruning ratio
    df = df.sort_values(by=['Method', 'Pruning_Ratio'])
    
    # Save to CSV
    df.to_csv('pruning_results/pareto_efficient_solutions.csv', index=False)
    
    # Also save to a more human-readable format
    with open('pruning_results/pareto_efficient_solutions.txt', 'w') as f:
        f.write("Summary of Pareto-Efficient Solutions for DQN Pruning\n")
        f.write("=" * 80 + "\n\n")
        
        for method in methods:
            f.write(f"\n{method}\n")
            f.write("-" * len(method) + "\n")
            f.write(f"Pruning Ratio | Parameters | Average Reward\n")
            f.write("-" * 70 + "\n")
            
            method_data = [item for item in all_data if item['Method'] == method]
            for item in sorted(method_data, key=lambda x: x['Pruning_Ratio']):
                f.write(f"{item['Pruning_Percentage']:12} | {item['Parameters']:10} | {item['Average_Reward']:6.2f}\n")
            
            f.write("\n")
    
    print(f"\nPareto-efficient solutions saved to 'pruning_results/pareto_efficient_solutions.csv' and 'pruning_results/pareto_efficient_solutions.txt'")
    return df





if __name__ == "__main__":
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    
    # n %s between 0 and 1
    n_prune_ratios = 10
    custom_prune_ratios = np.linspace(0, 0.99, n_prune_ratios)
    
    # Run with multiple pruning methods and specified pruning ratios
    base_model, all_pareto_fronts, pruning_methods = prune_dqn_with_multiple_methods(
        env_name='CartPole-v1',
        hidden_dims=[256, 128, 64, 16],
        train_episodes=20000,  # Max episodes, early stopping is used
        eval_episodes=50,     # Reduced for faster execution but still statistically meaningful
        early_stop_reward=300.0,  # CartPole-v1 is considered solved at 195.0
        render_every=None,        # Disable rendering during training for faster execution
        render_final=True,        # Render final performance after training
        visualize_pruning=True,   # Visualize model at each pruning level
        prune_ratios=custom_prune_ratios  # Use the custom pruning ratios
    )

    
    
    # Additionally, perform a focused evaluation of each method at key pruning levels
    # This allows seeing the actual performance degradation as pruning increases
    for method in pruning_methods:
        evaluate_and_visualize_pruning_levels('CartPole-v1', base_model, method)
        
    print("\nAll pruning methods have been evaluated and visualized!")