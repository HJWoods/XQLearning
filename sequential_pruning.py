import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def sequential_optimal_pruning(model, eval_env, prune_percentages=None, num_eval_episodes=50, 
                              performance_threshold=0.05, use_gradient=True, device='cuda'):
    """
    Find optimal pruning percentages for each layer sequentially and apply purification.
    
    Args:
        model: The DQN model to prune
        eval_env: Environment for evaluating model performance
        prune_percentages: List of pruning percentages to try for each layer (e.g. [0.2, 0.4, 0.6, 0.8])
        num_eval_episodes: Number of episodes to evaluate each model configuration
        performance_threshold: Maximum allowed performance drop ratio compared to the baseline
        use_gradient: Whether to use gradient-based pruning (True) or magnitude-based pruning (False)
        device: Device to run the model on
    
    Returns:
        Pruned model with optimized layer-wise pruning
    """
    # Default pruning percentages to try
    if prune_percentages is None:
        prune_percentages = [0.2, 0.4, 0.6, 0.8, 0.9]
    
    # Evaluate baseline performance
    baseline_reward = evaluate_dqn(eval_env, model, episodes=num_eval_episodes)
    print(f"Baseline performance: {baseline_reward:.4f} over {num_eval_episodes} episodes")
    
    # Minimum acceptable performance
    min_acceptable_reward = baseline_reward * (1 - performance_threshold)
    print(f"Minimum acceptable performance: {min_acceptable_reward:.4f}")
    
    # Create a working copy of the model
    pruned_model = deepcopy(model)
    
    # Identify all linear layers in the model
    linear_layers = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    print(f"Found {len(linear_layers)} linear layers to optimize")
    print(f"Using {'gradient-based' if use_gradient else 'magnitude-based'} pruning")
    
    # Process layers in order (from input to output, skip the last output layer)
    for i, (layer_name, layer) in enumerate(linear_layers[:-1]):  # Skip the output layer
        print(f"\nOptimizing layer {i+1}/{len(linear_layers)-1}: {layer_name}")
        
        best_prune_pct = 0
        best_performance = baseline_reward
        best_layer_state = deepcopy(layer.state_dict())
        
        # Try different pruning percentages for this layer
        for prune_pct in prune_percentages:
            # Create a temporary model with this layer pruned
            temp_model = deepcopy(pruned_model)
            
            # Prune the layer based on the specified method
            if use_gradient:
                # Gradient-based pruning
                prune_layer_by_gradient(temp_model, layer_name, prune_pct, eval_env)
            else:
                # Magnitude-based pruning
                temp_layer = None
                for name, module in temp_model.named_modules():
                    if name == layer_name:
                        temp_layer = module
                        break
                
                if temp_layer is None:
                    continue
                    
                prune_layer_by_magnitude(temp_layer, prune_pct)
            
            # Apply purification to propagate zeros
            temp_model = purification(temp_model)
            
            # Evaluate the model
            reward = evaluate_dqn(eval_env, temp_model, episodes=num_eval_episodes)
            
            print(f"  Pruning ratio {prune_pct*100:.1f}%: Reward = {reward:.2f}, "
                  f"Non-zero parameters: {temp_model.count_nonzero_parameters()}/{temp_model.count_parameters()}")
            
            # If performance is acceptable and better than previous best pruning level, update
            if reward >= min_acceptable_reward and prune_pct > best_prune_pct:
                best_prune_pct = prune_pct
                best_performance = reward
                best_layer_state = deepcopy(temp_layer.state_dict())
        
        # Apply the best pruning percentage to this layer
        for name, module in pruned_model.named_modules():
            if name == layer_name:
                module.load_state_dict(best_layer_state)
                break
        
        # Apply purification after each layer optimization to propagate zeros
        pruned_model = purification(pruned_model)
        
        pruned_count = pruned_model.count_nonzero_parameters()
        total_count = pruned_model.count_parameters()
        print(f"Layer {i+1} optimized: Best pruning ratio = {best_prune_pct*100:.1f}%, "
              f"Performance = {best_performance:.2f}, "
              f"Parameters remaining: {pruned_count}/{total_count} ({pruned_count/total_count*100:.1f}%)")
    
    # Final purification pass
    pruned_model = purification(pruned_model)
    
    final_reward = evaluate_dqn(eval_env, pruned_model, episodes=num_eval_episodes)
    pruned_count = pruned_model.count_nonzero_parameters()
    total_count = pruned_model.count_parameters()
    
    print(f"\nFinal model: Performance = {final_reward:.2f}, "
          f"Parameters: {pruned_count}/{total_count} ({pruned_count/total_count*100:.1f}%)")
    
    return pruned_model

def prune_layer_by_gradient(model, layer_name, prune_ratio, eval_env, num_samples=100):
    """
    Prune a layer by gradient sensitivity.
    
    Args:
        model: The full model containing the layer
        layer_name: Name of the layer to prune
        prune_ratio: Percentage of weights to prune (0.0 to 1.0)
        eval_env: Environment to get sample states for gradient computation
        num_samples: Number of state samples to use for gradient computation
    """
    # Get the layer
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break
    
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in the model")
    
    # Get sample states from the environment
    states = []
    for _ in range(num_samples):
        state, _ = eval_env.reset()
        states.append(state)
    
    # Convert to tensor
    states_tensor = torch.FloatTensor(np.array(states))
    
    # Enable gradients
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = True
    
    # Forward pass
    outputs = model(states_tensor)
    
    # Use mean of Q-values as a simple loss function
    loss = outputs.mean()
    
    # Backward pass to compute gradients
    loss.backward()
    
    # Calculate sensitivity as |weight × gradient|
    if layer.weight.grad is None:
        print(f"Warning: No gradient computed for {layer_name}, falling back to magnitude pruning")
        return prune_layer_by_magnitude(layer, prune_ratio)
    
    sensitivity = (layer.weight.data * layer.weight.grad).abs().cpu().numpy()
    
    # Calculate the threshold for pruning
    threshold = np.percentile(sensitivity, prune_ratio * 100)
    
    # Create a mask for weights below sensitivity threshold
    mask = (sensitivity > threshold).astype(np.float32)
    
    # Apply the mask
    layer.weight.data.mul_(torch.FloatTensor(mask))
    
    # Disable gradients again
    for param in model.parameters():
        param.requires_grad = False
    
    return layer

def prune_layer_by_magnitude(layer, prune_ratio):
    """
    Prune a layer by magnitude.
    
    Args:
        layer: PyTorch layer to prune
        prune_ratio: Percentage of weights to prune (0.0 to 1.0)
    """
    # Get the weights
    weights = layer.weight.data.cpu().numpy()
    
    # Calculate the threshold for pruning
    threshold = np.percentile(np.abs(weights), prune_ratio * 100)
    
    # Create a mask for weights below threshold
    mask = (np.abs(weights) > threshold).astype(np.float32)
    
    # Apply the mask
    layer.weight.data.mul_(torch.FloatTensor(mask))
    
    return layer

def purification(model):
    """
    Remove any neurons that all weights are negative or zero, if they are followed by a ReLU layer.
    Also remove any weights that are connected to this neuron in the next layer.
    This works because ReLU(x) = max(0, x), and the sum of negative weights is always negative.
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
        
        if iteration_count % 5 == 0 or iteration_count == 1:
            print(f"Iteration {iteration_count}, non-zero parameters: {pruned_model.count_nonzero_parameters()}/{pruned_model.count_parameters()}")
    
    print(f"Finished Purification, number of non-zero parameters: {pruned_model.count_nonzero_parameters()}/{pruned_model.count_parameters()}")
    return pruned_model

def evaluate_dqn(env, model, episodes=100):
    """Evaluate the DQN model on the given environment."""
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state]))
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def analyze_layer_contributions(model, eval_env, num_eval_episodes=20):
    """
    Analyze each layer's contribution to the overall model performance
    by zeroing out one layer at a time and measuring performance drop.
    """
    baseline_reward = evaluate_dqn(eval_env, model, episodes=num_eval_episodes)
    print(f"Baseline performance: {baseline_reward:.2f}")
    
    # Identify all linear layers in the model
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    results = []
    # Test performance with each layer zeroed out
    for i, (layer_name, _) in enumerate(linear_layers[:-1]):  # Skip output layer
        temp_model = deepcopy(model)
        
        # Zero out this layer
        for name, module in temp_model.named_modules():
            if name == layer_name:
                module.weight.data.zero_()
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # Evaluate
        zeroed_reward = evaluate_dqn(eval_env, temp_model, episodes=num_eval_episodes)
        performance_drop = (baseline_reward - zeroed_reward) / baseline_reward * 100
        
        print(f"Layer {i+1}: {layer_name} - Performance with layer zeroed: {zeroed_reward:.2f} (Drop: {performance_drop:.1f}%)")
        results.append((layer_name, zeroed_reward, performance_drop))
    
    # Sort layers by their importance (performance drop)
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    print("\nLayers ranked by importance:")
    for i, (name, reward, drop) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {drop:.1f}% performance drop when zeroed")
    
    return sorted_results

# Function to apply the optimized sequential pruning to a DQN
def optimize_dqn_pruning(env_name, model, eval_episodes=50, performance_threshold=0.05, use_gradient=True):
    """
    Apply sequential optimal pruning to a DQN model.
    
    Args:
        env_name: Gym environment name
        model: Trained DQN model
        eval_episodes: Number of episodes for evaluation
        performance_threshold: Maximum acceptable performance drop
        use_gradient: Whether to use gradient-based pruning (True) or magnitude-based pruning (False)
    
    Returns:
        Optimized pruned model
    """
    import gym
    
    # Create evaluation environment
    eval_env = gym.make(env_name)
    
    # First analyze layer contributions to understand which layers are more important
    print("\nAnalyzing layer contributions to model performance...")
    layer_importance = analyze_layer_contributions(model, eval_env, num_eval_episodes=20)
    
    # Apply sequential pruning
    print("\nApplying sequential layer-wise pruning...")
    pruned_model = sequential_optimal_pruning(
        model, 
        eval_env,
        prune_percentages=[0.3, 0.5, 0.7, 0.8, 0.9],
        num_eval_episodes=eval_episodes,
        performance_threshold=performance_threshold,
        use_gradient=use_gradient
    )
    
    # Final evaluation
    final_performance = evaluate_dqn(eval_env, pruned_model, episodes=100)
    base_performance = evaluate_dqn(eval_env, model, episodes=100)
    
    print(f"\nFinal evaluation (100 episodes):")
    print(f"Original model: {base_performance:.2f}")
    print(f"Pruned model: {final_performance:.2f} ({final_performance/base_performance*100:.1f}% of original)")
    
    pruned_count = pruned_model.count_nonzero_parameters()
    total_count = pruned_model.count_parameters()
    compression_ratio = total_count / pruned_count if pruned_count > 0 else float('inf')
    
    print(f"Compression achieved: {pruned_count}/{total_count} parameters remaining ({pruned_count/total_count*100:.2f}%)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return pruned_model


if __name__ == '__main__':
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    from pruning import DQN, train_dqn
    import gym
    render_every = 100
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    model = DQN(state_dim=4, action_dim=2, hidden_dims=[64, 64, 64, 64])
    render_env = None
    if render_every is not None:
        render_env = gym.make(env_name, render_mode="human")
        
    train_rewards = train_dqn(env, env_name, model, episodes=20000, 
                              early_stop_reward=300,
                              render_every=render_every,
                              render_final=True,
                              render_env=render_env)
    optimized_model = optimize_dqn_pruning('CartPole-v1', model)
# Usage example:
# from your_dqn_module import DQN
# model = DQN(state_dim=4, action_dim=2, hidden_dims=[64, 64])
# # Train your model first
# optimized_model = optimize_dqn_pruning('CartPole-v1', model)