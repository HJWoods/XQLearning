import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import json

class PyTorchToPolyC:
    def __init__(self, input_vars, output_vars, constants=None, env_vars=None):
        """
        Initialize the converter with variable definitions
        
        Args:
            input_vars (dict): Input variable names mapped to their dimensions
            output_vars (dict): Output/action variable names mapped to their dimensions
            constants (dict, optional): Constant values
            env_vars (dict, optional): Environment variables
        """
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.constants = constants or {}
        self.env_vars = env_vars or {}
        self.state_vars = {}
        self.constraints = []
        self.goals = []
    


    def generate_polyc_code(self, model_name="PyTorchModel"):
        """Generate the PolyC policy from the extracted neural network"""
        if not self.weights:
            raise ValueError("No model has been extracted. Call extract_model_architecture first.")
        
        # If there are fewer activation types than layers, add identity activation for the missing layers
        while len(self.activation_types) < len(self.weights):
            self.activation_types.append("identity")
        
        polyc_code = [f"# PolyC representation of PyTorch model: {model_name}"]
        
        # Add variable declarations
        polyc_code.append("\n# Input variables")
        for name, dim in self.input_vars.items():
            if isinstance(dim, int) and dim > 1:
                polyc_code.append(f"input float[{dim}] {name}")
            else:
                polyc_code.append(f"input float {name}")
        
        polyc_code.append("\n# Action variables")
        for name, dim in self.output_vars.items():
            if isinstance(dim, int) and dim > 1:
                polyc_code.append(f"action float[{dim}] {name}")
            else:
                polyc_code.append(f"action float {name}")
        
        polyc_code.append("\n# Constants")
        for name, value in self.constants.items():
            polyc_code.append(f"const float {name}")
        
        polyc_code.append("\n# Environment variables")
        for name, dim in self.env_vars.items():
            if isinstance(dim, int) and dim > 1:
                polyc_code.append(f"env float[{dim}] {name}")
            else:
                polyc_code.append(f"env float {name}")
        
        polyc_code.append("\n# State variables")
        # Add user-defined state variables
        for name, dim in self.state_vars.items():
            if isinstance(dim, int) and dim > 1:
                polyc_code.append(f"var float[{dim}] {name}")
            else:
                polyc_code.append(f"var float {name}")
        
        # Add additional state variable for input combination (to match network's expectation)
        first_layer_size = self.weights[0].shape[1]
        polyc_code.append(f"var float[{first_layer_size}] combined_inputs")
        
        # Add layer weights as constants and define layer outputs
        for i, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            input_size, output_size = weights.shape[1], weights.shape[0]
            
            # Add weights as constants
            for in_idx in range(input_size):
                for out_idx in range(output_size):
                    weight_value = weights[out_idx, in_idx]
                    polyc_code.append(f"const float w{i}_{in_idx}_{out_idx} = {weight_value}")
            
            # Add biases as constants
            for out_idx in range(output_size):
                bias_value = bias[out_idx]
                polyc_code.append(f"const float b{i}_{out_idx} = {bias_value}")
            
            # Define layer output variables with correct dimensions
            if output_size > 1:
                polyc_code.append(f"var float[{output_size}] layer{i}_output")
            else:
                polyc_code.append(f"var float layer{i}_output")
        
        # Add helper variables
        polyc_code.append("var float temp")
        
        # Set constant values
        for name, value in self.constants.items():
            if name not in ["leaky_relu_alpha"]:  # Skip activation function constants that are set above
                polyc_code.append(f"{name} = {value}")
        
        # Add constraints section
        if self.constraints:
            polyc_code.append("\nconstraints [")
            for constraint in self.constraints:
                polyc_code.append(f"    {constraint}")
            polyc_code.append("]")
        
        # Add goals section
        if self.goals:
            polyc_code.append("\ngoals [")
            for goal in self.goals:
                polyc_code.append(f"    {goal}")
            polyc_code.append("]")
        
        # Add activation functions
        polyc_code.append("\n# Activation functions")
        
        # ReLU activation
        polyc_code.append("relu(x) {")
        polyc_code.append("    if x < 0 {")
        polyc_code.append("        return 0")
        polyc_code.append("    }")
        polyc_code.append("    return x")
        polyc_code.append("}")
        
        # Sigmoid activation (piece-wise linear approximation)
        polyc_code.append("\nsigmoid(x) {")
        polyc_code.append("    # Approximation of sigmoid using piece-wise linear function")
        polyc_code.append("    if x < -5 {")
        polyc_code.append("        return 0")
        polyc_code.append("    }")
        polyc_code.append("    if x > 5 {")
        polyc_code.append("        return 1")
        polyc_code.append("    }")
        polyc_code.append("    if x < 0 {")
        polyc_code.append("        return 0.25 * x + 0.5")
        polyc_code.append("    }")
        polyc_code.append("    return 0.25 * x + 0.5")
        polyc_code.append("}")
        
        # Tanh activation (piece-wise linear approximation)
        polyc_code.append("\ntanh(x) {")
        polyc_code.append("    # Approximation of tanh using piece-wise linear function")
        polyc_code.append("    if x < -3 {")
        polyc_code.append("        return -1")
        polyc_code.append("    }")
        polyc_code.append("    if x > 3 {")
        polyc_code.append("        return 1")
        polyc_code.append("    }")
        polyc_code.append("    return 0.33 * x")
        polyc_code.append("}")
        
        # Leaky ReLU activation
        polyc_code.append("\nleaky_relu(x) {")
        polyc_code.append("    if x < 0 {")
        polyc_code.append("        return leaky_relu_alpha * x")
        polyc_code.append("    }")
        polyc_code.append("    return x")
        polyc_code.append("}")
        
        # Identity activation
        polyc_code.append("\nidentity(x) {")
        polyc_code.append("    return x")
        polyc_code.append("}")
        
        # Main function
        polyc_code.append("\n# Main policy function")
        polyc_code.append("main() {")
        
        # Create an input mapping that matches the order expected by the network
        if hasattr(self, 'input_order') and self.input_order:
            # User provided explicit order
            input_order = self.input_order
        else:
            # Default: alphabetical
            input_order = sorted(self.input_vars.keys())
        
        # First combine all inputs into a single array for processing
        polyc_code.append("    # Combine individual state variables into input array")
        for i, name in enumerate(input_order):
            polyc_code.append(f"    combined_inputs[{i}] = {name}")
        
        # Forward pass through the neural network
        for layer_idx, ((in_size, out_size), activation) in enumerate(zip(self.layer_sizes, self.activation_types)):
            polyc_code.append(f"\n    # Layer {layer_idx}: {in_size} -> {out_size}, {activation}")
            
            # Determine if current output is an array
            current_is_array = out_size > 1
            
            # For each output in this layer
            for j in range(out_size):
                # Get the correct output accessor
                output_accessor = f"[{j}]" if current_is_array else ""
                
                # Initialize with bias
                polyc_code.append(f"    layer{layer_idx}_output{output_accessor} = b{layer_idx}_{j}")
                
                # For the first layer, use combined_inputs
                if layer_idx == 0:
                    # Use the combined inputs
                    for i in range(in_size):
                        polyc_code.append(f"    layer{layer_idx}_output{output_accessor} = layer{layer_idx}_output{output_accessor} + (combined_inputs[{i}] * w{layer_idx}_{i}_{j})")
                else:
                    # Use previous layer's outputs
                    prev_is_array = self.weights[layer_idx-1].shape[0] > 1
                    for i in range(in_size):
                        # Get the correct input accessor
                        input_accessor = f"[{i}]" if prev_is_array else ""
                        polyc_code.append(f"    layer{layer_idx}_output{output_accessor} = layer{layer_idx}_output{output_accessor} + (layer{layer_idx-1}_output{input_accessor} * w{layer_idx}_{i}_{j})")
                
                # Apply activation function
                polyc_code.append(f"    layer{layer_idx}_output{output_accessor} = {activation}(layer{layer_idx}_output{output_accessor})")
        
        # Copy final layer output to action variables
        final_layer_idx = len(self.weights) - 1
        final_output_size = self.weights[final_layer_idx].shape[0]
        final_is_array = final_output_size > 1
        
        output_vars_list = list(self.output_vars.keys())
        output_var_name = output_vars_list[0]
        output_dim = self.output_vars[output_var_name]
        print(f"Output variable: {output_var_name} with dimension {output_dim}")
        output_is_array = isinstance(output_dim, int) and output_dim > 1
        
        polyc_code.append(f"\n    # Output layer: copy to action variable")
        if output_is_array:
            # Handle vector outputs
            for i in range(min(final_output_size, output_dim)):
                final_output_accessor = f"[{i}]" if final_is_array else ""
                polyc_code.append(f"    {output_var_name}[{i}] = layer{final_layer_idx}_output{final_output_accessor}")
        else:
            # Handle scalar outputs
            if final_is_array:
                # If we have an array output but scalar action, use the first element
                polyc_code.append(f"    {output_var_name} = layer{final_layer_idx}_output[0]")
            else:
                polyc_code.append(f"    {output_var_name} = layer{final_layer_idx}_output")
        
        polyc_code.append("}")
        
        return "\n".join(polyc_code)

       
    def add_state_vars(self, state_vars):
        """Add state variables needed for computations"""
        self.state_vars.update(state_vars)
    
    def add_constraints(self, constraints):
        """Add constraints for the PolyC program"""
        self.constraints.extend(constraints)
    
    def add_goals(self, goals):
        """Add goals for the PolyC program"""
        self.goals.extend(goals)
    
    def extract_model_architecture(self, model):
        """Extract the architecture and parameters of a PyTorch model."""
        self.weights = []
        self.biases = []
        self.activation_types = []
        self.layer_sizes = []
        
        # Get all layers that have parameters
        linear_layers = []
        activation_layers = []
        
        # Extract layers from the model
        if hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
            # Model has a .layers attribute (like our CartPoleModel)
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    linear_layers.append(layer)
                elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
                    activation_layers.append(layer)
        elif hasattr(model, 'features') and hasattr(model, 'output_layer'):
            # Model has features + output layer structure
            if isinstance(model.features, nn.Sequential):
                for layer in model.features:
                    if isinstance(layer, nn.Linear):
                        linear_layers.append(layer)
                    elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
                        activation_layers.append(layer)
            linear_layers.append(model.output_layer)
        else:
            # Try traversing all modules
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
                elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
                    activation_layers.append(module)
        
        # Extract weights and biases from linear layers
        for layer in linear_layers:
            weights = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy() if layer.bias is not None else np.zeros(layer.out_features)
            self.weights.append(weights)
            self.biases.append(bias)
            self.layer_sizes.append((layer.in_features, layer.out_features))
        
        # Match activations to layers (this is a heuristic since model structure can vary)
        curr_activation_idx = 0
        for i in range(len(linear_layers)):
            if i < len(linear_layers) - 1 and curr_activation_idx < len(activation_layers):
                # Assume activation follows the linear layer
                if isinstance(activation_layers[curr_activation_idx], nn.ReLU):
                    self.activation_types.append("relu")
                elif isinstance(activation_layers[curr_activation_idx], nn.Sigmoid):
                    self.activation_types.append("sigmoid")
                elif isinstance(activation_layers[curr_activation_idx], nn.Tanh):
                    self.activation_types.append("tanh")
                elif isinstance(activation_layers[curr_activation_idx], nn.LeakyReLU):
                    self.activation_types.append("leaky_relu")
                else:
                    self.activation_types.append("identity")
                curr_activation_idx += 1
            else:
                # Last layer often has no activation or has a specific output activation
                if hasattr(model, 'output_activation'):
                    if isinstance(model.output_activation, nn.Tanh):
                        self.activation_types.append("tanh")
                    elif isinstance(model.output_activation, nn.Sigmoid):
                        self.activation_types.append("sigmoid")
                    else:
                        self.activation_types.append("identity")
                else:
                    self.activation_types.append("identity")
        
        # Print out the extracted architecture
        print("Extracted model architecture:")
        for i, ((in_size, out_size), activation) in enumerate(zip(self.layer_sizes, self.activation_types)):
            print(f"  Layer {i}: {in_size} -> {out_size}, Activation: {activation}")
        
        if len(self.weights) != len(self.activation_types):
            print(f"Warning: Mismatch between number of layers ({len(self.weights)}) and activations ({len(self.activation_types)})")

    
    def _process_sequential(self, sequential):
        """Process a sequential container"""
        for module in sequential.children():
            if isinstance(module, nn.Linear):
                self._process_linear(module)
            elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
                self._process_activation(module)
            elif isinstance(module, nn.Sequential):
                self._process_sequential(module)
    
    def _process_linear(self, linear):
        """Process a linear layer"""
        weights = linear.weight.data.numpy()
        self.weights.append(weights)
        
        if linear.bias is not None:
            bias = linear.bias.data.numpy()
        else:
            bias = np.zeros(weights.shape[0])
        
        self.biases.append(bias)
        self.layer_sizes.append((weights.shape[1], weights.shape[0]))  # (input_size, output_size)
    
    def _process_activation(self, activation):
        """Process an activation layer"""
        if isinstance(activation, nn.ReLU):
            self.activation_types.append("relu")
        elif isinstance(activation, nn.Sigmoid):
            self.activation_types.append("sigmoid")
        elif isinstance(activation, nn.Tanh):
            self.activation_types.append("tanh")
        elif isinstance(activation, nn.LeakyReLU):
            self.activation_types.append("leaky_relu")
            # Store alpha value for leaky ReLU
            self.constants[f"leaky_relu_alpha"] = float(activation.negative_slope)
    
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
            print("Attempting alternative loading method...")
            
            # Re-create the model architecture and load just the state_dict
            policy_model = CartPolePolicyModel().to(device)
            policy_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        policy_model = model
    
    # Define configuration for PolyC - keeping individual state variables
    config = {
        "input_vars": {
            "cart_pos": 1,      # Cart position
            "cart_vel": 1,      # Cart velocity
            "pole_angle": 1,    # Pole angle
            "pole_vel": 1       # Pole angular velocity
        },
        "input_order": ["cart_pos", "cart_vel", "pole_angle", "pole_vel"],  # Order of inputs for the neural network
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
        input_order=config["input_order"],
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




def convert_pytorch_model(model, input_vars, output_vars, model_name="PyTorchModel", 
                         constraints=None, goals=None, constants=None, env_vars=None):
    """
    Convert a PyTorch model to PolyC code
    
    Args:
        model: The PyTorch model to convert
        input_vars: Dictionary mapping input variable names to their dimensions
        output_vars: Dictionary mapping output/action variable names to their dimensions
        model_name: Name for the generated model
        constraints: List of constraint expressions
        goals: List of goal expressions
        constants: Dictionary of constants
        env_vars: Dictionary of environment variables
        
    Returns:
        String containing the PolyC code
    """
    converter = PyTorchToPolyC(input_vars, output_vars, constants, env_vars)
    
    if constraints:
        converter.add_constraints(constraints)
    
    if goals:
        converter.add_goals(goals)
    
    converter.extract_model_architecture(model)
    return converter.generate_polyc_code(model_name)


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch neural networks to PolyC code')
    parser.add_argument('model_path', help='Path to the saved PyTorch model (.pt or .pth)')
    parser.add_argument('--config', required=True, help='Path to the configuration JSON file')
    parser.add_argument('--output', help='Output file path for the PolyC code (defaults to model_path with .polyc extension)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load the PyTorch model
    model = torch.load(args.model_path)
    model.eval()  # Set to evaluation mode
    
    # Generate PolyC code
    polyc_code = convert_pytorch_model(
        model,
        input_vars=config.get('input_vars', {}),
        output_vars=config.get('output_vars', {}),
        model_name=config.get('model_name', 'PyTorchModel'),
        constraints=config.get('constraints', []),
        goals=config.get('goals', []),
        constants=config.get('constants', {}),
        env_vars=config.get('env_vars', {})
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.model_path)[0]
        output_path = f"{base_name}.polyc"
    
    # Write the PolyC code to file
    with open(output_path, 'w') as f:
        f.write(polyc_code)
    
    print(f"Successfully converted model to {output_path}")


if __name__ == '__main__':
    main()