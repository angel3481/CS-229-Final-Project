import sys
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
import os

import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.linear_regression._linear_regression import load_data, one_hot_encode, target_encode

def initialize_parameters(layer_dims):
    """
    Initialize neural network parameters using He initialization.
    
    Args:
        layer_dims: List containing dimensions of each layer
        
    Returns:
        parameters: Dictionary containing weights (W), biases (b)
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        # He initialization for weights
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        # Initialize biases with small positive values to prevent dead ReLUs
        parameters['b' + str(l)] = np.ones((layer_dims[l], 1)) * 0.01
    
    return parameters

def relu(Z):
    """
    Compute Leaky ReLU activation function.
    
    Args:
        Z: Input numpy array
        
    Returns:
        A: Output of Leaky ReLU(Z), max(0.01*Z, Z)
        cache: Input Z, stored for backward propagation
    """
    A = np.where(Z > 0, Z, 0.01 * Z)
    cache = Z
    return A, cache

def linear_forward(A, W, b):
    """
    Implement forward propagation for a single linear layer.
    
    Args:
        A: activations from previous layer (or input data), shape (size of previous layer, batch size)
        W: weights matrix, shape (size of current layer, size of previous layer)
        b: bias vector, shape (size of current layer, 1)
        
    Returns:
        Z: the input of the activation function, shape (size of current layer, batch size)
        linear_cache: tuple containing (A, W, b) for use in backward pass
    """
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache

def identity(Z):
    """
    Identity activation function.
    
    Args:
        Z: Input numpy array
        
    Returns:
        A: Output (same as input)
        cache: Input Z, stored for backward propagation
    """
    return Z, Z

def linear_activation_forward(A_prev, W, b, activation, keep_prob=1.0, is_training=True):
    """
    Forward propagation for a single layer with dropout.
    
    Args:
        A_prev: activations from previous layer (size of prev layer, batch size)
        W: weights matrix (size of current layer, size of prev layer)
        b: bias vector (size of current layer, 1)
        activation: activation function to use ("softmax" or "relu")
        keep_prob: probability of keeping a neuron active during dropout
        is_training: whether the network is in training mode
        
    Returns:
        A: output activations
        cache: tuple of (linear_cache, activation_cache, dropout_mask)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "identity":
        A, activation_cache = identity(Z)
        D = None
    elif activation == "relu":
        A, activation_cache = relu(Z)
        if is_training and keep_prob < 1.0:
            D = np.random.rand(*A.shape) < keep_prob
            A = (A * D) / keep_prob
        else:
            D = None
    else:
        raise ValueError(f"Activation function {activation} not recognized")
    
    cache = (linear_cache, activation_cache, D)
    return A, cache

def L_model_forward(X, parameters, keep_prob=1.0, is_training=True):
    """
    Forward propagation for the entire neural network with dropout.
    
    Args:
        X: input data of shape (input size, batch size)
        parameters: dictionary containing network parameters W and b
        keep_prob: probability of keeping a neuron active during dropout
        is_training: whether network is in training mode
        
    Returns:
        AL: final layer output
        caches: list of caches for each layer containing:
               (linear_cache, activation_cache, dropout_mask)
    """
    caches = []
    A = X
    L = len([k for k in parameters.keys() if k.startswith('W')])

    # Hidden layers with ReLU activation and dropout
    for l in range(1, L):
        A, cache = linear_activation_forward(
            A, 
            parameters[f'W{l}'], 
            parameters[f'b{l}'], 
            activation="relu",
            keep_prob=keep_prob,
            is_training=is_training
        )
        caches.append(cache)

    # Output layer with identity activation (no dropout)
    AL, cache = linear_activation_forward(
        A, 
        parameters[f'W{L}'], 
        parameters[f'b{L}'], 
        activation="identity"
    )
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y, parameters, lambd=0.01):
    """
    Compute the mean squared error cost with L2 regularization and RMSD.
    
    Args:
        AL: output of the neural network
        Y: true values
        parameters: dictionary containing network parameters W and b
        lambd: L2 regularization parameter (default 0.01)
    
    Returns:
        cost: total cost (MSE + L2 regularization)
        rmsd: root mean square deviation
    """
    m = Y.shape[1]
    
    # Mean squared error
    mse_cost = np.mean(np.square(AL - Y))
    
    # L2 regularization
    L2_cost = 0
    L = len([k for k in parameters.keys() if k.startswith('W')])
    for l in range(L):
        L2_cost += np.sum(np.square(parameters[f'W{l+1}']))
    L2_cost *= (lambd / (2 * m))
    
    # Root mean square deviation
    rmsd = np.sqrt(mse_cost)
    
    return mse_cost + L2_cost, rmsd

def clip_gradients(gradients, threshold=1.0):
    """
    Clips gradients to prevent exploding gradients by limiting their values to [-threshold, threshold].
    
    Args:
        gradients: dictionary of gradients for weights and biases
        threshold: maximum absolute value allowed for gradients (default: 1.0)
    
    Returns:
        gradients: dictionary with clipped gradient values
    """
    for key in gradients:
        np.clip(gradients[key], -threshold, threshold, out=gradients[key])
    return gradients

def linear_backward(dZ, linear_cache, lambd):
    """
    Implements the linear portion of backward propagation with L2 regularization.
    
    Args:
        dZ: Gradient of cost with respect to linear output (dL/dZ)
        linear_cache: Tuple of (A_prev, W, b) containing:
            A_prev: Activations from previous layer
            W: Weight matrix
            b: Bias vector
        lambd: L2 regularization parameter
        
    Returns:
        dA_prev: Gradient of cost with respect to previous layer activation
        dW: Gradient of cost with respect to weights, includes L2 regularization
        db: Gradient of cost with respect to biases
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """
    Compute gradient of the Leaky ReLU activation function.
    
    Args:
        dA: post-activation gradient
        activation_cache: 'Z' value from forward propagation
        
    Returns:
        dZ: Gradient of the cost with respect to Z
    """
    Z = activation_cache
    dZ = dA * np.where(Z > 0, 1, 0.01)  # ReLU derivative: 1 if Z > 0, 0.01 otherwise
    return dZ

def linear_activation_backward(dA, cache, activation, lambd):
    """
    Backward propagation for a single layer with dropout.
    
    Args:
        dA: gradient of cost with respect to current layer's activation
        cache: tuple containing (linear_cache, activation_cache, dropout_mask)
        activation: activation function used in this layer ("relu" or "sigmoid")
        lambd: L2 regularization parameter
        
    Returns:
        dA_prev: gradient of cost with respect to previous layer's activation
        dW: gradient of cost with respect to weights
        db: gradient of cost with respect to biases
    """
    linear_cache, activation_cache, D = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        if D is not None:  # Apply dropout mask if used in forward pass
            dZ = dZ * D
    else:
        raise ValueError(f"Activation {activation} not recognized")
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    return dA_prev, dW, db

def update_parameters_adam(parameters, grads, v, s, t, learning_rate=0.01,
                         beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimization.
    
    Args:
        parameters: dictionary containing network parameters
        grads: dictionary containing gradients
        v: dictionary of exponentially weighted average of gradients
        s: dictionary of exponentially weighted average of squared gradients
        t: iteration number
        learning_rate: learning rate
        beta1: exponential decay rate for first moment estimates
        beta2: exponential decay rate for second moment estimates
        epsilon: small number to prevent division by zero
    
    Returns:
        parameters: updated parameters
        v: updated exponentially weighted average of gradients
        s: updated exponentially weighted average of squared gradients
    """
    L = len([k for k in parameters.keys() if k.startswith('W')])
    v_corrected = {}
    s_corrected = {}
    
    # Prevent t from being 0
    t = max(1, t)  # Start t at 1 instead of 0
    
    # Update rule for each parameter
    for l in range(L):
        # Update for W
        v[f'dW{l+1}'] = beta1 * v[f'dW{l+1}'] + (1 - beta1) * grads[f'dW{l+1}']
        s[f'dW{l+1}'] = beta2 * s[f'dW{l+1}'] + (1 - beta2) * np.square(grads[f'dW{l+1}'])
        
        # Compute bias-corrected first and second moment estimates
        v_corrected[f'dW{l+1}'] = v[f'dW{l+1}'] / (1 - np.power(beta1, t))
        s_corrected[f'dW{l+1}'] = s[f'dW{l+1}'] / (1 - np.power(beta2, t))
        
        # Update W with numerical stability check
        update = learning_rate * (v_corrected[f'dW{l+1}'] / 
                                (np.sqrt(s_corrected[f'dW{l+1}']) + epsilon))
        parameters[f'W{l+1}'] -= update
        
        # Update for b
        v[f'db{l+1}'] = beta1 * v[f'db{l+1}'] + (1 - beta1) * grads[f'db{l+1}']
        s[f'db{l+1}'] = beta2 * s[f'db{l+1}'] + (1 - beta2) * np.square(grads[f'db{l+1}'])
        
        # Compute bias-corrected first and second moment estimates
        v_corrected[f'db{l+1}'] = v[f'db{l+1}'] / (1 - np.power(beta1, t))
        s_corrected[f'db{l+1}'] = s[f'db{l+1}'] / (1 - np.power(beta2, t))
        
        # Update b with numerical stability check
        update = learning_rate * (v_corrected[f'db{l+1}'] / 
                                (np.sqrt(s_corrected[f'db{l+1}']) + epsilon))
        parameters[f'b{l+1}'] -= update
    
    return parameters, v, s

def L_model_backward(AL, Y, caches, lambd):
    """
    Backward propagation for the entire neural network.
    
    Args:
        AL: Output of the neural network (num_classes, batch_size)
        Y: True labels (num_classes, batch_size)
        caches: List of caches from forward propagation containing:
               (linear_cache, activation_cache, dropout_mask) for each layer
        lambd: L2 regularization parameter
        
    Returns:
        grads: Dictionary containing gradients for each parameter:
              dW1, db1, dW2, db2, etc.
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # Output layer gradient (linear)
    dZL = 2 * (AL - Y) / Y.shape[1]
    
    current_cache = caches[L-1]
    linear_cache, _, _ = current_cache
    dA_prev, dW, db = linear_backward(dZL, linear_cache, lambd)
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db

    # Hidden layers gradients (ReLU)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(
            dA_prev, current_cache, "relu", lambd
        )
        grads[f'dW{l+1}'] = dW
        grads[f'db{l+1}'] = db
    
    return grads

def predict(X, parameters):
    """
    Make predictions using the trained neural network.
    
    Args:
        X: input data of shape (n_features, m_samples)
        parameters: dictionary containing trained parameters
        
    Returns:
        predictions: continuous output values
    """
    # Forward propagate through network (using training=False to disable dropout)
    predictions, _ = L_model_forward(X, parameters, keep_prob=1.0, is_training=False)
    return predictions

    
def multiple_layer_model(x_train, y_train, x_val, y_val, layers_dims, initial_lr=0.001, 
                        num_epochs=3000, print_cost=False, lambd=0.01, decay_rate=0.1,
                        batch_size=32, beta1=0.9, beta2=0.999, epsilon=1e-8, keep_prob=0.8):
    """Train a neural network with L layers and dropout regularization."""
    parameters = initialize_parameters(layers_dims)
    v, s = initialize_adam(parameters)
    t = 0
    train_costs = []
    val_costs = []
    
    lr = initial_lr
    m = x_train.shape[1]
    
    try:
        for i in range(num_epochs):
            permutation = np.random.permutation(m)
            shuffled_X = x_train[:, permutation]
            shuffled_Y = y_train[:, permutation]
            
            # Calculate number of batches (including partial batch)
            num_batches = int(np.ceil(m / batch_size))
            
            for k in range(num_batches):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, m)
                minibatch_X = shuffled_X[:, start_idx:end_idx]
                minibatch_Y = shuffled_Y[:, start_idx:end_idx]
                
                AL_train, caches = L_model_forward(minibatch_X, parameters, 
                                                 keep_prob=keep_prob, 
                                                 is_training=True)
                
                # Calculate cost for this batch            
                grads = L_model_backward(AL_train, minibatch_Y, caches, lambd)
                t += 1
                parameters, v, s = update_parameters_adam(
                    parameters, grads, v, s, t, lr, beta1, beta2, epsilon
                )

            AL_train, _ = L_model_forward(x_train, parameters, keep_prob=1.0, is_training=False)
            train_cost, train_rmsd = compute_cost(AL_train, y_train, parameters, lambd)
                
            AL_val, _ = L_model_forward(x_val, parameters,  keep_prob=1.0, is_training=False)
            val_cost, val_rmsd = compute_cost(AL_val, y_val, parameters, lambd)
                
            train_costs.append(train_rmsd)
            val_costs.append(val_rmsd)
                
            if print_cost:
                print(f"Epoch {i}: Train RMSD = {train_rmsd:.4f}, Val RMSD = {val_rmsd:.4f}")
            
            lr = initial_lr * (1.0 / (1.0 + decay_rate * i))
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    return parameters, (train_costs, val_costs)

def initialize_adam(parameters):
    """
    Initialize Adam parameters v and s for all parameters.
    
    Args:
        parameters: dictionary containing network parameters
    
    Returns:
        v: dictionary of exponentially weighted average of gradients
        s: dictionary of exponentially weighted average of squared gradients
    """
    L = len([k for k in parameters.keys() if k.startswith('W')])
    v = {}
    s = {}
    
    for l in range(L):
        v[f'dW{l+1}'] = np.zeros_like(parameters[f'W{l+1}'])
        v[f'db{l+1}'] = np.zeros_like(parameters[f'b{l+1}'])
        s[f'dW{l+1}'] = np.zeros_like(parameters[f'W{l+1}'])
        s[f'db{l+1}'] = np.zeros_like(parameters[f'b{l+1}'])
    
    return v, s

def main(encoding_type, initial_lr, num_epochs, print_cost, lambd, decay_rate, batch_size, beta1, beta2, epsilon, keep_prob, i=None):
    """
    Main function to train and evaluate the neural network model.
    Loads data, trains model, plots learning curves and reports accuracies.
    """
    # Update the data path to use the parent directory
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    if encoding_type == "one-hot":
        X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test)
    elif encoding_type == "target":
        X_train, X_val, X_test = target_encode(X_train, X_val, X_test, Y_train)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.T, Y_train.T, X_val.T, Y_val.T, X_test.T, Y_test.T


    # Define network architecture
    n_features = X_train.shape[0]
    n_classes = Y_train.shape[0]
    layers_dims = [n_features, 16, 16, 16, n_classes]
    print(f"Architecture: {layers_dims}")
    
    # Train model with Adam optimizer
    parameters, costs = multiple_layer_model(
        X_train, Y_train,
        X_val, Y_val,
        layers_dims=layers_dims,
        initial_lr=initial_lr,
        num_epochs=num_epochs,
        print_cost=print_cost,
        lambd=lambd,  
        decay_rate=decay_rate,
        batch_size=batch_size,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        keep_prob=keep_prob
    )
        
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(costs[0])) , costs[0], 'b-', label='Train')
    plt.plot(np.arange(len(costs[1])) , costs[1], 'r-', label='Validation')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Cost over time')
    plt.legend()
    plt.tight_layout()
    if i is not None:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/small_nn_cost_{encoding_type}.png')
    else:
        plt.savefig(f'plots/small_nn_{encoding_type}.png')
    plt.close()
        
    # Calculate and report final MSE
    train_predictions = predict(X_train, parameters)
    val_predictions = predict(X_val, parameters)
    test_predictions = predict(X_test, parameters)
    train_rmse = np.sqrt(np.mean((train_predictions - Y_train) ** 2))
    val_rmse = np.sqrt(np.mean((val_predictions - Y_val) ** 2))
    test_rmse = np.sqrt(np.mean((test_predictions - Y_test) ** 2))


    print(f"\nFinal Mean Squared Error:")
    print(f"Training Set RMSE: {train_rmse:.4f}")
    print(f"Validation Set RMSE: {val_rmse:.4f}")
    print(f"Test Set RMSE: {test_rmse:.4f}")
if __name__ == "__main__":
    #main("label", 0.001, 60, True, 0.0, 0.0, 32, 0.9, 0.999, 1e-8, 1.0)
    #main("target", 0.001, 60, True, 0.0, 0.0, 32, 0.9, 0.999, 1e-8, 1.0)
    lrs = [10 ** np.random.uniform(-3.5, -3) for _ in range(5)]
    batch_sizes = [2 ** np.random.randint(3, 5) for _ in range(5)]
    for i in range(1):
        print(f"Running experiment {i+1} of 1")
        print(f"Learning rate: {lrs[i]:.4f}, Batch size: {batch_sizes[i]}")
        main("one-hot", 0.004, 100, True, 0.0, 0.1, 16, 0.9, 0.999, 1e-8, 1.0, i+17)
