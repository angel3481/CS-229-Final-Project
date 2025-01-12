import numpy as np
import os
import matplotlib.pyplot as plt

def load_data():
    """
    Loads training, development and test data from CSV files.
    
    The data files are expected to be in a 'datasets' directory one level up from 
    the current file location. The data has been preprocessed and scaled using 
    min-max scaling.
    
    Returns:
        tuple: A tuple containing:
            - x_train (np.ndarray): Training features
            - y_train (np.ndarray): Training target values (prices), shape (n_samples, 1)
            - x_dev (np.ndarray): Development features  
            - y_dev (np.ndarray): Development target values, shape (n_samples, 1)
            - x_test (np.ndarray): Test features
            - y_test (np.ndarray): Test target values, shape (n_samples, 1)
    """
    # Get the path to the datasets directory (one level up from current file)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets')

    # Use usecols to skip first column and specify dtype as float32
    x_train = np.loadtxt(os.path.join(dataset_path, 'x_train.csv'), 
                         delimiter=',', 
                         dtype=np.float32,
                         skiprows=1,  # Skip header row if it exists
                         usecols=range(1, 13))  # Changed from 100 to 13
    y_train = np.loadtxt(os.path.join(dataset_path, 'y_train.csv'), 
                         delimiter=',',
                         dtype=np.float32,
                         skiprows=1).reshape(-1, 1)

    x_dev = np.loadtxt(os.path.join(dataset_path, 'x_dev.csv'), 
                       delimiter=',',
                       dtype=np.float32,
                       skiprows=1,
                       usecols=range(1, 13))
    y_dev = np.loadtxt(os.path.join(dataset_path, 'y_dev.csv'), 
                       delimiter=',',
                       dtype=np.float32,
                       skiprows=1).reshape(-1, 1)

    x_test = np.loadtxt(os.path.join(dataset_path, 'x_test.csv'), 
                        delimiter=',',
                        dtype=np.float32,
                        skiprows=1,
                        usecols=range(1, 13))
    y_test = np.loadtxt(os.path.join(dataset_path, 'y_test.csv'), 
                        delimiter=',',
                        dtype=np.float32,
                        skiprows=1).reshape(-1, 1)

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def one_hot_encode(x_train, x_dev, x_test):
    """
    Performs one-hot encoding on the first 8 categorical features of the input data.
    
    Args:
        x_train (np.ndarray): Training features
        x_dev (np.ndarray): Development features
        x_test (np.ndarray): Test features
        
    Returns:
        tuple: Transformed versions of x_train, x_dev, and x_test with one-hot encoded features
    """
    # Extract numerical features (after first 8 columns)
    num_features_train = x_train[:, 8:]
    num_features_dev = x_dev[:, 8:]
    num_features_test = x_test[:, 8:]
    
    # Initialize separate lists for each dataset
    encoded_features_train = []
    encoded_features_dev = []
    encoded_features_test = []
    
    # Process each categorical feature
    for feature_idx in range(8):
        # Get unique values from training set for this feature
        unique_values = np.unique(x_train[:, feature_idx])
        
        # Create encoded arrays for each dataset separately
        encoded_train = np.zeros((x_train.shape[0], len(unique_values)))
        encoded_dev = np.zeros((x_dev.shape[0], len(unique_values)))
        encoded_test = np.zeros((x_test.shape[0], len(unique_values)))
        
        for i, value in enumerate(unique_values):
            encoded_train[:, i] = (x_train[:, feature_idx] == value).astype(float)
            encoded_dev[:, i] = (x_dev[:, feature_idx] == value).astype(float)
            encoded_test[:, i] = (x_test[:, feature_idx] == value).astype(float)
            
        encoded_features_train.append(encoded_train)
        encoded_features_dev.append(encoded_dev)
        encoded_features_test.append(encoded_test)
    
    # Combine encoded categorical features with numerical features
    x_train_encoded = np.hstack([np.hstack(encoded_features_train), num_features_train])
    x_dev_encoded = np.hstack([np.hstack(encoded_features_dev), num_features_dev])
    x_test_encoded = np.hstack([np.hstack(encoded_features_test), num_features_test])

    return x_train_encoded, x_dev_encoded, x_test_encoded

def target_encode(x_train, x_dev, x_test, y_train):
    """
    Performs target encoding on the first 8 categorical features of the input data.
    
    Args:
        x_train (np.ndarray): Training features
        x_dev (np.ndarray): Development features
        x_test (np.ndarray): Test features
        y_train (np.ndarray): Training target values
        
    Returns:
        tuple: Transformed versions of x_train, x_dev, and x_test with target encoded features
    """
    # Initialize list to store encoded features
    encoded_features_train = []
    encoded_features_dev = []
    encoded_features_test = []
    
    # Process each categorical feature
    for feature_idx in range(8):
        # Get unique values from training set for this feature
        unique_values = np.unique(x_train[:, feature_idx])
        
        # Calculate mean of y_train for each unique value
        target_means = {value: y_train[x_train[:, feature_idx] == value].mean() for value in unique_values}
        
        # Encode each dataset
        encoded_train = np.array([target_means[value] for value in x_train[:, feature_idx]]).reshape(-1, 1)
        encoded_dev = np.array([target_means.get(value, np.mean(y_train)) for value in x_dev[:, feature_idx]]).reshape(-1, 1)
        encoded_test = np.array([target_means.get(value, np.mean(y_train)) for value in x_test[:, feature_idx]]).reshape(-1, 1)
        
        encoded_features_train.append(encoded_train)
        encoded_features_dev.append(encoded_dev)
        encoded_features_test.append(encoded_test)
    
    # Combine encoded categorical features with numerical features
    num_features_train = x_train[:, 8:]
    num_features_dev = x_dev[:, 8:]
    num_features_test = x_test[:, 8:]
    
    x_train_encoded = np.hstack(encoded_features_train + [num_features_train])
    x_dev_encoded = np.hstack(encoded_features_dev + [num_features_dev])
    x_test_encoded = np.hstack(encoded_features_test + [num_features_test])
    
    return x_train_encoded, x_dev_encoded, x_test_encoded

def initialize_parameters(n_features):
    """
    Initialize parameters for linear regression model.
    
    Args:
        n_features (int): Number of input features
        
    Returns:
        tuple: A tuple containing:
            - w (np.ndarray): Weight parameters of shape (n_features, 1), initialized to zeros
            - b (float): Bias parameter, initialized to 0
    """
    # Initialize weights to zeros vector
    w = np.zeros(n_features).reshape(-1, 1)
    # Initialize bias to zero 
    b = 0
    return w, b

def compute_cost(w, y_pred, y, lambda_reg):
    """
    Computes the cost function for linear regression with L2 regularization.
    
    Args:
        w (np.ndarray): Weight parameters of shape (n_features, 1)
        y_pred (np.ndarray): Predicted values of shape (n_samples, 1)
        y (np.ndarray): True target values of shape (n_samples, 1)
        lambda_reg (float): L2 regularization parameter
        
    Returns:
        float: Total cost (MSE loss + L2 regularization)
    """
    # Get number of training examples
    m = len(y)
    cost = np.sum((y_pred - y) ** 2) / (2 * m)
    reg_cost = lambda_reg * np.sum(w ** 2) / (2 * m)
    rmsd = np.sqrt(np.mean((y_pred - y) ** 2))

    # Return total cost
    return cost + reg_cost, rmsd



def fit(x_train, y_train, batch_size=32, learning_rate=0.001, num_epochs=100, lambda_reg=0, print_cost=False):
    """
    Fits a linear regression model using mini-batch gradient descent.
    
    Args:
        x_train (np.ndarray): Training features of shape (n_samples, n_features)
        y_train (np.ndarray): Training labels of shape (n_samples, 1)
        batch_size (int): Size of mini-batches for gradient descent
        learning_rate (float): Learning rate for gradient descent
        num_epochs (int): Number of training epochs
        lambda_reg (float): L2 regularization parameter
        print_cost (bool): Whether to print cost during training
        
    Returns:
        tuple: Trained model parameters (weights w, bias b)
    """
    # Initialize model parameters
    w, b = initialize_parameters(x_train.shape[1])
    
    for epoch in range(num_epochs):
        # Shuffle the data
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        num_batches = len(x_train) // batch_size
        for i in range(num_batches):
            # Get mini-batch
            x_batch = x_train_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = y_train_shuffled[i*batch_size:(i+1)*batch_size]
            
            # Forward pass
            y_pred = np.dot(x_batch, w) + b
            
            # Compute gradients with gradient clipping
            grad_w = (np.dot(x_batch.T, (y_pred - y_batch)) / batch_size) + (lambda_reg * w / batch_size)
            grad_b = np.mean(y_pred - y_batch)
            
            # Clip gradients to prevent explosion
            grad_w = np.clip(grad_w, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)
            
            # Update parameters
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b

        # Print cost metrics if requested
        if print_cost and epoch % 10 == 0:
            y_pred = np.dot(x_train, w) + b
            cost, rmsd = compute_cost(w, y_pred, y_train, lambda_reg)
            print(f"Cost after epoch {epoch + 1}: {cost}")
            print(f"RMSD after epoch {epoch + 1}: {rmsd}")

    return w, b

def predict(x, w, b):
    """
    Predicts the target values for new data using the trained model.
    """
    return np.dot(x, w) + b

def linear_regression_model(dataset, batch_size=32, learning_rate=0.01, num_epochs=100, lambda_reg=0, print_cost=True, test_time=False):
    """
    Main function to run the linear regression model.
    
    Args:
        dataset (tuple): Tuple containing (x_train, y_train, x_dev, y_dev, x_test, y_test)
        batch_size (int): Size of mini-batches for training
        learning_rate (float): Learning rate for gradient descent
        num_epochs (int): Number of training epochs
        lambda_reg (float): L2 regularization parameter
        print_cost (bool): Whether to print cost during training
        
    Returns:
        tuple: Contains:
            w (ndarray): Trained model weights
            b (float): Trained model bias
            train_cost (float): Final training cost
            dev_cost (float): Final development cost  
            train_rmsd (float): Final training RMSD
            dev_rmsd (float): Final development RMSD
    """
    x_train, y_train, x_dev, y_dev, x_test, y_test = dataset

    w, b = fit(x_train, y_train, batch_size=batch_size, learning_rate=learning_rate, 
               num_epochs=num_epochs, lambda_reg=lambda_reg, print_cost=print_cost)
    
    train_cost, train_rmsd = compute_cost(w, predict(x_train, w, b), y_train, lambda_reg)
    dev_cost, dev_rmsd = compute_cost(w, predict(x_dev, w, b), y_dev, lambda_reg)
    
    if test_time:
        test_cost, test_rmsd = compute_cost(w, predict(x_test, w, b), y_test, lambda_reg)
        print(f"Final test RMSD: {test_rmsd:.6f}")
    
    print(f"Final training RMSD: {train_rmsd:.6f}")
    print(f"Final development RMSD: {dev_rmsd:.6f}")
    
    return w, b, train_cost, dev_cost, train_rmsd, dev_rmsd

def hyperparameter_search(encoding_type, num_trials=10, bounds=None):
    """
    Performs random search over hyperparameters for the linear regression model.
    
    Args:
        encoding_type (str): Type of encoding to use ('one-hot' or 'target')
        num_trials (int): Number of random hyperparameter combinations to try
        
    Returns:
        tuple: Contains:
            results (list): List of tuples with (batch_size, learning_rate, train_rmsd, dev_rmsd)
            best_train_rmsd (float): Best RMSD achieved on training set
            best_dev_rmsd (float): Best RMSD achieved on development set
    """
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()
    if encoding_type == "one-hot":
        x_train, x_dev, x_test = one_hot_encode(x_train, x_dev, x_test)
    elif encoding_type == "target":
        x_train, x_dev, x_test = target_encode(x_train, x_dev, x_test, y_train)
    elif encoding_type == "label":
        pass

    dataset = (x_train, y_train, x_dev, y_dev, x_test, y_test)
    
    if bounds is not None:
        batch_sizes_min, batch_sizes_max = bounds[0]
        learning_rates_min, learning_rates_max = bounds[1]
    else:
        batch_sizes_min, batch_sizes_max = 4, 8
        learning_rates_min, learning_rates_max = -4, -2

    best_train_rmsd = float('inf')
    best_dev_rmsd = float('inf')
    results = []

    for i in range(num_trials):
        print(f"Trial {i + 1} of {num_trials}")

        batch_size = 2 ** np.random.randint(batch_sizes_min, batch_sizes_max+1)
        learning_rate = 10 ** np.random.uniform(learning_rates_min, learning_rates_max)

        w, b, train_cost, dev_cost, train_rmsd, dev_rmsd = linear_regression_model(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=80,
            lambda_reg=0,
            print_cost=True,
            test_time=True # Set to True to test on test set
        )

        results.append((batch_size, learning_rate, train_rmsd, dev_rmsd))

        if train_rmsd < best_train_rmsd:
            best_train_rmsd = train_rmsd

        if dev_rmsd < best_dev_rmsd:
            best_dev_rmsd = dev_rmsd

    return results, best_train_rmsd, best_dev_rmsd

def plot_results(results, encoding_type, dataset_type):
    """
    Creates a scatter plot visualizing the relationship between hyperparameters and model performance.
    
    Args:
        results (list): List of tuples containing (batch_size, learning_rate, train_rmsd, dev_rmsd)
        encoding_type (str): Type of encoding used ('one-hot', 'target', or 'label')
        dataset_type (str): Which dataset results to plot ('train' or 'dev')
        
    The plot shows batch size vs learning rate, with RMSD values represented by color.
    Saves the plot as a PNG file named '{encoding_type}_{dataset_type}_rmsd.png'.
    """
    batch_sizes, learning_rates, rmsds = zip(*[(r[0], r[1], r[2] if dataset_type == 'train' else r[3]) for r in results])
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(batch_sizes, learning_rates, c=rmsds, cmap='viridis_r', s=100)
    plt.colorbar(scatter, label='RMSD')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title(f'{encoding_type.capitalize()} Encoding - {dataset_type.capitalize()} Set RMSD')
    plt.savefig(f'{encoding_type}_{dataset_type}_rmsd.png')
    plt.close()

if __name__ == "__main__":
    
    #results, best_train_rmsd, best_dev_rmsd = hyperparameter_search("label", num_trials=20, bounds=((4, 8), (-5, -3)))
    #plot_results(results, "label", 'train')
    #plot_results(results, "label", 'dev')
    #print(f"Best train RMSD for label encoding: {best_train_rmsd:.6f}")
    #print(f"Best dev RMSD for label encoding: {best_dev_rmsd:.6f}")

    results, best_train_rmsd, best_dev_rmsd = hyperparameter_search("one-hot", num_trials=1, bounds=((4, 4), (-2.0, -1.9)))
    plot_results(results, "one-hot", 'train')
    plot_results(results, "one-hot", 'dev')
    print(f"Best train RMSD for one-hot encoding: {best_train_rmsd:.6f}")
    print(f"Best dev RMSD for one-hot encoding: {best_dev_rmsd:.6f}")

    #results, best_train_rmsd, best_dev_rmsd = hyperparameter_search("target", num_trials=20, bounds=((4,8), (-2.5, -1.5)))
    #plot_results(results, "target", 'train')
    #plot_results(results, "target", 'dev')
    #print(f"Best train RMSD for target encoding: {best_train_rmsd:.6f}")
    #print(f"Best dev RMSD for target encoding: {best_dev_rmsd:.6f}")

