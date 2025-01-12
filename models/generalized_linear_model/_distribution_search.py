import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt

def find_best_distribution(data):
    """
    Analyze data to find the best-fitting distribution among common choices for positive data.
    
    Parameters:
    -----------
    data : array-like
        The positive numerical data to analyze
        
    Returns:
    --------
    dict : Contains best distribution name and its fit statistics
    """
    # Common GLM distributions for continuous data
    distributions = [
        ('gamma', stats.gamma),      # For positive continuous data with constant CV
        ('gaussian', stats.norm),    # For normally distributed continuous data
        ('inverse_gaussian', stats.invgauss)  # For positive continuous data with larger variance
    ]
    
    best_dist = None
    best_fit = float('inf')
    best_params = None
    
    for name, distribution in distributions:
        try:
            # Fit distribution parameters
            params = distribution.fit(data)
            
            # Calculate Akaike Information Criterion (AIC)
            try:
                log_likelihood = np.sum(distribution.logpdf(data, *params))
                if np.isfinite(log_likelihood):  # Check for valid likelihood
                    k = len(params)
                    aic = 2 * k - 2 * log_likelihood
                    
                    if aic < best_fit:
                        best_fit = aic
                        best_dist = name
                        best_params = params
            except (RuntimeWarning, RuntimeError, ValueError, OverflowError):
                continue
                
        except (RuntimeWarning, RuntimeError, ValueError, OverflowError):
            continue
            
    if best_dist is None:
        raise ValueError("Could not fit any distribution to the data")
            
    return {
        'distribution': best_dist,
        'parameters': best_params,
        'aic': best_fit
    }

def plot_distribution_fit(data, dist_info):
    """
    Plot histogram of data with fitted distribution overlay
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
    
    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 100)
    dist = getattr(stats, dist_info['distribution'])
    y = dist.pdf(x, *dist_info['parameters'])
    plt.plot(x, y, 'r-', label=f'Fitted {dist_info["distribution"]}')
    
    plt.title('Data Distribution with Best Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def main():
    # If the CSV has only one column, we should use usecols=0 instead of 1
    # since Python uses 0-based indexing
    data = np.loadtxt('../../datasets/y_train.csv', delimiter=',', skiprows=1, usecols=0)
    dist_info = find_best_distribution(data)
    plot_distribution_fit(data, dist_info)

if __name__ == "__main__":
    main()