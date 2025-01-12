import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.special import digamma
import pandas as pd
import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from models.linear_regression._linear_regression import load_data, one_hot_encode, target_encode

class GammaGLM:
    def __init__(self, link='log', max_iter=100, tol=1e-5):
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.scale_ = None
    
    def _link_function(self, mu):
        """Log link function"""
        return np.log(mu)
    
    def _inverse_link(self, eta):
        """Inverse of log link function"""
        return np.exp(eta)
    
    def fit(self, X, y):
        """
        Fit the Gamma GLM using IRLS (Iteratively Reweighted Least Squares)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (must be positive)
        """
        if np.any(y <= 0):
            raise ValueError("Target values must be positive for Gamma GLM")
        
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        # Initialize coefficients using linear regression on log(y)
        self.coef_ = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
        
        for i in range(self.max_iter):
            print(f"Iteration {i}")
            
            old_coef = self.coef_.copy()
            
            # 1. Compute current predictions
            eta = X @ self.coef_
            mu = self._inverse_link(eta)
            
            # 2. Compute working dependent variable and weights
            z = eta + (y - mu) / mu
            weights = mu ** 2
            
            # 3. Update coefficients using weighted least squares
            weighted_X = X * np.sqrt(weights.reshape(-1, 1))
            weighted_z = z * np.sqrt(weights)
            self.coef_ = np.linalg.lstsq(weighted_X, weighted_z, rcond=None)[0]
            
            change = np.sum((self.coef_ - old_coef) ** 2)
            print(f"Change: {change}")
            # Check convergence
            if change < self.tol:
                break
        
        # Estimate scale parameter (shape parameter of gamma distribution)
        residuals = (y - mu) / mu
        self.scale_ = np.mean(residuals ** 2)
        
        return self
    
    def predict(self, X):
        """Predict using the fitted model"""
        X = np.asarray(X)
        eta = X @ self.coef_
        return self._inverse_link(eta)
    
    def score(self, X, y):
        """
        Calculate the pseudo R-squared (based on deviance)
        
        Returns:
        --------
        score : float
            Pseudo R-squared score
        """
        y_pred = self.predict(X)
        null_deviance = -2 * np.sum(np.log(y/np.mean(y)) - (y - np.mean(y))/np.mean(y))
        model_deviance = -2 * np.sum(np.log(y/y_pred) - (y - y_pred)/y_pred)
        return 1 - model_deviance/null_deviance
    
    @classmethod
    def load_and_prepare_data(cls):
        """
        Load and prepare the bike sharing dataset
        
        Returns:
        --------
        tuple : (X_train, X_dev, X_test, y_train, y_dev, y_test)
            Prepared data splits
        """
        
        # Load all datasets
        X_train, y_train, X_dev, y_dev, X_test, y_test = load_data()
        X_train, X_dev, X_test = one_hot_encode(X_train, X_dev, X_test)
        
        # Add a small positive constant to ensure all values are positive
        epsilon = 1e-6
        y_train = y_train + epsilon
        y_dev = y_dev + epsilon
        y_test = y_test + epsilon
        
        return (X_train, X_dev, X_test,
                y_train, y_dev, y_test)

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_dev, X_test, y_train, y_dev, y_test = GammaGLM.load_and_prepare_data()
    print("Data shapes:")
    print(f"Training: {X_train.shape}, {y_train.shape}")
    print(f"Dev: {X_dev.shape}, {y_dev.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    
    # Initialize and train the model
    model = GammaGLM(max_iter=100, tol=1e-4)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    # Calculate RMSD for both sets
    y_pred_train = model.predict(X_train)
    y_pred_dev = model.predict(X_dev)
    y_pred_test = model.predict(X_test)
    
    rmsd_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    rmsd_dev = np.sqrt(np.mean((y_dev - y_pred_dev) ** 2))
    rmsd_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    
    print("\nModel Performance:")
    print(f"Training set - RMSD: {100*rmsd_train:.2f}%")
    print(f"Dev set     - RMSD: {100*rmsd_dev:.2f}%")
    print(f"Test set    - RMSD: {100*rmsd_test:.2f}%")