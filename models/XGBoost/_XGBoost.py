import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import os

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        # Navigate to dataset directory
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
        
        # Load training data
        X_train = pd.read_csv(os.path.join(base_path, 'x_train.csv')).iloc[:, 1:]
        y_train = pd.read_csv(os.path.join(base_path, 'y_train.csv'))
        
        # Load validation data
        X_dev = pd.read_csv(os.path.join(base_path, 'x_dev.csv')).iloc[:, 1:]
        y_dev = pd.read_csv(os.path.join(base_path, 'y_dev.csv'))
        
        # Load test data
        X_test = pd.read_csv(os.path.join(base_path, 'x_test.csv')).iloc[:, 1:]
        y_test = pd.read_csv(os.path.join(base_path, 'y_test.csv'))
        
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    
    def preprocess_data(self, X):
        # Assuming the first 6 columns are categorical and the last 6 are numerical
        categorical_cols = X.columns[:6]
        numerical_cols = X.columns[6:]
        
        # Scale numerical features
        X_numerical = self.scaler.fit_transform(X[numerical_cols])
        
        # Combine categorical and scaled numerical features
        X_processed = pd.concat([
            X[categorical_cols].reset_index(drop=True),
            pd.DataFrame(X_numerical, columns=numerical_cols)
        ], axis=1)
        
        return X_processed
    
    def objective(self, trial, X_train, y_train, X_dev, y_dev):
        # Define hyperparameter search space
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        }
        
        # Create and train model
        model = XGBRegressor(**param, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_dev)
        rmse = np.sqrt(mean_squared_error(y_dev, y_pred))
        
        return rmse
    
    def train(self, X_train, y_train, X_dev, y_dev):
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_dev, y_dev),
                      n_trials=50)
        
        # Get best parameters
        best_params = study.best_params
        
        # Train final model with best parameters
        self.model = XGBRegressor(**best_params, random_state=42)
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        return {
            'RMSE': rmse,
            'R2': r2
        }
    
    def run(self):
        # Load data
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.load_data()
        
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train)
        X_dev_processed = self.preprocess_data(X_dev)
        X_test_processed = self.preprocess_data(X_test)
        
        # Train model
        print("Training model...")
        self.train(X_train_processed, y_train, X_dev_processed, y_dev)
        
        # Evaluate on all sets
        print("\nEvaluation Results:")
        print("Training Set:", self.evaluate(X_train_processed, y_train))
        print("Validation Set:", self.evaluate(X_dev_processed, y_dev))
        print("Test Set:", self.evaluate(X_test_processed, y_test))
        
        return self.model

if __name__ == "__main__":
    xgb_model = XGBoostModel()
    model = xgb_model.run()
