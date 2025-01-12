import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DeepRegressor(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(DeepRegressor, self).__init__()
        
        # First hidden layer
        self.layer1 = nn.Sequential(
            nn.Linear(497, 192),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )
        
        # First pair of 192-neuron layers with skip connection
        self.block1 = nn.Sequential(
            nn.Linear(192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )
        
        # Transition layer to 128
        self.transition = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Second pair of 128-neuron layers with skip connection
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Output layer
        self.final = nn.Linear(128, 1)
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        self.dropout_rate = dropout_rate

    def update_dropout(self, p):
        """Update dropout rate during training"""
        self.dropout_rate = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def forward(self, x):
        # First layer
        x = self.layer1(x)
        
        # First skip connection block (192)
        identity1 = x
        x = self.block1(x)
        x = x + identity1
        
        # Transition to 128
        x = self.transition(x)
        
        # Second skip connection block (128)
        identity2 = x
        x = self.block2(x)
        x = x + identity2
        
        return self.final(x)

def train_model(model, train_loader, val_loader, device, max_time=30*60, 
                initial_lr=0.001, max_grad_norm=1.0):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5, 
                                                    verbose=True)
    
    train_losses = []
    val_losses = []
    start_time = time()
    epoch = 0
    best_val_loss = float('inf')
    
    while (time() - start_time) < max_time:
        model.train()
        train_loss = 0
        
        # Get total number of batches
        total_batches = len(train_loader)
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Create progress bar
            progress = int(30 * (batch_idx + 1) / total_batches)
            progress_bar = f"\r[{'=' * progress}{' ' * (30-progress)}] |{batch_idx+1}/{total_batches}"
            print(progress_bar, end='', flush=True)
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
        
        print()  # New line after progress bar
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_train_loss = np.sqrt(train_loss/len(train_loader))
        avg_val_loss = np.sqrt(val_loss/len(val_loader))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print current metrics as percentages
        print(f'Epoch {epoch:3d} | '
              f'Train RMSD: {avg_train_loss*100:.2f}% | '
              f'Dev RMSD: {avg_val_loss*100:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Learning rate scheduling based on validation loss
        scheduler.step(avg_val_loss)
        
        epoch += 1
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Save hyperparameters and results together with model weights
    model_save = {
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'initial_lr': initial_lr,
            'max_grad_norm': max_grad_norm,
            'dropout_rate': model.dropout_rate,
            'batch_size': train_loader.batch_size,
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'architecture': {
                'input_dim': 497,
                'hidden_dims': [192, 192, 128, 128],
                'output_dim': 1
            }
        },
        'results': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch,
            'training_time': time() - start_time,
            'final_lr': optimizer.param_groups[0]['lr']
        }
    }
    
    # Save complete model info
    torch.save(model_save, 'model_complete.pt')
    
    # Save training history to CSV
    pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_rmse': train_losses,
        'val_rmse': val_losses
    }).to_csv('training_history.csv', index=False)
    
    # Final plot save
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training RMSD')
    plt.plot(val_losses, label='Validation RMSD')
    plt.xlabel('Epoch')
    plt.ylabel('RMSD')
    plt.title('Training and Validation RMSD over Time')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    return model, train_losses, val_losses, model_save

class RegressionDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = torch.FloatTensor(pd.read_csv(x_path).values)
        self.y = torch.FloatTensor(pd.read_csv(y_path).values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
train_dataset = RegressionDataset('csv/x_train.csv', 'csv/y_train.csv')
dev_dataset = RegressionDataset('csv/x_dev.csv', 'csv/y_dev.csv')
test_dataset = RegressionDataset('csv/x_test.csv', 'csv/y_test.csv')

# Check input dimension before proceeding
print(f"Training data input dimension: {train_dataset.X.shape[1]}")
print(f"First few features of first sample: {train_dataset.X[0][:5]}")

# Verify the shape matches our expected architecture
assert train_dataset.X.shape[1] == 497, f"Expected input dimension of 497, but got {train_dataset.X.shape[1]}"

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Initialize model and train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepRegressor()
model, train_losses, val_losses, model_save = train_model(model, train_loader, val_loader, device)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # Handle both full dictionary and direct state dict saves
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Store predictions and actual values
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # Calculate RMSD
    avg_rmsd = np.sqrt(total_loss/len(test_loader))
    
    # Convert to numpy arrays for easier analysis
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Save predictions to CSV
    pd.DataFrame({
        'actual': actuals.flatten(),
        'predicted': predictions.flatten()
    }).to_csv('test_predictions.csv', index=False)
    
    print(f'Test Set RMSD: {avg_rmsd*100:.2f}%')
    
    return avg_rmsd, predictions, actuals

# After loading the best model, evaluate it
model = load_checkpoint(model, 'best_model.pt')
test_rmsd, test_predictions, test_actuals = evaluate_model(model, test_loader, device)

