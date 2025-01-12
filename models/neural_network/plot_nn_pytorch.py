import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file, skip only row index 1 (second row)
history_0 = pd.read_csv('nn_huge/training_history_0.csv', skiprows=[1,2,3,4,5,6])
history_1 = pd.read_csv('nn_huge/training_history_1.csv', skiprows=[0])
history = pd.concat([history_0, history_1])

# Create the plot
plt.figure(figsize=(14, 6))
plt.plot(history['epoch'], history['train_rmse'], label='Training RMSE', marker='o', markersize=3)
plt.plot(history['epoch'], history['val_rmse'], label='Validation RMSE', marker='o', markersize=3)

# Customize the plot
plt.title('Neural Network Training History')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('nn_huge/training_history_plot_huge.png')
plt.close()
