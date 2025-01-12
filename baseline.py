import numpy as np
import pandas as pd

# Read the CSV files
predictions = pd.read_csv('datasets/benchmark_test.csv').values.flatten()
y_test = pd.read_csv('datasets/y_test.csv').values.flatten()

# Calculate RMSD
rmsd = np.sqrt(np.mean((predictions - y_test) ** 2))

# Calculate NRMSD (normalized by mean of y_test)
nrmsd = rmsd / np.mean(y_test)

print(f"""
      Manheim Market Report NRMSD on the test set: {100*rmsd:.4f}%
""")

