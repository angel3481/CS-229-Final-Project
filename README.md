# CS-229 Final Project: Used Car Price Prediction

This project aims to predict used car prices using machine learning models, surpassing the performance of the state-of-the-art benchmark. The Manheim Market Report (MMR) achieved a Normalized Root Mean Squared Deviation (NRMSD) of **3.30%** on the test set, while our model achieved a competitive NRMSD of **3.48%**.

---

## **Project Overview**
Predicting used car prices accurately is a challenging problem due to variability in car attributes, market trends, and conditions. This project leverages advanced machine learning techniques and enriched features derived by tracking Vehicle Identification Numbers (VINs) to create a robust price prediction model.

The original dataset was sourced from [Kaggle's Used Car Auction Prices dataset](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices). Additional features were added by processing the VINs, making the dataset more comprehensive and improving the model's predictive capabilities.

---

## **Inputs and Outputs**
- **Input Features**:
  - `vin`: Vehicle Identification Number
  - `make`: Car manufacturer (e.g., Toyota, Ford)
  - `model`: Specific model (e.g., Camry, Mustang)
  - `body`: Body type (e.g., sedan, SUV)
  - `transmission`: Transmission type (e.g., automatic, manual)
  - `state`: Registration state
  - `exterior_color`: Car's exterior color
  - `interior_color`: Car's interior color
  - `fuel_type`: Type of fuel (e.g., gasoline, diesel, electric)
  - `year`: Manufacturing year
  - `condition`: Car condition (e.g., excellent, good)
  - `odometer`: Mileage (in miles)
  - `engine_volume`: Engine size (in liters)

- **Output**:
  - **Predicted car price**: The estimated market value of the car in USD.

---

## **Project Structure**
- **`data_processing/`**: 
  - **`data_preprocessing.py`**: Script for data cleaning, preprocessing, and feature engineering.
- **`datasets/`**: Processed datasets used for training and evaluation.
- **`models/`**: Contains implementations of the following models:
  1. **Generalized Linear Model**:
     - `generalized_linear_model.py`
  2. **Linear Regression**:
     - `linear_regression.py`
  3. **XGBoost**:
     - `XGBoost.py`
  4. **Neural Networks**:
     - **Small Model**: `neural_network_numpy_small.py`
     - **Medium Model**: `neural_network_pytorch_medium.py`
     - **Large Model**: `neural_network_pytorch_large.py`
     - **Huge Model**: `neural_network_pytorch_huge.py`
- **`baseline.py`**: A baseline implementation for comparison.
- **`README.md`**: This file.

---

## **How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/angel3481/CS-229-Final-Project.git
   cd CS-229-Final-Project
   ```

2. **Run Data Preprocessing**:
   Process the raw dataset and add VIN-based features:
   ```bash
   python data_processing/data_preprocessing.py
   ```

3. **Train Models**:
   Select and train any of the models from the `models/` directory. For example:
   - Generalized Linear Model:
     ```bash
     python models/generalized_linear_model.py
     ```
   - Neural Network (Huge):
     ```bash
     python models/neural_network_pytorch_huge.py
     ```

4. **Predict Prices**:
   Use a trained model to predict car prices. Example:
   ```bash
   python models/neural_network_pytorch_huge.py --input <path_to_input_file> --output <path_to_output_file>
   ```

---

## **Results**
- **Benchmark Performance**:
  - **Manheim Market Report (MMR)**: 3.30% NRMSD
- **Our Performance**:
  - **Neural Network (Large)**: 3.51% NRMSD
  - **Neural Network (Huge)**: 3.48% NRMSD
  
These results demonstrate competitive performance, with the huge neural network achieving state-of-the-art results.

---

## Documentation
- [Final Project Paper (PDF)](docs/CS_229_Final_Project.pdf): Detailed report describing the methodology, experiments, and results of this project.

---

## **Dataset**
- **Original Dataset**:
  - Sourced from [Kaggle's Used Car Auction Prices dataset](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices).
- **Enhanced Dataset**:
  - Additional features were created by processing VINs, enabling more detailed predictions.

---

## **Future Work**
- Further optimize the huge neural network for improved performance.
- Integrate external market trend data for enhanced accuracy.

---

## **Contributors**
- Angel Raychev ([GitHub](https://github.com/angel3481))
