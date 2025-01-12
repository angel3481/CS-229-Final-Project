import pandas as pd
import numpy as np
import os

def preprocess_data_1(file_path):
    # Read the CSV file and ensure consistent number of columns
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Make column names consistent and rename specific columns
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns={
        'color': 'exterior_color',
        'interior': 'interior_color',
        'sellingprice': 'price'
    })

    # Drop unwanted columns
    columns_to_drop = ['trim', 'mmr', 'saledate', 'seller']
    df = df.drop(columns=columns_to_drop)

    # Process string columns
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].apply(lambda x: x.str.lower().str.strip())
    df[string_columns] = df[string_columns].apply(lambda x: x.str.replace(r'\s+', '_', regex=True))

    # Handle missing values - simplified approach
    df = df.replace(['N/A', '', 'none'], np.nan)

    # Drop rows with missing values - simplified
    df = df.dropna()

    # Save the cleaned dataset
    df.to_csv(file_path, index=False)
    print(f"Data preprocessing completed. Cleaned file saved as '{file_path}'")

def preprocess_data_2(file_path):
    df = pd.read_csv(file_path)
    
    # Calculate counts for filtering
    make_counts = df['make'].value_counts()
    model_counts = df['model'].value_counts()
    body_counts = df['body'].value_counts()
    
    # Filter cars based on all criteria
    filtered_df = df[
        (df['year'] > 2000) &
        (df['odometer'] >= 1000) & (df['odometer'] <= 500000) &
        (df['price'] >= 1000) & (df['price'] <= 100000) &
        (df['make'].isin(make_counts[make_counts >= 1000].index)) &
        (df['model'].isin(model_counts[model_counts >= 100].index)) &
        (df['body'].isin(body_counts[body_counts >= 1000].index))
    ]
    
    # Save the filtered dataset
    filtered_df.to_csv('car_prices_2.csv', index=False)
    print(f"Filtered data saved to 'car_prices_2.csv'")
    print(f"Original number of records: {len(df)}")
    print(f"Filtered number of records: {len(filtered_df)}")

def preprocess_data_3(file_path):
    df = pd.read_csv(file_path)
    
    # Print initial count
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    
    # Remove exact duplicates
    df_no_duplicates = df.drop_duplicates()
    
    # Print results
    final_count = len(df_no_duplicates)
    duplicates_removed = initial_count - final_count
    print(f"Number of duplicate rows removed: {duplicates_removed}")
    print(f"Final number of rows: {final_count}")
    
    # Save the deduplicated dataset
    df_no_duplicates.to_csv('car_prices_3.csv', index=False)
    print(f"Deduplicated data saved to 'car_prices_3.csv'")

def preprocess_data_4(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Get total number of cars
    total_cars = len(df)
    print(f"\nTotal number of cars in dataset: {total_cars}")
    
    # Filter cars based on criteria
    filtered_cars = df[
        (df['odometer'] < 220000) &
        (df['price'] < 50000)
    ]
    filtered_count = len(filtered_cars)
    
    print(f"Cars with odometer < 220,000 and price < $50,000: {filtered_count}")
    print(f"Percentage of total: {(filtered_count/total_cars*100):.1f}%")
    
    # Save the filtered dataset
    filtered_cars.to_csv('car_prices_4.csv', index=False)
    print(f"Filtered data saved to 'car_prices_4.csv'")

def preprocess_data_5(file_path):
    # Read the main dataset
    df = pd.read_csv(file_path)
    print(f"Number of examples in {file_path}: {len(df)}")
    
    # Initialize empty DataFrame for batch data
    batch_data = pd.DataFrame()
    
    # Read and combine all batch files
    for i in range(436):
        batch_file = f'batches/batch_{i}.csv'
        try:
            batch_df = pd.read_csv(batch_file)
            batch_data = pd.concat([batch_data, batch_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
        if i % 30 == 0:
            print(f"Processed {i} batches")
    
    print(f"Number of records in batch data: {len(batch_data)}")
    
    # Merge main dataset with batch data
    merged_df = df.merge(batch_data, on='vin', how='left')
    
    # Validate the merge
    missing_matches = merged_df['fuel_type'].isna().sum()
    print(f"Records without matching batch data: {missing_matches}")
    print(f"Final number of records: {len(merged_df)}")
    
    # Save the combined dataset
    merged_df.to_csv('car_prices_5.csv', index=False)
    print(f"Combined data saved to 'car_prices_5.csv'")

def preprocess_data_6(file_path):
    df = pd.read_csv(file_path)
    
    # Standardize fuel_type strings
    df['fuel_type'] = df['fuel_type'].str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
    
    # Move price column to the end
    cols = [col for col in df.columns if col != 'price'] + ['price']
    df = df[cols]
    
    # Save the cleaned dataset
    df.to_csv('car_prices_6.csv', index=False)
    print(f"Cleaned data saved to 'car_prices_6.csv'")

def preprocess_data_7(file_path):
    df = pd.read_csv(file_path)
    
    # Print initial count
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    
    # Remove null values
    df = df.dropna()
    print(f"Rows after removing null values: {len(df)}")
    
    # Filter engine volume
    df = df[(df['engine_volume'] >= 1) & (df['engine_volume'] <= 7)]
    print(f"Rows after filtering engine volume: {len(df)}")
    
    # Remove rare fuel types
    fuel_counts = df['fuel_type'].value_counts()
    common_fuel_types = fuel_counts[fuel_counts >= 120].index
    df = df[df['fuel_type'].isin(common_fuel_types)]
    print(f"Rows after filtering rare fuel types: {len(df)}")

    # Save the cleaned dataset
    df.to_csv('car_prices_7.csv', index=False)
    print(f"Cleaned data saved to 'car_prices_7.csv'")

def preprocess_data_8(file_path):
    # Read the current dataset
    df_current = pd.read_csv(file_path)
    
    # Read the original dataset with error handling
    df_original = pd.read_csv('car_prices_0.csv', on_bad_lines='skip')
    
    # Standardize column names in original dataset
    df_original.columns = df_original.columns.str.lower().str.strip()
    df_original = df_original.rename(columns={'sellingprice': 'price'})
    
    # Create a composite key for matching
    df_original['match_key'] = df_original['vin'] + '_' + \
                              df_original['odometer'].astype(str) + '_' + \
                              df_original['price'].astype(str)
    
    # Create a mapping dictionary for quick lookup
    mmr_dict = dict(zip(df_original['match_key'], df_original['mmr']))
    
    # Create the same composite key in current dataset
    df_current['match_key'] = df_current['vin'] + '_' + \
                             df_current['odometer'].astype(str) + '_' + \
                             df_current['price'].astype(str)
    
    # Print initial counts
    print(f"Current dataset rows: {len(df_current)}")
    print(f"Original dataset rows: {len(df_original)}")
    
    # Add MMR column and fill with matched values
    df_current['mmr'] = df_current['match_key'].map(mmr_dict)
    
    # Remove the temporary match_key column
    df_current = df_current.drop('match_key', axis=1)
    
    # Print matching statistics
    matched_count = df_current['mmr'].notna().sum()
    print(f"Successfully matched MMR values: {matched_count}")
    print(f"Matching rate: {(matched_count/len(df_current))*100:.2f}%")

    # Save the dataset with MMR values
    df_current.to_csv('car_prices_8.csv', index=False)
    print(f"Enhanced data saved to 'car_prices_8.csv'")

def preprocess_data_9(file_path):
    df = pd.read_csv(file_path) 
    
    # Move vin column to the front
    cols = ['vin'] + [col for col in df.columns if col != 'vin']
    df = df[cols]
    
    # Save the cleaned dataset
    df.to_csv('car_prices_9.csv', index=False)
    print(f"Cleaned data saved to 'car_prices_9.csv'")


def preprocess_data_10(file_path):
    df = pd.read_csv(file_path)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train, dev, test
    test_df = df[:15000]
    dev_df = df[15000:30000]
    train_df = df[30000:]
    
    # Identify categorical and numerical columns
    categorical_cols = ['make', 'model', 'body', 'transmission', 'state', 
                       'exterior_color', 'interior_color', 'fuel_type']
    regression_cols = ['year', 'condition', 'odometer', 'engine_volume']
    
    # Create label encoders for categorical features
    label_encoders = {}
    for col in categorical_cols:
        # Store the integer encodings
        encoded_vals = pd.factorize(df[col])[0]
        label_encoders[col] = encoded_vals
        
        # Use .loc for proper indexing
        train_df.loc[:, col] = label_encoders[col][30000:]
        dev_df.loc[:, col] = label_encoders[col][15000:30000]
        test_df.loc[:, col] = label_encoders[col][:15000]
    
    # Before any scaling operations, convert numeric columns to float32
    numeric_cols = regression_cols + ['price', 'mmr']
    
    # Convert all numeric columns to float32 in the dataframes
    train_df = train_df.astype({col: 'float32' for col in numeric_cols})
    dev_df = dev_df.astype({col: 'float32' for col in numeric_cols})
    test_df = test_df.astype({col: 'float32' for col in numeric_cols})
    
    # Min-max scaling for regression features
    scalers = {}
    for col in regression_cols + ['price', 'mmr']:  # Include mmr here
        if col != 'mmr':  # Only compute scalers for non-mmr columns
            min_val = df[col].min()
            max_val = df[col].max()
            scalers[col] = (min_val, max_val)
            
            # Apply scaling: (x - min) / (max - min)
            train_df.loc[:, col] = (train_df[col] - min_val) / (max_val - min_val)
            dev_df.loc[:, col] = (dev_df[col] - min_val) / (max_val - min_val)
            test_df.loc[:, col] = (test_df[col] - min_val) / (max_val - min_val)
    
    # Scale MMR using price's min-max values
    price_min, price_max = scalers['price']
    train_df.loc[:, 'mmr'] = (train_df['mmr'] - price_min) / (price_max - price_min)
    dev_df.loc[:, 'mmr'] = (dev_df['mmr'] - price_min) / (price_max - price_min)
    test_df.loc[:, 'mmr'] = (test_df['mmr'] - price_min) / (price_max - price_min)
    
    # Reorder columns: vin, categorical, regression
    column_order = ['vin'] + categorical_cols + regression_cols
    
    # Create datasets directory if it doesn't exist
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    for dataset, suffix in [(train_df, 'train'), (dev_df, 'dev'), (test_df, 'test')]:
        # Features (x)
        features_df = dataset[column_order]
        features_df.to_csv(os.path.join(dataset_dir, f'x_{suffix}.csv'), index=False)
        
        # Price (y)
        dataset[['price']].to_csv(os.path.join(dataset_dir, f'y_{suffix}.csv'), index=False)
        
        # MMR (benchmark)
        dataset[['mmr']].to_csv(os.path.join(dataset_dir, f'benchmark_{suffix}.csv'), index=False)
    
    # Save scalers and encoders in the datasets directory
    np.save(os.path.join(dataset_dir, 'scalers.npy'), scalers)
    np.save(os.path.join(dataset_dir, 'label_encoders.npy'), label_encoders)
    
    print("Data preprocessing completed. Files saved in '../datasets/' directory.")

if __name__ == "__main__":
    preprocess_data_1('car_prices_0.csv')
    preprocess_data_2('car_prices_1.csv')
    preprocess_data_3('car_prices_2.csv')
    preprocess_data_4('car_prices_3.csv')
    preprocess_data_5('car_prices_4.csv')
    preprocess_data_6('car_prices_5.csv')
    preprocess_data_7('car_prices_6.csv')
    preprocess_data_8('car_prices_7.csv')
    preprocess_data_9('car_prices_8.csv')
    preprocess_data_10('car_prices_9.csv')
