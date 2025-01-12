import pandas as pd
import aiohttp
import asyncio
from tqdm import tqdm
import random

def get_vin_batch(filename, batch_size=1000, batch_number=0):
    """Extract a batch of VINs from the input file"""
    try:
        # Skip rows based on batch number
        skip_rows = batch_number * batch_size if batch_number > 0 else None
        
        # Read the CSV file with no headers, and specify the VIN column by index (5)
        df = pd.read_csv(filename, 
                        header=None, 
                        skiprows=skip_rows, 
                        nrows=batch_size,
                        names=['year', 'make', 'model', 'body', 'transmission', 'vin', 'state', 
                              'condition', 'odometer', 'exterior', 'interior', 'price'])
        
        return df['vin'].tolist()
        
    except Exception as e:
        print(f"Error reading batch {batch_number}: {e}")
        return []

async def fetch_batch_data(vins):
    """Fetch data for a batch of VINs"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for vin in vins:
            # Add delay before each request
            await asyncio.sleep(random.uniform(0.02, 0.03))
            # Create task but don't await it yet
            task = asyncio.create_task(fetch_single_vin(vin, session))
            tasks.append(task)
            
        # Now gather all responses
        all_results = await asyncio.gather(*tasks)
            
        return all_results

async def fetch_single_vin(vin, session):
    """Fetch data for a single VIN"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                print(f"Error fetching data for VIN {vin}: HTTP {response.status}")
                print(f"Response text: {await response.text()}")
                return {'vin': vin, 'fuel_type': None, 'engine_volume': None}
            
            data = await response.json()
            
            # Extract values from the Results array
            if data.get("Results"):
                fuel_type = None
                engine_volume = None
                
                for item in data["Results"]:
                    variable = item.get("Variable", "")
                    value = item.get("Value")
                    
                    if variable == "Fuel Type - Primary":
                        fuel_type = value
                    elif variable == "Displacement (L)":
                        engine_volume = value
                
                return {
                    'vin': vin,
                    'fuel_type': fuel_type,
                    'engine_volume': engine_volume
                }
            return {'vin': vin, 'fuel_type': None, 'engine_volume': None}
            
    except Exception as e:
        print(f"Exception for VIN {vin}: {str(e)}")
        return {'vin': vin, 'fuel_type': None, 'engine_volume': None}

def extract_vehicle_data(vin, data):
    """Extract specific vehicle information"""
    # Debug print
    print(f"\nRaw data for VIN {vin}:")
    print(data)
    
    results_dict = {
        item["Variable"]: item["Value"]
        for item in data.get("Results", [])
        if item.get("Variable") in ["Fuel Type - Primary", "Displacement (L)"]
    }
    
    # Debug print
    print("Extracted results_dict:")
    print(results_dict)
    
    return {
        'vin': vin,
        'fuel_type': results_dict.get("Fuel Type - Primary"),
        'engine_volume': results_dict.get("Displacement (L)"),
    }

def save_batch_results(batch_data, batch_number):
    """Save batch results to CSV"""
    import os
    
    # Create batches directory if it doesn't exist
    os.makedirs('batches', exist_ok=True)
    
    filename = os.path.join('batches', f'batch_{batch_number}.csv')
    df = pd.DataFrame(batch_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(batch_data)} records to {filename}")

async def main():
    input_file = 'car_prices_4.csv'
    batch_size = 1000
    batch_number = 372
    
    while True:
        # Get batch of VINs
        vins = get_vin_batch(input_file, batch_size, batch_number)
        if not vins:
            break
            
        print(f"Processing batch {batch_number} with {len(vins)} VINs")
        
        # Fetch data for the batch
        batch_data = await fetch_batch_data(vins)
        
        # Save batch results
        save_batch_results(batch_data, batch_number)
        
        batch_number += 1
        break
if __name__ == "__main__":
    asyncio.run(main())
