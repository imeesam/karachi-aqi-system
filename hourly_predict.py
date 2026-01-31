import requests
import pandas as pd
import joblib
import json
import logging
from datetime import datetime, timedelta
from huggingface_hub import login, HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REPO_ID = "imeesam/karachi-aqi-predictor"
DATA_REPO_ID = "imeesam/karachi-aqi-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Create .env file with HF_TOKEN=your_token")

LATITUDE = 24.8607
LONGITUDE = 67.0011

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def login_to_hf():
    """Login to Hugging Face"""
    login(token=HF_TOKEN)
    return HfApi()

def get_current_aqi():
    """Get current AQI and PM2.5 from Open-Meteo API"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "pm2_5",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pm25 = float(data['current']['pm2_5'])
        timestamp = data['current']['time']
        
        # Calculate AQI from PM2.5 (simplified formula)
        aqi = round((pm25 / 35.4) * 100)
        aqi = max(0, min(500, aqi))
        
        logging.info(f"Fetched current AQI: {aqi}, PM2.5: {pm25}")
        return aqi, timestamp, pm25
    except Exception as e:
        logging.error(f"Error fetching current AQI: {e}")
        # Fallback values
        return 100, datetime.now().isoformat(), 35.4

def get_yesterday_aqi():
    """Get yesterday's AQI from dataset"""
    try:
        dataset = load_dataset(DATA_REPO_ID)
        df = dataset['train'].to_pandas()
        
        if len(df) >= 24:
            # Get AQI from 24 hours ago
            return int(df['aqi'].iloc[-24])
        elif len(df) > 0:
            # Get latest available
            return int(df['aqi'].iloc[-1])
        else:
            return 100  # Default fallback
    except Exception as e:
        logging.warning(f"Error getting yesterday's AQI: {e}")
        return 100

def create_features():
    """Create features for prediction"""
    current_aqi, current_time, pm25 = get_current_aqi()
    dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
    
    yesterday_aqi = get_yesterday_aqi()
    
    features = {
        'timestamp': dt.isoformat(),
        'aqi': int(current_aqi),
        'pm2_5': float(pm25),
        'hour': int(dt.hour),
        'day_of_week': int(dt.weekday()),  # 0=Monday, 6=Sunday
        'month': int(dt.month),
        'year': int(dt.year),
        'aqi_yesterday': int(yesterday_aqi),
        'aqi_change_24h': int(current_aqi - yesterday_aqi)
    }
    
    logging.info(f"Created features: {features}")
    return features

def load_model_and_scaler(day_num):
    """Load the LATEST model and scaler from Hugging Face"""
    try:
        # First, get list of available files
        api_url = f"https://huggingface.co/api/models/imeesam/karachi-aqi-predictor/tree/main/models"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            logging.error(f"Failed to list files: {response.status_code}")
            return None, None, None
        
        files = response.json()
        
        # Find the latest model file for this day
        model_pattern = f"best_model_day{day_num}_"
        scaler_pattern = f"scaler_day{day_num}_"
        columns_pattern = f"feature_columns_day{day_num}_"
        
        model_files = [f for f in files if f.get('path', '').startswith(f'models/{model_pattern}')]
        scaler_files = [f for f in files if f.get('path', '').startswith(f'models/{scaler_pattern}')]
        columns_files = [f for f in files if f.get('path', '').startswith(f'models/{columns_pattern}')]
        
        if not model_files or not scaler_files or not columns_files:
            logging.error(f"Could not find files for day {day_num}")
            return None, None, None
        
        # Sort by timestamp (newest first)
        model_files.sort(key=lambda x: x['path'], reverse=True)
        scaler_files.sort(key=lambda x: x['path'], reverse=True)
        columns_files.sort(key=lambda x: x['path'], reverse=True)
        
        # Get latest files
        latest_model = model_files[0]['path']
        latest_scaler = scaler_files[0]['path']
        latest_columns = columns_files[0]['path']
        
        logging.info(f"Latest model for day {day_num}: {latest_model}")
        
        # Download files
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=latest_model,
            token=HF_TOKEN,
            repo_type="model"
        )
        
        scaler_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=latest_scaler,
            token=HF_TOKEN,
            repo_type="model"
        )
        
        columns_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=latest_columns,
            token=HF_TOKEN,
            repo_type="model"
        )
        
        # Load them
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(columns_path, 'r') as f:
            feature_columns = json.load(f)
        
        logging.info(f"Loaded model, scaler, and feature columns for day {day_num}")
        return model, scaler, feature_columns
        
    except Exception as e:
        logging.error(f"Error loading model for day {day_num}: {e}")
        return None, None, None

def prepare_prediction_features(features_dict, scaler, expected_columns):
    """Prepare features for prediction with proper encoding"""
    # Create DataFrame with the raw features
    raw_features = pd.DataFrame([{
        'hour': features_dict['hour'],
        'day_of_week': str(features_dict['day_of_week']),  # Convert to string for one-hot
        'month': str(features_dict['month']),              # Convert to string for one-hot
        'aqi': features_dict['aqi'],
        'aqi_yesterday': features_dict['aqi_yesterday'],
        'aqi_change_24h': features_dict['aqi_change_24h'],
        'pm2_5': features_dict['pm2_5']
    }])
    
    # One-hot encode categorical features
    raw_features = pd.get_dummies(raw_features, columns=['day_of_week', 'month'], drop_first=True)
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in raw_features.columns:
            raw_features[col] = 0
    
    # Reorder columns to match training
    raw_features = raw_features[expected_columns]
    
    # Scale features
    features_scaled = scaler.transform(raw_features)
    
    return features_scaled

def update_dataset_with_new_data(features):
    """Update the dataset with new hourly data"""
    try:
        # Load existing dataset
        dataset = load_dataset(DATA_REPO_ID)
        df = dataset['train'].to_pandas()
        
        # Convert timestamp to consistent format
        dt = datetime.fromisoformat(features['timestamp'].replace('Z', '+00:00'))
        
        # Create new row - ensure timestamp is int
        new_row = {
            'id': int(len(df)),
            'timestamp': int(dt.timestamp()),  # Ensure int
            'aqi': int(features['aqi']),
            'pm2_5': float(features['pm2_5']),
            'hour': int(features['hour']),
            'day_of_week': int(features['day_of_week']),
            'month': int(features['month']),
            'year': int(features['year']),
            'aqi_yesterday': int(features['aqi_yesterday']),
            'aqi_change_24h': int(features['aqi_change_24h']),
            'target_day1': None,
            'target_day2': None,
            'target_day3': None
        }
        
        # Add new row
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Update target values for old records
        updated_count = 0
        current_ts = int(dt.timestamp())
        
        for idx, row in df.iterrows():
            row_ts = row['timestamp']
            
            # Ensure row_ts is int for comparison
            if pd.isna(row_ts):
                continue
                
            row_ts = int(float(row_ts))  # Convert to int
            
            # Calculate hours difference
            hours_diff = (current_ts - row_ts) / 3600
            
            # Update target_day1 for records ~24 hours old
            if 23 <= hours_diff <= 25 and pd.isna(row.get('target_day1')):
                df.at[idx, 'target_day1'] = float(features['aqi'])
                updated_count += 1
            
            # Update target_day2 for records ~48 hours old
            elif 47 <= hours_diff <= 49 and pd.isna(row.get('target_day2')):
                df.at[idx, 'target_day2'] = float(features['aqi'])
                updated_count += 1
            
            # Update target_day3 for records ~72 hours old
            elif 71 <= hours_diff <= 73 and pd.isna(row.get('target_day3')):
                df.at[idx, 'target_day3'] = float(features['aqi'])
                updated_count += 1
        
        # Save back to Hugging Face
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(DATA_REPO_ID, token=HF_TOKEN)
        
        logging.info(f"Dataset updated with new data. Filled {updated_count} target values.")
        return updated_count
        
    except Exception as e:
        logging.error(f"Error updating dataset: {e}")
        return 0

def make_predictions():
    """Make predictions for next 3 days"""
    api = login_to_hf()
    features = create_features()
    
    predictions = {}
    
    for day_num in [1, 2, 3]:
        model, scaler, feature_columns = load_model_and_scaler(day_num)
        
        if model and scaler and feature_columns:
            try:
                # Prepare features for this model
                features_scaled = prepare_prediction_features(features, scaler, feature_columns)
                
                # Make prediction
                pred = model.predict(features_scaled)[0]
                predictions[f'day{day_num}'] = float(pred)
                logging.info(f"Day {day_num} prediction: {pred:.1f}")
            except Exception as e:
                logging.error(f"Error making prediction for day {day_num}: {e}")
                predictions[f'day{day_num}'] = float(features['aqi'])  # Fallback
        else:
            logging.warning(f"Using fallback prediction for day {day_num}")
            predictions[f'day{day_num}'] = float(features['aqi'])
    
    # Update dataset with new data
    targets_updated = update_dataset_with_new_data(features)
    
    # Create prediction data
    prediction_data = {
        'prediction_timestamp': datetime.now().isoformat() + 'Z',
        'current_timestamp': features['timestamp'],
        'current_aqi': features['aqi'],
        'current_pm25': features['pm2_5'],
        'predictions': predictions,
        'features': features,
        'targets_updated': targets_updated,
        'model_version': 'v2.0'
    }
    
    # Save predictions to Hugging Face
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save detailed prediction
    detailed_filename = f'pred_{timestamp}.json'
    api.upload_file(
        path_or_fileobj=json.dumps(prediction_data, indent=2).encode(),
        path_in_repo=f"predictions/{detailed_filename}",
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    # Also save as latest.json for easy access
    api.upload_file(
        path_or_fileobj=json.dumps(prediction_data, indent=2).encode(),
        path_in_repo="predictions/latest.json",
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    logging.info(f"Predictions saved: {predictions}")
    return prediction_data

if __name__ == "__main__":
    try:
        prediction_data = make_predictions()
        
        print("\n" + "="*50)
        print("HOURLY PREDICTION COMPLETE")
        print("="*50)
        print(f"Timestamp: {prediction_data['current_timestamp']}")
        print(f"Current AQI: {prediction_data['current_aqi']}")
        print(f"Current PM2.5: {prediction_data['current_pm25']:.1f}")
        print("\nPredictions:")
        for day, value in prediction_data['predictions'].items():
            print(f"  {day}: {value:.1f}")
        print(f"\nTarget values updated: {prediction_data['targets_updated']}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Hourly prediction failed: {e}")
        raise