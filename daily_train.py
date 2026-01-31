import pandas as pd
import joblib
import json
import logging
import numpy as np
from huggingface_hub import login, HfApi, hf_hub_download
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REPO_ID = "imeesam/karachi-aqi-predictor"
DATA_REPO_ID = "imeesam/karachi-aqi-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Create a .env file with HF_TOKEN=your_token")

FEATURE_COLUMNS = ['hour', 'day_of_week', 'month', 'aqi', 'aqi_yesterday', 'aqi_change_24h', 'pm2_5']
TARGET_DAYS = [1, 2, 3]
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def login_to_hf():
    """Login to Hugging Face"""
    login(token=HF_TOKEN)
    return HfApi()

def load_training_data():
    """Load dataset from Hugging Face"""
    try:
        dataset = load_dataset(DATA_REPO_ID)
        df = dataset['train'].to_pandas()
        
        # Ensure required columns exist
        required_columns = FEATURE_COLUMNS + ['target_day1', 'target_day2', 'target_day3']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"Missing columns in dataset: {missing_columns}")
            # Try to create missing columns
            if 'target_day1' not in df.columns and len(df) >= 24:
                df['target_day1'] = df['aqi'].shift(-24)
            if 'target_day2' not in df.columns and len(df) >= 48:
                df['target_day2'] = df['aqi'].shift(-48)
            if 'target_day3' not in df.columns and len(df) >= 72:
                df['target_day3'] = df['aqi'].shift(-72)
        
        df = df.dropna(subset=FEATURE_COLUMNS + ['target_day1', 'target_day2', 'target_day3'])
        logging.info(f"Loaded {len(df)} training samples")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def prepare_features(df):
    """Prepare features for training"""
    X = df[FEATURE_COLUMNS].copy()
    
    # Convert categorical features to string type for one-hot encoding
    X['day_of_week'] = X['day_of_week'].astype(str)
    X['month'] = X['month'].astype(str)
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['day_of_week', 'month'], drop_first=True)
    
    # Get the column order for future predictions
    feature_columns = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)
    
    return X_scaled, scaler, feature_columns

def train_models():
    """Train models for each day ahead"""
    api = login_to_hf()
    df = load_training_data()
    
    results = {}
    
    for day_num in TARGET_DAYS:
        target_col = f'target_day{day_num}'
        
        # Prepare features
        X_scaled, scaler, feature_columns = prepare_features(df)
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logging.info(f"Training Day {day_num} models on {len(X_train)} samples...")
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }
        
        best_model = None
        best_name = ""
        best_mae = float('inf')
        best_r2 = 0
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"  {name}: MAE={mae:.2f}, R2={r2:.3f}")
                
                if mae < best_mae:
                    best_mae = mae
                    best_r2 = r2
                    best_model = model
                    best_name = name
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
        
        if best_model is None:
            logging.error(f"No model trained for Day {day_num}")
            continue
        
        logging.info(f"Day {day_num} best model: {best_name} (MAE={best_mae:.2f})")
        
        # Save model and scaler locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        model_filename = f'best_model_day{day_num}.pkl'
        scaler_filename = f'scaler_day{day_num}.pkl'
        columns_filename = f'feature_columns_day{day_num}.json'
        
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        
        with open(columns_filename, 'w') as f:
            json.dump(feature_columns, f)
        
        # Upload to Hugging Face
        api.upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=f"models/best_model_day{day_num}.pkl",  # Already correct!
                repo_id=REPO_ID,
                repo_type="model"
        )
        
        api.upload_file(
            path_or_fileobj=scaler_filename,
            path_in_repo=f"models/scaler_day{day_num}.pkl",
            repo_id=REPO_ID,
            repo_type="model"
        )
        
        api.upload_file(
            path_or_fileobj=columns_filename,
            path_in_repo=f"models/feature_columns_day{day_num}.json",
            repo_id=REPO_ID,
            repo_type="model"
        )
        
        # Save model info
        model_info = {
            'model_name': best_name,
            'mae': float(best_mae),
            'r2': float(best_r2),
            'features': FEATURE_COLUMNS,
            'target': target_col,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(df),
            'feature_columns_count': len(feature_columns)
        }
        
        info_filename = f'model_info_day{day_num}_{timestamp}.json'
        with open(info_filename, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        api.upload_file(
            path_or_fileobj=info_filename,
            path_in_repo=f"models/model_info_day{day_num}.json",
            repo_id=REPO_ID,
            repo_type="model"
        )
        
        results[f'day{day_num}'] = {
            'model': best_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'mae': best_mae,
            'r2': best_r2
        }
        
        # Cleanup local files
        for f in [model_filename, scaler_filename, columns_filename, info_filename]:
            if os.path.exists(f):
                os.remove(f)
    
    logging.info("All models trained and uploaded successfully!")
    return results

if __name__ == "__main__":
    try:
        results = train_models()
        logging.info("Training pipeline completed successfully!")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise