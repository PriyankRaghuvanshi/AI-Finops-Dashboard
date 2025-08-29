# Updated train_model.py

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODEL_FILE_PATH
from ai_model import AIPatternLearner

def load_historical_data(filepath='historical_data_sample.csv'):
    """
    Loads historical, labeled data from a CSV file.
    In a real project, you would run a data engineering job to generate this file.
    """
    print(f"Loading historical training data from '{filepath}'...")
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Training file '{filepath}' not found.")
        print("Please create it with historical data or use the provided sample.")
        return None

def train_and_save_model(training_data, model_path):
    # (This function remains the same as before)
    print("Training AI models...")
    ai_learner = AIPatternLearner()
    
    feature_columns = ['size_gb', 'avg_age_days', 'access_freq', 'cost_trend', 'object_count', 'bucket_age_days', 'cost_volatility', 'region_factor', 'compliance_score']
    X = training_data[feature_columns].values
    
    ai_learner.scalers['features'] = StandardScaler().fit(X)
    X_scaled = ai_learner.scalers['features'].transform(X)
    
    ai_learner.label_encoders['optimization'] = LabelEncoder().fit(training_data['optimization'])
    y_optimization = ai_learner.label_encoders['optimization'].transform(training_data['optimization'])
    ai_learner.models['cost_optimizer'].fit(X_scaled, y_optimization)
    
    usage_patterns = ['high_usage' if f > 1.0 else 'low_usage' if f < 0.2 else 'moderate_usage' for f in training_data['access_freq']]
    ai_learner.label_encoders['usage'] = LabelEncoder().fit(usage_patterns)
    y_usage = ai_learner.label_encoders['usage'].transform(usage_patterns)
    ai_learner.models['usage_classifier'].fit(X_scaled, y_usage)
    
    ai_learner.models['anomaly_detector'].fit(X_scaled, (training_data['cost_trend'] > 0.5).astype(int))
    ai_learner.models['savings_estimator'].fit(X_scaled, training_data['savings_potential'])
    
    ai_learner.is_trained = True
    
    with open(model_path, 'wb') as f:
        pickle.dump(ai_learner, f)
    print(f"✅ Models trained and saved to '{model_path}'")

if __name__ == "__main__":
    training_df = load_historical_data()
    if training_df is not None:
        train_and_save_model(training_df, MODEL_FILE_PATH)