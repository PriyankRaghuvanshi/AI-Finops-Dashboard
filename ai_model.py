# ai_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import ANOMALY_CONFIDENCE_THRESHOLD

class AIPatternLearner:
    """ML models that learn from AWS patterns instead of hardcoded rules"""
    
    def __init__(self):
        self.models = {
            'cost_optimizer': RandomForestClassifier(n_estimators=100, random_state=42),
            'storage_predictor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'usage_classifier': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42),
            'anomaly_detector': RandomForestClassifier(n_estimators=50, random_state=42),
            'savings_estimator': MLPRegressor(hidden_layer_sizes=(32, 16), random_state=42)
        }
        self.scalers = {}
        self.label_encoders = {}
        self.is_trained = False

    def predict_optimization(self, aws_features):
        """Use trained models to predict optimization strategies"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        feature_columns = ['size_gb', 'avg_age_days', 'access_freq', 'cost_trend', 'object_count', 'bucket_age_days', 'cost_volatility', 'region_factor', 'compliance_score']
        feature_array = np.array([[aws_features[col] for col in feature_columns]])
        
        feature_scaled = self.scalers['features'].transform(feature_array)
        
        # Get predictions
        opt_pred_encoded = self.models['cost_optimizer'].predict(feature_scaled)[0]
        usage_pred_encoded = self.models['usage_classifier'].predict(feature_scaled)[0]

        return {
            'optimization_type': self.label_encoders['optimization'].inverse_transform([opt_pred_encoded])[0],
            'optimization_confidence': max(self.models['cost_optimizer'].predict_proba(feature_scaled)[0]),
            'usage_pattern': self.label_encoders['usage'].inverse_transform([usage_pred_encoded])[0],
            'is_anomaly': bool(self.models['anomaly_detector'].predict(feature_scaled)[0]),
            'anomaly_confidence': self.models['anomaly_detector'].predict_proba(feature_scaled)[0][1],
            'savings_potential': self.models['savings_estimator'].predict(feature_scaled)[0]
        }

    def generate_ai_recommendations(self, predictions, aws_features):
        """Generate intelligent recommendations by synthesizing outputs from multiple ML models."""
        optimization_type = predictions['optimization_type']
        
        recommendation_actions = {
            'glacier_transition': {'priority': 3, 'action': 'Transition to Glacier Deep Archive', 'implementation': 'Set lifecycle policy: Standard → IA (30d) → Glacier (90d)'},
            'intelligent_tiering': {'priority': 2, 'action': 'Enable S3 Intelligent Tiering', 'implementation': 'Enable Intelligent-Tiering with monitoring'},
            'delete_unused': {'priority': 3, 'action': 'Delete unused resources', 'implementation': 'Review and delete after data retention compliance check'},
            'monitor': {'priority': 1, 'action': 'Continue monitoring', 'implementation': 'Set up CloudWatch alerts for cost anomalies'},
            'investigate_anomaly': {'priority': 3, 'action': 'Investigate cost anomaly', 'implementation': 'Deep dive into usage patterns'}
        }
        
        if optimization_type not in recommendation_actions:
            return []

        rec_details = recommendation_actions[optimization_type].copy()
        
        # Use the real S3 price passed in aws_features
        s3_price = aws_features.get('s3_storage_price_per_gb', 0.023)
        predicted_savings = aws_features['size_gb'] * s3_price * predictions['savings_potential']

        recommendation = {
            'confidence': predictions['optimization_confidence'],
            'savings_potential_percent': predictions['savings_potential'] * 100,
            'predicted_monthly_savings': predicted_savings,
            'ai_reasoning': f"Primary model classified as '{optimization_type}' with {predictions['optimization_confidence']:.0%} confidence.",
            'bucket_name': aws_features.get('bucket_name', 'Unknown'),
            **rec_details
        }
        
        # AI Synthesis Layer
        if predictions['is_anomaly'] and predictions['anomaly_confidence'] > ANOMALY_CONFIDENCE_THRESHOLD:
            recommendation['priority'] = 3
            recommendation['ai_reasoning'] += f" | Anomaly detected with {predictions['anomaly_confidence']:.0%} confidence, increasing urgency."
        
        return [recommendation]