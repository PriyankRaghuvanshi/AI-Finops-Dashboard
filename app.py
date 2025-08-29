# app.py
import streamlit as st
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pickle
import json
import os
# Add this line at the top of app.py
from ai_model import AIPatternLearner
from botocore.exceptions import ClientError
from config import MIN_BUCKET_SIZE_MB, MODEL_FILE_PATH, COST_EXPLORER_TAG, ANOMALY_CONFIDENCE_THRESHOLD

# --- DATA COLLECTOR CLASS (Connects to real AWS data) ---
class AIDataCollector:
    """Collect REAL AWS data and prepare features for ML models"""
    
    def __init__(self, region):
        self.region = region
        self.clients = self._get_clients()
        self.s3_storage_price = self._get_s3_storage_price()

    def _get_clients(self):
        return {
            's3': boto3.client('s3', region_name=self.region),
            'cloudwatch': boto3.client('cloudwatch', region_name=self.region),
            'ce': boto3.client('ce', region_name='us-east-1'), # Cost Explorer is a global service
            'pricing': boto3.client('pricing', region_name='us-east-1') # Pricing is a global service
        }
    


   # Final debugging version for app.py

    # The definitive, robust pricing function for app.py

    # Final discovery version for app.py

  # The Final Version for app.py

    def _get_s3_storage_price(self):
        """Get S3 Standard storage price from the AWS Pricing API using a broad search."""
        
        REGION_NAME_MAP = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'ap-southeast-2': 'Asia Pacific (Sydney)',
            'eu-west-1': 'EU (Ireland)'
        }
        
        # This new map handles the special usagetype names
        USAGETYPE_REGION_MAP = {
            'ap-southeast-2': 'APS2'
        }
        
        location_name = REGION_NAME_MAP.get(self.region, self.region)
        # Use the map to get the correct region prefix, otherwise default to the standard name
        usage_region_prefix = USAGETYPE_REGION_MAP.get(self.region, self.region)
        target_usage_type = f'{usage_region_prefix}-TimedStorage-ByteHrs'

        try:
            paginator = self.clients['pricing'].get_paginator('get_products')
            for page in paginator.paginate(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location_name},
                    {'Type': 'TERM_MATCH', 'Field': 'storageClass', 'Value': 'General Purpose'}
                ]
            ):
                for price_item_str in page['PriceList']:
                    price_item = json.loads(price_item_str)
                    
                    if price_item.get('product', {}).get('attributes', {}).get('usagetype') == target_usage_type:
                        on_demand_terms = price_item.get('terms', {}).get('OnDemand', {})
                        if not on_demand_terms: continue

                        first_term = list(on_demand_terms.values())[0]
                        price_dimensions = first_term.get('priceDimensions', {})
                        if not price_dimensions: continue

                        first_dimension = list(price_dimensions.values())[0]
                        price_per_unit = first_dimension.get('pricePerUnit', {})
                        price_per_gb_str = price_per_unit.get('USD')

                        if price_per_gb_str:
                            price_per_gb = float(price_per_gb_str)
                            st.sidebar.metric("S3 Price/GB/Month", f"${price_per_gb:.4f}")
                            return price_per_gb
            
            st.sidebar.warning(f"Could not locate S3 Standard price for {self.region}. Using default.")
            return 0.023

        except (ClientError, Exception) as e:
            st.sidebar.error(f"An error occurred fetching pricing: {type(e).__name__}")
            st.sidebar.warning("Using default price.")
            return 0.023
        
    # ... all other data collection methods remain the same as the previous version ...
    def collect_ml_features(self):
        """Collect AWS data and convert to ML features, respecting the size threshold."""
        
        try:
            buckets_response = self.clients['s3'].list_buckets()
            ml_dataset = []
            total_buckets = len(buckets_response.get('Buckets', []))
            skipped_count = 0
            
            st.info(f"Scanning {total_buckets} buckets. This may take a moment...")
            progress_bar = st.progress(0)

            for i, bucket in enumerate(buckets_response.get('Buckets', [])):
                features = self._extract_bucket_features(bucket)
                if features:
                    ml_dataset.append(features)
                else:
                    skipped_count += 1
                progress_bar.progress((i + 1) / total_buckets)
            
            return ml_dataset, total_buckets, skipped_count
            
        except Exception as e:
            st.error(f"Error collecting ML features: {str(e)}")
            return [], 0, 0
    
    def _extract_bucket_features(self, bucket):
        """Extract ML features from a single bucket using real AWS data."""
        
        try:
            bucket_name = bucket['Name']
            min_size_bytes = MIN_BUCKET_SIZE_MB * 1024**2
            
            # Get bucket objects and total size
            objects, total_size = [], 0
            try:
                paginator = self.clients['s3'].get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket_name):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            age_days = (datetime.now(timezone.utc) - obj['LastModified']).days
                            objects.append({'size': obj['Size'], 'age_days': age_days})
                            total_size += obj['Size']
            except ClientError:
                return None 
            
            if total_size < min_size_bytes: return None
                
            access_freq = self._get_access_frequency(bucket_name)
            cost_trend, cost_volatility = self._get_cost_metrics(bucket_name)
            compliance_score = self._get_compliance_score(bucket_name)
            
            size_gb = total_size / (1024**3)
            avg_age_days = np.mean([obj['age_days'] for obj in objects]) if objects else 0
            bucket_age_days = (datetime.now(timezone.utc) - bucket['CreationDate']).days
            
            return {
                'bucket_name': bucket_name, 'size_gb': size_gb, 'avg_age_days': avg_age_days,
                'access_freq': access_freq, 'cost_trend': cost_trend, 'object_count': len(objects),
                'bucket_age_days': bucket_age_days, 'cost_volatility': cost_volatility,
                'region_factor': 1.0, 'compliance_score': compliance_score
            }
        except Exception:
            return None

    def _get_access_frequency(self, bucket_name):
        """Get access frequency from CloudWatch GetRequests metric."""
        try:
            response = self.clients['cloudwatch'].get_metric_statistics(
                Namespace='AWS/S3', MetricName='GetRequests',
                Dimensions=[{'Name': 'BucketName', 'Value': bucket_name}, {'Name': 'FilterId', 'Value': 'EntireBucket'}],
                StartTime=datetime.now() - timedelta(days=90), EndTime=datetime.now(),
                Period=86400 * 90, Statistics=['Sum']
            )
            return response['Datapoints'][0]['Sum'] / 90.0 if response['Datapoints'] else 0.0
        except ClientError:
            return 0.0 # Return a neutral value on error

    def _get_cost_metrics(self, bucket_name):
        """Get cost trend and volatility from Cost Explorer. REQUIRES BUCKET TAGS."""
        try:
            today = datetime.now()
            p1_start = (today - timedelta(days=60)).strftime('%Y-%m-%d')
            p1_end = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            p2_start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            p2_end = today.strftime('%Y-%m-%d')

            cost1_resp = self.clients['ce'].get_cost_and_usage(TimePeriod={'Start': p1_start, 'End': p1_end}, Granularity='MONTHLY', Metrics=['UnblendedCost'], Filter={'Tags': {'Key': 
COST_EXPLORER_TAG, 'Values': [bucket_name]}})
            cost1 = float(cost1_resp['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])

            cost2_resp = self.clients['ce'].get_cost_and_usage(TimePeriod={'Start': p2_start, 'End': p2_end}, Granularity='MONTHLY', Metrics=['UnblendedCost'], Filter={'Tags': {'Key': 
COST_EXPLORER_TAG, 'Values': [bucket_name]}})
            cost2 = float(cost2_resp['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
            
            cost_trend = ((cost2 - cost1) / cost1) if cost1 > 0 else 0.0

            daily_costs_resp = self.clients['ce'].get_cost_and_usage(TimePeriod={'Start': p2_start, 'End': p2_end}, Granularity='DAILY', Metrics=['UnblendedCost'], Filter={'Tags': {'Key': 
COST_EXPLORER_TAG, 'Values': [bucket_name]}})
            daily_amounts = [float(day['Total']['UnblendedCost']['Amount']) for day in daily_costs_resp['ResultsByTime']]
            cost_volatility = np.std(daily_amounts) / np.mean(daily_amounts) if np.mean(daily_amounts) > 0 else 0.0
            
            return cost_trend, cost_volatility
        except (ClientError, IndexError):
            return 0.0, 0.0 # Return neutral values if tags are missing or permissions fail

    def _get_compliance_score(self, bucket_name):
        """Calculate a compliance score based on S3 best practices."""
        score = 0.0
        try:
            if self.clients['s3'].get_public_access_block(Bucket=bucket_name)['PublicAccessBlockConfiguration']['BlockPublicAcls']: score += 0.4
        except ClientError: pass
        try:
            if self.clients['s3'].get_bucket_versioning(Bucket=bucket_name).get('Status') == 'Enabled': score += 0.3
        except ClientError: pass
        try:
            self.clients['s3'].get_bucket_encryption(Bucket=bucket_name)
            score += 0.3
        except ClientError: pass
        return score

# --- MAIN STREAMLIT APP ---
# --- MAIN STREAMLIT APP (Complete Version) ---
def main():
    st.set_page_config(page_title="AI FinOps Dashboard", layout="wide")
    st.title("🤖 AI FinOps Dashboard")
    st.markdown("**Powered by Real AWS Data & Dynamic Pricing**")
    
    # Load the pre-trained AI model
    if not os.path.exists(MODEL_FILE_PATH):
        st.error(f"Model file not found at '{MODEL_FILE_PATH}'. Please run 'python train_model.py' first.")
        return
    with open(MODEL_FILE_PATH, 'rb') as f:
        ai_learner = pickle.load(f)

    # Sidebar
    st.sidebar.header("AI Configuration")
    region = st.sidebar.selectbox("AWS Region", ['us-east-1', 'us-west-2', 'ap-southeast-2', 'eu-west-1'], index=2)
    
    if st.sidebar.button("🧠 Run AI Analysis", type="primary"):
        data_collector = AIDataCollector(region)
        
        with st.spinner("Extracting REAL ML features from your AWS environment..."):
            ml_features, total_buckets, skipped_count = data_collector.collect_ml_features()
        
        st.info(f"🔎 Scanned {total_buckets} buckets. Skipped {skipped_count} buckets (size < {MIN_BUCKET_SIZE_MB} MB or access issues).")

        if not ml_features:
            st.warning("⚠️ No buckets met the criteria for analysis.")
            return

        # Generate recommendations using the loaded model and real data
        all_recommendations = []
        for features in ml_features:
            predictions = ai_learner.predict_optimization(features)
            # Use dynamic S3 price in recommendation generation
            features['s3_storage_price_per_gb'] = data_collector.s3_storage_price
            recommendations = ai_learner.generate_ai_recommendations(predictions, features)
            all_recommendations.extend(recommendations)
        
        # --- THIS IS THE MISSING DISPLAY LOGIC ---
        if all_recommendations:
            st.success(f"🎯 AI Engine Generated {len(all_recommendations)} recommendations.")
            
            # Sort by priority
            all_recommendations.sort(key=lambda x: x.get('priority', 0), reverse=True)
            
            for i, rec in enumerate(all_recommendations, 1):
                priority_emoji = {3: "🔴", 2: "🟡", 1: "🟢"}.get(rec.get('priority', 1), "⚪")
                
                with st.expander(f"{priority_emoji} Recommendation #{i}: **{rec.get('action')}** for bucket `{rec.get('bucket_name')}`", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Priority", f"{priority_emoji} {rec.get('priority', 1)}")
                    with col2:
                        st.metric("AI Confidence", f"{rec.get('confidence', 0):.0%}")
                    with col3:
                        st.metric("Predicted Savings", f"${rec.get('predicted_monthly_savings', 0):.2f}/mo")
                    with col4:
                        st.metric("Savings %", f"{rec.get('savings_potential_percent', 0):.0f}%")
                    
                    st.info(f"**🧠 AI Reasoning:** {rec.get('ai_reasoning', 'N/A')}")
                    st.write(f"**⚙️ Implementation:** `{rec.get('implementation', 'N/A')}`")

        else:
            st.info("🎉 AI analysis complete - No optimization recommendations needed!")

if __name__ == "__main__":
    main()