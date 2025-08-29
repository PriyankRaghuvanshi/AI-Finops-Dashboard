# config.py

# --- Application Settings ---
# The minimum size in MB for a bucket to be analyzed
MIN_BUCKET_SIZE_MB = 200

# The file path to save/load the trained AI model
MODEL_FILE_PATH = 'ai_model.pkl'

# --- AWS Settings ---
# The tag key used in Cost Explorer to identify costs for a specific S3 bucket.
# You MUST tag your buckets with this key for cost analysis to work.
COST_EXPLORER_TAG = 'CostCenter'

# --- AI Model Thresholds ---
# The confidence level (0.0 to 1.0) above which an anomaly is flagged
ANOMALY_CONFIDENCE_THRESHOLD = 0.75
