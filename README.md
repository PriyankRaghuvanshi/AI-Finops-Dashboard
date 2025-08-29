# AI FinOps Dashboard

An intelligent dashboard that uses machine learning to provide dynamic cost-saving recommendations for AWS S3.

## The Problem
Cloud costs can quickly spiral out of control. Traditional analysis is often manual, time-consuming, and based on static rules that don't adapt to changing usage patterns.

## Solution
This tool connects directly to your AWS account to analyze real-time metrics for usage, cost, and compliance. It feeds this data into a trained AI model to generate actionable recommendations, 
helping you proactively manage your S3 spending.

## Architecture
*(Here you would embed a screenshot of the architecture diagram)*

## Features
- **AI-Driven Recommendations**: No hardcoded rules. The model learns from data.
- **Live AWS Integration**: Uses real-time data from CloudWatch, Cost Explorer, and the AWS Pricing API.
- **Dynamic Pricing**: Savings calculations are based on the precise S3 cost for your selected region.
- **Configurable**: Easily change settings like the analysis threshold and cost tags in `config.py`.

## Setup & Usage

**Prerequisites:**
- Python 3.8+
- An AWS account with programmatic access (credentials configured)

**Installation:**
1.  Clone the repository: `git clone <your-repo-url>`
2.  Navigate to the directory: `cd finops-ai-tool`
3.  Install dependencies: `pip install -r requirements.txt`

**Running the Application:**
1.  **Train the Model**: Populate `historical_data_sample.csv` with your real training data and run the training script:
    ```bash
    python train_model.py
    ```
2.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```
