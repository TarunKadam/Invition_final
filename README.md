## Overview

The project is a Forex anomaly detection and compliance system designed to monitor trading, login, financial, behavioral, and network activities in real-time. It combines statistical models, temporal analysis, and LLM-generated summaries to detect suspicious or high-risk patterns across user accounts. Alerts and detailed reports are generated for compliance teams to investigate, and the system supports advanced network-level and temporal anomaly detection. The architecture is scalable, integrates Kafka for real-time events, and can be extended with an interactive dashboard for visualization and auditing.


## Architecture Overview
```text
                ┌────────────────────┐
                │ Generate Synthetic │
                │   Trade Dataset    │
                │ (data_generator.py)│
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Feature Engineering│
                │  (processor.py)    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Train ML Models  │
                │ (train_models.py)  │
                └─────────┬──────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │  FastAPI Inference Layer   │
             │         (app.py)           │
             └─────────┬──────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 ┌────────────┐   ┌────────────────┐   ┌────────────────────┐
 │ Isolation  │   │LSTM Autoencoder│   │    Rule Engine     │
 │ Forest     │   │ (Temporal ML)  │   │ (Heuristic Checks) │
 └────────────┘   └────────────────┘   └────────────────────┘
        │                 │                 │
        └──────────┬──────┴──────┬──────────┘
                   ▼             ▼
            ┌────────────────────────┐
            │  Final Risk Decision   │
            └──────────┬─────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ LLM Risk Summary Generator   │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ API Response / Kafka Alerts  │
        └──────────────────────────────┘
```
## What I have done
1. data_generator.py:
- The code generates a synthetic forex events dataset with diverse users, including user_id, country, device_preference, risk_profile, and kyc_status. 
- It simulates a wide range of events (logins, logouts, trades, transactions, KYC, document uploads, orders, margin calls, stop-loss/profit triggers) with event-specific numeric and categorical features. 
- Anomalies include geo-anomalies, device anomalies, high-risk withdrawals, trade anomalies, and sequences of consecutive anomalous events, all of which can be amplified for LSTM detection. 
- Certain high-risk events are flagged as is_high_risk_fraud. The final output is a DataFrame with all features, timestamps, anomaly flags, and fraud indicators for model training.
- I have also inclluded noise in the dataset to replicate real-world datasets

2. processor.py:
- Reads the forex events CSV and ensures critical columns exist.
- It generates user-level features including rolling statistics (mean, std, z-score of amount), inter-event times, PnL volatility, IP/device anomaly proxies, clustered trades, and login velocity. 
- The pipeline captures temporal patterns, unusual trading behavior, and potential anomalies, then outputs a cleaned, feature-rich dataset (featured_events.csv) for model training.
- I have used Polars instead of Pandas for the following reasons
   - Polars is significantly faster than Pandas because it uses multi-core parallel processing, unlike Pandas which is single-threaded.
   -  It also supports lazy evaluation, optimizing the entire query before execution to reduce memory usage and computation time.
   -   Additionally, its Apache Arrow-based columnar storage and lack of index complexity make it more memory-efficient and less error-prone for large-scale data processing.

3. train_model.py:
- Implements a baseline Isolation Forest and an advanced LSTM Autoencoder for anomaly detection on forex event sequences.
- It preprocesses and encodes features, converts them into user-level sequences, and trains the LSTM to reconstruct normal behavior. 
- Both models generate anomaly scores, which are compared using precision, recall, and F1 metrics, helping evaluate their effectiveness on temporal anomalies.
- Scaled the LSTM Autoencoder anomaly score to a 0–1 range using a sigmoid function.

4. app.py:
- This FastAPI app implements ForexGuard compliance monitoring by detecting anomalies across multiple domains: login patterns, financial transactions, trading behavior, behavioral metrics, temporal events, account risk, and graph/network-level anomalies (e.g., multiple accounts on same IP, synchronized trades).
- It accepts trade activity via a TradeActivity JSON payload, computes anomaly scores using pre-trained models (Isolation Forest + LSTM), and performs rule-based checks for specific high-risk patterns. 
- Detected anomalies are summarized using an LLM and optionally sent to Kafka for alerting. 
- The app also tracks per-user and per-IP histories to identify patterns over time.
- It pushes alerts to a compliance_alerts, a topic in Redpanda cluster, helping in real-time messaging.

## Setup Instructions
1. Clone Repository
git clone <your-repo-url>
cd ForexGuard
2. Install Dependencies
pip install -r requirements.txt

3. Environment Variables
Set the following (optional but recommended):
export HF_TOKEN=your_huggingface_token
export KAFKA_BROKER=localhost:9092

4. Run data_generator.py
python data_generator.py
This creates forex_events.csv

5. Run processor.py: 
python processor.py
This creates featured_events.csv

6. Run train_models.py:
python train_models.py
advanced_model.pt is generated in \models folder 

7. For hosting, Render is used with Redpanda for storing alerts
8. Put these environment variables as follows in Render:
   HF_TOKEN=your_huggingface_token
  KAFKA_BOOTSTRAP_SERVERS=your_kafka_bootstrap_servers
  KAFKA_BROKER=your_kafka_broker
  KAFKA_USERNAME=your_kafka_username
  KAFKA_PASSWORD=your_kafka_password
  KAFKA_SASL_MECHANISM=SCRAM-SHA-256
  KAFKA_SECURITY_PROTOCOL=SASL_SSL
9. Deploy in Render.
10. You will encounter this message:  Available at your primary URL https://invition-final.onrender.com. Give your input on this URL.
11. On executing it, an alert is sent to Redpanda compliance_alerts topic if the input is an anomaly.




## Model Explanation

### Isolation Forest (Baseline Model)
  1. It is a classical machine learning model for anomaly detection.
  2. Isolation Forest works by randomly partitioning data points in feature space.
  3. Anomalies are easier to “isolate” because they are few and different from normal points.
  4. The fewer splits needed to isolate a point, the more likely it is an anomaly.
  5. X_raw is the flattened sequence of features for all users.
  6. contamination=0.01 tells the model that ~1% of the data is expected to be anomalous.
  7. SHAP Explainer provides interpretability by showing which features contribute most to each anomaly score
  8. Higher anomaly_scores indicate more anomalous points.

#### Strengths:
  1. Fast, non-parametric, works well with tabular data.
  2. Does not require labeled anomalies.

#### Limitations:
  1. Ignores temporal/sequential dependencies.
  2. Performance depends on feature scaling and representation.

#### Justification:
1. Unsupervised anomaly detection:
  In real-world event data (like forex or user sessions), labeled anomalies are rare. Isolation Forest does not require labeled anomalies to learn, making it ideal as a baseline.
2. Fast and interpretable:
  Works efficiently on large datasets.
  Using SHAP, it can explain which features contribute most to an anomaly, helping in auditing or compliance.
3. Widely accepted baseline:
  Isolation Forest is commonly used in fraud detection, cybersecurity, and sensor anomaly detection.
Provides a benchmark to compare advanced models against.
4. Robust to high-dimensional data:
  Can handle dozens of numeric/categorical features without overfitting easily.
5. It is good for large datasets unlike LOF(Local Outlier Factor), which is good for local anomalies.

### LSTM Autoencoder (Advanced Model)
  1. Deep learning model for sequence-based anomaly detection.
  2. LSTM (Long Short-Term Memory) networks capture temporal patterns in sequences.
  3. Autoencoder learns to reconstruct normal sequences.
  4. Anomalies produce high reconstruction error, because the model hasn’t seen similar sequences during training.
  5. Encoder: Compresses input sequence into a hidden representation.
  6. Decoder: Attempts to reconstruct the original sequence from hidden state.
  7. Mean Squared Error (MSE) between input and reconstruction is optimized.
  8. Higher MSE implies sequence is anomalous.
  9. Threshold is set at the 95th percentile of reconstruction errors to classify anomalies.

#### Strengths:

  1. Captures temporal dependencies in sequential data.
  2. Can handle both numeric and encoded categorical features.

#### Limitations:

  1. Requires more data and compute for training.
  2. Hyperparameter tuning (hidden_dim, sequence length, learning rate) affects performance.

#### Justification:
1. Justification for Choosing LSTM over Transformer and VAE
- Transformers require massive amounts of data to properly learn their self-attention weights. 
- Our dataset is relatively small (~50,000 rows) with simple features. 
- Using a Transformer in this scenario would likely overfit the data. 
- LSTMs, in contrast, are much more suitable for small-to-medium sequential datasets, capturing temporal patterns without over-parameterization.

- We used an LSTM Autoencoder because our data is sequential, and LSTMs capture temporal patterns that VAEs cannot.
- VAEs model probabilistic latent spaces but ignore sequence order, making them less effective here.
- LSTMs are simpler to train, more stable on moderate data, and provide real-time anomaly scores. They are also easier to debug and interpret compared to VAEs or Transformers.

2. Temporal dependency modeling:
- User-event or trading data is sequential in nature.
- LSTM networks capture patterns over time, which Isolation Forest cannot.

3. Unsupervised sequence reconstruction:
- Autoencoder learns the normal patterns of sequences.
- Anomalies result in high reconstruction errors, allowing detection of subtle temporal anomalies like unusual trading sequences or login     behaviors.

4. Flexibility for categorical + numeric features:
With proper encoding and scaling, LSTM Autoencoder can learn from heterogeneous feature types.

5. Scalability:
Can be extended to transformers or variational autoencoders (VAE) for more complex temporal or probabilistic anomaly modeling.

##Hosting
I chose Render because it allows very fast and simple deployment of backend APIs with minimal configuration. My project involves FastAPI endpoints and streaming components, which are not well supported by Hugging Face. While AWS provides full flexibility, it requires significant setup and infrastructure management. Since this was a prototype with time constraints, Render was the most efficient and practical choice.

## Assumptions, Improvements and Limitations
1. LLM Generated explanation of anaomaly could not be successfully incorporated due to time constraints
2. Creating a frontend dashboard
3. Experiment tracking using MLflow or Weights & Biases was planned; however, it was not implemented due to the evolving nature of the pipeline and prioritization of core system functionality. This can be incorporated in future iterations once the experimentation workflow is stabilized.
4. The synthetic dataset generated is symbolic of real-world data, hence its very unbalanced with respect to the number of anomalies among total entries. This causes Isolation Model to have similar performance metrics compared to that of LSTM Autoencoder model.


## Project Structure
```text
ForexGuard/
│
├── data/
│   ├── forex_events.csv
│   └── featured_events.csv
│
├── models/
│   ├── baseline.pkl
│   ├── advanced_model.pt
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   └── feature_names.pkl
│
├── src/
│   ├── app.py
│   ├── processor.py
│   ├── data_generator.py
│   └── train_models.py
│   
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```







