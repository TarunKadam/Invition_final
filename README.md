## Overview

The project is a Forex anomaly detection and compliance system designed to monitor trading, login, financial, behavioral, and network activities in real-time. It combines statistical models, temporal analysis, and LLM-generated summaries to detect suspicious or high-risk patterns across user accounts. Alerts and detailed reports are generated for compliance teams to investigate, and the system supports advanced network-level and temporal anomaly detection. The architecture is scalable, integrates Kafka for real-time events, and can be extended with an interactive dashboard for visualization and auditing.


## Architecture Overview

                ┌────────────────────┐
                │Generate Synthetic  |
                |     dataset of     |
                |  raw trade events  |
                |(data_generator.py) |
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Feature Engineering│
                │ (processor.py)     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Train ML Models  │
                │  (train_models.py) │
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
 │ Isolation  │   │LSTM Autoencoder│   │     Rule Engine    │
 │ Forest     │   │ (Temporal ML)  │   │ (Decting Anomalies)│
 └────────────┘   └────────────────┘   └────────────────────┘
        │                 │                 │
        └──────────┬──────┴──────┬──────────┘
                   ▼             ▼
                ┌────────────────────────┐
                │ Final Risk Decision    │
                └──────────┬─────────────┘
                           ▼
            ┌──────────────────────────────┐
            │ LLM Risk Summary Generator   │
            └──────────────┬───────────────┘
                           ▼
            ┌──────────────────────────────┐
            │ API Response / Kafka Alerts  │
            └──────────────────────────────┘

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
- It pushes alerts to a compliance_alerts topic, helping in real-time messaging

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

7. Start FastAPI Server
python app.py
4 files, baseline.pkl, scaler.pkl, shap_explainer.pkl and feature_names.pkl, are generated in \models folder

8. Have Docker Desktop open.

9. Open Swagger UI 
http://localhost:7860/docs#/

10. Open RedPanda Compliance Alert website, a user dashboard
Write a test JSON in Swagger UI and check here. The output will be visible here. 
http://localhost:8080/topics/compliance_alerts/?s=200&pageSize=10&sort=
Confirm that the Docker container, redpanda-console, is running for local hosting.

12. For hosting, HuggingFace is used. We push the Dockerfile into HuggingFace for hosting purposes
Dockerfile and docker-compose.yml files are present in this repository.




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

## Assumptions, Improvements and Limitations
1. An attempt was made to deploy the project on Hugging Face Spaces for demonstration purposes. However, deployment was unsuccessful due to resource constraints and dependency limitations, particularly with heavy libraries such as PyTorch and SHAP, which require higher memory and longer build times. Additionally, the model size and runtime requirements exceeded the platform’s free-tier capabilities, leading to build failures and execution issues. As a result, the project was successfully demonstrated in a local environment, ensuring full functionality, real-time processing, and stable performance without resource limitations.

2. AWS, while powerful, is overkill for a quick demo due to complex setup, higher resource requirements, and potential unexpected costs.
3. Render, though simpler, struggles with heavy ML dependencies like PyTorch and SHAP due to strict RAM limits and long build times. Both platforms introduce friction for fast deployment, making them less suitable for rapid prototyping.
4. The system design includes integration with the Mistral-7B model via the Hugging Face API, as reflected in the app.py implementation. However, due to API access constraints (such as missing/limited API token usage and runtime restrictions), the LLM inference calls were not executed during deployment and testing. Despite this, the architecture is fully prepared for LLM-based summarization, and the integration can be activated seamlessly once valid API access and sufficient resources are available.
5. Creating a frontend dashboard
6. Experiment tracking using MLflow or Weights & Biases was planned; however, it was not implemented due to the evolving nature of the pipeline and prioritization of core system functionality. This can be incorporated in future iterations once the experimentation workflow is stabilized.
7. The synthetic dataset generated is symbolic of real-world data, hence its very unbalanced with respect to the number of anomalies among total entries. This causes Isolation Model to have similar performance metrics compared to that of LSTM Autoencoder model.


## Project Structure
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








