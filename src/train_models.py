import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

# ===========================
# LSTM Autoencoder Definition
# ===========================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeat_hidden = hidden.permute(1, 0, 2).repeat(1, x.size(1), 1)
        x_recon, _ = self.decoder(repeat_hidden)
        return x_recon

# ===========================
# Utility Functions
# ===========================
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def save_objects(objects_dict, model_dir):
    for name, obj in objects_dict.items():
        joblib.dump(obj, os.path.join(model_dir, name))

# ===========================
# Train Isolation Forest
# ===========================
def train_isolation_forest(X_raw, model_dir, shap_subset=2000):
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_raw)

    explainer = shap.Explainer(iso_forest.predict, X_raw[:min(shap_subset,len(X_raw))])

    save_objects({
        "baseline.pkl": iso_forest,
        "shap_explainer.pkl": explainer
    }, model_dir)
    
    print("Baseline & SHAP Explainer Saved.")
    return iso_forest

# ===========================
# Sequence Preparation
# ===========================
def create_sequences(df, feature_cols, seq_len=20):
    df_sorted = df.sort_values(['user_id','timestamp'])
    sequences = []
    for user_id, user_df in df_sorted.groupby('user_id'):
        # Handle missing columns
        user_features = user_df.reindex(columns=feature_cols, fill_value=0).values
        for i in range(len(user_features) - seq_len + 1):
            sequences.append(user_features[i:i+seq_len])
    return np.array(sequences)

# ===========================
# Encode categorical features
# ===========================
def encode_categoricals(df, feature_cols):
    encoders = {}
    df_encoded = df.copy()
    categorical_cols = df_encoded[feature_cols].select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

# ===========================
# Train LSTM Autoencoder
# ===========================
def train_lstm_autoencoder(df, feature_cols, model_dir, hidden_dim=8, epochs=80, lr=0.01, seq_len=5):
    df_encoded, encoders = encode_categoricals(df, feature_cols)

    # Fill missing numeric columns with 0
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Scale numeric features
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_encoded[feature_cols].fillna(0))
    
    save_objects({
        "scaler.pkl": scaler,
        "feature_names.pkl": feature_cols,
        "categorical_encoders.pkl": encoders
    }, model_dir)

    # Convert to sequences
    sequences = create_sequences(df_scaled, feature_cols, seq_len=seq_len)
    X_tensor = torch.tensor(sequences).float()
    
    model = LSTMAutoencoder(input_dim=len(feature_cols), hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training LSTM on sequences")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
    
    torch.save(model.state_dict(), os.path.join(model_dir, "advanced_model.pt"))
    print("Advanced Model Saved.")
    return model, sequences

# ===========================
# Anomaly Scoring
# ===========================
def lstm_anomaly_score(model, sequences):
    model.eval()
    X_tensor = torch.tensor(sequences).float()
    with torch.no_grad():
        recon = model(X_tensor)
        mse = torch.mean((recon - X_tensor) ** 2, dim=[1,2]).numpy()
    return mse

def iso_anomaly_score(iso_model, X_raw):
    scores = iso_model.decision_function(X_raw)
    anomaly_scores = -scores
    return anomaly_scores

# ===========================
# Compare Models
# ===========================
def compare_models(sequences, iso_model, lstm_model, df, feature_cols, seq_len=5):
    # Flatten sequences for Isolation Forest
    n_seq, seq_len_seq, n_feat = sequences.shape
    X_seq_flat = sequences.reshape(n_seq, seq_len_seq * n_feat)

    # Isolation Forest
    iso_scores = iso_model.decision_function(X_seq_flat)
    iso_threshold = np.percentile(-iso_scores, 95)
    iso_pred = (-iso_scores > iso_threshold).astype(int)

    # LSTM Autoencoder
    lstm_mse = lstm_anomaly_score(lstm_model, sequences)
    lstm_threshold = np.percentile(lstm_mse, 95)
    lstm_pred = (lstm_mse > lstm_threshold).astype(int)

    # Align y_true for sequences: last element in each sequence
    y_true_seq = []
    df_sorted = df.sort_values(['user_id','timestamp'])
    for user_id, user_df in df_sorted.groupby('user_id'):
        user_labels = user_df['is_anomaly'].values
        for i in range(len(user_labels) - seq_len + 1):
            y_true_seq.append(user_labels[i + seq_len - 1])
    y_true_seq = np.array(y_true_seq)

    print(f"--- Model Comparison ---")
    print(f"Isolation Forest detected {iso_pred.sum()} anomalies")
    print(f"LSTM Autoencoder detected {lstm_pred.sum()} anomalies")

    print("\n--- Performance Metrics ---")
    for name, pred in [("Isolation Forest", iso_pred), ("LSTM Autoencoder", lstm_pred)]:
        precision = precision_score(y_true_seq, pred)
        recall = recall_score(y_true_seq, pred)
        f1 = f1_score(y_true_seq, pred)
        print(f"{name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# ===========================
# Main Training Function
# ===========================
def train_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/forex_events.csv")  # use new temporal anomaly dataset
    model_dir = os.path.join(base_dir, "../models")
    os.makedirs(model_dir, exist_ok=True)

    features = [
        'session_duration', 'is_anomaly', 'is_high_risk_fraud', 'device_anomaly', 'geo_anomaly',
        'amount', 'lot_size', 'price', 'leverage', 'margin_usage', 'failed_attempts', 'device_trusted',
        'kyc_level', 'review_time_sec', 'profit_amount', 'loss_amount', 'file_size_kb',
        'security_check_passed', 'fields_changed', 'change_frequency', 'margin_level', 'equity'
    ]
    
    df = load_data(data_path)
    
    # Train LSTM Autoencoder
    lstm_model, sequences = train_lstm_autoencoder(df, features, model_dir, hidden_dim=8, epochs=80, lr=0.01, seq_len=5)
    
    # Flatten sequences and train Isolation Forest on same flattened sequences
    n_seq, seq_len_seq, n_feat = sequences.shape
    X_seq_flat = sequences.reshape(n_seq, seq_len_seq * n_feat)
    iso_forest = train_isolation_forest(X_seq_flat, model_dir, shap_subset=2000)
    
    # Compare models
    compare_models(sequences, iso_forest, lstm_model, df, features, seq_len=5)

# ===========================
# Entry Point
# ===========================
if __name__ == "__main__":
    train_all()