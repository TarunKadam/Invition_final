import os
import joblib
import torch
import numpy as np
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaProducer
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

# ===========================
# 1. Infrastructure & Globals
# ===========================
# Using the keys you set in Render
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_USER = os.getenv("KAFKA_USERNAME")
KAFKA_PASS = os.getenv("KAFKA_PASSWORD")
KAFKA_ENABLED = False
producer = None

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

ip_history = defaultdict(set)
user_login_stats = defaultdict(lambda: {"last_login": None, "last_ip": None, "failed_count": 0})
user_finance_stats = defaultdict(lambda: {"total_deposited": 0.0, "total_withdrawn": 0.0, "minimal_trading": False})
user_trade_history = defaultdict(list)
user_bonus_stats = defaultdict(lambda: {"bonus_credit": 0.0, "required_volume": 0.0, "current_volume": 0.0})
last_kyc_change = defaultdict(lambda: None)
NEWS_EVENTS = [(13, 30), (14, 0), (19, 0)]  # HH:MM of important news
UNUSUAL_HOURS = list(range(23, 24)) + list(range(0, 6))

# Updated Kafka Connection with SASL_SSL Security
if KAFKA_BROKER and KAFKA_USER and KAFKA_PASS:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-256",
            sasl_plain_username=KAFKA_USER,
            sasl_plain_password=KAFKA_PASS,
            request_timeout_ms=5000  # Higher timeout for cloud latency
        )
        KAFKA_ENABLED = True
        print("✅ Kafka Connected to Redpanda Cloud")
    except Exception as e:
        print(f"❌ Running in Demo Mode: Kafka connection failed ({e})")
else:
    print("⚠️ Kafka Credentials missing. Running in Demo Mode.")

# ===========================
# 2. LSTM Skeleton for Loading
# ===========================
class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        seq_len = x.size(1)
        repeat_hidden = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)
        out, _ = self.decoder(repeat_hidden)
        return out

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if producer:
        producer.flush()

# ===========================
# 3. FastAPI Setup
# ===========================
app = FastAPI(title="ForexGuard Compliance API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 4. Load Models & Assets
# ===========================
base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure this path is correct relative to your app.py on Render
model_dir = os.path.join(base_dir, "..", "models")

iso_forest = joblib.load(os.path.join(model_dir, "baseline.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
shap_explainer = joblib.load(os.path.join(model_dir, "shap_explainer.pkl"))
features_list = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

lstm_model = LSTMAutoencoder(input_dim=len(features_list), hidden_dim=8)
lstm_model.load_state_dict(torch.load(os.path.join(model_dir, "advanced_model.pt"), map_location=torch.device('cpu')))
lstm_model.eval()

# ===========================
# 5. Pydantic Input Model
# ===========================
class TradeActivity(BaseModel):
    user_id: str
    user_ip: str
    session_duration: float
    is_anomaly: float
    is_high_risk_fraud: float
    device_anomaly: float
    geo_anomaly: float
    amount: float
    lot_size: float
    price: float
    leverage: float
    margin_usage: float
    failed_attempts: float
    device_trusted: float
    kyc_level: float
    review_time_sec: float
    profit_amount: float
    loss_amount: float
    file_size_kb: float
    security_check_passed: float
    fields_changed: float
    change_frequency: float
    margin_level: float
    equity: float
    pnl: float
    avg_volume_7d: float
    std_volume_7d: float
    volume_zscore: float
    inter_event_time_delta: float
    pnl_volatility: float
    ip_deviation_score: float
    is_clustered_trade: float
    login_velocity_6h: float

# ===========================
# 6. Helper Functions
# ===========================
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def generate_llm_summary(user_id, culprit, amount, ip):
    if not HF_TOKEN:
        return f"High risk detected for {user_id}. Primary driver: {culprit}."
    prompt = f"<s>[INST] <<SYS>> You are a Senior Forex Compliance Officer. Summarize: User {user_id}, Flag: {culprit}, Amount: ${amount}, IP: {ip}. [/INST]"
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        return response.json()[0]['generated_text'].split("[/INST]")[-1].strip()
    except:
        return f"Manual audit required: {culprit} detected for ${amount}."

# ===========================
# 7. Anomaly Detection Helpers
# ===========================
def lstm_score(data_array):
    tensor = torch.tensor(data_array).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        recon = lstm_model(tensor)
        mse = float(torch.mean((recon - tensor) ** 2).numpy())
    return mse

def iso_score(data_array):
    X_flat = np.tile(data_array, (5,1)).reshape(1, -1)
    return -iso_forest.decision_function(X_flat)[0]

# ===========================
# 8. Graph / Network-Level Anomaly
# ===========================
def graph_network_anomaly(user_id, user_ip, trade_time, lot_size):
    anomaly_flags = []
    users_on_ip = ip_history[user_ip]
    if len(users_on_ip) > 3:
        anomaly_flags.append("IP_HUB")

    for other_user, trades in user_trade_history.items():
        if other_user == user_id:
            continue
        for t_time, t_lot in trades[-5:]:
            if abs((trade_time - t_time).total_seconds()) < 10 and t_lot == lot_size:
                anomaly_flags.append("SYNCHRONIZED_TRADES")
                break

    user_trade_history[user_id].append((trade_time, lot_size))
    ip_history[user_ip].add(user_id)
    return anomaly_flags

# ===========================
# 9. Anomaly Detection Endpoint
# ===========================
@app.post("/score")
async def score_trade(data: TradeActivity):
    data_array = np.array([getattr(data, f) for f in features_list])

    mse = lstm_score(data_array)
    iso_anom = iso_score(data_array)
    is_anomaly = (mse > 10.0) or (iso_anom < -0.5) # Simplified threshold check

    current_hour = datetime.now().hour
    culprit = "Normal"

    # Rule-based overrides
    if current_hour in UNUSUAL_HOURS or data.failed_attempts >= 3 or (data.user_id in user_login_stats and user_login_stats[data.user_id]["last_ip"] != data.user_ip):
        culprit = "Login_Anomaly"
        is_anomaly = True

    if data.amount > 5000 and user_finance_stats[data.user_id]["total_withdrawn"] == 0:
        culprit = "Financial_Anomaly"
        is_anomaly = True

    if data.lot_size > 3 or data.margin_usage > 85 or data.is_clustered_trade:
        culprit = "Trading_Anomaly"
        is_anomaly = True

    if data.session_duration > 7200 or data.device_anomaly or data.geo_anomaly:
        culprit = "Behavioral_Anomaly"
        is_anomaly = True

    for news_h, news_m in NEWS_EVENTS:
        if abs(datetime.now().hour - news_h) < 1:
            culprit = "Temporal_Anomaly"
            is_anomaly = True

    if data.fields_changed > 1 or data.kyc_level > 1:
        culprit = "Account_Risk"
        is_anomaly = True

    trade_time = datetime.now()
    graph_flags = graph_network_anomaly(data.user_id, data.user_ip, trade_time, data.lot_size)
    if graph_flags:
        culprit = "Graph/Network Anomaly: " + ",".join(graph_flags)
        is_anomaly = True

    summary = generate_llm_summary(data.user_id, culprit, data.amount, data.user_ip) if is_anomaly else "Normal transaction patterns observed."

    result = {"user_id": data.user_id, "is_anomaly": is_anomaly, "culprit": culprit, "summary": summary}
    result_safe = sanitize_for_json(result)

    if is_anomaly and KAFKA_ENABLED:
        try:
            producer.send('compliance_alerts', value=result_safe)
        except Exception as e:
            print(f"Error sending to Kafka: {e}")

    return result_safe

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)