import pandas as pd 
import uuid
from datetime import datetime, timedelta
import random
import numpy as np

def random_choice_weighted(options, weights=None):
    return random.choices(options, weights=weights)[0]

def mark_anomaly(row, probability=1.0, amplify=False):
    if random.random() < probability:
        row['is_anomaly'] = max(row['is_anomaly'], 1)
        if amplify:
            # Amplify numeric features to make anomalies stand out for LSTM
            for k in ['amount','lot_size','price','margin_usage','session_duration','file_size_kb','profit_amount','loss_amount','equity']:
                if k in row:
                    row[k] = row[k] * random.uniform(2.0, 3.5)  # stronger amplification
        return True
    return False

def generate_user():
    return {
        "user_id": str(uuid.uuid4()),
        "country": random_choice_weighted(["IN", "US", "UK", "AE", "SG", "DE"], weights=[0.35,0.2,0.15,0.1,0.1,0.1]),
        "device_preference": random_choice_weighted(["mobile", "desktop", "tablet"], weights=[0.65,0.3,0.05]),
        "device_history": [],
        "risk_profile": random_choice_weighted(["low", "medium", "high"], weights=[0.6,0.3,0.1]),
        "account_age_days": random.randint(1, 1000),
        "kyc_status": random.choice(["pending", "verified", "rejected"])
    }

def generate_ip(country, country_ip_map):
    prefix = random.choice(country_ip_map[country])
    return f"{prefix}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def select_device(user):
    if random.random() < 0.85:
        return user["device_preference"]
    return random_choice_weighted(["mobile","desktop","tablet"], weights=[0.65,0.3,0.05])

def update_event_row(row, event_type, device):
    login_events = ['login_success','login_failure']
    profile_events = ['profile_update']
    kyc_events = ['kyc_submitted','kyc_approved','kyc_rejected']
    doc_events = ['document_upload']
    txn_events = ['deposit','withdrawal']
    trade_events = ['trade_open','trade_close']

    if event_type in login_events:
        row.update({
            'login_method': random.choice(['password','otp']),
            'failed_attempts': random.randint(0,5),
            'device_trusted': device == row['device']
        })
    elif event_type == 'logout':
        row['session_duration'] = random.randint(10,7200)
    elif event_type == 'password_change':
        row.update({
            'change_method': random.choice(['manual','forgot_password']),
            'security_check_passed': random.choice([True,False])
        })
    elif event_type in profile_events:
        row.update({
            'fields_changed': random.choice(['email','phone','address']),
            'change_frequency': random.randint(1,5)
        })
    elif event_type in kyc_events:
        row.update({
            'document_type': random.choice(['passport','aadhaar','license']),
            'kyc_level': random.choice(['basic','advanced']),
            'review_time_sec': random.randint(60,10000)
        })
    elif event_type in doc_events:
        row.update({
            'document_type': random.choice(['passport','aadhaar','license']),
            'file_size_kb': random.randint(100,5000)
        })
    elif event_type in txn_events:
        amount = round(random.uniform(100,10000),2)
        row.update({
            'amount': amount,
            'payment_method': random.choice(['card','bank','crypto']),
            'currency': random.choice(['USD','INR','EUR'])
        })
    elif event_type in trade_events:
        lot_size = round(random.uniform(0.01,5.0),2)
        row.update({
            'instrument': random.choice(['EURUSD','GBPUSD','XAUUSD','BTCUSD']),
            'lot_size': lot_size,
            'price': round(random.uniform(1.0,2000.0),2),
            'leverage': random.choice([10,50,100]),
            'margin_usage': round(random.uniform(5,100),2)
        })
    elif event_type == 'order_placed':
        row.update({
            'instrument': random.choice(['EURUSD','GBPUSD','XAUUSD']),
            'order_type': random.choice(['market','limit','stop']),
            'lot_size': round(random.uniform(0.01,3.0),2)
        })
    elif event_type == 'order_cancelled':
        row['cancel_reason'] = random.choice(['user_request','timeout','margin_issue'])
    elif event_type == 'margin_call':
        row.update({
            'margin_level': round(random.uniform(0,50),2),
            'equity': round(random.uniform(100,10000),2)
        })
    elif event_type == 'stop_loss_triggered':
        row.update({
            'instrument': random.choice(['EURUSD','GBPUSD','XAUUSD']),
            'loss_amount': round(random.uniform(10,1000),2)
        })
    elif event_type == 'take_profit_triggered':
        row.update({
            'instrument': random.choice(['EURUSD','GBPUSD','XAUUSD']),
            'profit_amount': round(random.uniform(10,2000),2)
        })

def inject_anomalies(row, user, event_type, consecutive_count=0):
    """
    Amplify anomalies in sequences: consecutive anomalous events
    """
    # High-risk withdrawal
    if event_type == 'withdrawal' and row.get('amount',0) > 5000:
        mark_anomaly(row, probability=0.9, amplify=True)
    
    # Multiple anomalies in sequence for LSTM
    if consecutive_count > 0:
        mark_anomaly(row, probability=0.8, amplify=True)
    
    # Login anomalies
    if event_type == 'login_failure' and row.get('failed_attempts',0) >= 3:
        mark_anomaly(row, probability=0.9)
    if event_type == 'login_success' and row.get('device_trusted') is False:
        mark_anomaly(row, probability=0.5)
    
    # Trade anomalies
    if event_type == 'trade_open' and (row.get('lot_size',0) > 3 or row.get('margin_usage',0) > 85):
        mark_anomaly(row, probability=0.85, amplify=True)
    
    # Document anomalies
    if event_type == 'document_upload' and row.get('file_size_kb',0) > 4000:
        mark_anomaly(row, probability=0.8, amplify=True)
    
    # Slight rare anomalies
    mark_anomaly(row, 0.02)

def generate_forex_data(n_events=50000, n_users=500, seq_anomaly_length=3):
    users = [generate_user() for _ in range(n_users)]
    data = []
    last_country = {}
    last_timestamp = {}
    consecutive_anomaly_counter = {u['user_id']:0 for u in users}
    
    country_ip_map = {
        "IN": ["49", "103", "117"],
        "US": ["3", "18", "54"],
        "UK": ["51", "81"],
        "AE": ["5", "94"],
        "SG": ["13", "45"],
        "DE": ["18", "91"]
    }

    for i in range(n_events):
        user = random.choice(users)
        timestamp = datetime.now() - timedelta(minutes=n_events-i)
        user_id = user['user_id']

        # Pick event type
        event_type = random_choice_weighted([
            'login_success','login_failure','logout','password_change','profile_update',
            'kyc_submitted','kyc_approved','kyc_rejected','document_upload',
            'deposit','withdrawal',
            'trade_open','trade_close','order_placed','order_cancelled','margin_call',
            'stop_loss_triggered','take_profit_triggered'
        ])
        
        final_country = user["country"] if random.random() < 0.9 else random.choice(list(country_ip_map.keys()))
        device = select_device(user)
        ip_address = generate_ip(final_country, country_ip_map)

        row = {
            'timestamp': timestamp,
            'user_id': user_id,
            'event_type': event_type,
            'ip_address': ip_address,
            'country': final_country,
            'session_id': str(uuid.uuid4())[:10],
            'session_duration': random.randint(10, 3600),
            'device': device,
            'is_anomaly': 0,
            'is_high_risk_fraud': 0,
            'device_anomaly': False,
            'geo_anomaly': False,
            'amount': random.uniform(100,5000),
            'lot_size': random.uniform(0.01,5.0),
            'price': random.uniform(1,2000),
            'margin_usage': random.uniform(5,100),
            'file_size_kb': random.randint(100,5000),
            'profit_amount': random.uniform(0,2000),
            'loss_amount': random.uniform(0,1000),
            'equity': random.uniform(100,10000)
        }

        # GEO anomaly
        if user_id in last_country:
            prev_country = last_country[user_id]
            prev_time = last_timestamp[user_id]
            if prev_country != final_country and abs((timestamp - prev_time).total_seconds()) < 1800:
                if random.random() < 0.7:
                    row['is_anomaly'] = 1
                    row['geo_anomaly'] = True

        # Device switching anomaly
        user["device_history"].append(device)
        if len(user["device_history"]) >= 3 and len(set(user["device_history"][-3:])) > 1:
            if random.random() < 0.6:
                row['is_anomaly'] = 1
                row['device_anomaly'] = True

        # Event-specific fields
        update_event_row(row, event_type, device)

        # Inject temporal anomalies
        consecutive_count = consecutive_anomaly_counter[user_id]
        inject_anomalies(row, user, event_type, consecutive_count)
        
        # Update consecutive anomaly counter
        if row['is_anomaly']:
            consecutive_anomaly_counter[user_id] = min(seq_anomaly_length, consecutive_anomaly_counter[user_id]+1)
        else:
            consecutive_anomaly_counter[user_id] = 0

        # High-risk withdrawal logic
        if row['is_anomaly'] and event_type == 'withdrawal' and random.random() < 0.9:
            row['is_high_risk_fraud'] = 1

        data.append(row)
        last_country[user_id] = final_country
        last_timestamp[user_id] = timestamp

    df = pd.DataFrame(data)
    return df

# Usage
if __name__ == "__main__":
    df = generate_forex_data(n_events=50000, n_users=500, seq_anomaly_length=3)
    df.to_csv("../data/forex_events.csv", index=False)
    print(f"Generated {len(df)} events")