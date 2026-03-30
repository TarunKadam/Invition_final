import polars as pl
import os

# ===========================
# Path Configuration
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "../data/forex_events.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../data/featured_events.csv")


# ===========================
# Utility Functions
# ===========================
def load_data(file_path: str) -> pl.DataFrame:
    """
    Load CSV and parse timestamp to datetime.
    """
    print(f"Reading: {file_path}")
    df = pl.read_csv(file_path).with_columns(
        pl.col("timestamp").str.to_datetime()
    )
    return df


def ensure_columns(df: pl.DataFrame, defaults: dict) -> pl.DataFrame:
    """
    Ensure critical columns exist; inject defaults if missing.
    """
    for col, default in defaults.items():
        if col not in df.columns:
            print(f"'{col}' missing! Injecting default: {default}")
            df = df.with_columns(pl.lit(default).alias(col))
    return df


# ===========================
# Feature Engineering Functions
# ===========================
def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling statistics features: mean, std, z-score of amount.
    """
    df = df.with_columns([
        pl.col("amount").mean().over("user_id").alias("avg_volume_7d"),
        pl.col("amount").std().over("user_id").alias("std_volume_7d")
    ]).with_columns(
        ((pl.col("amount") - pl.col("avg_volume_7d")) / 
         (pl.col("std_volume_7d") + 1e-6)).alias("volume_zscore")
    )
    return df


def add_inter_event_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute time delta between consecutive events per user.
    """
    df = df.with_columns(
        (pl.col("timestamp").diff().dt.total_seconds().over("user_id"))
        .fill_null(0)
        .alias("inter_event_time_delta")
    )
    return df


def add_pnl_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute PnL volatility per user.
    """
    df = df.with_columns(
        pl.col("pnl").std().over("user_id").alias("pnl_volatility")
    )
    return df


def add_device_ip_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add device/IP anomaly proxy features.
    """
    df = df.with_columns(
        pl.col("user_ip").n_unique().over("user_id").alias("ip_deviation_score")
    )
    return df


def add_trade_pattern_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flag clustered trades (events within 10 seconds).
    """
    df = df.with_columns(
        pl.when(pl.col("inter_event_time_delta") < 10)
        .then(1).otherwise(0)
        .alias("is_clustered_trade")
    )
    return df


def add_login_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute login velocity (number of login events per user).
    """
    df = df.with_columns(
        pl.col("event_type").filter(pl.col("event_type") == "login")
        .count()
        .over("user_id")
        .alias("login_velocity_6h")
    )
    return df


# ===========================
# Main Pipeline
# ===========================
def engineer_features(file_path: str) -> pl.DataFrame:
    """
    Main feature engineering pipeline.
    """
    # Load and validate data
    df = load_data(file_path)
    df = ensure_columns(df, {
        "pnl": 0.0,
        "user_ip": "0.0.0.0",
        "event_type": "trade",
        "amount": 0.0
    })

    # Sort by user and timestamp for proper rolling calculations
    df = df.sort(["user_id", "timestamp"])

    # Apply feature engineering
    df = add_rolling_features(df)
    df = add_inter_event_features(df)
    df = add_pnl_features(df)
    df = add_device_ip_features(df)
    df = add_trade_pattern_features(df)
    df = add_login_features(df)

    # Fill remaining nulls
    return df.fill_null(0)


# ===========================
# Entry Point
# ===========================
if __name__ == "__main__":
    try:
        featured_df = engineer_features(INPUT_PATH)
        featured_df.write_csv(OUTPUT_PATH)
        print(f"Success! Created: {OUTPUT_PATH}")
        print(f"Final Feature Set: {featured_df.columns}")
    except Exception as e:
        print(f"Error during processing: {e}")