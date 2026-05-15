"""
prepare_data.py -- Buat gold_advanced_features.csv dari monthly.csv (raw price data).
Menghasilkan semua feature engineering yang dibutuhkan oleh train.py.
Jalankan sekali sebelum train.py.
"""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "monthly.csv")
OUT_PATH = os.path.join(ROOT, "data", "gold_advanced_features.csv")

df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

# Normalise column names (Date -> date, Price -> price)
df = df.rename(columns={"Date": "date", "Price": "price"})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# ── Time features ────────────────────────────────────────────────────────────
df['year']    = df['date'].dt.year
df['month']   = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ── Lag features ─────────────────────────────────────────────────────────────
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df['price'].shift(lag).fillna(df['price'])

# ── Rolling statistics ────────────────────────────────────────────────────────
for w in [3, 6, 12]:
    df[f'roll_mean_{w}'] = df['price'].shift(1).rolling(w, min_periods=1).mean()
    df[f'roll_std_{w}']  = df['price'].shift(1).rolling(w, min_periods=1).std().fillna(0)

# ── Momentum & returns ────────────────────────────────────────────────────────
df['momentum_1']   = df['price'] - df['price'].shift(1).fillna(df['price'])
df['momentum_3']   = df['price'] - df['price'].shift(3).fillna(df['price'])
df['pct_change_1'] = df['price'].pct_change(1).fillna(0)
df['pct_change_3'] = df['price'].pct_change(3).fillna(0)

# ── EWM ──────────────────────────────────────────────────────────────────────
df['ewm_3'] = df['price'].shift(1).ewm(span=3, adjust=False).mean().fillna(df['price'])
df['ewm_6'] = df['price'].shift(1).ewm(span=6, adjust=False).mean().fillna(df['price'])

# ── Price ratios ──────────────────────────────────────────────────────────────
df['price_to_roll3']  = df['price'] / df['roll_mean_3'].replace(0, np.nan).fillna(df['price'])
df['price_to_roll12'] = df['price'] / df['roll_mean_12'].replace(0, np.nan).fillna(df['price'])

# rename roll_std_3 -> volatility_3 (alias, keep both)
df['volatility_3'] = df['roll_std_3']

df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} rows -> {OUT_PATH}")
print(df.shape)
print(df.tail(3)[['date','price']].to_string())
