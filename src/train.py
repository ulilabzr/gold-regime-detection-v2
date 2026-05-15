# -*- coding: utf-8 -*-
"""
train.py -- Gold Price Regime Detection Pipeline
Menggunakan pendekatan klasifikasi untuk memprediksi arah harga emas bulan depan.

Sesuai kode Kaggle notebook (struktur identik, akurasi mirip).

Catatan Metrik Historis (Versi Regresi Sebelumnya):
Metrik Evaluasi Kinerja model diukur menggunakan tiga metrik regresi standar:
- RMSE (Root Mean Squared Error): Menunjukkan rata-rata penyimpangan prediksi terhadap nilai asli dalam satuan Dolar.
- MAE (Mean Absolute Error): Menunjukkan rata-rata selisih absolut antara prediksi dan nilai aktual.
- R-Squared (R²): Menunjukkan seberapa besar persentase varians harga emas yang dapat dijelaskan oleh model.
(Saat ini kode berjalan sebagai klasifikasi dan dievaluasi menggunakan Accuracy & AUC).

Pipeline:
    Step 1 : Load & EDA
    Step 2 : Feature Engineering (target creation)
    Step 3 : Train / Test Split (time-series safe, 80/20)
    Step 4 : Training 4 model (LR, RF, GBM, XGBoost)
    Step 5 : Evaluasi + MLflow logging
    Step 6 : Simpan best model sebagai pickle

Cara menjalankan:
    python src/train.py
"""

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend untuk server/CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# =============================================================================
# KONSTANTA
# =============================================================================

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "gold_advanced_features.csv")
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH= os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PLOTS_DIR = os.path.join(ROOT, "reports", "figures")

EXPERIMENT_NAME = "Gold Regime Detection"
TRAIN_RATIO     = 0.8

# 25 fitur (sama persis dengan Kaggle notebook)
FEATURES = [
    'price', 'year', 'month', 'quarter',
    'month_sin', 'month_cos',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'roll_mean_3', 'roll_mean_6', 'roll_mean_12',
    'roll_std_3', 'roll_std_6',
    'momentum_1', 'momentum_3',
    'pct_change_1', 'pct_change_3',
    'ewm_3', 'ewm_6',
    'price_to_roll3', 'price_to_roll12',
    'volatility_3'
]
TARGET = 'target'


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load CSV dan konversi kolom date ke datetime."""
    print(f"[Step 1] Memuat dataset: {path}")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"         Shape: {df.shape}")
    return df


# =============================================================================
# STEP 1b: EDA — Simpan plot sebagai PNG untuk MLflow artifacts
# =============================================================================

def run_eda(df: pd.DataFrame, out_dir: str) -> list:
    """
    Jalankan EDA dan simpan semua plot ke folder out_dir.
    Return list path gambar yang dihasilkan.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    sns.set(style="whitegrid")

    # 1. Price trend + rolling means
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['price'],        label='Price',          color='blue',   lw=1.5)
    ax.plot(df['date'], df['roll_mean_3'],  label='Roll Mean 3',    color='orange', lw=1)
    ax.plot(df['date'], df['roll_mean_6'],  label='Roll Mean 6',    color='green',  lw=1)
    ax.plot(df['date'], df['roll_mean_12'], label='Roll Mean 12',   color='red',    lw=1)
    ax.set_title("Price Trend with Rolling Means")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "01_price_trend.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 2. Lag features
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['price'], label='Price', color='black', lw=1.5)
    ax.plot(df['date'], df['lag_1'], label='Lag 1', color='purple', lw=1)
    ax.plot(df['date'], df['lag_2'], label='Lag 2', color='cyan',   lw=1)
    ax.plot(df['date'], df['lag_3'], label='Lag 3', color='magenta',lw=1)
    ax.set_title("Lag Features Comparison")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "02_lag_features.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 3. Momentum & percentage change
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['momentum_1'],  label='Momentum 1', color='brown', lw=1)
    ax.plot(df['date'], df['momentum_3'],  label='Momentum 3', color='pink',  lw=1)
    ax.plot(df['date'], df['pct_change_1'],label='% Change 1', color='olive', lw=1)
    ax.plot(df['date'], df['pct_change_3'],label='% Change 3', color='teal',  lw=1)
    ax.set_title("Momentum & Percentage Change")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "03_momentum.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 4. EWM
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['price'], label='Price', color='black', lw=1.5)
    ax.plot(df['date'], df['ewm_3'], label='EWM 3', color='red',  lw=1)
    ax.plot(df['date'], df['ewm_6'], label='EWM 6', color='blue', lw=1)
    ax.set_title("Exponential Weighted Moving Averages")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "04_ewm.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 5. Volatility
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['roll_std_3'],  label='Rolling Std 3', color='orange', lw=1)
    ax.plot(df['date'], df['roll_std_6'],  label='Rolling Std 6', color='green',  lw=1)
    ax.plot(df['date'], df['volatility_3'],label='Volatility 3',  color='red',    lw=1)
    ax.set_title("Volatility Analysis")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "05_volatility.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 6. Cyclical encoding
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df['month_sin'], df['month_cos'], c=df['month'], cmap='viridis', s=10)
    plt.colorbar(sc, ax=ax, label='Month')
    ax.set_title("Cyclical Encoding of Month")
    ax.set_xlabel("Month Sin"); ax.set_ylabel("Month Cos")
    plt.tight_layout()
    p = os.path.join(out_dir, "06_cyclical.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 7. Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df.drop(columns=['date']).corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    p = os.path.join(out_dir, "07_correlation.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # 8. Price ratios
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['price_to_roll3'],  label='Price / Roll3',  color='blue', lw=1)
    ax.plot(df['date'], df['price_to_roll12'], label='Price / Roll12', color='red',  lw=1)
    ax.set_title("Price Ratio Features")
    ax.legend(); plt.tight_layout()
    p = os.path.join(out_dir, "08_price_ratios.png")
    plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    print(f"         EDA: {len(saved)} plot disimpan ke {out_dir}")
    return saved


# =============================================================================
# STEP 2: FEATURE ENGINEERING — Target creation
# =============================================================================

def prepare_features(df: pd.DataFrame):
    """
    Buat kolom target: 1 jika harga bulan depan NAIK, 0 jika TURUN.
    Tidak ada data leakage karena menggunakan shift(-1).
    """
    print("\n[Step 2] Feature engineering...")
    df = df.copy()
    df[TARGET] = (df['price'].shift(-1) > df['price']).astype(int)
    df = df.dropna(subset=[TARGET] + FEATURES)

    X = df[FEATURES]
    y = df[TARGET]

    print(f"         Total sampel  : {len(X)}")
    print(f"         Target dist  : UP={y.sum()} ({y.mean()*100:.1f}%)  DOWN={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    return X, y


# =============================================================================
# STEP 3: TRAIN / TEST SPLIT + SCALING
# =============================================================================

def split_and_scale(X: pd.DataFrame, y: pd.Series):
    """
    Time-series safe split (NO shuffle).
    Scaler di-fit hanya di train set.
    """
    print("\n[Step 3] Train/test split & scaling...")
    split_idx = int(len(X) * TRAIN_RATIO)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"         Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler


# =============================================================================
# STEP 4: TRAINING + EVALUASI 4 MODEL
# =============================================================================

def train_and_evaluate(X_train_sc, X_test_sc, X_train, X_test, y_train, y_test, out_dir):
    """
    Latih 4 model (sama persis dengan Kaggle notebook):
    - Logistic Regression (scaled input)
    - Random Forest       (scaled input)
    - Gradient Boosting   (scaled input)
    - XGBoost             (unscaled input — tree-based, tidak butuh scaling)

    Return dict hasil dan model terbaik berdasarkan AUC.
    """
    print("\n[Step 4] Training & evaluasi model...")

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Random Forest"       : RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(),
        "XGBoost"             : XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
    }

    results   = {}
    roc_fig, roc_ax = plt.subplots(figsize=(10, 6))

    for name, model in models.items():
        # XGBoost pakai data unscaled (konsisten dengan Kaggle)
        Xtr = X_train if name == "XGBoost" else X_train_sc
        Xte = X_test  if name == "XGBoost" else X_test_sc

        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"model": model, "acc": acc, "auc": auc,
                         "y_pred": y_pred, "y_proba": y_proba}

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    AUC      : {auc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["DOWN","UP"],
                                    zero_division=0))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    roc_ax.plot([0,1],[0,1],'--', color='gray', label='Random')
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve Comparison")
    roc_ax.legend(); plt.tight_layout()
    roc_path = os.path.join(out_dir, "09_roc_curves.png")
    roc_fig.savefig(roc_path, dpi=120); plt.close(roc_fig)

    # Summary table
    summary = pd.DataFrame(
        {n: {"Accuracy": v["acc"], "AUC": v["auc"]} for n, v in results.items()}
    ).T.sort_values("AUC", ascending=False)
    print("\nFinal Model Comparison:")
    print(summary.to_string())

    best_name = summary.index[0]
    print(f"\n  Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    return results, summary, roc_path, best_name


# =============================================================================
# STEP 5: MLflow LOGGING (per model)
# =============================================================================

def log_to_mlflow(results, summary, eda_plots, roc_path, scaler):
    """Log semua model dan metrics ke MLflow."""
    print("\n[Step 5] Logging ke MLflow...")

    tracking_uri = os.path.join(ROOT, "mlruns")
    mlflow.set_tracking_uri(f"file:///{tracking_uri.replace(os.sep, '/')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1 parent run untuk semua model
    with mlflow.start_run(run_name="GoldRegimeDetection_AllModels") as parent:

        # Log EDA plots
        for p in eda_plots:
            mlflow.log_artifact(p, artifact_path="eda")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Log summary table sebagai CSV
        summary_path = os.path.join(PLOTS_DIR, "model_summary.csv")
        summary.to_csv(summary_path)
        mlflow.log_artifact(summary_path, artifact_path="results")

        for name, info in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_name", name)
                mlflow.log_metric("accuracy", info["acc"])
                mlflow.log_metric("auc",      info["auc"])

    print("         MLflow logging selesai.")


# =============================================================================
# STEP 6: SIMPAN BEST MODEL
# =============================================================================

def save_best_model(results, best_name, scaler, summary):
    """Simpan best model + scaler + summary ke folder model/."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(results[best_name]["model"], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Simpan metadata model (untuk app.py)
    meta = {
        "best_model_name": best_name,
        "accuracy"       : results[best_name]["acc"],
        "auc"            : results[best_name]["auc"],
        "features"       : FEATURES,
        "all_results"    : {n: {"acc": v["acc"], "auc": v["auc"]}
                            for n, v in results.items()},
    }
    import json
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Step 6] Model disimpan:")
    print(f"         {MODEL_PATH}")
    print(f"         {SCALER_PATH}")


# =============================================================================
# PIPELINE UTAMA
# =============================================================================

def run_pipeline():
    print("=" * 60)
    print("  GOLD REGIME DETECTION -- Classification Pipeline")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Step 1: Load
    df = load_data(DATA_PATH)

    # Step 1b: EDA
    eda_plots = run_eda(df, PLOTS_DIR)

    # Step 2: Feature engineering
    X, y = prepare_features(df)

    # Step 3: Split + scale
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(X, y)

    # Step 4: Train + evaluate
    results, summary, roc_path, best_name = train_and_evaluate(
        X_train_sc, X_test_sc, X_train, X_test, y_train, y_test, PLOTS_DIR
    )

    # Step 5: MLflow
    log_to_mlflow(results, summary, eda_plots, roc_path, scaler)

    # Step 6: Save
    save_best_model(results, best_name, scaler, summary)

    print("\n[Done] Pipeline selesai!")
    print("       MLflow UI : mlflow ui  ->  http://localhost:5000")
    print("       Streamlit : streamlit run src/app.py  ->  http://localhost:8501")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
