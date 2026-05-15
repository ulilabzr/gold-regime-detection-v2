# -*- coding: utf-8 -*-
"""
train.py -- Script training model Random Forest untuk prediksi harga emas (XAU/USD)
dengan MLflow experiment tracking.

Pipeline:
    Step 1: Load & preprocess data
    Step 2: Training model RandomForestRegressor
    Step 3: Evaluasi model (RMSE, MAE, R²)
    Step 4: Log semua hasil ke MLflow
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Path folder penyimpanan model pickle (untuk deployment Streamlit Cloud)
MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

# =============================================================================
# KONSTANTA — Sesuai spesifikasi KONTEKS.md
# =============================================================================

# Path dataset relatif terhadap root proyek
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gold_advanced_features.csv")

# 6 fitur input yang digunakan untuk prediksi
FEATURES = ["lag_1", "lag_2", "roll_mean_3", "roll_mean_6", "volatility_3", "momentum_1"]

# Kolom target
TARGET = "price"

# Hyperparameter model
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

# Rasio train/test split (no shuffle — menjaga urutan waktu)
TRAIN_RATIO = 0.8

# Nama experiment MLflow
EXPERIMENT_NAME = "Gold Price Prediction - Random Forest"


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data(path: str):
    """
    Load dan preprocess dataset dari file CSV.

    Langkah:
    1. Baca file CSV dengan pd.read_csv()
    2. Pilih 6 fitur + kolom target
    3. Hapus baris dengan NaN (dihasilkan dari lag/rolling)
    4. Split secara time-series (NO shuffle) — 80% train, 20% test

    Parameters
    ----------
    path : str
        Path absolut atau relatif menuju file CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    print(f"[Step 1] Memuat dataset dari: {path}")

    # Baca CSV
    df = pd.read_csv(path)
    print(f"         Total baris sebelum dropna: {len(df)}")

    # Pilih kolom yang diperlukan
    df = df[FEATURES + [TARGET]]

    # Hapus baris NaN (dari perhitungan lag dan rolling)
    df = df.dropna()
    print(f"         Total baris setelah dropna: {len(df)}")

    # Time-series split — tanpa shuffle untuk menjaga urutan kronologis
    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    print(f"         Train size: {len(X_train)} baris | Test size: {len(X_test)} baris")
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 2: TRAINING MODEL
# =============================================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Melatih model RandomForestRegressor dengan hyperparameter default.

    Algoritma Random Forest dipilih karena:
    - Menangani interaksi non-linear antar fitur time-series
    - Tidak memerlukan feature scaling
    - Robust terhadap outlier
    - Mudah diinterpretasi via feature importance

    Parameters
    ----------
    X_train : pd.DataFrame
        Fitur training.
    y_train : pd.Series
        Target training (harga emas).

    Returns
    -------
    model : RandomForestRegressor
        Model yang sudah dilatih.
    """
    print(f"\n[Step 2] Melatih RandomForestRegressor...")
    print(f"         n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(f"         Training selesai.")
    return model


# =============================================================================
# STEP 2b: EVALUASI MODEL
# =============================================================================

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Mengevaluasi performa model pada data test.

    Metrik yang dihitung:
    - RMSE : Root Mean Squared Error (error dalam satuan dolar)
    - MAE  : Mean Absolute Error (rata-rata error absolut)
    - R²   : Coefficient of Determination (seberapa baik model menjelaskan varians)

    Parameters
    ----------
    model : RandomForestRegressor
        Model yang sudah dilatih.
    X_test : pd.DataFrame
        Fitur test.
    y_test : pd.Series
        Target aktual test.

    Returns
    -------
    metrics : dict
        Dictionary berisi nilai RMSE, MAE, dan R².
    """
    print(f"\n[Step 2b] Mengevaluasi model pada data test...")

    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    print(f"         RMSE : {rmse:.4f}")
    print(f"         MAE  : {mae:.4f}")
    print(f"         R²   : {r2:.4f}")

    return metrics, y_pred


# =============================================================================
# HELPER: BUAT VISUALISASI UNTUK ARTIFACT MLFLOW
# =============================================================================

def _save_actual_vs_predicted_plot(y_test, y_pred, output_path: str):
    """Simpan plot aktual vs prediksi sebagai file gambar."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(y_test)), y_test.values, label="Aktual", color="#2ecc71", linewidth=2)
    ax.plot(range(len(y_pred)), y_pred,         label="Prediksi", color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_title("Harga Emas: Aktual vs Prediksi (Test Set)", fontsize=14)
    ax.set_xlabel("Index Test")
    ax.set_ylabel("Harga (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_feature_importance_plot(model, feature_names, output_path: str):
    """Simpan plot feature importance sebagai file gambar."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_feat = [feature_names[i] for i in idx]
    palette = sns.color_palette("viridis", len(sorted_feat))
    sns.barplot(x=importances[idx], y=sorted_feat, hue=sorted_feat, palette=palette, legend=False, ax=ax)
    ax.set_title("Feature Importance — Random Forest", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# STEP 3: MLflow TRACKING + PIPELINE UTAMA
# =============================================================================

def run_pipeline():
    """
    Menjalankan end-to-end pipeline:
    1. Load data
    2. Train model
    3. Evaluasi model
    4. Log semua ke MLflow (params, metrics, model, artifacts)

    MLflow menyimpan semua hasil ke folder mlruns/ di root proyek.
    Jalankan `mlflow ui` untuk melihat hasilnya di browser.
    """
    print("=" * 60)
    print("  GOLD PRICE PREDICTION -- Random Forest Pipeline")
    print("=" * 60)

    # ─── Load Data ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # ─── Train Model ──────────────────────────────────────────────
    model = train_model(X_train, y_train)

    # ─── Evaluasi Model ───────────────────────────────────────────
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # ─── MLflow Tracking ──────────────────────────────────────────
    print(f"\n[Step 3] Logging ke MLflow...")

    # Set tracking URI ke root proyek (relatif)
    tracking_uri = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
    mlflow.set_tracking_uri(f"file:///{tracking_uri.replace(os.sep, '/')}")

    # Set atau buat experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest_v1") as run:

        # --- Log Parameters ---
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth",    MAX_DEPTH)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("features",     str(FEATURES))
        print(f"         Parameters logged.")

        # --- Log Metrics ---
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("r2",   metrics["r2"])
        print(f"         Metrics logged: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

        # --- Simpan & Log Visualisasi ---
        tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_plots")
        os.makedirs(tmp_dir, exist_ok=True)

        plot_avp  = os.path.join(tmp_dir, "actual_vs_predicted.png")
        plot_fi   = os.path.join(tmp_dir, "feature_importance.png")

        _save_actual_vs_predicted_plot(y_test, y_pred, plot_avp)
        _save_feature_importance_plot(model, FEATURES, plot_fi)

        mlflow.log_artifact(plot_avp,  artifact_path="plots")
        mlflow.log_artifact(plot_fi,   artifact_path="plots")
        print(f"         Visualisasi artifacts logged.")

        # --- Log Model ---
        mlflow.sklearn.log_model(model, "random_forest_model")
        print(f"         Model artifact logged.")

        run_id = run.info.run_id
        print(f"\n         [OK] Run ID: {run_id}")

    # ─── Simpan Model sebagai Pickle (untuk Streamlit Cloud) ──────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[Pickle] Model disimpan ke: {MODEL_PATH}")

    print("\n[Done] Pipeline selesai!")
    print("       Jalankan: mlflow ui")
    print("       Buka browser: http://localhost:5000")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()
