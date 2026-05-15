# 📋 KONTEKS PROYEK: Data Mining - Prediksi Harga Emas (XAU/USD)

> **Untuk Vibe Coding Session** — Baca file ini sebelum mulai coding. Semua keputusan teknis sudah terdefinisi di sini.

---

## 🎯 1. TUJUAN PROYEK

Membangun **end-to-end Machine Learning pipeline** untuk memprediksi harga emas (XAU/USD) menggunakan pendekatan ML tradisional yang dilengkapi dengan:
- Experiment tracking via **MLflow**
- Web application deployment via **Streamlit**

**Output akhir yang diharapkan:**
1. Script `train.py` — melatih model dan log ke MLflow
2. Script `app.py` — aplikasi web Streamlit untuk prediksi
3. Laporan akademik lengkap (PDF/DOCX)

---

## 🛠️ 2. TECH STACK (JANGAN GANTI)

| Komponen | Library/Tool | Versi Minimum |
|---|---|---|
| Language | Python | 3.9+ |
| Data Processing | Pandas, NumPy | Latest |
| Modelling | Scikit-Learn | 1.3+ |
| Experiment Tracking | MLflow | 2.x |
| Web Deployment | Streamlit | 1.x |
| Visualisasi | Matplotlib, Seaborn | Latest |

```bash
# Instalasi semua dependency
pip install pandas numpy scikit-learn mlflow streamlit matplotlib seaborn
```

---

## 📁 3. STRUKTUR DIREKTORI PROYEK

```
gold-price-prediction/
│
├── KONTEKS.md                  ← File ini (baca dulu!)
│
├── data/
│   └── gold_advanced_features.csv   ← Dataset utama (ganti ke daily jika perlu)
│
├── src/
│   ├── train.py                ← Script training + MLflow logging
│   └── app.py                  ← Streamlit web app
│
├── notebooks/
│   └── eda.ipynb               ← Exploratory Data Analysis (opsional)
│
├── reports/
│   └── laporan.md              ← Draft laporan akademik
│
├── mlruns/                     ← Auto-generated oleh MLflow (jangan edit manual)
│
└── requirements.txt            ← Dependency list
```

---

## 📊 4. SPESIFIKASI DATASET

### Sumber
- **Platform:** Kaggle
- **Nama Dataset:** Gold Price History
- **File:** `gold_advanced_features.csv`
- **Catatan:** Untuk sesi ini, dataset harian (`daily`) akan digunakan sebagai pengganti

### Kolom yang Digunakan

| Nama Kolom | Tipe | Keterangan |
|---|---|---|
| `lag_1` | Float | Harga emas 1 hari sebelumnya |
| `lag_2` | Float | Harga emas 2 hari sebelumnya |
| `roll_mean_3` | Float | Rata-rata harga 3 hari terakhir |
| `roll_mean_6` | Float | Rata-rata harga 6 hari terakhir |
| `volatility_3` | Float | Standar deviasi harga 3 hari terakhir |
| `momentum_1` | Float | Selisih harga hari ini vs kemarin |
| **`price`** | **Float** | **TARGET — Harga emas hari ini (XAU/USD)** |

### Preprocessing Rules
- ✅ Drop NaN (dihasilkan dari perhitungan rolling/lag)
- ✅ Split data dengan **urutan waktu** (NO shuffle) — 80% train, 20% test
- ❌ TIDAK perlu scaling (Random Forest tidak memerlukannya)
- ❌ TIDAK perlu encoding (semua fitur sudah numerik)

---

## 🔁 5. ARSITEKTUR PIPELINE (4 LANGKAH — WAJIB DIIKUTI)

```
[CSV File]
    │
    ▼
┌─────────────────────────────────┐
│  STEP 1: LOAD DATA              │
│  - pd.read_csv()                │
│  - Pilih 6 fitur + target       │
│  - dropna()                     │
│  - Train/test split (no shuffle)│
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  STEP 2: MODELLING              │
│  - RandomForestRegressor        │
│  - n_estimators=100             │
│  - max_depth=10                 │
│  - model.fit(X_train, y_train)  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  STEP 3: MLflow TRACKING        │
│  - mlflow.start_run()           │
│  - Log params: n_estimators,    │
│    max_depth                    │
│  - Log metrics: RMSE, MAE, R²   │
│  - Log model sebagai artifact   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  STEP 4: DEPLOYMENT (Streamlit) │
│  - Load model dari mlruns/      │
│  - Sidebar input 6 fitur        │
│  - Tampilkan prediksi harga     │
└─────────────────────────────────┘
```

---

## 🤖 6. SPESIFIKASI MODEL

### Algoritma: Random Forest Regressor
**Alasan pemilihan:**
- Menangani interaksi non-linear antar fitur time-series dengan baik
- Tidak memerlukan feature scaling
- Robust terhadap outlier
- Mudah diinterpretasi via feature importance

### Hyperparameter Default

```python
RandomForestRegressor(
    n_estimators=100,   # Jumlah pohon dalam forest
    max_depth=10,       # Kedalaman maksimum tiap pohon
    random_state=42     # Reproducibility
)
```

### Metrik Evaluasi

| Metrik | Formula | Keterangan |
|---|---|---|
| **RMSE** | √(Σ(y_pred - y_true)²/n) | Error dalam satuan dolar |
| **MAE** | Σ\|y_pred - y_true\|/n | Rata-rata error absolut |
| **R²** | 1 - SS_res/SS_tot | Seberapa baik model menjelaskan varians data |

---

## 📊 7. SPESIFIKASI MLflow TRACKING

### Apa yang di-log:

```python
# PARAMETERS (input model)
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_param("random_state", 42)
mlflow.log_param("train_size", len(X_train))
mlflow.log_param("test_size", len(X_test))

# METRICS (hasil evaluasi)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)

# MODEL ARTIFACT
mlflow.sklearn.log_model(model, "random_forest_model")
```

### Cara Menjalankan MLflow UI:
```bash
mlflow ui
# Buka browser: http://localhost:5000
```

---

## 🌐 8. SPESIFIKASI STREAMLIT APP

### Fitur Wajib:
1. **Sidebar** — Input 6 nilai fitur dari user
2. **Main panel** — Tampilkan prediksi harga emas
3. **Auto-load model** — Otomatis ambil model terbaru dari `mlruns/`

### Cara Menjalankan:
```bash
streamlit run src/app.py
# Buka browser: http://localhost:8501
```

### Input Range (untuk slider di Streamlit):

| Fitur | Min | Max | Default |
|---|---|---|---|
| `lag_1` | 1000 | 3000 | 1900 |
| `lag_2` | 1000 | 3000 | 1900 |
| `roll_mean_3` | 1000 | 3000 | 1900 |
| `roll_mean_6` | 1000 | 3000 | 1900 |
| `volatility_3` | 0 | 100 | 15 |
| `momentum_1` | -100 | 100 | 0 |

---

## 🔑 9. ATURAN CODING (WAJIB DIPATUHI)

### ✅ BOLEH / HARUS:
- Gunakan `scikit-learn` untuk semua kebutuhan ML
- Gunakan `mlflow.sklearn.log_model()` untuk logging model
- Gunakan time-series split (no shuffle) untuk train/test
- Tambahkan komentar di setiap blok kode (untuk laporan)
- Gunakan `if __name__ == "__main__":` di train.py

### ❌ DILARANG:
- Deep Learning (LSTM, CNN, Transformer, dll.)
- Library selain yang ada di Tech Stack
- Shuffle data saat split (merusak urutan waktu)
- Hardcode path file (gunakan `os.path`)

---

## 📝 10. TEMPLATE KODE (REFERENSI CEPAT)

### train.py — Skeleton

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os

# --- KONSTANTA ---
DATA_PATH = "data/gold_advanced_features.csv"
FEATURES = ["lag_1", "lag_2", "roll_mean_3", "roll_mean_6", "volatility_3", "momentum_1"]
TARGET = "price"
N_ESTIMATORS = 100
MAX_DEPTH = 10

def load_data(path):
    """Step 1: Load dan preprocess data"""
    pass

def train_model(X_train, y_train):
    """Step 2: Training model"""
    pass

def evaluate_model(model, X_test, y_test):
    """Step 2b: Evaluasi model"""
    pass

def run_pipeline():
    """Step 3: Jalankan pipeline + MLflow tracking"""
    pass

if __name__ == "__main__":
    run_pipeline()
```

### app.py — Skeleton

```python
import streamlit as st
import mlflow
import pandas as pd
import os

def load_latest_model():
    """Load model terbaru dari MLflow"""
    pass

def main():
    st.title("🥇 Gold Price Predictor (XAU/USD)")
    # Sidebar untuk input fitur
    # Main panel untuk hasil prediksi
    pass

if __name__ == "__main__":
    main()
```

---

## 🚀 11. URUTAN PENGERJAAN (WORKFLOW)

```
1. [ ] Siapkan dataset di folder data/
2. [ ] Buat train.py (Step 1 + Step 2 + Step 3)
3. [ ] Jalankan: python src/train.py
4. [ ] Verifikasi di MLflow UI: mlflow ui
5. [ ] Buat app.py (Step 4)
6. [ ] Jalankan: streamlit run src/app.py
7. [ ] Test semua fitur
8. [ ] Tulis laporan akademik
```

---

## 📚 12. REFERENSI AKADEMIK

| Topik | Referensi |
|---|---|
| Random Forest | Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. |
| Time Series Splitting | Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice |
| MLflow | Zaharia et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow |
| Gold Price Prediction | Parisi et al. (2008). Forecasting gold price changes using ARIMA and machine learning |

---

## ⚡ 13. TROUBLESHOOTING UMUM

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: mlflow` | `pip install mlflow` |
| MLflow UI tidak bisa dibuka | Pastikan `mlruns/` ada di direktori aktif |
| Model tidak ditemukan di app.py | Jalankan `train.py` terlebih dahulu |
| NaN di dataset | Sudah ditangani dengan `dropna()` di `load_data()` |
| Data terlalu sedikit setelah dropna | Normal — lag dan rolling menghasilkan NaN di awal |

---

*File ini adalah single source of truth untuk proyek ini. Update jika ada perubahan requirement.*
