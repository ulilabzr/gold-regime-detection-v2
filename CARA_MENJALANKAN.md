# Cara Menjalankan Proyek Gold Price Prediction

## Urutan Pengerjaan

### 1. Install Dependency (sudah dilakukan)
```bash
pip install pandas numpy scikit-learn mlflow streamlit matplotlib seaborn
```

### 2. Jalankan Training
```bash
python src/train.py
```
Output yang diharapkan:
```
[Step 1] Memuat dataset ... Train size: 78 | Test size: 20
[Step 2] Melatih RandomForestRegressor...
[Step 2b] RMSE : ... | MAE : ... | R² : ...
[Step 3] Logging ke MLflow...
[OK] Run ID: ...
[Done] Pipeline selesai!
```

### 3. Buka MLflow UI (di terminal terpisah)
```bash
python -m mlflow ui --backend-store-uri "file:///R:/CODING/DATA SCIENCE/Gold-Regime-Detections/mlruns" --host 127.0.0.1 --port 5000
```
Buka browser: **http://localhost:5000**
- Cari experiment: **Gold Price Prediction - Random Forest**
- Lihat metrics: RMSE, MAE, R²
- Lihat plots di tab Artifacts

### 4. Jalankan Aplikasi Streamlit (setelah training)
```bash
streamlit run src/app.py
```
Buka browser: **http://localhost:8501**

---

## Catatan Teknis: Kenapa R² Negatif?

Dataset berisi 98 baris bulanan (2018–2026). Harga emas mengalami **tren kenaikan kuat** di 2024–2026 (dari ~$2000 ke >$4000). Karena split time-series 80/20:
- **Train**: 2018–2024 (78 baris)
- **Test**: 2024–2026 (20 baris, harga jauh lebih tinggi)

Model yang dilatih di range $1200–$2500 kesulitan memprediksi $3000–$5000. Ini adalah fenomena normal untuk **time-series extrapolation** — bukan bug.

Untuk laporan akademik, jelaskan bahwa:
- Model bekerja baik **dalam distribusi data training**
- Tren kenaikan harga emas 2024-2026 bersifat struktural (konflik geopolitik, krisis dolar)
- Solusi: tambah data atau gunakan model yang lebih adaptif (misalnya rolling retraining)
