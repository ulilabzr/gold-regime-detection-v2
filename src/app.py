"""
app.py -- Streamlit Web App untuk prediksi harga emas (XAU/USD)
Load model dari file pickle (model/random_forest_model.pkl).
Kompatibel dengan Streamlit Community Cloud.

Cara menjalankan lokal:
    streamlit run src/app.py

Deploy:
    Push repo ke GitHub, lalu connect ke https://share.streamlit.io
"""

import os
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="Gold Price Predictor | XAU/USD",
    page_icon="gold",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Konstanta fitur (harus sama dengan train.py)
FEATURES = ["lag_1", "lag_2", "roll_mean_3", "roll_mean_6", "volatility_3", "momentum_1"]

# Path root proyek (dua level di atas src/app.py)
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "random_forest_model.pkl")

# Informasi model (hardcoded dari hasil training, untuk tampilan di UI)
MODEL_INFO = {
    "algorithm"   : "Random Forest Regressor",
    "n_estimators": 100,
    "max_depth"   : 10,
    "random_state": 42,
    "train_size"  : 78,
    "test_size"   : 20,
    "rmse"        : 1261.59,
    "mae"         : 1015.21,
    "r2"          : -1.8206,
}


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Font import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Main background */
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 12px;
        padding: 16px;
    }

    /* Divider */
    hr { border-color: rgba(255,215,0,0.15); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    """
    Load model RandomForest dari file pickle.
    Menggunakan st.cache_resource agar hanya dimuat sekali per session.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# =============================================================================
# HELPER: FEATURE IMPORTANCE CHART
# =============================================================================

def plot_feature_importance(model) -> plt.Figure:
    """Bar chart feature importance dari model Random Forest."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_feats  = [FEATURES[i] for i in idx]
    sorted_imps   = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(sorted_feats)))
    bars = ax.barh(sorted_feats[::-1], sorted_imps[::-1], color=colors[::-1], edgecolor="none")

    for bar, val in zip(bars, sorted_imps[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=9, color="#ffd700")

    ax.set_title("Feature Importance", fontsize=12, color="#ffd700", fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score", fontsize=9, color="#aaa")
    ax.tick_params(colors="#ccc", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.15, color="#555")
    plt.tight_layout()
    return fig


# =============================================================================
# HELPER: PREDICTION BOX
# =============================================================================

def render_prediction_box(value: float):
    """Tampilkan kotak hasil prediksi bergaya premium."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #b8860b 0%, #ffd700 50%, #ff8c00 100%);
            border-radius: 20px;
            padding: 36px 24px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(255,215,0,0.3);
            margin: 16px 0;
        ">
            <p style="font-size:13px; color:#3d2600; margin:0; font-weight:600; letter-spacing:2px; text-transform:uppercase;">
                Prediksi Harga Emas
            </p>
            <h1 style="font-size:56px; color:#1a0d00; margin:8px 0; font-weight:700; letter-spacing:-1px;">
                ${value:,.2f}
            </h1>
            <p style="font-size:12px; color:#5c3d00; margin:0; font-weight:500;">
                XAU/USD &mdash; per Troy Ounce
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# APLIKASI UTAMA
# =============================================================================

def main():

    # ─── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 4px 0;">
            <h1 style="color:#ffd700; font-size:2.4rem; font-weight:700; margin:0;">
                Gold Price Predictor
            </h1>
            <p style="color:#aaa; font-size:1rem; margin:4px 0 0 0;">
                XAU/USD &mdash; Random Forest Regressor &mdash; Monthly Data (2018–2026)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ─── Load Model ──────────────────────────────────────────────────────────
    model = load_model()

    if model is None:
        st.error(
            "**Model belum ditemukan** di `model/random_forest_model.pkl`.  \n\n"
            "Jalankan training terlebih dahulu:\n"
            "```bash\npython src/train.py\n```"
        )
        st.stop()

    # ─── SIDEBAR: Input Fitur ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Input Fitur")
        st.markdown("Geser slider untuk mengatur nilai setiap fitur prediksi.")
        st.divider()

        lag_1        = st.slider("lag_1 — Harga 1 bulan lalu",     1000, 6000, 3300, 10,
                                 help="Harga penutupan XAU/USD bulan sebelumnya")
        lag_2        = st.slider("lag_2 — Harga 2 bulan lalu",     1000, 6000, 3200, 10,
                                 help="Harga penutupan XAU/USD 2 bulan sebelumnya")
        roll_mean_3  = st.slider("roll_mean_3 — Rata-rata 3 bulan", 1000, 6000, 3200, 10,
                                 help="Rata-rata harga 3 bulan terakhir")
        roll_mean_6  = st.slider("roll_mean_6 — Rata-rata 6 bulan", 1000, 6000, 3100, 10,
                                 help="Rata-rata harga 6 bulan terakhir")
        volatility_3 = st.slider("volatility_3 — Volatilitas",      0,    500,  80,   5,
                                 help="Standar deviasi harga 3 bulan terakhir")
        momentum_1   = st.slider("momentum_1 — Momentum",          -500,  500,  50,  10,
                                 help="Selisih harga bulan ini vs bulan lalu")

        st.divider()
        predict_btn = st.button(
            "Prediksi Harga Emas",
            use_container_width=True,
            type="primary",
        )

    # ─── Layout Utama ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    input_df = pd.DataFrame(
        [[lag_1, lag_2, roll_mean_3, roll_mean_6, volatility_3, momentum_1]],
        columns=FEATURES,
    )

    # ─── KIRI: Prediksi ───────────────────────────────────────────────────────
    with col_left:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            prediction = model.predict(input_df)[0]
            render_prediction_box(prediction)
            st.success(f"Prediksi berhasil: **${prediction:,.2f} USD** per troy ounce")
        else:
            st.info("Atur nilai fitur di sidebar kiri, lalu klik **Prediksi Harga Emas**.")

        with st.expander("Lihat nilai input"):
            styled = input_df.T.rename(columns={0: "Nilai"})
            st.dataframe(styled, use_container_width=True)

    # ─── KANAN: Info Model ────────────────────────────────────────────────────
    with col_right:
        st.subheader("Informasi Model")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE",  f"${MODEL_INFO['rmse']:,.0f}")
        c2.metric("MAE",   f"${MODEL_INFO['mae']:,.0f}")
        c3.metric("R²",    f"{MODEL_INFO['r2']:.3f}")

        st.markdown(
            f"""
            | Parameter | Nilai |
            |:---|:---|
            | Algoritma | Random Forest |
            | n_estimators | {MODEL_INFO['n_estimators']} |
            | max_depth | {MODEL_INFO['max_depth']} |
            | Training samples | {MODEL_INFO['train_size']} baris |
            | Test samples | {MODEL_INFO['test_size']} baris |
            | Data range | 2018 – 2026 (bulanan) |
            """
        )

    # ─── Feature Importance ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Importance")
    st.caption("Fitur dengan importance tertinggi paling berpengaruh terhadap prediksi model.")
    fig = plot_feature_importance(model)
    st.pyplot(fig, use_container_width=True)

    # ─── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Dibuat untuk keperluan akademik | "
        "Model: Random Forest Regressor | "
        "Experiment Tracking: MLflow | "
        "Data: XAU/USD Monthly Historical 2018–2026"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
