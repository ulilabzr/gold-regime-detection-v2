"""
app.py -- Gold Price Regime Detection — Streamlit Web App
Memprediksi arah harga emas bulan depan: NAIK atau TURUN.

Load model dari model/best_model.pkl (hasil train.py).
Kompatibel dengan Streamlit Community Cloud.
"""

import os
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Gold Regime Detector",
    page_icon="gold",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PATHS
# =============================================================================

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(ROOT, "model", "best_model.pkl")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.pkl")
META_PATH   = os.path.join(ROOT, "model", "model_meta.json")

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

# =============================================================================
# CUSTOM CSS — Dark Gold Theme
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
}
[data-testid="stSidebar"] * { color: #ddd !important; }
[data-testid="stSidebar"] .stSlider label { color: #ffd700 !important; font-size:12px; }

.stApp { background-color: #0a0a14; color: #e0e0e0; }
[data-testid="metric-container"] {
    background: rgba(255,215,0,0.07);
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 12px;
    padding: 12px;
}
.stTabs [data-baseweb="tab"] { color: #aaa !important; }
.stTabs [aria-selected="true"] { color: #ffd700 !important; border-bottom-color:#ffd700 !important; }
hr { border-color: rgba(255,215,0,0.1); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD MODEL & META
# =============================================================================

@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    meta   = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return model, scaler, meta


# =============================================================================
# HELPERS
# =============================================================================

def derive_time_features(year: int, month: int):
    """Hitung quarter, month_sin, month_cos dari year & month."""
    quarter   = (month - 1) // 3 + 1
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return quarter, month_sin, month_cos


def render_prediction_card(label: str, proba_up: float):
    """Tampilkan kartu prediksi dengan warna dinamis."""
    is_up   = label == "UP"
    color   = "linear-gradient(135deg,#1a472a,#2ecc71)" if is_up else \
              "linear-gradient(135deg,#7b1a1a,#e74c3c)"
    icon    = "↑" if is_up else "↓"
    txt_col = "#a8ffcc" if is_up else "#ffaaaa"
    pct     = proba_up * 100 if is_up else (1 - proba_up) * 100

    st.markdown(
        f"""
        <div style="
            background:{color};
            border-radius:20px; padding:36px 24px;
            text-align:center;
            box-shadow:0 8px 32px rgba(0,0,0,0.4);
            margin:12px 0;
        ">
            <p style="font-size:13px;color:{txt_col};margin:0;
                      font-weight:600;letter-spacing:2px;text-transform:uppercase;">
                Prediksi Harga Bulan Depan
            </p>
            <h1 style="font-size:72px;color:#fff;margin:8px 0;font-weight:700;">
                {icon} {label}
            </h1>
            <p style="font-size:16px;color:{txt_col};margin:0;">
                Probabilitas : <b>{pct:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_proba_gauge(proba_up: float) -> plt.Figure:
    """Bar chart sederhana UP vs DOWN probability."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0a0a14")
    colors = ["#2ecc71", "#e74c3c"]
    vals   = [proba_up, 1 - proba_up]
    labels = ["UP", "DOWN"]
    bars   = ax.barh(labels, vals, color=colors, height=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", color="#fff", fontsize=11)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probabilitas", color="#aaa", fontsize=9)
    ax.tick_params(colors="#ccc")
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names) -> plt.Figure | None:
    """Bar chart feature importance (hanya untuk tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]          # top 15 fitur
    feats = [feature_names[i] for i in idx]
    imps  = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0a0a14")
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(feats)))
    ax.barh(feats, imps, color=colors, edgecolor="none")
    ax.set_title("Top 15 Feature Importance", color="#ffd700",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="#ccc", labelsize=8)
    ax.set_xlabel("Importance Score", color="#aaa", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.1, color="#555")
    plt.tight_layout()
    return fig


def plot_model_comparison(all_results: dict) -> plt.Figure:
    """Bar chart perbandingan Accuracy & AUC semua model."""
    names = list(all_results.keys())
    accs  = [all_results[n]["acc"] for n in names]
    aucs  = [all_results[n]["auc"] for n in names]
    x     = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0a0a14")
    w = 0.35
    ax.bar(x - w/2, accs, w, label='Accuracy', color='#3498db', alpha=0.85)
    ax.bar(x + w/2, aucs, w, label='AUC',      color='#ffd700', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha='right',
                                          fontsize=9, color="#ccc")
    ax.tick_params(colors="#ccc")
    ax.set_ylim(0, 0.8)
    ax.axhline(0.5, color='#555', linestyle='--', lw=1, label='Baseline (0.5)')
    ax.legend(fontsize=9, labelcolor='white',
              facecolor='#1a1a2e', edgecolor='#333')
    ax.set_title("Perbandingan Semua Model", color="#ffd700",
                 fontsize=12, fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", alpha=0.1, color="#555")
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:8px 0 4px 0;">
        <h1 style="color:#ffd700;font-size:2.2rem;font-weight:700;margin:0;">
            Gold Regime Detector
        </h1>
        <p style="color:#aaa;font-size:1rem;margin:4px 0 0 0;">
            Prediksi arah harga XAU/USD bulan depan &mdash; UP atau DOWN
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Load Artifacts ────────────────────────────────────────────────────────
    model, scaler, meta = load_artifacts()

    if model is None:
        st.error("**Model belum ditemukan!** Jalankan dulu:\n```bash\npython src/train.py\n```")
        st.stop()

    best_name   = meta.get("best_model_name", "Model")
    all_results = meta.get("all_results", {})

    # ── SIDEBAR: Input Fitur ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Input Data Pasar")
        st.markdown("Masukkan nilai indikator emas saat ini.")
        st.divider()

        # --- Tab 1: Time ---
        st.markdown("**Waktu Prediksi**")
        col_y, col_m = st.columns(2)
        year  = col_y.number_input("Tahun",  min_value=1834, max_value=2030, value=2026, step=1)
        month = col_m.number_input("Bulan",  min_value=1,    max_value=12,   value=3,    step=1)

        quarter, month_sin, month_cos = derive_time_features(int(year), int(month))

        st.divider()
        st.markdown("**Harga & Lag (USD)**")
        price  = st.number_input("Harga saat ini (price)",  100, 10000, 3200, 10)
        lag_1  = st.number_input("lag_1  (1 bln lalu)",     100, 10000, 3100, 10)
        lag_2  = st.number_input("lag_2  (2 bln lalu)",     100, 10000, 3000, 10)
        lag_3  = st.number_input("lag_3  (3 bln lalu)",     100, 10000, 2900, 10)
        lag_6  = st.number_input("lag_6  (6 bln lalu)",     100, 10000, 2700, 10)
        lag_12 = st.number_input("lag_12 (12 bln lalu)",    100, 10000, 2300, 10)

        st.divider()
        st.markdown("**Rolling Statistics**")
        roll_mean_3  = st.number_input("roll_mean_3",  100, 10000, 3100, 10)
        roll_mean_6  = st.number_input("roll_mean_6",  100, 10000, 3000, 10)
        roll_mean_12 = st.number_input("roll_mean_12", 100, 10000, 2800, 10)
        roll_std_3   = st.number_input("roll_std_3 / volatility_3", 0, 1000, 150, 5)
        roll_std_6   = st.number_input("roll_std_6",  0, 1000, 180, 5)

        st.divider()
        st.markdown("**Momentum & Returns**")
        momentum_1   = st.number_input("momentum_1",    -2000, 2000, 100,  10)
        momentum_3   = st.number_input("momentum_3",    -2000, 2000, 300,  10)
        pct_change_1 = st.number_input("pct_change_1",  -0.5,  0.5,  0.03, 0.01, format="%.3f")
        pct_change_3 = st.number_input("pct_change_3",  -0.5,  0.5,  0.09, 0.01, format="%.3f")

        st.divider()
        st.markdown("**EWM & Ratios**")
        ewm_3          = st.number_input("ewm_3",          100, 10000, 3150, 10)
        ewm_6          = st.number_input("ewm_6",          100, 10000, 3000, 10)
        price_to_roll3 = st.number_input("price_to_roll3", 0.5, 2.0,  1.03, 0.01, format="%.3f")
        price_to_roll12= st.number_input("price_to_roll12",0.5, 2.5,  1.14, 0.01, format="%.3f")
        volatility_3   = roll_std_3     # alias

        st.divider()
        predict_btn = st.button("Prediksi Regime", use_container_width=True, type="primary")

    # ── Build Input ───────────────────────────────────────────────────────────
    input_values = [
        price, int(year), int(month), quarter,
        month_sin, month_cos,
        lag_1, lag_2, lag_3, lag_6, lag_12,
        roll_mean_3, roll_mean_6, roll_mean_12,
        roll_std_3, roll_std_6,
        momentum_1, momentum_3,
        pct_change_1, pct_change_3,
        ewm_3, ewm_6,
        price_to_roll3, price_to_roll12,
        volatility_3
    ]
    input_df = pd.DataFrame([input_values], columns=FEATURES)

    # Scale (kecuali XGBoost)
    from xgboost import XGBClassifier as _XGB
    need_scale = not isinstance(model, _XGB)
    input_ready = scaler.transform(input_df) if (need_scale and scaler) else input_df.values

    # ── LAYOUT ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["  Prediksi  ", "  Analisis Model  ", "  Data Input  "])

    # ─── TAB 1: PREDIKSI ─────────────────────────────────────────────────────
    with tab1:
        col_pred, col_gauge = st.columns([1.2, 0.8], gap="large")

        with col_pred:
            st.subheader("Hasil Prediksi")
            if predict_btn:
                proba    = model.predict_proba(input_ready)[0]
                proba_up = proba[1]
                label    = "UP" if proba_up >= 0.5 else "DOWN"
                render_prediction_card(label, proba_up)
                st.caption(f"Model: **{best_name}** | Probabilitas UP: `{proba_up:.4f}` | DOWN: `{1-proba_up:.4f}`")
            else:
                st.info("Atur nilai di sidebar kiri, lalu klik **Prediksi Regime**.")

        with col_gauge:
            st.subheader("Probabilitas")
            if predict_btn:
                fig_g = plot_proba_gauge(proba_up)
                st.pyplot(fig_g, use_container_width=True)

        st.divider()

        # Metric row — model terbaik
        st.subheader("Performa Model Terbaik")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model",  best_name)
        c2.metric("Accuracy",    f"{meta.get('accuracy',0):.4f}")
        c3.metric("AUC",         f"{meta.get('auc',0):.4f}")

    # ─── TAB 2: ANALISIS ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Perbandingan Semua Model")
        if all_results:
            fig_cmp = plot_model_comparison(all_results)
            st.pyplot(fig_cmp, use_container_width=True)

            res_df = pd.DataFrame(all_results).T
            res_df.columns = ["Accuracy", "AUC"]
            res_df = res_df.sort_values("AUC", ascending=False)
            st.dataframe(res_df.style.format("{:.4f}"), use_container_width=True)

        st.divider()
        st.subheader("Feature Importance")
        fig_fi = plot_feature_importance(model, FEATURES)
        if fig_fi:
            st.pyplot(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance tidak tersedia untuk model ini (Logistic Regression).")

    # ─── TAB 3: DATA INPUT ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Nilai Input yang Digunakan")
        styled = input_df.T.rename(columns={0: "Nilai"})
        st.dataframe(styled, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Gold Regime Detection | "
        "Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost | "
        "Experiment Tracking: MLflow | Data: XAU/USD 1833–2026"
    )


if __name__ == "__main__":
    main()
