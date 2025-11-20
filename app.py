# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Coba import SHAP (opsional, hanya jalan kalau terinstal)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan LMS",
    layout="wide"
)

# ======================================
# HEADER & SIDEBAR
# ======================================
st.title("Mini Tool Analisis Kepuasan LMS")
st.caption("Versi demo – Kerja Praktek | Universitas Sebelas April (UNSAP)")

with st.sidebar:
    st.header("Info Aplikasi")
    st.write(
        """
        Mini tool ini digunakan untuk:
        - Mengunggah data survei kepuasan LMS (CSV)  
        - Melihat statistik dasar dan visualisasi  
        - Membangun model sederhana (Random Forest)  
        - (Opsional) Analisis faktor penting dengan XAI (SHAP, jika tersedia)
        """
    )
    st.markdown("---")
    st.write("**Pengembang**: Wildan Hidayatulloh")
    st.write("**Prodi**: Informatika FTI UNSAP")

# ======================================
# 1. UPLOAD DATA
# ======================================
st.subheader("1. Upload Data Survei")

uploaded_file = st.file_uploader("Upload file CSV hasil survei LMS", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload file `.csv` terlebih dahulu untuk memulai analisis.")
    st.stop()

# Baca data
df = pd.read_csv(uploaded_file)

# ======================================
# 2. PREVIEW & RINGKASAN DATA
# ======================================
st.write("### 2. Preview Data")
st.dataframe(df.head())

st.write("**Ringkasan Data:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Baris", df.shape[0])
with col2:
    st.metric("Jumlah Kolom", df.shape[1])
with col3:
    st.metric("Jumlah Nilai Kosong", int(df.isna().sum().sum()))

# ======================================
# 3. KONFIGURASI ANALISIS
# ======================================
st.markdown("### 3. Konfigurasi Analisis")

# Ambil hanya kolom numerik untuk model
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik yang ditemukan. Pastikan data berisi skor Likert (1–5) atau angka.")
    st.stop()

# Coba deteksi kolom target default
default_target = None
for cand in ["Kepuasan", "Overall Satisfaction", "Overall_Satisfaction", "Avg_Satisfaction"]:
    if cand in numeric_cols:
        default_target = cand
        break

target_col = st.selectbox(
    "Pilih kolom target kepuasan (misalnya: Kepuasan / Overall Satisfaction):",
    options=numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

feature_cols = [c for c in numeric_cols if c != target_col]

if len(feature_cols) == 0:
    st.error("Tidak ada fitur selain kolom target. Tambahkan kolom fitur lain (SQ1, SQ2, dan seterusnya).")
    st.stop()

st.success(f"Model akan memprediksi: `{target_col}`, menggunakan {len(feature_cols)} fitur.")

# ======================================
# 4. STATISTIK DESKRIPTIF & VISUALISASI
# ======================================
st.markdown("### 4. Statistik Deskriptif & Visualisasi")

# Rata-rata per fitur
mean_scores = df[feature_cols].mean().sort_values(ascending=False)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("#### Rata-rata Skor per Fitur")
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_scores.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rata-rata Skor")
    ax.set_xlabel("Fitur")
    ax.set_title("Rata-rata Skor Fitur LMS")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    st.write("Ringkasan Cepat")
    top3 = mean_scores.head(3)
    bottom3 = mean_scores.tail(3)
    st.write("Tiga aspek dengan skor tertinggi:")
    for idx, val in top3.items():
        st.write(f"- {idx} (rata-rata: {val:.2f})")

    st.write("Tiga aspek dengan skor terendah:")
    for idx, val in bottom3.items():
        st.write(f"- {idx} (rata-rata: {val:.2f})")

# Distribusi target
st.write("#### Distribusi Skor Kepuasan")
fig2, ax2 = plt.subplots(figsize=(6, 3))
df[target_col].hist(bins=5, ax=ax2)
ax2.set_xlabel("Skor Kepuasan")
ax2.set_ylabel("Frekuensi")
ax2.set_title(f"Distribusi {target_col}")
plt.tight_layout()
st.pyplot(fig2)

# ======================================
# 5. PEMBANGUNAN MODEL RANDOM FOREST
# ======================================
st.markdown("### 5. Model Prediksi Kepuasan (Random Forest)")

X = df[feature_cols]
y = df[target_col]

# Bagi data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inisialisasi dan latih model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluasi
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("R² (Koefisien Determinasi)", f"{r2:.3f}")
with col_m2:
    st.metric("RMSE", f"{rmse:.3f}")

st.caption(
    "Semakin mendekati 1, nilai R² menunjukkan model semakin baik menjelaskan variasi data. "
    "RMSE yang lebih kecil menunjukkan kesalahan prediksi rata-rata yang lebih rendah."
)

# ======================================
# 6. PENTINGNYA FITUR (FEATURE IMPORTANCE)
# ======================================
st.markdown("### 6. Pentingnya Fitur (Model Random Forest)")

importances = rf_model.feature_importances_
fi_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 4))
fi_series.plot(kind="barh", ax=ax3)
ax3.set_xlabel("Importance")
ax3.set_title("Pentingnya Fitur Berdasarkan Random Forest")
plt.tight_layout()
st.pyplot(fig3)

st.write("Top 5 fitur paling berpengaruh menurut model:")
for feat, val in fi_series.sort_values(ascending=False).head(5).items():
    st.write(f"- {feat} (importance: {val:.3f})")

# ======================================
# 7. EXPLAINABLE AI (XAI) DENGAN SHAP (OPSIONAL)
# ======================================
st.markdown("### 7. Explainable AI (SHAP) – Opsional")

if not SHAP_AVAILABLE:
    st.info(
        "Library `shap` belum terinstal di environment ini.\n\n"
        "Jika ingin menjalankan analisis XAI secara lokal:\n"
        "  pip install shap\n\n"
        "Di Streamlit Cloud, Anda dapat menambahkan `shap` ke requirements.txt, "
        "namun instalasi terkadang gagal karena konflik dependency."
    )
else:
    st.success("SHAP tersedia. Menjalankan analisis XAI...")

    max_samples = min(100, len(X))
    X_sample = X.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    st.write("#### SHAP Summary Plot (Dampak Fitur terhadap Prediksi)")
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.write("#### SHAP Bar Plot (Rata-rata |SHAP| per Fitur)")
    plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.caption(
        "Semakin besar nilai rata-rata |SHAP|, semakin besar pengaruh fitur tersebut "
        "terhadap prediksi kepuasan."
    )

st.markdown("---")
st.write("Analisis selesai. Anda dapat mengganti file CSV atau kolom target untuk eksperimen lain.")
