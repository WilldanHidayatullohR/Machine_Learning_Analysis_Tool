# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan",
    layout="wide"
)

# ===== HEADER & SIDEBAR =====
st.title("Mini Tool Analisis Kepuasan Berbasis Data")
st.caption("Versi demo Kerja Praktek – dapat digunakan untuk berbagai survei kepuasan (sistem, layanan, aplikasi, dan lain-lain).")

with st.sidebar:
    st.header("Informasi Aplikasi")
    st.write(
        """
        Mini tool ini dirancang untuk:
        - Mengunggah data survei kepuasan dalam format CSV
        - Melihat statistik deskriptif dan visualisasi
        - Membangun model prediksi sederhana (Linear Regression dan Random Forest)
        - Menghasilkan insight otomatis dari data
        """
    )
    st.markdown("---")
    st.write("Pengembang: Wildan Hidayatulloh")
    st.write("Program Studi: Informatika, FTI UNSAP")

# ===== FUNGSI BANTU =====

def hitung_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def klasifikasi_level_mean(mean_value):
    """
    Mengembalikan label sederhana berdasarkan nilai rata-rata Likert 1–5.
    """
    if mean_value >= 4.0:
        return "sangat baik"
    elif mean_value >= 3.0:
        return "cukup baik"
    else:
        return "perlu perhatian"

# ===== 1. UPLOAD DATA =====
st.subheader("1. Upload Data Survei")

uploaded_file = st.file_uploader("Upload file CSV hasil survei (skala Likert 1–5 atau angka).", type=["csv"])

if uploaded_file is not None:
    # Simpan ke session_state agar tidak hilang saat rerun
    if "df" not in st.session_state:
        st.session_state["df"] = pd.read_csv(uploaded_file)
    else:
        # Jika user upload file baru, update df
        new_df = pd.read_csv(uploaded_file)
        st.session_state["df"] = new_df

if "df" not in st.session_state:
    st.info("Silakan upload file CSV terlebih dahulu untuk melanjutkan.")
    st.stop()

df = st.session_state["df"]

# ===== 2. PREVIEW DATA & INFO UMUM =====
st.subheader("2. Preview Data dan Informasi Umum")

st.write("Preview 5 baris pertama:")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Baris", df.shape[0])
with col2:
    st.metric("Jumlah Kolom", df.shape[1])
with col3:
    st.metric("Total Nilai Kosong", int(df.isna().sum().sum()))

# ===== 3. PEMILIHAN KOLUMN TARGET =====
st.subheader("3. Konfigurasi Analisis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Tidak ditemukan kolom numerik. Pastikan file berisi nilai angka (misalnya skala Likert 1–5).")
    st.stop()

st.write("Kolom numerik yang terdeteksi:")
st.write(", ".join(numeric_cols))

# Tebak target secara otomatis jika memungkinkan
default_target = None
for cand in ["Kepuasan", "Overall_Satisfaction", "Overall Satisfaction", "Avg_Satisfaction"]:
    if cand in numeric_cols:
        default_target = cand
        break

target_col = st.selectbox(
    "Pilih kolom target kepuasan yang akan dianalisis:",
    options=numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

feature_cols = [c for c in numeric_cols if c != target_col]

if len(feature_cols) == 0:
    st.error("Tidak ada fitur numerik selain kolom target. Tambahkan kolom fitur lain (misalnya SQ1, SQ2, UX1, dan sebagainya).")
    st.stop()

st.success(f"Target yang dipilih: {target_col}. Jumlah fitur yang digunakan dalam model: {len(feature_cols)}.")

# ===== TABS UNTUK ALUR ANALISIS =====
tab_data, tab_stat, tab_model, tab_insight = st.tabs(
    ["Data & Ringkasan", "Statistik & Visualisasi", "Model Prediksi", "Insight Otomatis"]
)

# ===== TAB 1: DATA & RINGKASAN =====
with tab_data:
    st.write("### Ringkasan Statistik Dasar (Kolom Numerik)")
    st.dataframe(df[numeric_cols].describe().T)

    st.write("### Distribusi Nilai Target")
    fig_t, ax_t = plt.subplots(figsize=(6, 3))
    df[target_col].hist(bins=5, ax=ax_t)
    ax_t.set_xlabel(f"Nilai {target_col}")
    ax_t.set_ylabel("Frekuensi")
    ax_t.set_title(f"Distribusi {target_col}")
    plt.tight_layout()
    st.pyplot(fig_t)

# ===== TAB 2: STATISTIK & VISUALISASI =====
with tab_stat:
    st.write("### Rata-rata Skor per Fitur")

    mean_scores = df[feature_cols].mean().sort_values(ascending=False)

    fig_m, ax_m = plt.subplots(figsize=(8, 4))
    mean_scores.plot(kind="bar", ax=ax_m)
    ax_m.set_ylabel("Rata-rata Skor")
    ax_m.set_xlabel("Fitur")
    ax_m.set_title("Rata-rata Skor Fitur")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_m)

    # Korelasi fitur dengan target
    st.write("### Korelasi Fitur dengan Target")
    corr_with_target = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_sorted = corr_with_target.sort_values(ascending=False)

    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.barh(corr_sorted.index, corr_sorted.values)
    ax_c.set_xlabel("Koefisien Korelasi")
    ax_c.set_title("Korelasi Fitur terhadap Target")
    ax_c.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_c)

    # Heatmap korelasi sederhana (jika fitur tidak terlalu banyak)
    if len(feature_cols) <= 25:
        st.write("### Heatmap Korelasi (Fitur dan Target)")
        corr_matrix = df[feature_cols + [target_col]].corr()

        fig_h, ax_h = plt.subplots(figsize=(6, 5))
        cax = ax_h.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax_h.set_xticks(range(len(corr_matrix.columns)))
        ax_h.set_yticks(range(len(corr_matrix.index)))
        ax_h.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax_h.set_yticklabels(corr_matrix.index)
        fig_h.colorbar(cax)
        plt.tight_layout()
        st.pyplot(fig_h)

# ===== TAB 3: MODEL PREDIKSI =====
with tab_model:
    st.write("### Pembangunan Model Prediksi")

    X = df[feature_cols]
    y = df[target_col]

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Linear Regression (baseline)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)
    rmse_lin = hitung_rmse(y_test, y_pred_lin)

    # Model 2: Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = hitung_rmse(y_test, y_pred_rf)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("Hasil Evaluasi Linear Regression:")
        st.write(f"R²: {r2_lin:.3f}")
        st.write(f"RMSE: {rmse_lin:.3f}")
    with col_m2:
        st.write("Hasil Evaluasi Random Forest:")
        st.write(f"R²: {r2_rf:.3f}")
        st.write(f"RMSE: {rmse_rf:.3f}")

    st.write("### Perbandingan Kinerja Model")

    perf_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R2": [r2_lin, r2_rf],
        "RMSE": [rmse_lin, rmse_rf]
    })

    st.dataframe(perf_df)

    fig_p, ax_p = plt.subplots(figsize=(6, 3))
    ax_p.bar(perf_df["Model"], perf_df["R2"])
    ax_p.set_ylabel("R²")
    ax_p.set_title("Perbandingan R² Antar Model")
    plt.tight_layout()
    st.pyplot(fig_p)

    # Feature importance dari Random Forest
    st.write("### Pentingnya Fitur Menurut Random Forest")

    importances = rf_model.feature_importances_
    fi_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    fi_series.plot(kind="barh", ax=ax_fi)
    ax_fi.set_xlabel("Importance")
    ax_fi.set_title("Pentingnya Fitur (Random Forest)")
    plt.tight_layout()
    st.pyplot(fig_fi)

    st.write("Lima fitur paling berpengaruh menurut Random Forest:")
    for feat, val in fi_series.sort_values(ascending=False).head(5).items():
        st.write(f"- {feat}: importance {val:.3f}")

# ===== TAB 4: INSIGHT OTOMATIS =====
with tab_insight:
    st.write("### Ringkasan Insight Otomatis")

    mean_scores = df[feature_cols].mean()
    top3 = mean_scores.sort_values(ascending=False).head(3)
    bottom3 = mean_scores.sort_values(ascending=True).head(3)

    st.write("Aspek dengan rata-rata skor tertinggi:")
    for feat, val in top3.items():
        level = klasifikasi_level_mean(val)
        st.write(f"- {feat}: rata-rata {val:.2f} ({level})")

    st.write("Aspek dengan rata-rata skor terendah:")
    for feat, val in bottom3.items():
        level = klasifikasi_level_mean(val)
        st.write(f"- {feat}: rata-rata {val:.2f} ({level})")

    st.markdown("---")

    st.write("Interpretasi umum:")
    st.write(
        """
        1. Fitur dengan skor rata-rata tinggi menunjukkan area yang sudah berjalan baik 
           dan dapat dipertahankan kualitasnya.
        2. Fitur dengan skor rata-rata rendah dan korelasi positif yang tinggi terhadap target 
           menjadi prioritas utama untuk perbaikan, karena peningkatan pada aspek tersebut 
           berpotensi berdampak langsung pada kenaikan kepuasan.
        3. Hasil analisis model (terutama Random Forest) dapat digunakan untuk memvalidasi 
           apakah fitur yang dianggap penting oleh pengguna juga berpengaruh signifikan 
           secara prediktif terhadap tingkat kepuasan.
        """
    )

st.markdown("---")
st.write("Analisis selesai. Anda dapat mengganti file CSV atau kolom target untuk melakukan eksperimen lain.")
