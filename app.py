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


# ==== 4. STATISTIK & VISUALISASI ====
st.markdown("### 4. Statistik & Visualisasi")

tab_bar, tab_box, tab_corr, tab_scatter, tab_radar = st.tabs(
    ["Rata-rata Fitur", "Sebaran (Boxplot)", "Korelasi", "Scatter Plot", "Radar Aspek"]
)

# -------------------------------------------------
# TAB 1: BAR CHART RATA-RATA PER FITUR + INSIGHT
# -------------------------------------------------
with tab_bar:
    st.subheader("Rata-rata Skor per Fitur")

    mean_scores = df[feature_cols].mean().sort_values(ascending=False)

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    mean_scores.plot(kind="bar", ax=ax_bar)
    ax_bar.set_ylabel("Rata-rata Skor")
    ax_bar.set_xlabel("Fitur")
    ax_bar.set_title("Rata-rata Skor Fitur")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_bar)

    st.markdown("**Ringkasan:**")
    top3 = mean_scores.head(3)
    bottom3 = mean_scores.tail(3)

    st.write("Aspek dengan rata-rata skor **tertinggi**:")
    for idx, val in top3.items():
        st.write(f"- **{idx}** (rata-rata: {val:.2f})")

    st.write("Aspek dengan rata-rata skor **terendah**:")
    for idx, val in bottom3.items():
        st.write(f"- **{idx}** (rata-rata: {val:.2f})")

    st.caption(
        "Nilai rata-rata tinggi menunjukkan area yang sudah berjalan baik dan dapat dipertahankan. "
        "Sebaliknya, rata-rata rendah dapat menjadi kandidat prioritas perbaikan."
    )

# -------------------------------------------------
# TAB 2: BOXPLOT – SEBARAN NILAI PER FITUR
# -------------------------------------------------
with tab_box:
    st.subheader("Sebaran Skor per Fitur (Boxplot)")

    fig_box, ax_box = plt.subplots(figsize=(10, 4))
    df[feature_cols].plot(kind="box", ax=ax_box)
    ax_box.set_ylabel("Skor")
    ax_box.set_title("Sebaran Skor Fitur")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_box)

    st.caption(
        "Boxplot menunjukkan median, sebaran, dan potensi outlier pada setiap fitur. "
        "Fitur dengan sebaran sangat lebar menandakan persepsi pengguna yang beragam."
    )

# -------------------------------------------------
# TAB 3: HEATMAP KORELASI
# -------------------------------------------------
with tab_corr:
    st.subheader("Korelasi Antar Fitur dan Target")

    # Ambil hanya fitur numerik + target
    corr_cols = feature_cols + [target_col]
    corr = df[corr_cols].corr()

    st.write("Matriks Korelasi:")
    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    cax = ax_corr.imshow(corr, interpolation="nearest", cmap="coolwarm")
    ax_corr.set_xticks(range(len(corr_cols)))
    ax_corr.set_yticks(range(len(corr_cols)))
    ax_corr.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr_cols)
    fig_corr.colorbar(cax)
    ax_corr.set_title("Heatmap Korelasi")
    plt.tight_layout()
    st.pyplot(fig_corr)

    st.caption(
        "Nilai korelasi mendekati 1 atau -1 menunjukkan hubungan yang kuat. "
        "Perhatikan fitur dengan korelasi tinggi terhadap target kepuasan."
    )

# -------------------------------------------------
# TAB 4: SCATTER PLOT FITUR vs TARGET
# -------------------------------------------------
with tab_scatter:
    st.subheader("Scatter Plot Fitur vs Target")

    selected_feature = st.selectbox(
        "Pilih satu fitur untuk dibandingkan dengan target:",
        options=feature_cols,
        index=0
    )

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    ax_scatter.scatter(df[selected_feature], df[target_col], alpha=0.7)
    ax_scatter.set_xlabel(selected_feature)
    ax_scatter.set_ylabel(target_col)
    ax_scatter.set_title(f"{selected_feature} vs {target_col}")
    plt.tight_layout()
    st.pyplot(fig_scatter)

    st.caption(
        "Scatter plot membantu melihat pola: apakah kenaikan nilai pada fitur "
        "tersebut cenderung diikuti oleh kenaikan nilai target kepuasan."
    )

# -------------------------------------------------
# TAB 5: RADAR CHART PER ASPEK
# -------------------------------------------------
with tab_radar:
    st.subheader("Profil Aspek Kepuasan (Radar Chart)")

    # Mapping grup aspek → prefix kolom
    aspect_groups = {
        "System Quality": ["SQ1", "SQ2", "SQ3", "SQ4"],
        "Information Quality": ["IQ1", "IQ2", "IQ3", "IQ4"],
        "Service Quality": ["SVQ1", "SVQ2", "SVQ3", "SVQ4"],
        "User Experience": ["UX1", "UX2", "UX3", "UX4"],
        "Expected Satisfaction": ["ES1", "ES2", "ES3", "ES4"],
    }

    aspect_labels = []
    aspect_means = []

    for aspect_name, cols in aspect_groups.items():
        # Hanya ambil kolom yang benar-benar ada di data
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) == 0:
            continue
        aspect_labels.append(aspect_name)
        aspect_means.append(df[valid_cols].mean().mean())

    if len(aspect_labels) < 3:
        st.info(
            "Radar chart membutuhkan minimal beberapa kelompok aspek. "
            "Pastikan nama kolom mengikuti pola: SQ1–SQ4, IQ1–IQ4, SVQ1–SVQ4, UX1–UX4, ES1–ES4."
        )
    else:
        # Siapkan data untuk radar
        angles = np.linspace(0, 2 * np.pi, len(aspect_labels), endpoint=False)
        values = np.array(aspect_means)
        # Tutup kembali ke titik pertama
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig_rad, ax_rad = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax_rad.plot(angles, values, "o-", linewidth=2)
        ax_rad.fill(angles, values, alpha=0.25)
        ax_rad.set_thetagrids(angles[:-1] * 180 / np.pi, aspect_labels)
        ax_rad.set_title("Profil Rata-rata per Aspek", pad=20)
        ax_rad.set_ylim(1, 5)  # asumsi skala Likert 1–5
        st.pyplot(fig_rad)

        st.caption(
            "Radar chart memperlihatkan profil kekuatan dan kelemahan tiap aspek. "
            "Aspek dengan nilai rata-rata lebih tinggi menunjukkan area yang relatif lebih kuat."
        )
