# =========================================================
# MINI TOOL ANALISIS KEPUASAN (VERSI STABIL STREAMLIT CLOUD)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from io import BytesIO

# --------- Import ReportLab (Opsional untuk PDF Lokal) ---------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --------- ML Library ---------
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# =================================================================
# KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan",
    layout="wide"
)

st.title("Mini Tool Analisis Kepuasan Berbasis Data")
st.caption("Versi demo Kerja Praktek – dapat digunakan untuk berbagai survei kepuasan umum.")

with st.sidebar:
    st.header("Informasi Aplikasi")
    st.write("""
    Tool ini digunakan untuk:
    - Mengunggah file survei (CSV)
    - Menampilkan visualisasi otomatis
    - Membuat model prediksi sederhana
    - Memberikan insight otomatis
    """)
    st.markdown("---")
    st.write("Pengembang: Wildan Hidayatulloh")
    st.write("Prodi: Informatika – FTI UNSAP")


# =================================================================
# FUNGSI BANTU
# =================================================================
def hitung_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def klasifikasi_level_mean(m):
    if m >= 4.0:
        return "sangat baik"
    elif m >= 3.0:
        return "cukup baik"
    return "perlu perhatian"


# =================================================================
# 1. UPLOAD CSV
# =================================================================
st.subheader("1. Upload Data Survei")

uploaded_file = st.file_uploader("Upload file CSV (skala Likert 1–5 atau angka)", type=["csv"])

if uploaded_file is not None:
    st.session_state["df"] = pd.read_csv(uploaded_file)

if "df" not in st.session_state:
    st.info("Silakan upload file CSV untuk melanjutkan.")
    st.stop()

df = st.session_state["df"]


# =================================================================
# 2. PREVIEW DATA
# =================================================================
st.subheader("2. Preview Data dan Informasi Umum")

st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
with col1: st.metric("Jumlah Baris", df.shape[0])
with col2: st.metric("Jumlah Kolom", df.shape[1])
with col3: st.metric("Total Nilai Kosong", int(df.isna().sum().sum()))


# =================================================================
# 3. PILIH KOLUMN TARGET
# =================================================================
st.subheader("3. Konfigurasi Analisis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik ditemukan.")
    st.stop()

default_target = None
for c in ["Kepuasan", "Overall_Satisfaction", "Overall Satisfaction", "Avg_Satisfaction"]:
    if c in numeric_cols:
        default_target = c
        break

target_col = st.selectbox(
    "Pilih kolom target kepuasan",
    numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

feature_cols = [c for c in numeric_cols if c != target_col]

st.success(f"Target dipilih: {target_col} | Fitur dipakai: {len(feature_cols)}")


# =================================================================
# 4. TAB VISUALISASI
# =================================================================
st.markdown("### 4. Statistik & Visualisasi")

tab_bar, tab_box, tab_corr, tab_scatter, tab_radar = st.tabs(
    ["Rata-rata", "Boxplot", "Korelasi", "Scatter Plot", "Radar Aspek"]
)


# ======================================================
# TAB 1: RATA-RATA
# ======================================================
with tab_bar:
    st.subheader("Rata-rata Skor per Fitur")
    mean_scores = df[feature_cols].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,4))
    mean_scores.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.caption("Nilai rata-rata tinggi = area sudah baik. Nilai rendah = perlu perhatian.")


# ======================================================
# TAB 2: BOX PLOT
# ======================================================
with tab_box:
    st.subheader("Sebaran Skor (Boxplot)")
    fig, ax = plt.subplots(figsize=(10,4))
    df[feature_cols].plot(kind='box', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ======================================================
# TAB 3: KORELASI
# ======================================================
with tab_corr:
    st.subheader("Heatmap Korelasi")
    corr = df[feature_cols + [target_col]].corr()

    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)


# ======================================================
# TAB 4: SCATTER
# ======================================================
with tab_scatter:
    st.subheader("Scatter Plot Fitur vs Target")
    feat = st.selectbox("Pilih fitur:", feature_cols)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df[feat], df[target_col], alpha=0.7)
    ax.set_xlabel(feat)
    ax.set_ylabel(target_col)
    st.pyplot(fig)


# ======================================================
# TAB 5: RADAR CHART
# ======================================================
with tab_radar:
    st.subheader("Radar Chart per Aspek")

    aspect_groups = {
        "System Quality": ["SQ1","SQ2","SQ3","SQ4"],
        "Information Quality": ["IQ1","IQ2","IQ3","IQ4"],
        "Service Quality": ["SVQ1","SVQ2","SVQ3","SVQ4"],
        "User Experience": ["UX1","UX2","UX3","UX4"],
        "Expected Satisfaction": ["ES1","ES2","ES3","ES4"]
    }

    labels = []
    values = []

    for asp, cols in aspect_groups.items():
        valid = [c for c in cols if c in df.columns]
        if valid:
            labels.append(asp)
            values.append(df[valid].mean().mean())

    if len(labels) >= 3:
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        values = np.array(values)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.2)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        ax.set_ylim(1,5)
        st.pyplot(fig)
    else:
        st.info("Radar membutuhkan minimal 3 aspek valid.")


# =================================================================
# 5. MODEL PREDIKSI
# =================================================================
st.markdown("### 5. Model Prediksi")

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Linear Regression
model_lin = LinearRegression().fit(X_train, y_train)
pred_lin = model_lin.predict(X_test)

# Model 2: Random Forest
model_rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

colL, colR = st.columns(2)
with colL:
    st.write("**Linear Regression**")
    st.write(f"R²: {r2_score(y_test, pred_lin):.3f}")
    st.write(f"RMSE: {hitung_rmse(y_test, pred_lin):.3f}")
with colR:
    st.write("**Random Forest**")
    rf_r2 = r2_score(y_test, pred_rf)
    st.write(f"R²: {rf_r2:.3f}")
    st.write(f"RMSE: {hitung_rmse(y_test, pred_rf):.3f}")


# =================================================================
# 6. INSIGHT OTOMATIS
# =================================================================
st.markdown("### 6. Insight Otomatis")

mean_scores = df[feature_cols].mean()
corr_target = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)

st.write("**Top 3 fitur skor tertinggi:**")
for f, v in mean_scores.sort_values(ascending=False).head(3).items():
    st.write(f"- {f}: {v:.2f}")

st.write("**Top 3 fitur skor terendah:**")
for f, v in mean_scores.sort_values().head(3).items():
    st.write(f"- {f}: {v:.2f}")

st.write("**Fitur paling berkorelasi positif dengan kepuasan:**")
top_corr = corr_target.sort_values(ascending=False).head(5)
for f, v in top_corr.items():
    st.write(f"- {f}: {v:.2f}")


# =================================================================
# 7. PDF EXPORT (Hanya Lokal)
# =================================================================
st.markdown("### 7. Export Laporan (PDF)")

if REPORTLAB_AVAILABLE:

    st.success("ReportLab terpasang. PDF bisa dibuat.")

    summary_text = (
        f"Target kepuasan: {target_col}\n"
        f"Jumlah responden: {df.shape[0]}\n"
        f"Model Random Forest R²: {rf_r2:.2f}\n\n"
        "Insight utama:\n"
        "- Fitur yang sangat berpengaruh berada pada daftar korelasi tertinggi.\n"
        "- Fitur dengan skor terendah menjadi prioritas perbaikan.\n"
    )

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    text = c.beginText(40, 800)
    text.setFont("Helvetica", 11)

    for line in summary_text.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        "Download PDF",
        data=pdf_buffer,
        file_name="laporan_kepuasan.pdf",
        mime="application/pdf"
    )

else:
    st.info("Fitur PDF hanya tersedia saat dijalankan lokal (ReportLab tidak support di Cloud).")

st.markdown("---")
st.write("Selesai. Silakan ubah file CSV atau target untuk analisis lainnya.")
