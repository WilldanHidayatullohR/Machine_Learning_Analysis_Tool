# Mini Tool Analisis Kepuasan LMS (UNSAP)

Aplikasi **Streamlit** sederhana untuk menganalisis hasil survei kepuasan pengguna **Learning Management System (LMS)** di lingkungan Universitas Sebelas April (UNSAP).

## 🎯 Fitur Utama

- Upload data survei dalam format **CSV**
- Preview dan ringkasan data (jumlah baris, kolom, missing value)
- Visualisasi:
  - Rata-rata skor per fitur (bar chart)
  - Distribusi skor kepuasan
- Pemodelan sederhana dengan **Random Forest Regressor**
  - Train–test split otomatis
  - Evaluasi dengan R² dan RMSE
  - Feature importance (top fitur paling berpengaruh)
- (Opsional) **Explainable AI (XAI)** menggunakan **SHAP**
  - SHAP summary plot
  - SHAP bar plot (mean |SHAP|)

## 🗂 Struktur Folder

```text
lms-minitool/
├─ app.py
├─ requirements.txt
├─ README.md
├─ assets/
│   └─ logo.png
└─ sample_data/
    └─ sample_lms_survey.csv
