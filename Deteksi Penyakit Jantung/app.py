import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model yang Sudah Disimpan ---
try:
    model_rf = joblib.load('random_forest_model.joblib')
    st.success("Model Random Forest berhasil dimuat.")
except FileNotFoundError:
    st.error("Error: File 'random_forest_model.joblib' tidak ditemukan. Pastikan Anda sudah melatih dan menyimpan model.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

# --- 2. Judul Aplikasi Streamlit ---
st.title("Aplikasi Prediksi Penyakit Jantung")
st.markdown("Aplikasi ini memprediksi kemungkinan penyakit jantung berdasarkan fitur yang dimasukkan.")

# --- 3. Input Pengguna ---
st.header("Masukkan Data Pasien:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia", 20, 80, 50)
    sex = st.radio("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
    cp = st.selectbox("Tipe Nyeri Dada (cp)", options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.slider("Tekanan Darah Istirahat (trestbps)", 90, 200, 120, help="Tekanan darah saat istirahat (mm Hg)")
    chol = st.slider("Kolesterol Serum (chol)", 100, 600, 200, help="Kolesterol serum (mg/dl)")
    fbs = st.radio("Gula Darah Puasa > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    restecg = st.selectbox("Hasil Elektrokardiografi Istirahat (restecg)", options=[0, 1, 2], help="0: Normal, 1: Memiliki kelainan gelombang ST-T, 2: Hipertrofi ventrikel kiri")

with col2:
    thalach = st.slider("Detak Jantung Maksimum (thalach)", 70, 202, 150, help="Detak jantung maksimum yang tercapai")
    exang = st.radio("Angina Akibat Olahraga (exang)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    oldpeak = st.slider("Depresi ST Akibat Olahraga (oldpeak)", 0.0, 6.0, 1.0, 0.1, help="Depresi ST yang diinduksi olahraga relatif terhadap istirahat")
    slope = st.selectbox("Slope Puncak Segment ST Olahraga (slope)", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.selectbox("Jumlah Pembuluh Darah Besar (ca)", options=[0, 1, 2, 3], help="Jumlah pembuluh darah besar (0-3) yang diwarnai oleh fluoroscopy")
    thal = st.selectbox("Thal (thal)", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Tidak diketahui")
    # Catatan: Asumsi thal 0, 1, 2, 3 sesuai dengan dataset aslinya. Jika ada makna khusus, sesuaikan.

# --- 4. Tombol Prediksi ---
if st.button("Prediksi"):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Lakukan prediksi
    prediction = model_rf.predict(input_data)[0]
    prediction_proba = model_rf.predict_proba(input_data)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error(f"**Berdasarkan data yang dimasukkan, pasien memiliki kemungkinan besar MENGALAMI Penyakit Jantung.**")
        st.write(f"Probabilitas Penyakit Jantung: **{prediction_proba[1]*100:.2f}%**")
        st.write(f"Probabilitas Tidak Ada Penyakit Jantung: {prediction_proba[0]*100:.2f}%")
    else:
        st.success(f"**Berdasarkan data yang dimasukkan, pasien memiliki kemungkinan kecil MENGALAMI Penyakit Jantung.**")
        st.write(f"Probabilitas Tidak Ada Penyakit Jantung: **{prediction_proba[0]*100:.2f}%**")
        st.write(f"Probabilitas Penyakit Jantung: {prediction_proba[1]*100:.2f}%")

    st.markdown("---")
    st.info("Catatan: Prediksi ini hanyalah model statistik dan tidak menggantikan diagnosis medis profesional.")