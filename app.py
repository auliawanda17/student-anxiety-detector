import streamlit as st
import numpy as np
import joblib

# ----------------- Load Model & Tools -----------------
model = joblib.load("svm_anxiety_model.pkl")
feature_names = joblib.load("feature_names (1).pkl")
scaler = joblib.load("scaler (1).pkl")

# ----------------- Judul Aplikasi -----------------
st.title("🧠 Klasifikasi Tingkat Kecemasan Mahasiswa Sebelum Ujian")
st.markdown("Berdasarkan faktor gaya hidup selama pembelajaran daring")

# ----------------- Input Pengguna -----------------
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.slider("Usia", 17, 30, 20)
education = st.selectbox("Tingkat Pendidikan", ["Mahasiswa", "Siswa/Siswi", "Diploma", "Sarjana", "Magister", "Doktor"])
screen_time = st.slider("Durasi Screen Time (jam/hari)", 0, 24, 6)
sleep_duration = st.slider("Durasi Tidur (jam/hari)", 0, 12, 7)
physical_activity = st.slider("Aktivitas Fisik (jam/minggu)", 0, 20, 3)
academic_change = st.selectbox("Perubahan Performa Akademik", ["Meningkat", "Tetap", "Menurun"])

# ----------------- Pra-pemrosesan -----------------

# Encode input
gender = 1 if gender == "Laki-laki" else 0
education_map = {"Mahasiswa": 0, "Siswa/Siswi": 1, "Diploma": 2, "Sarjana": 3, "Magister": 4, "Doktor": 5}
education = education_map[education]
academic_map = {"Meningkat": 2, "Tetap": 1, "Menurun": 0}
academic_change = academic_map[academic_change]

# Gabungkan semua fitur ke array numpy
input_data = np.array([[gender, age, education, screen_time, sleep_duration, physical_activity, academic_change]])

# Skalakan data
input_scaled = scaler.transform(input_data)

# ----------------- Prediksi -----------------
if st.button("Prediksi Tingkat Kecemasan"):
    prediction = model.predict(input_scaled)[0]
    label_map = {0: "Tidak Cemas", 1: "Cemas"}
    st.success(f"Tingkat kecemasan kamu sebelum ujian: **{label_map[prediction]}**")
