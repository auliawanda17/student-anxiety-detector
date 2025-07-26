import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler (1).pkl')

# Judul aplikasi
st.title("📘 Prediksi Kecemasan Siswa Sebelum Ujian")

st.markdown("""
Silakan isi data berikut untuk memprediksi apakah seorang siswa mengalami kecemasan sebelum ujian.
""")

# Input pengguna dalam Bahasa Indonesia
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
usia = st.number_input("Usia", min_value=10, max_value=100, value=20)
pendidikan = st.selectbox("Tingkat Pendidikan", ["SMA", "Diploma", "Sarjana", "Magister"])
waktu_layar = st.number_input("Durasi Waktu Layar (jam/hari)", min_value=0.0, value=5.0)
durasi_tidur = st.number_input("Durasi Tidur (jam)", min_value=0.0, value=6.0)
aktivitas_fisik = st.number_input("Aktivitas Fisik (jam/minggu)", min_value=0.0, value=2.0)
perubahan_performa = st.selectbox("Perubahan Performa Akademik", ["Meningkat", "Menurun", "Tidak Berubah"])

# Mapping input ke format numerik (sesuai LabelEncoder yang digunakan sebelumnya)
gender_encoded = 0 if jenis_kelamin == "Perempuan" else 1
pendidikan_map = {"SMA": 0, "Diploma": 1, "Sarjana": 2, "Magister": 3}
performa_map = {"Meningkat": 0, "Menurun": 1, "Tidak Berubah": 2}

pendidikan_encoded = pendidikan_map[pendidikan]
performa_encoded = performa_map[perubahan_performa]

# Gabungkan ke dalam array
input_data = np.array([[gender_encoded, usia, pendidikan_encoded, waktu_layar,
                        durasi_tidur, aktivitas_fisik, performa_encoded]])

# Normalisasi input
input_scaled = scaler.transform(input_data)

# Tombol Prediksi
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)
    hasil = 'Mengalami Kecemasan Sebelum Ujian' if prediction[0] == 1 else 'Tidak Mengalami Kecemasan'

    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{hasil}**")

    # Tampilkan catatan motivasi
    if hasil == 'Mengalami Kecemasan Sebelum Ujian':
        st.warning("🌱 Tetap semangat ya! Kecemasan itu hal yang wajar. Yuk, coba perbaiki pola tidur, kurangi waktu layar, dan lakukan aktivitas menyenangkan. Kamu tidak sendiri!")
    else:
        st.info("💪 Hebat! Kamu tampaknya cukup tenang menghadapi ujian. Tetap pertahankan pola hidup sehat dan semangat belajarnya ya!")
