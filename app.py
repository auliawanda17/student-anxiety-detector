import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler (1).pkl')

# Judul aplikasi
st.title("🎓 Prediksi Kecemasan Siswa Sebelum Ujian")

st.markdown("""
Masukkan data berikut untuk memprediksi apakah seorang siswa mengalami kecemasan sebelum ujian.
""")

# Input pengguna
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=10, max_value=100, value=20)
education = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master"])
screen_time = st.number_input("Screen Time (hrs/day)", min_value=0.0, value=5.0)
sleep_duration = st.number_input("Sleep Duration (hrs)", min_value=0.0, value=6.0)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, value=2.0)
performance_change = st.selectbox("Academic Performance Change", ["Increased", "Decreased", "No Change"])

# Mapping ke angka sesuai LabelEncoder
gender_encoded = 0 if gender == "Female" else 1
education_map = {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3}
performance_map = {"Increased": 0, "Decreased": 1, "No Change": 2}

education_encoded = education_map[education]
performance_encoded = performance_map[performance_change]

# Gabungkan input
input_data = np.array([[gender_encoded, age, education_encoded, screen_time,
                        sleep_duration, physical_activity, performance_encoded]])

# Normalisasi
input_scaled = scaler.transform(input_data)

# Prediksi dan tampilkan hasil
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)
    result = 'Anxious Before Exam' if prediction[0] == 1 else 'Not Anxious'
    
    # Tampilkan hasil utama
    st.success(f"Hasil Prediksi: **{result}**")
    
    # Tambahkan catatan semangat sesuai hasil
    if result == 'Anxious Before Exam':
        st.warning("🌱 Tetap semangat! Kecemasan itu wajar, yuk atur waktu tidur, kurangi screen time, dan cari aktivitas yang bikin rileks. Kamu nggak sendiri!")
    else:
        st.info("💪 Mantap! Kamu terlihat cukup stabil secara mental menjelang ujian. Tetap jaga pola tidur dan rutinitas sehatmu ya!")
