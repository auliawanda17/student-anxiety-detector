import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler (1).pkl')

# Judul halaman
st.title("Prediksi Kecemasan Siswa Sebelum Ujian")
st.markdown("Masukkan data siswa untuk memprediksi kemungkinan mengalami kecemasan sebelum ujian.")

# Input pengguna
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=10, max_value=100, value=20)
education = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master"])
screen_time = st.number_input("Screen Time (hrs/day)", min_value=0.0, value=4.0)
sleep_duration = st.number_input("Sleep Duration (hrs)", min_value=0.0, value=7.0)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, value=2.0)
performance_change = st.selectbox("Academic Performance Change", ["Increased", "Decreased", "No Change"])

# Encode input sesuai hasil LabelEncoder
gender_encoded = 0 if gender == "Female" else 1
education_map = {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3}
performance_map = {"Increased": 0, "Decreased": 1, "No Change": 2}

education_encoded = education_map[education]
performance_encoded = performance_map[performance_change]

# Gabungkan data input
input_data = np.array([[gender_encoded, age, education_encoded, screen_time,
                        sleep_duration, physical_activity, performance_encoded]])

# Normalisasi data
input_scaled = scaler.transform(input_data)

# Tombol prediksi
if st.button("Prediksi"):
    # Hitung probabilitas
    probability = model.predict_proba(input_scaled)[0][1]  # Probabilitas kelas 'Anxious'
    prediction = model.predict(input_scaled)[0]

    # Interpretasi hasil
    if probability >= 0.8:
        message = "⚠️ Sangat besar kemungkinan mengalami kecemasan sebelum ujian."
    elif probability >= 0.6:
        message = "⚠️ Cenderung mengalami kecemasan sebelum ujian."
    elif probability >= 0.4:
        message = "🔸 Kemungkinan sedang mengalami sedikit kecemasan."
    elif probability >= 0.2:
        message = "✅ Kemungkinan kecil mengalami kecemasan."
    else:
        message = "✅ Hampir tidak ada kecemasan yang terdeteksi."

    st.subheader("Hasil Prediksi:")
    st.write(f"**Probabilitas kecemasan:** {probability:.2f}")
    st.write(message)
