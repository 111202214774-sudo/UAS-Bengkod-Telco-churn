import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI ANTARMUKA ---
st.set_page_config(
    page_title="Analytics: Customer Retention",
    page_icon="📊",
    layout="centered"
)

# --- LOAD ENGINE PREDIKSI ---
@st.cache_resource
def load_engine():
    try:
        return joblib.load('model_churn_terbaik.pkl')
    except FileNotFoundError:
        return None

predictor = load_engine()

if predictor is None:
    st.error("Sistem gagal memuat model. Pastikan file 'model_churn_terbaik.pkl' tersedia.")
    st.stop()

# --- HEADER ---
st.title("Dashboard Prediksi Retensi Pelanggan")
st.info("Gunakan panel di samping untuk memasukkan profil pelanggan dan menekan tombol 'Analisis' di bawah.")

# --- INPUT DATA PELANGGAN ---
st.sidebar.header("📋 Input Data Pelanggan")

def tangkap_input_user():
    # Mengelompokkan input ke dalam Expander agar lebih rapi
    with st.sidebar.expander("Profil Dasar", expanded=True):
        gen = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        snr = st.selectbox("Status Lansia", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        ptn = st.selectbox("Memiliki Pasangan", ["Yes", "No"])
        dep = st.selectbox("Tanggungan Keluarga", ["Yes", "No"])
        tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 12)

    with st.sidebar.expander("Layanan Koneksi", expanded=True):
        phn = st.selectbox("Layanan Telepon", ["Yes", "No"])
        mul = st.selectbox("Multi-Line", ["Yes", "No", "No phone service"])
        int_srv = st.selectbox("Provider Internet", ["DSL", "Fiber optic", "No"])
        sec = st.selectbox("Keamanan Online", ["Yes", "No", "No internet service"])
        bck = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
        prot = st.selectbox("Proteksi Perangkat", ["Yes", "No", "No internet service"])
        sup = st.selectbox("Dukungan Teknis", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.sidebar.expander("Detail Kontrak & Biaya", expanded=True):
        cnt = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
        pap = st.selectbox("Tagihan Digital (Paperless)", ["Yes", "No"])
        pay = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        mth = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Akumulasi Biaya ($)", 0.0, 10000.0, 800.0)

    # Struktur DataFrame (sesuaikan dengan urutan model asli)
    data_map = {
        'gender': gen, 'SeniorCitizen': snr, 'Partner': ptn, 'Dependents': dep,
        'tenure': tenure, 'PhoneService': phn, 'MultipleLines': mul,
        'InternetService': int_srv, 'OnlineSecurity': sec, 'OnlineBackup': bck,
        'DeviceProtection': prot, 'TechSupport': sup, 'StreamingTV': tv,
        'StreamingMovies': mov, 'Contract': cnt, 'PaperlessBilling': pap,
        'PaymentMethod': pay, 'MonthlyCharges': mth, 'TotalCharges': total
    }
    return pd.DataFrame([data_map])

input_data = tangkap_input_user()

# --- RINGKASAN PROFIL ---
st.subheader("Ringkasan Profil")
with st.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("Tenure", f"{input_data['tenure'].values[0]} Bln")
    c2.metric("Biaya/Bln", f"${input_data['MonthlyCharges'].values[0]}")
    c3.metric("Kontrak", input_data['Contract'].values[0])

# --- PROSES PREDIKSI ---
if st.button("Jalankan Analisis", type="primary", use_container_width=True):
    hasil = predictor.predict(input_data)
    
    try:
        prob_score = predictor.predict_proba(input_data)
        confidence = np.max(prob_score) * 100
    except:
        confidence = None

    st.divider()
    
    # Tampilan Hasil
    if hasil[0] == 1:
        st.warning("### ⚠️ Hasil: Berisiko Tinggi (Churn)")
        st.write(f"Sistem mendeteksi probabilitas sebesar **{confidence:.2f}%** bahwa pelanggan ini akan berhenti.")
    else:
        st.success("### ✅ Hasil: Loyal (Non-Churn)")
        st.write(f"Tingkat keyakinan model: **{confidence:.2f}%**.")

    # Rekomendasi
    with st.expander("Lihat Rekomendasi Tindakan"):
        if hasil[0] == 1:
            st.write("- Tawarkan diskon perpanjangan kontrak.")
            st.write("- Hubungi pelanggan untuk menanyakan kendala teknis.")
        else:
            st.write("- Pertahankan kualitas layanan saat ini.")
            st.write("- Tawarkan program loyalitas atau upgrade paket.")
