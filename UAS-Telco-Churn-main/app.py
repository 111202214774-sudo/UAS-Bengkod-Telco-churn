import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman (Page Config)
st.set_page_config(
    page_title="Telco Churn Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Model
try:
    model = joblib.load('model_churn_terbaik.pkl')
except FileNotFoundError:
    st.error("Critical Error: Model file 'model_churn_terbaik.pkl' not found in the directory.")
    st.stop()

# 3. Header Utama
st.title("Sistem Prediksi Churn Pelanggan")
st.markdown("""
Aplikasi ini menggunakan algoritma Machine Learning untuk memprediksi probabilitas pelanggan berhenti berlangganan (Churn).
Silakan masukkan parameter pelanggan pada panel di sebelah kiri untuk memulai analisis.
""")
st.markdown("---")

# 4. Sidebar Input
st.sidebar.title("Parameter Input")
st.sidebar.info("Sesuaikan parameter di bawah ini dengan profil pelanggan.")

def user_input_features():
    # Group 1: Demografi
    st.sidebar.subheader("Profil Demografi")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    # Group 2: Layanan
    st.sidebar.subheader("Layanan Berlangganan")
    tenure = st.sidebar.number_input("Tenure (Bulan)", min_value=0, max_value=72, value=12)
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    sec = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
    backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    dev_prot = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_sup = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    # Group 3: Akun & Pembayaran
    st.sidebar.subheader("Informasi Akun")
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=0.0)

    # Konstruksi DataFrame
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi_lines,
        'InternetService': internet, 'OnlineSecurity': sec, 'OnlineBackup': backup,
        'DeviceProtection': dev_prot, 'TechSupport': tech_sup, 'StreamingTV': tv,
        'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
    }
    return pd.DataFrame(data, index=[0])

# Eksekusi Input
input_df = user_input_features()

# 5. Review Data (Main Panel) - DIPERBAIKI (SPLIT 2 KOLOM)
st.subheader("Tinjauan Data Input")

with st.expander("Lihat Detail Data (Klik untuk membuka)", expanded=True):
    # Ubah dataframe menjadi vertikal (Transpose)
    df_transposed = input_df.T
    df_transposed.columns = ["Value"] # Beri nama kolom

    # Bagi menjadi 2 Kolom Tampilan
    col_kiri, col_kanan = st.columns(2)

    # Tentukan titik tengah untuk membagi data
    half_point = 10 

    with col_kiri:
        st.markdown("**📂 Profil & Layanan Dasar**")
        # Tampilkan separuh data pertama
        st.table(df_transposed.iloc[:half_point])

    with col_kanan:
        st.markdown("**💳 Layanan Tambahan & Akun**")
        # Tampilkan separuh data sisanya
        st.table(df_transposed.iloc[half_point:])

# 6. Logika Prediksi
if st.button("Proses Analisis", type="primary"):
    
    # Melakukan Prediksi
    prediction = model.predict(input_df)
    
    # Mengambil Probabilitas (Jika didukung model)
    try:
        proba = model.predict_proba(input_df)
        probability = np.max(proba) * 100
    except:
        probability = 0

    st.markdown("---")
    st.subheader("Hasil Analisis")

    # Layout Kolom untuk Hasil
    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("Status: CHURN (Berisiko)")
            st.markdown("**Kesimpulan:** Pelanggan memiliki indikasi tinggi untuk berhenti berlangganan.")
        else:
            st.success("Status: NON-CHURN (Aman)")
            st.markdown("**Kesimpulan:** Pelanggan diprediksi akan tetap menggunakan layanan.")

    with col2:
        if probability > 0:
            st.metric(label="Tingkat Keyakinan Model (Probability)", value=f"{probability:.2f}%")
        else:
            st.metric(label="Prediction Output", value=str(prediction[0]))
