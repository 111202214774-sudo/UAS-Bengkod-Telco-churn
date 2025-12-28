import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="centered"
)

# 2. Load Model
try:
    # PASTIKAN NAMA FILE INI SAMA PERSIS DENGAN YANG DIUPLOAD
    model = joblib.load('model_churn_Rizalda.pkl') 
except FileNotFoundError:
    st.error("Error: File 'model_churn_Rizalda.pkl' tidak ditemukan. Pastikan file model sudah diupload ke GitHub dengan nama yang sesuai.")
    st.stop()

# 3. Judul & Header
st.title("📊 Analisis Retensi Pelanggan")
st.write("Gunakan aplikasi ini untuk mendeteksi potensi pelanggan berhenti berlangganan (Churn).")
st.markdown("---")

# 4. Form Input
with st.form("churn_form"):
    st.subheader("Profil Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        partner = st.selectbox("Punya Pasangan?", ["Yes", "No"])
        dependents = st.selectbox("Punya Tanggungan?", ["Yes", "No"])
        tenure = st.number_input("Lama Langganan (Bulan)", 0, 72, 12)

    with col2:
        phone = st.selectbox("Layanan Telepon", ["Yes", "No"])
        internet = st.selectbox("Jenis Internet", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Metode Bayar", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        charges = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 50.0)

    # Input Tambahan
    with st.expander("Fitur Tambahan (Opsional)"):
        tech_sup = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        dev_prot = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        online_sec = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_back = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        stream_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        stream_mov = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        paperless = st.selectbox("Tagihan Paperless", ["Yes", "No"])
        multi_line = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        total_charges = st.number_input("Total Tagihan ($)", 0.0, 10000.0, charges * tenure)

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Analisis Risiko Churn")

# 5. Logika Prediksi
if submitted:
    # Siapkan Data
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi_line,
        'InternetService': internet, 'OnlineSecurity': online_sec, 'OnlineBackup': online_back,
        'DeviceProtection': dev_prot, 'TechSupport': tech_sup, 'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': charges, 'TotalCharges': total_charges
    }
    
    df_input = pd.DataFrame(data, index=[0])

    # Prediksi
    prediction = model.predict(df_input)
    
    try:
        proba = model.predict_proba(df_input)
        prob_churn = proba[0][1] * 100
        prob_loyal = proba[0][0] * 100
    except:
        prob_churn = 0

    st.markdown("---")
    
    # Tampilan Hasil
    if prediction[0] == 1:
        st.error(f"⚠️ PERINGATAN: Risiko Churn Tinggi ({prob_churn:.1f}%)")
        st.progress(int(prob_churn))
        st.write("Saran: Tawarkan diskon atau kontrak jangka panjang segera.")
    else:
        st.success(f"✅ AMAN: Pelanggan Loyal ({prob_loyal:.1f}%)")
        st.progress(int(prob_loyal))
