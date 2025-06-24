import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("modelXGboost.pkl")

# Title
st.title("📊 Prediksi Berdasarkan 3 Fitur Masukan")
st.write("Masukkan data di bawah ini untuk mendapatkan hasil prediksi dari model XGBoost.")

# Input fields
country = st.text_input("🌍 Country/Region", "Indonesia")
adult_prevalence = st.text_input("💉 Adult prevalence (%)", "1.5")
annual_deaths = st.text_input("⚰️ Annual deaths", "15000")
year_of_estimate = st.text_input("📆 Year of estimate", "2022")

# Process input
def preprocess_input(adult_prev, deaths, year):
    try:
        df = pd.DataFrame({
            "Adult_Prevalence": [float(adult_prev)],
            "Annual_Deaths": [int(deaths)],
            "Year_of_estimate": [int(year)]
        })
        return df
    except ValueError:
        st.error("Pastikan semua input numerik diisi dengan benar.")
        return None

# Predict button
if st.button("🔍 Prediksi"):
    input_df = preprocess_input(adult_prevalence, annual_deaths, year_of_estimate)
    
    if input_df is not None:
        try:
            prediction = model.predict(input_df)
            st.success(f"🎯 Hasil Prediksi: {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
