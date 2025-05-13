import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data(file_path='D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal/Data/combined_climate_data.csv'):
    df = pd.read_csv(file_path)
    return df

class Feature_Engineering:
    @staticmethod
    def show(df):
        st.title("🌾 Feature Engineering for Climate & Agriculture Data")

        if df is None or df.empty:
            st.error("Data not found. Please load the dataset from the Home page.")
            return

        data = df.copy()
        st.session_state.data = data

        st.header("1. Current Dataset Columns")
        st.write(data.columns.tolist())

        st.header("2. Feature Engineering Options")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌡️ Temperature Features")
            if st.checkbox("Add Temperature Range (avg_max_temp - avg_min_temp)"):
                data["temp_range"] = data["avg_max_temp"] - data["avg_min_temp"]

            if st.checkbox("Add Temperature Anomaly (from avg_mean_temp)"):
                mean_temp = data["avg_mean_temp"].mean()
                data["temp_anomaly"] = data["avg_mean_temp"] - mean_temp

        with col2:
            st.subheader("📅 Temporal Features")
            if st.checkbox("Add Decade (year//10 * 10)"):
                data["decade"] = (data["year"] // 10) * 10

            if st.checkbox("Add Years from 2000"):
                data["years_from_2000"] = data["year"] - 2000

        st.subheader("🧠 Derived Agro-Climate Indicators")

        if st.checkbox("Add Rainfall Intensity Index (precipitation_max / annual_rainfall)"):
            data["rainfall_intensity_index"] = data["precipitation_max"] / (data["annual_rainfall"] + 1e-5)

        if st.checkbox("Add Fertilizer Pressure (fertilizer / agri_land_area)"):
            data["fertilizer_pressure"] = data["fertilizer_kg_per_ha"] / (data["agri_land_area"] + 1e-5)

        if st.checkbox("Add Climate Stress Index (humidity * temp_range)"):
            if "temp_range" not in data.columns:
                data["temp_range"] = data["avg_max_temp"] - data["avg_min_temp"]
            data["climate_stress_index"] = data["relative_humidity"] * data["temp_range"]

        st.header("3. Select Features for Modeling")
        default_features = [col for col in data.columns if col != "avg_max_temp"]
        selected_features = st.multiselect(
            "Select input features:",
            options=default_features,
            default=default_features,
        )

        final_features = selected_features + ["avg_max_temp"]
        processed_data = data[final_features]

        if st.button("✅ Apply Feature Engineering"):
            st.session_state.data = processed_data
            st.success(f"✅ Feature engineering applied! {processed_data.shape[1]} features, {processed_data.shape[0]} rows.")
            st.subheader("🔍 Preview of Engineered Data")
            st.dataframe(processed_data.head())

        st.markdown("---")
        st.info("Tip: Proceed to the model training page after applying feature engineering.")
