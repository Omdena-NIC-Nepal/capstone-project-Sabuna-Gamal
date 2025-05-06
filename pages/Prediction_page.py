import streamlit as st
import numpy as np
import sys

sys.path.append("D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal")

from model import load_model
from prediction import make_prediction, get_historical_context, get_historical_average
from visualization import plot_prediction_context

def show(df):
    """
    Display the predictions page
    """
    st.header("Climate and Agricultural Prediction")

    # Load or fetch model
    if 'model' not in st.session_state:
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train the model first.")
            st.stop()
        st.session_state['model'] = model
        st.session_state['model_type'] = "Pre-trained Model"

    st.info(f"Using {st.session_state['model_type']} for predictions")

    # Input features from the user
    st.subheader("Enter Input Features for Prediction:")

    user_input = {
        'year': st.slider("Year", 2000, 2035, 2025),
        'avg_mean_temp': st.number_input("Avg Mean Temperature (°C)", 0.0, 50.0, 26.0),
        'avg_min_temp': st.number_input("Avg Min Temperature (°C)", 0.0, 40.0, 18.0),
        'avg_max_temp': st.number_input("Avg Max Temperature (°C)", 0.0, 60.0, 34.0),
        'relative_humidity': st.number_input("Relative Humidity (%)", 0.0, 100.0, 70.0),
        'precipitation_max': st.number_input("Max Precipitation (mm)", 0.0, 1000.0, 300.0),
        'annual_rainfall': st.number_input("Annual Rainfall (mm)", 0.0, 3000.0, 1400.0),
        'agri_land_area': st.number_input("Agricultural Land Area (sq km)", 0.0, 100000.0, 25000.0),
        'cropland_pct': st.number_input("Cropland Percentage (%)", 0.0, 100.0, 60.0),
        'fertilizer_kg_per_ha': st.number_input("Fertilizer Used (kg/ha)", 0.0, 500.0, 100.0),
        'population_density': st.number_input("Population Density (per sq km)", 0.0, 1000.0, 220.0)
    }

    # Predict Button
    if st.button("Predict Target Variable"):
        model = st.session_state['model']
        prediction = make_prediction(model, user_input)

        # Display result
        st.success(f"Predicted Value: {prediction:.2f}")

        # Historical comparison if needed
        st.subheader("Prediction in Historical Context")
        hist_avg = get_historical_average(df, target_column='avg_mean_temp', filter_column='year', filter_value=user_input['year'])
        st.write(f"Historical Average (for year {user_input['year']}): {hist_avg:.2f}°C")

        diff = prediction - hist_avg
        if diff > 0:
            st.write(f"Prediction is {diff:.2f}°C **higher** than historical average")
        else:
            st.write(f"Prediction is {abs(diff):.2f}°C **lower** than historical average")

        # Historical plot
        hist_temps = get_historical_context(df, target_column='avg_mean_temp', filter_column='year', filter_value=user_input['year'])
        fig = plot_prediction_context(hist_temps, user_input['year'], None, prediction)
        st.pyplot(fig)

