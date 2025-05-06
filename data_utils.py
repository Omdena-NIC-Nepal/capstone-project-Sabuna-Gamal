# climate_agriculture_data.py

import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data(file_path = 'D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal/Data/combined_climate_data.csv'):
    #def load_data(file_path = 'D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal/Data/combined_data_climate.csv'):
    # Load the CSV file
    df = pd.read_csv(file_path)

    return df

def prepare_features(df):
    """
    Prepare features and target variable for model training.
    """
    # Define the feature columns
    feature_cols = [
        'year',
        'avg_mean_temp',
        'avg_min_temp', 
        'avg_max_temp', 
        'relative_humidity', 
        'precipitation_max', 
        'annual_rainfall', 
        'agri_land_area', 
        'cropland_pct', 
        'fertilizer_kg_per_ha', 
        'population_density'
    ]

    X = df[feature_cols].values  # Features matrix

    # Define the target variable
    y = df['avg_mean_temp'].values  # Predicting average mean temperature

    return X, y

