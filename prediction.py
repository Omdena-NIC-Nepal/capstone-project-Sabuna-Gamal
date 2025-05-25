
import pandas as pd
import numpy as np

def make_prediction(model, input_features: dict):
    """
    Make a prediction using the trained model and the given input features.

    Parameters:
    - model: Trained ML model (e.g., LinearRegression, RandomForestRegressor)
    - input_features: Dictionary of features required by the model

    Returns:
    - prediction: float, the predicted output from the model
    """
    # Ensure the input follows the order expected by the model
    feature_order = [
        'year', 'avg_mean_temp', 'avg_min_temp', 'avg_max_temp',
        'relative_humidity', 'precipitation_max', 'annual_rainfall',
        'agri_land_area', 'cropland_pct', 'fertilizer_kg_per_ha',
        'population_density'
    ]
    
    # Extract features in the correct order and reshape for prediction
    features = np.array([[input_features[feature] for feature in feature_order]])
    prediction = model.predict(features)[0]
    return prediction

def get_historical_context(df: pd.DataFrame, target_column: str, filter_column: str, filter_value):
    """
    Retrieve historical values of a target column filtered by a specific value.

    Parameters:
    - df: DataFrame containing historical data
    - target_column: The column of interest (e.g., 'avg_mean_temp')
    - filter_column: The column used for filtering (e.g., 'month', 'region')
    - filter_value: The specific value to filter by

    Returns:
    - List of (year, target_column value) tuples
    """
    records = df[df[filter_column] == filter_value]
    return list(zip(records['year'], records[target_column]))

def get_historical_average(df: pd.DataFrame, target_column: str, filter_column: str, filter_value):
    """
    Get the average value of a target column filtered by a specific column value.

    Parameters:
    - df: DataFrame containing data
    - target_column: Column to compute the mean on (e.g., 'avg_mean_temp')
    - filter_column: Column to apply the filter on (e.g., 'month')
    - filter_value: Specific filter value (e.g., 6 for June)

    Returns:
    - Mean of the target column for the filtered data (float)
    """
    filtered_data = df[df[filter_column] == filter_value]
    return filtered_data[target_column].mean()
