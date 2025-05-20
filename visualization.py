
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_temperature_trends(df):
    """
    Plot avg_mean_temp, avg_min_temp, and avg_max_temp over years.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['avg_mean_temp'], marker='o', label='Avg Mean Temp (°C)')
    ax.plot(df['year'], df['avg_min_temp'], marker='x', linestyle='--', label='Avg Min Temp (°C)')
    ax.plot(df['year'], df['avg_max_temp'], marker='^', linestyle='-.', label='Avg Max Temp (°C)')
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Trends Over Years")
    ax.legend()
    ax.grid(True)
    return fig

def plot_rainfall_trends(df):
    """
    Plot annual rainfall and precipitation max over years.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['annual_rainfall'], marker='o', label='Annual Rainfall (mm)')
    ax.plot(df['year'], df['precipitation_max'], marker='s', linestyle='--', label='Max Precipitation (mm)')
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title("Rainfall Trends Over Years")
    ax.legend()
    ax.grid(True)
    return fig

def plot_humidity_trends(df):
    """
    Plot relative humidity over years.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['relative_humidity'], marker='o', color='purple')
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Humidity (%)")
    ax.set_title("Relative Humidity Trends Over Years")
    ax.grid(True)
    return fig

def plot_agriculture_trends(df):
    """
    Plot agriculture land area and cropland percentage over years.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Agricultural Land Area (sq km)', color=color)
    ax1.plot(df['year'], df['agri_land_area'], color=color, marker='o', label='Agricultural Land Area')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axis
    color = 'tab:blue'
    ax2.set_ylabel('Cropland (%)', color=color)
    ax2.plot(df['year'], df['cropland_pct'], color=color, marker='s', linestyle='--', label='Cropland %')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Agriculture Land Area and Cropland % Trends')
    return fig

def plot_fertilizer_use(df):
    """
    Plot fertilizer use per hectare over years.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['fertilizer_kg_per_ha'], marker='o', color='brown')
    ax.set_xlabel("Year")
    ax.set_ylabel("Fertilizer Use (kg/ha)")
    ax.set_title("Fertilizer Usage Per Hectare Over Years")
    ax.grid(True)
    return fig

def plot_population_density(df):
    """
    Plot population density over years.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['population_density'], marker='o', color='darkred')
    ax.set_xlabel("Year")
    ax.set_ylabel("Population Density (people/sq km)")
    ax.set_title("Population Density Trends Over Years")
    ax.grid(True)
    return fig

def plot_actual_vs_predicted(y_test, y_pred, label=""):
    """
    Plot the actual vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel(f"Actual {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"Actual vs Predicted {label}")
    ax.grid(True)
    return fig

def plot_prediction_context(hist_data, pred_year, prediction, ylabel="Value"):
    """
    Plot for me the prediction in historical context.
    """
    years, values = zip(*hist_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(years, values, label="Historical Data", color="blue")
    ax.plot(years, values, 'b--', alpha=0.6)

    # Plot prediction
    ax.scatter([pred_year], [prediction], color='red', s=100, label="Prediction")

    # Add trend line
    z = np.polyfit(years, values, 1)
    p = np.poly1d(z)
    ax.plot(range(min(years), pred_year+1), p(range(min(years), pred_year+1)), 'g-', label="Trend")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Historical and Predicted {ylabel}")
    ax.legend()
    ax.grid(True)
    return fig

def plot_correlation_heatmap(df):
    """
    Plot heatmap of correlation matrix among all variables.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

