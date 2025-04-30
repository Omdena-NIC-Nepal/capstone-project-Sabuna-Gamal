import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------- Visualization Functions -----------------------

def plot_temperature_trends(df):
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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['relative_humidity'], marker='o', color='purple')
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Humidity (%)")
    ax.set_title("Relative Humidity Trends Over Years")
    ax.grid(True)
    return fig

def plot_agriculture_trends(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Agricultural Land Area (sq km)', color=color)
    ax1.plot(df['year'], df['agri_land_area'], color=color, marker='o', label='Agricultural Land Area')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cropland (%)', color=color)
    ax2.plot(df['year'], df['cropland_pct'], color=color, marker='s', linestyle='--', label='Cropland %')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Agriculture Land Area and Cropland % Trends')
    return fig

def plot_fertilizer_use(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['fertilizer_kg_per_ha'], marker='o', color='brown')
    ax.set_xlabel("Year")
    ax.set_ylabel("Fertilizer Use (kg/ha)")
    ax.set_title("Fertilizer Usage Per Hectare Over Years")
    ax.grid(True)
    return fig

def plot_population_density(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['year'], df['population_density'], marker='o', color='darkred')
    ax.set_xlabel("Year")
    ax.set_ylabel("Population Density (people/sq km)")
    ax.set_title("Population Density Trends Over Years")
    ax.grid(True)
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

# ----------------------- Streamlit Page -----------------------

def show(df):
    st.title("🌍 Climate and Agriculture Data Exploration")

    # Display data
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(10))

    st.subheader("📈 Statistical Summary")
    st.write(df.describe())

    # Plots
    st.subheader("🌡️ Temperature Trends")
    fig = plot_temperature_trends(df)
    st.pyplot(fig)

    st.subheader("🌧️ Rainfall Trends")
    fig = plot_rainfall_trends(df)
    st.pyplot(fig)

    st.subheader("💧 Relative Humidity Trends")
    fig = plot_humidity_trends(df)
    st.pyplot(fig)

    st.subheader("🌾 Agricultural Land and Cropland Trends")
    fig = plot_agriculture_trends(df)
    st.pyplot(fig)

    st.subheader("🧪 Fertilizer Usage Trends")
    fig = plot_fertilizer_use(df)
    st.pyplot(fig)

    st.subheader("👥 Population Density Trends")
    fig = plot_population_density(df)
    st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)
