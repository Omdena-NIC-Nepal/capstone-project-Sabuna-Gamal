
import streamlit as st
import sys

sys.path.append("D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal")
from visualization import plot_temperature_trends, plot_rainfall_trends, plot_prediction_context, plot_actual_vs_predicted, plot_agriculture_trends, plot_correlation_heatmap, plot_fertilizer_use, plot_humidity_trends, plot_population_density

def show(df):
    st.title("🌍 Climate and Agriculture Data Exploration")

    # Display data
    st.subheader("🔍 Data Preview")
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

