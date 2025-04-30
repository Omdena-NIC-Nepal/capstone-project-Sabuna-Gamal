import streamlit as st
from data_utils import load_data
import sys

sys.path.append("D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal/pages")
from pages import EDA, Model_Training, Prediction_page

# Set the page configuration
st.set_page_config(
    page_title = "Climate Change Impact Assessment and Prediction System for Nepal",
    page_icon=' ',
    layout = 'wide'
)

# Give the title
st.title("Climate Change Impact Assessment and Prediction System for Nepal")
st.markdown("Analyze historical Temperatures data and Precipitation data and predict future trends")

df = load_data()


# Give the sidebar for the app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model_Training", 'Prediction_page'])


# Display the selected page
if page == "EDA":
    EDA.show(df)
elif page == "Model_Training":
    Model_Training.show(df)
else: # Prediction_page
    Prediction_page.show(df)