import streamlit as st
from data_utils import load_data
import sys

#sys.path.append("D:/Assignment/Capstone Project/capstone-project-Sabuna-Gamal/pages")
from pages import Data_Exploration, Model_Training, Prediction_page
from pages.Feature_Engineering import Feature_Engineering 


# Set the page configuration
st.set_page_config(
    page_title = "Climate Change Impact Assessment and Prediction System for Nepal ",
    page_icon=' ',
    layout = 'wide'
)

# Give the title
st.title("Climate Change Impact Assessment and Prediction System for Nepal")
st.markdown("Analyze historical Temperatures data and Precipitation data and predict future trends")

df = load_data()


# Give the sidebar for the app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [" Data_Exploration", "Feature_Engineering", "Model_Training", "Prediction_page"])


# Display the selected page
if page == " Data_Exploration":
    Data_Exploration.show(df)
elif page == "Feature_Engineering":
    Feature_Engineering.show(df)
elif page == "Model_Training":
    Model_Training.show(df)
else: #Prediction
    Prediction_page.show(df)
    