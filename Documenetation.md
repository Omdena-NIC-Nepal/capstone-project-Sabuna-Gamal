# Climate Change Impact Assessment and Prediction System for Nepal Documentation

## Project Goal: 
- Develop an end-to-end data analysis system that monitors, analyzes, and predicts climate change impacts in Nepal with a focus on vulnerable regions
### Target Audience: 
- Recent data science graduates applying their skills to real-world climate problem
## Project Overview

- Data Exploration
- Feature Engineering
- Machine Learning Model Training and Evaluation
- Prediction

**Target Variable**: Average Maximum Temperature (`avg_max_temp`)

##Core Features
- **Interactive Data Exploration**: Visualize trends, distributions, and correlations
- **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression, Ridge Regression
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Feature Engineering**: Create and select features for modeling
- **Prediction**: Make new predictions with trained models

## Setup Instructions
Clone the repository:
   ```cmd
   https://github.com/Omdena-NIC-Nepal/capstone-project-Sabuna-Gamal.git
   cd capstone-project-Sabuna-Gamal
   ```

Create and activate a virtual environment (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```



 Place your `combined_climate_data.csv` file in the directory: `../data'

## Running the Application
Start the Streamlit application:
```cmd
streamlit run app.py
```
## Streamlit
https://omdena-nic-nepal-capstone-project-sabuna-gamal-app-dwetrk.streamlit.app/


## Navigation Guide
1. **Data Exploration**: Explore and visualize the data
3. **Feature Engineering**: Create and select features
4. **Model Training**: Train machine learning models
6. **Prediction**: Make new predictions

## File Structure

```
capstone-project-Sabuna-Gamal
│
├── data/                # Main application file
│   ├── raw/       # raw datas
│   │   ├── climate       # raw climate datas
│   │ 
│   │   └── socio-economic       # raw socio-economic datas
│   │       
│   ├── processed_data/       # raw datas
│   │   └── combined_climate_data.csv       # combination of all the raw data with selected features and data preprocessing
│   
├── app.py                # Main application file
├── data_utils.py         # Utility functions for data processing
├── visualization.py      # data visualization
├── model.py              # train and evaluate machine leaning models
├── prediction.py         # prediction

├── pages/
│   ├── Data_Exploration.py        # Exploratory Data Analysis
│   ├── Model_Training.py  # Model training page
│   ├── Prediction_page.py      # Prediction interface
│   └── Feature_Engineering.py # Feature engineering
├── requirements.txt      # Dependencies
├── data_preprocessing.ipynb               # combining the data from different data sources and handeling the missing and duplicate data and saved to `../data/combined_climate_data.csv`
├── README.txt        # Project instruction
└── Documentation.md        # Documentation of the project with installation and process
```
