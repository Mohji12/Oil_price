
import streamlit as st
import joblib
import pandas as pd 

# Load the saved Random Forest model
model = joblib.load('random_forest_model_save14.pkl')

st.header('Oil Price Prediction Using Random Forest',divider='rainbow')

# Input features for prediction

st.text("Input for Random Forst Model")
year = st.number_input('Year', value=2023)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
volume = st.number_input('Volume', value=100000)

# Make a prediction
prediction_data = {'Year': [year], 'Month': [month], 'Day': [day], 'Volume': [volume]}
input_df = pd.DataFrame(prediction_data)
prediction = model.predict(input_df)

st.write('Predicted Oil Price:', prediction[0])

# Information
st.info(
    "This Streamlit app presents the entire Oil Price Prediction project. "
    "It includes data preprocessing, exploratory data analysis (EDA)"
    "ARIMA modeling and Random Forest Model. Adjust the settings in the sidebar to predict future oil prices using ARIMA."
)


st.header('_Streamlit_ is :blue[cool] :sunglasses:')